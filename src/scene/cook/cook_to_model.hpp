#pragma once
// ---------------------------------------------------------------------------
// nk CookToModel — SceneIR -> {nk::Model, SceneMap} (the design /).
//
// Drives the EXISTING cook (scene::CookScene for the heavy lifting: V-HACD /
// SDF / filters; runtime::articulation::CookArticulations for the kinematic
// tree) and TRANSCRIBES the resulting CookedBlob + articulation topology into
// the nk::Model field tables. The env template is replicated env_count times
// (env-major). The physics-material bucket table is built from the cooked
// per-shape friction/contact params. SceneMap binds each entity to its cooked
// rows (record order == row order for bodies/joints; shapes expand by V-HACD
// pieces — see the .cpp for the mapping choice).
//
// PURE C++ — zero CUDA tokens. (The Model's device upload happens later, in
// nk::World; CookToModel only fills host tables.)
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "nk/model/model.hpp"
#include "scene/cook/media_render_surface.hpp"
#include "scene/scene_ir.hpp"
#include "scene/scene_map.hpp"
#include "scene/terrain/heightfield.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace nuka::scene {
struct CookedBlob;
}

namespace nuka::scene::cook {

struct CookToModelResult {
    nk::Model model;
    SceneMap  scene_map;
    // The baked terrain grid when the scene carried a TerrainRecord (empty/invalid
    // otherwise). Retained so obs/spawn/render sample the SAME grid the cook staged
    // for contact (SampleHeightFieldZ). IsValid(terrain) == false for a terrain-free scene.
    ::nuka::terrain::HeightField terrain;
    // The cooked tables this model was transcribed from, for host-side mirrors.
    std::shared_ptr<const CookedBlob> blob;
};

// Cook a SceneIR into an nk::Model replicated across `env_count` envs, plus the
// EntityId<->row SceneMap. env_count must be >= 1 (clamped to 1 if 0).
CookToModelResult CookToModel(const SceneIR& scene, int env_count);

// B1 (general contact pipeline (PairDriven)): cook options. contact_family selects the
// stepped contact path. : the legacy FusedFoot runtime is DELETED, so the cook
// always routes the GENERAL LBVH -> cvx narrowphase -> mixed-island solve path —
// PairDriven is now the DEFAULT. The cook resizes the per-env row budget for the
// general per-candidate-slot row layout and sets model.contact_family = PairDriven
// (so the pipeline routes the general assembly/solve). H1 grasp keeps UnionCsr
// (union_cook). Cross-env filtering is left ON for PairDriven multi-dog-in-one-env
// (intra-env collidables collide). The FusedFoot enum value is retained (dead)
// until the enum collapse.
enum class CookContactFamily { FusedFoot, PairDriven };
struct CookToModelOptions {
    CookContactFamily contact_family = CookContactFamily::PairDriven;
    // enable_contacts == false cooks a CONTACTS-OFF world: the per-env contact /
    // candidate-slot budget is zeroed (max_contacts_per_env == max_rows_per_env ==
    // 0) so the pipeline emits NO broadphase/narrowphase/solve ops -- the world is
    // pure articulation + rigid dynamics. This is the general-path equivalent of the
    // legacy single-env `enable_contacts == false` (the dynamics oracle): a fixed-
    // base / free-space scene with nothing to collide. The cooked shape_table /
    // link_geom rows stay sized consistently (just never read). Default true.
    bool enable_contacts = true;
    // Produce optional visual distance fields without binding them to colliders.
    bool bake_link_sdf = false;
};
CookToModelResult CookToModel(const SceneIR& scene, int env_count,
                              const CookToModelOptions& options);

// ---------------------------------------------------------------------------
// The ONE cook-time heightfield bake. Copies a HeightField's normalized [0,1]
// grid into model.heightfield_heights, derives the HeightfieldData device locator
// from its placement, and registers a STATIC heightfield collidable (a body_init
// row with inv_mass==0 + body_id==-1 in shape_table) so the field enters the LBVH
// as one big leaf and any overlapping body yields a (body, heightfield) candidate
// pair routed to the per-cell TRIANGLE_PRISM midphase. The same grid drives obs
// (the bilinear sampler) and render, so they agree by construction. Returns the
// heightfield collidable's body row. Sets model.capacities.max_heightfield_cells.
// The grid must be square-celled (cell_x == cell_y); a non-square grid returns ~0u
// (a build error, not a silent reinterpretation -- the device locator is one cell).
uint32_t CookHeightfieldGrid(nk::Model& model,
                             const ::nuka::terrain::HeightField& hf);

// Bake a HeightField into the model as the static terrain collidable AND recompute
// the contact budget -- the ONE terrain bake both the editor cook (from a SceneIR
// TerrainRecord) and the c_abi built-scene path (from a nuka_world_desc terrain) call
// so terrain enters the model through identical code. Seeds the heightfield collidable
// at `orig_bodies` (overwriting the static ground-plane shape row appended there, so
// the collidable count is unchanged), then sets bodies_per_env / max_bodies_total /
// the per-candidate-slot row budget by the SAME (dynamic+1 static)*4 rule the body cook
// uses, and grows the body<->particle reserve. Returns the heightfield collidable's body row.
uint32_t CookTerrainIntoModel(nk::Model& model,
                              const ::nuka::terrain::HeightField& hf,
                              uint32_t orig_bodies);

// Reserve one-point particle contacts above the rigid manifold budget.
// row_exempt is the grid-owned particle count from Model::MpmParticlesPerEnv().
void GrowContactBudgetForParticles(nk::ModelCapacities& cap, uint32_t rigid_base,
                                   uint32_t row_exempt = 0u);

void SetOgcContactCapacity(nk::Model& model, uint32_t capacity);

// ---------------------------------------------------------------------------
// Particle cook: stage vertex-block cloth/cable or MLS-MPM material points into the
// nk::Model particle block. The cook layer is pure C++ and takes the cookers' POD products.
// ---------------------------------------------------------------------------

struct CookParticleSurface {
    std::vector<uint32_t> triangles;
    float half_thickness = 0.0f;
    float friction = 0.6f;
    std::vector<uint32_t> edges;
};

struct VbdCookInput {
    std::vector<math::Vec3>      positions;     // per-particle rest state
    std::vector<math::Vec3>      velocities;    // per-particle initial velocity
    std::vector<float>           inv_mass;      // 1/mass (0 == pinned)
    // Cloth aerodynamic-drag surface triangles (3 particle indices / triangle).
    // Empty => no drag (the drag op is inert; the cook stays byte-identical).
    std::vector<std::array<uint32_t, 3>> aero_triangles;
    // Lumped anisotropic air-drag coeffs (0.5*rho*Cn, 0.5*rho*Ct) + per-step
    // impulse clamp. All 0 => drag off. Cn >> Ct seeds flutter on a falling sheet.
    float    aero_drag_normal  = 0.0f;
    float    aero_drag_tangent = 0.0f;
    float    aero_drag_max_dv  = 0.0f;
    // body<->soft contact mu, finite so a foot grips/drags the cloth (solmix=max).
    float    friction = 0.6f;
    std::vector<CookParticleSurface> surfaces;
    // Vertex-block elements over input particles [vbd_begin, vbd_begin + vbd_count), indexed by input particle.
    std::vector<nk::VbdElement> vbd_elements;
    uint32_t vbd_begin = 0u, vbd_count = 0u;
};

// Stage a vertex-block soft body into the Model (single-env template;
// SeedInitialState replicates env-major).
void CookVbdParticles(nk::Model& model, uint32_t env_count,
                       const VbdCookInput& in);

// One MLS-MPM material the cook appends to model.mpm_materials (the constitutive
// branch, later, reads it). Mirrors nk::MpmMaterial 1:1.
struct MpmMaterialInput {
    float youngs = 0.0f, poisson = 0.0f, density = 0.0f;
    float dp_friction = 0.0f, dp_cohesion = 0.0f;
    // 0 = fixed-corotated elastic, 1 = Drucker-Prager (reserved), 2 = Neo-Hookean,
    // 3 = weakly-compressible fluid (Tait EOS, reads the three fields below).
    float model_kind = 0.0f;
    float bulk_modulus = 0.0f, tait_gamma = 0.0f, viscosity = 0.0f;
    float yield_stress = 0.0f, hardening_modulus = 0.0f;
};

struct MpmCookInput {
    uint32_t contact_capacity = 0u;  // Zero reserves one slot per reachable grid node.
    std::vector<math::Vec3> positions;     // per-particle rest state
    std::vector<math::Vec3> velocities;    // per-particle initial velocity
    std::vector<float>      inv_mass;      // 1/mass (0 == pinned)
    std::vector<float>      vol0;           // per-particle sampling-lattice volume
    MpmMaterialInput        material;       // single-material cook: appended as id 0
    // Heterogeneous cook: N material rows + a per-particle material index into them.
    // Empty (default) => the single `material` above, all particles id 0 (byte-identical).
    std::vector<MpmMaterialInput> materials;
    std::vector<uint32_t>         material_id;
    math::Vec3 grid_origin{0.0f, 0.0f, 0.0f};  // world corner of node (0,0,0)
    uint32_t   grid_dims[3] = {0u, 0u, 0u};    // per-env node resolution
    float      dx = 0.0f;                       // uniform node spacing (> 0).
    // Internal explicit substeps per World.Step (CFL headroom for a stiff medium).
    uint32_t   substeps = 1u;
    // Static floor plane the grid BC projects against (z-up). normal need not be
    // unit (the BC normalizes); friction is the Coulomb mu on the tangential drag.
    math::Vec3 floor_normal{0.0f, 0.0f, 1.0f};
    float      floor_d = 0.0f;
    float      floor_friction = 0.4f;
    // Stress joins the shared row solve as one block of rows per occupied grid cell.
    bool       implicit_stress = false;
};

// Stage an MLS-MPM bulk-soft body into the Model (single-env template; F=identity,
// C=0, vol0 from the lattice). Sets ParticleMode::Mpm + the grid/material caps;
// reuses the SAME particle-slot reserve as CookVbdParticles (coupling one-path).
void CookMpmParticles(nk::Model& model, uint32_t env_count,
                      const MpmCookInput& in);

// Validate the (kind x method) of every medium: vertex blocks for cloth/cable, MLS-MPM otherwise.
void ValidateMedia(const std::vector<MediaRecord>& media);


// Co-resident [mpm | vertex blocks] cook: MPM occupies [0, n_mpm), the soft set the rest.
// One empty side reduces to the single-medium cook; contact_d_min is the body<->particle diameter.
void CookMpmVbd(nk::Model& model, uint32_t env_count, const MpmCookInput& mpm,
                 const VbdCookInput& soft, float contact_d_min = 0.0f);

// ---------------------------------------------------------------------------
// Media records -> particle cook. The per-medium builders read a scene
// MediaRecord and reuse the SAME topology/lattice cookers
// (soft::BuildClothVertexBlocks / import::cooker::CookParticleBox) -- never
// reimplemented. The dispatch concatenates vertex-block media into the soft slice and
// MLS-MPM media onto one grid, composing them with CookMpmVbd when both are present.
// ---------------------------------------------------------------------------

// Whether a medium solves as vertex blocks (cloth and cable); these end the soft slice.
bool IsVertexBlockMedium(const MediaRecord& media);

// Build the soft-slice input of a cloth MediaRecord: a baked mesh or a flat (nx x ny)
// lattice, lumped area masses, and StVK membrane plus hinge vertex blocks from
// soft::BuildClothVertexBlocks. Empty when the geometry is absent.
VbdCookInput BuildClothVertexBlockInput(const MediaRecord& media);

// Build the soft-slice input of a cable MediaRecord: segments+1 vertex blocks from
// start to end joined by axial springs and segment bends, the pinned endpoint(s)
// kinematic, and an optional slab whose 8 corners form a 28-spring lattice welded to
// the loaded end by 4 springs. Empty when segments < 1 or radius <= 0.
VbdCookInput BuildCableVertexBlockInput(const MediaRecord& media);

// Build the MLS-MPM cook input for a tet-soft / fluid MediaRecord: particles sampled
// from the geometry (sphere lattice / CookParticleBox), material + grid from MediaMpmMaterial.
MpmCookInput BuildMpmInput(const MediaRecord& media);

// Concatenate MPM media and material tables onto one grid with matching spacing and floor.
MpmCookInput BuildMpmInput(const std::vector<MediaRecord>& media);

// The cloth lattice render-surface triangle list (two triangles per quad, the SAME
// row-major winding BuildClothVertexBlockInput meshes the elements with). Empty when the
// grid extent is absent. Indexes the [0, nx*ny) cloth particles (laid out first).
std::vector<uint32_t> BuildClothSurfaceTriangles(const MediaRecord& media);

// Fixed, density and grain surfaces reference the same particle order as CookSceneMedia.
std::vector<MediaRenderSurface> BuildSceneMediaRenderSurfaces(
    const std::vector<MediaRecord>& media);

// Dispatch a validated media list onto an ALREADY-cooked Model: vertex-block media through
// CookVbdParticles, MLS-MPM media through CookMpmParticles, both through CookMpmVbd.
void CookSceneMedia(nk::Model& model, uint32_t env_count,
                    const std::vector<MediaRecord>& media,
                    float particle_contact_radius = 0.0f);

// Orchestrator: the rigid/articulation CookToModel followed by the media-list
// dispatch over the SAME SceneIR (ONE path). A media-free scene is byte-identical to
// CookToModel; a scene carrying cloth/fluid media cooks them onto the rigid Model.
CookToModelResult CookSceneToModel(const SceneIR& scene, int env_count,
                                   const CookToModelOptions& options);

} // namespace nuka::scene::cook
