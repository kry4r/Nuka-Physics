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
#include <vector>

namespace nuka::scene::cook {

struct CookToModelResult {
    nk::Model model;
    SceneMap  scene_map;
    // The baked terrain grid when the scene carried a TerrainRecord (empty/invalid
    // otherwise). Retained so obs/spawn/render sample the SAME grid the cook staged
    // for contact (SampleHeightFieldZ). IsValid(terrain) == false for a terrain-free scene.
    ::nuka::terrain::HeightField terrain;
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
    // Bake a per-body SDF from that body's VISUAL trimesh and switch its collision
    // row to kShapeSdfMesh, so particle/MPM/rigid contact rides the true silhouette
    // (not the inset primitive). A no-op for primitive-only scenes (no visual mesh
    // to bake). Default false leaves every existing cook byte-identical.
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

// Grow the body-contact budget by a DISJOINT reserve above `rigid_base` (the cooked
// body<->body slot count) for body<->particle rows; idempotent, byte-identical when
// particles_per_env == 0 (the reserve is then 0). The c_abi terrain cook recomputes
// the rigid base after appending terrain collidables and re-applies this so a world
// with BOTH terrain and particles keeps the rigid + particle slot ranges disjoint.
// `row_exempt` = per-env particles that never emit body rows (the grid-coupled MPM
// slice under MpmXpbd); they reserve no slots.
void GrowContactBudgetForParticles(nk::ModelCapacities& cap, uint32_t rigid_base,
                                   uint32_t row_exempt = 0u);

// ---------------------------------------------------------------------------
// particle cook (the design "粒子 XPBD/PBF"). Stage an XPBD soft body or a
// PBF fluid into the nk::Model particle block + set the particle/constraint
// capacities. The cook layer is PURE C++ (nk_engine lint scope: zero CUDA
// tokens), so it CANNOT include the legacy runtime::soft/fluid world headers
// (those pull <cuda_runtime.h>). Instead the cook takes the cookers' PLAIN
// PRODUCTS as POD arrays (the caller runs import::cooker::CookXpbdSoftBody /
// CookFluidBox — which ARE the reuse — and hands the de-interleaved constraint
// arrays + the rest state here). The de-interleave mirrors the legacy
// soft-upload / fluid-upload byte layout, so the device-staged particle /
// constraint tables (after env-major replication) reproduce the legacy world's
// layout — the byte-exact port contract for the XPBD/PBF 件套.
// ---------------------------------------------------------------------------

// One XPBD distance constraint (the de-interleaved XpbdDistanceConstraint).
struct CookDistanceCon { uint32_t a, b; float rest_length, compliance_alpha; };
// One XPBD bend constraint (4 particles + 4 cooked gradient vectors K_i).
struct CookBendCon { uint32_t p[4]; math::Vec3 k[4]; float compliance_alpha; };
// One XPBD volume constraint (4 particles + 6*rest_volume + compliance).
struct CookVolumeCon { uint32_t p[4]; float rest_volume_times6, compliance_alpha; };
// One XPBD shape-match cluster (; the de-interleaved XpbdShapeMatchCluster).
// Variable-size cluster pulled toward the rigid transform of its rest shape.
// particle[i] indexes the soft particle set; rest_positions[i] is x_i^0; the
// cluster weight m_i defaults to the particle's mass when omitted (see the cook).
struct CookShapeMatchCluster {
    std::vector<uint32_t>   particle;        // cluster particle indices (size n>=1)
    std::vector<math::Vec3> rest_positions;  // x_i^0, size n (same order)
    std::vector<float>      cluster_mass;    // m_i weight, size n (>0)
    float stiffness = 1.0f;                  // goal-pull fraction in [0,1].
};

struct XpbdCookInput {
    std::vector<math::Vec3>      positions;     // per-particle rest state
    std::vector<math::Vec3>      velocities;    // per-particle initial velocity
    std::vector<float>           inv_mass;      // 1/mass (0 == pinned)
    std::vector<CookDistanceCon> distance;
    std::vector<CookBendCon>     bend;
    std::vector<CookVolumeCon>   volume;
    std::vector<CookShapeMatchCluster> shape_match;  // (id 9).
    // Cloth aerodynamic-drag surface triangles (3 particle indices / triangle).
    // Empty => no drag (the drag op is inert; the cook stays byte-identical).
    std::vector<std::array<uint32_t, 3>> aero_triangles;
    // Lumped anisotropic air-drag coeffs (0.5*rho*Cn, 0.5*rho*Ct) + per-step
    // impulse clamp. All 0 => drag off. Cn >> Ct seeds flutter on a falling sheet.
    float    aero_drag_normal  = 0.0f;
    float    aero_drag_tangent = 0.0f;
    float    aero_drag_max_dv  = 0.0f;
    uint16_t solver_iterations = 1u;
    // body<->soft contact mu, finite so a foot grips/drags the cloth (solmix=max).
    float    friction = 0.6f;
    // Per-medium solver selection carried from the Scene IR SoftBodyComponent.
    // Xpbd (default) cooks the existing XPBD soft body byte-identically; Mpm
    // routes the cook to CookMpmParticles instead (ParticleMode::Mpm + material).
    nk::Model::ParticleMode solver = nk::Model::ParticleMode::Xpbd;
};

// Stage an XPBD soft body into the Model (single-env template; SeedInitialState
// replicates env-major). Sets particles_per_env / dist/bend/vol_cons_per_env.
void CookXpbdParticles(nk::Model& model, uint32_t env_count,
                       const XpbdCookInput& in);

// One MLS-MPM material the cook appends to model.mpm_materials (the constitutive
// branch, later, reads it). Mirrors nk::MpmMaterial 1:1.
struct MpmMaterialInput {
    float youngs = 0.0f, poisson = 0.0f, density = 0.0f;
    float dp_friction = 0.0f, dp_cohesion = 0.0f;
    // 0 = fixed-corotated elastic, 1 = Drucker-Prager (reserved), 2 = Neo-Hookean,
    // 3 = weakly-compressible fluid (Tait EOS, reads the three fields below).
    float model_kind = 0.0f;
    float bulk_modulus = 0.0f, tait_gamma = 0.0f, viscosity = 0.0f;
};

struct MpmCookInput {
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
};

// Stage an MLS-MPM bulk-soft body into the Model (single-env template; F=identity,
// C=0, vol0 from the lattice). Sets ParticleMode::Mpm + the grid/material caps;
// reuses the SAME particle-slot reserve as CookXpbdParticles (coupling one-path).
void CookMpmParticles(nk::Model& model, uint32_t env_count,
                      const MpmCookInput& in);

// Per-medium solver dispatch for a BULK-SOFT body (the only medium MPM admits).
// Routes by in.solver: Xpbd -> CookXpbdParticles (byte-identical default); Mpm ->
// CookMpmParticles built from `mpm` (the grid/material the XpbdCookInput lacks).
// `mpm` is consulted ONLY for the Mpm branch. Throws LOUDLY on an unsupported
// in.solver (anything but Xpbd/Mpm).
void CookSoftBodyParticles(nk::Model& model, uint32_t env_count,
                           const XpbdCookInput& in, const MpmCookInput& mpm);

// LOUD cook-time validation of a media list against the (kind x method) legal set
// (cloth = XPBD; tet-soft = XPBD or MLS-MPM; fluid = PBF or MLS-MPM) plus the rules
// that an MLS-MPM medium may not co-reside with a PBF medium (XPBD is legal) and a
// Model holds at most one PBF fluid slice / one MLS-MPM medium. Throws on an illegal
// pair or mix; an empty list and the existing legal cases pass (so every existing
// scene cooks byte-identically). The single call site is CookSceneMedia.
void ValidateMedia(const std::vector<MediaRecord>& media);

struct PbfCookInput {
    std::vector<math::Vec3> positions;     // CookFluidBox lattice
    std::vector<math::Vec3> velocities;    // initial velocity (usually zero)
    std::vector<float>      inv_mass;      // 1/mass (0 == pinned boundary)
    float    particle_mass   = 0.0f;       // uniform mass = rho0 * spacing^3
    float    rest_density    = 0.0f;       // rho0
    float    support_radius  = 0.0f;       // h (== grid cell size + query radius)
    float    cfm_epsilon     = 1.0e-6f;
    uint16_t iters           = 4u;
    bool     clamp_overdensity = true;
    float    xsph_viscosity  = 0.0f;
    float    surface_tension = 0.0f;
    // Uniform grid domain (the op rebuilds the grid each step over the predicted
    // positions but sizes it from this cooked AABB + cell size == support radius).
    math::Vec3 grid_min{0.0f, 0.0f, 0.0f};
    uint32_t   grid_dims[3] = {0u, 0u, 0u};
    bool       boundary_enabled = false;
    float      floor_z = 0.0f;             // z-up boundary floor.
    // body<->fluid contact mu, ~0 so a foot slides (splash stays normal/PBF-driven).
    float      friction = 0.0f;
    // Pinned-boundary confinement: walls_enabled appends boundary_layers floor-slab
    // layers + a side-wall ring (all inv_mass 0). Off => byte-identical fluid cook.
    bool       walls_enabled   = false;
    math::Vec3 walls_min{0.0f, 0.0f, 0.0f};
    math::Vec3 walls_max{0.0f, 0.0f, 0.0f};
    uint32_t   boundary_layers = 0u;
};

// Stage a PBF fluid into the Model (single-env template; SeedInitialState
// replicates env-major). Sets particles_per_env.
void CookPbfParticles(nk::Model& model, uint32_t env_count,
                      const PbfCookInput& in);

// Cross-system — cross-system contact params for the co-residence cook.
// The class-blind unilateral non-penetration co-step over the FULL [soft | fluid]
// union (op-ified cross-system particle co-step). d_min ==
// 2*contact_radius (uniform radius); <= 0 disables the op (the default). The cook
// widens the union grid query_radius/cell_size to >= d_min so the neighbor list
// covers every contact pair (the grid is built over the union by ParticleGridBuild).
struct SoftFluidContactInput {
    float    contact_d_min      = 0.0f;  // 2*contact_radius (<= 0 => no cross-contact)
    float    compliance_alpha   = 0.0f;  // XPBD alpha (0 == rigid)
    uint32_t solver_iterations  = 1u;    // Jacobi gather+apply sweeps (>= 1)
};

// Two-system cook: stage BOTH a soft (XPBD) particle set AND a fluid (PBF)
// particle set co-resident into ONE Model with a contiguous [soft | fluid] layout
// (the soft particles occupy [0, n_soft), the fluid [n_soft, particles_per_env)).
// The soft XPBD constraint indices stay in the soft slice; the fluid particles are
// appended AFTER the soft set, so the PBF density solve (scoped to [n_soft, P) by
// the SoftFluid ops) never sees a soft particle. Mode = ParticleMode::SoftFluid.
// STRICT SUPERSET: a soft-only cook (fluid empty) is byte-identical to
// CookXpbdParticles + the shape-match block; a fluid-only cook (soft empty,
// n_soft 0) is byte-identical to CookPbfParticles. The two single-system cooks
// remain the canonical paths; this is the co-residence composer. The optional
// `contact` block enables the cross-system non-penetration co-step over the
// union (default-off => behavior: the two slices co-step independently).
void CookSoftFluidParticles(nk::Model& model, uint32_t env_count,
                            const XpbdCookInput& soft, const PbfCookInput& fluid,
                            const SoftFluidContactInput& contact = {});

// Two-system cook: stage BOTH an MLS-MPM medium AND an XPBD (cloth/tet) set
// co-resident into ONE Model with a contiguous [mpm | xpbd] layout (MPM occupies
// [0, n_mpm), the XPBD set [n_mpm, particles_per_env)). The XPBD constraint indices
// are rebased by n_mpm; the MPM continuum fields stay sized to the MPM slice.
// Mode = ParticleMode::MpmXpbd. STRICT SUPERSET: an MPM-only cook (soft empty) is
// byte-identical to CookMpmParticles, an XPBD-only cook (mpm empty) to
// CookXpbdParticles, each plus the same body<->particle contact setup.
void CookMpmXpbd(nk::Model& model, uint32_t env_count, const MpmCookInput& mpm,
                 const XpbdCookInput& soft, const SoftFluidContactInput& contact = {});

// ---------------------------------------------------------------------------
// Media records -> particle cook. The per-medium builders read a scene
// MediaRecord and reuse the SAME constraint/lattice cookers
// (soft::BuildClothConstraints / import::cooker::CookFluidBox) -- never
// reimplemented. The dispatch concatenates XPBD media (cloth) into the soft slice
// and the one PBF fluid into the fluid slice via CookSoftFluidParticles, so the
// media list rides the EXISTING [soft | fluid] layout (no new ParticleMode).
// ---------------------------------------------------------------------------

// Build the XPBD cook input for a cloth MediaRecord: a flat (nx x ny) lattice at the
// authored origin (perimeter pinned unless free), meshed into triangles whose
// stretch+bend constraints come from soft::BuildClothConstraints. Empty when the
// grid extent is absent (nx<2 / ny<2 / spacing<=0).
XpbdCookInput BuildClothXpbdInput(const MediaRecord& media);

// Build the XPBD cook input for a tet-soft MediaRecord from its sphere rest-lattice
// (soft::BuildSphereTetLattice + BuildTetMeshConstraints). Empty when extent absent.
XpbdCookInput BuildSoftTetXpbdInput(const MediaRecord& media);

// Build the XPBD cook input for a cable MediaRecord: a distance chain of segments+1
// particles from start to end (inextensible unless distance_alpha > 0), the pinned
// endpoint(s) kinematic, an optional bend (skip-one) row set, and an optional rigid
// slab (a shape-match cluster welded to the loaded end). Empty when segments < 1 or
// radius <= 0. Mirrors BuildClothXpbdInput; slab corners follow the chain particles.
XpbdCookInput BuildCableXpbdInput(const MediaRecord& media);

// Build the MLS-MPM cook input for a tet-soft / fluid MediaRecord: particles sampled
// from the geometry (sphere lattice / CookFluidBox), material + grid from MediaMpmMaterial.
MpmCookInput BuildMpmInput(const MediaRecord& media);

// The cloth lattice render-surface triangle list (two triangles per quad, the SAME
// row-major winding BuildClothXpbdInput meshes the constraints with). Empty when the
// grid extent is absent. Indexes the [0, nx*ny) cloth particles (laid out first).
std::vector<uint32_t> BuildClothSurfaceTriangles(const MediaRecord& media);

// The per-medium render surfaces of a media list, base-offset to match
// CookSceneMedia's [soft|fluid] particle layout: cloth -> the lattice faces
// (BuildClothSurfaceTriangles), soft-tet -> the tet boundary faces
// (runtime::soft::ExtractBoundaryTriangles over the SAME sphere rest-lattice the
// XPBD cook meshes). A fluid (no triangulated surface) and a lone MLS-MPM medium (a
// dense sample, not the lattice vertices) yield no surface. ONE pass in the SAME
// medium order CookSceneMedia lays particles out -- the per-medium topology is data,
// so the live beauty render draws each surface uniformly (no cloth-vs-tet fork).
std::vector<MediaRenderSurface> BuildSceneMediaRenderSurfaces(
    const std::vector<MediaRecord>& media);

// Build the PBF cook input for a fluid MediaRecord: an AABB box filled on a uniform
// lattice by import::cooker::CookFluidBox, the uniform grid sized to enclose the box
// + headroom. Empty when the box extent is absent.
PbfCookInput BuildFluidPbfInput(const MediaRecord& media);

// Dispatch a validated media list onto an ALREADY-cooked Model: an all-XPBD/PBF list
// routes through CookSoftFluidParticles, a lone MLS-MPM medium through CookSoftBodyParticles.
void CookSceneMedia(nk::Model& model, uint32_t env_count,
                    const std::vector<MediaRecord>& media,
                    float particle_contact_radius = 0.0f);

// Orchestrator: the rigid/articulation CookToModel followed by the media-list
// dispatch over the SAME SceneIR (ONE path). A media-free scene is byte-identical to
// CookToModel; a scene carrying cloth/fluid media cooks them onto the rigid Model.
CookToModelResult CookSceneToModel(const SceneIR& scene, int env_count,
                                   const CookToModelOptions& options);

} // namespace nuka::scene::cook
