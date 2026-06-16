#pragma once
// ---------------------------------------------------------------------------
// nk CookToModel — SceneIR -> {nk::Model, SceneMap} (plan §3.3 / M3).
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
#include "scene/scene_ir.hpp"
#include "scene/scene_map.hpp"

#include <cstdint>
#include <vector>

namespace nuka::scene::cook {

struct CookToModelResult {
    nk::Model model;
    SceneMap  scene_map;
};

// Cook a SceneIR into an nk::Model replicated across `env_count` envs, plus the
// EntityId<->row SceneMap. env_count must be >= 1 (clamped to 1 if 0).
CookToModelResult CookToModel(const SceneIR& scene, int env_count);

// B1 (general contact pipeline Phase 1B): cook options. contact_family selects the
// stepped contact path. FusedFoot (the DEFAULT, used by the 2-arg overload above)
// keeps the legacy single-dog foot pipeline BYTE-IDENTICAL; PairDriven flips the
// world to the GENERAL LBVH -> cvx narrowphase -> mixed-island solve path (the
// multi-body / body-body cook). When PairDriven is set the per-env row budget is
// resized for the general per-candidate-slot row layout and the cook sets
// model.contact_family = PairDriven (so the pipeline gates the dog_dog hack OFF
// and routes the general assembly/solve). H1 grasp keeps UnionCsr (union_cook);
// single-dog go2 keeps FusedFoot (the 2-arg cook). Cross-env filtering is left ON
// for PairDriven multi-dog-in-one-env (intra-env collidables collide).
enum class CookContactFamily { FusedFoot, PairDriven };
struct CookToModelOptions {
    CookContactFamily contact_family = CookContactFamily::FusedFoot;
};
CookToModelResult CookToModel(const SceneIR& scene, int env_count,
                              const CookToModelOptions& options);

// ---------------------------------------------------------------------------
// M6 — particle cook (plan §3.10 "粒子 XPBD/PBF"). Stage an XPBD soft body or a
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
// One XPBD shape-match cluster (M9 T11; the de-interleaved XpbdShapeMatchCluster).
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
    std::vector<CookShapeMatchCluster> shape_match;  // M9 T11 (id 9).
    uint16_t solver_iterations = 1u;
};

// Stage an XPBD soft body into the Model (single-env template; SeedInitialState
// replicates env-major). Sets particles_per_env / dist/bend/vol_cons_per_env.
void CookXpbdParticles(nk::Model& model, uint32_t env_count,
                       const XpbdCookInput& in);

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
};

// Stage a PBF fluid into the Model (single-env template; SeedInitialState
// replicates env-major). Sets particles_per_env.
void CookPbfParticles(nk::Model& model, uint32_t env_count,
                      const PbfCookInput& in);

// M9 T11 Phase 2 — id-10 cross-system contact params for the co-residence cook.
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

// M9 T11 two-system cook: stage BOTH a soft (XPBD) particle set AND a fluid (PBF)
// particle set co-resident into ONE Model with a contiguous [soft | fluid] layout
// (the soft particles occupy [0, n_soft), the fluid [n_soft, particles_per_env)).
// The soft XPBD constraint indices stay in the soft slice; the fluid particles are
// appended AFTER the soft set, so the PBF density solve (scoped to [n_soft, P) by
// the SoftFluid ops) never sees a soft particle. Mode = ParticleMode::SoftFluid.
// STRICT SUPERSET: a soft-only cook (fluid empty) is byte-identical to
// CookXpbdParticles + the shape-match block; a fluid-only cook (soft empty,
// n_soft 0) is byte-identical to CookPbfParticles. The two single-system cooks
// remain the canonical paths; this is the co-residence composer. The optional
// `contact` block enables the id-10 cross-system non-penetration co-step over the
// union (default-off => Phase-1 behavior: the two slices co-step independently).
void CookSoftFluidParticles(nk::Model& model, uint32_t env_count,
                            const XpbdCookInput& soft, const PbfCookInput& fluid,
                            const SoftFluidContactInput& contact = {});

} // namespace nuka::scene::cook
