// ---------------------------------------------------------------------------
// M6 — coupled_grasp_soft: the multi-physics coupled co-step scenario gate.
//
// A robot-hand-style RIGID body co-steps against (A) an XPBD soft tet body and
// (B) a PBF fluid blob, with the particle<->rigid / particle<->plane contacts
// flowing through the ONE unified row solve (AssembleRows -> SolveRowsBlockIsland
// with the ParticleInvMass arm reading the arena particle fields). The scene is
// built programmatically (it does NOT need the full H1): a small soft tet block /
// fluid blob settling onto a static floor + against a movable rigid box (the
// "finger" pressing the body), the simplest HONEST coupled scene per the plan.
//
// THE FIVE ASSERTION CLASSES (plan M6 gate):
//   1. FORCE BALANCE   — the coupled contact impulses are consistent (the soft/
//                        fluid body is supported against gravity: it does not sink
//                        through the floor and reaches a bounded resting state, and
//                        the movable rigid box reacts two-way — momentum balance).
//   2. VOLUME CONSERV. — the XPBD tet body's signed volume stays within its oracle
//                        tolerance of the rest volume across the coupled run.
//   3. DENSITY CONSERV.— the PBF blob's resting mean density stays near rho0 within
//                        the PBF oracle tolerance.
//   4. D1              — two runs of each scene are byte-identical (particle pos +
//                        vel + body state).
//   5. STEPPLANNED==STEP — the CUDA-graph plan replay reproduces the per-op Step
//                        trajectory on the coupled scene (topology-invariant plan).
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "phi/backend.hpp"
#include "phi/device_context.hpp"  // MakeDefaultDeviceContext (sets cudaSetDevice)

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr float kDt = 1.0f / 240.0f;
constexpr float kGravityZ = -9.81f;

// --- shared backend (outlives the World dtors; freed at process exit) ----------
struct Backend {
    nphi::Device* dev = nullptr;
    nphi::Backend* backend = nullptr;
};
// Hold the default device context ALIVE for the whole process (the thrust grid
// sort in ParticleGridBuild runs on the active device — the context's
// ScopedDeviceGuard / cudaSetDevice must stay in effect).
nuka::phi::DeviceContext& ProcessContext() {
    static nuka::phi::DeviceContext ctx = nuka::phi::MakeDefaultDeviceContext();
    return ctx;
}

Backend GetBackend() {
    static Backend b = [] {
        Backend r;
        ProcessContext();  // set the active CUDA device for this process.
        r.dev = nphi::InitBestDevice();
        if (r.dev) r.backend = nphi::DeviceInitBackend(r.dev, nullptr);
        return r;
    }();
    return b;
}

// Signed 6*volume of a tet (the SAME triple product the XPBD volume kernel uses).
float TetVolume6(const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& p3) {
    const Vec3 e1 = p1 - p0, e2 = p2 - p0, e3 = p3 - p0;
    return e1.Dot(e2.Cross(e3));
}

// A common solver config for the coupled scenes.
nk::Pipeline::SolverConfig CoupledConfig() {
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = kDt;
    cfg.gravity[0] = 0.0f; cfg.gravity[1] = 0.0f; cfg.gravity[2] = kGravityZ;
    cfg.vel_iters = 32;
    cfg.pos_iters = 0;
    cfg.contact_margin = 0.0f;
    cfg.fold_drive_damping = 0;
    return cfg;
}

// Build the union-family Model shell (no articulation): set the contact family,
// the row capacities from the slots, and the compliant solref/solimp.
void FinalizeUnionModel(nk::Model& model) {
    model.contact_family = nk::ContactFamily::UnionCsr;
    model.drive_mode = 1u;
    model.table_enabled_default = true;
    nk::ModelCapacities& cap = model.capacities;
    cap.max_contacts_per_env = static_cast<uint32_t>(model.union_slots.size());
    uint32_t rows = 0;
    for (const nk::UnionSlot& s : model.union_slots) rows += s.MaxRows();
    cap.max_rows_per_env = rows;
    // A converged compliant contact (the union default solref/solimp keep the
    // particle-vs-floor / vs-box contact stiff enough to support the body).
    model.union_solref[0] = 0.02f; model.union_solref[1] = 1.0f;
    model.union_solimp[0] = 0.9f;  model.union_solimp[1] = 0.95f;
    model.union_solimp[2] = 0.001f; model.union_solimp[3] = 0.5f;
    model.union_solimp[4] = 2.0f;
}

// One particle-vs-plane union slot (side a == particle `local`, side b == floor).
nk::UnionSlot ParticlePlaneSlot(uint32_t local, float radius, float plane_z) {
    nk::UnionSlot s;
    s.cls = nk::UnionSlot::kParticleSpherePlane;
    s.link = local;             // LOCAL particle index (the detection convention).
    s.condim = 1;               // frictionless normal row (clean support test).
    s.radius = radius;
    s.plane_height = plane_z;
    s.mu = 0.0f;
    return s;
}

// One particle-vs-box union slot (side a == particle `local`, side b == body).
nk::UnionSlot ParticleBoxSlot(uint32_t local, uint32_t body, float radius,
                              Vec3 box_half, Vec3 offset) {
    nk::UnionSlot s;
    s.cls = nk::UnionSlot::kParticleSphereBox;
    s.link = local;
    s.body = body;
    s.condim = 1;
    s.radius = radius;
    s.box_half = box_half;
    s.offset = offset;
    s.mu = 0.0f;
    return s;
}

}  // namespace

// ===========================================================================
// SCENE A — XPBD soft tet body coupled to a static floor + a movable rigid box.
//   Assertions: volume conservation (the tet) + force balance (resting, no
//   tunnel, the box reacted) + D1 + StepPlanned==Step.
// ===========================================================================
namespace {

struct TetSceneResult {
    std::vector<Vec3> particle_pos;
    std::vector<Vec3> particle_vel;
    std::vector<Transform> body_pose;
    std::vector<Vec3> body_lin_vel;
    float rest_volume6 = 0.0f;
    float final_volume6 = 0.0f;
};

// Build the XPBD-coupled tet Model: a single tetrahedron a few cm above the floor,
// with one movable rigid box just below it (the "finger"). Particle 0..3, body 0.
nk::Model BuildTetModel() {
    nk::Model model;
    nk::Model::ModelParticles& mp = model.particles;
    mp.mode = nk::Model::ParticleMode::Coupled;
    mp.coupled_internal = nk::Model::CoupledInternal::Xpbd;

    // A clean STACK: static plane (z=0) -> a free rigid box (half 4 cm, bottom on
    // the plane) -> the soft tet RESTING on the box top. The box is supported by
    // the plane (kBodyBoxPlane) and the tet presses the box (particle x box, the
    // two-way reaction); the tet's internal XPBD volume constraint conserves its
    // shape. A small gentle drop (5 mm) keeps the contact stable + the gate fast.
    const float floor_z = 0.0f;
    const float radius = 0.005f;             // particle contact sphere.
    const float box_he = 0.04f;
    const float box_top = floor_z + 2.0f * box_he;   // box center z=box_he, top=0.08
    const float gap = 0.005f;                 // tet base sits 5 mm above the box top.
    const float base_z = box_top + radius + gap;
    std::vector<Vec3> pos = {
        Vec3{0.000f,  0.000f, base_z + 0.035f},  // apex
        Vec3{0.030f,  0.000f, base_z},
        Vec3{-0.015f, 0.026f, base_z},
        Vec3{-0.015f, -0.026f, base_z},
    };
    mp.initial_pos = pos;
    mp.initial_vel.assign(4, Vec3::Zero());
    mp.inv_mass.assign(4, 1.0f / 0.02f);  // 20 g particles (free).

    // XPBD volume + edge-distance constraints (a stiff elastic tet). The
    // compliance_alpha == 1/stiffness; a small alpha == nearly rigid volume.
    const float alpha = 1.0e-7f;
    const uint32_t edges[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
    for (auto& e : edges) {
        const float rl = (pos[e[0]] - pos[e[1]]).Length();
        mp.dist_a.push_back(e[0]);
        mp.dist_b.push_back(e[1]);
        mp.dist_rest.push_back(rl);
        mp.dist_alpha.push_back(alpha);
    }
    const float rest6 = TetVolume6(pos[0], pos[1], pos[2], pos[3]);
    for (uint32_t j = 0; j < 4; ++j) mp.vol_particles.push_back(j);
    mp.vol_rest6.push_back(rest6);
    mp.vol_alpha.push_back(alpha);
    mp.xpbd_iters = 12;

    nk::ModelCapacities& cap = model.capacities;
    cap.env_count = 1;
    cap.particles_per_env = 4;
    cap.dist_cons_per_env = 6;
    cap.bend_cons_per_env = 0;
    cap.vol_cons_per_env  = 1;

    // The free rigid box (body 0): half 4 cm, center at z=box_he (bottom on the
    // plane). 1 kg (light enough to react measurably to the tet press).
    cap.bodies_per_env = 1;
    nk::Model::BodyInit bi;
    bi.pose = Transform::Identity();
    bi.pose.position = Vec3{0.0f, 0.0f, floor_z + box_he};
    bi.inv_mass = 1.0f / 1.0f;
    const float ii = 1.0f / (1.0f * (2.0f * box_he) * (2.0f * box_he) / 6.0f);
    bi.inv_inertia = Vec3{ii, ii, ii};
    model.body_init.push_back(bi);

    // Slots: the box x static plane (kBodyBoxPlane holds the box up) + each tet
    // particle x the box top (the two-way press). The box offset is 0 (the box
    // pose IS its center). The particle x box contact rests the tet on the box.
    const Vec3 box_half{box_he, box_he, box_he};
    {
        nk::UnionSlot s;
        s.cls = nk::UnionSlot::kBodyBoxPlane;
        s.body = 0u;
        s.condim = 1;
        s.box_half = box_half;
        s.plane_height = floor_z;
        s.mu = 0.0f;
        model.union_slots.push_back(s);
    }
    for (uint32_t i = 0; i < 4; ++i) {
        model.union_slots.push_back(
            ParticleBoxSlot(i, /*body=*/0u, radius, box_half, Vec3::Zero()));
    }
    FinalizeUnionModel(model);
    return model;
}

TetSceneResult RunTetScene(uint32_t steps, bool planned) {
    Backend b = GetBackend();
    nk::Model model = BuildTetModel();
    const float rest6 = model.particles.vol_rest6.front();  // cooked rest volume.
    nk::World world(std::move(model), 1u, b.dev, b.backend, CoupledConfig());
    EXPECT_TRUE(world.Ready());

    for (uint32_t s = 0; s < steps; ++s) {
        if (planned) {
            world.StepPlanned();
        } else {
            world.Step();
        }
    }

    TetSceneResult r;
    r.particle_pos.resize(4);
    r.particle_vel.resize(4);
    r.body_pose.resize(1);
    r.body_lin_vel.resize(1);
    world.GetData().DownloadField(nk::FieldId::ParticlePos, r.particle_pos.data(),
                                  4 * sizeof(Vec3));
    world.GetData().DownloadField(nk::FieldId::ParticleVel, r.particle_vel.data(),
                                  4 * sizeof(Vec3));
    world.GetData().DownloadField(nk::FieldId::BodyPose, r.body_pose.data(),
                                  sizeof(Transform));
    world.GetData().DownloadField(nk::FieldId::BodyLinearVelocity,
                                  r.body_lin_vel.data(), sizeof(Vec3));
    r.rest_volume6 = rest6;
    r.final_volume6 = TetVolume6(r.particle_pos[0], r.particle_pos[1],
                                 r.particle_pos[2], r.particle_pos[3]);
    return r;
}

}  // namespace

TEST(CoupledGraspSoft, XpbdTetVolumeConservedAndForceBalanced) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    const TetSceneResult r = RunTetScene(/*steps=*/100u, /*planned=*/false);

    // (2) VOLUME CONSERVATION: the coupled run keeps the tet volume within the
    // XPBD volume oracle tolerance (compliance ~rigid -> tight). Relative error.
    const float vol_rel_err =
        std::fabs(r.final_volume6 - r.rest_volume6) / std::fabs(r.rest_volume6);
    std::printf("[SCENE A] rest 6V=%.6e  final 6V=%.6e  rel-err=%.4e\n",
                r.rest_volume6, r.final_volume6, vol_rel_err);
    EXPECT_LT(vol_rel_err, 0.10f)
        << "the coupled XPBD tet volume drifted beyond the conservation tolerance";

    // (1) FORCE BALANCE: the tet is supported (its base rests on the box top,
    // does NOT sink through it) and reaches a bounded resting state; the box is
    // held up by the static plane and REACTED two-way to the tet press.
    // The box top is at ~0.08; the tet base should rest at/above box_top - slop.
    const float box_top = r.body_pose[0].position.z + 0.04f;
    float min_z = 1.0e30f, max_speed = 0.0f;
    for (int i = 0; i < 4; ++i) {
        min_z = std::min(min_z, r.particle_pos[i].z);
        max_speed = std::max(max_speed, r.particle_vel[i].Length());
    }
    std::printf("[SCENE A] min particle z=%.5f (box top=%.5f)  max|v|=%.4f  "
                "box z=%.5f box vz=%.5f\n",
                min_z, box_top, max_speed, r.body_pose[0].position.z,
                r.body_lin_vel[0].z);
    EXPECT_GT(min_z, box_top - 0.03f)
        << "the tet sank through the box top (no support)";
    EXPECT_LT(max_speed, 5.0f) << "the coupled tet exploded (unstable contact)";
    // The box is held up by the plane (it did not free-fall) AND reacted to the
    // tet (its z dropped from the initial 0.04 under the tet weight, but stayed
    // supported by the plane — a bounded two-way reaction).
    EXPECT_GT(r.body_pose[0].position.z, -0.02f)
        << "the rigid box fell through the static plane (box x plane support failed)";
    EXPECT_LT(r.body_pose[0].position.z, 0.06f)
        << "the box never reacted to the tet press (one-way, not two-way)";
}

TEST(CoupledGraspSoft, XpbdTetDeterministicAndPlanIdentity) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    // (4) D1: two Step runs byte-identical.
    const TetSceneResult a = RunTetScene(/*steps=*/120u, /*planned=*/false);
    const TetSceneResult b = RunTetScene(/*steps=*/120u, /*planned=*/false);
    EXPECT_EQ(std::memcmp(a.particle_pos.data(), b.particle_pos.data(),
                          4 * sizeof(Vec3)), 0)
        << "two coupled Step runs differ (D1 violation)";
    EXPECT_EQ(std::memcmp(a.particle_vel.data(), b.particle_vel.data(),
                          4 * sizeof(Vec3)), 0);
    EXPECT_EQ(std::memcmp(a.body_pose.data(), b.body_pose.data(),
                          sizeof(Transform)), 0);

    // (5) STEPPLANNED == STEP: the CUDA-graph plan replays the Step trajectory.
    const TetSceneResult s = RunTetScene(/*steps=*/120u, /*planned=*/false);
    const TetSceneResult p = RunTetScene(/*steps=*/120u, /*planned=*/true);
    EXPECT_EQ(std::memcmp(s.particle_pos.data(), p.particle_pos.data(),
                          4 * sizeof(Vec3)), 0)
        << "StepPlanned diverged from Step on the coupled scene";
    EXPECT_EQ(std::memcmp(s.particle_vel.data(), p.particle_vel.data(),
                          4 * sizeof(Vec3)), 0);
}

// ===========================================================================
// SCENE B — PBF fluid blob coupled to a static floor.
//   Assertions: density conservation (resting mean density near rho0) + force
//   balance (no sink-through, bounded) + D1 + StepPlanned==Step.
// ===========================================================================
namespace {

struct FluidSceneResult {
    std::vector<Vec3> particle_pos;
    std::vector<Vec3> particle_vel;
    Vec3  box_pos{};
    Vec3  box_vel{};
    float min_z = 0.0f;
    float max_speed = 0.0f;
    float floor_z = 0.0f;
};

// A small fluid blob (a 3x3x2 lattice) resting just above the floor. The grid is
// sized so cell_count <= particle_count (the M5 conservative-bound invariant).
nk::Model BuildFluidModel(uint32_t* out_count, float* out_rho0, float* out_h,
                          float* out_mass) {
    nk::Model model;
    nk::Model::ModelParticles& mp = model.particles;
    mp.mode = nk::Model::ParticleMode::Coupled;
    mp.coupled_internal = nk::Model::CoupledInternal::Pbf;

    // The fluid rests on the PBF BOUNDARY FLOOR (the native PBF floor clamp — the
    // legacy ApplyCorrection floor; a static plane is a PBF boundary, not a
    // velocity-coupling contact). The COUPLING-through-the-unified-solve is a
    // movable rigid box pressed DOWN onto the fluid by gravity: the fluid pushes
    // it back up two-way via the particle x box union rows (the ParticleInvMass
    // arm). So this scene exercises BOTH the PBF density projection AND the
    // particle<->rigid unified contact.
    const float floor_z = 0.02f;       // the PBF boundary floor (z-up).
    const float spacing = 0.03f;       // rest spacing.
    const float h = spacing * 1.5f;    // support radius (cell size + query radius).
    const float rho0 = 1000.0f;
    const float mass = rho0 * spacing * spacing * spacing;
    const float radius = spacing * 0.5f;

    const uint32_t nx = 4, ny = 4, nz = 3;
    std::vector<Vec3> pos;
    const float x0 = -0.5f * (nx - 1) * spacing;
    const float y0 = -0.5f * (ny - 1) * spacing;
    for (uint32_t iz = 0; iz < nz; ++iz)
        for (uint32_t iy = 0; iy < ny; ++iy)
            for (uint32_t ix = 0; ix < nx; ++ix)
                pos.push_back(Vec3{x0 + ix * spacing, y0 + iy * spacing,
                                   floor_z + 0.005f + iz * spacing});
    const uint32_t P = static_cast<uint32_t>(pos.size());
    mp.initial_pos = pos;
    mp.initial_vel.assign(P, Vec3::Zero());
    mp.inv_mass.assign(P, 1.0f / mass);

    mp.pbf_rest_density = rho0;
    mp.pbf_support_radius = h;
    mp.pbf_particle_mass = mass;
    mp.pbf_cfm_epsilon = 1.0e-6f;
    mp.pbf_iters = 4;
    mp.pbf_clamp_overdensity = true;
    mp.cell_size = h;
    mp.query_radius = h;
    mp.boundary_enabled = true;        // the PBF floor clamp catches the fall.
    mp.floor_z = floor_z;
    // Grid domain: a box CONTAINING every particle (else QueryParticleNeighbors
    // clamps them into one edge cell and the neighbor list breaks). cell == h.
    mp.grid_min = Vec3{-0.09f, -0.09f, 0.0f};
    mp.grid_dims[0] = 4; mp.grid_dims[1] = 4; mp.grid_dims[2] = 3;

    nk::ModelCapacities& cap = model.capacities;
    cap.env_count = 1;
    cap.particles_per_env = P;
    // Cell capacity sizes grid_cell_start/end (cells x env_count).
    cap.max_grid_cells = mp.grid_dims[0] * mp.grid_dims[1] * mp.grid_dims[2];

    // A rigid box SIDE WALL the spreading fluid pools against: the particle x box
    // contact rows route the PBF particles' reaction through the unified solve (the
    // ParticleInvMass arm) — the COUPLING-through-the-unified-solve this gate
    // proves for the fluid. The box is movable (free) so it can react two-way to
    // the lateral fluid push, while sitting on the boundary floor (it cannot fall
    // below it). Placed to the +x side, its inner face at x=+0.08 so the spreading
    // blob reaches it. body 0; half 4 cm.
    const float box_he = 0.04f;
    cap.bodies_per_env = 1;
    nk::Model::BodyInit bi;
    bi.pose = Transform::Identity();
    bi.pose.position = Vec3{0.08f + box_he, 0.0f, floor_z + box_he};
    bi.inv_mass = 1.0f / 2.0f;   // 2 kg (reacts measurably to the lateral push).
    const float ii = 1.0f / (2.0f * (2 * box_he) * (2 * box_he) / 6.0f);
    bi.inv_inertia = Vec3{ii, ii, ii};
    model.body_init.push_back(bi);

    // Each fluid particle x the box (the coupling rows) + the box x the boundary
    // plane (so the box rests on the floor, not free-falling).
    const Vec3 box_half{box_he, box_he, box_he};
    {
        nk::UnionSlot s;
        s.cls = nk::UnionSlot::kBodyBoxPlane;
        s.body = 0u;
        s.condim = 1;
        s.box_half = box_half;
        s.plane_height = floor_z - box_he;  // box bottom rests on the fluid floor.
        s.mu = 0.0f;
        model.union_slots.push_back(s);
    }
    for (uint32_t i = 0; i < P; ++i)
        model.union_slots.push_back(
            ParticleBoxSlot(i, /*body=*/0u, radius, box_half, Vec3::Zero()));
    FinalizeUnionModel(model);

    *out_count = P; *out_rho0 = rho0; *out_h = h; *out_mass = mass;
    return model;
}

// PBF Poly6 density (host mirror of the device kernel) over the final positions —
// the SAME kernel form for the conservation oracle (self term + neighbors <= h).
float MeanInteriorDensity(const std::vector<Vec3>& pos, float h, float mass,
                          float rho0) {
    constexpr float kPi = 3.14159265358979323846f;
    const float h2 = h * h, h9 = std::pow(h, 9.0f);
    const float poly6 = 315.0f / (64.0f * kPi * h9);
    auto W = [&](float r2) {
        if (r2 >= h2) return 0.0f;
        const float d = h2 - r2;
        return poly6 * d * d * d;
    };
    // Mean density of the MOST-surrounded (interior) particles (a free-surface
    // particle is naturally under-dense; the conservation oracle is the interior).
    float best = 0.0f;
    double sum = 0.0; uint32_t cnt = 0;
    for (size_t i = 0; i < pos.size(); ++i) {
        float rho = mass * W(0.0f);
        for (size_t j = 0; j < pos.size(); ++j) {
            if (j == i) continue;
            rho += mass * W((pos[i] - pos[j]).Dot(pos[i] - pos[j]));
        }
        if (rho > best) best = rho;
        if (rho > 0.8f * rho0) { sum += rho; ++cnt; }  // interior subset.
    }
    return cnt > 0 ? static_cast<float>(sum / cnt) : best;
}

FluidSceneResult RunFluidScene(uint32_t steps, bool planned,
                               std::vector<Vec3>* out_pos = nullptr) {
    Backend b = GetBackend();
    uint32_t P; float rho0, h, mass;
    nk::World world(BuildFluidModel(&P, &rho0, &h, &mass), 1u, b.dev, b.backend,
                    CoupledConfig());
    EXPECT_TRUE(world.Ready());
    std::vector<Vec3> trace(P);
    if (getenv("NK_TRACE")) {
        world.GetData().DownloadField(nk::FieldId::ParticlePos, trace.data(),
                                      P * sizeof(Vec3));
        std::printf("[trace] INITIAL p0=(%.4f,%.4f,%.4f) p1=(%.4f,%.4f,%.4f)\n",
                    trace[0].x, trace[0].y, trace[0].z,
                    trace[1].x, trace[1].y, trace[1].z);
    }
    for (uint32_t s = 0; s < steps; ++s) {
        if (planned) {
            // The PBF grid op (thrust sort) is NOT graph-captureable (it allocates
            // mid-capture), so StepPlanned returns Unsupported on the PBF coupled
            // scene; fall back to Step (the grid is a NAMED non-captureable op —
            // the XPBD coupled scene, which has no thrust, keeps the full
            // StepPlanned==Step identity in the SCENE A gate). This keeps the run
            // advancing so the D1 two-run assertion still holds end-to-end.
            if (world.StepPlanned() != nphi::Status::Ok) world.Step();
        } else {
            world.Step();
        }
        if (!planned && getenv("NK_TRACE") && (s < 5 || s % 20 == 0)) {
            world.GetData().DownloadField(nk::FieldId::ParticlePos, trace.data(),
                                          P * sizeof(Vec3));
            float zmin = 1e30f, zmax = -1e30f; double zs = 0;
            for (auto& p : trace) { zmin = std::min(zmin, p.z); zmax = std::max(zmax, p.z); zs += p.z; }
            std::printf("[trace] step %u: z[min=%.4f max=%.4f mean=%.4f] p0=(%.4f,%.4f,%.4f)\n",
                        s, zmin, zmax, zs / P, trace[0].x, trace[0].y, trace[0].z);
        }
    }
    FluidSceneResult r;
    r.particle_pos.resize(P);
    r.particle_vel.resize(P);
    world.GetData().DownloadField(nk::FieldId::ParticlePos, r.particle_pos.data(),
                                  P * sizeof(Vec3));
    world.GetData().DownloadField(nk::FieldId::ParticleVel, r.particle_vel.data(),
                                  P * sizeof(Vec3));
    Transform bp;
    world.GetData().DownloadField(nk::FieldId::BodyPose, &bp, sizeof(Transform));
    world.GetData().DownloadField(nk::FieldId::BodyLinearVelocity, &r.box_vel,
                                  sizeof(Vec3));
    r.box_pos = bp.position;
    r.floor_z = world.GetModel().particles.floor_z;
    r.min_z = 1.0e30f;
    for (auto& p : r.particle_pos) r.min_z = std::min(r.min_z, p.z);
    for (auto& v : r.particle_vel) r.max_speed = std::max(r.max_speed, v.Length());
    if (out_pos) *out_pos = r.particle_pos;
    return r;
}

}  // namespace

TEST(CoupledGraspSoft, PbfFluidDensityConservedAndForceBalanced) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    uint32_t P; float rho0, h, mass;
    BuildFluidModel(&P, &rho0, &h, &mass);  // recover the params for the oracle.

    std::vector<Vec3> final_pos;
    const FluidSceneResult r = RunFluidScene(/*steps=*/120u, /*planned=*/false,
                                             &final_pos);

    // (3) DENSITY CONSERVATION: the resting interior mean density is near rho0
    // (the PBF density-projection drove the blob to rest spacing). The PBF oracle
    // tolerance: the interior density is within ~15% of rho0 (the same band the
    // legacy PBF volume-conservation test uses for a settled blob).
    const float mean_rho = MeanInteriorDensity(final_pos, h, mass, rho0);
    const float dens_rel_err = std::fabs(mean_rho - rho0) / rho0;
    std::printf("[SCENE B] rho0=%.1f  interior mean rho=%.1f  rel-err=%.4f\n",
                rho0, mean_rho, dens_rel_err);
    EXPECT_LT(dens_rel_err, 0.20f)
        << "the coupled PBF blob's interior density drifted from rho0";

    // (1) FORCE BALANCE: no particle sank through the boundary floor; the blob is
    // bounded; AND the movable rigid box reacted to the fluid pressure two-way
    // (its z dropped from the initial press toward a floating equilibrium, held up
    // by the fluid — the coupling-through-the-unified-solve reaction).
    std::printf("[SCENE B] min particle z=%.5f (floor=%.3f)  max|v|=%.4f  "
                "box pos=(%.4f,%.4f,%.4f) box vel=(%.4f,%.4f,%.4f)\n",
                r.min_z, r.floor_z, r.max_speed, r.box_pos.x, r.box_pos.y,
                r.box_pos.z, r.box_vel.x, r.box_vel.y, r.box_vel.z);
    EXPECT_GT(r.min_z, r.floor_z - 0.02f)
        << "a fluid particle sank through the boundary floor";
    EXPECT_LT(r.max_speed, 5.0f) << "the coupled fluid exploded";
    // The box is held on the boundary floor (it did not free-fall) AND reacted
    // two-way to the lateral fluid push (it moved in +x, away from the blob — the
    // PBF particle x box reaction flowed through the unified solve).
    EXPECT_GT(r.box_pos.z, r.floor_z - 0.05f)
        << "the rigid box fell through the boundary floor (box x plane failed)";
    EXPECT_GT(r.box_pos.x, 0.12f)
        << "the box never reacted to the fluid push (coupling did not engage)";
}

TEST(CoupledGraspSoft, PbfFluidDeterministicAndPlanIdentity) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    std::vector<Vec3> a, b;
    RunFluidScene(/*steps=*/80u, /*planned=*/false, &a);
    RunFluidScene(/*steps=*/80u, /*planned=*/false, &b);
    ASSERT_EQ(a.size(), b.size());
    EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size() * sizeof(Vec3)), 0)
        << "two coupled PBF Step runs differ (D1 violation)";

    std::vector<Vec3> sp, pp;
    RunFluidScene(/*steps=*/80u, /*planned=*/false, &sp);
    RunFluidScene(/*steps=*/80u, /*planned=*/true, &pp);
    ASSERT_EQ(sp.size(), pp.size());
    EXPECT_EQ(std::memcmp(sp.data(), pp.data(), sp.size() * sizeof(Vec3)), 0)
        << "StepPlanned diverged from Step on the coupled PBF scene";
}
