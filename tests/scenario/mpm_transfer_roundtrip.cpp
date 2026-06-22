// ---------------------------------------------------------------------------
// MLS-MPM transfer ROUND-TRIP IDENTITY gate (the inert-scaffold proof).
//
// Builds a small MLS-MPM particle set + an env-private background grid, seeds
// C=0 / F=I / zero grid force / a known particle velocity, and dispatches
// MpmGridClear -> MpmP2G -> MpmG2P DIRECTLY through the backend (the transfer-only
// path; NOT the step loop). Asserts:
//   (1) the G2P-recovered velocity reproduces the seeded velocity (the APIC
//       transfer with C=0 is the identity on a constant velocity field);
//   (2) MpmP2G run twice is BIT-identical (the deterministic gather, NO atomics);
//   (3) the snapshot/restore round-trips F/C (the three readout.cu sites);
//   (4) a particle outside the grid AABB flags the env-status escape bit.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "phi/backend.hpp"
#include "phi/op_schema.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
using nuka::math::Vec3;

struct Backend {
    nphi::Device* dev = nullptr;
    nphi::Backend* backend = nullptr;
};
Backend GetBackend() {
    static Backend b = [] {
        Backend r;
        r.dev = nphi::InitBestDevice();
        if (r.dev) r.backend = nphi::DeviceInitBackend(r.dev, nullptr);
        return r;
    }();
    return b;
}

nk::Pipeline::SolverConfig Cfg() {
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[0] = 0.0f; cfg.gravity[1] = 0.0f; cfg.gravity[2] = -9.81f;
    return cfg;
}

constexpr uint32_t kDim = 6u;   // 6^3 = 216 nodes/env.
constexpr float    kDx  = 0.1f;
const Vec3 kOrigin{0.0f, 0.0f, 0.0f};

// A small block of MLS-MPM particles well inside the grid AABB, each with the SAME
// (constant) velocity so the APIC C=0 transfer is the exact identity. The grid is
// kDim^3 nodes at spacing kDx from kOrigin.
nk::Model BuildMpmModel(const Vec3& seed_vel, bool escape = false) {
    nk::Model m;
    nk::Model::ModelParticles& mp = m.particles;
    mp.mode = nk::Model::ParticleMode::Mpm;
    // Particles inside [0.2, 0.4] (cells 2..4) so the full 3^3 stencil stays in grid.
    std::vector<Vec3> pos;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k)
                pos.push_back(Vec3{0.25f + 0.05f * i, 0.25f + 0.05f * j,
                                   0.25f + 0.05f * k});
    if (escape) pos.push_back(Vec3{5.0f, 5.0f, 5.0f});  // outside the grid AABB.
    const size_t n = pos.size();
    mp.initial_pos = pos;
    mp.initial_vel.assign(n, seed_vel);
    mp.inv_mass.assign(n, 1.0f);  // unit mass.
    mp.initial_F.assign(n * 9u, 0.0f);
    for (size_t p = 0; p < n; ++p) {
        mp.initial_F[p * 9u + 0u] = 1.0f;
        mp.initial_F[p * 9u + 4u] = 1.0f;
        mp.initial_F[p * 9u + 8u] = 1.0f;
    }
    mp.initial_vol0.assign(n, kDx * kDx * kDx);
    mp.initial_material_id.assign(n, 0u);

    nk::MpmMaterial mat;  // one elastic material with recognizable nonzero values.
    mat.youngs = 1.0e4f; mat.poisson = 0.3f; mat.density = 1000.0f;
    m.mpm_materials = {mat};

    nk::ModelCapacities& cap = m.capacities;
    cap.particles_per_env = static_cast<uint32_t>(n);
    cap.mpm_grid_nodes_per_env = kDim * kDim * kDim;
    cap.mpm_material_count = 1u;
    mp.mpm_grid_min = kOrigin;
    mp.mpm_grid_dims[0] = kDim; mp.mpm_grid_dims[1] = kDim; mp.mpm_grid_dims[2] = kDim;
    mp.mpm_cell_size = kDx;
    return m;
}

nphi::MpmGridParams MakeParams(const nk::Model& m) {
    nphi::MpmGridParams p{};
    p.particle_count = m.capacities.particles_per_env;  // env_count == 1.
    p.particles_per_env = m.capacities.particles_per_env;
    p.env_count = 1u;
    p.nodes_per_env = m.capacities.mpm_grid_nodes_per_env;
    p.grid_dims[0] = kDim; p.grid_dims[1] = kDim; p.grid_dims[2] = kDim;
    p.grid_origin[0] = kOrigin.x; p.grid_origin[1] = kOrigin.y;
    p.grid_origin[2] = kOrigin.z;
    p.dx = kDx;
    p.dt = 1.0f / 240.0f;
    return p;
}

bool RunTransfer(nk::World& w, const nphi::MpmGridParams& p) {
    return w.DispatchOp(nphi::NkOp::MpmGridClear, &p) == nphi::Status::Ok &&
           w.DispatchOp(nphi::NkOp::MpmP2G, &p) == nphi::Status::Ok &&
           w.DispatchOp(nphi::NkOp::MpmG2P, &p) == nphi::Status::Ok;
}

}  // namespace

// (1) round-trip identity: clear -> P2G -> G2P with C=0/F=I reproduces velocity.
TEST(MpmTransferRoundtrip, P2GThenG2PReproducesVelocity) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();
    const Vec3 seed{0.7f, -0.3f, 1.1f};
    nk::Model m = BuildMpmModel(seed);
    const uint32_t np = m.capacities.particles_per_env;

    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
    ASSERT_TRUE(w.Ready());
    nphi::MpmGridParams p = MakeParams(w.GetModel());
    ASSERT_TRUE(RunTransfer(w, p));

    std::vector<Vec3> out(np, Vec3::Zero());
    ASSERT_TRUE(w.GetData().DownloadField(nk::FieldId::ParticleVel, out.data(),
                                          out.size() * sizeof(Vec3)));
    float max_err = 0.0f;
    for (uint32_t i = 0; i < np; ++i) {
        max_err = std::max(max_err, std::fabs(out[i].x - seed.x));
        max_err = std::max(max_err, std::fabs(out[i].y - seed.y));
        max_err = std::max(max_err, std::fabs(out[i].z - seed.z));
        EXPECT_FALSE(std::isnan(out[i].x) || std::isnan(out[i].y) ||
                     std::isnan(out[i].z));
    }
    std::fprintf(stderr, "[mpm-roundtrip] np=%u max_err=%.3e\n", np, max_err);
    EXPECT_LE(max_err, 1.0e-5f)
        << "APIC C=0 transfer must reproduce the constant velocity field";
}

// (2) the deterministic gather is byte-identical run-to-run (NO float atomics).
TEST(MpmTransferRoundtrip, P2GGatherByteIdenticalRunToRun) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();

    auto run = [&](std::vector<float>& mass, std::vector<Vec3>& mom) -> bool {
        nk::Model m = BuildMpmModel(Vec3{0.5f, 0.5f, 0.5f});
        const uint32_t nodes = m.capacities.mpm_grid_nodes_per_env;
        nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
        if (!w.Ready()) return false;
        nphi::MpmGridParams p = MakeParams(w.GetModel());
        if (w.DispatchOp(nphi::NkOp::MpmGridClear, &p) != nphi::Status::Ok) return false;
        if (w.DispatchOp(nphi::NkOp::MpmP2G, &p) != nphi::Status::Ok) return false;
        mass.assign(nodes, 0.0f);
        mom.assign(nodes, Vec3::Zero());
        return w.GetData().DownloadField(nk::FieldId::GridMass, mass.data(),
                                         mass.size() * sizeof(float)) &&
               w.GetData().DownloadField(nk::FieldId::GridMomentum, mom.data(),
                                         mom.size() * sizeof(Vec3));
    };
    std::vector<float> m1, m2;
    std::vector<Vec3> p1, p2;
    ASSERT_TRUE(run(m1, p1));
    ASSERT_TRUE(run(m2, p2));
    EXPECT_EQ(0, std::memcmp(m1.data(), m2.data(), m1.size() * sizeof(float)))
        << "grid mass differs across runs (atomic scatter would)";
    EXPECT_EQ(0, std::memcmp(p1.data(), p2.data(), p1.size() * sizeof(Vec3)))
        << "grid momentum differs across runs (atomic scatter would)";
    // Sanity: total mass == particle count (unit masses, partition of unity).
    double total = 0.0;
    for (float v : m1) total += v;
    std::fprintf(stderr, "[mpm-gather] total_mass=%.6f nodes=%zu\n", total, m1.size());
    EXPECT_NEAR(total, static_cast<double>(m1.empty() ? 0 : 8), 1.0e-3);
}

// (3) snapshot/restore round-trips F/C: mutate F on device, restore, F == cooked I.
TEST(MpmTransferRoundtrip, SnapshotRestoreRoundTripsF) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();
    nk::Model m = BuildMpmModel(Vec3{0.0f, 0.0f, 0.0f});
    const uint32_t np = m.capacities.particles_per_env;
    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
    ASSERT_TRUE(w.Ready());  // ctor takes the construction-time snapshot (F=I).

    // Clobber the live F with garbage, then Reset (the bulk RestoreState) should
    // recover the cooked identity from the construction-time snapshot.
    std::vector<float> junk(static_cast<size_t>(np) * 9u, 7.0f);
    ASSERT_TRUE(w.GetData().UploadField(nk::FieldId::ParticleF, junk.data(),
                                        junk.size() * sizeof(float)));
    ASSERT_EQ(nphi::Status::Ok, w.Reset());

    std::vector<float> F(static_cast<size_t>(np) * 9u, 0.0f);
    ASSERT_TRUE(w.GetData().DownloadField(nk::FieldId::ParticleF, F.data(),
                                          F.size() * sizeof(float)));
    for (uint32_t p = 0; p < np; ++p) {
        const float* f = F.data() + static_cast<size_t>(p) * 9u;
        EXPECT_FLOAT_EQ(f[0], 1.0f); EXPECT_FLOAT_EQ(f[4], 1.0f); EXPECT_FLOAT_EQ(f[8], 1.0f);
        EXPECT_FLOAT_EQ(f[1], 0.0f); EXPECT_FLOAT_EQ(f[5], 0.0f); EXPECT_FLOAT_EQ(f[7], 0.0f);
    }
}

// Run clear -> P2G and return the env-status word (escape-bit probe).
uint32_t EscapeStatusAfterP2G(Backend& b, nk::Model m) {
    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
    EXPECT_TRUE(w.Ready());
    nphi::MpmGridParams p = MakeParams(w.GetModel());
    EXPECT_EQ(nphi::Status::Ok, w.DispatchOp(nphi::NkOp::MpmGridClear, &p));
    EXPECT_EQ(nphi::Status::Ok, w.DispatchOp(nphi::NkOp::MpmP2G, &p));
    uint32_t status = 0u;
    EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::EnvStatus, &status,
                                          sizeof(uint32_t)));
    return status;
}

// (4) the escape bit is two-sided: an all-in-grid set leaves it CLEAR, an out-of-
// AABB particle SETS it (a kernel that always-sets or never-sets is caught).
TEST(MpmTransferRoundtrip, OutOfGridParticleFlagsEnvStatus) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();
    const uint32_t in_grid =
        EscapeStatusAfterP2G(b, BuildMpmModel(Vec3{0.0f, 0.0f, 0.0f}, /*escape=*/false));
    EXPECT_EQ(0u, in_grid & nphi::kEnvStatusMpmGridEscape)
        << "an all-in-grid set must NOT flag the escape bit (always-set kernel caught)";
    const uint32_t escaped =
        EscapeStatusAfterP2G(b, BuildMpmModel(Vec3{0.0f, 0.0f, 0.0f}, /*escape=*/true));
    EXPECT_NE(0u, escaped & nphi::kEnvStatusMpmGridEscape)
        << "an out-of-AABB MPM particle must flag the escape bit (never a silent drop)";
}

// (5) a non-finite particle position must flag the escape bit: NaN/Inf saturate
// floor->int so the signed bounds test misses it; the nonfinite guard catches it.
TEST(MpmTransferRoundtrip, NonFinitePositionFlagsEnvStatus) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();
    const float nan_x = std::numeric_limits<float>::quiet_NaN();
    for (const Vec3 bad : {Vec3{nan_x, 0.25f, 0.25f}, Vec3{1.0e30f, 0.25f, 0.25f}}) {
        nk::Model m = BuildMpmModel(Vec3{0.0f, 0.0f, 0.0f});
        m.particles.initial_pos.back() = bad;  // poison one in-set particle.
        const uint32_t status = EscapeStatusAfterP2G(b, std::move(m));
        EXPECT_NE(0u, status & nphi::kEnvStatusMpmGridEscape)
            << "a non-finite MPM particle position must flag the escape bit";
    }
}

// (6) the cooked MPM material table stages to the device (the Model-move fix): a
// recognizable material reaches the device table (all-zero without the move).
TEST(MpmTransferRoundtrip, MaterialTableStagesToDevice) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();
    nk::Model m = BuildMpmModel(Vec3{0.0f, 0.0f, 0.0f});  // youngs/poisson/density set.
    const nk::MpmMaterial seed = m.mpm_materials.front();
    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
    ASSERT_TRUE(w.Ready());
    const uint32_t stride = nk::MpmMaterial::kValueCount;
    std::vector<float> table(stride, 0.0f);
    ASSERT_TRUE(w.GetData().DownloadField(nk::FieldId::MpmMaterialTable, table.data(),
                                          table.size() * sizeof(float)));
    EXPECT_FLOAT_EQ(table[0], seed.youngs);
    EXPECT_FLOAT_EQ(table[1], seed.poisson);
    EXPECT_FLOAT_EQ(table[2], seed.density)
        << "the cooked MPM material is all-zero on device if the Model move drops it";
}
