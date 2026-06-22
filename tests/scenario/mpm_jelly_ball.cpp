// ---------------------------------------------------------------------------
// MLS-MPM elastic jelly-ball gate (the headline physics probe).
//
// A fixed-corotated MLS-MPM ball (sampled densely on a lattice, sim_method=mlsmpm
// through the config selector) is dropped onto the STATIC floor plane (the grid BC,
// no dynamic body). Asserts the gate metrics: peak compression <= ~25%, recovery
// (rebound), no collapse, no bottom spikes, >= 5 s sustained, zero NaN/Inf, no
// grid-AABB escape bit, and two-run byte-identity (the deterministic gather).
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "phi/backend.hpp"
#include "phi/op_schema.hpp"
#include "scene/cook/cook_to_model.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace cook = nuka::scene::cook;
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

// Ball + grid geometry (shared with the demo's intent: a dense lattice fill).
constexpr float kBallRadius = 0.10f;
constexpr float kFloorZ     = 0.0f;
constexpr float kDropGap    = 0.05f;   // bottom of the ball this far above the floor.
constexpr float kDx         = 0.025f;  // grid node spacing.

// Sample the ball volume on a lattice at dx/2 spacing (8 particles/cell), build the
// MpmCookInput with a soft elastic material + an env-private grid spanning the drop.
cook::MpmCookInput BuildJellyInput() {
    cook::MpmCookInput in;
    const float center_z = kFloorZ + kDropGap + kBallRadius;
    const float pdx = kDx * 0.5f;                       // particle spacing.
    const int r_cells = static_cast<int>(kBallRadius / pdx) + 1;
    for (int i = -r_cells; i <= r_cells; ++i)
        for (int j = -r_cells; j <= r_cells; ++j)
            for (int k = -r_cells; k <= r_cells; ++k) {
                const Vec3 local{i * pdx, j * pdx, k * pdx};
                if (local.LengthSq() <= kBallRadius * kBallRadius)
                    in.positions.push_back(Vec3{local.x, local.y, center_z + local.z});
            }
    const size_t n = in.positions.size();
    in.velocities.assign(n, Vec3::Zero());
    const float vol0 = pdx * pdx * pdx;
    const float density = 1000.0f;
    const float pmass = density * vol0;
    in.inv_mass.assign(n, 1.0f / pmass);
    in.vol0.assign(n, vol0);
    in.material.youngs = 3.0e4f;     // soft jelly.
    in.material.poisson = 0.3f;
    in.material.density = density;
    in.material.model_kind = 0.0f;   // fixed-corotated elastic.
    // Env-private grid: a box around the drop, a few cells of margin on every side.
    const float span = kBallRadius + kDropGap + 4.0f * kDx;
    in.grid_origin = Vec3{-span, -span, kFloorZ - 4.0f * kDx};
    const float height = (center_z + kBallRadius) - in.grid_origin.z + 4.0f * kDx;
    in.grid_dims[0] = static_cast<uint32_t>(2.0f * span / kDx) + 1u;
    in.grid_dims[1] = in.grid_dims[0];
    in.grid_dims[2] = static_cast<uint32_t>(height / kDx) + 1u;
    in.dx = kDx;
    in.substeps = 20u;               // CFL headroom for the stiff explicit step.
    in.floor_normal = Vec3{0.0f, 0.0f, 1.0f};
    in.floor_d = kFloorZ;
    in.floor_friction = 0.5f;
    return in;
}

// Cook via the config SELECTOR (sim_method == mlsmpm), not a hardcoded mode.
nk::Model BuildJellyModel() {
    nk::Model m;
    m.capacities.env_count = 1u;
    cook::XpbdCookInput soft;          // the bulk-soft descriptor; selector = Mpm.
    soft.solver = nk::Model::ParticleMode::Mpm;
    cook::CookSoftBodyParticles(m, 1u, soft, BuildJellyInput());
    return m;
}

struct Extent { float min_z, max_z, height, span_xy; };
Extent MeasureExtent(const std::vector<Vec3>& pos, uint32_t n) {
    float zlo = 1e30f, zhi = -1e30f, xlo = 1e30f, xhi = -1e30f, ylo = 1e30f, yhi = -1e30f;
    for (uint32_t i = 0; i < n; ++i) {
        zlo = std::min(zlo, pos[i].z); zhi = std::max(zhi, pos[i].z);
        xlo = std::min(xlo, pos[i].x); xhi = std::max(xhi, pos[i].x);
        ylo = std::min(ylo, pos[i].y); yhi = std::max(yhi, pos[i].y);
    }
    return Extent{zlo, zhi, zhi - zlo, 0.5f * ((xhi - xlo) + (yhi - ylo))};
}

bool AnyNonFinite(const std::vector<Vec3>& v, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i)
        if (!std::isfinite(v[i].x) || !std::isfinite(v[i].y) || !std::isfinite(v[i].z))
            return true;
    return false;
}

}  // namespace

// The full gate-(a) arc: drop -> squash -> recover, >= 5 s, all metrics.
TEST(MpmJellyBall, DropSquashRecoverNoCollapseNoSpike) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();
    nk::Model m = BuildJellyModel();
    const uint32_t P = m.capacities.particles_per_env;
    ASSERT_GT(P, 200u) << "the lattice fill must be dense (no spike-prone sparse set)";
    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
    ASSERT_TRUE(w.Ready());

    std::vector<Vec3> pos(P, Vec3::Zero()), vel(P, Vec3::Zero());
    auto dl = [&] {
        w.GetData().DownloadField(nk::FieldId::ParticlePos, pos.data(), P * sizeof(Vec3));
        w.GetData().DownloadField(nk::FieldId::ParticleVel, vel.data(), P * sizeof(Vec3));
    };
    dl();
    const Extent rest = MeasureExtent(pos, P);

    // 5 s at dt = 1/240 == 1200 steps. Track the squash, rebound, floor penetration,
    // finiteness, and the grid-escape status bit over the whole trajectory.
    constexpr uint32_t kSteps = 1200u;
    float min_height = 1e30f;             // deepest squash (peak compression).
    float min_z_ever = 1e30f;             // deepest floor penetration.
    float min_span_xy = 1e30f;            // collapse guard (the ball never implodes).
    float max_height_after_squash = 0.0f; // rebound after the squash.
    int squash_step = -1;
    bool any_nonfinite = false;
    uint32_t escape = 0u;
    for (uint32_t s = 0; s < kSteps; ++s) {
        w.Step();
        dl();
        const Extent e = MeasureExtent(pos, P);
        if (e.height < min_height) { min_height = e.height; squash_step = static_cast<int>(s); }
        min_z_ever = std::min(min_z_ever, e.min_z);
        min_span_xy = std::min(min_span_xy, e.span_xy);
        if (squash_step >= 0 && static_cast<int>(s) > squash_step)
            max_height_after_squash = std::max(max_height_after_squash, e.height);
        any_nonfinite = any_nonfinite || AnyNonFinite(pos, P) || AnyNonFinite(vel, P);
        uint32_t st = 0u;
        w.GetData().DownloadField(nk::FieldId::EnvStatus, &st, sizeof(uint32_t));
        escape |= st & nphi::kEnvStatusMpmGridEscape;
        if (s % 120u == 0u)
            std::fprintf(stderr, "  [jelly] s=%4u h=%.4f min_z=%.4f span_xy=%.4f\n",
                         s, e.height, e.min_z, e.span_xy);
    }
    dl();
    const Extent fin = MeasureExtent(pos, P);

    const float peak_compression = 1.0f - min_height / rest.height;
    std::fprintf(stderr,
                 "[jelly] rest_h=%.4f min_h=%.4f peak_compression=%.1f%% "
                 "rebound_h=%.4f min_z_ever=%.4f min_span_xy=%.4f rest_span=%.4f "
                 "final_h=%.4f nonfinite=%d escape=%u\n",
                 rest.height, min_height, 100.0f * peak_compression,
                 max_height_after_squash, min_z_ever, min_span_xy, rest.span_xy,
                 fin.height, any_nonfinite ? 1 : 0, escape);

    EXPECT_FALSE(any_nonfinite) << "MPM trajectory must be finite (no NaN/Inf)";
    EXPECT_EQ(0u, escape) << "no particle stencil may leave the cooked grid AABB";
    EXPECT_GE(min_z_ever, kFloorZ - kDx)
        << "no bottom spike / tunnel below the floor (min particle z >= floor - dx)";
    EXPECT_LE(peak_compression, 0.30f)
        << "peak compression must stay near the ~25% gate (squash, not pancake)";
    EXPECT_GT(peak_compression, 0.02f) << "the ball must visibly squash on impact";
    EXPECT_GT(max_height_after_squash, min_height + 0.1f * rest.height)
        << "the ball must recover/bounce after the squash (not stay flattened)";
    EXPECT_GT(min_span_xy, 0.5f * rest.span_xy)
        << "the ball must not collapse/implode (horizontal extent stays bounded)";
}

// Two runs produce a byte-identical MPM trajectory (the deterministic gather).
TEST(MpmJellyBall, TwoRunByteIdentical) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();
    auto run = [&](std::vector<Vec3>& out) -> bool {
        nk::Model m = BuildJellyModel();
        const uint32_t P = m.capacities.particles_per_env;
        nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
        if (!w.Ready()) return false;
        for (uint32_t s = 0; s < 120u; ++s) w.Step();
        out.assign(P, Vec3::Zero());
        return w.GetData().DownloadField(nk::FieldId::ParticlePos, out.data(),
                                         P * sizeof(Vec3));
    };
    std::vector<Vec3> a, c;
    ASSERT_TRUE(run(a));
    ASSERT_TRUE(run(c));
    ASSERT_EQ(a.size(), c.size());
    EXPECT_EQ(0, std::memcmp(a.data(), c.data(), a.size() * sizeof(Vec3)))
        << "MPM trajectory differs run-to-run (an atomic scatter would)";
}
