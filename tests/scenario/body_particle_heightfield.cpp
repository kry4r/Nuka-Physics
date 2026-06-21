// ---------------------------------------------------------------------------
// Body/artic <-> particle coupling against a HEIGHTFIELD — the terrain arm gate.
//
// A particle is a SPHERE on the ONE general path, so a particle resting on a
// cooked heightfield grid is the SAME contact problem as a sphere foot on the
// terrain: the body<->particle narrowphase walks the overlapped cells and emits a
// face contact via the deep-sink-robust SphereTriangleFace, identical in spirit to
// the rigid sphere-vs-heightfield handler. Before this gate the heightfield arm was
// a SILENT NO-OP STUB (it built a frame, discarded it, returned an empty manifold),
// so a particle tunnelled straight through the terrain with ZERO contacts AND zero
// diagnostic.
//
// Asserts:
//   (a) truncated-to-solve: a NkRow carries a kNkSideParticle side with the global
//       particle id, and its normal multiplier lambda > 0 (the new handler emits a
//       real push-out contact — the stub emitted 0 rows);
//   (b) full Step: the particle does NOT tunnel — its position rests AT or ABOVE the
//       cooked terrain surface (the stub let it fall forever, far below the grid).
//   (c) two runs of the full step are BYTE-IDENTICAL (D1 for the terrain arm).
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/terrain/heightfield.hpp"
#include "scene/terrain/heightfield_loaders.hpp"
#include "scene/terrain/heightfield_sample.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace cook = nuka::scene::cook;
namespace terrain = nuka::terrain;
using nuka::math::Transform;
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
    cfg.contact_margin = 0.0f;
    cfg.vel_iters = 32u;
    cfg.pos_iters = 0u;
    cfg.max_pairs = 32u;
    return cfg;
}

// A flat heightfield grid + one particle just above its surface, moving DOWN. The
// heightfield is the only collidable; bodies_per_env == 1 exercises the direct-scan
// gather (no LBVH). The particle is a sphere of d_min/2 on the ONE path.
terrain::HeightField MakeFlatHf() {
    terrain::TerrainGenConfig cfg;
    cfg.nrow = 9u; cfg.ncol = 9u;
    cfg.cell_x = 0.25f; cfg.cell_y = 0.25f;
    cfg.origin = Vec3{-0.5f * 8.0f * 0.25f, -0.5f * 8.0f * 0.25f, 0.0f};
    cfg.base_z = 0.0f;
    cfg.ring_rise = 0.0f; cfg.ring_width = 0.0f;
    cfg.ring_platform = 0.0f; cfg.ring_count = 0u;
    terrain::HeightField hf;
    terrain::GenerateHeightField(cfg, hf);
    return hf;
}

nk::Model BuildModel(const terrain::HeightField& hf, float start_z) {
    nk::Model m;
    const uint32_t hf_row = cook::CookHeightfieldGrid(m, hf);
    (void)hf_row;

    nk::ModelCapacities& cap = m.capacities;
    const uint32_t bodies = static_cast<uint32_t>(m.body_init.size());
    cap.env_count = 1u;
    cap.bodies_per_env = bodies;
    cap.max_bodies_total = bodies;
    cap.max_contacts_per_env = 16u;
    cap.max_rows_per_env = 16u * nk::kPairDrivenRowsPerSlot;
    cap.particles_per_env = 1u;
    m.contact_family = nk::ContactFamily::PairDriven;
    m.filter_cross_env = true;

    nk::Model::ModelParticles& mp = m.particles;
    mp.mode = nk::Model::ParticleMode::Xpbd;
    // Particle radius 0.02 (d_min 0.04) over flat surface z=0, moving DOWN at -1 m/s
    // so the terrain contact row must brake it (the stub emitted 0 rows / 0 contacts).
    mp.initial_pos = {Vec3{0.0f, 0.0f, start_z}};
    mp.initial_vel = {Vec3{0.0f, 0.0f, -1.0f}};
    mp.inv_mass = {1.0f / 0.01f};
    mp.xpbd_iters = 1;
    mp.pp_contact_d_min = 0.04f;
    return m;
}

bool StepToSolve(nk::World& w) {
    for (const nphi::OpCall& call : w.GetPipeline().Calls()) {
        if (w.DispatchOp(call.op, call.params) != nphi::Status::Ok) return false;
        if (call.op == nphi::NkOp::SolveRowsBlockIsland) return true;
    }
    return true;
}

struct RowSides { uint32_t a_kind, b_kind, a_index, b_index; bool active; float upper; };
RowSides DecodeRow(const std::vector<float>& urows, uint32_t row) {
    const float* r = urows.data() + static_cast<size_t>(row) * 32u;
    auto u = [&](int i) { uint32_t v; std::memcpy(&v, &r[i], 4); return v; };
    RowSides s;
    s.active = (u(0) & 1u) != 0u;
    s.upper = r[7];
    s.a_kind = u(16); s.a_index = u(17);
    s.b_kind = u(24); s.b_index = u(25);
    return s;
}

}  // namespace

// (a) the terrain arm emits a kNkSideParticle contact row with lambda > 0 (the stub
// emitted ZERO rows -> this would have FAILED before the fix).
TEST(BodyParticleHeightfield, ParticleSideRowAgainstTerrain) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();
    const terrain::HeightField hf = MakeFlatHf();
    nk::Model m = BuildModel(hf, 0.015f);  // bottom -0.005 -> shallow overlap at step 1.
    const uint32_t rows = m.capacities.max_rows_per_env;

    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
    ASSERT_TRUE(w.Ready());
    ASSERT_TRUE(StepToSolve(w));

    std::vector<float> urows(static_cast<size_t>(rows) * 32u, 0.0f);
    std::vector<float> lambda(rows, 0.0f);
    nk::Data& d = w.GetData();
    ASSERT_TRUE(d.DownloadField(nk::FieldId::Urows, urows.data(),
                                urows.size() * sizeof(float)));
    ASSERT_TRUE(d.DownloadField(nk::FieldId::Lambda, lambda.data(),
                                lambda.size() * sizeof(float)));

    uint32_t particle_rows = 0u, active_normal = 0u;
    float max_lambda = 0.0f;
    for (uint32_t row = 0u; row < rows; ++row) {
        const RowSides rs = DecodeRow(urows, row);
        if (!rs.active) continue;
        const bool a_part = rs.a_kind == nk::kNkSideParticle;
        const bool b_part = rs.b_kind == nk::kNkSideParticle;
        if (a_part || b_part) {
            ++particle_rows;
            const uint32_t pidx = a_part ? rs.a_index : rs.b_index;
            EXPECT_EQ(pidx, 0u) << "particle side must carry the GLOBAL particle id";
            if (rs.upper > 1.0e30f) { ++active_normal; max_lambda = std::max(max_lambda, lambda[row]); }
        }
    }
    std::fprintf(stderr,
                 "[bp-hf] particle_rows=%u active_normal=%u max_lambda=%.6f\n",
                 particle_rows, active_normal, max_lambda);
    EXPECT_GT(particle_rows, 0u) << "no kNkSideParticle row vs the heightfield (stub regressed)";
    EXPECT_GT(active_normal, 0u) << "no active particle NORMAL row vs the terrain";
    EXPECT_GT(max_lambda, 0.0f) << "particle terrain normal lambda must be > 0";
}

// (b) full Step: the particle does NOT tunnel through the terrain — it rests AT or
// ABOVE the cooked surface (the stub let it fall forever, far below the grid).
TEST(BodyParticleHeightfield, ParticleDoesNotTunnelTerrain) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();
    const terrain::HeightField hf = MakeFlatHf();
    nk::Model m = BuildModel(hf, 0.30f);  // a real drop from 30 cm onto the terrain.

    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
    ASSERT_TRUE(w.Ready());
    for (uint32_t s = 0; s < 240u; ++s) ASSERT_TRUE(w.Step().AllOk()) << s;

    std::vector<Vec3> ppos(1, Vec3::Zero());
    nk::Data& d = w.GetData();
    ASSERT_TRUE(d.DownloadField(nk::FieldId::ParticlePos, ppos.data(),
                                ppos.size() * sizeof(Vec3)));
    const float radius = 0.02f;
    const float surf = terrain::SampleHeightFieldZ(hf, ppos[0].x, ppos[0].y);
    const float bottom = ppos[0].z - radius;
    std::fprintf(stderr,
                 "[bp-hf] final particle=(%.4f,%.4f,%.4f) surf=%.4f bottom=%.4f\n",
                 ppos[0].x, ppos[0].y, ppos[0].z, surf, bottom);

    ASSERT_TRUE(std::isfinite(ppos[0].z)) << "particle position NaN/inf";
    // The DECISIVE tunnel check: the particle's lowest point stays at/above the
    // terrain surface (small penetration tolerance). The stub let it fall to
    // z << surf (a clean tunnel-through); the fix brakes it at the surface.
    EXPECT_GT(bottom, surf - 0.02f) << "particle tunnelled through the heightfield (穿模)";
    EXPECT_LT(ppos[0].z, 0.5f) << "particle launched away from the terrain";
}

// (c) two runs of the full step are byte-identical (D1 for the terrain arm).
TEST(BodyParticleHeightfield, TwoRunsByteIdentical) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();

    auto run = [&]() -> std::vector<Vec3> {
        const terrain::HeightField hf = MakeFlatHf();
        nk::Model m = BuildModel(hf, 0.30f);
        nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
        if (!w.Ready()) return {};
        for (uint32_t s = 0; s < 200u; ++s)
            if (!w.Step().AllOk()) return {};
        std::vector<Vec3> ppos(1, Vec3::Zero());
        w.GetData().DownloadField(nk::FieldId::ParticlePos, ppos.data(),
                                  ppos.size() * sizeof(Vec3));
        return ppos;
    };

    const std::vector<Vec3> a = run();
    const std::vector<Vec3> c = run();
    ASSERT_EQ(a.size(), 1u);
    ASSERT_EQ(c.size(), 1u);
    EXPECT_EQ(std::memcmp(a.data(), c.data(), sizeof(Vec3)), 0)
        << "particle terrain position differs across runs";
}
