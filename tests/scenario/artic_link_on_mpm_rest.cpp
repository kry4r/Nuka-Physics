// ---------------------------------------------------------------------------
// Articulated-LINK <-> MLS-MPM coupling gate: fixed-base prismatic-Z links rest on
// an MPM bed. A link body is cooked inv_mass==0 (its mass lives in the
// articulation's M), so the grid->body reaction cannot be applied to the body row
// directly; instead the per-substep reaction is gathered (linear + angular) and the
// per-articulation deposit seeds qdot_flat += M^-1 J^T wrench, which the row solve
// reads. This is the SAME general path foot<->ground uses, exercised on the
// simplest articulated balance. Asserts:
//   * support + balance: the links do not sink (min_z stays near the settled rest,
//     prismatic qdot ~0), Sigma reaction.z over a step ~= the links' m*g*dt, and the
//     supported links stay deposit-eligible (the one-way analytic bit stays clear);
//   * BITE: disabling the dynamic-body grid coupling (its BC AND deposit) drops the link;
//   * two runs are byte-identical (LinkPose + Qdot) -- the deposit is deterministic;
//   * a 2-links-on-one-articulation case lands a deposit on BOTH prismatic DOFs;
//   * a 2-env world keeps each env's deposit independent (byte-identical envs).
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "import/cooker/xpbd_cooker_types.hpp"
#include "import/usd_importer.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"
#include "phi/backend.hpp"
#include "phi/op_schema.hpp"  // kEnvStatusMpmOneWayBody
#include "runtime/sdf/sparse_sdf_query.cuh"
#include "scene/cook/cook_to_model.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace cook = nuka::scene::cook;
using nuka::math::Transform;
using nuka::math::Vec3;

// Scene geometry. The MPM bed (a dense slab) sits on the static floor; the
// prismatic link's collidable foot sphere rests a hair into the bed top.
constexpr float kFloorZ    = 0.0f;
constexpr float kDx        = 0.02f;
constexpr float kBedHalfXY = 0.15f;  // covers both feet (x=0 and x=0.10, r=0.04).
constexpr float kBedTopZ   = 0.20f;
constexpr float kFootR     = 0.04f;
constexpr float kLinkMass  = 0.5f;   // matches prismatic_z_link.usda physics:mass.

std::filesystem::path ScenePath() {
    return std::filesystem::path(NUKA_SOURCE_DIR) / "examples" / "scenes" /
           "prismatic_z_link.usda";
}

namespace sdfq = nuka::runtime::sdf;

// Bind a narrow-band sphere SDF to a body row so the MPM grid BC can rasterize it
// (the BC samples a body's cooked SDF; an analytic sphere alone carries none). The
// general MPM body BC + reaction path is identical for a free-rigid or a link row;
// only an SDF-cooked collidable participates, so the link's foot needs one.
void AddSphereSdfToBodyRow(nk::Model& m, uint32_t body_row, float radius) {
    const float vh = kDx;
    const float band = 3.0f * vh;
    const float ext = radius + band;
    const int n = static_cast<int>(std::ceil(ext / vh)) + 1;
    const Vec3 origin{-static_cast<float>(n) * vh, -static_cast<float>(n) * vh,
                      -static_cast<float>(n) * vh};
    const uint32_t base = static_cast<uint32_t>(m.sdf_cell_values.size());
    uint32_t count = 0u;
    for (int i = 0; i <= 2 * n; ++i)
        for (int j = 0; j <= 2 * n; ++j)
            for (int k = 0; k <= 2 * n; ++k) {
                const Vec3 p{origin.x + i * vh, origin.y + j * vh, origin.z + k * vh};
                const float len = std::sqrt(p.LengthSq());
                const float phi = len - radius;
                if (std::fabs(phi) > band) continue;
                const Vec3 grad = len > 1e-8f ? p * (1.0f / len) : Vec3{0, 0, 1};
                m.sdf_cell_keys.push_back(sdfq::PackSdfCellKey(
                    static_cast<uint32_t>(i), static_cast<uint32_t>(j),
                    static_cast<uint32_t>(k)));
                m.sdf_cell_values.push_back(phi);
                m.sdf_cell_gradients.push_back(grad);
                ++count;
            }
    nk::Model::SdfGrid sg;
    sg.origin = origin;
    sg.voxel_size = vh;
    sg.dims[0] = sg.dims[1] = sg.dims[2] = static_cast<uint32_t>(2 * n + 1);
    sg.cell_offset = base;
    sg.cell_count = count;
    const uint32_t grid_idx = static_cast<uint32_t>(m.sdf_grids.size());
    m.sdf_grids.push_back(sg);
    m.shape_table_rows[body_row].sdf_grid = grid_idx;
}

struct Backend { nphi::Device* dev = nullptr; nphi::Backend* backend = nullptr; };
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
    cfg.max_pairs = 64u;
    return cfg;
}

// A moderate MPM bed (dx*0.85 lattice, full depth on the static floor) under the
// links' feet. The lattice + Young's + substeps are tuned so the foot settles to a
// clean static rest (Sigma reaction.z balances m*g*dt) yet steps a gate test fast.
cook::MpmCookInput BuildBedInput() {
    cook::MpmCookInput in;
    const float pdx = kDx * 0.85f;
    const float bed_lo_z = kFloorZ + pdx;
    for (float x = -kBedHalfXY; x <= kBedHalfXY + 1e-4f; x += pdx)
        for (float y = -kBedHalfXY; y <= kBedHalfXY + 1e-4f; y += pdx)
            for (float z = bed_lo_z; z <= kBedTopZ + 1e-4f; z += pdx)
                in.positions.push_back(Vec3{x, y, z});
    const size_t n = in.positions.size();
    in.velocities.assign(n, Vec3::Zero());
    const float vol0 = pdx * pdx * pdx;
    const float density = 1000.0f;
    in.inv_mass.assign(n, 1.0f / (density * vol0));
    in.vol0.assign(n, vol0);
    in.material.youngs = 3.0e5f;       // holds the foot up without a stiff bounce.
    in.material.poisson = 0.3f;
    in.material.density = density;
    in.material.model_kind = 0.0f;     // fixed-corotated elastic.
    const float span = kBedHalfXY + 4.0f * kDx;
    in.grid_origin = Vec3{-span, -span, kFloorZ - 4.0f * kDx};
    const float top = 0.26f + kFootR + 6.0f * kDx;   // above the slider rest height.
    in.grid_dims[0] = static_cast<uint32_t>(2.0f * span / kDx) + 1u;
    in.grid_dims[1] = in.grid_dims[0];
    in.grid_dims[2] = static_cast<uint32_t>((top - in.grid_origin.z) / kDx) + 1u;
    in.dx = kDx;
    in.substeps = 10u;
    in.floor_normal = Vec3{0.0f, 0.0f, 1.0f};
    in.floor_d = kFloorZ;
    in.floor_friction = 0.5f;
    return in;
}

// Cook the prismatic articulation (general PairDriven family) + the MPM bed below,
// replicated across env_count homogeneous environments.
nk::Model BuildModel(bool bite, uint32_t env_count = 1u) {
    nuka::scene::SceneIR scene = nuka::import::LoadUsd(ScenePath().string());
    cook::CookToModelOptions opt;
    opt.contact_family = cook::CookContactFamily::PairDriven;
    nk::Model m = cook::CookToModel(scene, static_cast<int>(env_count), opt).model;

    nk::ModelCapacities& cap = m.capacities;
    cap.max_contacts_per_env = 16u;
    cap.max_rows_per_env = cap.max_contacts_per_env * nk::kPairDrivenRowsPerSlot;

    // Give each collidable link's foot a cooked SDF so the MPM grid BC rasterizes
    // it (the same SDF a free-rigid box carries in rigid_on_mpm_rest).
    const uint32_t L = cap.links_per_env;
    const std::vector<uint32_t>& kind = m.articulation.link_geom_kind;
    const std::vector<float>& params = m.articulation.link_geom_params;
    for (uint32_t b = 0; b < m.body_to_link.size(); ++b) {
        const uint32_t link = m.body_to_link[b];
        if (link == ~0u || link >= L) continue;
        if (link < kind.size() && kind[link] != 0u) {
            const float r = params[static_cast<size_t>(link) * 4u];
            AddSphereSdfToBodyRow(m, b, r);
        } else if (b < m.shape_table_rows.size()) {
            // A geomless link row (the kinematic base) is not a collidable: clear
            // its contype so the MPM body BC does not flag it as a one-way analytic.
            m.shape_table_rows[b].contype = 0u;
        }
    }
    cap.max_sdf_grids = static_cast<uint32_t>(m.sdf_grids.size());
    cap.max_sdf_cells = static_cast<uint32_t>(m.sdf_cell_values.size());

    // Joint damping dissipates the stiff grid-contact bounce so the link settles to
    // a steady rest (a physical property, not a fudge: the deposit still balances g).
    if (m.articulation.joint_damping.size() < L)
        m.articulation.joint_damping.assign(L, 0.0f);
    for (uint32_t l = 0; l < L; ++l)
        if (l < kind.size() && kind[l] != 0u) m.articulation.joint_damping[l] = 8.0f;

    cook::XpbdCookInput soft;
    soft.solver = nk::Model::ParticleMode::Mpm;
    cook::CookSoftBodyParticles(m, env_count, soft, BuildBedInput());

    m.particles.mpm_body_friction = 0.5f;
    m.particles.mpm_bite_disable_dynamic_bc = bite;
    return m;
}

// The link whose collidable foot sphere descends to the bed; its prismatic DOF.
struct LinkInfo { uint32_t link = ~0u; uint32_t dof = ~0u; float radius = 0.0f; };

std::vector<LinkInfo> CollidableLinks(const nk::Model& m) {
    std::vector<LinkInfo> out;
    const uint32_t L = m.capacities.links_per_env;
    const std::vector<uint32_t>& kind = m.articulation.link_geom_kind;
    const std::vector<float>& params = m.articulation.link_geom_params;
    for (uint32_t l = 0; l < L; ++l) {
        if (l < kind.size() && kind[l] != 0u) {
            LinkInfo li;
            li.link = l;
            li.radius = params[static_cast<size_t>(l) * 4u];
            out.push_back(li);
        }
    }
    return out;
}

}  // namespace

// Support + balance: the link rests on the bed (no sink, qdot ~ 0 at rest) and the
// per-step grid reaction balances gravity through the prismatic deposit.
TEST(ArticLinkOnMpmRest, RestReactionBalancesGravityThroughDeposit) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    if (!std::filesystem::exists(ScenePath())) GTEST_SKIP() << "scene not present";
    Backend b = GetBackend();
    nk::Model m = BuildModel(/*bite=*/false);
    ASSERT_GT(m.capacities.particles_per_env, 200u) << "the bed must be dense";
    const uint32_t L = m.capacities.links_per_env;
    const std::vector<LinkInfo> links = CollidableLinks(m);
    ASSERT_GE(links.size(), 1u) << "no collidable prismatic link cooked";
    const uint32_t foot_link = links.front().link;

    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
    ASSERT_TRUE(w.Ready());

    const float dt = Cfg().dt;
    const float weight_impulse = kLinkMass * 9.81f * dt;  // one link's m*g*dt.

    auto foot_z = [&]() {
        std::vector<Transform> lp(L);
        w.GetData().DownloadField(nk::FieldId::LinkPose, lp.data(), L * sizeof(Transform));
        return lp[foot_link].position.z;
    };
    const float z0 = foot_z();
    constexpr uint32_t kSettle = 70u;
    {
        std::vector<float> qd(L, 0.0f);
        for (uint32_t s = 0; s < kSettle; ++s) {
            w.Step();
            if (s % 20u == 0u || s + 5u > kSettle) {
                w.GetData().DownloadField(nk::FieldId::Qdot, qd.data(), L * sizeof(float));
                std::fprintf(stderr, "[DBG s=%u] footz=%.4f qd_foot=%.4f\n", s,
                             foot_z(), qd.size() > foot_link ? qd[foot_link] : 0.f);
            }
        }
    }
    const float z_settled = foot_z();

    constexpr uint32_t kWindow = 40u;
    double sum_rz = 0.0;
    float min_z = 1e30f;
    bool nonfinite = false;
    std::vector<Vec3> reaction(/*bodies*/ 0);
    std::vector<float> qdot(L, 0.0f);
    float max_qdot = 0.0f;
    // The collidable link's body row holds the linear reaction; download all body
    // rows and pick the maximum upward reaction (the single foot row carries it).
    const uint32_t B = w.GetModel().capacities.bodies_per_env;
    reaction.assign(B, Vec3::Zero());
    for (uint32_t s = 0; s < kWindow; ++s) {
        w.Step();
        w.GetData().DownloadField(nk::FieldId::MpmBodyReaction, reaction.data(),
                                  B * sizeof(Vec3));
        float rz = 0.0f;
        for (const Vec3& r : reaction) rz += r.z;  // total upward grid reaction.
        w.GetData().DownloadField(nk::FieldId::Qdot, qdot.data(), L * sizeof(float));
        const float z = foot_z();
        min_z = std::min(min_z, z);
        nonfinite = nonfinite || !std::isfinite(z) || !std::isfinite(rz);
        max_qdot = std::max(max_qdot, std::fabs(qdot[foot_link]));
        sum_rz += rz;
    }
    const float avg_rz = static_cast<float>(sum_rz / kWindow);
    // Two links rest on the bed: the total reaction balances both weights.
    const float total_weight = static_cast<float>(links.size()) * weight_impulse;
    const float rel_err = std::fabs(avg_rz - total_weight) / total_weight;

    std::fprintf(stderr,
                 "[artic-rest] links=%zu z0=%.4f settled=%.4f min_z=%.4f "
                 "weight=%.6e avg_reaction.z=%.6e rel_err=%.3f max|qdot|=%.5f\n",
                 links.size(), z0, z_settled, min_z, total_weight, avg_rz, rel_err,
                 max_qdot);

    EXPECT_FALSE(nonfinite) << "the rest trajectory must be finite";
    EXPECT_GT(min_z, 0.9f * z_settled) << "the link sank away from its rest height";
    EXPECT_LT(max_qdot, 0.5f) << "the prismatic DOF is not at rest (large qdot)";
    EXPECT_GT(avg_rz, 0.0f) << "the reaction must push the link UP";
    EXPECT_LT(rel_err, 0.10f)
        << "Sigma reaction.z must balance the links' m*g*dt through the deposit";

    // The supported links are deposit-eligible, not one-way analytic bodies: a
    // failed SDF bind would mark a foot analytic-only (no reaction) and raise this.
    uint32_t env_status = 0u;
    w.GetData().DownloadField(nk::FieldId::EnvStatus, &env_status, sizeof(uint32_t));
    EXPECT_EQ(env_status & nphi::kEnvStatusMpmOneWayBody, 0u)
        << "a supported link was flagged a one-way analytic body (deposit bypassed)";
}

// BITE: disabling the dynamic-body grid coupling (its per-substep BC AND its per-Step
// M^-1 J^T deposit, both off one flag) drops the link; the static floor BC still
// holds the bed, so the link free-falls onto a bed that itself stays put.
TEST(ArticLinkOnMpmRest, FreeFallBiteWhenBcDisabled) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    if (!std::filesystem::exists(ScenePath())) GTEST_SKIP() << "scene not present";
    Backend b = GetBackend();
    nk::Model m = BuildModel(/*bite=*/true);
    const uint32_t L = m.capacities.links_per_env;
    const uint32_t B = m.capacities.bodies_per_env;
    const uint32_t foot_link = CollidableLinks(m).front().link;
    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
    ASSERT_TRUE(w.Ready());

    const float dt = Cfg().dt;
    auto foot_z = [&]() {
        std::vector<Transform> lp(L);
        w.GetData().DownloadField(nk::FieldId::LinkPose, lp.data(), L * sizeof(Transform));
        return lp[foot_link].position.z;
    };
    (void)dt;
    // First step runs the forward kinematics so the prismatic link reaches its
    // world rest height (the pre-FK seed pose carries the link-local z, not world).
    w.Step();
    const float z0 = foot_z();
    constexpr uint32_t kSteps = 50u;
    float max_react = 0.0f;
    std::vector<Vec3> reaction(B, Vec3::Zero());
    for (uint32_t s = 0; s < kSteps; ++s) {
        w.Step();
        w.GetData().DownloadField(nk::FieldId::MpmBodyReaction, reaction.data(),
                                  B * sizeof(Vec3));
        for (const Vec3& r : reaction) max_react = std::max(max_react, std::fabs(r.z));
    }
    const float z1 = foot_z();
    const float drop = z0 - z1;
    // max_react is diagnostic only: under BITE the reaction probe is cleared each step
    // and never refilled (its writer is bite-gated), so it reads 0 by bookkeeping.
    std::fprintf(stderr, "[artic-bite] z0=%.4f z1=%.4f drop=%.4f max_react=%.6e\n",
                 z0, z1, drop, max_react);
    // With the coupling disabled the link is not held up: it descends well below its
    // rest (joint damping resists, so this is a damped fall, not undamped 1/2 g t^2).
    EXPECT_GT(drop, 0.05f) << "the link must fall under the BITE (no deposit holds it)";
}

// Two runs of the rest scene produce a byte-identical link trajectory (LinkPose +
// Qdot) -- the per-articulation deposit is deterministic (fixed body-row order, no
// float atomics on the qdot_flat accumulation).
TEST(ArticLinkOnMpmRest, RestTwoRunByteIdentical) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    if (!std::filesystem::exists(ScenePath())) GTEST_SKIP() << "scene not present";
    Backend b = GetBackend();
    auto run = [&](std::vector<Transform>& lp_out, std::vector<float>& qdot_out) -> bool {
        nk::Model m = BuildModel(/*bite=*/false);
        const uint32_t L = m.capacities.links_per_env;
        nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
        if (!w.Ready()) return false;
        for (uint32_t s = 0; s < 55u; ++s) w.Step();
        lp_out.assign(L, Transform::Identity());
        qdot_out.assign(L, 0.0f);
        return w.GetData().DownloadField(nk::FieldId::LinkPose, lp_out.data(),
                                         L * sizeof(Transform)) &&
               w.GetData().DownloadField(nk::FieldId::Qdot, qdot_out.data(),
                                         L * sizeof(float));
    };
    std::vector<Transform> la, lc;
    std::vector<float> qa, qc;
    ASSERT_TRUE(run(la, qa));
    ASSERT_TRUE(run(lc, qc));
    ASSERT_EQ(la.size(), lc.size());
    ASSERT_EQ(qa.size(), qc.size());
    EXPECT_EQ(0, std::memcmp(la.data(), lc.data(), la.size() * sizeof(Transform)))
        << "link poses differ run-to-run (a non-deterministic deposit would)";
    EXPECT_EQ(0, std::memcmp(qa.data(), qc.data(), qa.size() * sizeof(float)))
        << "joint velocities differ run-to-run";
}

// The per-articulation reduction: both sibling prismatic links rest on the bed, so
// BOTH prismatic DOFs receive a deposit (the deposit sums over the articulation's
// link bodies and writes the shared qdot_flat tile once). Each foot is held up.
TEST(ArticLinkOnMpmRest, TwoLinksOnOneArticulationBothSupported) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    if (!std::filesystem::exists(ScenePath())) GTEST_SKIP() << "scene not present";
    Backend b = GetBackend();
    nk::Model m = BuildModel(/*bite=*/false);
    const uint32_t L = m.capacities.links_per_env;
    const std::vector<LinkInfo> links = CollidableLinks(m);
    ASSERT_EQ(links.size(), 2u) << "the scene must cook two collidable links";
    // The two sibling links must belong to the SAME articulation (one tree).
    ASSERT_EQ(m.capacities.articulations_per_env, 1u)
        << "both prismatic links must live on ONE articulation (shared qdot tile)";

    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
    ASSERT_TRUE(w.Ready());

    auto foot_z = [&](uint32_t link) {
        std::vector<Transform> lp(L);
        w.GetData().DownloadField(nk::FieldId::LinkPose, lp.data(), L * sizeof(Transform));
        return lp[link].position.z;
    };
    const float z0_a = foot_z(links[0].link);
    const float z0_b = foot_z(links[1].link);
    constexpr uint32_t kSettle = 90u;
    float min_a = 1e30f, min_b = 1e30f;
    for (uint32_t s = 0; s < kSettle; ++s) {
        w.Step();
        min_a = std::min(min_a, foot_z(links[0].link));
        min_b = std::min(min_b, foot_z(links[1].link));
    }
    const float settled_a = foot_z(links[0].link);
    const float settled_b = foot_z(links[1].link);
    std::fprintf(stderr,
                 "[artic-2link] z0=(%.4f,%.4f) settled=(%.4f,%.4f) min=(%.4f,%.4f)\n",
                 z0_a, z0_b, settled_a, settled_b, min_a, min_b);
    // Each link descended onto the bed and was held there (did not pass through it).
    EXPECT_GT(settled_a, kBedTopZ - kFootR - 0.03f)
        << "link A sank through the bed (no deposit on its DOF)";
    EXPECT_GT(settled_b, kBedTopZ - kFootR - 0.03f)
        << "link B sank through the bed (no deposit on its DOF)";
    EXPECT_TRUE(std::isfinite(settled_a) && std::isfinite(settled_b));
}

// A 2-env world keeps each env's deposit independent: the homogeneous replicas must
// produce byte-identical link trajectories. A per-env stride bug in the deposit
// (env*base_link_count, row0, or the qdot tile) would desync env 1 from env 0.
TEST(ArticLinkOnMpmRest, MultiEnvDepositPerEnvStridesByteIdentical) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    if (!std::filesystem::exists(ScenePath())) GTEST_SKIP() << "scene not present";
    Backend b = GetBackend();
    constexpr uint32_t kEnvs = 2u;
    nk::Model m = BuildModel(/*bite=*/false, kEnvs);
    const uint32_t L = m.capacities.links_per_env;
    const uint32_t foot_link = CollidableLinks(m).front().link;
    nk::World w(std::move(m), kEnvs, b.dev, b.backend, Cfg());
    ASSERT_TRUE(w.Ready());
    for (uint32_t s = 0; s < 70u; ++s) w.Step();

    std::vector<Transform> lp(static_cast<size_t>(L) * kEnvs);
    ASSERT_TRUE(w.GetData().DownloadField(nk::FieldId::LinkPose, lp.data(),
                                          lp.size() * sizeof(Transform)));
    const float z0 = lp[foot_link].position.z;
    const float z1 = lp[L + foot_link].position.z;
    std::fprintf(stderr, "[artic-2env] foot_z env0=%.4f env1=%.4f\n", z0, z1);
    // Each env's foot is held on the bed: the deposit landed in that env's qdot tile.
    EXPECT_GT(z0, kBedTopZ - kFootR - 0.03f) << "env 0 foot sank (no deposit)";
    EXPECT_GT(z1, kBedTopZ - kFootR - 0.03f) << "env 1 foot sank (deposit mis-strided)";
    // The two homogeneous envs must be byte-identical (the deposit is per-env disjoint).
    EXPECT_EQ(0, std::memcmp(lp.data(), lp.data() + L, L * sizeof(Transform)))
        << "env 1 link poses differ from env 0 -- a per-env deposit stride bug";
}
