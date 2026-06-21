// ---------------------------------------------------------------------------
// Articulated-body LINK <-> particle TWO-WAY coupling on the ONE general row
// solver. A minimal single-revolute pendulum (a pinned base + one swinging arm
// whose collidable tip sphere hangs on an X lever) is cooked through the SAME
// production path a robot uses (LoadUsd -> CookToModel -> nk::World). The arm's
// LINK is a collidable body row posed by SyncLinkBodyPose; a cloth patch is
// placed under the resting tip. The link presses the particles through the SAME
// detection -> body<->particle narrowphase -> emit -> SolveRowsBlockIsland the
// foot uses on the ground: no per-robot code, an articulation link is just
// another collidable body row and a particle is a sphere.
//
// The crux, proven numerically against a no-particle control:
//   (a) LINK-SIDE ROW: a body<->particle contact row exists whose particle side
//       kind == kNkSideParticle AND whose BODY side maps to an ARTICULATION LINK
//       (body_to_link[body] != ~0) -- not a rigid free body.
//   (b) TWO-WAY: the particles are displaced/braked by the link AND the reaction
//       reaches the ARTICULATION -- the single revolute qdot and the arm tip's
//       link pose differ from a no-particle control. The impulse flows through
//       the articulated dynamics exactly as foot<->ground does.
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
#include "runtime/soft/cloth_topology.hpp"
#include "scene/cook/cook_to_model.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace cook = nuka::scene::cook;
namespace soft = nuka::runtime::soft;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr float kContactDMin = 0.030f;       // particle sphere radius = d_min/2.
constexpr float kParticleMass = 0.05f;       // 50 g per particle (resists the arm).
constexpr uint32_t kGridN = 11u;             // 11x11 = 121 particles (a patch).
constexpr float kSpacing = 0.02f;            // ~0.20 m square patch.
constexpr uint16_t kXpbdIters = 20u;
constexpr uint32_t kSettleSteps = 400u;
// The arm starts horizontal (q==0, tip out at +X); a positive revolute target
// rotates it DOWN about +Y so the descending tip presses a patch placed under it.
constexpr float kDriveTarget = 1.1f;

std::filesystem::path PendulumScenePath() {
    return std::filesystem::path(NUKA_SOURCE_DIR) / "examples" / "scenes" /
           "pendulum_tip.usda";
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

// A flat cloth patch centred at (cx,cy) lying at height z. Two far corners pinned
// so the patch does not slide out from under the tip while it is pressed.
cook::XpbdCookInput BuildCloth(float cx, float cy, float z) {
    std::vector<Vec3> rest;
    rest.reserve(kGridN * kGridN);
    const float c0 = -0.5f * static_cast<float>(kGridN - 1u) * kSpacing;
    for (uint32_t j = 0; j < kGridN; ++j) {
        for (uint32_t i = 0; i < kGridN; ++i) {
            rest.push_back(Vec3{cx + c0 + static_cast<float>(i) * kSpacing,
                                cy + c0 + static_cast<float>(j) * kSpacing, z});
        }
    }
    auto idx = [](uint32_t i, uint32_t j) { return j * kGridN + i; };
    std::vector<soft::ClothTriangle> tris;
    for (uint32_t j = 0; j + 1 < kGridN; ++j) {
        for (uint32_t i = 0; i + 1 < kGridN; ++i) {
            tris.push_back(soft::ClothTriangle{{idx(i, j), idx(i + 1, j),
                                                idx(i + 1, j + 1)}});
            tris.push_back(soft::ClothTriangle{{idx(i, j), idx(i + 1, j + 1),
                                                idx(i, j + 1)}});
        }
    }
    soft::ClothTopologyOptions opts;
    opts.distance_compliance_alpha = 0.0f;
    opts.bend_compliance_alpha = 1.0e-4f;
    soft::XpbdConstraintSet cs;
    soft::BuildClothConstraints(rest, tris, opts, cs);

    cook::XpbdCookInput in;
    in.positions = rest;
    in.velocities.assign(rest.size(), Vec3::Zero());
    in.inv_mass.assign(rest.size(), 1.0f / kParticleMass);
    // Pin the four corners so the patch stays put under the lateral lever motion.
    const uint32_t last = kGridN - 1u;
    for (uint32_t c : {idx(0, 0), idx(last, 0), idx(0, last), idx(last, last)}) {
        in.inv_mass[c] = 0.0f;
    }
    for (const auto& dc : cs.distance) {
        cook::CookDistanceCon c;
        c.a = dc.particle_a; c.b = dc.particle_b;
        c.rest_length = dc.rest_length; c.compliance_alpha = dc.compliance_alpha;
        in.distance.push_back(c);
    }
    for (const auto& bc : cs.bend) {
        cook::CookBendCon c;
        for (uint32_t k = 0; k < 4u; ++k) { c.p[k] = bc.particle[k]; c.k[k] = bc.k[k]; }
        c.compliance_alpha = bc.compliance_alpha;
        in.bend.push_back(c);
    }
    in.solver_iterations = kXpbdIters;
    in.friction = 0.6f;
    return in;
}

// Cook the pendulum (general PairDriven family) optionally with a cloth patch
// under the tip. patch_present == false cooks the bare articulation control.
nk::Model BuildScene(bool patch_present, float cx, float cy, float z) {
    nuka::scene::SceneIR scene = nuka::import::LoadUsd(PendulumScenePath().string());
    cook::CookToModelOptions opt;
    opt.contact_family = cook::CookContactFamily::PairDriven;
    nk::Model m = cook::CookToModel(scene, 1, opt).model;

    nk::ModelCapacities& cap = m.capacities;
    // Generous rigid candidate budget for the single articulation's collidables.
    cap.max_contacts_per_env = 16u;
    cap.max_rows_per_env = cap.max_contacts_per_env * nk::kPairDrivenRowsPerSlot;
    if (patch_present) {
        const cook::XpbdCookInput cloth = BuildCloth(cx, cy, z);
        cook::CookXpbdParticles(m, 1u, cloth);   // appends cloth + grows the budget.
        m.particles.pp_contact_d_min = kContactDMin;
    }
    return m;
}

// NkRow packs to 32 f32: [0]=flags [7]=upper, a.kind [16] a.index [17],
// b.kind [24] b.index [25].
struct RowSides {
    uint32_t a_kind, b_kind, a_index, b_index;
    bool active; float upper;
};
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

void DownloadParticles(nk::World& w, std::vector<Vec3>* pos) {
    const uint32_t P = w.GetModel().capacities.particles_per_env;
    pos->assign(P, Vec3::Zero());
    w.GetData().DownloadField(nk::FieldId::ParticlePos, pos->data(), P * sizeof(Vec3));
}

float MinZ(const std::vector<Vec3>& pos) {
    float lo = 1.0e9f;
    for (const Vec3& p : pos) lo = std::min(lo, p.z);
    return lo;
}

}  // namespace

// The arm LINK presses a cloth patch through the general body<->particle path:
// a coupling row whose BODY side is an articulation link, and the reaction reaches
// the single revolute joint (qdot + tip pose differ from a no-particle control).
TEST(ArticulatedLinkParticleCoupling, ArmLinkPressesParticlesAndJointRecoils) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    if (!std::filesystem::exists(PendulumScenePath())) {
        GTEST_SKIP() << "pendulum_tip.usda not present";
    }
    Backend b = GetBackend();

    // --- locate the resting tip with a bare-articulation pre-run ----------------
    // Cook the control once, settle it, and read where the collidable tip link
    // comes to rest so the cloth patch can be placed directly under it.
    nk::Model probe = BuildScene(/*patch_present=*/false, 0, 0, 0);
    const uint32_t L = probe.capacities.links_per_env;
    const std::vector<uint32_t> body_to_link = probe.body_to_link;
    const std::vector<uint32_t> link_geom_kind = probe.articulation.link_geom_kind;
    const std::vector<float> link_geom_params = probe.articulation.link_geom_params;
    const std::vector<Transform> link_geom_local = probe.articulation.link_geom_local;
    ASSERT_GT(L, 0u);

    // Find the single collidable link (the arm) and its tip sphere radius.
    uint32_t tip_link = ~0u; float tip_radius = 0.0f;
    for (uint32_t l = 0; l < L; ++l) {
        if (l < link_geom_kind.size() && link_geom_kind[l] != 0u) {
            tip_link = l; tip_radius = link_geom_params[static_cast<size_t>(l) * 4u];
            break;
        }
    }
    ASSERT_NE(tip_link, ~0u) << "no collidable link cooked for the pendulum arm";

    nk::World wp(std::move(probe), 1u, b.dev, b.backend, Cfg());
    ASSERT_TRUE(wp.Ready());
    for (uint32_t s = 0; s < kSettleSteps; ++s) wp.Step();
    std::vector<Transform> link_pose(L);
    ASSERT_TRUE(wp.GetData().DownloadField(nk::FieldId::LinkPose, link_pose.data(),
                                           L * sizeof(Transform)));
    const Transform tip_world = link_pose[tip_link] * link_geom_local[tip_link];
    const Vec3 tip_rest = tip_world.position;  // tip at the horizontal rest (q==0).
    std::fprintf(stderr,
                 "[artic-particle] L=%u tip_link=%u tip_radius=%.4f tip rest(q=0) @ "
                 "(%.4f,%.4f,%.4f)\n",
                 L, tip_link, tip_radius, tip_rest.x, tip_rest.y, tip_rest.z);

    // The driven tip swings DOWN through an arc (x shrinks, z drops). Place a
    // horizontal patch the descending tip must cross: under the rest height and
    // pulled inward toward the mid-arc, sized to span the arc's horizontal reach.
    const Vec3 patch_xy{tip_rest.x - 0.06f, tip_rest.y, 0.0f};
    const float patch_z = tip_rest.z - 0.12f;

    // --- runs a scene driving the arm DOWN onto the patch; collects the deepest
    // link<->particle row (body side maps to a link), the body-side reaction, the
    // final joint coordinate / velocity, and the tip pose. --------------------
    struct Run {
        bool link_particle_row = false;   // a row with a kNkSideParticle side whose
                                          // body side maps to an articulation link.
        uint32_t particle_rows = 0u;
        float max_body_lambda = 0.0f;
        float q_joint = 0.0f;             // the revolute coordinate (the 1 DOF).
        std::vector<float> qdot;
        Vec3 tip{};
        float patch_min_z = 1.0e9f;
    };
    auto run_scene = [&](bool patch_present) -> Run {
        Run out;
        nk::Model m = BuildScene(patch_present, patch_xy.x, patch_xy.y, patch_z);
        const uint32_t rows = m.capacities.max_rows_per_env;
        nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
        EXPECT_TRUE(w.Ready());
        nk::Data& d = w.GetData();
        // Command the arm DOWN onto the patch (the production PD-hold drive).
        std::vector<float> tgt(L, 0.0f);
        tgt[tip_link] = kDriveTarget;
        std::vector<float> urows(static_cast<size_t>(rows) * 32u, 0.0f);
        std::vector<float> lambda(rows, 0.0f);
        for (uint32_t s = 0; s < kSettleSteps; ++s) {
            d.UploadField(nk::FieldId::DriveTarget, tgt.data(), L * sizeof(float));
            w.Step();
            if (!patch_present) continue;
            d.DownloadField(nk::FieldId::Urows, urows.data(), urows.size() * sizeof(float));
            d.DownloadField(nk::FieldId::Lambda, lambda.data(), lambda.size() * sizeof(float));
            for (uint32_t row = 0u; row < rows; ++row) {
                const RowSides rs = DecodeRow(urows, row);
                if (!rs.active) continue;
                const bool a_part = rs.a_kind == nk::kNkSideParticle;
                const bool b_part = rs.b_kind == nk::kNkSideParticle;
                if (!(a_part || b_part)) continue;
                EXPECT_FALSE(a_part && b_part) << "row has two particle sides";
                ++out.particle_rows;
                // The body side is the non-particle side; its index is an env-local
                // body row. body_to_link maps a link-owned body row to its link.
                const uint32_t body_idx = a_part ? rs.b_index : rs.a_index;
                if (body_idx < body_to_link.size() &&
                    body_to_link[body_idx] != ~uint32_t(0)) {
                    out.link_particle_row = true;
                }
                if (rs.upper > 1.0e30f)  // a normal row (friction spokes cap upper).
                    out.max_body_lambda = std::max(out.max_body_lambda, lambda[row]);
            }
        }
        std::vector<float> q(L, 0.0f);
        out.qdot.assign(L, 0.0f);
        d.DownloadField(nk::FieldId::Q, q.data(), L * sizeof(float));
        d.DownloadField(nk::FieldId::Qdot, out.qdot.data(), L * sizeof(float));
        out.q_joint = q[tip_link];
        std::vector<Transform> lp(L);
        d.DownloadField(nk::FieldId::LinkPose, lp.data(), L * sizeof(Transform));
        out.tip = (lp[tip_link] * link_geom_local[tip_link]).position;
        if (patch_present) {
            std::vector<Vec3> p; DownloadParticles(w, &p);
            out.patch_min_z = MinZ(p);
        }
        return out;
    };

    const Run with_patch = run_scene(/*patch_present=*/true);
    const Run control = run_scene(/*patch_present=*/false);

    // The patch holds the arm SHORT of the target the free control reaches: the
    // contact reaction reduces the driven joint coordinate (the reaction reaches
    // the articulation). qdot also differs from the control.
    const float q_lag = control.q_joint - with_patch.q_joint;
    double qdot_delta = 0.0;
    ASSERT_EQ(with_patch.qdot.size(), control.qdot.size());
    for (size_t i = 0; i < with_patch.qdot.size(); ++i)
        qdot_delta += std::fabs(static_cast<double>(with_patch.qdot[i]) - control.qdot[i]);
    // The patch's rest lay (no contact) vs its pressed min z (the link pushed it).
    const float patch_lay_z = patch_z;

    std::fprintf(stderr,
                 "[artic-particle] particle_rows=%u link_particle_row=%d "
                 "body_lambda=%.6f patch_lay_z=%.4f patch_min_z=%.4f\n",
                 with_patch.particle_rows, with_patch.link_particle_row ? 1 : 0,
                 with_patch.max_body_lambda, patch_lay_z, with_patch.patch_min_z);
    std::fprintf(stderr,
                 "[artic-particle] q_joint with_patch=%.4f control=%.4f (lag=%.4f) "
                 "tip_z with_patch=%.4f control=%.4f qdot_delta_L1=%.6f\n",
                 with_patch.q_joint, control.q_joint, q_lag,
                 with_patch.tip.z, control.tip.z, qdot_delta);

    // (a) a body<->particle row exists whose BODY side is an articulation LINK.
    EXPECT_GT(with_patch.particle_rows, 0u) << "no body<->particle coupling rows emitted";
    EXPECT_TRUE(with_patch.link_particle_row)
        << "the body side of every coupling row was a free/static body -- the "
           "articulation LINK never coupled to a particle";
    EXPECT_GT(with_patch.max_body_lambda, 0.0f)
        << "the link produced no body-side reaction impulse on the particles";

    // (b) two-way, direction 1: the link pushed the particles DOWN below the lay.
    EXPECT_LT(with_patch.patch_min_z, patch_lay_z - 0.005f)
        << "the arm link did not push the particles down (no displacement)";
    // (b) two-way, direction 2: the reaction reached the revolute JOINT -- the patch
    // held the driven arm short of the target the free control reaches, and qdot
    // differs. The impulse flowed through the articulated dynamics (foot<->ground).
    EXPECT_GT(q_lag, 0.01f)
        << "the patch did not hold the driven joint back vs the no-patch control -- "
           "the contact reaction never reached the articulation (one-sided = a bug)";
    EXPECT_GT(qdot_delta, 1.0e-4)
        << "the articulation qdot did not change vs the no-particle control";
}
