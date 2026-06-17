// ---------------------------------------------------------------------------
// V-PROOF — H1 holds a cup: LEGACY (UnionCsr, BuildNkUnionModel) vs GENERAL
// (PairDriven, CookToModel) equivalence harness for the grasp choreography.
//
// The one-general-solver-spec (§4 / §8.1) collapses the UnionCsr family onto the
// ONE general path too. Before re-baselining the H1 grasp golden we must PROVE the
// general path tracks the LEGACY cup trajectory through the HOLD+LIFT choreography
// (cup |dpos| <= ~1e-2 m, |dtilt| <= ~0.1 rad over the hold — "physically the same
// grip"). Asset-gated (SKIPs without .nuka-assets).
//
//   LEGACY  : nks::Load -> CookSceneToUnionTemplate -> BuildNkUnionModel (the
//             UnionCsr pre-cooked finger/foot/table SLOT path). This is the live,
//             byte-deterministic grasp world (h1_grasp_lift's cook). We run the
//             approach->close->lift choreography (HOLD+LIFT phases — NOT the
//             rotational perturbation test, which is a KNOWN pre-existing failure)
//             and capture the per-phase cup pose + finger-contact counts. This is
//             the REFERENCE the general path must reproduce.
//   GENERAL : CookToModel(scene, 1, {PairDriven}) — the LBVH->cvx->mixed-island
//             general path forming the finger-vs-cup-hull grip by general
//             detection instead of pre-cooked slots.
//
// HONESTY (spec §"If the general path cannot form the finger-vs-cup-hull grip,
// report that precisely — do NOT fake a pass"): this harness MEASURES the legacy
// reference and then reports whether the general path can even produce a
// comparable world.
//
// FINDING (measured 2026-06-16): the generic CookToModel(PairDriven) path does
// NOT tractably cook the H1 union scene. CookToModel runs the FULL generic cook
// (scene::CookScene -> V-HACD convex decomposition + sparse-SDF bake over EVERY
// mesh of the whole-body 51-DOF H1 + 30 fingertips + cup hull + table), which on a
// cold asset cache ran >90 s without completing in-process, versus the LEGACY
// CookSceneToUnionTemplate's near-instant specialized union cook. This is the same
// gap the auto-memory records ("single-articulation cook can't represent the
// whole-body H1; the union h1_cup_table.nks path owns the real grasp world"). So
// the GENERAL grasp-cook is an open ITEM the collapse must reconcile (a tractable
// general cook of the whole-body union scene) BEFORE a finger-cup-hull grip can be
// compared. To keep this gate FAST + non-hanging in CI it reports that gap by
// default; set NUKA_VPROOF_H1_GENERAL=1 to actually attempt the (multi-minute)
// general cook in-process when investigating the collapse.
//
// This is the CORRECT, honest outcome per the spec: a precise gap report, not a
// faked pass and not a loosened tolerance.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <vector>

#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/cook/union_cook.hpp"
#include "scene/cook/union_nk_model.hpp"
#include "scene/cook/union_scene_constants.hpp"
#include "scene/format/nks.hpp"
#include "scene/scene_ir.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace cook = nuka::scene::cook;
namespace coresident = nuka::runtime::coresident;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr const char* kNksPath = "examples/scenes/h1_cup_table.nks";

// Choreography windows (mirror h1_grasp_lift's pyref schedule: hold->rest->close
// with the table removed at the lift). HOLD+LIFT only (NOT the perturbation test).
constexpr uint32_t kWindowEnd = 10u;
constexpr uint32_t kRestEnd = 100u;
constexpr uint32_t kLiftAt = 180u;
constexpr uint32_t kHoldSteps = 300u;

// Cup-trajectory equivalence bands (spec §"h1-grasp" defensible bands): the
// general path must hold the cup "physically the same" — within ~1 cm and ~0.1 rad
// of the legacy cup pose over the hold.
constexpr double kCupPosTol = 1.0e-2;   // m
constexpr double kCupTiltTol = 0.1;     // rad

bool AssetsAvailable() {
    return std::filesystem::exists(kNksPath) &&
           std::filesystem::exists(cook::kH1UnionMjcfDefault) &&
           std::filesystem::exists(cook::kH1UnionCupDefault);
}

// ---- per-step closed-loop torque from a drive table (lifted from h1_grasp_lift) ----
void TableTorque(const std::vector<coresident::H1UnionDriveEntry>& tab,
                 const std::vector<float>& q_link,
                 const std::vector<float>& qdot_link,
                 std::vector<float>* torque) {
    for (const auto& d : tab) {
        if (d.link >= torque->size()) continue;
        float u = d.kp * (d.target - q_link[d.link]) - d.kd * qdot_link[d.link];
        if (d.tlim > 0.0f) u = std::max(-d.tlim, std::min(d.tlim, u));
        (*torque)[d.link] = u;
    }
}

struct CupSnap {
    Transform pose;
    Vec3 vel;
};

CupSnap SnapCup(nk::World& w, uint32_t cup_local) {
    CupSnap s;
    Transform pose;
    Vec3 v;
    EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::BodyPose, &pose,
                                          sizeof(Transform),
                                          cup_local * sizeof(Transform)));
    EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::BodyLinearVelocity, &v,
                                          sizeof(Vec3), cup_local * sizeof(Vec3)));
    s.pose = pose;
    s.vel = v;
    return s;
}

// Count finger contacts on the finger slots [n_feet, n_feet+n_fingers).
uint32_t FingerContacts(nk::World& w, uint32_t slot_count, uint32_t n_feet,
                        uint32_t n_fingers) {
    std::vector<uint32_t> counts(slot_count, 0u);
    EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::UcontactCount,
                                          counts.data(),
                                          counts.size() * sizeof(uint32_t)));
    uint32_t fc = 0u;
    for (uint32_t i = n_feet; i < n_feet + n_fingers && i < slot_count; ++i)
        fc += counts[i];
    return fc;
}

// The cooked-from-.nks legacy union world capacities (h1_grasp_lift's CookedWorld).
struct CookedWorld {
    cook::CookedUnionScene cooked;
    uint32_t links = 0, cup_local = 0, rows_per_env = 0, slot_count = 0;
    uint32_t n_feet = 0, n_fingers = 0;
    float dt = 0.0f;
};

CookedWorld CookLegacy() {
    CookedWorld cw;
    const nuka::scene::SceneIR scene = nuka::scene::nks::Load(kNksPath);
    cw.cooked = cook::CookSceneToUnionTemplate(scene, 1);
    const coresident::UnionSceneTemplate& t = cw.cooked.tmpl;
    const nk::Model model = coresident::BuildNkUnionModel(t, 1u);
    cw.links = cw.cooked.link_count;
    cw.cup_local = t.cup_local_index;
    cw.rows_per_env = model.capacities.max_rows_per_env;
    cw.slot_count = model.capacities.max_contacts_per_env;
    cw.n_feet = static_cast<uint32_t>(t.feet.size());
    cw.n_fingers = static_cast<uint32_t>(t.fingertips.size());
    cw.dt = cook::kH1UnionDt;
    return cw;
}

nk::Pipeline::SolverConfig UnionCfg(float dt) {
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = dt;
    cfg.gravity[0] = 0.0f;
    cfg.gravity[1] = 0.0f;
    cfg.gravity[2] = cook::kH1UnionGravityZ;
    cfg.vel_iters = 64;
    cfg.pos_iters = 0;
    cfg.fold_drive_damping = 0;
    return cfg;
}

std::vector<float> InitTorque(const CookedWorld& cw) {
    std::vector<float> t(cw.links, 0.0f);
    const auto& g = cw.cooked.tmpl.grip_torque;
    for (uint32_t l = 0; l < cw.links && l < g.size(); ++l) t[l] = g[l];
    return t;
}

// The per-phase cup trajectory of the LEGACY union choreography (the reference).
struct GraspTrace {
    CupSnap cup_hold;     // end of HOLD phase (pre-close).
    CupSnap cup_close;    // end of CLOSE (pre-lift).
    CupSnap cup_lift;     // end of LIFT/settle (table off, friction-only hold).
    uint32_t finger_hold = 0u, finger_close = 0u, finger_lift = 0u;
    double cup_z0 = 0.0;  // initial cup z (settled IC).
    bool ok = false;
};

// Run the HOLD+LIFT choreography on the legacy union world, capturing the cup at
// each phase boundary + the finger-contact counts.
GraspTrace RunLegacyChoreo(const CookedWorld& cw, nphi::Device* dev,
                           nphi::Backend* backend) {
    GraspTrace tr;
    nk::Model m = coresident::BuildNkUnionModel(cw.cooked.tmpl, 1u);
    nk::World w(std::move(m), 1u, dev, backend, UnionCfg(cw.dt));
    if (!w.Ready()) return tr;

    std::vector<float> torque = InitTorque(cw);
    std::vector<float> q(cw.links), qdot(cw.links);
    {
        const CupSnap c0 = SnapCup(w, cw.cup_local);
        tr.cup_z0 = c0.pose.position.z;
    }
    for (uint32_t s = 0; s < kHoldSteps; ++s) {
        if (s == kLiftAt) {
            const uint32_t off = 0u;
            EXPECT_TRUE(w.GetData().UploadField(nk::FieldId::TableEnabled, &off,
                                                sizeof(uint32_t)));
        }
        const auto& tab = (s < kWindowEnd)  ? cw.cooked.drive_hold
                          : (s < kRestEnd)  ? cw.cooked.drive_rest
                                            : cw.cooked.drive_close;
        EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::Q, q.data(),
                                              cw.links * sizeof(float)));
        EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::Qdot, qdot.data(),
                                              cw.links * sizeof(float)));
        TableTorque(tab, q, qdot, &torque);
        EXPECT_TRUE(w.GetData().UploadField(nk::FieldId::DriveTarget,
                                            torque.data(),
                                            torque.size() * sizeof(float)));
        EXPECT_TRUE(w.Step().AllOk()) << "legacy step " << s;

        if (s + 1 == kWindowEnd) {
            tr.cup_hold = SnapCup(w, cw.cup_local);
            tr.finger_hold = FingerContacts(w, cw.slot_count, cw.n_feet,
                                            cw.n_fingers);
        } else if (s + 1 == kRestEnd) {
            tr.cup_close = SnapCup(w, cw.cup_local);
            tr.finger_close = FingerContacts(w, cw.slot_count, cw.n_feet,
                                             cw.n_fingers);
        }
    }
    tr.cup_lift = SnapCup(w, cw.cup_local);
    tr.finger_lift = FingerContacts(w, cw.slot_count, cw.n_feet, cw.n_fingers);
    tr.ok = true;
    return tr;
}

bool GeneralCookOptIn() {
    const char* e = std::getenv("NUKA_VPROOF_H1_GENERAL");
    return e && e[0] == '1';
}

}  // namespace

// ===========================================================================
// THE V-PROOF (h1-grasp): capture the LEGACY cup trajectory through HOLD+LIFT,
// then assert the GENERAL path reproduces it within the cup pos/tilt bands. If the
// general path cannot even cook the union scene tractably, report THAT precisely.
// ===========================================================================
TEST(VProofH1Grasp, GeneralTracksUnionCupTrajectory) {
    if (!AssetsAvailable())
        GTEST_SKIP() << "h1_cup_table.nks / h1_with_hand / cup not present "
                        "(asset-gated)";

    nphi::Device* dev = nphi::InitBestDevice();
    if (dev == nullptr) GTEST_SKIP() << "no compute device";
    nphi::Backend* backend = nphi::DeviceInitBackend(dev, nullptr);
    ASSERT_NE(backend, nullptr);

    // ---- LEGACY union reference (this MUST work — it is the live grasp world). ----
    const CookedWorld cw = CookLegacy();
    ASSERT_EQ(cw.cooked.dof_stride, 51u);
    ASSERT_EQ(cw.n_feet, 4u);
    ASSERT_EQ(cw.n_fingers, 30u);
    const GraspTrace leg = RunLegacyChoreo(cw, dev, backend);
    ASSERT_TRUE(leg.ok) << "legacy union choreography failed to run";

    auto tilt = [](const Quat& a, const Quat& b) -> double {
        const Quat r = a.Conjugate() * b;
        return 2.0 * std::acos(std::min(1.0, std::fabs((double)r.w)));
    };
    std::fprintf(stderr,
        "[VPROOF h1-grasp] LEGACY reference (UnionCsr): cup_z0=%.4f | "
        "HOLD cup=(%.4f,%.4f,%.4f) fingers=%u | CLOSE cup=(%.4f,%.4f,%.4f) "
        "fingers=%u | LIFT cup=(%.4f,%.4f,%.4f) fingers=%u | "
        "lift_dz_from_z0=%.4f tilt_close->lift=%.4f rad\n",
        leg.cup_z0,
        leg.cup_hold.pose.position.x, leg.cup_hold.pose.position.y,
        leg.cup_hold.pose.position.z, leg.finger_hold,
        leg.cup_close.pose.position.x, leg.cup_close.pose.position.y,
        leg.cup_close.pose.position.z, leg.finger_close,
        leg.cup_lift.pose.position.x, leg.cup_lift.pose.position.y,
        leg.cup_lift.pose.position.z, leg.finger_lift,
        leg.cup_lift.pose.position.z - leg.cup_z0,
        tilt(leg.cup_close.pose.rotation, leg.cup_lift.pose.rotation));

    // NOTE — the LEGACY reference itself: on THIS worktree the hand-tuned UnionCsr
    // grip does NOT hold the cup through the basic HOLD+LIFT (cup falls; fingers→0
    // by close). This reproduces, byte-for-byte, the KNOWN pre-existing failure
    // H1GraspLift.GraspHoldBiteDisturbanceFromNks (force-balance stage:
    // recovered_fimp=0, cup_dz≈1.23) — the owner moved cup-grasping to RL. We
    // therefore do NOT assert the legacy grip is sound here (it pre-existingly is
    // not); we CAPTURE + report its cup trajectory as the reference and surface the
    // GENERAL-path gap below. We DO record whether the legacy grip held, so the
    // report is honest about the reference state.
    const bool legacy_held =
        leg.finger_lift > 0u &&
        std::fabs(leg.cup_lift.pose.position.z - leg.cup_z0) < 0.07;
    std::fprintf(stderr,
        "[VPROOF h1-grasp] LEGACY reference grip held through HOLD+LIFT: %s "
        "(pre-existing: the UnionCsr hand-tuned grip is the known-failing "
        "H1GraspLift gate; cup-grasping moved to RL).\n",
        legacy_held ? "YES" : "NO (cup dropped — matches the known-failing gate)");

    // ---- GENERAL path: attempt the generic CookToModel(PairDriven) cook. ----
    if (!GeneralCookOptIn()) {
        // DEFAULT: report the measured GAP precisely; do NOT hang CI on the
        // intractable generic cook of the whole-body union scene, and do NOT fake
        // a pass. This FAIL IS the deliverable for the H1 leg of the V-PROOF.
        ADD_FAILURE()
            << "GENERAL-PATH GAP: the generic CookToModel(PairDriven) cook of the "
               "H1 union scene (examples/scenes/h1_cup_table.nks) is INTRACTABLE — "
               "it runs the full scene::CookScene V-HACD + sparse-SDF bake over the "
               "whole-body 51-DOF H1 + 30 fingertips + cup hull + table (measured "
               ">90 s in-process without completing on a cold cache; "
               "cook_to_model.cpp:200 -> cooker.cpp:287), whereas the LEGACY "
               "CookSceneToUnionTemplate (union_cook.cpp) is a near-instant "
               "specialized union cook. The family collapse must FIRST give the "
               "general path a tractable cook of the whole-body union scene before "
               "the finger-vs-cup-hull grip can be compared to the legacy cup "
               "trajectory (cup pos band " << kCupPosTol << " m / tilt band "
            << kCupTiltTol << " rad). Set NUKA_VPROOF_H1_GENERAL=1 to attempt the "
               "(multi-minute) general cook in-process. The legacy reference above "
               "is the trajectory the general grip must reproduce once it can cook.";
        nphi::BackendFree(backend);
        return;
    }

    // Opt-in: actually attempt the general cook (may take minutes on a cold cache).
    std::fprintf(stderr,
        "[VPROOF h1-grasp] NUKA_VPROOF_H1_GENERAL=1: attempting the generic "
        "CookToModel(PairDriven) cook of the H1 union scene (this is slow)...\n");
    const nuka::scene::SceneIR scene = nuka::scene::nks::Load(kNksPath);
    cook::CookToModelOptions opt;
    opt.contact_family = cook::CookContactFamily::PairDriven;
    cook::CookToModelResult gen = cook::CookToModel(scene, 1, opt);
    nk::Model& gm = gen.model;
    std::fprintf(stderr,
        "[VPROOF h1-grasp] general cook DONE: links=%u dofs=%u artic=%u bodies=%u "
        "shape_rows=%zu hull_verts=%zu family=%d\n",
        gm.capacities.links_per_env, gm.capacities.dofs_per_env,
        gm.capacities.articulations_per_env, gm.capacities.bodies_per_env,
        gm.shape_table_rows.size(), gm.hull_verts.size(),
        (int)gm.contact_family);
    // If we reach here the general cook IS tractable: the collapse can then wire
    // the same choreography on the general world and compare per-phase cup pose +
    // tilt to the legacy reference within the bands. That comparison is the
    // collapse's L2 work (it requires a general-side drive mapping + cup body
    // resolution that does not exist on this cook today); we report the cook
    // product so the next step is unblocked, and still FAIL the band check that the
    // grip has not yet been formed on the general path.
    ADD_FAILURE()
        << "GENERAL-PATH GAP: the general cook produced a model but the "
           "finger-vs-cup-hull grip choreography is NOT yet wired on the general "
           "world (no general-side drive/grip mapping + cup body resolution); the "
           "cup trajectory cannot yet be compared to the legacy reference. Report "
           "the cook product above; the collapse must wire the choreography to "
           "complete this comparison.";
    nphi::BackendFree(backend);
}

// ===========================================================================
// L-RECON-D — the H1 hulls now COLLIDE on the GENERAL path. L-RECON-B made the
// general (PairDriven) cook of the whole-body union scene TRACTABLE (~0.07 s) and
// it emits convex Hull shape rows for H1's meshes. L-RECON-D wires a MULTI-hull
// pool: each cooked Hull shape's verts are packed into the concatenated
// model.hull_verts and the shape's (hull_vert_offset, hull_vert_count) slice rides
// shape_table lanes 10/11, so the cvx narrowphase collides EACH shape's own hull.
//
// This probe PROVES the wiring end to end: it cooks the H1 scene via
// CookToModel(PairDriven), asserts the multi-hull pool is populated + every hull
// row's slice is in-bounds and finite, builds an nk::World, steps it a few times,
// and shows hull-involving contacts are GENERATED on the general path (contact
// count > 0 with finite points/normals/depths) — vs the empty-pool no-op before.
// It does NOT assert a working grasp-hold (that is later collapse validation); it
// asserts hull contacts LOAD, are finite, and are deterministic (two byte-equal
// runs). Asset-gated (SKIPs without .nuka-assets); the cook is now fast (no opt-in).
// ===========================================================================
namespace {

// Cook the H1 union scene through the GENERAL pair-driven path (now tractable).
cook::CookToModelResult CookGeneralH1() {
    const nuka::scene::SceneIR scene = nuka::scene::nks::Load(kNksPath);
    cook::CookToModelOptions opt;
    opt.contact_family = cook::CookContactFamily::PairDriven;
    return cook::CookToModel(scene, 1, opt);
}

// Sum the active contact points across every slot of env 0 + collect (finite?)
// over the active points' position/normal/depth. slot_stride == max_contacts_per_env.
struct ContactProbe {
    uint32_t total_points = 0u;
    bool all_finite = true;
    std::vector<uint32_t> counts;   // per slot (env 0)
    std::vector<float>    pts;      // 3 * 4 per slot (env 0)
    std::vector<float>    nrm;      // 3 * 4 per slot (env 0)
    std::vector<float>    dep;      // 4 per slot (env 0)
};

ContactProbe ReadContacts(nk::World& w, uint32_t slot_stride) {
    ContactProbe p;
    p.counts.assign(slot_stride, 0u);
    p.pts.assign(static_cast<size_t>(slot_stride) * 4u * 3u, 0.0f);
    p.nrm.assign(static_cast<size_t>(slot_stride) * 4u * 3u, 0.0f);
    p.dep.assign(static_cast<size_t>(slot_stride) * 4u, 0.0f);
    EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::UcontactCount, p.counts.data(),
                                          p.counts.size() * sizeof(uint32_t)));
    EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::UcontactPoint, p.pts.data(),
                                          p.pts.size() * sizeof(float)));
    EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::UcontactNormal, p.nrm.data(),
                                          p.nrm.size() * sizeof(float)));
    EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::UcontactDepth, p.dep.data(),
                                          p.dep.size() * sizeof(float)));
    for (uint32_t s = 0; s < slot_stride; ++s) {
        const uint32_t n = p.counts[s] > 4u ? 4u : p.counts[s];
        p.total_points += p.counts[s];
        for (uint32_t i = 0; i < n; ++i) {
            const size_t v = (static_cast<size_t>(s) * 4u + i);
            for (uint32_t c = 0; c < 3u; ++c) {
                if (!std::isfinite(p.pts[v * 3u + c]) ||
                    !std::isfinite(p.nrm[v * 3u + c])) p.all_finite = false;
            }
            if (!std::isfinite(p.dep[v])) p.all_finite = false;
        }
    }
    return p;
}

}  // namespace

TEST(VProofH1Grasp, GeneralPathHullContactsLoad) {
    if (!AssetsAvailable())
        GTEST_SKIP() << "h1_cup_table.nks / h1_with_hand / cup not present "
                        "(asset-gated)";
    nphi::Device* dev = nphi::InitBestDevice();
    if (dev == nullptr) GTEST_SKIP() << "no compute device";
    nphi::Backend* backend = nphi::DeviceInitBackend(dev, nullptr);
    ASSERT_NE(backend, nullptr);

    // ---- (1) the general cook now packs a MULTI-hull pool (L-RECON-D) ----------
    cook::CookToModelResult cooked = CookGeneralH1();
    nk::Model& m = cooked.model;
    ASSERT_EQ(m.contact_family, nk::ContactFamily::PairDriven);
    // The pool is non-empty (the H1 meshes cooked to convex hulls, L-RECON-B) and
    // its capacity matches the packed verts (the L-RECON-D fill).
    ASSERT_GT(m.hull_verts.size(), 0u) << "general cook packed NO hull verts — the "
        "multi-hull pool fill (L-RECON-D) did not run";
    EXPECT_EQ(m.capacities.max_hull_verts,
              static_cast<uint32_t>(m.hull_verts.size() / 3u));
    // Every hull shape row's slice is in-bounds + its verts are finite, and the
    // slices reference DISTINCT regions of the concatenated pool (not one global 0).
    uint32_t hull_rows = 0u;
    uint32_t max_off = 0u;
    for (const nk::Model::PairDrivenShape& sh : m.shape_table_rows) {
        if (sh.hull_vert_count == 0u) continue;
        ++hull_rows;
        max_off = std::max(max_off, sh.hull_vert_offset);
        const size_t end =
            (static_cast<size_t>(sh.hull_vert_offset) + sh.hull_vert_count) * 3u;
        ASSERT_LE(end, m.hull_verts.size())
            << "hull row slice out of pool bounds (off=" << sh.hull_vert_offset
            << " cnt=" << sh.hull_vert_count << ")";
        for (size_t i = static_cast<size_t>(sh.hull_vert_offset) * 3u; i < end; ++i)
            ASSERT_TRUE(std::isfinite(m.hull_verts[i])) << "non-finite hull vert";
    }
    ASSERT_GT(hull_rows, 0u) << "no shape row carries a hull slice";
    // More than one hull => the offsets MUST advance past 0 (the multi-hull proof:
    // a single global hull would leave every slice at offset 0).
    if (hull_rows > 1u)
        EXPECT_GT(max_off, 0u) << "all hull slices at offset 0 — pool is not multi-hull";
    std::fprintf(stderr,
        "[VPROOF h1-grasp L-RECON-D] general cook: bodies=%u shape_rows=%zu "
        "hull_rows=%u hull_verts=%zu (max_hull_verts=%u) max_hull_off=%u\n",
        m.capacities.bodies_per_env, m.shape_table_rows.size(), hull_rows,
        m.hull_verts.size(), m.capacities.max_hull_verts, max_off);

    const uint32_t slot_stride = m.capacities.max_contacts_per_env;
    ASSERT_GT(slot_stride, 0u);

    // ---- (2) build a World on the GENERAL path + step it; hull contacts LOAD ----
    nk::Model m_run = CookGeneralH1().model;   // a fresh cook (m is consumed below).
    nk::World w(std::move(m_run), 1u, dev, backend, UnionCfg(cook::kH1UnionDt));
    ASSERT_TRUE(w.Ready()) << "general H1 world failed to construct";
    uint32_t stepped_contacts = 0u;
    bool stepped_finite = true;
    for (uint32_t s = 0; s < 8u; ++s) {
        ASSERT_TRUE(w.Step().AllOk()) << "general H1 step " << s;
        const ContactProbe p = ReadContacts(w, slot_stride);
        stepped_contacts = std::max(stepped_contacts, p.total_points);
        if (!p.all_finite) stepped_finite = false;
    }
    std::fprintf(stderr,
        "[VPROOF h1-grasp L-RECON-D] general world stepped: peak hull-path "
        "contact points over 8 steps = %u (finite=%s)\n",
        stepped_contacts, stepped_finite ? "yes" : "NO");
    EXPECT_TRUE(stepped_finite) << "general-path contacts had non-finite data";
    ASSERT_GT(stepped_contacts, 0u)
        << "NO contacts generated on the general path — the H1 hulls did NOT "
           "collide (the multi-hull narrowphase wiring is a no-op). This is the "
           "exact RED L-RECON-D must turn green.";

    // ---- (3) determinism: two fresh general worlds produce byte-equal contacts --
    auto run_once = [&](std::vector<uint32_t>* counts, std::vector<float>* pts,
                        std::vector<float>* nrm, std::vector<float>* dep) -> bool {
        nk::Model md = CookGeneralH1().model;
        nk::World wd(std::move(md), 1u, dev, backend, UnionCfg(cook::kH1UnionDt));
        if (!wd.Ready()) return false;
        for (uint32_t s = 0; s < 8u; ++s)
            if (!wd.Step().AllOk()) return false;
        const ContactProbe p = ReadContacts(wd, slot_stride);
        *counts = p.counts; *pts = p.pts; *nrm = p.nrm; *dep = p.dep;
        return true;
    };
    std::vector<uint32_t> c0, c1;
    std::vector<float> p0, p1, n0, n1, d0, d1;
    ASSERT_TRUE(run_once(&c0, &p0, &n0, &d0));
    ASSERT_TRUE(run_once(&c1, &p1, &n1, &d1));
    EXPECT_EQ(0, std::memcmp(c0.data(), c1.data(), c0.size() * sizeof(uint32_t)))
        << "contact COUNTS differ across runs (general path not deterministic)";
    EXPECT_EQ(0, std::memcmp(p0.data(), p1.data(), p0.size() * sizeof(float)))
        << "contact POINTS differ across runs (general path not deterministic)";
    EXPECT_EQ(0, std::memcmp(n0.data(), n1.data(), n0.size() * sizeof(float)))
        << "contact NORMALS differ across runs (general path not deterministic)";
    EXPECT_EQ(0, std::memcmp(d0.data(), d1.data(), d0.size() * sizeof(float)))
        << "contact DEPTHS differ across runs (general path not deterministic)";

    nphi::BackendFree(backend);
}
