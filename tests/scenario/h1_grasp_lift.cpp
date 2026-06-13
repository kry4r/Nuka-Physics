// ---------------------------------------------------------------------------
// M7 T5 — h1_grasp_lift: THE PERMANENT GRASP GATE on the PURE .nks path.
//
// The grasp crux of M7. Reproduces the dev-spike force-balance / BITE /
// disturbance / D1 assertions (previously proven only by the factory-fed
// co-resident spikes) on a union nk::World cooked from the AUTHORED scene file
// alone — NO factory call:
//
//     scene::format::Load("examples/scenes/h1_cup_table.nks")
//   → cook::CookSceneToUnionTemplate(scene, env_count)         (src/scene/cook)
//   → coresident::BuildNkUnionModel(cooked.tmpl, env_count)    (the M4 bridge)
//   → nk::World                                                (the device core)
//
// The cook reads the baked settled IC + grasp config (it reproduced the now-
// deleted 1026-line h1_union_scene_factory's FK/placement/200-step-settle
// product to ~1e-8, the equivalence the transitional T4 gate pinned before T6
// deleted both it and the factory). This gate then drives the SAME
// approach→close→lift choreography the parity oracle uses (TableTorque /
// SnapNk / ReportNk, lifted here) on the .nks-cooked world and asserts the
// grasp physics:
//
//   placement_found — the authored scene carried a placed cup (R4).
//   hold_gravity    — close the grip, remove the table, hold; the cup's
//                     recovered VERTICAL SUPPORT IMPULSE ≈ weight_kick within
//                     ±35% (the dev-spike recover_ok), the cup does not fall,
//                     no static-friction cheat (no table row persists).
//   bite_drops      — grip→0 (BITE), step ~120: the cup DROPS (the honesty
//                     discriminator; a hold surviving grip=0 is a passive wedge).
//   hold_rotation   — sustained 1.0 g lateral (along the escape-gap bisector) +
//                     a one-shot 2.5 rad/s tilt with the grip held: bounded
//                     disp/tilt/spin.
//   D1              — the hold run twice → the cup final pose + qdot byte-
//                     identical (memcmp).
//
// weight_kick = cup_mass·(−gravity_z)·dt = 0.2·9.81·(1/240) ≈ 8.175e-3.
//
// PURE .nks: this gate links nuka_scene_cook (the cook) + the coresident bridge
// (BuildNkUnionModel + the UnionSceneTemplate struct, both alive to M9), NOT
// the factory. The inline constexpr scene constants (kH1UnionDt /
// kH1UnionGravityZ / the asset paths) come from scene/cook/union_scene_constants.hpp
// (their post-factory home, M7 T6); the factory is fully deleted.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <vector>

#include "nk/model/generated/field_ids.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"
#include "phi/device_context.hpp"
#include "scene/cook/union_nk_model.hpp"
#include "scene/cook/union_cook.hpp"
#include "scene/cook/union_scene_constants.hpp"  // constants (post-factory home)
#include "scene/format/nks.hpp"
#include "scene/scene_ir.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace coresident = nuka::runtime::coresident;
namespace cook = nuka::scene::cook;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr const char* kNksPath = "examples/scenes/h1_cup_table.nks";

bool AssetsAvailable() {
    return std::filesystem::exists(kNksPath) &&
           std::filesystem::exists(cook::kH1UnionMjcfDefault) &&
           std::filesystem::exists(cook::kH1UnionCupDefault);
}

// ---- the dev-spike constants (test_h1_dense_grasp.cpp / feasibility probe) ----
constexpr double kPi = 3.14159265358979323846;
// Choreography windows (mirror h1_union_parity's pyref schedule: hold→rest→close
// with the table removed at the lift). The cup is then held by the fingers.
constexpr uint32_t kWindowEnd = 10u;    // hold-phase end.
constexpr uint32_t kRestEnd = 100u;     // rest end == close start.
constexpr uint32_t kLiftAt = 180u;      // table off → friction-only hold.
constexpr uint32_t kHoldSteps = 300u;   // total steps to a settled friction hold.
constexpr uint32_t kLateWindow = 100u;  // late settled window for the force balance.

// Disturbance (the pre-committed worst case, ForceClosureLiftWithDisturbance /
// ConjointHonestGateVerdict): 1.0 g sustained lateral along the escape bisector
// + a one-shot 2.5 rad/s tilt, grip held FIXED.
constexpr float kFcLatAccelG = 1.0f;
constexpr uint32_t kFcPushSteps = 30u;
constexpr uint32_t kFcReleaseSettle = 70u;
constexpr float kFcTiltKickW = 2.5f;
constexpr double kFcMaxDisp = 0.07;
constexpr double kFcMaxTilt = 0.35;
constexpr double kFcMaxFinalW = 1.20;

// BITE.
constexpr uint32_t kFall = 120u;

// ---- per-step closed-loop torque from a drive table (lifted from parity) ----
// u = kp*(target - q) - kd*qdot, clamped to ±tlim; per-LINK (ApplyTorqueDrive
// layout). kill_grip zeroes the wrap-driven grip columns (the BITE switch).
void TableTorque(const std::vector<coresident::H1UnionDriveEntry>& tab,
                 const std::vector<float>& q_link,
                 const std::vector<float>& qdot_link, bool kill_grip,
                 std::vector<float>* torque) {
    for (const auto& d : tab) {
        if (d.link >= torque->size()) continue;
        if (kill_grip && d.grip != 0u) {
            (*torque)[d.link] = 0.0f;
            continue;
        }
        float u = d.kp * (d.target - q_link[d.link]) - d.kd * qdot_link[d.link];
        if (d.tlim > 0.0f) u = std::max(-d.tlim, std::min(d.tlim, u));
        (*torque)[d.link] = u;
    }
}

// ---- NK-side cup snapshot (downloads the fields the comparison reads) ----
struct CupSnap {
    Transform pose;     // cup body_pose (position + orientation).
    Vec3 vel;           // cup body_linear_velocity.
    Vec3 angvel;        // cup body_angular_velocity.
    std::vector<float> qdot;  // per-link qdot (for D1).
};

CupSnap SnapCup(nk::World& w, uint32_t links, uint32_t cup_local) {
    CupSnap s;
    Transform pose;
    Vec3 v, wv;
    EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::BodyPose, &pose,
                                          sizeof(Transform),
                                          cup_local * sizeof(Transform)));
    EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::BodyLinearVelocity, &v,
                                          sizeof(Vec3), cup_local * sizeof(Vec3)));
    EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::BodyAngularVelocity, &wv,
                                          sizeof(Vec3), cup_local * sizeof(Vec3)));
    s.pose = pose;
    s.vel = v;
    s.angvel = wv;
    s.qdot.resize(links);
    EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::Qdot, s.qdot.data(),
                                          links * sizeof(float)));
    return s;
}

// Inject a velocity impulse on the cup (the nk equivalent of the legacy
// coresident stepper's ApplyCupImpulse: dv on linear, dw on angular,
// applied BEFORE the next Step()'s contact phase).
void ApplyCupImpulse(nk::World& w, uint32_t cup_local, const Vec3& dv,
                     const Vec3& dw) {
    Vec3 v, wv;
    ASSERT_TRUE(w.GetData().DownloadField(nk::FieldId::BodyLinearVelocity, &v,
                                          sizeof(Vec3), cup_local * sizeof(Vec3)));
    ASSERT_TRUE(w.GetData().DownloadField(nk::FieldId::BodyAngularVelocity, &wv,
                                          sizeof(Vec3), cup_local * sizeof(Vec3)));
    v += dv;
    wv += dw;
    ASSERT_TRUE(w.GetData().UploadField(nk::FieldId::BodyLinearVelocity, &v,
                                        sizeof(Vec3), cup_local * sizeof(Vec3)));
    ASSERT_TRUE(w.GetData().UploadField(nk::FieldId::BodyAngularVelocity, &wv,
                                        sizeof(Vec3), cup_local * sizeof(Vec3)));
}

// ---- NK-side contact/impulse report (lifted from parity ReportNk) ----
struct NkReport {
    uint32_t foot_normal_rows = 0;
    uint32_t finger_contacts = 0;
    uint32_t table_rows = 0;     // cup×table normal rows (the static-cheat detector).
    double cup_vertical_impulse = 0.0;    // finger rows: lambda * j_cup.z.
    double table_vertical_impulse = 0.0;  // table rows: lambda * j_cup.z.
};

NkReport ReportNk(nk::World& w, uint32_t rows_per_env, uint32_t slot_count,
                  uint32_t n_feet, uint32_t n_fingers) {
    NkReport r;
    std::vector<nk::NkRow> rows(rows_per_env);
    std::vector<float> lambda(rows_per_env);
    EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::Urows, rows.data(),
                                          rows.size() * sizeof(nk::NkRow)));
    EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::Lambda, lambda.data(),
                                          lambda.size() * sizeof(float)));
    for (uint32_t i = 0; i < rows_per_env; ++i) {
        const nk::NkRow& row = rows[i];
        if (!(row.flags & nk::nk_row_flags::kActive)) continue;
        const bool friction = (row.flags & nk::nk_row_flags::kFriction) != 0;
        const bool a_art = row.a.kind == nk::kNkSideArtic;
        const bool b_rigid = row.b.kind == nk::kNkSideRigid;
        const bool a_rigid = row.a.kind == nk::kNkSideRigid;
        const bool b_static = row.b.kind == nk::kNkSideStatic;
        if (a_art && b_static) {  // foot × ground.
            if (!friction) ++r.foot_normal_rows;
        } else if (a_art && b_rigid) {  // finger × cup.
            r.cup_vertical_impulse +=
                static_cast<double>(lambda[i]) * row.b.jlin.z;
        } else if (a_rigid && b_static) {  // cup × table.
            r.table_vertical_impulse +=
                static_cast<double>(lambda[i]) * row.a.jlin.z;
            if (!friction) ++r.table_rows;
        }
    }
    // Finger contact points = UcontactCount on the finger slots [n_feet,
    // n_feet+n_fingers) (the FingerSphereHull class; slot order feet→fingers→table).
    std::vector<uint32_t> counts(slot_count);
    EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::UcontactCount,
                                          counts.data(),
                                          counts.size() * sizeof(uint32_t)));
    for (uint32_t i = n_feet; i < n_feet + n_fingers && i < slot_count; ++i)
        r.finger_contacts += counts[i];
    return r;
}

// ---- the cooked-from-.nks union template (NO factory) ----
// nk::Model is MOVE-ONLY (model.hpp:364), so we cache only the cooked product +
// the capacity metadata and BUILD A FRESH BuildNkUnionModel per World (exactly as
// h1_union_parity rebuilds its model + replay twin).
struct CookedWorld {
    cook::CookedUnionScene cooked;
    uint32_t links = 0;
    uint32_t cup_local = 0;
    uint32_t rows_per_env = 0;
    uint32_t slot_count = 0;
    uint32_t n_feet = 0;
    uint32_t n_fingers = 0;
    float dt = 0.0f;
};

// Cook the .nks (the heavy part; done once) + read the UnionCsr model capacities.
CookedWorld CookFromNks() {
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

// A fresh union nk::Model cooked from the .nks template (no factory). nk::World
// is not movable (deleted copy + user dtor), so the caller constructs the World
// in-place from this model.
nk::Model FreshModel(const CookedWorld& cw) {
    return coresident::BuildNkUnionModel(cw.cooked.tmpl, 1u);
}

// The union SolverConfig the parity oracle / perf gate use (64 vel iters, no
// damping fold — the legacy union CRBA).
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

// Seed the per-link torque buffer with the template grip torque (the legacy
// coresident world's action_torque_host_ init the parity gate mirrors).
std::vector<float> InitTorque(const CookedWorld& cw) {
    std::vector<float> t(cw.links, 0.0f);
    const auto& g = cw.cooked.tmpl.grip_torque;
    for (uint32_t l = 0; l < cw.links && l < g.size(); ++l) t[l] = g[l];
    return t;
}

// Drive ONE step of the approach→close→lift choreography on the nk world,
// returning the post-step report. `s` selects the drive table by phase;
// `kill_grip` forces the grip columns to 0 (the BITE). `tilt_off` removes the
// table once (idempotent — the caller passes it at kLiftAt).
NkReport DriveStep(nk::World& w, const CookedWorld& cw, uint32_t s,
                   bool kill_grip, std::vector<float>* torque,
                   std::vector<float>* q, std::vector<float>* qdot) {
    if (s == kLiftAt) {
        const uint32_t off = 0u;
        EXPECT_TRUE(w.GetData().UploadField(nk::FieldId::TableEnabled, &off,
                                            sizeof(uint32_t)));
    }
    const auto& tab = (s < kWindowEnd)  ? cw.cooked.drive_hold
                      : (s < kRestEnd)  ? cw.cooked.drive_rest
                                        : cw.cooked.drive_close;
    EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::Q, q->data(),
                                          cw.links * sizeof(float)));
    EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::Qdot, qdot->data(),
                                          cw.links * sizeof(float)));
    TableTorque(tab, *q, *qdot, kill_grip, torque);
    EXPECT_TRUE(w.GetData().UploadField(nk::FieldId::DriveTarget, torque->data(),
                                        torque->size() * sizeof(float)));
    const nk::StepResult sr = w.Step();
    EXPECT_TRUE(sr.AllOk()) << "an nk op failed/unsupported at step " << s;
    return ReportNk(w, cw.rows_per_env, cw.slot_count, cw.n_feet, cw.n_fingers);
}

// Step the FULL hold (approach→close, table off at kLiftAt) for `n_steps`,
// accumulating the late-window force balance. Returns the final cup snapshot.
// `out_recovered` = mean cup vertical impulse over the late window;
// `out_any_static` = a table (static) row persisted in the late window;
// `out_min_finger` = min finger contacts in the late window.
CupSnap RunHold(nk::World& w, const CookedWorld& cw, uint32_t n_steps,
                double* out_recovered, bool* out_any_static,
                uint32_t* out_min_finger, double* out_cup_z0) {
    std::vector<float> torque = InitTorque(cw);
    for (uint32_t l = 0; l < cw.links && l < cw.cooked.tmpl.grip_torque.size(); ++l)
        torque[l] = cw.cooked.tmpl.grip_torque[l];
    std::vector<float> q(cw.links), qdot(cw.links);
    double rec_sum = 0.0;
    uint32_t rec_n = 0u;
    bool any_static = false;
    uint32_t min_finger = ~0u;
    // HONESTY GUARD (M7 review, one-sided-filter finding): the force balance
    // below averages ONLY cvi>0 finger-contact steps. Count any late finger-
    // contact step whose cup vertical impulse is non-positive so we can pin that
    // the filter drops NOTHING (else the mean would be optimistically inflated).
    uint32_t late_nonpos = 0u;
    {
        // record the initial cup z (pre-step) for the bounded-fall check.
        Transform pose0;
        EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::BodyPose, &pose0,
                                              sizeof(Transform),
                                              cw.cup_local * sizeof(Transform)));
        *out_cup_z0 = pose0.position.z;
    }
    const uint32_t late_start = (n_steps > kLateWindow) ? n_steps - kLateWindow : 0u;
    for (uint32_t s = 0; s < n_steps; ++s) {
        const NkReport rep =
            DriveStep(w, cw, s, /*kill_grip=*/false, &torque, &q, &qdot);
        if (s >= late_start) {
            // table off (post kLiftAt): the cup is held by fingers ONLY.
            if (rep.table_rows > 0u) any_static = true;
            if (rep.finger_contacts > 0u && rep.cup_vertical_impulse > 0.0) {
                rec_sum += rep.cup_vertical_impulse;
                ++rec_n;
            }
            if (rep.finger_contacts > 0u && rep.cup_vertical_impulse <= 0.0)
                ++late_nonpos;
            min_finger = std::min(min_finger, rep.finger_contacts);
        }
    }
    // The one-sided cvi>0 filter must be a no-op: every late finger-contact step
    // is genuinely supporting (cvi>0). If a regression makes the grip intermittent
    // (a contact step pushes the cup non-upward), this fires instead of the mean
    // silently averaging only the favorable steps. (late_nonpos==0 at authoring.)
    EXPECT_EQ(late_nonpos, 0u)
        << "one-sided cvi>0 filter dropped " << late_nonpos
        << " late finger-contact step(s) -> force-balance mean optimistically biased";
    *out_recovered = rec_n ? rec_sum / rec_n : 0.0;
    *out_any_static = any_static;
    *out_min_finger = (min_finger == ~0u) ? 0u : min_finger;
    return SnapCup(w, cw.links, cw.cup_local);
}

}  // namespace

// ===========================================================================
// THE GATE. placement_found + hold_gravity (force balance) + bite_drops (BITE)
// + hold_rotation (disturbance) + D1, on the PURE .nks-cooked union nk::World.
// ===========================================================================
TEST(H1GraspLift, GraspHoldBiteDisturbanceFromNks) {
    if (!AssetsAvailable())
        GTEST_SKIP() << "h1_cup_table.nks / h1_with_hand / cup not present";

    nphi::Device* dev = nphi::InitBestDevice();
    ASSERT_NE(dev, nullptr);
    nphi::Backend* backend = nphi::DeviceInitBackend(dev, nullptr);
    ASSERT_NE(backend, nullptr);

    // ---- the cook (NO factory): Load(.nks) → cook → BuildNkUnionModel ----
    CookedWorld cw = CookFromNks();
    ASSERT_EQ(cw.cooked.dof_stride, 51u);
    ASSERT_EQ(cw.n_feet, 4u);
    ASSERT_EQ(cw.n_fingers, 30u);
    const double weight_kick =
        static_cast<double>(cw.cooked.cup_mass) *
        static_cast<double>(-cook::kH1UnionGravityZ) *
        static_cast<double>(cw.dt);

    // ===== placement_found (R4: the authored scene baked a placed cup) =====
    EXPECT_TRUE(cw.cooked.place_found)
        << "the .nks did not carry a placed cup (place_found=false)";

    const nk::Pipeline::SolverConfig cfg = UnionCfg(cw.dt);

    // ===== hold_gravity (FORCE BALANCE — the non-negotiable core) =====
    double recovered_fimp = 0.0, cup_z0 = 0.0;
    bool any_static = false;
    uint32_t min_finger = 0u;
    CupSnap hold_final;
    {
        nk::Model m = FreshModel(cw);  // fresh per World (Model is move-only).
        nk::World w(std::move(m), 1u, dev, backend, cfg);
        ASSERT_TRUE(w.Ready());
        hold_final = RunHold(w, cw, kHoldSteps, &recovered_fimp, &any_static,
                             &min_finger, &cup_z0);
    }
    const double fimp_dev = std::fabs(recovered_fimp - weight_kick);
    const double fimp_pct = 100.0 * fimp_dev / weight_kick;
    const double cup_dz = std::fabs(hold_final.pose.position.z - cup_z0);
    const bool recover_ok = fimp_dev < 0.35 * weight_kick;
    const bool fall_ok = cup_dz < 0.07;  // the cup did NOT fall away.
    const bool static_ok = !any_static;  // no table-row static-friction cheat.
    const bool contact_ok = min_finger > 0u;
    std::printf("[H1-GRASP-LIFT] hold_gravity: recovered_fimp=%.4e (mg.dt=%.4e, "
                "dev=%.1f%%) cup_dz=%.4f finger_min=%u any_static=%d "
                "recover_ok=%d fall_ok=%d static_ok=%d\n",
                recovered_fimp, weight_kick, fimp_pct, cup_dz, min_finger,
                any_static ? 1 : 0, recover_ok ? 1 : 0, fall_ok ? 1 : 0,
                static_ok ? 1 : 0);
    EXPECT_TRUE(recover_ok)
        << "FORCE-BALANCE FAIL: recovered cup support " << recovered_fimp
        << " vs weight_kick " << weight_kick << " (dev " << fimp_pct
        << "%, bound 35%)";
    EXPECT_NEAR(recovered_fimp, weight_kick, 0.35 * weight_kick);
    EXPECT_TRUE(fall_ok) << "the cup fell away during the friction-only hold: dz="
                         << cup_dz;
    EXPECT_TRUE(static_ok)
        << "a cup×table (static) row persisted after the lift — static cheat";
    EXPECT_TRUE(contact_ok) << "the fingers lost the cup in the late hold window";

    // ===== bite_drops (BITE — the honesty discriminator, non-negotiable) =====
    double bite_z_pre = 0.0, bite_vz = 0.0, bite_drop = 0.0;
    bool falls = false;
    {
        nk::Model m = FreshModel(cw);
        nk::World w(std::move(m), 1u, dev, backend, cfg);
        ASSERT_TRUE(w.Ready());
        std::vector<float> torque = InitTorque(cw);
        std::vector<float> q(cw.links), qdot(cw.links);
        // hold to a settled friction grip (table off at kLiftAt), grip ON.
        for (uint32_t s = 0; s < kHoldSteps; ++s)
            DriveStep(w, cw, s, /*kill_grip=*/false, &torque, &q, &qdot);
        const CupSnap pre = SnapCup(w, cw.links, cw.cup_local);
        bite_z_pre = pre.pose.position.z;
        // BITE: kill the grip (grip torque → 0) for kFall steps, table stays off.
        for (uint32_t s = 0; s < kFall; ++s)
            DriveStep(w, cw, kHoldSteps, /*kill_grip=*/true, &torque, &q, &qdot);
        const CupSnap post = SnapCup(w, cw.links, cw.cup_local);
        bite_vz = post.vel.z;
        bite_drop = bite_z_pre - post.pose.position.z;
        falls = (bite_drop > 0.02) || (bite_vz < -0.10);
    }
    const double free_fall = 0.5 * (-cook::kH1UnionGravityZ) *
                             (kFall * cw.dt) * (kFall * cw.dt);
    std::printf("[H1-GRASP-LIFT] bite_drops: z_pre=%.5f drop=%.5f (free_fall=%.4f) "
                "vz=%.4f -> %s\n",
                bite_z_pre, bite_drop, free_fall, bite_vz,
                falls ? "FALLS (active grip)" : "HOLDS (passive wedge — FAKE)");
    EXPECT_TRUE(falls)
        << "BITE FAIL: grip=0 did NOT drop the cup (drop=" << bite_drop
        << " vz=" << bite_vz << ") — a passive wedge, not an active grasp";

    // ===== hold_rotation (disturbance robustness) =====
    double peak_disp = 0.0, peak_tilt = 0.0, final_w = 0.0;
    {
        nk::Model m = FreshModel(cw);
        nk::World w(std::move(m), 1u, dev, backend, cfg);
        ASSERT_TRUE(w.Ready());
        std::vector<float> torque = InitTorque(cw);
        std::vector<float> q(cw.links), qdot(cw.links);
        // hold to a settled friction grip (table off at kLiftAt), grip ON.
        for (uint32_t s = 0; s < kHoldSteps; ++s)
            DriveStep(w, cw, s, /*kill_grip=*/false, &torque, &q, &qdot);
        const CupSnap held = SnapCup(w, cw.links, cw.cup_local);
        const Vec3 c_held = held.pose.position;
        const Quat q_held = held.pose.rotation;

        // Escape-gap bisector: the cup hull is ~axisymmetric and the wrap is an
        // opposed 2-sided grip, so the worst-case lateral escape direction is
        // ill-defined in-plane (no preferred gap). Push along the FIXED world +X
        // bisector (a definite, on-record worst case along the hand-open axis);
        // the in-plane tilt axis is perpendicular (a roll about the escape line),
        // exactly the dev-spike's worst-case (RunForceClosureDist) geometry.
        const Vec3 push_dir{1.0f, 0.0f, 0.0f};
        const Vec3 tilt_axis{-push_dir.y, push_dir.x, 0.0f};
        const Vec3 dv_step =
            push_dir * (kFcLatAccelG * (-cook::kH1UnionGravityZ) * cw.dt);

        const uint32_t kTotal = kFcPushSteps + kFcReleaseSettle;
        for (uint32_t s = 0; s < kTotal; ++s) {
            if (s < kFcPushSteps) {
                if (s == 0u)  // ONE-SHOT tilt (impulse accumulates → apply once).
                    ApplyCupImpulse(w, cw.cup_local, dv_step,
                                    tilt_axis * kFcTiltKickW);
                else
                    ApplyCupImpulse(w, cw.cup_local, dv_step, Vec3::Zero());
            }
            // grip held FIXED at the close table (no table; no BITE).
            DriveStep(w, cw, kHoldSteps, /*kill_grip=*/false, &torque, &q, &qdot);
            const CupSnap now = SnapCup(w, cw.links, cw.cup_local);
            peak_disp = std::max(
                peak_disp,
                static_cast<double>((now.pose.position - c_held).Length()));
            const Quat q_rel = q_held.Conjugate() * now.pose.rotation;
            const double tilt =
                2.0 * std::acos(std::min(1.0, std::fabs((double)q_rel.w)));
            peak_tilt = std::max(peak_tilt, tilt);
        }
        final_w = SnapCup(w, cw.links, cw.cup_local).angvel.Length();
    }
    const bool disp_ok = peak_disp < kFcMaxDisp;
    const bool tilt_ok = peak_tilt < kFcMaxTilt;
    const bool spin_ok = final_w < kFcMaxFinalW;
    std::printf("[H1-GRASP-LIFT] hold_rotation: peak_disp=%.4f (bound %.3f) "
                "peak_tilt=%.4f (bound %.2f) final|w|=%.3f (bound %.2f) "
                "disp_ok=%d tilt_ok=%d spin_ok=%d\n",
                peak_disp, kFcMaxDisp, peak_tilt, kFcMaxTilt, final_w,
                kFcMaxFinalW, disp_ok ? 1 : 0, tilt_ok ? 1 : 0, spin_ok ? 1 : 0);
    EXPECT_TRUE(disp_ok) << "disturbance: cup translated past the bound: "
                         << peak_disp;
    EXPECT_TRUE(tilt_ok) << "disturbance: cup tilted past the bound: " << peak_tilt;
    EXPECT_TRUE(spin_ok) << "disturbance: residual spin past the bound: " << final_w;

    // ===== D1 determinism (ForceClosureLiftWithDisturbanceDeterministicTwoRun) =====
    // Two independent holds → the cup final pose + qdot byte-identical.
    CupSnap d1a, d1b;
    {
        double rr;
        bool as;
        uint32_t mf;
        double z0;
        nk::Model ma = FreshModel(cw);
        nk::World wa(std::move(ma), 1u, dev, backend, cfg);
        ASSERT_TRUE(wa.Ready());
        d1a = RunHold(wa, cw, kHoldSteps, &rr, &as, &mf, &z0);
        nk::Model mb = FreshModel(cw);
        nk::World wb(std::move(mb), 1u, dev, backend, cfg);
        ASSERT_TRUE(wb.Ready());
        d1b = RunHold(wb, cw, kHoldSteps, &rr, &as, &mf, &z0);
    }
    const int pose_cmp =
        std::memcmp(&d1a.pose, &d1b.pose, sizeof(Transform));
    const int vel_cmp = std::memcmp(&d1a.vel, &d1b.vel, sizeof(Vec3));
    const int angvel_cmp = std::memcmp(&d1a.angvel, &d1b.angvel, sizeof(Vec3));
    ASSERT_EQ(d1a.qdot.size(), d1b.qdot.size());
    const int qdot_cmp = std::memcmp(d1a.qdot.data(), d1b.qdot.data(),
                                     d1a.qdot.size() * sizeof(float));
    std::printf("[H1-GRASP-LIFT] D1: cup pose memcmp=%d vel=%d angvel=%d "
                "qdot=%d (0 == byte-identical)\n",
                pose_cmp, vel_cmp, angvel_cmp, qdot_cmp);
    EXPECT_EQ(pose_cmp, 0) << "D1: cup final pose diverged across 2 runs";
    EXPECT_EQ(vel_cmp, 0) << "D1: cup final velocity diverged across 2 runs";
    EXPECT_EQ(angvel_cmp, 0) << "D1: cup final angular velocity diverged across 2 runs";
    EXPECT_EQ(qdot_cmp, 0) << "D1: gripper qdot diverged across 2 runs";

    nphi::BackendFree(backend);
}

// ===========================================================================
// PLAN-REPLAY D1 — folded from h1_union_parity (M9 T10, before T11 deletes it).
//
// h1_union_parity's UNIQUE non-legacy coverage was an nk twin replaying the
// RECORDED per-step action stream through StepPlanned() (the CUDA-graph plan:
// ONE capture, kHoldSteps replays with AssembleRows/SolveRowsBlockIsland inside
// the graph) landing BYTE-IDENTICAL to the Step() world — i.e. that the planned
// graph path reproduces the eager Step() path EXACTLY on the full H1 UNION
// articulation+contact scene (51-DOF floating base + feet + 30 fingertips + cup
// + table). The grasp gate's existing D1 above is a two-run Step()-vs-Step()
// check (run-to-run determinism); coupled_grasp_soft asserts StepPlanned==Step
// but only on the TET/PARTICLE coupled scene, NOT the union articulation
// row-solve graph. This test folds that union-scene plan==step byte-identity
// onto the pure .nks cook so the coverage survives h1_union_parity's deletion.
// ===========================================================================
TEST(H1GraspLift, GraspPlanReplayByteIdenticalFromNks) {
    if (!AssetsAvailable())
        GTEST_SKIP() << "h1_cup_table.nks / h1_with_hand / cup not present";

    nphi::Device* dev = nphi::InitBestDevice();
    ASSERT_NE(dev, nullptr);
    nphi::Backend* backend = nphi::DeviceInitBackend(dev, nullptr);
    ASSERT_NE(backend, nullptr);

    CookedWorld cw = CookFromNks();
    ASSERT_EQ(cw.cooked.dof_stride, 51u);
    const nk::Pipeline::SolverConfig cfg = UnionCfg(cw.dt);

    // ---- the Step() world: run the hold choreography, RECORD the per-step
    // DriveTarget stream (DriveStep computes the closed-loop torque from the
    // world's OWN q/qdot and uploads it; the post-call `torque` buffer IS the
    // DriveTarget uploaded that step — exactly what h1_union_parity records). ---
    std::vector<std::vector<float>> recorded_torque;
    recorded_torque.reserve(kHoldSteps);
    CupSnap eager;
    {
        nk::Model m = FreshModel(cw);
        nk::World w(std::move(m), 1u, dev, backend, cfg);
        ASSERT_TRUE(w.Ready());
        std::vector<float> torque = InitTorque(cw);
        std::vector<float> q(cw.links), qdot(cw.links);
        for (uint32_t s = 0; s < kHoldSteps; ++s) {
            DriveStep(w, cw, s, /*kill_grip=*/false, &torque, &q, &qdot);
            recorded_torque.push_back(torque);  // the DriveTarget uploaded at s.
        }
        eager = SnapCup(w, cw.links, cw.cup_local);
    }
    ASSERT_EQ(recorded_torque.size(), kHoldSteps);

    // ---- the StepPlanned() twin: replay the RECORDED stream through the
    // CUDA-graph plan (ONE capture, kHoldSteps replays). Table off at kLiftAt
    // exactly as the Step() world did inside DriveStep. ----
    CupSnap planned;
    {
        nk::Model m = FreshModel(cw);
        nk::World w(std::move(m), 1u, dev, backend, cfg);
        ASSERT_TRUE(w.Ready());
        for (uint32_t s = 0; s < kHoldSteps; ++s) {
            if (s == kLiftAt) {
                const uint32_t off = 0u;
                ASSERT_TRUE(w.GetData().UploadField(nk::FieldId::TableEnabled,
                                                    &off, sizeof(uint32_t)));
            }
            ASSERT_TRUE(w.GetData().UploadField(
                nk::FieldId::DriveTarget, recorded_torque[s].data(),
                recorded_torque[s].size() * sizeof(float)));
            ASSERT_EQ(w.StepPlanned(), nphi::Status::Ok)
                << "plan execute failed at step " << s;
        }
        planned = SnapCup(w, cw.links, cw.cup_local);
    }

    const int pose_cmp = std::memcmp(&eager.pose, &planned.pose, sizeof(Transform));
    const int vel_cmp = std::memcmp(&eager.vel, &planned.vel, sizeof(Vec3));
    const int angvel_cmp =
        std::memcmp(&eager.angvel, &planned.angvel, sizeof(Vec3));
    ASSERT_EQ(eager.qdot.size(), planned.qdot.size());
    const int qdot_cmp = std::memcmp(eager.qdot.data(), planned.qdot.data(),
                                     eager.qdot.size() * sizeof(float));
    std::printf("[H1-GRASP-LIFT] plan-replay D1: %u StepPlanned replays vs Step -> "
                "cup pose memcmp=%d vel=%d angvel=%d qdot=%d (0 == byte-identical)\n",
                kHoldSteps, pose_cmp, vel_cmp, angvel_cmp, qdot_cmp);
    EXPECT_EQ(pose_cmp, 0)
        << "plan-replay: cup final pose diverged from Step on the union scene";
    EXPECT_EQ(vel_cmp, 0)
        << "plan-replay: cup final velocity diverged from Step";
    EXPECT_EQ(angvel_cmp, 0)
        << "plan-replay: cup final angular velocity diverged from Step";
    EXPECT_EQ(qdot_cmp, 0)
        << "plan-replay: gripper qdot diverged from Step (StepPlanned != Step)";

    nphi::BackendFree(backend);
}
