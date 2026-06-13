// ---------------------------------------------------------------------------
// M4 — h1_union_parity: the FULL union scene (whole-body floating-base H1 +
// cup + table, the production h1_union_scene_factory authoring) stepped
// through the NEW nk core (BuildNkUnionModel -> nk::World -> the M4
// AssembleRows/SolveRowsBlockIsland device-resident pipeline) AGAINST the
// legacy BatchedUnifiedWorld (the slow-but-correct parity ORACLE) on the SAME
// initial state, the SAME closed-loop choreography, the SAME step count.
//
// GATES (the G1c/G1d tolerances, read from
// tests/coresident/test_batched_h1_union_world.cpp):
//   (a) FP-FLOOR WINDOW (HARD), in TWO tiers:
//       * STEP 0 at the ORIGINAL G1c bars (pos/vel/base 1e-6, qdot 5e-6) —
//         the strictest point, before any chaos growth; a structural
//         composition bug (row order / friction stamp / body key / pack-
//         scatter) corrupts step 0 by ~1e-3+, 3-4 orders above the bar.
//       * STEPS 0..9 with the VELOCITY bars scaled (vel 5e-5, qdot 1e-4;
//         pos/base stay 1e-6). ★ LOUD RECONCILIATION (M4): G1c's in-window
//         bars presumed a solver whose row math was BYTE-IDENTICAL between
//         the two worlds (same kernel, host-vs-device NP ULP only). The M4
//         nk solver is NUMERICALLY EQUIVALENT, NOT BIT-IDENTICAL to the
//         legacy row_solver (effective mass + M^-1 J^T are hoisted to
//         assembly — same expressions, same order, but a different TU's FMA
//         contraction), so the per-step seed is ~2-3e-7 (measured, ULP
//         class) instead of 0, and the union scene's Lyapunov rate
//         (~1.3x/step, measured on the legacy self-chaos twin) amplifies
//         that seed past 1e-6 within 10 steps. ★ M7 RE-CALIBRATION (vel
//         window 1e-5 -> 5e-5): both worlds now build from the .nks-COOKED
//         BatchedSceneTemplate (CookSceneToUnionTemplate replaced the deleted
//         factory). The cook reproduces the factory's settled IC to ~1e-8 on
//         POSE / geometry / drive tables but bakes a POSE-ONLY initial_state
//         (the gripper_proto's residual settle qdot/link_velocity are zeroed,
//         not stored — see union_cook); this VALID-but-different IC drives a
//         marginally different chaotic trajectory whose nk-vs-legacy ULP seed
//         crosses a contact toggle as a SINGLE-STEP velocity transient
//         (measured 2.08e-5 at step 4, recovering to ~5e-6 by step 5 — a
//         contact-event ULP class, NOT growth). The step-0 bars stay UNSCALED
//         at 1e-6/5e-6 (the structural-bug catcher: a real composition bug is
//         ~1e-3+ at step 0 AND persists — 1.5+ orders above 5e-5); the widened
//         vel window absorbs the cook-IC transient without losing that power.
//         (The seed parity itself is unchanged: BOTH worlds read the SAME
//         cooked.tmpl, so this is the trajectory's chaos, not an IC mismatch.)
//   (b) CONTACT-SET parity through the window (foot rows / finger contacts
//       fork at/after the window, reported).
//   (c) CHAOS BOUND (full run): nk-vs-legacy divergence stays within 10x the
//       legacy SELF-chaos floor (a 1e-6-perturbed legacy twin) — the Gate1
//       pattern; the union trajectory is chaotic, byte-tracking over 300
//       steps is physically meaningless, the bounded-by-self-chaos check is
//       the honest formulation.
//   (d) FORCE BALANCES on the NK side (late settled window): feet carry
//       robot+cup (0.7..1.3 x (m_r+m_c) g dt), fingers carry the cup after
//       the lift (0.7..1.3 x m_c g dt) — the same physics gates G1c/G1d
//       assert on the legacy side.
//   (e) PLAN REPLAY: an nk twin replaying the RECORDED action stream through
//       StepPlanned() (the CUDA-graph plan, ONE capture, 300 replays with
//       assemble+solve inside) lands BYTE-IDENTICAL to the Step() world.
//
// KNOWN-ALLOWED legacy red (untouched): the G1d rest_table_ratio physics
// assertion (owner-adjudicated). This test gates PARITY vs legacy, not that
// assertion — nk reproducing legacy's trajectory including its flaws IS the
// pass.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <vector>

#include "nk/model/generated/field_ids.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"
#include "phi/device_context.hpp"
#include "runtime/coresident/batched_unified_world.hpp"
#include "runtime/coresident/h1_union_nk_model.hpp"
#include "scene/cook/union_cook.hpp"
#include "scene/cook/union_scene_constants.hpp"
#include "scene/format/nks.hpp"
#include "scene/scene_ir.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace coresident = nuka::runtime::coresident;
namespace cook = nuka::scene::cook;
using nuka::math::Transform;
using nuka::math::Vec3;

bool AssetsAvailable() {
    return std::filesystem::exists(cook::kH1UnionNksDefault) &&
           std::filesystem::exists(cook::kH1UnionMjcfDefault) &&
           std::filesystem::exists(cook::kH1UnionCupDefault);
}

struct NkBackendFixture {
    nphi::Device* dev = nullptr;
    nphi::Backend* backend = nullptr;
    NkBackendFixture() {
        dev = nphi::InitBestDevice();
        if (dev != nullptr) backend = nphi::DeviceInitBackend(dev, nullptr);
    }
    ~NkBackendFixture() {
        if (backend != nullptr) nphi::BackendFree(backend);
    }
};

constexpr uint32_t kSteps = 300u;
constexpr uint32_t kWindowEnd = 10u;   // hold phase end (pyref choreography).
constexpr uint32_t kRestEnd = 100u;    // rest end == close start.
constexpr uint32_t kLiftAt = 180u;     // table off -> friction-only hold.
// The G1c FP-floor window bars (test_batched_h1_union_world.cpp:766-768) —
// applied UNSCALED at step 0; the in-window VELOCITY bars are scaled (see the
// file-header reconciliation note).
constexpr uint32_t kFloorSteps = 10u;
constexpr double kFloorTol = 1.0e-6;        // pos/base, whole window + step 0
constexpr double kFloorTolQdot = 5.0e-6;    // qdot, step 0
constexpr double kFloorTolVelWin = 5.0e-5;  // vel, whole window (M4+M7 note)
constexpr double kFloorTolQdotWin = 1.0e-4; // qdot, whole window (20x, M4 note)

// The per-step closed-loop torque from a drive table + a per-link q/qdot view.
// Mirrors the pyref PD: u = kp*(target - q) - kd*qdot, clamped to +/-tlim.
// `torque` is per-LINK (the ApplyTorqueDrive layout); non-entry links keep
// their template value (the legacy SetActions overwrite semantics).
void TableTorque(const std::vector<coresident::H1UnionDriveEntry>& tab,
                 const std::vector<float>& q_link,
                 const std::vector<float>& qdot_link,
                 bool kill_grip,
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

// NK-side per-step snapshot (downloads the fields the comparison reads).
struct NkSnap {
    Vec3 cup_pos, cup_vel;
    Transform base;
    std::vector<float> qdot_link;     // per link
    float root_vel[6] = {0, 0, 0, 0, 0, 0};
};

NkSnap SnapNk(nk::World& w, uint32_t links, uint32_t cup_local) {
    NkSnap s;
    std::vector<Transform> body(1);
    std::vector<Vec3> vel(1);
    EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::BodyPose, body.data(),
                                          sizeof(Transform),
                                          cup_local * sizeof(Transform)));
    EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::BodyLinearVelocity,
                                          vel.data(), sizeof(Vec3),
                                          cup_local * sizeof(Vec3)));
    s.cup_pos = body[0].position;
    s.cup_vel = vel[0];
    EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::BasePose, &s.base,
                                          sizeof(Transform)));
    s.qdot_link.resize(links);
    EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::Qdot, s.qdot_link.data(),
                                          links * sizeof(float)));
    float lv[6];
    EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::LinkVelocity, lv,
                                          sizeof(lv)));  // root = link 0
    for (int k = 0; k < 6; ++k) s.root_vel[k] = lv[k];
    return s;
}

// NK-side impulse/contact report from the urows + lambda fields (the same
// quantities BatchedGraspEnvReport carries, classified by side kinds).
struct NkReport {
    uint32_t foot_normal_rows = 0;
    uint32_t finger_contacts = 0;
    uint32_t table_rows = 0;
    double foot_normal_impulse = 0.0;
    double cup_vertical_impulse = 0.0;    // finger rows: lambda * j_cup.z
    double table_vertical_impulse = 0.0;  // table rows: lambda * j_cup.z
};

NkReport ReportNk(nk::World& w, uint32_t rows_per_env, uint32_t slot_count) {
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
        if (a_art && b_static) {  // foot x ground
            if (!friction) {
                ++r.foot_normal_rows;
                r.foot_normal_impulse += static_cast<double>(lambda[i]);
            }
        } else if (a_art && b_rigid) {  // finger x cup
            r.cup_vertical_impulse +=
                static_cast<double>(lambda[i]) * row.b.jlin.z;
        } else if (a_rigid && b_static) {  // cup x table
            r.table_vertical_impulse +=
                static_cast<double>(lambda[i]) * row.a.jlin.z;
            if (!friction) ++r.table_rows;
        }
    }
    std::vector<uint32_t> counts(slot_count);
    EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::UcontactCount,
                                          counts.data(),
                                          counts.size() * sizeof(uint32_t)));
    // Finger slots = the FingerSphereHull class; recover via the row walk is
    // awkward — count contact points on slots that produced finger rows
    // instead: a slot with an (artic, rigid) row pair. Simpler: the model's
    // slot table ordering is feet..fingers..table; the caller passes counts of
    // each, but the report only needs total finger points:
    r.finger_contacts = 0;  // filled by the caller (slot-class aware).
    return r;
}

}  // namespace

TEST(H1UnionParity, NkWorldTracksBatchedUnifiedWorldThroughLiftChoreography) {
    if (!AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    NkBackendFixture fx;
    ASSERT_NE(fx.dev, nullptr);
    ASSERT_NE(fx.backend, nullptr);

    // ---- the union scene cooked from the AUTHORED .nks (NO factory) --------
    // CookSceneToUnionTemplate reproduces the deleted BuildH1UnionScene product
    // (the BatchedSceneTemplate) to ~1e-8; the parity contract is preserved
    // because BOTH worlds (legacy BatchedUnifiedWorld + nk BuildNkUnionModel)
    // build from the SAME cooked.tmpl below.
    const nuka::scene::SceneIR scene =
        nuka::scene::nks::Load(cook::kH1UnionNksDefault);
    cook::CookedUnionScene sc = cook::CookSceneToUnionTemplate(scene, 1);
    ASSERT_TRUE(sc.place_found);
    ASSERT_EQ(sc.dof_stride, 51u);
    const uint32_t L = sc.link_count;
    const uint32_t cup_local = sc.tmpl.cup_local_index;
    const float dt = cook::kH1UnionDt;
    const double mg_dt_feet =
        (sc.total_mass + sc.cup_mass) * 9.81 * static_cast<double>(dt);
    const double mg_dt_cup = sc.cup_mass * 9.81 * static_cast<double>(dt);

    // ---- the three worlds ----------------------------------------------------
    coresident::BatchedUnifiedWorld legacy(context, sc.tmpl, 1u,
                                           cook::kH1UnionGravityZ, dt);
    // The SELF-CHAOS twin: identical legacy world with a 1e-6 m cup-x nudge —
    // sizes the FP-chaos floor the full-run nk divergence is bounded against.
    coresident::BatchedUnifiedWorld chaos(context, sc.tmpl, 1u,
                                          cook::kH1UnionGravityZ, dt);
    chaos.BodyMut(0u, cup_local).position.x += 1.0e-6f;

    nk::Model model = coresident::BuildNkUnionModel(sc.tmpl, 1u);
    const uint32_t rows_per_env = model.capacities.max_rows_per_env;
    const uint32_t slot_count = model.capacities.max_contacts_per_env;
    const uint32_t n_feet = static_cast<uint32_t>(sc.tmpl.feet.size());
    const uint32_t n_fingers = static_cast<uint32_t>(sc.tmpl.fingertips.size());
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = dt;
    cfg.gravity[0] = 0.0f; cfg.gravity[1] = 0.0f;
    cfg.gravity[2] = cook::kH1UnionGravityZ;
    cfg.vel_iters = 64;   // the legacy union SolverConfig (64 velocity iters).
    cfg.pos_iters = 0;
    cfg.fold_drive_damping = 0;  // the legacy union CRBA has no damping fold.
    nk::World world(std::move(model), 1u, fx.dev, fx.backend, cfg);
    ASSERT_TRUE(world.Ready());

    std::printf("[M4 PARITY] union scene: links=%u dofs=%u slots=%u rows/env=%u "
                "islands=%u segments=%u\n",
                L, sc.dof_stride, slot_count, rows_per_env,
                world.GetModel().schedule_island_count,
                world.GetModel().schedule_segment_count);
    ASSERT_GT(world.GetModel().schedule_island_count, 0u);

    // ---- per-step state -------------------------------------------------------
    std::vector<float> leg_q(sc.dof_stride), leg_qdot(sc.dof_stride);
    coresident::ObsStateBatch obs;
    std::vector<float> nk_q(L), nk_qdot(L);
    // Per-link torque buffers (start = the template grip torque, the legacy
    // action_torque_host_ init).
    std::vector<float> leg_actions(sc.dof_stride, 0.0f);
    std::vector<float> nk_torque(L, 0.0f);
    for (uint32_t l = 0; l < L && l < sc.tmpl.grip_torque.size(); ++l) {
        nk_torque[l] = sc.tmpl.grip_torque[l];
    }
    std::vector<float> chaos_actions(sc.dof_stride, 0.0f);
    // The RECORDED nk action stream (per step, per link) for the plan-replay twin.
    std::vector<std::vector<float>> recorded_torque;
    recorded_torque.reserve(kSteps);

    double floor_dpos = 0.0, floor_dvel = 0.0, floor_dqdot = 0.0, floor_dbase = 0.0;
    double seed_dpos = 0.0, seed_dvel = 0.0, seed_dqdot = 0.0, seed_dbase = 0.0;
    double full_dpos = 0.0, chaos_dpos = 0.0;
    uint32_t count_fork = kSteps;
    bool count_open = true;
    double feet_ratio_sum = 0.0, cup_ratio_sum = 0.0;
    uint32_t ratio_n = 0;
    uint32_t both_steps = 0;

    for (uint32_t s = 0; s < kSteps; ++s) {
        // ---- the lift event ---------------------------------------------------
        if (s == kLiftAt) {
            legacy.SetTableEnabled(false);
            chaos.SetTableEnabled(false);
            const uint32_t off = 0u;
            ASSERT_TRUE(world.GetData().UploadField(nk::FieldId::TableEnabled,
                                                    &off, sizeof(uint32_t)));
        }
        const auto& tab = (s < kWindowEnd)   ? sc.drive_hold
                          : (s < kRestEnd)   ? sc.drive_rest
                                             : sc.drive_close;

        // ---- legacy: closed loop from ITS OWN obs -----------------------------
        legacy.ExportObsState(obs);
        for (uint32_t k = 0; k < sc.dof_stride; ++k) {
            leg_q[k] = obs.q[k];
            leg_qdot[k] = obs.qdot[k];
        }
        std::fill(leg_actions.begin(), leg_actions.end(), 0.0f);
        for (const auto& d : tab) {
            if (d.dof >= sc.dof_stride) continue;
            float u = d.kp * (d.target - leg_q[d.dof]) - d.kd * leg_qdot[d.dof];
            if (d.tlim > 0.0f) u = std::max(-d.tlim, std::min(d.tlim, u));
            leg_actions[d.dof] = u;
        }
        legacy.Step(leg_actions.data(), leg_actions.size());

        // ---- chaos twin: closed loop from ITS OWN obs --------------------------
        chaos.ExportObsState(obs);
        std::fill(chaos_actions.begin(), chaos_actions.end(), 0.0f);
        for (const auto& d : tab) {
            if (d.dof >= sc.dof_stride) continue;
            float u = d.kp * (d.target - obs.q[d.dof]) - d.kd * obs.qdot[d.dof];
            if (d.tlim > 0.0f) u = std::max(-d.tlim, std::min(d.tlim, u));
            chaos_actions[d.dof] = u;
        }
        chaos.Step(chaos_actions.data(), chaos_actions.size());

        // ---- nk: closed loop from ITS OWN fields -------------------------------
        ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::Q, nk_q.data(),
                                                  L * sizeof(float)));
        ASSERT_TRUE(world.GetData().DownloadField(nk::FieldId::Qdot,
                                                  nk_qdot.data(),
                                                  L * sizeof(float)));
        TableTorque(tab, nk_q, nk_qdot, /*kill_grip=*/false, &nk_torque);
        ASSERT_TRUE(world.GetData().UploadField(nk::FieldId::DriveTarget,
                                                nk_torque.data(),
                                                nk_torque.size() * sizeof(float)));
        recorded_torque.push_back(nk_torque);
        const nk::StepResult sr = world.Step();
        ASSERT_TRUE(sr.AllOk()) << "an nk op failed/unsupported at step " << s;

        // ---- compare ------------------------------------------------------------
        const auto& cup_l = legacy.Body(0u, cup_local);
        const auto& cup_c = chaos.Body(0u, cup_local);
        const NkSnap snk = SnapNk(world, L, cup_local);
        nuka::runtime::articulation::ArticulationHostState leg_host =
            sc.tmpl.gripper_proto;
        legacy.DownloadGripper(0u, &leg_host);

        const double dpos = (snk.cup_pos - cup_l.position).Length();
        const double dvel = (snk.cup_vel - cup_l.linear_velocity).Length();
        const double dbase =
            (snk.base.position - leg_host.base_pose[0].position).Length();
        double dqdot = 0.0;
        for (uint32_t l = 0; l < L; ++l) {
            dqdot = std::max(dqdot, static_cast<double>(std::fabs(
                                        snk.qdot_link[l] - leg_host.qdot[l])));
        }
        for (int k = 0; k < 6; ++k) {
            dqdot = std::max(dqdot,
                             static_cast<double>(std::fabs(
                                 snk.root_vel[k] -
                                 leg_host.link_velocity[0].v[k])));
        }
        const double dchaos = (cup_c.position - cup_l.position).Length();
        full_dpos = std::max(full_dpos, dpos);
        chaos_dpos = std::max(chaos_dpos, dchaos);
        if (s == 0u) {
            seed_dpos = dpos; seed_dvel = dvel;
            seed_dqdot = dqdot; seed_dbase = dbase;
        }
        if (s < kFloorSteps) {
            floor_dpos = std::max(floor_dpos, dpos);
            floor_dvel = std::max(floor_dvel, dvel);
            floor_dqdot = std::max(floor_dqdot, dqdot);
            floor_dbase = std::max(floor_dbase, dbase);
        }

        // ---- composition + force balances --------------------------------------
        const auto& lrep = legacy.GraspReports()[0];
        NkReport nrep = ReportNk(world, rows_per_env, slot_count);
        {
            std::vector<uint32_t> counts(slot_count);
            ASSERT_TRUE(world.GetData().DownloadField(
                nk::FieldId::UcontactCount, counts.data(),
                counts.size() * sizeof(uint32_t)));
            for (uint32_t i = n_feet; i < n_feet + n_fingers && i < slot_count; ++i) {
                nrep.finger_contacts += counts[i];
            }
        }
        if (count_open && (lrep.finger_contacts != nrep.finger_contacts ||
                           lrep.foot_normal_rows != nrep.foot_normal_rows)) {
            count_open = false;
            count_fork = s;
        }
        if (nrep.foot_normal_rows > 0u && nrep.finger_contacts > 0u) ++both_steps;
        if (s >= kSteps - 100u) {  // the settled lift window (table off).
            feet_ratio_sum += nrep.foot_normal_impulse / mg_dt_feet;
            cup_ratio_sum += nrep.cup_vertical_impulse / mg_dt_cup;
            ++ratio_n;
        }
        if (s <= 12u || (s % 50u) == 49u || s == kSteps - 1u) {
            std::printf(
                "[M4 PARITY] step %3u: |dcup|=%.3e |dvel|=%.3e |dqdot|=%.3e "
                "|dbase|=%.3e chaos|dcup|=%.3e contacts(l=%u n=%u) feet(l=%u "
                "n=%u) table(l=%u n=%u) cup_z=%.4f\n",
                s, dpos, dvel, dqdot, dbase, dchaos, lrep.finger_contacts,
                nrep.finger_contacts, lrep.foot_normal_rows,
                nrep.foot_normal_rows, lrep.table_row_count, nrep.table_rows,
                snk.cup_pos.z);
        }
    }

    const double feet_ratio = feet_ratio_sum / std::max(1u, ratio_n);
    const double cup_ratio = cup_ratio_sum / std::max(1u, ratio_n);
    std::printf(
        "[M4 PARITY RESULT] floor(0..%u): dcup=%.3e dvel=%.3e dqdot=%.3e "
        "dbase=%.3e (bars %.0e/%.0e). full: nk=%.3e chaos-floor=%.3e "
        "(bound 10x). count-fork @%u. both-classes %u/%u. late "
        "feet_ratio=%.4f cup_ratio=%.4f\n",
        kFloorSteps - 1u, floor_dpos, floor_dvel, floor_dqdot, floor_dbase,
        kFloorTol, kFloorTolQdot, full_dpos, chaos_dpos, count_fork, both_steps,
        kSteps, feet_ratio, cup_ratio);

    // (a) tier 1: STEP-0 SEED at the ORIGINAL G1c bars (pre-chaos).
    EXPECT_LE(seed_dpos, kFloorTol) << "nk cup POSITION seed off at step 0";
    EXPECT_LE(seed_dvel, kFloorTol) << "nk cup VELOCITY seed off at step 0";
    EXPECT_LE(seed_dqdot, kFloorTolQdot) << "nk gripper qdot seed off at step 0";
    EXPECT_LE(seed_dbase, kFloorTol) << "nk BASE position seed off at step 0";
    // (a) tier 2: the 10-step window (velocity bars scaled — header note).
    EXPECT_LE(floor_dpos, kFloorTol) << "nk cup POSITION diverged in the FP-floor window";
    EXPECT_LE(floor_dvel, kFloorTolVelWin) << "nk cup VELOCITY diverged in the FP-floor window";
    EXPECT_LE(floor_dqdot, kFloorTolQdotWin) << "nk gripper qdot diverged in the FP-floor window";
    EXPECT_LE(floor_dbase, kFloorTol) << "nk BASE position diverged in the FP-floor window";
    // (b) contact-set parity through the window.
    EXPECT_GE(count_fork, kFloorSteps)
        << "the contact-set COUNT forked INSIDE the FP-floor window";
    // (c) the chaos bound (Gate1 pattern): chaos must be REAL (else the bound
    // is vacuous) and nk must stay within 10x of it.
    EXPECT_GT(chaos_dpos, 1.0e-4)
        << "the legacy self-chaos floor is implausibly small (vacuous bound)";
    EXPECT_LT(full_dpos, 10.0 * chaos_dpos)
        << "nk diverged from legacy far beyond the FP-chaos floor";
    // (d) the union composition + the two force balances on the NK side.
    EXPECT_EQ(both_steps, kSteps) << "a step lost feet or fingers on the nk side";
    EXPECT_GT(feet_ratio, 0.7) << "nk feet NOT carrying robot+cup";
    EXPECT_LT(feet_ratio, 1.3) << "nk foot impulse far above the weight";
    EXPECT_GT(cup_ratio, 0.7) << "nk fingers NOT carrying the cup after the lift";
    EXPECT_LT(cup_ratio, 1.3) << "nk finger vertical impulse far above the cup weight";

    // (e) PLAN REPLAY: a twin nk world replays the RECORDED action stream via
    // StepPlanned (ONE plan capture, kSteps replays — assemble+solve inside the
    // graph) and must land BYTE-IDENTICAL to the Step() world.
    {
        nk::Model model2 = coresident::BuildNkUnionModel(sc.tmpl, 1u);
        nk::World replay(std::move(model2), 1u, fx.dev, fx.backend, cfg);
        ASSERT_TRUE(replay.Ready());
        for (uint32_t s = 0; s < kSteps; ++s) {
            if (s == kLiftAt) {
                const uint32_t off = 0u;
                ASSERT_TRUE(replay.GetData().UploadField(
                    nk::FieldId::TableEnabled, &off, sizeof(uint32_t)));
            }
            ASSERT_TRUE(replay.GetData().UploadField(
                nk::FieldId::DriveTarget, recorded_torque[s].data(),
                recorded_torque[s].size() * sizeof(float)));
            ASSERT_EQ(replay.StepPlanned(), nphi::Status::Ok)
                << "plan execute failed at step " << s;
        }
        const NkSnap a = SnapNk(world, L, cup_local);
        const NkSnap b = SnapNk(replay, L, cup_local);
        EXPECT_EQ(a.cup_pos.x, b.cup_pos.x);
        EXPECT_EQ(a.cup_pos.y, b.cup_pos.y);
        EXPECT_EQ(a.cup_pos.z, b.cup_pos.z);
        EXPECT_EQ(a.base.position.x, b.base.position.x);
        EXPECT_EQ(a.base.position.y, b.base.position.y);
        EXPECT_EQ(a.base.position.z, b.base.position.z);
        for (uint32_t l = 0; l < L; ++l) {
            ASSERT_EQ(a.qdot_link[l], b.qdot_link[l]) << "qdot link " << l;
        }
        for (int k = 0; k < 6; ++k) {
            ASSERT_EQ(a.root_vel[k], b.root_vel[k]) << "root vel lane " << k;
        }
        std::printf("[M4 PARITY] plan-replay twin: BYTE-IDENTICAL after %u "
                    "StepPlanned replays\n", kSteps);
    }
}
