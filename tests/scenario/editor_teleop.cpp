// ---------------------------------------------------------------------------
// Editor keyboard-teleop gate.
//
// Teleop is a THIRD drive producer beside the Drive sliders and the --policy
// controller. Two layers are covered:
//
//   (A) The PURE mapping (ApplyTeleop, no device): synthetic key intents nudge the
//       selected DOF's drive target by EXACTLY the step + latch drive_dirty (the seam
//       the viewer uploads), select/clamp the active DOF, and turn WASD/QE into a
//       [vx,vy,wyaw] command. The GATE is a first-class parameter: a disabled tick
//       (toggle off OR ImGui owns the keyboard) changes NOTHING -- the sensitivity
//       guard, since a leaked write would move the robot while typing.
//
//   (B) The LIVE world (cooks go2.nks; SKIP when no phi device): a teleop nudge
//       written through the SAME UploadField seam REACHES nk::Data (the DriveTarget
//       field changes by the step) and, with the general PD-hold gains applied, the
//       joint actually SWINGS toward the nudged target under the solver. A command
//       reaches a command-consuming controller through the general base interface.
//
// HOST-ONLY: the pure tests need no device; the live tests cook on the phi device.
// ---------------------------------------------------------------------------

#include "nk/model/generated/field_ids.hpp"
#include "phi/backend.hpp"
#include "runtime/app/scene_controller.hpp"
#include "runtime/app/simulation.hpp"
#include "runtime/app/viewer/drive_hold.hpp"
#include "runtime/app/viewer/editor_scene.hpp"
#include "runtime/app/viewer/teleop.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;
namespace viewer = nuka::runtime::app::viewer;
using nuka::nk::FieldId;

namespace {

// Size the per-DOF drive state to `n` zeroed targets (what SeedDriveTargets produces).
viewer::ViewerUiState MakeUi(uint32_t n) {
    viewer::ViewerUiState ui;
    ui.drive_targets.assign(n, 0.0f);
    ui.drive_dirty.assign(n, 0u);
    return ui;
}

uint8_t AnyDirty(const std::vector<uint8_t>& d) {
    uint8_t any = 0u;
    for (uint8_t x : d) any |= x;
    return any;
}

}  // namespace

// ===========================================================================
// (A) Pure mapping -- no device.
// ===========================================================================

// A '[' / ']' (or , / .) nudge moves the SELECTED DOF's target by EXACTLY the step and
// latches ONLY that DOF dirty -- the seam the viewer's per-DOF UploadField loop sends.
TEST(TeleopMapping, PerDofNudgeWritesTheDriveSeam) {
    viewer::ViewerUiState ui = MakeUi(4);
    ui.teleop_dof = 2;
    ui.teleop_step = 0.05f;

    viewer::TeleopKeys plus; plus.dof_plus = true;
    viewer::ApplyTeleop(true, plus, ui, 4);
    EXPECT_FLOAT_EQ(ui.drive_targets[2], 0.05f);
    EXPECT_EQ(ui.drive_dirty[2], 1u);
    // Only the selected DOF moved / latched.
    EXPECT_FLOAT_EQ(ui.drive_targets[0], 0.0f);
    EXPECT_FLOAT_EQ(ui.drive_targets[3], 0.0f);
    EXPECT_EQ(ui.drive_dirty[0], 0u);

    viewer::TeleopKeys minus; minus.dof_minus = true;
    viewer::ApplyTeleop(true, minus, ui, 4);
    EXPECT_FLOAT_EQ(ui.drive_targets[2], 0.0f);  // back down by one step
}

// The GATE: a disabled tick (toggle off OR ImGui owns the keyboard) must change NOTHING.
TEST(TeleopMapping, DisabledTickIsInert) {
    viewer::ViewerUiState ui = MakeUi(4);
    ui.teleop_dof = 1;
    ui.teleop_step = 0.10f;

    viewer::TeleopKeys keys; keys.dof_plus = true; keys.fwd = true;
    const viewer::TeleopResult r = viewer::ApplyTeleop(/*enabled=*/false, keys, ui, 4);
    EXPECT_FALSE(r.command_active);
    EXPECT_FLOAT_EQ(r.command[0], 0.0f);
    EXPECT_FLOAT_EQ(ui.drive_targets[1], 0.0f);  // no leaked nudge
    EXPECT_EQ(AnyDirty(ui.drive_dirty), 0u);     // nothing queued for upload
}

// DOF selection steps with - / = and CLAMPS to [0, dof_count).
TEST(TeleopMapping, DofSelectClamps) {
    viewer::ViewerUiState ui = MakeUi(3);
    ui.teleop_dof = 0;
    viewer::TeleopKeys prev; prev.dof_prev = true;
    viewer::ApplyTeleop(true, prev, ui, 3);
    EXPECT_EQ(ui.teleop_dof, 0);  // clamped at the bottom

    ui.teleop_dof = 2;
    viewer::TeleopKeys next; next.dof_next = true;
    viewer::ApplyTeleop(true, next, ui, 3);
    EXPECT_EQ(ui.teleop_dof, 2);  // clamped at the top

    ui.teleop_dof = 0;
    viewer::ApplyTeleop(true, next, ui, 3);
    EXPECT_EQ(ui.teleop_dof, 1);  // steps up by one
}

// WASD plane + QE yaw -> a [vx,vy,wyaw] command scaled by the editable magnitudes;
// releasing all keys yields the zero command (so the robot stops, not coasts).
TEST(TeleopMapping, LocomotionCommand) {
    viewer::ViewerUiState ui = MakeUi(0);
    ui.teleop_lin = 1.5f;
    ui.teleop_yaw = 0.8f;

    viewer::TeleopKeys fwd_yaw; fwd_yaw.fwd = true; fwd_yaw.yaw_l = true;
    viewer::TeleopResult r = viewer::ApplyTeleop(true, fwd_yaw, ui, 0);
    EXPECT_TRUE(r.command_active);
    EXPECT_FLOAT_EQ(r.command[0], 1.5f);
    EXPECT_FLOAT_EQ(r.command[1], 0.0f);
    EXPECT_FLOAT_EQ(r.command[2], 0.8f);

    viewer::TeleopKeys back_strafe; back_strafe.back = true; back_strafe.strafe_r = true;
    r = viewer::ApplyTeleop(true, back_strafe, ui, 0);
    EXPECT_FLOAT_EQ(r.command[0], -1.5f);
    EXPECT_FLOAT_EQ(r.command[1], -1.5f);

    viewer::TeleopKeys none;
    r = viewer::ApplyTeleop(true, none, ui, 0);
    EXPECT_FALSE(r.command_active);
    EXPECT_FLOAT_EQ(r.command[0], 0.0f);
    EXPECT_FLOAT_EQ(ui.teleop_cmd[2], 0.0f);
}

// The command reaches a command-consuming SceneController through the GENERAL base
// interface (the exact path the viewer uses; Go2PolicyController overrides it too).
namespace {
class RecordingController : public nuka::runtime::app::SceneController {
public:
    void OnStep(nuka::nk::World&, uint32_t) override {}
    bool AcceptsCommand() const override { return true; }
    void SetCommandVector(const float* c, uint32_t n) override {
        ++calls;
        for (uint32_t i = 0; i < 3u; ++i) cmd[i] = (i < n) ? c[i] : 0.0f;
    }
    float cmd[3] = {0.0f, 0.0f, 0.0f};
    int   calls  = 0;
};
}  // namespace

TEST(TeleopMapping, CommandReachesController) {
    viewer::ViewerUiState ui = MakeUi(0);
    ui.teleop_lin = 2.0f;
    RecordingController ctrl;

    viewer::TeleopKeys fwd; fwd.fwd = true;
    const viewer::TeleopResult r = viewer::ApplyTeleop(true, fwd, ui, 0);
    // The viewer feeds a command-consuming controller every frame (zero on release).
    nuka::runtime::app::SceneController* base = &ctrl;
    ASSERT_TRUE(base->AcceptsCommand());
    base->SetCommandVector(r.command, 3);
    EXPECT_EQ(ctrl.calls, 1);
    EXPECT_FLOAT_EQ(ctrl.cmd[0], 2.0f);
    EXPECT_FLOAT_EQ(ctrl.cmd[1], 0.0f);
}

// The default SceneController ignores commands (no command sink) -- the general base
// safely no-ops so a teleop producer never assumes a consumer.
TEST(TeleopMapping, DefaultControllerHasNoCommandSink) {
    class Plain : public nuka::runtime::app::SceneController {
        void OnStep(nuka::nk::World&, uint32_t) override {}
    } plain;
    EXPECT_FALSE(plain.AcceptsCommand());
    const float c[3] = {1.0f, 2.0f, 3.0f};
    plain.SetCommandVector(c, 3);  // must not crash / do anything
}

// ===========================================================================
// (B) Live world -- cooks go2.nks on the phi device.
// ===========================================================================

class TeleopLive : public ::testing::Test {
protected:
    void SetUp() override {
        dev_ = nuka::phi::InitBestDevice();
        if (!dev_) GTEST_SKIP() << "no phi device";
        backend_ = nuka::phi::DeviceInitBackend(dev_, nullptr);
        if (!backend_) GTEST_SKIP() << "no phi backend";
        if (!fs::exists(kScene)) GTEST_SKIP() << "go2.nks missing";
    }
    static std::vector<float> DlF(nuka::nk::World& w, FieldId id, uint32_t n) {
        std::vector<float> out(n, 0.0f);
        if (n > 0u)
            w.GetData().DownloadField(id, out.data(),
                                      static_cast<uint64_t>(n) * sizeof(float), 0);
        return out;
    }
    const char* kScene = "examples/scenes/go2.nks";
    nuka::phi::Device*  dev_ = nullptr;
    nuka::phi::Backend* backend_ = nullptr;
};

// A teleop nudge reaches nk::Data through the EXACT viewer upload, and the PD-held
// joint SWINGS toward the nudged target under the solver. Sensitive: the live drive
// field must change by the step, and the joint must move in the nudge direction.
TEST_F(TeleopLive, NudgeReachesWorldAndSwingsJoint) {
    auto es = viewer::LoadEditorScene(kScene, dev_, backend_, 1.0f / 240.0f);
    ASSERT_NE(es, nullptr);
    nuka::nk::World& w = *es->world;
    const uint32_t L = es->caps.links_per_env;
    ASSERT_GT(L, 2u) << "go2 should have a floating base + actuated joints";

    // Seed the per-DOF drive editor from the live field (SeedDriveTargets' contract).
    viewer::ViewerUiState ui;
    ui.drive_targets = DlF(w, FieldId::DriveTarget, L);
    ui.drive_dirty.assign(L, 0u);

    // General PD-hold so drive targets are effective (motor-actuated robots cook zero
    // gains; the hold uploads uniform gains to the actuated joints, skipping the base).
    viewer::ApplyHoldGains(w, es->caps.env_count, 20.0f, 0.5f);

    // Nudge a thigh (a non-base joint with range to swing).
    const int dof = 2;
    ui.teleop_dof = dof;
    ui.teleop_step = 0.4f;
    const float target_before = ui.drive_targets[dof];

    viewer::TeleopKeys plus; plus.dof_plus = true;
    viewer::ApplyTeleop(true, plus, ui, L);
    ASSERT_EQ(ui.drive_dirty[dof], 1u);
    EXPECT_FLOAT_EQ(ui.drive_targets[dof], target_before + 0.4f);

    // The viewer's per-DOF UploadField (env 0): write ONLY the dirtied DOF.
    const std::vector<float> q_before = DlF(w, FieldId::Q, L);
    w.GetData().UploadField(FieldId::DriveTarget, &ui.drive_targets[dof], sizeof(float),
                            static_cast<uint64_t>(dof) * sizeof(float));

    // (1) The write REACHED the live world.
    const std::vector<float> drv_live = DlF(w, FieldId::DriveTarget, L);
    EXPECT_NEAR(drv_live[dof], target_before + 0.4f, 1e-5f)
        << "teleop nudge did not reach nk::Data";

    // (2) The PD-held joint SWINGS toward the new target under stepping.
    for (int i = 0; i < 150; ++i) w.Step();
    const std::vector<float> q_after = DlF(w, FieldId::Q, L);
    const float moved = q_after[dof] - q_before[dof];
    const float to_target_before = std::fabs(q_before[dof] - drv_live[dof]);
    const float to_target_after  = std::fabs(q_after[dof]  - drv_live[dof]);
    std::printf("[teleop] dof %d  q %.4f -> %.4f  target %.4f  (moved %.4f)\n", dof,
                q_before[dof], q_after[dof], drv_live[dof], moved);
    EXPECT_GT(moved, 0.05f) << "joint did not swing in the nudge (+) direction";
    EXPECT_LT(to_target_after, to_target_before) << "joint did not move toward the target";
}

// The GATE on the live world: a disabled tick queues NOTHING, so a re-download of the
// drive field is unchanged -- typing in a field can never perturb the robot.
TEST_F(TeleopLive, GatedNudgeNeverReachesWorld) {
    auto es = viewer::LoadEditorScene(kScene, dev_, backend_, 1.0f / 240.0f);
    ASSERT_NE(es, nullptr);
    nuka::nk::World& w = *es->world;
    const uint32_t L = es->caps.links_per_env;

    viewer::ViewerUiState ui;
    ui.drive_targets = DlF(w, FieldId::DriveTarget, L);
    ui.drive_dirty.assign(L, 0u);
    const std::vector<float> drv0 = ui.drive_targets;

    viewer::TeleopKeys plus; plus.dof_plus = true;
    viewer::ApplyTeleop(/*enabled=*/false, plus, ui, L);  // gated
    EXPECT_EQ(AnyDirty(ui.drive_dirty), 0u);

    // Nothing dirty -> the viewer uploads nothing -> the live field is untouched.
    const std::vector<float> drv1 = DlF(w, FieldId::DriveTarget, L);
    for (uint32_t i = 0; i < L; ++i)
        EXPECT_FLOAT_EQ(drv1[i], drv0[i]) << "gated teleop perturbed DOF " << i;
}
