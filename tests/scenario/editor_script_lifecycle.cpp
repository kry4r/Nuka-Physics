// ---------------------------------------------------------------------------
// Editor live-world script LIFECYCLE gate.
//
// Cooks a real EditorScene, stands up the embedded-Python ScriptHost bound to it,
// registers a /script node defining on_ready/on_step/on_reset, and drives the SAME
// step + reset path the viewer does -- asserting each callback fires at the right
// time, through the real nuka.set_drive seam, with the right CARDINALITY:
//   * on_ready runs EXACTLY once (a second dispatch does not re-fire it),
//   * on_step runs once per Play/Step (the drive target equals the step count each
//     step -- off-by-one sensitive) and the value reaches the LIVE world,
//   * on_reset runs right after ResetEditorScenePhysics (it, not the reset, sets the
//     sentinel),
//   * a throwing on_step is caught (the editor keeps stepping) and DISABLED after a
//     bounded retry -- not on the first failure -- while a sibling script keeps running.
//
// HOST-ONLY: cooks on the phi device (SKIP when none); embeds CPython (SKIP when the
// interpreter is unavailable). One ScriptHost per process (one Py_Initialize).
// ---------------------------------------------------------------------------

#include "nk/model/generated/field_ids.hpp"
#include "phi/backend.hpp"
#include "runtime/app/simulation.hpp"
#include "runtime/app/viewer/editor_edits.hpp"
#include "runtime/app/viewer/editor_scene.hpp"
#include "runtime/app/viewer/imgui_layer.hpp"
#include "runtime/app/viewer/script_host.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace viewer = nuka::runtime::app::viewer;
using nuka::nk::FieldId;
using Callback = viewer::ScriptHost::Callback;

namespace {

constexpr float kDt = 1.0f / 240.0f;

// Defines all three lifecycle hooks; each writes a DISTINCT drive DOF through the
// real seam so the host can observe exactly which ran and how many times.
//   ready -> dof 0 = 100 + (#ready),  step -> dof 1 = (#step),  reset -> dof 2 = 500 + (#reset)
const char* kGoodScript = R"PY(
import nuka
state = {'ready': 0, 'step': 0, 'reset': 0}
def on_ready():
    state['ready'] += 1
    nuka.set_drive(0, 100.0 + state['ready'])
def on_step():
    state['step'] += 1
    nuka.set_drive(1, float(state['step']))
def on_reset():
    state['reset'] += 1
    nuka.set_drive(2, 500.0 + state['reset'])
)PY";

// A script whose on_step always raises -- the isolation / disable subject.
const char* kThrowingStep = R"PY(
def on_step():
    raise ValueError('boom')
)PY";

// Mirror the viewer's end-of-frame drive drain: upload each dirty DOF (env 0) into
// the live world's DriveTarget field, then clear the latch.
void DrainDrive(viewer::ViewerUiState& ui, nuka::nk::World& world) {
    for (uint32_t d = 0; d < ui.drive_targets.size(); ++d) {
        if (!ui.drive_dirty[d]) continue;
        world.GetData().UploadField(FieldId::DriveTarget, &ui.drive_targets[d],
                                    sizeof(float),
                                    static_cast<uint64_t>(d) * sizeof(float));
        ui.drive_dirty[d] = 0u;
    }
}

}  // namespace

// One comprehensive scenario (a single embedded interpreter for the process).
TEST(EditorScriptLifecycle, CallbacksFireOnceStepResetIsolatedAndDisabled) {
    nuka::phi::Device* dev = nuka::phi::InitBestDevice();
    if (!dev) GTEST_SKIP() << "no phi device";
    nuka::phi::Backend* backend = nuka::phi::DeviceInitBackend(dev, nullptr);
    if (!backend) GTEST_SKIP() << "no phi backend";

    const std::string scene = "examples/scenes/go2.nks";
    if (!fs::exists(scene)) GTEST_SKIP() << "go2.nks missing";
    auto es = viewer::LoadEditorScene(scene, dev, backend, kDt);
    ASSERT_NE(es, nullptr) << "cook failed: " << scene;
    nuka::nk::World& world = *es->world;
    nuka::runtime::app::Simulation& sim = *es->sim;
    const uint32_t L = es->caps.links_per_env;
    ASSERT_GE(L, 3u) << "scene needs >=3 DOFs for the three callback probes";

    viewer::EntityRecordIndex index;
    index.Build(es->scene);
    viewer::ViewerUiState ui;
    ui.drive_targets.assign(L, 0.0f);
    ui.drive_dirty.assign(L, 0u);

    viewer::ScriptHost host;
    if (!host.Valid()) GTEST_SKIP() << "embedded python unavailable";
    host.SetContext({es.get(), &index, &ui});

    const uint64_t kGood = 1;
    const std::string reg_out = host.RegisterScript(kGood, kGoodScript);
    ASSERT_EQ(host.ScriptCount(), 1u) << "register: " << reg_out;

    // -- on_ready fires EXACTLY once -----------------------------------------
    host.DispatchReady();
    EXPECT_FLOAT_EQ(ui.drive_targets[0], 101.0f) << "on_ready did not run on load";
    host.DispatchReady();  // a second dispatch must NOT re-fire it
    EXPECT_FLOAT_EQ(ui.drive_targets[0], 101.0f) << "on_ready re-fired (not once)";

    // -- on_step fires once per Play/Step, exact + reaching the live world ----
    constexpr int kSteps = 30;
    for (int i = 1; i <= kSteps; ++i) {
        sim.FramePublish(nullptr, /*do_step=*/true);  // consume the prior step's drive
        host.DispatchStep();                          // on_step sets dof 1 = i
        EXPECT_FLOAT_EQ(ui.drive_targets[1], static_cast<float>(i))
            << "on_step did not run exactly once on step " << i;
        DrainDrive(ui, world);                        // stage for the next FramePublish
    }
    std::vector<float> dt(L, 0.0f);
    world.GetData().DownloadField(FieldId::DriveTarget, dt.data(),
                                  static_cast<uint64_t>(L) * sizeof(float), 0);
    EXPECT_NEAR(dt[1], static_cast<float>(kSteps), 1e-4f)
        << "on_step's set_drive never reached the live world (the body was not driven)";

    // -- on_reset fires right after the reset entry --------------------------
    ASSERT_TRUE(viewer::ResetEditorScenePhysics(*es)) << "reset failed";
    const float reset_before = ui.drive_targets[2];
    host.DispatchReset();
    EXPECT_FLOAT_EQ(ui.drive_targets[2], 501.0f) << "on_reset did not run after reset";
    EXPECT_NE(reset_before, 501.0f) << "DispatchReset is what set the reset sentinel";

    // -- a throwing on_step is caught, bounded-disabled, and isolated --------
    const uint64_t kBad = 99;
    host.RegisterScript(kBad, kThrowingStep);
    ASSERT_EQ(host.ScriptCount(), 2u);
    const float good_before = ui.drive_targets[1];

    host.DispatchStep();  // first failure
    EXPECT_FALSE(host.CallbackDisabled(kBad, Callback::Step))
        << "a throwing callback was disabled on its FIRST failure (not a bounded retry)";
    EXPECT_GT(ui.drive_targets[1], good_before)
        << "the throwing script's failure stopped the healthy script (no isolation)";

    for (int k = 0; k < 8; ++k) host.DispatchStep();  // exceed the retry bound
    EXPECT_TRUE(host.CallbackDisabled(kBad, Callback::Step))
        << "a repeatedly-throwing callback was never disabled (would spam every frame)";

    // the editor still steps, and the healthy script still advances.
    const float good_mid = ui.drive_targets[1];
    host.DispatchStep();
    sim.FramePublish(nullptr, /*do_step=*/true);  // must not crash with a disabled callback
    EXPECT_GT(ui.drive_targets[1], good_mid)
        << "a disabled bad callback froze the editor's dispatch of the healthy one";

    host.SetContext({nullptr, &index, &ui});  // drop the world before it tears down
}
