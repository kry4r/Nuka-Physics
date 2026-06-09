# H1 Cup Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete H1 cup demo pipeline that validates a continuous standing+cup manipulation rollout and renders a polished PBR-style MP4.

**Architecture:** Keep RL scoped to standing: the exported H1 standing actor drives the legs through the existing C++ bridge. A deterministic full-world controller drives upper body/hand phases for reach, close, lift, place, and release. The validated rollout is captured once, then rendered by a real geometry/PBR-style renderer; the Go2 stick renderer is not acceptable as final output.

**Tech Stack:** C++ co-resident stepper/tests, `TinyMlp<32,64,10>` bridge artifacts, CUDA RT/two-level renderer (`src/rt`), Python orchestration/encoding, `ffmpeg`, existing H1/cup assets.

---

## File Structure

### New/Modified Core Demo Files

- Create `tests/coresident/h1_demo_shared.hpp`
  - Shared H1 helper code factored from `test_h1_bridge_spike.cpp` and `test_h1_dense_grasp.cpp`.
  - Responsibilities: H1 asset constants, link lookup, FK download, leg bridge observation, standing policy torque mapping, foot sphere setup, dense cup/grasp placement helpers needed by demo tests.

- Create `tests/coresident/test_h1_cup_sequence_demo.cpp`
  - Hard gate for one continuous demo sequence.
  - Uses RL standing bridge for legs and deterministic upper-body/hand phases.
  - Writes optional capture/metrics when `NUKA_H1_DEMO_CAPTURE_DIR` is set.

- Modify `tests/CMakeLists.txt`
  - Add `nuka_h1_cup_sequence_demo_test` target.
  - Do not include or depend on owner-excluded `tests/scene/test_scene_compose_h1_cup_table.cpp`.

- Create `examples/demo/h1_cup_capture_sequence.py`
  - Wrapper that builds/runs `nuka_h1_cup_sequence_demo_test` with capture enabled and fails on skip.

- Create `examples/demo/h1_cup_pbr_render.py`
  - Converts captured poses/phase metadata into PBR-style frames.
  - Uses a geometry renderer, not stick figures. Preferred path: C++/CUDA RT helper executable; fallback inside the plan is a polished analytic proxy over link poses with PBR shading and shadows.

- Create `src/apps/h1_cup_rt_render.cpp` and optionally `src/apps/h1_cup_rt_render.hpp`
  - CLI renderer that loads capture manifest/pose data and writes PPM frames using `rt::TwoLevelScene` / `rt::RenderFrame` public APIs.
  - Builds reusable robot/cup/table/floor meshes or analytic primitives from captured poses.
  - Do not modify existing `src/rt/**` internals without owner approval.

- Modify `src/CMakeLists.txt` to add `nuka_h1_cup_rt_render` if the C++ RT renderer path is used.
- Modify `tests/CMakeLists.txt` only for new test target registration, using partial staging so the pre-existing owner hunk is not committed.

- Create `examples/demo/h1_cup_render_demo.py`
  - Top-level command: build, validate/capture, render frames, encode MP4, write final manifest.

- Create `docs/examples/h1_cup_demo.md`
  - User-facing run instructions and honesty statement: standing is RL, manipulation is deterministic state machine unless later changed.

### Artifacts

- Output directory: `out/h1_cup_demo/`
  - `rollout.jsonl` or `rollout.npz`
  - `metrics.json`
  - `frames/frame_00000.ppm` ...
  - `h1_cup_demo.mp4`
  - `manifest.json`

---

## Task 1: PBR Renderer Feasibility Gate

**Files:**
- Test: `tests/rt/test_h1_cup_pbr_render.cpp`
- Create: `src/apps/h1_cup_rt_render.cpp` only if needed for CLI smoke; otherwise keep this task test-only.
- Modify: `tests/CMakeLists.txt` for test target only.
- Read: `src/rt/two_level_render.hpp`, `src/rt/material.hpp`, `tests/rt/test_two_level_render.cpp`

- [ ] **Step 1: Write a failing renderer smoke test**

Create `tests/rt/test_h1_cup_pbr_render.cpp` with a minimal scene: floor plane/box, table, cup proxy sphere/cylinder mesh, and a humanoid proxy made of capsules/spheres/boxes. Test that `rt::RenderFrame` produces non-background pixels, shadows/contact shadows, distinct material colors, and stable camera framing. This test is a render-quality smoke, not a stick/debug renderer test.

- [ ] **Step 2: Run test to verify it fails to compile**

Run:

```bash
export PATH=/root/.nuka-toolchain-gcc14/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/opt/cuda-12.8-root/lib64:${LD_LIBRARY_PATH:-}
cmake --build build-cuda128 --target nuka_h1_cup_pbr_render_test -j$(nproc)
```

Expected: FAIL because target/file does not exist.

- [ ] **Step 3: Implement minimal CMake target and RT scene builder**

Add target for `tests/rt/test_h1_cup_pbr_render.cpp`, linking the same RT libraries used by `nuka_rt_two_level_test`.

Implement helper functions in the test first:

- `BuildUvSphereMesh(...)`
- `BuildBoxMesh(...)`
- `BuildCapsuleProxyMesh(...)` if feasible, else capsule as cylinder+spheres.
- `BuildHeroCamera(...)` low 3/4 view.

Use `rt::Material` with varied albedo/roughness/metallic.

- [ ] **Step 4: Run smoke test to verify real shaded geometry**

Run:

```bash
cmake --build build-cuda128 --target nuka_h1_cup_pbr_render_test -j$(nproc)
build-cuda128/tests/nuka_h1_cup_pbr_render_test --gtest_filter=H1CupPbrRender.SmokeHeroScene
```

Expected: PASS, nonzero hit pixels, non-black color variance, shadowed pixels present, robot/cup/table visible in expected image regions.

- [ ] **Step 5: Commit renderer feasibility**

```bash
git add tests/rt/test_h1_cup_pbr_render.cpp
git add -p tests/CMakeLists.txt  # stage only the new render test target, not owner-excluded hunks
git commit -m "Add H1 cup PBR render smoke [skip ci]"
```

---

## Task 2: Shared H1 Demo Helpers

**Files:**
- Create: `tests/coresident/h1_demo_shared.hpp`
- Modify: `tests/coresident/test_h1_bridge_spike.cpp`
- Modify: `tests/coresident/test_h1_dense_grasp.cpp` only if helpers can be extracted without destabilizing; otherwise duplicate minimally in new sequence test.
- Test: existing `nuka_h1_bridge_spike_test`, `nuka_h1_dense_grasp_test`

- [ ] **Step 1: Add header with constants and link helpers**

Move or duplicate carefully:

- H1 asset paths.
- `kLegLinkNames`.
- `LegLimit(slot)`.
- `ForwardKinematics`.
- `TiltDeg`.
- `ResolveLegLinks` / `LinkByName` style helpers.

- [ ] **Step 2: Add standing bridge helper**

Expose:

```cpp
std::vector<float> BuildH1BridgeObs(...);
std::vector<float> PolicyTorquesFromBridge(...);
```

Contract:

- 32 obs dims, same field order as current Gate2.
- 10 normalized actions, clamp `[-1,1]`, multiply by per-joint limits.

- [ ] **Step 3: Add dense grasp helper seam**

Expose only what sequence needs:

- cup hull load/scale
- dense fingertip setup
- hand curl target names
- PD target creation

Do not rewrite dense grasp physics.

- [ ] **Step 4: Rebuild existing gates**

Run:

```bash
cmake --build build-cuda128 --target nuka_h1_bridge_spike_test nuka_h1_dense_grasp_test -j$(nproc)
build-cuda128/tests/nuka_h1_bridge_spike_test --gtest_filter=H1BridgeSpike.Gate1BridgeParity:H1BridgeSpike.Gate2ClosedLoopStandSoft
build-cuda128/tests/nuka_h1_dense_grasp_test --gtest_filter=H1DenseGraspLargeCup.ForceClosureLiftWithDisturbance:H1DenseGraspLargeCup.FingerOnlyFallbackBiteGripOffVsOn
```

Expected: PASS, no behavior changes.

- [ ] **Step 5: Commit shared helpers**

```bash
git add tests/coresident/h1_demo_shared.hpp tests/coresident/test_h1_bridge_spike.cpp tests/coresident/test_h1_dense_grasp.cpp
git commit -m "Factor H1 demo helper contracts [skip ci]"
```

---

## Task 3: Continuous Sequence Gate Skeleton

**Files:**
- Create: `tests/coresident/test_h1_cup_sequence_demo.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write failing sequence target test**

Test name:

```cpp
TEST(H1CupSequenceDemo, StandsReachLiftPlaceRelease)
```

Initial expectations:

- skips if H1/cup/bridge artifacts missing.
- loads standing MLP.
- constructs full H1 stand scene plus cup/table/grasp config.
- runs phases and accumulates metrics.
- asserts metrics from spec.

- [ ] **Step 2: Build to confirm missing target/test failure**

Run:

```bash
cmake --build build-cuda128 --target nuka_h1_cup_sequence_demo_test -j$(nproc)
```

Expected: FAIL target missing.

- [ ] **Step 3: Add CMake target**

Add `nuka_h1_cup_sequence_demo_test` linking the same libraries as `nuka_h1_bridge_spike_test` and `nuka_h1_dense_grasp_test`, including `src/runtime/coresident/unified_coresident_stepper.cpp` if needed.

- [ ] **Step 4: Implement establish-only sequence**

First green target only runs standing policy for 300-500 control steps with cup/table present but no arm motion.

Expected metrics:

- `tilt_max < 15`
- `min_contacts == 4`
- `within_torque == true`
- no NaN

- [ ] **Step 5: Run establish-only gate**

```bash
build-cuda128/tests/nuka_h1_cup_sequence_demo_test --gtest_filter=H1CupSequenceDemo.StandsReachLiftPlaceRelease
```

Expected: PASS for establish, with later phases temporarily TODO/disabled only inside the test implementation until next task.

- [ ] **Step 6: Commit skeleton**

```bash
git add tests/coresident/test_h1_cup_sequence_demo.cpp tests/CMakeLists.txt
git commit -m "Add H1 cup sequence gate skeleton [skip ci]"
```

---

## Task 4: Full-World Coexistence Feasibility Gate

**Files:**
- Modify: `tests/coresident/test_h1_cup_sequence_demo.cpp`
- Possibly Modify: `tests/coresident/h1_demo_shared.hpp`

- [ ] **Step 1: Add an explicit coexistence test phase**

Before reach/lift/place, prove the mechanics can coexist in one world without claiming the final sequence. The gate runs:

- floating full H1
- RL leg policy active
- non-leg PD/hold active
- dense fingertip geometry registered
- cup/table contact enabled
- no arm motion yet except a small deterministic hand-close probe away from the cup if needed

- [ ] **Step 2: Assert coexistence metrics**

Required assertions:

```cpp
EXPECT_LT(metrics.tilt_max_deg, 15.0);
EXPECT_EQ(metrics.min_foot_contacts, 4u);
EXPECT_TRUE(metrics.within_torque);
EXPECT_FALSE(metrics.went_nan);
EXPECT_TRUE(metrics.cup_table_support_seen);
EXPECT_TRUE(metrics.hand_drive_command_reaches_solver);
```

This catches integration blockers before the full state machine is tuned.

- [ ] **Step 3: Run coexistence gate**

```bash
build-cuda128/tests/nuka_h1_cup_sequence_demo_test --gtest_filter=H1CupSequenceDemo.CoexistenceFeasibility
```

Expected: PASS. If it fails, debug coexistence plumbing before any reach/place tuning.

- [ ] **Step 4: Commit coexistence gate**

```bash
git add tests/coresident/test_h1_cup_sequence_demo.cpp tests/coresident/h1_demo_shared.hpp
git commit -m "Add H1 cup coexistence gate [skip ci]"
```

---

## Task 5: Reach/Close/Lift/Place/Release State Machine

**Files:**
- Modify: `tests/coresident/test_h1_cup_sequence_demo.cpp`
- Possibly Modify: `tests/coresident/h1_demo_shared.hpp`

- [ ] **Step 1: Add phase enum and metrics struct**

```cpp
enum class DemoPhase { Establish, Reach, Pregrasp, Close, Lift, Place, Release, HeroHold };
struct DemoMetrics { ... };
```

- [ ] **Step 2: Implement deterministic joint target interpolation**

Use named upper-body/hand links from H1. Keep leg torque slots exclusively controlled by RL. Non-leg joints get PD targets and torque clamps.

- [ ] **Step 3: Implement close/lift using dense grasp validated targets**

Use dense grasp hand curl and cup placement from `test_h1_dense_grasp.cpp` helpers. During lift, table support must go to zero or the cup must separate from table support, depending on available stepper mode.

- [ ] **Step 4: Implement place/release**

Lower cup to table-supported state. Open hand or reduce grip torque. Run post-release settle window.

- [ ] **Step 5: Add hard assertions**

Required assertions:

```cpp
EXPECT_LT(metrics.tilt_max_deg, 15.0);
EXPECT_EQ(metrics.min_foot_contacts, 4u);
EXPECT_TRUE(metrics.within_torque);
EXPECT_GE(metrics.lift_height_delta, 0.05);
EXPECT_TRUE(metrics.lift_table_support_absent);
EXPECT_LE(metrics.place_final_height_error, 0.01);
EXPECT_LE(metrics.place_final_xy_error, 0.03);
EXPECT_GE(metrics.post_release_table_contact_ratio, 0.90);
EXPECT_TRUE(metrics.release_grip_command_off);
EXPECT_LE(metrics.post_release_hand_vertical_impulse, metrics.near_zero_impulse);
EXPECT_LE(metrics.post_release_max_disp, 0.03);
EXPECT_LE(metrics.post_release_max_tilt, 0.20);
EXPECT_LE(metrics.post_release_max_speed, 0.10);
EXPECT_FALSE(metrics.went_nan);
```

- [ ] **Step 6: Run and iterate only on deterministic controller parameters**

Run:

```bash
build-cuda128/tests/nuka_h1_cup_sequence_demo_test --gtest_filter=H1CupSequenceDemo.StandsReachLiftPlaceRelease
```

Expected: PASS. If it fails, do not weaken thresholds silently; inspect phase metrics and adjust deterministic keyframes/durations/place height.

- [ ] **Step 7: Commit full sequence gate**

```bash
git add tests/coresident/test_h1_cup_sequence_demo.cpp tests/coresident/h1_demo_shared.hpp
git commit -m "Add continuous H1 cup sequence gate [skip ci]"
```

---

## Task 6: Capture Artifact Writer

**Files:**
- Modify: `tests/coresident/test_h1_cup_sequence_demo.cpp`
- Create: `examples/demo/h1_cup_capture_sequence.py`

- [ ] **Step 1: Add capture writer controlled by env var**

When `NUKA_H1_DEMO_CAPTURE_DIR=/path`, write:

- `rollout.jsonl` with per-render-frame data.
- `metrics.json`.

Each frame record includes:

- frame index, time, phase
- H1 link poses
- cup pose
- table transform
- camera target hints

- [ ] **Step 2: Add Python wrapper**

`examples/demo/h1_cup_capture_sequence.py`:

- sets toolchain env.
- builds target unless `--skip-build`.
- runs gtest with capture env.
- fails if gtest skips.

- [ ] **Step 3: Run capture**

```bash
/root/miniconda3/envs/nuka-v03/bin/python examples/demo/h1_cup_capture_sequence.py --out out/h1_cup_demo --skip-build
```

Expected: PASS, writes `rollout.jsonl` and `metrics.json`.

- [ ] **Step 4: Commit capture wrapper**

```bash
git add tests/coresident/test_h1_cup_sequence_demo.cpp examples/demo/h1_cup_capture_sequence.py
git commit -m "Capture H1 cup sequence rollout [skip ci]"
```

---

## Task 7: PBR Frame Renderer CLI

**Files:**
- Create: `src/apps/h1_cup_rt_render.cpp`
- Modify: `src/CMakeLists.txt` for the renderer executable target.
- Modify: `tests/CMakeLists.txt` only for renderer tests, using partial staging.
- Create: `examples/demo/h1_cup_pbr_render.py` if Python wrapper is still useful.
- Do not modify existing `src/rt/**` internals without owner approval.

- [ ] **Step 1: Define renderer input contract**

Accept:

```bash
build-cuda128/src/apps/nuka_h1_cup_rt_render \
  --rollout out/h1_cup_demo/rollout.jsonl \
  --frames out/h1_cup_demo/frames \
  --width 1920 --height 1080 --fps 30
```

- [ ] **Step 2: Build TwoLevelScene assets**

Use RT meshes:

- H1 analytic proxy links: capsules/spheres/boxes from link poses; material groups for torso/legs/arms/hand.
- Cup: cylinder/sphere/convex proxy with ceramic material, or imported mesh if feasible.
- Table/floor: boxes/planes with matte PBR material.

This is not stick rendering: use shaded 3D geometry, shadows, material roughness/metallic, and camera perspective. Add tone mapping and supersampling/anti-aliasing if single-sample frames look jagged. Use multiple/faked studio lights if the bounded RT API only supports one direct light; document the chosen lighting trick in the manifest.

- [ ] **Step 3: Implement camera path**

NVIDIA keynote-inspired camera:

- Establish: low 3/4 full-body shot.
- Grasp/lift/place: dolly to hand/cup close-up.
- Hero hold: slight orbit back to full-body composition.

Use `rt::BuildPinhole` with FOV around 35-45 degrees and smooth interpolation.

- [ ] **Step 4: Write PPM frames**

Convert `rt::Framebuffer.color` floats to tone-mapped RGB PPM.

- [ ] **Step 5: Add renderer smoke test**

Update `tests/rt/test_h1_cup_pbr_render.cpp` to run one captured mini-scene through the CLI or core builder.

- [ ] **Step 6: Run renderer on capture**

```bash
build-cuda128/src/apps/nuka_h1_cup_rt_render --rollout out/h1_cup_demo/rollout.jsonl --frames out/h1_cup_demo/frames --width 1920 --height 1080 --fps 30
```

Expected: frame count matches manifest, nonzero images, robot/cup/table visible in target frame regions, no blank frames. Generate a contact sheet for visual review.

- [ ] **Step 7: Commit renderer CLI**

```bash
git add src/apps/h1_cup_rt_render.cpp src/CMakeLists.txt tests/rt/test_h1_cup_pbr_render.cpp examples/demo/h1_cup_pbr_render.py
git add -p tests/CMakeLists.txt  # stage only new renderer test target hunks
git commit -m "Render H1 cup sequence with PBR RT [skip ci]"
```

---

## Task 8: Encode and One-Command Demo

**Files:**
- Create: `examples/demo/h1_cup_render_demo.py`
- Create: `examples/demo/h1_cup_render_demo.sh` optional shell wrapper
- Modify: `examples/demo/h1_cup_demo_gate.py` to call or reference full video gate, if appropriate

- [ ] **Step 1: Add top-level command**

`examples/demo/h1_cup_render_demo.py` does:

1. set env.
2. build `nuka_h1_cup_sequence_demo_test` and `nuka_h1_cup_rt_render`.
3. run capture gate.
4. render frames.
5. run `ffmpeg` to encode MP4.
6. write `manifest.json`.
7. fail on skip/missing assets/failed thresholds/missing frames/blank frames/wrong video metadata/visibility checks.

- [ ] **Step 2: Add ffmpeg encode**

Command:

```bash
ffmpeg -y -framerate 30 -i out/h1_cup_demo/frames/frame_%05d.ppm \
  -pix_fmt yuv420p -vf "scale=1920:1080:flags=lanczos" \
  out/h1_cup_demo/h1_cup_demo.mp4
```

- [ ] **Step 3: Add manifest and machine video checks**

Include manifest fields:

- commit SHA
- bridge artifact paths
- capture command
- render command
- encode command
- metrics summary
- renderer mode: `rt_pbr_proxy` or `rt_pbr_mesh`
- camera path name and frame ranges
- frame count, width, height, fps, duration

Machine checks must parse `ffprobe` and fail unless:

- resolution is 1920x1080
- FPS is 30
- duration is 8-12 seconds
- encoded frame count matches rendered frame count
- selected frames are nonblank and show expected robot/cup/table visibility regions
- no debug overlay flag is enabled

- [ ] **Step 4: Run full one-command demo**

```bash
/root/miniconda3/envs/nuka-v03/bin/python examples/demo/h1_cup_render_demo.py --out out/h1_cup_demo
```

Expected: PASS and writes `out/h1_cup_demo/h1_cup_demo.mp4`.

- [ ] **Step 5: Commit orchestration**

```bash
git add examples/demo/h1_cup_render_demo.py examples/demo/h1_cup_render_demo.sh examples/demo/h1_cup_demo_gate.py
git commit -m "Add one-command H1 cup video demo [skip ci]"
```

---

## Task 9: Documentation and Reproducibility

**Files:**
- Create: `docs/examples/h1_cup_demo.md`
- Modify: `examples/demo/README.md`

- [ ] **Step 1: Document run commands**

Include exact env:

```bash
export PATH=/root/.nuka-toolchain-gcc14/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/opt/cuda-12.8-root/lib64:${LD_LIBRARY_PATH:-}
/root/miniconda3/envs/nuka-v03/bin/python examples/demo/h1_cup_render_demo.py --out out/h1_cup_demo
```

- [ ] **Step 2: Document honesty boundary**

State clearly:

- RL drives standing.
- Manipulation is deterministic state machine.
- Renderer is PBR-style RT/proxy or mesh mode according to manifest.
- Full dexterous manipulation RL is future work unless implemented later.

- [ ] **Step 3: Document artifact regeneration**

Include standing checkpoint export command, especially if using calibrated bias:

```bash
/root/miniconda3/envs/nuka-v03/bin/python examples/training/export_h1_tiny_actor_bridge.py \
  --checkpoint runs/h1_stand_tiny_flatstrong384/nn/h1_stand_tiny_flatstrong384.pth \
  --action-bias 9 0.30
```

- [ ] **Step 4: Commit docs**

```bash
git add docs/examples/h1_cup_demo.md examples/demo/README.md docs/plans/2026-06-09-h1-cup-demo-spec.md docs/plans/2026-06-09-h1-cup-demo-implementation-plan.md
git commit -m "Document H1 cup demo pipeline [skip ci]"
```

---

## Task 10: Final Verification and Delivery

**Files:**
- No code changes expected.

- [ ] **Step 1: Run unit/gate suite**

```bash
/root/miniconda3/envs/nuka-v03/bin/python -m pytest python/tests/test_h1_stand_env.py -q
build-cuda128/tests/nuka_h1_bridge_spike_test --gtest_filter=H1BridgeSpike.Gate1BridgeParity:H1BridgeSpike.Gate2ClosedLoopStandSoft
build-cuda128/tests/nuka_h1_dense_grasp_test --gtest_filter=H1DenseGraspLargeCup.ForceClosureLiftWithDisturbance:H1DenseGraspLargeCup.ForceClosureLiftWithDisturbanceDeterministicTwoRun:H1DenseGraspLargeCup.FingerOnlyFallbackBiteGripOffVsOn
build-cuda128/tests/nuka_h1_cup_sequence_demo_test --gtest_filter=H1CupSequenceDemo.StandsReachLiftPlaceRelease
```

Expected: all pass, no skips.

- [ ] **Step 2: Run full video command**

```bash
/root/miniconda3/envs/nuka-v03/bin/python examples/demo/h1_cup_render_demo.py --out out/h1_cup_demo
```

Expected: writes `out/h1_cup_demo/h1_cup_demo.mp4`, manifest says all gates pass.

- [ ] **Step 3: Visual review**

Inspect representative frames:

```bash
ls out/h1_cup_demo/frames/frame_*.ppm | sed -n '1p;$p'
ffprobe out/h1_cup_demo/h1_cup_demo.mp4
/root/miniconda3/envs/nuka-v03/bin/python examples/demo/h1_cup_render_demo.py --out out/h1_cup_demo --verify-only
```

Use `preview_export` or a contact sheet if remote visual inspection is needed. Manual inspection must confirm the camera is not a debug/top-down view and the final MP4 is presentation-quality PBR-style geometry, not stick figures.

- [ ] **Step 4: Final commit if needed**

```bash
git status --short
git commit -m "Finalize H1 cup demo video [skip ci]"
```

Only if there are final doc/script changes. Do not commit generated MP4 unless explicitly requested.

---

## Known Constraints

- Do not modify `src/codegen/generated/**`, `tests/oracle/golden/**`, `world_stepper.cpp`, `gpu/batched_articulated_world.*`, or K3.
- Do not include owner-excluded pre-existing files: `tests/scene/test_scene_compose_h1_cup_table.cpp`, `grasp-code-recon.md`, `subagent-plans/`, or the current `tests/CMakeLists.txt` owner hunk unrelated to this plan.
- Allowed edit surface for MVP: new tests, new demo scripts, new renderer app, docs, `src/CMakeLists.txt` app registration, and `tests/CMakeLists.txt` target registration. Existing `src/runtime/**`, `src/collision/**`, `src/solver/**`, `src/constraint/**`, and `src/rt/**` internals require owner approval before modification.
- Commit steps are checkpoints; generated MP4/frames stay out of git unless the owner explicitly asks to track artifacts.
- Every commit must include `[skip ci]` and no `Co-Authored-By`.
- Do not push without owner approval.
