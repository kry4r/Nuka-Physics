# H1 Cup Demo Spec

**Status:** Draft for implementation
**Branch:** `master`
**Goal:** Produce a complete, presentable H1 cup demo video: a trained RL standing policy keeps H1 upright while a deterministic full-body controller reaches, closes the hand on a cup, lifts it, places it back down, releases it, and renders the sequence with PBR-style lighting/materials and a presentation-quality camera.

## Scope

### MVP Deliverable

A reproducible command produces:

1. A full-world co-resident rollout artifact containing H1 link poses, cup pose, table pose, phase labels, and validation metrics.
2. A hard validation gate proving:
   - RL standing bridge remains active throughout the sequence.
   - H1 remains upright: `tilt_max < 15 deg`.
   - Foot contacts remain stable: `min_contacts >= 4/4` except for explicitly reported one-frame contact solver jitter if any; target is zero low-contact frames.
   - Leg torques remain within H1 MJCF limits.
   - The hand closes on the cup and lifts it off table support.
   - The cup is placed back on the table.
   - After release, the cup remains stable without grip support: bounded displacement, bounded tilt, low velocity, and table support present.
   - No NaN/Inf in robot, cup, or renderer inputs.
3. A rendered MP4 at `out/h1_cup_demo/h1_cup_demo.mp4`, plus frames and a manifest.

### Explicit Non-MVP

The MVP does **not** require an RL policy for hand reaching or grasping. RL drives standing. Arm/hand/cup choreography is a deterministic full-world state machine using existing co-resident stepper mechanics. Training a manipulation RL policy is a later research task.

This split is intentional: current verified assets already prove standing RL and dense hand/cup force closure independently. The shortest credible demo is to integrate them into one continuous rollout and render it. Turning dexterous reach/grasp/place into RL would be a new long-running research campaign and should not block the video.

## Current Proven Building Blocks

- Standing policy gate: `tests/coresident/test_h1_bridge_spike.cpp` loads `python/spikes/out/h1_bridge_mlp.bin`, projects full-world H1 state to the 32-dim bridge observation, runs `TinyMlp<32,64,10>`, applies normalized action times per-joint H1 limits, and asserts S4 standing metrics.
- H1 reduced training env: `python/nuka/tasks/h1_stand.py` implements 32-dim observation and 10-dim normalized torque action.
- Dense hand/cup grasp gate: `tests/coresident/test_h1_dense_grasp.cpp` builds the H1 hand + large cup dense-contact scene and validates force-closure hold under disturbance, grip-off bite, and deterministic replay.
- Demo gate wrapper: `examples/demo/h1_cup_demo_gate.py` proves standing and dense grasp gates, but does not run one continuous sequence and does not render.
- Rendering precedent: Go2 demo uses capture-to-frames-to-MP4, but H1 needs a new PBR-oriented renderer/capture path.

## Full Demo Behavior

The sequence is phase-based:

1. **Establish:** H1 stands in the full co-resident world under the exported RL leg policy. Upper body is held in a neutral presentation posture. Cup rests on a table in front/right of the robot.
2. **Reach:** Upper-body and arm joints follow a deterministic pose schedule toward the prevalidated grasp pose. Leg policy continues at the bridge rate. The cup remains table-supported.
3. **Pregrasp:** Fingers open around the cup using the dense-grasp placement geometry. The hand approaches without destabilizing the base.
4. **Close:** Hand grip torque/PD target closes to the validated dense-contact grasp. Validation starts tracking finger/cup contacts and grip support.
5. **Lift:** Table support for the cup is removed or the cup is lifted upward with the hand, depending on what the existing co-resident grasp path exposes. The gate must prove cup support comes from hand contacts, not table support.
6. **Place:** Arm schedule lowers cup back to the table. The gate watches cup height, table support, tilt, and velocity.
7. **Release:** Grip torque/PD target opens. Cup remains on table for a post-release settle window.
8. **Hero Hold:** Robot stays standing with released cup visible on table for a final camera beat.

## Controller Architecture

### Legs

Use the already-exported standing policy artifact:

- Input: C++ full-world bridge observation, 32 dims.
- Network: `TinyMlp<32,64,10>`.
- Output: normalized leg actions, clamped to `[-1,1]`.
- Torque mapping: per-joint H1 limits in contract order `[200,200,200,300,40]` per leg.

The demo runner must not change the observation contract or introduce a hidden action bias. Any calibration must already be folded into the exported weights/goldens and covered by Gate 1 parity.

### Upper Body and Hand

Use a deterministic phase controller:

- Non-leg joints use PD targets and torque limits as in existing H1 grasp/standing tests.
- Arm poses are interpolated between named keyframes.
- Hand close/open uses the dense-grasp `CurlPose`/drive-link style from `test_h1_dense_grasp.cpp`.
- The state machine is deterministic and phase durations are fixed.

### Cup and Table

Use the dense-contact large cup setup already validated by `test_h1_dense_grasp.cpp`:

- Cup asset: `.nuka-assets/newton_assets/manipulation_objects/cup/model.usda`.
- H1 asset: `.nuka-assets/newton_assets/unitree_h1/mjcf/h1_with_hand.xml`.
- Table support is explicit and logged.
- Release gate proves the cup remains stable on the table without active grip.

## Validation Metrics

The sequence gate should emit a JSON manifest containing at least:

- `standing.tilt_max_deg`
- `standing.min_foot_contacts`
- `standing.low_contact_frames`
- `standing.peak_ankle_nm`, `peak_knee_nm`, `peak_hip_nm`
- `cup.lift_height_delta_m`
- `cup.max_lift_tilt_rad`
- `cup.place_final_height_error_m`
- `cup.post_release_max_displacement_m`
- `cup.post_release_max_tilt_rad`
- `cup.post_release_max_linear_speed_mps`
- `cup.post_release_table_contact_frames`
- `sequence.phase_frame_ranges`
- `render.output_mp4`, `render.frame_count`, `render.width`, `render.height`, `render.fps`

Hard pass thresholds for MVP:

- `tilt_max_deg < 15`
- `min_foot_contacts == 4`
- leg torques within limits
- `lift_height_delta_m >= 0.05`
- during lift, table support for the cup is absent or below a reported near-zero threshold
- `place_final_height_error_m <= 0.01`
- `place_final_xy_error_m <= 0.03`
- during post-release settle, table contact/support is present for at least 90% of settle frames
- after release, grip command is off/open and hand/finger vertical impulse is near zero
- `post_release_max_displacement_m <= 0.03`
- `post_release_max_tilt_rad <= 0.20`
- `post_release_max_linear_speed_mps <= 0.10`
- no NaN/Inf

Thresholds may be tightened after the first working rollout, but may not be loosened silently to pass.

## Capture Artifact

The runner writes `out/h1_cup_demo/rollout.npz` or equivalent containing:

- H1 link poses per frame.
- Cup pose per frame.
- Table pose and dimensions.
- Camera phase metadata.
- Phase labels per frame.
- Metric JSON.

The capture artifact is the source of truth for rendering. Rendering must not rerun physics with different behavior.

## Rendering Requirements

The final video must use a real PBR-style renderer, not the Go2 stick-figure/debug-raster path. The Go2 capture/render scripts are only useful as a pattern for `capture -> frames -> ffmpeg`; their visual approach is not acceptable for this demo.

Required output:

- MP4: `1920x1080`, 30 FPS, 8-12 seconds.
- Geometry-based rendering of H1, table, floor, and cup; no stick skeleton as the final artifact.
- PBR-style materials: dark studio floor, matte table, realistic cup albedo/roughness, H1 metal/plastic material separation where available.
- Lighting: large soft key light, cool rim light, subtle fill, visible ground/contact shadows.
- Camera: NVIDIA keynote style reference — low 3/4 hero angle, slight telephoto compression, slow dolly/orbit, robot and cup centered, no top-down debug view.
- Composition: establish feet/stance, cut or dolly to hand/cup close-up during grasp/lift/place, final hero shot after release.
- On-screen debug overlays are off by default; optional metrics overlay can be generated separately.

Renderer implementation requirement:

- Do **not** reuse `examples/demo/go2_demo_render.py` as final rendering. It is a stick/raster preview path.
- Prefer building a new app on top of the existing CUDA RT/two-level renderer APIs because they already provide triangle/sphere primitives, GGX-like material fields, shadows, camera, AOVs, and per-frame instance transforms. Do not modify existing `src/rt/**` without owner approval; first use the public API from a new app/test.
- If imported H1 mesh geometry is not directly available to RT on day one, use a high-quality analytic proxy renderer over captured link poses (capsules/spheres/boxes with PBR shading, shadows, depth, camera motion) as an intermediate, but it must be visually polished and still PBR-style. Mark any analytic proxy usage explicitly in the manifest.
- The production/debug Vulkan storage-image renderer is not sufficient by itself because it currently maps `RenderScene` to debug draw boxes/capsules and lacks the desired photoreal camera/material/shadow presentation.
- Because the current RT stack is bounded direct-light GGX/Lambert rather than a full path tracer, the MVP visual target is `PBR-style presentation-quality`: geometry, materials, shadows, tone mapping, supersampling/anti-aliasing if needed, studio lighting, and good camera motion. The manifest must disclose whether the renderer used analytic proxy geometry (`rt_pbr_proxy`) or imported mesh geometry (`rt_pbr_mesh`).

## Reproducibility

Add a manifest and regenerate command:

- Standing checkpoint used for export.
- Export command including `--action-bias 9 0.30` if that remains the accepted calibrated artifact.
- Bridge artifact paths.
- Build/run environment exports.
- Demo capture command.
- Render command.
- Encode command.

Ignored binary artifacts are allowed for local delivery, but the repo must contain enough scripts and documentation to regenerate them.

## Acceptance Command

The final top-level command should be one of:

```bash
examples/demo/h1_cup_render_demo.sh out/h1_cup_demo
```

or

```bash
/root/miniconda3/envs/nuka-v03/bin/python examples/demo/h1_cup_render_demo.py --out out/h1_cup_demo
```

It should build required targets if needed, run the sequence gate, render frames, encode MP4, write manifest, and exit nonzero on any failed gate or skipped asset.

## Risks and Fallbacks

1. **Full standing + grasp controller integration may expose co-resident API gaps.** Fallback: first build a sequence gate that reuses test-side helpers, then refactor into a demo runner.
2. **Production PBR renderer may not ingest captured articulation/cup poses.** Fallback: implement a polished headless renderer over captured poses, then upgrade to Vulkan/PBR ingestion later.
3. **Reach trajectory may destabilize standing.** Fallback: shorten arm motion, keep center of mass close, use a camera-framed tabletop pose near the validated grasp configuration.
4. **Release/place may fail with current cup/table contact.** Fallback: add a hard negative gate and tune deterministic place height/orientation, not physics constants.
5. **Artifacts are ignored.** Fallback: scripts regenerate artifacts and manifest records provenance.

## Non-Negotiables

- Do not change locked 32-dim standing observation contract.
- Do not change normalized 10-action to per-joint torque-limit mapping.
- Do not silently hard-code runtime action biases outside exported/golden-validated weights.
- Do not claim manipulation is RL-trained unless a manipulation RL training loop and acceptance gate are actually added.
- Do not ship a video that is not generated from a passing captured rollout.
- Allowed edit surface for MVP: new tests, new demo scripts, new renderer app, docs, and CMake registrations. Existing production runtime/collision/solver/constraint/RT internals require owner approval before modification.
- The top-level video command must machine-check MP4 resolution, FPS, duration, frame count, missing/blank frames, and basic robot/cup visibility; manual inspection is additional evidence, not the only video gate.
