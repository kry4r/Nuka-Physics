# Code Context

## Files Retrieved
1. `src/runtime/coresident/unified_coresident_stepper.hpp` (lines 94-158) - `CoResidentStepReport` fields used for hold/lift/contact metrics.
2. `src/runtime/coresident/unified_coresident_stepper.hpp` (lines 162-213) - grasp config types: fingertips, cup hull/body, torque vectors, and optional table.
3. `src/runtime/coresident/unified_coresident_stepper.hpp` (lines 219-285) - public stepper API: `Step`, `Cup`, `ApplyCupImpulse`, `SetGripTorque`, `SetTableEnabled`, `Download`.
4. `src/runtime/coresident/unified_coresident_stepper.cpp` (lines 228-258) - grasp constructor uploads torque/limits and initializes table toggle.
5. `src/runtime/coresident/unified_coresident_stepper.cpp` (lines 595-1028) - `StepGrasp()` data flow: torque drive, ABA, contact solve, metrics, qdot scatter, pose integration.
6. `src/runtime/coresident/unified_coresident_stepper.cpp` (lines 1045-1061) - runtime torque update and articulation download hooks.
7. `tests/coresident/test_h1_dense_grasp.cpp` (lines 240-316) - dense contact set, driven finger links, curl helpers.
8. `tests/coresident/test_h1_dense_grasp.cpp` (lines 600-735) - scale constants, `DenseGraspScene`, `BuildDenseGraspScene`, `MakePdTarget`, `DrivePd`.
9. `tests/coresident/test_h1_dense_grasp.cpp` (lines 833-1039) - `LiveCoverage`, force-closure disturbance constants, `RunForceClosureDist`.
10. `tests/coresident/test_h1_dense_grasp.cpp` (lines 1335-1517) - finger-only lift gate, force-closure gate, and force-closure D1 gate.
11. `tests/coresident/test_h1_dense_grasp.cpp` (lines 1519-1658) - finger-only BITE and deterministic two-run gates.
12. `tests/coresident/test_h1_dense_grasp.cpp` (lines 1659-1898) - large-cup lift/BITE metrics to mirror for pass/skip thresholds.
13. `tests/CMakeLists.txt` (lines 2235-2278) - `nuka_h1_dense_grasp_test` target wiring and dependencies.
14. `src/runtime/articulation/articulation_state.hpp` (lines 114-149) - host state fields available after `Download` (`q`, `qdot`, `link_pose`, topology arrays).
15. `.nuka-assets/newton_assets/unitree_h1/mjcf/h1_with_hand.xml` (lines 251-379) - right arm/body joint names and actuator ranges.

## Key Code

Stepper hooks already expose the needed test-side control/state surface:

```cpp
const BodyState& Cup() const;
void ApplyCupImpulse(const Vec3& dv, const Vec3& dw);
void SetGripTorque(const std::vector<float>& torque);
void SetTableEnabled(bool enabled);
void Download(articulation::ArticulationHostState* out) const;
```

Existing dense helper already computes host-side direct-torque PD and feeds it through the stepper:

```cpp
articulation::ArticulationHostState st; stepper.Download(&st);
for (uint32_t l : gs.drive_links)
    tau[l] = pd.Kp * (pd.q_target[l] - st.q[l]) - pd.Kd * st.qdot[l];
stepper.SetGripTorque(tau);
```

Existing per-frame state dump can be test-only: download articulation, recompute FK, and read the live cup body:

```cpp
articulation::ArticulationHostState st; stepper.Download(&st);
const auto poses = ForwardKinematics(context, st);
const BodyState cup = stepper.Cup();
```

Reusable helpers/constants from `test_h1_dense_grasp.cpp`:
- Assets/geometry: `AssetsAvailable`, `LoadCupHull`, `ScaleCupHull`, `CupRadius`.
- H1/topology/FK: `CookedH1`, `LoadH1Fixed`, `LinkName`, `LinkByName`, `ForwardKinematics`.
- Contact/placement: `WrapSpheres`, `MeasureSurround`, `BestPlacementShallowNoPalmCaging`, `BuildDenseGraspScene`, `LiveCoverage`.
- Control: `CurlForScale`, `OffsetCurl`, `ApplyCurl`, `MakePdTarget`, `DrivePd`, `RelTilt`.
- Finger-only constants: `kFingerOnlyFallbackSxy=1.8`, `kFingerOnlyFallbackSz=1.8`, zero curl deltas, `kKp=4`, `kKd=0.4`, `kCloseOffset=0.18`, `kMu=0.8`, `kCupMass=0.2`, `kDt=1/240`, `kGravityZ=-9.81`.
- Force-closure constants: `kFcLatAccelG=1.0`, `kFcLiftSettle=50`, `kFcPushSteps=30`, `kFcReleaseSettle=70`, `kFcTiltKickW=2.5`, `kFcMaxDisp=0.07`, `kFcMaxTilt=0.35`, `kFcMaxFinalW=1.20`.

## Architecture

`UnifiedCoResidentStepper` runs in grasp mode when constructed with `GraspConfig`: each `Step()` applies the current per-link torque buffer, runs ABA/integrates velocities, emits fingertip/cup plus optional table rows, solves contacts, scatters qdot back to the device, then integrates articulation/cup poses. `Cup()` is host-resident and always current; `Download()` snapshots articulation state from the device.

The current dense scene builder intentionally fixes every H1 body except `kWrapDriven` finger/thumb links. Therefore finger-only force closure works today, but right shoulder/elbow/hand joints are fixed in the existing path. Arm-joint scripted PD is possible with current stepper hooks only if a test-side additive builder leaves those bodies free by passing them into `LoadH1Fixed` along with `kWrapDriven`. Then one combined torque vector can drive both fingers and arm joints through `SetGripTorque` each frame. Do not call separate finger and arm torque setters in one frame because the latter overwrites the full buffer.

No production stepper change is required for per-frame dumps. Add a test-only helper, e.g. `DumpChoreoFrame(frame, phase, stepper, gs, selected_links, report, ostream*)`, that records `BodyState` fields, selected H1 link transforms from `ForwardKinematics`, and report metrics. Prefer stdout or an env-gated path like `NUKA_H1_CHOREO_DUMP`; avoid committing dumps. If a true arbitrary-place demo is required, current hooks are limited: `SetTableEnabled` only toggles the construction-time table plane, and there is no public table-height/cup-pose setter. A “place back onto original table” test is possible without production changes; arbitrary table target/place height would need a small additive hook and should be treated as a separate owner decision.

Concise implementation plan:
1. Add inside `test_h1_dense_grasp.cpp` to reuse anonymous-namespace helpers; no CMake change if staying in `nuka_h1_dense_grasp_test`.
2. Create `ChoreoTraceFrame` and `ChoreoResult` structs near `ForceClosureDistResult` to hold per-frame cup state, selected link poses, report metrics, and final D1 fields.
3. Add `DrivePdMerged(stepper, gs, finger_pd, arm_tracks, frame)` that builds one per-link torque vector from downloaded `q/qdot`; keep existing `DrivePd` unchanged.
4. Add a `BuildFingerOnlyChoreoScene` wrapper using `PlaceMode::FingerOnlyShallow`, `with_palm=false`, canonical 1.8x scale/curl; optionally accept `extra_free_bodies` for arm choreography.
5. Reuse the `RunForceClosureDist` phase structure: 70-step table settle, table removal, gravity hold, optional scripted arm/push phase, release/recovery; call the frame dumper at deterministic cadence or every frame.
6. Add a small additive test such as `H1DenseGraspLargeCup.FingerOnlyForceClosureChoreographyTrace` that asserts the same hold/security metrics and prints/dumps trace metadata.
7. Add D1 sibling only after the trace runner is deterministic; compare final `BodyState`, full `q`, full `qdot`, and optional trace byte equality if captured in memory.

Honest skip conditions:
- `AssetsAvailable()` false or cup hull vertex count is zero.
- `BuildDenseGraspScene(... PlaceMode::FingerOnlyShallow ...)` returns `!gs.place.found`.
- Required named links for dump or arm PD are missing (`LinkByName == kInvalidLink`).
- Arm choreography requested but the builder did not free the arm links; skip instead of pretending arm targets are active.
- Grip-on baseline does not hold before choreography: weak vertical impulse, large `vz`, excessive displacement, or missing contacts.
- `rep.any_static_row` appears after `SetTableEnabled(false)`.
- Vertical impulse bookkeeping mismatch exceeds `5.0e-2`; treat as numerically invalid, not a physics failure.
- Scripted arm motion breaks the grasp under precommitted thresholds; skip with metrics and do not tune gains/thresholds in the same change.

Metrics to mirror:
- PLACE/geometry: `gs.place.found`, `max_pen <= kShallowPenMax`, `covered_arc`, `max_gap`, `genuine`, `sphere_count`, plus live `LiveCoverage.n/max_gap/covered_arc` at settle/end.
- LIFT/hold: `mean_fimp` within `35%` of `mg*dt`, `contact_after >= steps - 12`, `max_disp < 0.05` for lift or `peak_disp < kFcMaxDisp` for force-closure disturbance, bounded tilt/spin, `any_static_after=false`, `cross < 5.0e-2`.
- BITE: after grip-on hold, zero torque for `120` steps; grip-on requires `|z_active - cup0.z| < 0.05`, `|vz_on| < 0.5`, `vimp > 0.5*mgdt`; grip-off “falls” if `drop > 0.02` or final `vz < -0.10`.
- D1: two identical rollouts compare `BodyState` via `memcmp`, compare full `qdot` and preferably full `q`; if trace frames are stored, compare same frame count and bytes.

Validation commands:
- Build target: `cmake --build build-cuda128 --target nuka_h1_dense_grasp_test -j$(nproc)`.
- Baseline relevant tests: `./build-cuda128/tests/nuka_h1_dense_grasp_test --gtest_filter='H1DenseGraspLargeCup.FingerOnlyFallbackScaleAndCurlSweep:H1DenseGraspLargeCup.FingerOnlyFallbackBiteGripOffVsOn:H1DenseGraspLargeCup.ForceClosureLiftWithDisturbance:H1DenseGraspLargeCup.ForceClosureLiftWithDisturbanceDeterministicTwoRun'`.
- New tests after implementation: `./build-cuda128/tests/nuka_h1_dense_grasp_test --gtest_filter='H1DenseGraspLargeCup.*Choreography*'`.
- CTest equivalent: `ctest --test-dir build-cuda128 -R 'H1DenseGraspLargeCup\..*(FingerOnly|ForceClosure|Choreography).*' --output-on-failure`.

## Start Here
Open `tests/coresident/test_h1_dense_grasp.cpp` at lines 833-1039 first. `RunForceClosureDist` is the exact rollout spine to copy/factor for additive choreography, and it already uses the canonical finger-only force-closure path, table removal, support metrics, disturbance, and D1-compatible final state capture.

## Supervisor coordination
No supervisor decision was needed. The main open decision for the parent is whether Task 2b wants only a trace/choreography helper inside the existing dense target, or a true arbitrary-place demo, which would likely require a small additive production hook for table height or external placement control.
