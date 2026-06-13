# Code Context

## Files Retrieved
1. `src/runtime/coresident/unified_coresident_stepper.hpp` (lines 60-135) - co-resident foot/box/ground report fields plus grasp/lift report metrics.
2. `src/runtime/coresident/unified_coresident_stepper.hpp` (lines 170-213) - grasp fingertips, cup hull, and optional table configuration.
3. `src/runtime/coresident/unified_coresident_stepper.hpp` (lines 219-283) - `UnifiedCoResidentStepper` public API available to tests.
4. `src/runtime/coresident/unified_coresident_stepper.cpp` (lines 68-193) - internal geometry/Jacobian/inertia/ref helpers.
5. `src/runtime/coresident/unified_coresident_stepper.cpp` (lines 595-1014) - `StepGrasp()` data flow: drive, contact direct-emission, table toggle, row solve, report metrics, qdot scatter.
6. `src/runtime/coresident/unified_coresident_stepper.cpp` (lines 1030-1063) - integration, torque setter, download implementation.
7. `tests/coresident/test_h1_scaled_cup_grasp.cpp` (lines 62-135) - constants, asset gates, cup hull loading/scaling.
8. `tests/coresident/test_h1_scaled_cup_grasp.cpp` (lines 143-319) - H1 import/fixed-base helpers, wrap sphere set, curl/coverage primitives.
9. `tests/coresident/test_h1_scaled_cup_grasp.cpp` (lines 321-624) - scaled placement, scene builder, PD, live coverage, scale sweep.
10. `tests/coresident/test_h1_scaled_cup_grasp.cpp` (lines 649-1100) - scaled-cup probe, step-zero sweep, lift, bite, dynamic load, D1 tests.
11. `tests/coresident/test_h1_dense_grasp.cpp` (lines 81-155) - dense-test constants, asset gates, cup hull scaling.
12. `tests/coresident/test_h1_dense_grasp.cpp` (lines 157-349) - H1 import/fixed-base helpers and dense wrap sphere-chain setup.
13. `tests/coresident/test_h1_dense_grasp.cpp` (lines 349-696) - dense placement filters, scene builder, unique handles, PD, live coverage.
14. `tests/coresident/test_h1_dense_grasp.cpp` (lines 698-1132) - dense-contact realization, shallow calibration, lift, bite, dynamic load, D1 tests.
15. `tests/CMakeLists.txt` (lines 1935-2278) - exact coresident/grasp executable targets and linked dependencies.
16. `CMakeLists.txt` (lines 24-31) - top-level build options controlling tests and render/demo validation.
17. `.github/PULL_REQUEST_TEMPLATE.md` (lines 13-16) and `CONTRIBUTING.md` (lines 118-135) - protected golden/generated-code risk constraints found by targeted grep.

## Key Code

### Stepper public helpers available
```cpp
const BodyState& Cup() const;
void ApplyCupImpulse(const Vec3& dv, const Vec3& dw);
void SetGripTorque(const std::vector<float>& torque);
void SetTableEnabled(bool enabled);
bool TableEnabled() const;
void Download(ArticulationHostState* out) const;
CoResidentEnergy Energy() const;
CoResidentStepReport Step();
```
Defined in `src/runtime/coresident/unified_coresident_stepper.hpp` lines 242-283.

### Step report fields useful for lift gates
`CoResidentStepReport` includes:
- contact presence/counts: `pair_found`, `manifold_count`, `row_count`, `finger_contacts`
- finger support: `cup_vertical_impulse`, `cup_dvz_impulse`, `finger_vimpulse_normal`, `finger_vimpulse_friction`
- cup state: `cup_vz`, `cup_z`, `box_dv_norm`, `qdot_delta_l1`
- table support/removal proof: `any_static_row`, `table_row_count`, `table_lambda`, `table_vertical_impulse`
- penetration/impulse: `contact_depth`, `lambda`
See `src/runtime/coresident/unified_coresident_stepper.hpp` lines 89-135.

### Grasp config helpers
`GraspConfig` already supports the scale/lift validation needs:
- `fingertips`: N `CoResidentFingertip` spheres with `link`, `local_offset`, `radius`, and optional independent `broadphase_handle`.
- `cup`: convex hull verts and `broadphase_body_id`.
- `cup_state`: movable rigid state/mass/inertia/pose.
- `grip_torque` and `drive_force_limits`: per-device-link drive input.
- `friction_mu`, `condim`: contact friction setup.
- `has_table`, `table_height`, `table_mu`, `table_broadphase_id`: inert-by-default table support for lift choreography.
See `src/runtime/coresident/unified_coresident_stepper.hpp` lines 170-213.

### StepGrasp data flow
`UnifiedCoResidentStepper::StepGrasp()` is already the correct spine:
1. Re-applies grip torque through `LaunchApplyTorqueDriveKernels()` before ABA (`src/runtime/coresident/unified_coresident_stepper.cpp` lines 600-607).
2. Runs ABA + velocity integration, gravity-kicks cup (`src/runtime/coresident/unified_coresident_stepper.cpp` lines 610-620).
3. Downloads articulation + FK poses, builds world cup hull (`src/runtime/coresident/unified_coresident_stepper.cpp` lines 623-633).
4. Direct-emits every fingertip sphere vs cup candidate pair using `broadphase_handle` (`src/runtime/coresident/unified_coresident_stepper.cpp` lines 640-663).
5. Optionally emits cup/table pair only when `grasp.has_table && table_enabled_` (`src/runtime/coresident/unified_coresident_stepper.cpp` lines 665-676).
6. Resolves sphere/cup/table shapes and builds manifolds/rows (`src/runtime/coresident/unified_coresident_stepper.cpp` lines 679-757).
7. Stamps finger/table friction, computes chain-J per finger row, maps handle back to real link (`src/runtime/coresident/unified_coresident_stepper.cpp` lines 761-891).
8. Runs `UnifiedSolve`, reports finger/table vertical impulse and contact metrics (`src/runtime/coresident/unified_coresident_stepper.cpp` lines 899-966).
9. Writes solved cup state, cup dvz impulse cross-check, qdot delta, and scatters qdot back to device (`src/runtime/coresident/unified_coresident_stepper.cpp` lines 991-1014).

### Test helpers available in scaled-cup TU
`tests/coresident/test_h1_scaled_cup_grasp.cpp` provides reusable local patterns:
- Asset gates: `AssetsAvailable()` checks `.nuka-assets/newton_assets/unitree_h1/mjcf/h1_with_hand.xml` and `.nuka-assets/newton_assets/manipulation_objects/cup/model.usda` (lines 68-74).
- Cup scale: `LoadCupHull()`, `ScaleCupHull()`, `CupRadius()`, `ScaledMass()` (lines 80-131, 298-307, 612-628).
- H1 setup: `LoadH1Fixed()`, `LinkByName()`, `ForwardKinematics()` (lines 143-209).
- Wrap geometry: `WrapSpheres()`, `CurlPose`, `ApplyCurl()`, `SphereCenter()`, `MeasureSurround()`, `BestPlacement()` (lines 229-455).
- Scene/drive: `ScaledGraspParams`, `BuildScaledGraspScene()`, `MakePdTarget()`, `DrivePd()` (lines 457-559).
- Live validation: `RelTilt()`, `LiveCoverage()`, `MaxPrePenetration()` (lines 561-610).
- Existing sweep: `kScaleSweep = {{1.4,1.2},{1.6,1.2},{1.8,1.2}}`, dynamics pinned at `kDynSxy=1.6`, `kDynSz=1.2`, `kCloseOffset=0.18`, `kKp=4`, `kKd=0.4` (lines 612-779).

### Test helpers available in dense TU
`tests/coresident/test_h1_dense_grasp.cpp` mirrors scaled-cup helpers but changes contact density:
- Dense sphere chain: 3 small spheres per phalanx plus palm chain, `kWrapRadius=0.006`, `kPalmRadius=0.010`, `WrapSpheres()` (lines 234-287).
- Shallow-depth discriminator: `kShallowPenMax=0.002`, `BestPlacementShallow()`, `BestPlacementShallowNoPalm()`, `BestPlacementVoidClosed()` (lines 446-532).
- Dense scene: `BuildDenseGraspScene()` uses `PlaceMode::{Shallow,VoidClosed}` and assigns unique `broadphase_handle` values starting at `9000u` while preserving real `link` for chain-J (lines 541-633).
- Drive/live helpers: `MakePdTarget()`, `DrivePd()`, `RelTilt()`, `LiveCoverage()` (lines 634-696).

## Architecture

The H1 grasp tests are not wired through rendering/demo entry points. They are standalone gtest executables that compile `src/runtime/coresident/unified_coresident_stepper.cpp` and `src/collision/contact_stream_driver.cpp` directly, link solver/constraint/runtime/import/cooker/articulation/collision/phi libs, and run from the repo root so asset-relative paths resolve.

The runtime stepper is the central simulation bridge. In grasp mode, tests construct `GraspConfig` with H1 fingertip spheres, scaled convex-hull cup vertices, cup rigid `BodyState`, friction, and optional table support. Each test loop computes PD torques host-side via `Download()`, feeds them through `SetGripTorque()`, calls `Step()`, and gates on `CoResidentStepReport` plus `Cup()` pose/velocity.

The table-removal lift gate is already supported without editing the stepper: construct with `has_table=true`, settle on table, call `SetTableEnabled(false)`, then assert `any_static_row == false`, `table_row_count == 0` implicitly via reports after removal, finger vertical impulse carries a meaningful fraction of `m*g*dt`, contact count remains high, cup translation/tilt stays bounded, and D1 remains byte-identical.

The scaled-cup test currently sweeps geometry in Step Zero only, then pins dynamics to 1.6x. The dense test pins dynamics to 1.6x and adds shallow/deep placement modes plus unique handles to validate dense contact realization. Both test files are local-helper-heavy; adding a new Phase 1 H1 large-cup validation should likely stay in `tests/coresident/` by extending or adding a sibling TU, not by touching runtime/rendering.

## Test Patterns

### Scaled-cup patterns
- Probe: `H1ScaledCupProbe.PalmAndFingerGeometry` prints palm/world geometry and offset sweep (`tests/coresident/test_h1_scaled_cup_grasp.cpp` lines 649-678).
- Step-zero scale sweep: `H1ScaledCupStepZero.ScaledCupClosesPalmVoid` loops over `kScaleSweep`, prints coverage/gap/palm contact, asserts some scale closes the void (`tests/coresident/test_h1_scaled_cup_grasp.cpp` lines 688-760).
- Calibration: `H1ScaledCupDynamics.PrePenetrationAndPlacementCalibration` reports max/palm pre-penetration at pinned dynamic scale (`tests/coresident/test_h1_scaled_cup_grasp.cpp` lines 803-837).
- Lift: `H1ScaledCupDynamics.LiftGateScaledWrapCagesCup` settles 70 steps with table, removes table, runs 220 lift steps, checks no static rows, support impulse, cross-check, translation, contacts, qdot, tilt/spin; skips with honest-negative if angular spread collapses (`tests/coresident/test_h1_scaled_cup_grasp.cpp` lines 847-969).
- BITE: `H1ScaledCupDynamics.ActiveGraspBiteGripOffFalls` validates active-vs-passive by zeroing torque after table removal; current protocol skips with finding if cup does not fall (`tests/coresident/test_h1_scaled_cup_grasp.cpp` lines 983-1027).
- Dynamic load: applies lateral/angular impulse and temporary upward acceleration, gates contact retention and tilt (`tests/coresident/test_h1_scaled_cup_grasp.cpp` lines 1035-1069).
- D1: two rollouts compare `BodyState` and qdot byte-for-byte (`tests/coresident/test_h1_scaled_cup_grasp.cpp` lines 1076-1100).

### Dense-grasp patterns
- Apparatus gate: `H1DenseGrasp.DenseContactsAreRealized` uses deep void-closed placement to force contact and asserts `max_contacts >= 25`; may skip after reporting dense bookkeeping limitation (`tests/coresident/test_h1_dense_grasp.cpp` lines 698-768).
- Shallow calibration: `H1DenseGrasp.ShallowPlacementCalibration` is the main feasibility discriminator for `<=2mm`, palm co-contact, and max-gap requirements (`tests/coresident/test_h1_dense_grasp.cpp` lines 785-865).
- Lift: `H1DenseGrasp.LiftGateDenseWrapCagesCup` mirrors scaled lift but first skips if shallow placement is infeasible; caged means support, translation, contact, qd, arc, tilt, no static rows, and impulse cross-check (`tests/coresident/test_h1_dense_grasp.cpp` lines 875-980).
- BITE: `H1DenseGrasp.ActiveBiteGripOffVsOn` separates active friction from passive/form closure at shallow depth (`tests/coresident/test_h1_dense_grasp.cpp` lines 991-1049).
- Dynamic load and D1 mirror scaled patterns; D1 deliberately uses deep placement to ensure contact-active determinism (`tests/coresident/test_h1_dense_grasp.cpp` lines 1056-1132).

## Exact Target Names

Coresident/grasp targets in `tests/CMakeLists.txt` lines 1949-2277:
- `nuka_foot_box_coresidence_test`
- `nuka_foot_box_mjx_parity_test`
- `nuka_unified_coresident_stepper_test`
- `nuka_grasp_hold_spike_test`
- `nuka_h1_grasp_spike_test`
- `nuka_h1_power_grasp_lift_test`
- `nuka_h1_scaled_cup_grasp_test`
- `nuka_h1_dense_grasp_test`

The two requested H1 large-cup-adjacent targets are exactly:
- `nuka_h1_scaled_cup_grasp_test` (`tests/CMakeLists.txt` lines 2201-2232)
- `nuka_h1_dense_grasp_test` (`tests/CMakeLists.txt` lines 2246-2277)

## Likely Minimal Place To Add Scale Sweep/Lift Gate

Best minimal path: add a focused test in `tests/coresident/test_h1_scaled_cup_grasp.cpp` near the existing dynamics section, because it already has:
- cup scale sweep (`kScaleSweep`, `ScaleCupHull`, `CurlForScale`),
- scene construction (`BuildScaledGraspScene`),
- lift choreography (`SetTableEnabled(false)` and 220-step window),
- validation metrics (`LiveCoverage`, `RelTilt`, `CoResidentStepReport` fields),
- exact target already built as `nuka_h1_scaled_cup_grasp_test`.

If the new Phase 1 H1 large-cup validation requires dense contacts or shallow-depth gating, use `tests/coresident/test_h1_dense_grasp.cpp` instead or add a sibling TU copying its helper structure. Dense has the unique-handle path and `BestPlacementShallow()` already; scaled does not.

Suggested local seam in scaled TU:
1. Extend `kScaleSweep` or add a new local `LargeCupScaleSpec` list near `tests/coresident/test_h1_scaled_cup_grasp.cpp` lines 612-621.
2. Add a helper like `RunLiftGateForScale(spec)` near `DynParams()` lines 768-793 or inline in a new `TEST(...)` after `ScaledCupClosesPalmVoid`/before current pinned dynamics.
3. Reuse `ScaledGraspParams` with `p.hull = ScaleCupHull(base, spec.sxy, spec.sz)`, `p.curl = CurlForScale(spec.sxy)`, `p.has_table=true`, `p.require_palm=true`, `p.cup_mass=ScaledMass(...)`.
4. Reuse current lift loop from `LiftGateScaledWrapCagesCup`, but record per-scale pass/fail metrics instead of hardcoding `kDynSxy=1.6`.
5. Keep BITE separate if Phase 1 only asks “large-cup grasp validation”; if it needs active-friction proof, include `grip=0` contrast from `ActiveGraspBiteGripOffFalls`.

Do not add this in `src/runtime/coresident/unified_coresident_stepper.*` unless new physics/report fields are required. Existing APIs already support scale sweep, table lift, PD torque, dynamic impulse, and reports.

## Validation Commands

Build only the relevant targets, using the existing CUDA build tree if configured:
```bash
cmake --build build-cuda128 --target nuka_h1_scaled_cup_grasp_test nuka_h1_dense_grasp_test -j
```

Run the scaled-cup gtests directly from repo root so asset-relative paths work:
```bash
./build-cuda128/tests/nuka_h1_scaled_cup_grasp_test --gtest_filter='H1ScaledCup*' --gtest_color=no
```

Run the dense gtests directly:
```bash
./build-cuda128/tests/nuka_h1_dense_grasp_test --gtest_filter='H1DenseGrasp*' --gtest_color=no
```

Run through CTest if test discovery is available:
```bash
ctest --test-dir build-cuda128 -R 'H1ScaledCup|H1DenseGrasp' --output-on-failure
```

Fallback build tree if `build-cuda128` is not the active local build:
```bash
cmake --build build --target nuka_h1_scaled_cup_grasp_test nuka_h1_dense_grasp_test -j
ctest --test-dir build -R 'H1ScaledCup|H1DenseGrasp' --output-on-failure
```

Assets are required or tests skip: `.nuka-assets/newton_assets/unitree_h1/mjcf/h1_with_hand.xml` and `.nuka-assets/newton_assets/manipulation_objects/cup/model.usda`.

## Forbidden Paths Risk

High-risk/no-touch for a minimal Phase 1 test addition:
- `src/codegen/generated/**`: PR template explicitly says no edits; regenerate via build if needed.
- Protected golden artifacts: `CONTRIBUTING.md` warns not to modify protected goldens without explicit reviewed protected-change proposal.
- Production integrator/world paths: comments in `unified_coresident_stepper.hpp` state this stepper reuses `FeatherstoneAba` and is additive/validated-not-wired; do not modify `BatchedArticulatedWorld`, `world_stepper`, or ABA integration to make a grasp test pass.
- `src/runtime/coresident/unified_coresident_stepper.*`: not forbidden, but broad blast radius because every coresident/grasp target compiles it directly; only edit if existing report/API is insufficient.
- `tests/coresident/test_h1_power_grasp_lift.cpp` and prior negative gates: comments in scaled/dense TUs say prior honest-negative gates are left intact; avoid mutating them unless explicitly asked.
- Rendering/demo entry points: unnecessary for Phase 1 H1 grasp validation; render/demo options are off by default (`NK_BUILD_VULKAN_VALIDATION`, `NK_BUILD_DEMO_TIMING_VALIDATION`) and should not be pulled into this validation.
- `.nuka-assets/**`: treat as external assets, not source patches.
- CMake outside the exact target block: only needed if adding a new sibling TU/target; otherwise avoid changes beyond `tests/coresident/test_h1_scaled_cup_grasp.cpp` or `test_h1_dense_grasp.cpp`.

## Start Here

Open `tests/coresident/test_h1_scaled_cup_grasp.cpp` first. It already owns the large-cup scale sweep, cup scaling, H1 fixed-base setup, palm/finger placement, PD close, table-removal lift gate, BITE gate, dynamic load, and D1 pattern. For dense/shallow validation, open `tests/coresident/test_h1_dense_grasp.cpp` next and reuse its unique-handle `BuildDenseGraspScene()` and `BestPlacementShallow()` patterns.
