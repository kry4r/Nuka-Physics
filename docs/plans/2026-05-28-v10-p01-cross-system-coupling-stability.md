# Nuka Physics v1.0 – Phase 1: Cross-System Coupling Stability Tuning

> **Master plan reference:** §3 Round 7 + §8 risk register (Featherstone ↔ XPBD coupling instability)
> **Prerequisites:** v0.7 closed (all 6 coupling pair types operational)
> **Blocks:** v1.0 Phases 7-8 (pour / wring demos depend on stable coupling)
> **Exit criteria gate:** v1.0
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

Make cross-system coupling **production-stable**. The risk register flags "Featherstone ↔ XPBD coupling instability" as the largest remaining technical risk; v0.7 Phase 11 delivered the functional coupling but only sufficient for the basic grasp demo (rigid cube). v1.0 needs the coupling to remain stable under:
- High-stiffness cloth + high-mass rigid (towel wringing forces).
- Fluid in motion + rigid containers (cup filling, sloshing).
- Multi-way coupling (cup + water + cloth simultaneously).

Approach: identify instability modes empirically; apply targeted fixes; bake the fixes into the row scheduler and solver config; document tuning knobs.

## Tech Stack

- All v0.7 infrastructure
- Diff-sim instrumentation (run sample tests with backward + observe gradient magnitudes; gradient blowup = instability signal)

## Files to Create

- `src/solver/coupling_stability.hpp` — tuning knob structure
- `src/solver/coupling_stability.cu`
- `src/solver/sub_step_orchestrator.hpp` — multi-sub-step coordinator
- `src/solver/sub_step_orchestrator.cu`
- `src/solver/lambda_warm_start_smoother.cuh` — smooth lambda transitions across substeps
- `src/solver/position_correction_clamp.cuh` — clamp per-particle correction magnitude
- `tests/regression/test_coupling_stability_stress_battery.cpp` — 10+ stress scenarios
- `tests/regression/test_coupling_stability_diff_sim.cpp` — gradient magnitudes bounded
- `docs/architecture/coupling-stability-tuning.md` — empirical study writeup

## Files to Modify

- `src/solver/gpu/row_solver.cu` — accept stability config; route through sub-step path
- `src/runtime/world_stepper.cpp` — sub-step orchestration
- `tools/codegen/templates/derivatives/*` — compliance-aware adjoint for stability

## Tasks

### Task 10.1.1 — Stress test battery

Build a catalog of intentionally adversarial scenarios:

| Scenario | Stress dimension |
|---|---|
| Stiff cloth (α=1e-7) compressed against rigid wall | High XPBD stiffness vs rigid contact |
| Fluid in rapidly-rotating container | Rigid motion → fluid response |
| Massive cube on thin cloth | High-mass-ratio rigid ↔ soft |
| Cup filled with water shaken | Fluid pressure on rigid wall |
| Cloth sandwich between two rigid plates squeezed | Compression with self-collision |
| Robot finger pressed into soft pillow | Featherstone-link ↔ soft |
| Water poured into beaker held by robot | Robot ↔ container ↔ fluid 3-way |
| Towel grasped + lifted | Robot ↔ cloth + cloth self-collision |
| Dropping object into pool of water | Rigid-fluid impact |
| Soft body inside fluid (buoyancy) | Soft-fluid steady-state |

Each scenario has expected qualitative behavior; passes if no NaN / no blowup / no penetration.

### Task 10.1.2 — Identify instability modes

Run the stress battery; collect failure modes:
- Position blow-up (particles fly to infinity)
- Constraint residual oscillates without converging
- Per-step lambda values flip sign rapidly (chattering)
- Gradient magnitude during backward exceeds 1e6 (numerical conditioning fail)
- Bodies tunneling through each other

Document each in `coupling-stability-tuning.md`.

### Task 10.1.3 — Sub-step orchestrator

The most general fix: run the constraint solve at smaller `dt` than the integration `dt`. Common ratios: 2-8 substeps per render frame.

```cpp
class SubStepOrchestrator {
public:
    void Run(WorldState& w, float main_dt, uint32_t substep_count) {
        float sub_dt = main_dt / substep_count;
        for (uint32_t s = 0; s < substep_count; ++s) {
            featherstone_.ComputeAccelerations(...);
            row_builder_.Build(...);
            row_solver_.Solve(..., sub_dt);
            integrator_.AdvanceArticulation(sub_dt);
            integrator_.AdvanceFreeBody(sub_dt);
            xpbd_.CorrectVelocities(sub_dt);
        }
    }
};
```

Cost: linearly more compute per main step. Benefit: dramatically smaller error per step → stability rescues many scenarios.

Heuristic auto-tuning: detect blow-up (NaN / position magnitude check) → fall back to higher substep count for next step.

### Task 10.1.4 — Lambda warm-start smoothing

When substepping, the lambda solved in substep N-1 is reused as initial guess for substep N. Sometimes this propagates instability. Solution: damped warm start:

```
lambda_init_N = (1 - alpha) * lambda_warm_start + alpha * lambda_final_{N-1}
```

`alpha` parameter; typical 0.5-0.9.

### Task 10.1.5 — Position correction clamping

XPBD large position correction in one step can cause divergence. Per particle clamp:

```
|delta_p| ≤ max_step_fraction * cell_size
```

`max_step_fraction` typical 0.5.

### Task 10.1.6 — Diff-sim stability check

Run stress battery scenarios through backward; gradient magnitudes must stay bounded.

```cpp
TEST(CouplingStability, GradientMagnitudeBounded) {
    auto world = MakeStressScenario(...);
    auto tape = MakeTape(world, /*K=*/50);
    /* ... forward 100 steps ... */
    auto loss = ComputeLoss(world);
    loss.backward();
    auto max_grad = MaxGradientMagnitude(world);
    EXPECT_LT(max_grad, 1e4f) << "Gradient blew up; coupling unstable for diff-sim";
}
```

### Task 10.1.7 — Documentation

`docs/architecture/coupling-stability-tuning.md`:

Sections:
- Catalog of identified instability modes.
- Tuning knobs and their effects.
- Recommendations per scenario type.
- Performance cost of stability (substep count vs throughput).

## Validation

- 10+ stress scenarios pass: no NaN, no blowup, no obvious physical violation.
- Diff-sim gradient magnitudes bounded.
- Performance: 4× substep count costs ~3× step time (not 4×, due to amortized contact generation).

## Exit Criteria for v1.0 Phase 1

1. Stress test battery runs cleanly on default config.
2. Sub-step orchestrator operational with auto-tuning.
3. Lambda warm-start smoothing + position correction clamping in place.
4. Diff-sim stability verified.
5. Tuning documentation written.

## What This Phase Does Not Do

- No new physics features.
- No new sensor types.
- Does not change row IR or codegen.
- Does not promise solving every conceivable case — research-grade scenarios may still fail.
