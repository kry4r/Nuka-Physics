# Physics acceptance and pipeline measurements

`nuka_pipeline_benchmark` records two different decisions. `status.valid` covers
world creation, stepping, state and sensor replay, reset, deformation limits and
required coupling workloads. `status.passes_full_physics_acceptance` requires all
physical acceptance layers to pass. A valid lifecycle can still have failed
physics or incomplete measurements.

## Physical readouts

`quality.physical_acceptance` uses the same production energy, stage momentum,
contact audit and solver residual fields consumed by Python's
`nuka.diagnostics.DiagnosticSession`. It requests these fields after the timed
window and reads every interval of the quality replay, including warmup. The
timed and replayed physical state must remain byte identical to bind the
observations to the measured run.

Surface contact endpoints fit rotation in local edge coordinates with double
precision intermediates. Their forward and transpose products also accumulate
in double precision before returning single precision state. The CUDA impulse
audit sums the production Jacobians in double precision so its own cancellation
does not dominate the closure check. Rounded triangle coefficients retain a unit
partition; focused checks verify rigid motion and both impulse momenta under
world translation.

Each check has a `status` and `coverage_complete`. An observed violation remains
`failed` when another part of its coverage is unavailable. Missing coverage alone
is `unmeasured`. These states also apply to Python diagnostic contact reports.

| Readout | Reported evidence | Limit |
|---|---|---|
| Contact impulse audit | Row visits, invalid rows, linear closure, friction cone and negative normal impulses | Active production contact rows |
| Committed gaps | Gap violations, maximum penetration and its step/environment/substep/row | Unsupported feature kinds remain unmeasured |
| Contact work rate | Signed impulse-direction displacement rates | Uses the declared acceptance velocity tolerance |
| Solver residual | Normal, tangent, joint and primal maxima in m/s | Only the last solve of each policy step is retained |
| Stage diagnostics | Finite state/execution, solved VBD momentum-equation error and positive passive-row work | An equation residual alone does not prove conservation |
| Energy ledger | Available work/loss terms, coverage bits, finite and unwritten record counts | Partial terms cannot establish a full energy balance |
| Motion truncation | Truncation count visits | Only the last physical interval per policy step is retained |

Counts are accumulated visits across replay intervals, rather than unique
contacts. Maximum witnesses use one-based policy steps and zero-based
environment/substep indices. Residual and truncation scopes are stated
separately because their production buffers do not retain every substep.

The audit retains production thresholds: relative linear closure `1e-6`, friction
cone factor `1.001`, nonnegative normal impulses and the configured position
slop. A positive `NUKA_SOLVER_VEL_TOLERANCE` controls the production solve and is
also reported as the residual/work-rate acceptance budget. A zero stopping
tolerance denotes a fixed iteration solve; the benchmark then checks residuals
and work rates against the existing production tile tolerance of `1e-5` m/s.
The report names that source and records both tolerances. This acceptance check
does not change the fixed iteration solve. A failure in the last substep remains
a failure, while a passing last-substep residual cannot certify earlier substeps.

The energy coverage mask follows `nk::energy_status`: nonfinite values,
unclassified rows, other particle materials, grid materials, external body loads,
floating position work, rigid angular position work, armature energy and
execution failure. Complete MPM material energy and prescribed boundary work are
not covered. Even with a clear coverage mask, the benchmark leaves the independent
balance criterion unmeasured. Analytic balances, matched external reference
results, timestep/grid convergence and independent CCD require separate evidence
and remain unmeasured in this program.

For detailed preserved traces, use `DiagnosticSession` and the maintained report
command:

```bash
python python/physics_diagnostics.py report out/physics-trace --output out/physics-report
```

The Python report additionally consumes bound analytic/reference, solver-budget,
independent state/elastic and CCD artifacts. Its full acceptance decision still
requires every required physical system and layer to be covered.

## Timing and memory

GPU step timing uses completion events. Physical readbacks and their
synchronization belong to the separately timed quality replay.
Physics-to-sensor and render-only measurements complete before physical
diagnostics are requested, so their physics timing excludes those observers.
`quality.timed_replay_bit_equal` and the physical report's binding field detect
an observer-induced change to the trajectory. Failed binding prevents acceptance.

`timed_memory` records storage before physical diagnostics are requested.
`memory` records storage after diagnostics. Persistent/workspace totals include
retained arenas and separately allocated replacement fields. The
`compact_layout_*` values describe the hypothetical packed layout at the current
capacities; replacements can retain the original arena allocation as well.
These are requested buffer bytes, excluding allocator reservation, backend plans
and renderer storage. Arena field offsets describe the original arena segments.

## Independent processes

```bash
python tools/perf/run_pipeline_sweep.py \
  --binary build/src/nuka_pipeline_benchmark \
  --output out/pipeline-observations --source-id source-manifest-sha256 \
  --scene robot-rigid-mpm-cloth --envs 2 --executions eager graph \
  --processes 5 --warmup 250 --steps 200 --velocity-iterations 48 \
  --state-sensors 1 --render-sensors 1 --render-width 128 --render-height 128 \
  --save-state --save-wrench --complete-failing-observations
```

The runner records loaded executable/library identities and hashes, commands,
inputs, solver budgets, GPU clocks, completion latency and cross-process
state/workload identity. Record a source manifest and the full input closure
alongside the supplied source ID.

By default a measured physics failure stops the sweep. The explicit
`--complete-failing-observations` option completes the requested independent
processes when lifecycle validation remains valid. Results retain failed
acceptance and the runner still exits unsuccessfully after writing its summary.
`--allow-unaccepted-baseline` permits an unmeasured baseline to finish successfully;
it does not waive a measured failure. A source/input hash, five processes or a
successful render cannot replace complete physical acceptance. Freeze a
performance denominator only after equal physical quality has been established.
