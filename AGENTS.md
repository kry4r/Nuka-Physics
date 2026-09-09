# Nuka-Physics — Coding Standards

These rules OVERRIDE default behavior. Follow them in every file you write or edit.

## Comments
- Keep code comments to **2 lines maximum**.
- **No temporary or process wording in code/comments.** Never reference milestones,
  phases, or task ids (`Phase 2`, `L1-c`, `Task 3`, `M10`, `S5`, `L-RECON-B`, …).
  A comment describes the code as it stands, not the project timeline that made it.

## Architecture
- **ONE general physics solving path. No case-by-case paths, no per-scene hacks.**
  Robot+ground and robot+grasped-object are the same contact problem and run the
  same general code. Reject special-cased solvers, fused fast-paths, scene-specific
  cooks, and magic-numbered data layouts. When the general path lacks something,
  build it generally — never add a shortcut.
- Preserve a multi-backend architecture. Keep physical contracts, model topology,
  scheduling dependencies, and public APIs independent of any device backend.
- Separate architecture changes from backend tuning. CUDA launch, occupancy,
  cooperative groups, streams, graphs, intrinsics, and CUDA libraries belong in
  the CUDA backend; do not expose their types or requirements in the core model.

## Validation
- Prioritize reasoning about and changing production code. Use existing evidence
  to make related fixes, then validate the batch with the existing pipeline.
- Do not keep expanding standalone tests or reference harnesses before making
  code changes; add coverage only for a concrete unresolved failure.
- Batch related changes across modules, then build and validate the complete batch.
- Avoid repeating full validation after each small edit; rerun for failures or new risks.
- Minimize new unit tests. Prefer a fixed representative environment that runs
  the complete production pipeline.
- Extend the robot + cloth + fluid pipeline regression to cover cooking, world
  creation, control, stepping, contacts/coupling, reset, and state readout.
- Include rendering when a change affects rendered output or sensor input.
- Add focused unit/oracle tests only when needed to isolate a failure or verify
  a physical invariant that the pipeline regression cannot measure adequately.
- Test counts do not replace complete pipeline acceptance.
- The previous engine is not a physical oracle. Judge constraint error, contact
  response, conservation, and timestep convergence independently of byte identity.
- Use Newton, MuJoCo, and Genesis simulation results as physical references where
  their models apply. Match inputs and document parameter/solver differences;
  resolve disagreements with analytic cases and invariants, not majority voting.
- Preserve failing baselines. Fix shared physical errors before freezing a new
  performance denominator; do not count changed physics as a speedup.

## Performance
- In performance modules, prioritize measured latency and throughput gains and keep pursuing remaining hotspots.
- Profile GPU completion time, scheduling, bandwidth, layout, occupancy, and memory reuse on the general path.
- Compare complete pipelines at equal physical quality; report regressions, memory cost, and five-process results.
- Classify each optimization as backend-independent architecture, CUDA backend
  implementation, or physical algorithm change, and validate its own boundary.
