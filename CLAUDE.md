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

## Debugging workflow
- Measure first with the unified diagnostics (`nuka.diagnostics.DiagnosticSession` and
  `python/physics_diagnostics.py`). When a measurement is missing, extend them generally;
  do not write one-off probes for a single run.
- When stuck, research the literature, documentation and issue trackers online.
- Then compare with the reference engines MuJoCo, Newton and Genesis (sources under
  `.nuka_cache/engine-review/`, runnable packages in `/root/nuka-physics-reference-20260909`):
  find how each implements the same mechanism and record the concrete implementation differences.
- Only after an implementation difference is confirmed with evidence, examine the architectural
  differences behind it (data layout, stage order, solver coupling, contact representation).
- Adopt nothing by guess; every change stays on the one general solving path.
