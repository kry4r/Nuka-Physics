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
