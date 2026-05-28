# AI Agent Boundary

The master plan defines files that AI agents may read but must not modify
without an explicit owner request. If one of these files needs to change, the
agent should stop and write a proposal for the human owner instead of editing.

## Protected Categories

1. `docs/plans/2026-05-28-nuka-physics-master-plan.md`, the project
   constitution.
2. `tools/codegen/schema/*.yaml`, the row IR source of truth once created.
3. `tests/oracle/golden/**`, Git LFS golden trajectories.
4. Numeric threshold constants. Current tracked list:
   - row oracle tolerances in future oracle test configs
   - energy invariant tolerances in future diagnostics configs
   - MJX / Pinocchio comparison tolerances in future oracle configs
5. `LICENSE`, `NOTICE`, and `CLA.md`.
6. Any `.cu` or `.cuh` file whose first line is:

   ```cpp
   // GENERATED — DO NOT EDIT
   ```

## How to Request Changes to Protected Files

Protected-file changes require an owner-only PR or explicit owner request. The
change must be single-purpose, include a rationale, and explain how downstream
generated files, tests, or golden artifacts are updated.
