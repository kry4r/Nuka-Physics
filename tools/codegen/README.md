# Codegen Pipeline

The row codegen pipeline is reserved by Phase 1 and implemented in Phase 2.
Generated CUDA files are contracts between the row IR and the runtime; they are
regenerated, not hand-edited.

## Source of Truth

Row IR lives at:

```text
tools/codegen/schema/row_v0_1.yaml
```

Phase 2 creates this schema for the four v0.1 base row classes:
`MaximalContactRow`, `MaximalJointRow`, `MaximalDriveRow`, and
`FeatherstoneContactRow`.

## Regeneration

Regenerate all codegen outputs with:

```bash
python tools/codegen/regen.py
```

Phase 2 creates `tools/codegen/regen.py`; until then this command is the
reserved interface.

## DO-NOT-EDIT Header

Every generated `.cu` / `.cuh` file starts with:

```cpp
// ============================================================
// GENERATED — DO NOT EDIT
// Source: <relative path to IR yaml>
// Regenerate: python tools/codegen/regen.py
// ============================================================
```

AI agents and human contributors must change the IR or generator, then rerun
the regeneration command. Direct edits to generated files are rejected by the
project workflow.

## Build Wiring

CMake uses `add_custom_command` to regenerate kernels whenever the row IR YAML
changes. The generated sources are then compiled as normal CUDA inputs.
