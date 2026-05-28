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

The generator validates the meta-schema and every row class YAML before writing
outputs. Schema errors print compiler-style diagnostics with the YAML path and
line number.

Generated outputs are:

```text
src/codegen/generated/maximal_contact_forward.cu
src/codegen/generated/maximal_joint_forward.cu
src/codegen/generated/maximal_drive_forward.cu
src/codegen/generated/featherstone_contact_forward.cu
src/codegen/generated/row_dispatch.cu
src/codegen/generated/row_class_registry.hpp
```

Use alternate paths only for tests:

```bash
python tools/codegen/regen.py \
  --classes-dir /tmp/row-ir \
  --output-dir /tmp/generated
```

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

CMake exposes the `nuka_codegen` target and wires it into the default build.
Changes to `tools/codegen/classes/*.yaml`, the schema, templates, or
`tools/codegen/regen.py` regenerate the files above. The generated CUDA sources
compile into `nuka_codegen_generated`, and `nuka_codegen_test` verifies the
round trip.

Useful validation commands:

```bash
cmake --build build --target nuka_codegen
ctest --test-dir build -R CodegenRoundtrip --output-on-failure
python tools/lint/physics_smell.py
```
