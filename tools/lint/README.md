# Physics-Smell Lint

`tools/lint/physics_smell.py` enforces the CUDA physics guardrails from the
v0.1 master plan. It scans scoped source files for deterministic-physics and
ABI hazards before they become silent simulation debt.

## What It Blocks

The current pattern set lives in `tools/lint/banned_patterns.yaml`:

- `float_atomic_add`: `atomicAdd` in CUDA physics paths. Float atomics break the
  D1 strong determinism contract; legacy integer counters are explicitly
  allowlisted until the CSR row migration removes them.
- `hot_path_cuda_malloc`: `cudaMalloc` or `cudaMallocAsync` inside hot solver,
  constraint, or runtime GPU paths.
- `shared_mem_cross_row_class`: shared-memory dispatches marked as spanning
  multiple row classes.
- `exception_across_c_abi`: `throw` in C ABI implementation files.
- `stl_in_public_header`: STL containers or strings in public wrapper headers.
- `generated_do_not_edit_header`: generated codegen outputs under
  `src/codegen/generated/` must start with `// GENERATED — DO NOT EDIT`.

## Usage

```bash
python tools/lint/physics_smell.py
python tools/lint/physics_smell.py --pattern float_atomic_add
python tools/lint/physics_smell.py --files src/solver/gpu/cuda_constraint_solver.cu
python tools/lint/physics_smell.py --fix-allowlist
```

Violations print in compiler style:

```text
src/runtime/gpu/foo.cu:42: error: [float_atomic_add] atomicAdd on float/double breaks D1 strong determinism
```

Exit code is `0` when clean and `1` when any non-allowlisted violation is found.

## Allowlist

`tools/lint/allowlist.yaml` is only for documented migration debt or validation
code paths. Every entry must include:

- `path`
- `pattern`
- `line` or `match`
- `reason`

Do not use the allowlist to land new physics code. If a rule is too broad,
tighten the scope or regex and document the change in this README.
