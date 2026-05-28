# Nuka Physics v0.1 – Phase 1: AI Agent Guardrails (V5 Layer)

> **Master plan reference:** §3 Round 11 (V5) + §5.4 AI agent boundary + §5.5 physics-smell lint
> **Prerequisites:** None — this is the first phase of the project
> **Blocks:** All subsequent phases (no code generation or row IR work begins until this lands)
> **Exit criteria gate:** v0.1
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

Build the V5 validation layer **before any other engineering work begins**. Solo + AI workflow has a known failure mode: AI agents produce code that compiles and passes naive tests but is physically wrong, with errors accumulating silently over months. This phase exists to enforce the rules that prevent silent rot.

Deliverables: a CI-enforced physics-smell lint, codegen conventions (DO-NOT-EDIT header + IR location + regeneration command documented), and an explicit AI-protected file list that AI agents must respect.

## Tech Stack

- Python 3.10+ (lint script)
- CMake (build target integration)
- GitHub Actions (CI hook); pre-commit fallback for local development
- ripgrep / regex (pattern matching)

## Files to Create

- `tools/lint/physics_smell.py` — the lint scanner
- `tools/lint/README.md` — what the lint does, how to extend, allowlist mechanism
- `tools/lint/banned_patterns.yaml` — declarative pattern list (data, not code)
- `tools/lint/allowlist.yaml` — per-file pattern allowances (for reference / validation code paths)
- `tools/codegen/README.md` — codegen pipeline overview, regeneration command, DO-NOT-EDIT header convention
- `docs/architecture/ai-agent-boundary.md` — the 6 categories of files AI agents must not modify (per master plan §5.4)
- `.github/workflows/lint.yml` — CI workflow running the lint on PR
- `scripts/pre-commit-lint.sh` — optional local hook

## Files to Modify

- `CMakeLists.txt` — add custom target `nuka_run_lint` invoking `tools/lint/physics_smell.py`
- `README.md` — add brief "Contributing / Lint" section pointing to `tools/lint/README.md`

## Tasks

### Task 1.1 — Author the banned-patterns YAML

`tools/lint/banned_patterns.yaml`:

```yaml
patterns:
  - id: float_atomic_add
    description: "atomicAdd on float/double breaks D1 strong determinism"
    regex: '\batomicAdd\s*\(\s*[^,)]+,\s*[^,)]+\s*\)'
    scope: physics_path     # see scope_definitions below
    severity: error

  - id: hot_path_cuda_malloc
    description: "cudaMalloc / cudaMallocAsync in per-step kernels = hot-path allocation"
    regex: '\b(cudaMalloc|cudaMallocAsync)\s*\('
    scope: hot_path
    severity: error

  - id: shared_mem_cross_row_class
    description: "__shared__ memory spanning multiple row classes within one dispatch is a determinism risk"
    regex: '__shared__\s+\w+\s+\w+\[.*\];.*//\s*ROW_CLASS:\s*\*'
    scope: physics_path
    severity: error

  - id: exception_across_c_abi
    description: "throw across extern \"C\" boundary is undefined behavior"
    regex: 'throw\s+'
    scope: c_abi_files
    severity: error

  - id: stl_in_public_header
    description: "std::vector / std::string / std::map etc. in public C++20 wrapper headers fragments ABI"
    regex: '\b(std::vector|std::string|std::map|std::unordered_map)\b'
    scope: public_headers
    severity: error

  - id: production_cpu_sim
    description: "CPU physics simulation forbidden in production code path (master plan §5.6 / decision #35)"
    regex: '\b(phi::PhysicsBackend::CpuReference|BackendSelectionPolicy::ForceCpuReference|allow_cpu_reference_validation\s*=\s*true)\b'
    scope: production_path
    severity: error

scope_definitions:
  physics_path:
    include:
      - "src/solver/**/*.cu"
      - "src/solver/**/*.cuh"
      - "src/constraint/**/*.cu"
      - "src/runtime/**/*.cu"
    exclude:
      - "src/**/reference/**"
      - "tests/**"

  hot_path:
    include:
      - "src/solver/**/*.cu"
      - "src/constraint/**/*.cu"
      - "src/runtime/gpu/**/*.cu"
    exclude:
      - "src/runtime/world_builder.cpp"
      - "src/import/**"

  c_abi_files:
    include:
      - "src/c_abi/**/*.cpp"
      - "src/c_abi/**/*.hpp"

  public_headers:
    include:
      - "src/include/nuka/**/*.hpp"
      - "src/include/nuka/**/*.h"

  production_path:
    include:
      - "src/**/*.cpp"
      - "src/**/*.cu"
      - "src/**/*.hpp"
      - "src/**/*.cuh"
      - "src/**/*.h"
    exclude:
      - "src/**/reference/**"
      - "src/**/*reference*.cpp"
      - "src/**/*reference*.hpp"
      - "tests/**"
      - "tools/oracle/**"
```

### Task 1.2 — Implement `tools/lint/physics_smell.py`

Behavior:

1. Load `banned_patterns.yaml` and `allowlist.yaml`.
2. For each pattern, compile regex + resolve scope file list (glob-include minus glob-exclude).
3. Scan each in-scope file line-by-line, recording `file:line:pattern_id:matched_text`.
4. Subtract allowlist entries.
5. Print violations as `<file>:<line>: error: [<pattern_id>] <description>` (compiler-style; editors jump to file:line).
6. Exit code 0 on clean, 1 on any violation.

Performance: full scan of `src/**` should complete in under 2 seconds on a 16-core dev machine. Cache compiled regex globally.

CLI:

```
python tools/lint/physics_smell.py            # scan everything
python tools/lint/physics_smell.py --pattern float_atomic_add
python tools/lint/physics_smell.py --files src/solver/gpu/cuda_constraint_solver.cu
python tools/lint/physics_smell.py --fix-allowlist  # interactively add to allowlist
```

### Task 1.3 — CI integration

`.github/workflows/lint.yml`:

```yaml
name: lint
on: [push, pull_request]
jobs:
  physics_smell:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install pyyaml
      - run: python tools/lint/physics_smell.py
```

`CMakeLists.txt` add:

```cmake
add_custom_target(nuka_run_lint
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/lint/physics_smell.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running physics-smell lint"
)
```

### Task 1.4 — Author `tools/codegen/README.md`

Document the codegen pipeline convention even before the codegen itself exists (Phase 2 will populate). Required content:

- Where row IR lives: `tools/codegen/schema/row_v0_1.yaml`
- Regeneration command: `python tools/codegen/regen.py` (script created in Phase 2; placeholder file with TODO acceptable here)
- DO-NOT-EDIT header convention: every generated `.cu` / `.cuh` starts with:

```
// ============================================================
// GENERATED — DO NOT EDIT
// Source: <relative path to IR yaml>
// Regenerate: python tools/codegen/regen.py
// ============================================================
```

- Build wiring: CMake `add_custom_command` regenerates kernels when YAML changes.

### Task 1.5 — Author `docs/architecture/ai-agent-boundary.md`

The 6 categories of files AI agents must not modify (per master plan §5.4):

1. `docs/plans/2026-05-28-nuka-physics-master-plan.md` (project constitution)
2. `tools/codegen/schema/*.yaml` (row IR — source of truth)
3. `tests/oracle/golden/**` (golden trajectories in Git LFS)
4. Numeric threshold constants (file-list in this doc; updated as the project grows)
5. `LICENSE`, `NOTICE`, `CLA.md`
6. `*.cu` / `*.cuh` files whose first line is `// GENERATED — DO NOT EDIT`

Add a section "How to request changes to protected files": owner-only PR, explicit rationale, single-purpose change.

### Task 1.6 — `agent.md` boundary section

Append to repository `agent.md`:

```
## AI-Protected Files

See docs/architecture/ai-agent-boundary.md. Do not edit files in these categories
without explicit owner request. If you believe an AI-protected file needs to change,
surface the proposal in a comment for the human owner rather than editing.
```

This makes future AI sessions (including new Claude instances) aware of the boundary on first read of `agent.md`.

## Validation

- **Self-test:** introduce a deliberate `atomicAdd<float>` in a sandbox file → lint must fail with the right `file:line:pattern_id`.
- **Performance:** lint full repo finishes < 2 s on dev machine.
- **CI:** push a PR with a violation; CI must block.
- **DO-NOT-EDIT discipline:** create a sample generated `.cu` with the header; document the regeneration step.

## Exit Criteria for Phase 1

All five must hold:

1. `python tools/lint/physics_smell.py` exits 0 on current repo (no false positives in existing code, even if some legacy code needs allowlisting — allowlist entries must be explicit and documented).
2. CI runs the lint on every PR; a deliberate violation blocks merge.
3. `tools/codegen/README.md` exists with the DO-NOT-EDIT convention, regeneration command, and IR location documented (Phase 2 will populate the actual codegen).
4. `docs/architecture/ai-agent-boundary.md` exists and is referenced from `agent.md`.
5. CMake `nuka_run_lint` target builds locally.

## What This Phase Does Not Do

- Does **not** implement codegen itself (Phase 2).
- Does **not** generate any CUDA kernels.
- Does **not** modify any existing physics code beyond adding the lint hook.
- Does **not** create the row IR YAML schema (Phase 2 — but the file path is reserved).

This is the discipline phase. Build the guardrails first; only then build the things they guard.
