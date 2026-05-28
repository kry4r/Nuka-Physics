# Nuka Physics v0.1 – Phase 2: Row IR Schema + Codegen Skeleton

> **Master plan reference:** §3 Round 4 (Row format) + §4 Universal Row IR + §5.4 AI boundary
> **Prerequisites:** Phase 1 complete (V5 guardrails operational)
> **Blocks:** Phase 5 (CSR Row Migration) — migration cannot start until codegen produces stub kernels
> **Exit criteria gate:** v0.1
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

Build the codegen pipeline that consumes Row IR YAML and produces CUDA forward kernel stubs. This phase ships the **structural backbone** — the row IR schema for the four base row classes (Contact, Joint, Drive, FeatherstoneContact) plus a codegen tool that emits forward-only stub kernels matching existing PGS behavior on trivial scenes.

**Adjoint generation is explicitly deferred to v0.5 Phase 1.** This phase delivers forward only.

The deliverable validates: (a) IR schema is expressive enough for v0.1 row catalog, (b) codegen toolchain works end-to-end (YAML → CUDA → compiled binary), (c) the CMake build hooks regenerate on YAML change.

## Tech Stack

- Python 3.10+ + Jinja2 (codegen template engine)
- PyYAML (IR loading)
- CMake `add_custom_command` (regeneration on YAML change)
- CUDA 12+ (existing toolchain)

## Files to Create

- `tools/codegen/schema/row_v0_1.yaml` — IR schema definition (meta-schema, declares the shape of row class entries)
- `tools/codegen/classes/maximal_contact.yaml` — IR for `MaximalContactRow`
- `tools/codegen/classes/maximal_joint.yaml` — IR for `MaximalJointRow`
- `tools/codegen/classes/maximal_drive.yaml` — IR for `MaximalDriveRow`
- `tools/codegen/classes/featherstone_contact.yaml` — IR for `FeatherstoneContactRow`
- `tools/codegen/regen.py` — main codegen entry point
- `tools/codegen/templates/forward_kernel.cu.j2` — Jinja2 template for forward kernel
- `tools/codegen/templates/row_dispatch.cu.j2` — dispatch table template
- `tools/codegen/templates/registry_header.hpp.j2` — generated header listing row classes + constants
- `src/codegen/generated/.gitkeep` — placeholder directory for generated outputs
- `tests/codegen/test_codegen_roundtrip.cpp` — sanity test: regenerate + compile + link

## Files to Modify

- `CMakeLists.txt` — wire `add_custom_command` so changes to `tools/codegen/classes/*.yaml` re-run `regen.py`
- `src/CMakeLists.txt` — include generated sources under `src/codegen/generated/`
- `tools/codegen/README.md` — populate the regeneration section (Phase 1 left a placeholder)
- `tools/lint/banned_patterns.yaml` — confirm DO-NOT-EDIT header regex enforced on `src/codegen/generated/**`

## Tasks

### Task 2.1 — Define the meta-schema

`tools/codegen/schema/row_v0_1.yaml`:

```yaml
# Meta-schema for Universal Row IR v0.1. Validates row class definitions.
schema_version: "0.1"

required_fields:
  - row_class_id      # uint32, unique
  - row_class_name    # string, CamelCase
  - body_count_mode   # enum: fixed | variable
  - body_count        # int (if fixed) or null
  - jacobian_kind     # enum: maximal_6vec | featherstone_chain_scalar | mixed
  - max_rows_per_block # int, default 6
  - supports_friction # bool
  - supports_compliance # bool, default true
  - default_gradient_mode  # enum: dense_adjoint | stop_grad_on_event | ift_at_convergence | none
  - default_recompute_mode # enum: tape | checkpoint | recompute_always
  - forward_evaluator: # codegen template hook
      input_fields: [list of row fields read]
      output_fields: [list of row fields written]
      body_index_field: body_list_offset
      jacobian_field: jacobian_offset
  - constraint_kind   # enum: equality | unilateral | unilateral_with_friction

flags_bitfield:
  - Equality
  - Unilateral
  - Friction
  - Coupled
  - GradActive
```

### Task 2.2 — Author the four base row class IRs

Each `classes/*.yaml` follows the meta-schema. Example for `maximal_contact.yaml`:

```yaml
row_class_id: 0
row_class_name: MaximalContactRow

body_count_mode: fixed
body_count: 2

jacobian_kind: maximal_6vec
max_rows_per_block: 6     # 1 normal + up to 4 friction (cone polygon) + buffer

supports_friction: true
supports_compliance: false   # v0.1: hard contact only; XPBD compliance comes with XPBD rows in v0.7
default_gradient_mode: stop_grad_on_event   # β contact discontinuity, master plan Round 5
default_recompute_mode: checkpoint
constraint_kind: unilateral_with_friction

forward_evaluator:
  input_fields: [body_a, body_b, jacobian_linear_a, jacobian_angular_a,
                 jacobian_linear_b, jacobian_angular_b, rhs, lower, upper,
                 lambda, effective_mass]
  output_fields: [lambda, position_error]
  body_index_field: body_list_offset
  jacobian_field: jacobian_offset

  # Codegen template substitutions
  inner_loop_strategy: "row_serial_jacobi"   # template selects body-segmented update
  warm_start: true
  friction_pyramid_sides: 4
```

Similar concise IR for `maximal_joint.yaml` (equality, 1–6 rows depending on joint type), `maximal_drive.yaml` (1 row, PD-controlled equality), `featherstone_contact.yaml` (1 maximal body + 1 articulation link; `jacobian_kind: mixed`).

### Task 2.3 — Implement `tools/codegen/regen.py`

Pipeline:

1. Load meta-schema; validate each `classes/*.yaml` against it. Fail with line numbers on schema violation.
2. Topologically sort row classes by dependencies (none in v0.1; reserved for future).
3. For each row class, instantiate `templates/forward_kernel.cu.j2`:
   - Inputs: row class IR.
   - Output: `src/codegen/generated/<class_name>_forward.cu`.
   - File header: DO-NOT-EDIT block (Phase 1 convention) + source IR path + regeneration command.
4. Emit `src/codegen/generated/row_dispatch.cu` from `row_dispatch.cu.j2` — a `switch(row_class_id)` dispatch into per-class evaluator.
5. Emit `src/codegen/generated/row_class_registry.hpp` from `registry_header.hpp.j2` — compile-time constants (row class IDs, name strings, max rows).

Stub semantics for v0.1: forward kernels reproduce existing PGS solver step's row update logic but route through the new IR-derived buffer layout. **Behavior parity with existing solver is required**, not new physics.

### Task 2.4 — Author the Jinja2 templates

`templates/forward_kernel.cu.j2` (skeleton):

```cuda
// =====================================================
// GENERATED — DO NOT EDIT
// Source: {{ source_yaml_path }}
// Regenerate: python tools/codegen/regen.py
// =====================================================

#include "codegen/generated/row_class_registry.hpp"
#include "math/vec3.hpp"
#include "solver/gpu/cuda_constraint_solver.cuh"

namespace nuka::solver::generated {

__global__ void {{ class_name | snake_case }}_forward_kernel(
    const Row* __restrict__ rows,
    {% for f in forward_evaluator.input_fields %}
    const {{ field_cuda_type(f) }}* __restrict__ {{ f }}_data,
    {% endfor %}
    {% for f in forward_evaluator.output_fields %}
    {{ field_cuda_type(f) }}* __restrict__ {{ f }}_out,
    {% endfor %}
    uint32_t row_count)
{
    uint32_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= row_count) return;

    const Row& row = rows[row_idx];
    if (row.row_class_id != {{ row_class_id }}) return;

    // === Behavior-equivalent to legacy PGS step ===
    {% if inner_loop_strategy == "row_serial_jacobi" %}
    // Body-segmented Jacobi update (placeholder for v0.1 — full island/coloring in Phase 5)
    // ...
    {% endif %}

    // Friction projection (v0.1: 4-side polygon)
    {% if supports_friction %}
    // ...
    {% endif %}
}

} // namespace nuka::solver::generated
```

The template emits placeholder bodies in v0.1; full PGS-equivalent implementations are filled in Phase 5 (CSR Row Migration) when migration happens with the diff-test bridge.

### Task 2.5 — CMake regeneration hook

`CMakeLists.txt`:

```cmake
file(GLOB ROW_IR_FILES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/tools/codegen/classes/*.yaml")
file(GLOB ROW_TEMPLATES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/tools/codegen/templates/*.j2")

add_custom_command(
    OUTPUT
        ${CMAKE_SOURCE_DIR}/src/codegen/generated/row_class_registry.hpp
        ${CMAKE_SOURCE_DIR}/src/codegen/generated/row_dispatch.cu
        # listed explicitly per class:
        ${CMAKE_SOURCE_DIR}/src/codegen/generated/maximal_contact_forward.cu
        ${CMAKE_SOURCE_DIR}/src/codegen/generated/maximal_joint_forward.cu
        ${CMAKE_SOURCE_DIR}/src/codegen/generated/maximal_drive_forward.cu
        ${CMAKE_SOURCE_DIR}/src/codegen/generated/featherstone_contact_forward.cu
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/codegen/regen.py
    DEPENDS ${ROW_IR_FILES} ${ROW_TEMPLATES}
            ${CMAKE_SOURCE_DIR}/tools/codegen/regen.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Regenerating row class CUDA kernels from IR"
)

add_custom_target(nuka_codegen ALL
    DEPENDS ${CMAKE_SOURCE_DIR}/src/codegen/generated/row_dispatch.cu
)
```

### Task 2.6 — Round-trip sanity test

`tests/codegen/test_codegen_roundtrip.cpp`:

- Test 1: `regen.py` exits 0 with current IR set.
- Test 2: Generated `.cu` files contain the DO-NOT-EDIT header.
- Test 3: Generated kernels compile (NVCC, against existing `Row` struct from Phase 5 — for v0.1, use a stub `Row` struct in a header until Phase 5 lands the real one).
- Test 4: Modifying a YAML field triggers regeneration on next build.
- Test 5: Lint accepts the generated files (DO-NOT-EDIT header satisfies the convention).

## Validation

- **Generated kernels behave equivalent to legacy PGS** on a trivial 2-body contact scene (this is a placeholder validation; full diff-test bridge comes in Phase 5).
- **CMake re-runs `regen.py`** on YAML edit but not on unrelated edits — verify by timestamp.
- **Schema rejection works**: introduce a malformed IR (e.g., missing `row_class_id`); `regen.py` must fail with line-numbered error.
- **All four row class IRs validate.**

## Exit Criteria for Phase 2

1. `tools/codegen/schema/row_v0_1.yaml` defines the meta-schema for v0.1.
2. Four IR files exist under `tools/codegen/classes/` and validate against the meta-schema.
3. `tools/codegen/regen.py` produces forward kernel stubs for all four classes, plus dispatch + registry.
4. Generated files carry the DO-NOT-EDIT header and are placed under `src/codegen/generated/`.
5. CMake automatically re-runs codegen when YAML changes.
6. Round-trip test passes.
7. `tools/codegen/README.md` (created skeleton in Phase 1) is now fully populated with operational instructions.

## What This Phase Does Not Do

- **No adjoint kernel generation.** Adjoint codegen ships in v0.5 Phase 1.
- **No PGS rewrite.** Generated kernels are stubs / placeholders; the actual migration of existing solver work happens in Phase 5 with the diff-test bridge.
- **No new row classes beyond the four base ones.** XPBD / PBF / SDF rows come in v0.7.
- **No island/coloring inside kernels.** Pure per-row update for now; coloring scheduler comes in Phase 5.
