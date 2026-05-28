# Nuka Physics v0.5 – Phase 1: Codegen Adjoint Kernel Generation + V3 FD Validation

> **Master plan reference:** §3 Round 4 (codegen) + §3 Round 5 (diff-sim) + §6 V3 + §5.3 per-row 5 gates
> **Prerequisites:** v0.3 closed; v0.1 Phase 2 codegen skeleton; v0.1 Phase 5 row scheduler
> **Blocks:** v0.5 Phases 2–5
> **Exit criteria gate:** v0.5
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

Extend the codegen pipeline (v0.1 Phase 2 emitted forward-only stubs) to **also emit adjoint kernels** for every row class. Pair each with **V3 finite-difference validation** so every row class is provably differentiable within tolerance before it can ship. This is the core enabler of the rest of v0.5.

Deliverables:

1. **Adjoint codegen** — Jinja2 templates for reverse-mode kernels per row class, driven by IR fields `gradient_mode` and `adjoint_kernel_id`.
2. **Adjoint registry** — generated dispatch into the right adjoint by `row_class_id`.
3. **V3 finite-difference harness** — for each row class, automatically generate test cases that perturb parameters, run forward at `(x, p)` and `(x, p+δ)`, compute numerical Jacobian, compare to analytical adjoint output. Tolerance < 1e-3 relative.
4. **β stop-gradient at events** — the contact-event bit flow. When event_flag fires forward, adjoint zeros that contribution (per master plan Round 5 decision).
5. **CI gating** — V3 FD check runs on PR; any new row class must pass before merge.

## Tech Stack

- Python (codegen, FD harness)
- C++20 / CUDA 12+ (generated kernels, test harness)
- Existing Phase 2 Jinja2 infrastructure

## Files to Create

### Codegen extensions

- `tools/codegen/templates/adjoint_kernel.cu.j2` — adjoint kernel template
- `tools/codegen/templates/adjoint_dispatch.cu.j2`
- `tools/codegen/adjoint_emitter.py` — adjoint codegen logic (called from regen.py)
- `tools/codegen/derivative_rules.py` — per-IR-field derivative rules (e.g., friction cone projection's adjoint)

### V3 FD validation harness

- `src/codegen/v3_validation/fd_harness.hpp`
- `src/codegen/v3_validation/fd_harness.cu`
- `src/codegen/v3_validation/numerical_jacobian.cu`
- `tests/codegen/test_adjoint_fd_maximal_contact.cpp`
- `tests/codegen/test_adjoint_fd_maximal_joint.cpp`
- `tests/codegen/test_adjoint_fd_maximal_drive.cpp`
- `tests/codegen/test_adjoint_fd_featherstone_contact.cpp`
- `tools/codegen/generate_v3_tests.py` — auto-generate test stubs from row IR

### Row IR extensions

- Each `tools/codegen/classes/*.yaml` gains:
  - `adjoint_evaluator` block (input/output gradient mapping)
  - `derivative_overrides` (for ops without standard derivatives, e.g., friction cone)

## Tasks

### Task 5.1.1 — Extend IR schema for adjoint

`tools/codegen/schema/row_v0_1.yaml` adds:

```yaml
required_fields:
  - adjoint_evaluator:
      grad_input_fields: [lambda_grad, position_grad, velocity_grad]
      grad_output_fields: [body_force_grad, body_torque_grad, parameter_grads]
      reverse_dependencies: [...]   # which forward fields are needed by reverse
      derivative_rules:
        # Per non-trivial op, a closed-form derivative expression
        - op: friction_cone_projection
          rule: friction_cone_adjoint
        - op: stop_grad_on_event
          rule: zero_out_if_event_bit_set
```

### Task 5.1.2 — Author adjoint kernel template

`tools/codegen/templates/adjoint_kernel.cu.j2`:

```cuda
// =====================================================
// GENERATED — DO NOT EDIT
// Source: {{ source_yaml_path }}
// Adjoint of: {{ class_name | snake_case }}_forward_kernel
// =====================================================

#include "codegen/generated/row_class_registry.hpp"
#include "math/vec3.hpp"

namespace nuka::solver::generated {

__global__ void {{ class_name | snake_case }}_adjoint_kernel(
    const Row* __restrict__ rows,
    {% for f in adjoint_evaluator.grad_input_fields %}
    const float* __restrict__ {{ f }}_in,
    {% endfor %}
    {% for f in adjoint_evaluator.grad_output_fields %}
    float* __restrict__ {{ f }}_out,
    {% endfor %}
    {% for f in adjoint_evaluator.reverse_dependencies %}
    const {{ field_cuda_type(f) }}* __restrict__ {{ f }}_data,
    {% endfor %}
    uint32_t row_count)
{
    uint32_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= row_count) return;
    const Row& row = rows[row_idx];
    if (row.row_class_id != {{ row_class_id }}) return;

    // Stop-gradient on event (β strategy)
    {% if default_gradient_mode == 'stop_grad_on_event' %}
    if (row.flags & row_flags::Friction) {
        // Friction stick→slip transition: zero gradient on lambda for that row component
        if (row.event_flag_field & EVENT_FLAG_FRICTION_TRANSITION) {
            return;
        }
    }
    {% endif %}

    // Analytical adjoint body — pulled from derivative_rules.py templates
    {% include 'derivatives/' + class_name | snake_case + '_adjoint_body.cu.j2' %}
}

} // namespace nuka::solver::generated
```

### Task 5.1.3 — Derivative rules per row class

`tools/codegen/derivative_rules.py`:

For each row class, codify the closed-form derivative of the forward update. For PGS contact:

- Forward: `lambda_new = clamp(lambda + (rhs - J·v + b) / Aii, lower, upper)`
- Adjoint:
  - `dL/d(rhs)`     = `dL/d(lambda_new) * (lambda was unclamped at this step ? 1/Aii : 0)`
  - `dL/d(J·v)`     = `-dL/d(rhs)`
  - `dL/d(b)`       = `dL/d(rhs)`
  - `dL/d(body_velocity)` via row's body Jacobian (`J^T * dL/d(J·v)`)
  - friction cone projection: gradient flows along the active facet; zero if at boundary (stop-grad)

`derivative_rules.py` emits these expressions as C++/CUDA snippets into the template at codegen time.

### Task 5.1.4 — Featherstone ABA adjoint

The Featherstone forward dynamics has a structured adjoint (three reverse passes mirroring the forward three passes). Reference: "Efficient Computation of the Inverse Dynamics Gradient" (Singh et al.) and MJX's implementation.

`tools/codegen/classes/featherstone_contact.yaml` adjoint section:

```yaml
adjoint_evaluator:
  # Reverse Featherstone: 3 passes in reverse order
  reverse_passes:
    - name: pass3_reverse_accelerations
      reads: [link_articulated_inertia, link_bias_force, joint_acceleration_grad]
      writes: [link_acceleration_grad, joint_torque_grad]
    - name: pass2_reverse_articulated_inertia
      reads: [link_inertia, joint_motion_subspace, link_acceleration_grad]
      writes: [link_articulated_inertia_grad]
    - name: pass1_reverse_kinematics
      reads: [link_velocity_grad, joint_motion_subspace]
      writes: [joint_velocity_grad, joint_position_grad]
```

The adjoint emitter generates kernels for each reverse pass, called in reverse temporal order from `world_step_backward`.

### Task 5.1.5 — V3 FD harness

`src/codegen/v3_validation/fd_harness.hpp`:

```cpp
namespace nuka::codegen::v3 {

// For a given row class, perturb each input parameter by ±δ, call forward twice,
// compute (out(x+δ) - out(x-δ)) / (2δ) as numerical Jacobian column.
// Compare against analytical adjoint (vT·J via the adjoint kernel).
class RowClassFdValidator {
public:
    RowClassFdValidator(uint32_t row_class_id, const phi::DeviceContext& ctx);

    // Build random test cases for this row class
    void GenerateTestCases(uint32_t count, uint32_t seed);

    // Run validation; returns max relative error
    struct Result { float max_rel_err; uint32_t worst_case_idx; };
    Result Run(float delta = 1e-4f);

private:
    uint32_t row_class_id_;
    const phi::DeviceContext& ctx_;
    // device buffers for forward inputs, perturbation deltas, numerical & analytical grads
};

} // namespace nuka::codegen::v3
```

### Task 5.1.6 — Per-row-class FD tests

`tests/codegen/test_adjoint_fd_maximal_contact.cpp`:

```cpp
TEST(AdjointFd, MaximalContactRow_RelErrUnder1eM3) {
    auto ctx = nuka::phi::MakeDeviceContext(0, nullptr);
    nuka::codegen::v3::RowClassFdValidator v(ROW_CLASS_ID_MAXIMAL_CONTACT, ctx);
    v.GenerateTestCases(/*count=*/100, /*seed=*/42);
    auto r = v.Run(/*delta=*/1e-4f);
    EXPECT_LT(r.max_rel_err, 1e-3f)
        << "Adjoint diverges from FD at case " << r.worst_case_idx;
}
```

One test file per row class; CI runs them all on every PR touching `tools/codegen/` or `src/codegen/generated/`.

### Task 5.1.7 — Adjoint dispatch wiring

`tools/codegen/templates/adjoint_dispatch.cu.j2`:

```cuda
__global__ void row_dispatch_adjoint_kernel(const Row* rows, uint32_t row_count, ...) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= row_count) return;
    switch (rows[i].row_class_id) {
        {% for c in row_classes %}
        case {{ c.row_class_id }}:
            {{ c.class_name | snake_case }}_adjoint_per_row(rows[i], ...);
            break;
        {% endfor %}
    }
}
```

Solver invokes this kernel during backward pass.

### Task 5.1.8 — Update lint to enforce adjoint header

Generated adjoint kernels carry the same DO-NOT-EDIT header; `physics_smell` lint extends the check to `*_adjoint*.cu` files.

### Task 5.1.9 — Master plan §5.3 per-row 5 gates enforcement

For every row class to ship, code path must run:

1. ✅ Forward oracle (matches MJX / Bullet / Vellum / Flex — already enforced by V1).
2. **✅ Adjoint vs FD (this phase)**.
3. ✅ Determinism (run twice, bit-exact).
4. ✅ Energy invariant (V2 smoke).
5. ✅ Demo non-regression.

Build a `tools/v3/run_all_row_class_gates.sh` script that runs all 5 gates for every row class; CI uses it on PRs touching row IR.

## Validation

- For each of the 4 v0.1 row classes (Maximal Contact, Maximal Joint, Maximal Drive, Featherstone Contact), the V3 FD check passes within 1e-3 relative error on 100 random test cases.
- Generated adjoint kernels compile with the DO-NOT-EDIT header.
- Lint accepts new adjoint files.
- Adjoint dispatch correctly routes by `row_class_id`.
- `tools/v3/run_all_row_class_gates.sh` runs cleanly.

## Exit Criteria for v0.5 Phase 1

1. Codegen emits forward + adjoint kernels for all v0.1 row classes.
2. Each adjoint passes V3 FD check (< 1e-3 relative).
3. Adjoint dispatch wired into solver (Phases 2–4 will use it).
4. CI gates new row class merges on all 5 quality gates.
5. Stop-gradient on contact events (β strategy) operational in friction adjoint.
6. Featherstone three-pass adjoint operational and oracle-verified against MJX `qddot` gradients (vs analytical AD in MJX).

## What This Phase Does Not Do

- No tape / checkpointing yet (Phase 2).
- No PyTorch backward yet (Phase 4 — needs tape from Phase 2).
- No IFT path (Phase 3).
- No sim-to-real noise (Phase 4).
- No new row classes (v0.7).
