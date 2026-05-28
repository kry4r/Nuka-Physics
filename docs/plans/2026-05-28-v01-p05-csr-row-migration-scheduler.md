# Nuka Physics v0.1 – Phase 5: CSR Universal Row Migration + Island/Coloring Scheduler

> **Master plan reference:** §3 Round 4 (Row format) + §3 Round 6 (D1 determinism) + §4 Row IR + §5 hard constraints
> **Prerequisites:** Phase 1 (lint), Phase 2 (codegen skeleton with stub kernels), Phase 3 (DeviceContext)
> **Blocks:** Phase 6 (Featherstone needs the row format) and v0.1 exit
> **Exit criteria gate:** v0.1
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

This is the largest single phase in v0.1. Two interlocked deliverables that **must land together**:

1. **Rewrite `ConstraintBlock` as the CSR-like Universal Row format** documented in master plan §4. Migrate the existing Contact / Joint / Drive paths to operate on the new format through codegen-produced kernels, using a **diff-test bridge** to verify behavior parity with the legacy PGS solver before retiring the old code.

2. **Implement the island / graph-coloring scheduler** required by D1 strong determinism. Without coloring, the new row solver cannot avoid float atomic adds; without atomic adds, parallel row updates collide on shared bodies. Coloring is the v1 must-have per master plan §3 Round 6.

Critical constraint: legacy `cuda_constraint_solver.cu` (1244 lines) is retired only after every diff-test passes within tolerance for the smoke scene set (Phase 4).

## Tech Stack

- C++20
- CUDA 12+
- Existing solver / constraint module structure (rewritten contents)
- Phase 2 codegen pipeline (kernels regenerated from IR)
- Phase 4 V2 invariants (used by diff-test verification)

## Files to Create

- `src/constraint/row.hpp` — the new Row struct (replaces ConstraintBlock)
- `src/constraint/row_buffers.hpp` — CSR buffer layout (rows, body_list, jacobian)
- `src/constraint/row_buffers.cpp`
- `src/constraint/row_builder.hpp` — contact/joint/drive → Row converter
- `src/constraint/row_builder.cpp`
- `src/solver/gpu/row_scheduler.hpp` — island detection + graph coloring + dispatch
- `src/solver/gpu/row_scheduler.cu` — CUDA implementation of coloring
- `src/solver/gpu/row_scheduler.cuh`
- `src/solver/gpu/row_solver.cu` — top-level solver loop using codegen kernels
- `src/solver/gpu/row_solver.cuh`
- `src/solver/diff_test_bridge.hpp` — runs legacy + new in parallel, asserts diff < tolerance
- `src/solver/diff_test_bridge.cpp`
- `tests/solver/test_csr_row_layout.cpp`
- `tests/solver/test_island_coloring.cpp` — verify deterministic coloring + no shared-body races
- `tests/solver/test_diff_test_bridge_smoke.cpp`
- `tests/solver/test_diff_test_bridge_4096env_batch.cpp`

## Files to Modify

- `src/constraint/constraint_block.hpp` — deprecated; kept temporarily for diff-test bridge only; deleted at phase exit
- `src/constraint/contact_builder.{hpp,cpp}` — output Row format directly (via row_builder)
- `src/runtime/articulation/joint_constraints.{hpp,cpp}` — output Row format
- `src/runtime/world_stepper.cpp` — invoke new row solver
- `src/solver/rigid_solver.{hpp,cpp}` — adapter that owns the row scheduler
- `src/solver/gpu/cuda_constraint_solver.cu` — kept during diff-test; removed at phase exit
- `tools/codegen/templates/forward_kernel.cu.j2` — implementations now real (not stubs), matching legacy PGS exactly
- `src/CMakeLists.txt`

## Tasks

### Task 5.1 — Define the Row struct

`src/constraint/row.hpp`:

```cpp
#pragma once
#include <cstdint>

namespace nuka::constraint {

// Universal Row IR — master plan §4. One layout for every constraint type.
struct Row {
    uint32_t row_class_id;      // dispatch into evaluator (see row_class_registry)
    uint32_t body_count;        // CSR neighborhood size
    uint32_t body_list_offset;  // index into RowBuffers::body_indices
    uint32_t jacobian_offset;   // index into RowBuffers::jacobian_data
    float    rhs;               // residual
    float    lambda;            // accumulated multiplier (warm start + tape entry)
    float    lower;             // bound
    float    upper;             // bound
    float    compliance_alpha;  // XPBD α; 0 → hard
    float    damping_beta;      // Baumgarte / XPBD damping
    uint16_t flags;             // bitfield: Equality(1) | Unilateral(2) | Friction(4) | Coupled(8) | GradActive(16)
    uint16_t adjoint_kernel_id; // reverse evaluator id (v0.5)
    uint8_t  gradient_mode;     // 0 dense | 1 stop_grad_event | 2 ift_at_conv | 3 none
    uint8_t  recompute_mode;    // 0 tape | 1 checkpoint | 2 recompute_always
    uint8_t  event_flag_field;  // bit position; stop-grad signal
    uint8_t  contact_softness;  // 0 hard | 1 soft_alpha
};

namespace row_flags {
    constexpr uint16_t Equality   = 1u << 0;
    constexpr uint16_t Unilateral = 1u << 1;
    constexpr uint16_t Friction   = 1u << 2;
    constexpr uint16_t Coupled    = 1u << 3;
    constexpr uint16_t GradActive = 1u << 4;
}

} // namespace nuka::constraint
```

### Task 5.2 — CSR buffer layout

`src/constraint/row_buffers.hpp`:

```cpp
namespace nuka::constraint {

// Device-side CSR buffer layout for the entire batch of envs.
// Each env contributes a contiguous slice of rows; row scheduler operates per-env.
struct RowBuffers {
    // Per-row records (SoA of Row above is also possible; AoS for v0.1 simplicity)
    Row*       rows;             // device buffer
    uint32_t   row_count;

    // CSR body indices: rows[i] references body_indices[rows[i].body_list_offset..+rows[i].body_count]
    uint32_t*  body_indices;
    uint32_t   body_index_count;

    // CSR Jacobian data: depends on row_class_id.
    //   For row_class_id == MaximalContactRow / MaximalJointRow / MaximalDriveRow:
    //     6-vec per body: [lin.x lin.y lin.z ang.x ang.y ang.z]
    //   For FeatherstoneContactRow:
    //     mixed layout (see codegen template featherstone_contact_forward.cu)
    float*     jacobian_data;
    uint32_t   jacobian_data_count;
};

// Per-env partitioning so the scheduler can do per-env work
struct RowBufferPartition {
    uint32_t row_offset;
    uint32_t row_count;
    uint32_t env_id;
};

} // namespace nuka::constraint
```

### Task 5.3 — Row builder: contact / joint / drive → Row

`src/constraint/row_builder.{hpp,cpp}` consumes the existing `ContactManifold`, joint constraint, and drive structures and emits `Row` records into `RowBuffers`. Each existing builder file (`contact_builder.cpp`, `joint_constraints.cpp`) gains a `BuildRows(...)` entry point.

Determinism: builder must produce identical row ordering across runs (no atomic appends; per-env prefix-sum then write).

### Task 5.4 — Graph coloring scheduler

`src/solver/gpu/row_scheduler.cu`:

Algorithm: greedy graph coloring at the row level.
- Each row "writes to" the bodies in its `body_list`.
- Two rows conflict if their write sets intersect.
- Color k = set of rows that pairwise do not conflict; each color is processed in one parallel batch (race-free).

Implementation:

1. Build a body→row index (which rows touch each body) — segment scan.
2. For each row, compute a candidate color = `max(color[neighbor_row] for neighbor_row sharing any body) + 1`.
3. Iterate until stable (Luby's algorithm variant for parallel graph coloring).
4. Output: for each color c, a list of row indices to process in parallel.

For v0.1 the row graph is small (Contact + Joint + Drive only); typical color count ~5-8 per scene. Document the worst-case bound.

Required property: **same scene → same coloring** (deterministic). Use stable ordering (row index ascending) as tie-break.

### Task 5.5 — Row solver (top-level loop)

`src/solver/gpu/row_solver.cu`:

```cpp
void RowSolver::Solve(const phi::DeviceContext& ctx, RowBuffers& buffers,
                      BodyState& bodies, const SolveConfig& cfg)
{
    // 1. Build color partitions (Task 5.4)
    ColorPartitions cp = scheduler_.PartitionColors(buffers);

    // 2. Iterate PGS-style; for each iteration, sweep colors sequentially.
    for (uint32_t it = 0; it < cfg.iterations; ++it) {
        for (uint32_t color = 0; color < cp.num_colors; ++color) {
            const auto& rows_in_color = cp.color_rows[color];
            // Launch dispatch kernel: switch on row_class_id, call codegen evaluator
            nuka_dispatch_forward_kernel<<<grid, block, 0, ctx.stream.Native()>>>(
                buffers.rows, rows_in_color.data(), rows_in_color.size(),
                bodies, /* other state */);
            // No barrier between rows of same color (they don't conflict by coloring).
        }
    }
}
```

The dispatch kernel (`row_dispatch.cu`, generated in Phase 2) reads `row_class_id` and calls the per-class evaluator. Per-class evaluators implement PGS-equivalent updates (Phase 2 left these as stubs; Phase 5 fills them in to match legacy `cuda_constraint_solver.cu`).

### Task 5.6 — Diff-test bridge

`src/solver/diff_test_bridge.{hpp,cpp}`:

Runs both solvers (legacy `CudaConstraintSolver` and new `RowSolver`) on the same input, compares output body state, asserts diff under tolerance.

```cpp
class DiffTestBridge {
public:
    explicit DiffTestBridge(const phi::DeviceContext& ctx);

    // Solve via both paths from identical input; assert agreement.
    void Solve(BodyState& bodies_in_out, const ConstraintInputs& inputs,
               float tolerance_pos = 1e-5f, float tolerance_vel = 1e-4f);

    // Inspect divergence per body (for debugging)
    struct Divergence { uint32_t body_id; float pos_err; float vel_err; };
    std::vector<Divergence> LastDivergence() const;

private:
    LegacyCudaConstraintSolver legacy_;
    RowSolver new_;
};
```

Gating: every smoke scene runs through the bridge. Phase exit is blocked if any smoke scene diverges over tolerance.

### Task 5.7 — Test suite

- `test_csr_row_layout.cpp`: build a 100-row CSR buffer, verify body_indices and jacobian_data slice correctly per row.
- `test_island_coloring.cpp`: random row graphs; verify (a) no color contains conflicting rows, (b) coloring deterministic across runs.
- `test_diff_test_bridge_smoke.cpp`: run all three Phase 4 smoke scenes; assert legacy == new within tolerance.
- `test_diff_test_bridge_4096env_batch.cpp`: run a batched 4096-env scene; assert per-env divergence < tolerance; this catches batching-related bugs.

### Task 5.8 — Retirement of legacy code

Once all diff-tests pass:

- Delete `src/solver/gpu/cuda_constraint_solver.cu` and its `.cuh`.
- Delete `src/constraint/constraint_block.hpp` (Row replaces it).
- Update `src/runtime/world_stepper.cpp` to use `RowSolver` directly.
- Run full test suite + V2 smoke scenes; all green.

This deletion is the symbolic phase exit: the engine has migrated.

## Validation

- All three Phase 4 smoke scenes diff-test agree within tolerance.
- Energy invariant (V2) still holds within thresholds — proves the new solver is at least as good as legacy.
- Determinism check: same scene + same seed → bit-exact output across runs (D1 contract; coloring is deterministic; no float atomics in the new path).
- Performance check: new path within 20% of legacy step time for the same scene (10-20% indirection cost is the accepted CSR tax per master plan Round 4).

## Exit Criteria for Phase 5

1. `Row` struct + `RowBuffers` + `row_builder` produce identical row content for current contact / joint / drive inputs as legacy code does internally.
2. Graph coloring scheduler operational; no row in a single color writes to a body another row in the same color writes to.
3. RowSolver produces bit-exact output across two runs on same input + GPU + driver (D1 strong).
4. Diff-test bridge proves agreement on all Phase 4 smoke scenes.
5. Legacy `cuda_constraint_solver.cu` and `constraint_block.hpp` deleted.
6. Step time within 20% of legacy for same scene.
7. Lint passes (no float atomics in new code).

## What This Phase Does Not Do

- No adjoint kernels (v0.5 Phase 1).
- No new row classes beyond the four base ones (v0.7).
- No XPBD compliance handling beyond IR field (no XPBD rows yet).
- No Featherstone-specific row class (Phase 6 lands `FeatherstoneContactRow` actual implementation).
