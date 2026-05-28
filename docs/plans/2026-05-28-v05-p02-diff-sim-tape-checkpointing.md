# Nuka Physics v0.5 – Phase 2: Diff-Sim Tape + Gradient Checkpointing

> **Master plan reference:** §3 Round 5 (path 2 + path 3 hybrid) + §3 Round 13 (risk: long-episode convergence)
> **Prerequisites:** v0.5 Phase 1 (adjoint kernels exist per row class)
> **Blocks:** v0.5 Phase 4 (PyTorch backward needs tape)
> **Exit criteria gate:** v0.5
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

Implement gradient checkpointing (master plan path 2) as the diff-sim tape strategy. Memory cap dictates this: S1's 4096 envs × 1000-step episode × 10KB/step would be 40 GB of full tape, unfeasible. Checkpointing every K=50–100 steps reduces this to (1000/K)×state + K×tape ≈ 1 GB; reverse pass recomputes from nearest checkpoint at 2-3× forward cost.

Path 3 (IFT-at-convergence) for guaranteed-convergent subsystems lands in Phase 3 (cuDSS). This phase focuses on the universal checkpointing path.

Deliverables:

1. **Checkpoint storage** — per-env GPU-resident state snapshots at K intervals during forward rollout.
2. **Tape recording** — between checkpoints, lightweight per-step records of constraint-row lambda + active flags (enough to recompute physics deterministically).
3. **Recompute orchestrator** — reverse pass loads nearest checkpoint, replays forward to target step, runs adjoint kernel for that step, accumulates gradients.
4. **PyTorch-friendly API** — `world.step_with_tape()` and `world.backward_to_checkpoint()` exposed via C ABI + Python.
5. **Configurable K** — episode length / GPU memory trade-off knob.

## Tech Stack

- C++20 / CUDA 12+
- Existing solver / Row infrastructure
- D1 determinism (recompute path must reproduce forward bit-exact)

## Files to Create

- `src/diffsim/tape.hpp` — tape data structure
- `src/diffsim/tape.cpp`
- `src/diffsim/tape.cu`
- `src/diffsim/checkpoint.hpp` — checkpoint snapshot
- `src/diffsim/checkpoint.cu`
- `src/diffsim/recompute_orchestrator.hpp`
- `src/diffsim/recompute_orchestrator.cu`
- `src/diffsim/backward_runner.hpp` — top-level reverse pass
- `src/diffsim/backward_runner.cu`
- `src/include/nuka/nuka_diffsim.h` — C ABI extension for diff-sim
- `src/c_abi/diffsim.cpp`
- `tests/diffsim/test_tape_determinism.cpp` — replay produces bit-exact state
- `tests/diffsim/test_checkpoint_round_trip.cpp` — save + recompute = original
- `tests/diffsim/test_backward_short_episode.cpp` — 50-step backward agreement vs full-tape baseline
- `tests/diffsim/test_backward_long_episode_checkpoint.cpp` — 1000 step with K=100

## C ABI extension

```c
typedef struct nuka_tape_t* nuka_tape_handle;

typedef struct {
    uint32_t checkpoint_interval;     /* K */
    uint32_t max_tape_entries;        /* preallocated; default K */
    uint32_t max_checkpoints;         /* default episode_length / K */
    uint8_t  recompute_on_backward;   /* 1 = checkpointing, 0 = full tape (debug) */
} nuka_tape_desc_t;

nuka_result_t nuka_tape_create(nuka_world_handle world, const nuka_tape_desc_t* desc,
                               nuka_tape_handle* out);
void          nuka_tape_destroy(nuka_tape_handle tape);

/* Step with tape recording */
nuka_result_t nuka_world_step_with_tape(nuka_world_handle world, nuka_tape_handle tape);

/* Reverse pass over the recorded steps. grad_observations_in is per-step gradient. */
nuka_result_t nuka_tape_backward(nuka_tape_handle tape,
                                  const void* grad_observations_in,
                                  void* grad_actions_out,
                                  void* grad_parameters_out);

/* Reset (after backward) */
nuka_result_t nuka_tape_reset(nuka_tape_handle tape);
```

## Tasks

### Task 5.2.1 — Define tape data structure

`src/diffsim/tape.hpp`:

```cpp
namespace nuka::diffsim {

// Per-step recorded data: minimum needed to (a) replay forward identically,
// (b) run adjoint at this step given the replayed state.
struct TapeEntry {
    uint32_t step_index;
    // Constraint-row accumulator state
    float*   lambda_per_row;          // device pointer; sized to row_count of this step
    uint32_t row_count;
    // Active-event bits per row (for stop-grad)
    uint32_t* event_flags_per_row;
    // Per-env action input
    float*   actions;                  // env_count × action_dim
    // Solver iteration count (some subsystems iterate variably; record for replay)
    uint32_t solver_iterations;
};

class Tape {
public:
    Tape(const phi::DeviceContext& ctx, const nuka_tape_desc_t& desc);
    void RecordStep(const StepRecord& step);
    void RecordCheckpoint(const Checkpoint& cp);
    uint32_t StepCount() const noexcept { return entries_.size(); }
    uint32_t CheckpointCount() const noexcept { return checkpoints_.size(); }
    const TapeEntry& Entry(uint32_t i) const { return entries_[i]; }
    const Checkpoint& CheckpointAt(uint32_t i) const { return checkpoints_[i]; }

private:
    const phi::DeviceContext& ctx_;
    nuka_tape_desc_t desc_;
    DeviceVector<TapeEntry> entries_;
    DeviceVector<Checkpoint> checkpoints_;
};

} // namespace
```

### Task 5.2.2 — Checkpoint snapshot

`src/diffsim/checkpoint.hpp`:

```cpp
struct Checkpoint {
    uint32_t step_index;
    // Free rigid body state
    float*   body_position;       // env × body × 3
    float*   body_quaternion;     // env × body × 4
    float*   body_linear_velocity;
    float*   body_angular_velocity;
    // Articulation state
    float*   joint_q;
    float*   joint_qdot;
    // Soft body / fluid state (deferred to v0.7 when these subsystems land)
    // ...
    // RNG state if used (sim-to-real noise — Phase 4)
    uint64_t rng_state;
};

class CheckpointManager {
public:
    CheckpointManager(const phi::DeviceContext& ctx, uint32_t max_checkpoints);
    void Capture(const WorldState& w, uint32_t step_index);
    void Restore(WorldState& w, uint32_t checkpoint_index);
    uint32_t NearestCheckpointBefore(uint32_t target_step) const;
private:
    DeviceVector<Checkpoint> store_;
};
```

### Task 5.2.3 — Recompute orchestrator

`src/diffsim/recompute_orchestrator.hpp`:

```cpp
// Drives the recompute window between checkpoint i and step j.
class RecomputeOrchestrator {
public:
    RecomputeOrchestrator(const phi::DeviceContext& ctx, RowSolver& solver,
                          CheckpointManager& cps, Tape& tape);

    // Replay forward from cps.checkpoint_index to target step, returning state at target_step.
    // Recompute is bit-exact under D1 (re-uses same Row buffers, same coloring, same iteration count
    // recorded in tape).
    void Replay(uint32_t from_checkpoint_idx, uint32_t to_step);

    // For each step in [from_checkpoint_idx_to_step, to_step], runs adjoint, accumulates grads.
    void RunAdjointInRange(uint32_t start_step, uint32_t end_step,
                           AdjointAccumulator& grads);
};
```

Determinism requirement: replay must produce bit-exact state matching the original forward. D1 contract holds because:
- Row coloring is deterministic (same scene → same colors)
- Solver iteration count is recorded in tape
- Lambda warm-start uses recorded values
- No float atomics (D1 path)

### Task 5.2.4 — Backward runner

`src/diffsim/backward_runner.hpp`:

```cpp
class BackwardRunner {
public:
    BackwardRunner(const phi::DeviceContext& ctx, Tape& tape, CheckpointManager& cps,
                   RecomputeOrchestrator& orch, RowSolver& solver);

    // grad_observations: per-step gradient of loss w.r.t. observations at that step
    //   shape [step_count, env_count, obs_dim]
    // Outputs: gradient w.r.t. actions, parameters
    void Run(const float* grad_observations,
             float* grad_actions,
             float* grad_parameters);
};
```

Algorithm:

```
adjoint state ← 0
for step in reverse_range(tape.step_count):
    if step is at a checkpoint boundary:
        cp_idx ← previous checkpoint
        orch.Replay(cp_idx, step)
    # State at this step is now reconstructed
    accumulate grad_obs[step] into adjoint state
    solver.RunAdjointKernel(tape.Entry(step), adjoint_state)
    extract grad_actions[step] from solver
    accumulate grad_parameters from row gradients
```

Memory: only one replay window in flight at a time → at most K steps of working state + 2 checkpoint snapshots. Bounded by `K × env × per-body-state`.

### Task 5.2.5 — Full-tape debug mode

For debugging adjoint correctness, allow `nuka_tape_desc_t.recompute_on_backward = 0`. This records full tape (no checkpointing); reverse pass is single-pass. Used in CI to validate that checkpoint-based backward matches full-tape backward.

### Task 5.2.6 — Tests

`tests/diffsim/test_tape_determinism.cpp`:

```cpp
TEST(TapeDeterminism, ReplayBitExact) {
    auto ctx = nuka::phi::MakeDeviceContext(0, nullptr);
    auto world = MakeTestWorld(ctx, "go2_stand.usda", /*env_count=*/16);
    auto tape = MakeTape(ctx, world, /*K=*/10);

    // Forward 50 steps with tape on
    for (int i = 0; i < 50; ++i) {
        nuka_world_step_with_tape(world.handle, tape.handle);
    }
    auto state_A = SnapshotState(world);

    // Replay from step 0 (checkpoint) to step 50
    RecomputeOrchestrator orch(ctx, ...);
    orch.Replay(/*from_checkpoint=*/0, /*to_step=*/50);
    auto state_B = SnapshotState(world);

    // Bit-exact match (D1)
    EXPECT_EQ(state_A, state_B);
}
```

`tests/diffsim/test_backward_long_episode_checkpoint.cpp`:

```cpp
TEST(BackwardLongEpisode, CheckpointVsFullTapeAgreement) {
    // Run 1000 steps with K=100 (10 checkpoints + tape)
    // Compute backward
    // Run same 1000 steps with K=1000 (full tape, single window)
    // Compute backward
    // Assert gradient outputs agree within 1e-5 relative
}
```

## Validation

- Replay is bit-exact under D1.
- Backward via checkpointing matches full-tape backward within 1e-5.
- Memory consumption scales as expected: ~K × env × state.
- Reverse pass is 2-3× forward pass time at K=50.
- V3 FD check (Phase 1) still passes after Phase 2 wiring.

## Exit Criteria for v0.5 Phase 2

1. Tape data structure operational; per-step records sufficient for replay + adjoint.
2. Checkpoint manager captures and restores full world state.
3. Recompute orchestrator replays forward bit-exactly.
4. Backward runner produces gradients matching full-tape baseline within 1e-5.
5. C ABI exposes `nuka_tape_create / step_with_tape / backward / destroy`.
6. 1000-step episode with K=100 fits in single-GPU memory envelope.
7. Reverse pass time 2-3× forward pass time at default K.

## What This Phase Does Not Do

- No IFT-at-convergence (Phase 3).
- No PyTorch backward integration (Phase 4).
- No JAX (Phase 4).
- No sim-to-real noise yet (Phase 4 — RNG state needs to be captured in checkpoint when sim2real lands).
- No soft body / fluid state in checkpoints (v0.7 adds those).
