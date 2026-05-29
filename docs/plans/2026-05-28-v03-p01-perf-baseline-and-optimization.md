# Nuka Physics v0.3 – Phase 1: Performance Baseline + 4096-env Optimization

> **Master plan reference:** §3 Round 6 (multi-GPU + envelope) + §3 Round 13 (S1 perf gates) + §7 v0.3 exit
> **Prerequisites:** v0.1 closed (all v0.1 phases done)
> **Blocks:** v0.3 Phase 4 (Go2 PPO needs sub-ms step time)
> **Exit criteria gate:** v0.3
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

Establish a hard performance baseline for 4096-env Go2 on RTX 4090 and execute the optimization work necessary to hit master plan §7 v0.3 exit criteria:

- **Step time < 1 ms per env-step on RTX 4090**
- **Energy drift < 2% over 1000 steps**
- (Other v0.3 exit items live in Phases 2–4)

Approach: instrument, measure, find hot kernels, optimize, repeat. No premature optimization; every change validated against V1 oracle (Featherstone) and V2 invariants to ensure perf gains do not introduce drift.

> **Hardware addendum (2026-05-29) — no RTX 4090 available.** The dev/optimization box is **2× RTX 4000 Ada Generation**
> (sm_89, ~20 GiB/ECC-off, 48 SMs, 360 GB/s) — ~3× below a 4090, so the < 1 ms@4096 absolute gate is **not reachable
> here**; this box records the relative `baseline.json` and hosts all optimization. Final perf validation is **deferred to
> the owner's RTX 5080** (Blackwell sm_120, 16 GiB, 960 GB/s, Nsight-capable) against a **supplementary v0.3-doc target of
> < 1.3 ms/env-step @ 4096** (relative-derived: 5080 ≈ 0.68× 4090 FP32, ≈ 0.95× BW; budget-weighted ×1.2–1.3 of the 1000 µs
> 4090 budget). The **master-plan §7 < 1 ms @ RTX 4090 gate is unchanged and protected** — the 5080 target is additive,
> recorded only in v0.3 plan docs (see entry-plan §7). Nsight (`ncu`/`nsys`) profiling (Task 3.1.3) also defers to the
> 5080; on this box, in-engine CUDA-event timers (Task 3.1.1) are the primary measurement path. CMake CUDA arch list must
> include `sm_120` so the owner can build for the 5080.

Critical constraint: **D1 strong determinism must not regress.** If a proposed optimization (e.g., atomic-based reduction) breaks D1, it is rejected. Per master plan Round 13 risk register, if D1 misses the perf bar, fall back to D2 (weak determinism) **for the training mode only**; D1 stays for oracle / debug.

## Tech Stack

- C++20, CUDA 12+
- NVIDIA Nsight Compute (kernel profiling)
- NVIDIA Nsight Systems (timeline tracing)
- Custom in-engine perf counters (`src/core/perf/`)

## Files to Create

- `src/core/perf/timing.hpp` — scoped CUDA event timers
- `src/core/perf/timing.cpp`
- `src/core/perf/perf_recorder.hpp` — aggregator: per-kernel mean / p50 / p99
- `src/core/perf/perf_recorder.cpp`
- `tools/perf/run_perf_baseline.py` — sweep N envs ∈ {64, 256, 1024, 4096}, log step time
- `tools/perf/compare_perf_runs.py` — diff two perf runs (regression detection)
- `tests/perf/test_go2_4096env_step_time.cpp` — automated gate
- `docs/architecture/perf-budget.md` — per-kernel time budget for sub-ms total

## Files to Modify

- `src/solver/gpu/row_solver.cu` — likely the largest hot spot; needs SoA review, occupancy tuning, possible kernel fusion
- `src/runtime/articulation/featherstone_aba.cu` — three-pass ABA; profile per-pass cost
- `src/constraint/gpu/contact_generation.cu` — contact builder runs every step
- `src/runtime/gpu/cuda_batched_world.cu` (etc., existing batched paths)
- `src/CMakeLists.txt` — add perf instrumentation translation units

## Per-Kernel Perf Budget (for 1 ms / env-step on 4096 envs)

Step time budget breakdown (target totals on RTX 4090, 4096 envs):

| Stage | Budget | Notes |
|---|---|---|
| Contact generation (broadphase SAP + narrow-phase) | 150 µs | current implementation; SDF cooker comes in v0.7 |
| Featherstone ABA (3 passes) | 80 µs | warp-per-articulation; 4096 articulations × 32-thread warp |
| Row builder (contact + joint + drive → rows) | 50 µs | per-env prefix-sum, no atomics |
| Graph coloring scheduler | 20 µs | only re-color on topology change; cached |
| Row solver (PGS iterations, ~10 iters × N colors × dispatch) | 500 µs | dominant; codegen per-class kernels |
| Integrator (symplectic Euler q, qd; v, x) | 30 µs | trivial element-wise |
| V2 invariant sampling (when enabled) | 50 µs | every 16 steps |
| Buffer mgmt / housekeeping | 20 µs | |
| **Headroom** | 100 µs | for stalls, kernel launch overhead, OS jitter |
| **Total target** | **1000 µs / step** | |

## Tasks

### Task 3.1.1 — Build perf timing infrastructure

`src/core/perf/timing.hpp`:

```cpp
namespace nuka::core::perf {

class ScopedCudaTimer {
public:
    ScopedCudaTimer(PerfRecorder& r, const char* tag, cudaStream_t stream);
    ~ScopedCudaTimer();
private:
    PerfRecorder& r_;
    const char* tag_;
    cudaStream_t stream_;
    cudaEvent_t  start_, stop_;
};

#define NUKA_CUDA_TIME(recorder, tag, stream) \
    nuka::core::perf::ScopedCudaTimer NUKA_UNIQUE_NAME(_t)(recorder, tag, stream)

} // namespace
```

`PerfRecorder` aggregates per-tag: total samples, mean, p50, p99, recent N for sliding stats.

Add `NUKA_CUDA_TIME` instrumentation around every major kernel in the step loop. Output: at end of run (or on demand), print a table.

### Task 3.1.2 — Baseline measurement

`tools/perf/run_perf_baseline.py`:

```python
# Runs go2_stand demo across N ∈ {64, 256, 1024, 4096} envs,
# 5000 steps each, records per-kernel times.
# Output: out/perf/baseline_2026-05-28.json
```

Capture v0.1 baseline before any optimization. This becomes the regression reference.

Document the baseline in `docs/architecture/perf-budget.md` together with the target budget above.

### Task 3.1.3 — Hot spot identification

Profile with Nsight Compute:
- Memory throughput (achieved vs theoretical)
- Warp occupancy
- Branch divergence
- Bank conflicts
- L1/L2 hit rates

For each kernel exceeding its budget, identify root cause (one of):
1. **Memory-bound** → SoA layout fix; coalesced loads; texture cache for Jacobians?
2. **Compute-bound** → unroll inner loops; use FMA; tensor-core for matrix ops where applicable
3. **Launch overhead** → fuse adjacent kernels; CUDA graphs for the step loop
4. **Underutilization** → grid sizing; persistent threads for variable workloads

### Task 3.1.4 — Likely optimization candidates (priority order)

Based on architecture analysis, expected optimization passes:

**a) RowSolver kernel fusion.** Currently dispatch + per-class evaluator is a separate launch per (color × row class). Fuse: one kernel reads `row_class_id` and switches inside the kernel. Trade-off: divergence within warp if mixed classes in same color — mitigate by sorting rows by class within color.

**b) Jacobian SoA conversion.** Phase 2's IR initially uses AoS `Row` struct with CSR Jacobian. For high row count, switch to SoA fields (separate arrays for `lambda`, `rhs`, etc.) so warp reads are coalesced. Codegen template change.

**c) CUDA Graphs for fixed step.** When env count and topology are constant (training), capture the entire step as a CUDA graph; subsequent steps execute the graph (saves per-launch overhead, ~5-20 µs × N kernels).

**d) Featherstone warp-level reductions.** Replace shared-memory inter-thread comms with warp shuffle where possible (and where determinism allows — shuffle reduction order is fixed → still D1).

**e) Contact narrow-phase batch.** Current narrow-phase runs per contact pair. Batch contact pairs by shape combinations and process in vectorized form.

**f) Drop redundant data.** Per master plan Round 4, the Row struct has fields the v0.1 row catalog doesn't use (e.g., `compliance_alpha = 0` for hard contacts). Codegen can elide these for classes that don't need them — smaller struct, more cache-friendly.

### Task 3.1.5 — D1 / D2 split at runtime

Per Round 13 risk register: if D1 fails to hit < 1 ms / step, fall back to D2 for training mode.

`src/solver/gpu/row_solver.hpp`:

```cpp
enum class DeterminismLevel : uint8_t {
    Strong = 0,    // D1: no float atomics; graph coloring; fixed reduction
    Weak   = 1,    // D2: atomics allowed; same seed in process is reproducible; cross-process not guaranteed
};

struct SolveConfig {
    uint32_t iterations = 10;
    DeterminismLevel determinism = DeterminismLevel::Strong;
};
```

The codegen emits two variants of the evaluator (D1 strict / D2 atomic-allowed) when `determinism == Weak` is configured. Oracle / debug / diff-sim is always D1. Training-mode `World` can be created with D2 if a flag in `nuka_world_desc_t` is set (extend the C ABI).

### Task 3.1.6 — Regression-gate perf test

`tests/perf/test_go2_4096env_step_time.cpp`:

The test must be **CI-safe on non-validation hardware** (this box has no 4090/5080) yet **assert on the designated
validation GPU**. It detects the GPU at runtime and is threshold-parameterized — it does **not** weaken the §7 1000 µs
constant, it keeps it as the default and only gates the *assertion* on the validation card. Knobs (env vars, with
defaults): `NUKA_PERF_GATE_US` (default `1000.0` = §7 4090 bar; owner sets `1300` for the 5080 run),
`NUKA_PERF_VALIDATION_GPU` (substring of the validation GPU name, e.g. `"RTX 5080"` or `"RTX 4090"`).

```cpp
TEST(Go2_4096env_StepTime, MeetsGatePerEnv) {
    auto dev = nuka::Device::Create(0, nullptr).value();
    auto world = nuka::World::CreateFromScene(dev, "examples/scenes/go2_stand.usda",
                                              4096, 1.f/240.f).value();
    world.StepN(100).value();           // warmup
    cudaDeviceSynchronize();

    auto t0 = now_us();
    constexpr int N = 1000;
    world.StepN(N).value();
    cudaDeviceSynchronize();
    double per_step_us = double(now_us() - t0) / N;

    const double   gate_us       = env_double("NUKA_PERF_GATE_US", 1000.0);
    const char*    validation_gpu = std::getenv("NUKA_PERF_VALIDATION_GPU"); // e.g. "RTX 5080"
    const bool     on_validation_hw = validation_gpu && gpu_name(dev).find(validation_gpu) != std::string::npos;

    // Always record the number (CI logs / baseline diffing) regardless of hardware.
    record_perf("go2_4096env_step_us", per_step_us);

    if (on_validation_hw) {
        EXPECT_LT(per_step_us, gate_us)
            << "Per-step " << per_step_us << " µs exceeds " << gate_us << " µs gate on " << gpu_name(dev);
    } else {
        GTEST_SKIP() << "Recorded " << per_step_us << " µs on " << gpu_name(dev)
                     << " (not the validation GPU; gate asserted only there). Gate=" << gate_us << " µs.";
    }
    // D1 strict by default; per-env determinism covered by tests/regression/.
}
```

**Gate disposition:** the formal §7 gate is **< 1000 µs on RTX 4090** (protected, unchanged). The owner validates on an
**RTX 5080** against the supplementary **< 1300 µs** v0.3-doc target (`NUKA_PERF_GATE_US=1300 NUKA_PERF_VALIDATION_GPU="RTX 5080"`).
On the dev box (RTX 4000 Ada) the test **records-and-skips** (no false CI failure); v0.3 exit perf-gate sign-off comes
from the owner's 5080 run.

## Validation

- Per-kernel timing visible via in-engine CUDA-event timers (Nsight timeline deferred to the owner's RTX 5080 run).
- Baseline JSON committed under `out/perf/` / `tools/perf/` (RTX-4000-Ada numbers; for regression diffing).
- After optimization: on the dev box, the gate test records-and-skips; **perf-gate sign-off is the owner's RTX 5080 run**
  against the supplementary **< 1.3 ms@4096** target (§7 4090 < 1 ms gate protected/unchanged).
- V1 Featherstone oracle still passes (no physics regression from optimization).
- V2 energy invariant still passes (no drift from optimization).
- D1 determinism still passes by default; D2 toggle exists as escape hatch.

## Exit Criteria for v0.3 Phase 1

1. Perf timing infrastructure operational; CI logs per-kernel times (in-engine timers on this box).
2. Baseline measured and documented (RTX-4000-Ada relative reference).
3. Parameterized gate test present and **green on the dev box** (records-and-skips off the validation GPU); the absolute
   < 1 ms@4090 / supplementary < 1.3 ms@5080 assertion is validated by the owner on the 5080 (deferred hand-off).
4. D1 / D2 toggle exposed in C ABI.
5. V1 + V2 validation continues to pass.

## What This Phase Does Not Do

- No new physics features.
- No PyTorch autograd (Phase 2).
- No RL training (Phase 3, 4).
- No multi-GPU scaling (M2 deferred until S3 phase).
- No SDF / BVH upgrades (v0.7).
