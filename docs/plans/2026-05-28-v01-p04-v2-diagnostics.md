# Nuka Physics v0.1 – Phase 4: V2 Invariant Monitoring + Diagnostics

> **Master plan reference:** §3 Round 11 (V2 layer) + §6 Validation Architecture
> **Prerequisites:** Phase 1 (V5 guardrails); can run in parallel with Phases 2 and 3
> **Blocks:** v0.1 exit criteria — invariant monitoring must be operational at v0.1 close
> **Exit criteria gate:** v0.1
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

Build V2 — the physics invariant monitoring layer. This catches silent physics drift before V1 oracles do: things like energy slowly leaking, momentum non-conservation under no-force conditions, joint range violations, NaN/Inf appearing, particle count drifting. V2 is **engine self-check, no oracle required**.

V2 must be:

- **Cheap** in production (sampled, not every step) — opt-in via dev-mode flag.
- **Always-on** in CI smoke tests (small scenes, every step).
- **Dashboardable** — trace output that can be plotted per-env over a training run.

## Tech Stack

- C++20
- CUDA reductions for GPU-resident state
- Existing `core` module conventions
- Optional: tinyplot / csv for trace export (Python-side plotting separate)

## Files to Create

- `src/core/diagnostics/invariants.hpp` — invariant declarations
- `src/core/diagnostics/invariants.cpp` — sampling and threshold logic
- `src/core/diagnostics/invariants_gpu.cu` — CUDA reductions for energy/momentum/etc.
- `src/core/diagnostics/invariants_gpu.cuh` — kernel declarations
- `src/core/diagnostics/trace_sink.hpp` — trace output abstraction (CSV / null / in-memory ring)
- `src/core/diagnostics/trace_sink.cpp`
- `tests/core/test_invariants_energy_conservation.cpp` — basic conservative scene
- `tests/core/test_invariants_nan_detection.cpp` — inject NaN, verify detection
- `tests/core/test_invariants_momentum.cpp` — no external force → momentum conserved
- `tools/scripts/plot_invariants.py` — plot CSV traces (one curve per env)

## Files to Modify

- `src/runtime/world_stepper.cpp` — invoke invariant sampler at end of step (if enabled)
- `src/runtime/physics_world.cpp` — surface enable/disable + sampling rate
- `src/CMakeLists.txt` — add diagnostics translation units

## Invariants Tracked

| Invariant | Subsystem | When violated |
|---|---|---|
| Total kinetic + potential + elastic energy | All | Drift > threshold under conservative external work |
| Linear momentum | All | Drift > threshold with zero external impulse |
| Angular momentum | All | Drift > threshold with zero external torque |
| Constraint residual: `\|J·v + b\|` after PGS converge | Solver | Exceeds tolerance → solver under-converged |
| Joint angle within `[lower, upper]` | Articulation | Excursion past limit by > slop |
| Featherstone link length stability | Articulation | Link rigid-body length drift > 0.1% |
| Particle count | Fluid (v0.7+) | Loss / duplication |
| NaN / Inf in position / velocity | All | Any single occurrence |
| Velocity envelope `\|v\| < V_max` | All | Per-config max-velocity violation |
| Position envelope (world AABB) | All | Body outside expected world bounds |

For v0.1, the following are operational immediately (existing code paths):
- Energy
- Momentum
- Constraint residual
- Joint range
- NaN/Inf
- Velocity / position envelope

Reserved for later phases:
- Link length (after Featherstone ABA — Phase 6)
- Particle count (after PBF — v0.7)

## Tasks

### Task 4.1 — Declare invariant API

`src/core/diagnostics/invariants.hpp`:

```cpp
#pragma once
#include "phi/device_context.hpp"
#include <cstdint>
#include <vector>

namespace nuka::core::diagnostics {

enum class Invariant : uint8_t {
    Energy,
    LinearMomentum,
    AngularMomentum,
    ConstraintResidual,
    JointRange,
    LinkLength,
    ParticleCount,
    NanInf,
    VelocityEnvelope,
    PositionEnvelope,
    _Count
};

struct InvariantThresholds {
    float energy_drift_rel = 0.02f;          // 2% over 1000 steps
    float momentum_drift_abs = 1e-3f;
    float constraint_residual_abs = 1e-3f;
    float joint_range_slop_rad = 1e-4f;
    float link_length_drift_rel = 1e-3f;
    float velocity_max = 1e3f;
    float position_max = 1e3f;
};

struct InvariantSample {
    Invariant which;
    uint32_t env_id;
    uint32_t step_index;
    float    value;
    bool     violation;     // value exceeded threshold
};

struct InvariantConfig {
    bool   enabled = false;     // off by default in production
    uint32_t sample_every_steps = 16;
    InvariantThresholds thresholds = {};
    bool   abort_on_nan = true;
    bool   trace_violations_only = true;
};

class InvariantSampler {
public:
    explicit InvariantSampler(const InvariantConfig& cfg);
    void Sample(const phi::DeviceContext& ctx,
                const class runtime::PhysicsWorld& world,
                uint32_t step_index,
                std::vector<InvariantSample>* out_violations);
    void Reset();

    const InvariantConfig& Config() const noexcept { return cfg_; }

private:
    InvariantConfig cfg_;
    // Per-env baseline values for drift detection
    std::vector<float> baseline_energy_;
    std::vector<float> baseline_linear_momentum_x_;
    // ...
};

} // namespace nuka::core::diagnostics
```

### Task 4.2 — CUDA kernels for sampling

`src/core/diagnostics/invariants_gpu.cu`:

```cuda
__global__ void compute_energy_kernel(
    const float* __restrict__ velocities,        // per-body linear velocity
    const float* __restrict__ angular_velocities,
    const float* __restrict__ inverse_masses,
    const float* __restrict__ inverse_inertias,
    const float* __restrict__ positions,         // for potential energy under gravity
    float gravity_z,
    uint32_t body_count_per_env,
    uint32_t env_count,
    float* __restrict__ out_energy_per_env)
{
    extern __shared__ float sdata[];
    uint32_t env_id = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t stride = blockDim.x;

    float partial = 0.f;
    for (uint32_t b = tid; b < body_count_per_env; b += stride) {
        uint32_t idx = env_id * body_count_per_env + b;
        // KE_linear
        float m_inv = inverse_masses[idx];
        if (m_inv > 0.f) {
            float vx = velocities[3*idx+0], vy = velocities[3*idx+1], vz = velocities[3*idx+2];
            partial += 0.5f * (vx*vx+vy*vy+vz*vz) / m_inv;
            // PE under gravity
            partial += (-gravity_z) * positions[3*idx+2] / m_inv;
        }
        // KE_angular (similar)
    }
    sdata[tid] = partial;
    __syncthreads();

    // Deterministic tree reduction (no atomics) — D1 contract
    for (uint32_t off = stride/2; off > 0; off >>= 1) {
        if (tid < off) sdata[tid] += sdata[tid + off];
        __syncthreads();
    }
    if (tid == 0) out_energy_per_env[env_id] = sdata[0];
}
```

Similar kernels for momentum (sum of `m * v` and `m * r × v`), constraint residual (read accumulated `\|J·v+b\|`), joint range (per-link check), NaN/Inf detect (single-pass scan).

### Task 4.3 — Trace sink

`src/core/diagnostics/trace_sink.hpp`:

```cpp
class TraceSink {
public:
    virtual ~TraceSink() = default;
    virtual void OnSample(const InvariantSample& s) = 0;
    virtual void Flush() = 0;
};

class CsvTraceSink : public TraceSink {
public:
    explicit CsvTraceSink(std::string_view path);
    void OnSample(const InvariantSample& s) override;
    void Flush() override;
private:
    // implementation uses FILE* not std::ofstream to keep public API STL-free
};

class InMemoryRingTraceSink : public TraceSink { ... };   // for unit tests
class NullTraceSink : public TraceSink { ... };           // production default
```

### Task 4.4 — Wire into world stepper

`src/runtime/world_stepper.cpp`, end of `Step`:

```cpp
if (invariant_sampler_ && step_count_ % invariant_sampler_->Config().sample_every_steps == 0) {
    std::vector<InvariantSample> violations;
    invariant_sampler_->Sample(*ctx_, world_, step_count_, &violations);
    for (const auto& v : violations) {
        if (trace_sink_) trace_sink_->OnSample(v);
        if (v.which == Invariant::NanInf && invariant_sampler_->Config().abort_on_nan) {
            // Abort: log + dump state + assert
            FailFastNanDetected(v);
        }
    }
}
```

### Task 4.5 — Smoke-test scenes

Create three CI smoke scenes used by every test run:

- `tests/data/smoke/single_falling_box.yaml` — single rigid box falling under gravity; energy conserved (KE+PE invariant); momentum monotonic; no NaN.
- `tests/data/smoke/double_pendulum.yaml` — 2-link articulation; chaotic but energy bounded; joint range respected.
- `tests/data/smoke/two_body_collision.yaml` — frontal elastic collision; momentum exactly conserved.

Each smoke scene defines:
- expected energy drift envelope over 1000 steps
- expected momentum drift envelope
- expected constraint residual ceiling

CI runs all three scenes with V2 enabled. Any threshold violation fails the build.

### Task 4.6 — Python plotting script

`tools/scripts/plot_invariants.py`:

```python
# Usage: python tools/scripts/plot_invariants.py out/run-2026-05-28/invariants.csv
# Produces: invariants_energy.png, invariants_momentum.png, invariants_residual.png
# Each plot: one curve per env, x = step_index, y = value, threshold band shaded
```

Used post-training to visually inspect drift trends across 4096 envs (v0.3 onward).

## Validation

- Single falling box conserves energy within 2% over 1000 steps (proven by V2).
- Two-body elastic collision conserves momentum to 1e-3 (proven by V2).
- Injecting `quiet_NaN` into velocity buffer is detected within one sampling window.
- V2 disabled mode has zero overhead (verified by step-time benchmark).

## Exit Criteria for Phase 4

1. All 6 invariants operational (energy, linear momentum, angular momentum, constraint residual, joint range, NaN/Inf, velocity envelope, position envelope).
2. Sampling rate configurable; default off in production, on in CI smoke.
3. Trace sink writes per-violation CSV rows: `step,env_id,invariant,value,threshold,violation`.
4. Three CI smoke scenes pass invariant checks consistently.
5. `tools/scripts/plot_invariants.py` produces clean plots from a sample CSV.
6. Step-time benchmark shows < 1% overhead with V2 disabled, < 5% with V2 enabled at default sampling rate.

## What This Phase Does Not Do

- Does not check link length (Phase 6 Featherstone ships that).
- Does not check particle count (v0.7 PBF).
- Does not provide a real-time dashboard — only post-hoc CSV + plot.
- Does not compare against an external oracle (V1, Phase 7).
