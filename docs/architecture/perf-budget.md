# Per-Kernel Performance Budget

This document records the per-kernel time budget for a 4096-env Go2 step and
the relative baseline captured on the dev/optimization box.

## Gates and hardware framing

The **protected formal gate** is master plan **§7**: a 4096-env Go2 step must
run in **< 1 ms per env-step on an RTX 4090**. That gate is unchanged and
authoritative.

The dev/optimization box is **2× RTX 4000 Ada Generation** (sm_89), roughly
**~3× below an RTX 4090**. The absolute < 1 ms @ 4096 gate is therefore **not
reachable on this hardware**, so this document records **RTX-4000-Ada-relative**
numbers — useful for tracking the effect of optimization passes, not for
asserting the formal gate. The owner validates on an **RTX 5080** against a
**supplementary < 1.3 ms @ 4096** target (a relative-derived, additive target
recorded only in the v0.3 docs; it does not replace the protected §7 4090 gate).

## Budget table (RTX 4090 targets, 4096 envs)

The budget column lists the target per-stage totals on an RTX 4090 at 4096
envs. The canonical per-tag column lists the tag names the in-engine
`PerfRecorder` and the `tools/perf/*` tooling use; these map 1:1 to the kernel
stages. `step_total` corresponds to the whole-step **Total**. **Headroom** is a
budget reserve (stalls, launch overhead, jitter) and has no dedicated tag.

The `RTX 4000 Ada baseline (p50 µs)` column is the measured steady-state value
captured by the frozen baseline run (see *Measured baseline* below). The
canonical tags are recorded a **non-uniform number of times per step**
(`row_builder` 4×, `integrator`/`contact_generation` 2×, the rest 1×), so this
column is **per-stage p50 and is NOT summable to `step_total`** — the summable
accounting is the detail sub-stage table further down. Counts are noted where
≠1×.

| Stage | Canonical tag | Budget (µs) | RTX 4000 Ada baseline (p50 µs) |
| --- | --- | ---: | ---: |
| Contact generation (broadphase SAP + narrow-phase) | `contact_generation` (2×/step) | 150 | 53.2 |
| Featherstone ABA (3 passes, warp-per-articulation) | `featherstone_aba` | 80 | 689.5 |
| Row builder (per-env prefix-sum, no atomics) | `row_builder` (4×/step) | 50 | 50.0 |
| Graph-coloring scheduler (cached unless topology changes) | `scheduler` | 20 | n/a (no separate kernel yet) |
| Row solver (PGS ~10 iters) — dominant | `row_solver` | 500 | 1565.7 |
| Integrator (symplectic Euler) | `integrator` (2×/step) | 30 | 20.2 |
| V2 invariant sampling (every 16 steps) | `v2_sampling` | 50 | n/a (not in this step path) |
| Buffer management | `buffer_mgmt` | 20 | 16.3 |
| Headroom (stalls, launch overhead, jitter) | _(no tag)_ | 100 | 220.0 (`launch_sync_overhead` residual) |
| **Total** | `step_total` | **1000** | **3892.2** |

`step_total` p50 = **3892 µs** is ≈ **3.9× the 1000 µs RTX-4090 budget**,
consistent with the "~3× below a 4090" hardware framing plus the detail-on event
syncs (see notes below). This is the whole-4096-env step; the master-plan §7
gate and the 1000 µs budget are both on `step_total`, **not** on
`step_total / n_envs`.

## Measured baseline (RTX 4000 Ada, frozen reference)

- **Frozen JSON:** `out/perf/baseline_rtx4000ada_4096_frozen.json` (named so the
  post-optimization capture in #6 will **not** overwrite it).
- **GPU (read via `cudaGetDeviceProperties`):** `NVIDIA RTX 4000 Ada Generation`.
- **Scene:** `examples/scenes/go2_stand.usda` (owner golden, unmodified).
- **Sweep:** `N ∈ {64, 256, 1024, 4096}`, 5000 measured steps after **250**
  warm-up steps (harness default `max(100, steps/20)` at 5000 steps), determinism
  `d1`, **detail timing ON**.

`step_total` across the sweep (µs/step):

| N | mean | p50 | p99 | per-env-step p50 (µs) |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 698.9 | 621.7 | 2070.5 | 9.713 |
| 256 | 805.5 | 708.6 | 2433.0 | 2.768 |
| 1024 | 1519.0 | 1315.9 | 5258.2 | 1.285 |
| 4096 | 4566.9 | 3892.2 | 17852.4 | 0.950 |

### Accounting-complete detail sub-stage breakdown @ 4096 envs (p50)

Each detail sub-stage is recorded exactly once per step, so this table **is**
summable: Σ(sub-stages) + `launch_sync_overhead` residual = `step_total` p50.

| Sub-stage (detail tag) | p50 µs | mean µs | p99 µs |
| --- | ---: | ---: | ---: |
| `solve_contact_rows` | 1555.5 | 2217.3 | 15503.4 |
| `factor_inertia_m_inv` | 887.8 | 887.9 | 896.0 |
| `aba_compute_accelerations` | 664.5 | 664.0 | 668.4 |
| `crba_inertia_m` | 383.0 | 382.7 | 387.2 |
| `update_world_poses` | 47.1 | 47.6 | 51.2 |
| `contact_effective_mass` | 40.2 | 40.7 | 44.0 |
| `chain_jacobians` | 29.7 | 30.4 | 37.9 |
| `assemble_rows` | 15.4 | 15.8 | 18.4 |
| `integrate_position` | 10.5 | 11.3 | 17.4 |
| `integrate_velocity` | 10.2 | 10.8 | 16.4 |
| `detect_foot_contacts` | 8.2 | 8.5 | 11.6 |
| `aba_apply_drives` | 7.9 | 7.9 | 11.1 |
| `link_pose_refresh_copy` | 7.2 | 7.7 | 14.3 |
| `contact_tangent_basis` | 5.1 | 5.1 | 9.2 |
| **Σ(sub-stages)** | **3672.2** | | |
| `launch_sync_overhead` (residual) | 220.0 | | |
| **`step_total`** | **3892.2** | | |

Σ(sub-stages) accounts for **94.3 %** of `step_total`; the 220 µs residual is
per-kernel launch latency + the implicit stream syncs the nested detail timers
force + host bookkeeping between scopes.

### Three dominant hotspots (the targets for #5 hotspot analysis and #6)

1. **`solve_contact_rows` (PGS contact solver) — p50 ≈ 1556 µs (~40 % of step).**
   This is the budget's predicted dominant stage (`row_solver` "— dominant").
   Confirmed real, not a detail-scope artifact: the independently-recorded
   canonical `row_solver` p50 = 1565.7 µs agrees with the detail
   `solve_contact_rows` p50 = 1555.5 µs. It is **high-variance**
   (mean 2217, p99 15503) — see the variance note below.
2. **`factor_inertia_m_inv` (dense joint-space inertia inversion) — p50 ≈ 888 µs.**
   Rock-steady (p99 896 ≈ p50). Corroborates the #3 short-window estimate
   (≈ 931 µs) within ≈ 5 %.
3. **`aba_compute_accelerations` (Featherstone ABA forward dynamics) — p50 ≈ 664 µs.**
   Steady (p99 668 ≈ p50). Corroborates the #3 estimate (≈ 702 µs).
   `crba_inertia_m` (≈ 383 µs, corroborates ≈ 409) is the fourth-largest and the
   third dense-linear-algebra kernel.

Together the three dense-LA kernels (factor / aba / crba ≈ 888 + 664 + 383 =
1935 µs) and the contact solver (1556 µs) are ≈ 90 % of `step_total`.

### Notes on how to read these numbers

- **Detail-on inflates `step_total`.** With detail timing enabled, each
  sub-stage scope forces a stream sync, so `step_total` here is **larger** than
  the production (detail-off) path. This baseline is therefore a **relative**
  reference only — comparisons stay apples-to-apples because the #6
  post-optimization capture is also detail-on. Do **not** read the 3892 µs
  absolute as the production step time.
- **The absolute < 1 ms @ 4096 (§7, RTX 4090) gate is unreachable on this box**
  (2× RTX 4000 Ada, ~3× below a 4090). The owner validates the absolute gate on
  an RTX 4090 / 5080; this box only records the relative regression reference.
- **Use p50 as the regression comparator.** `compare_perf_runs.py` defaults to
  `p50_us` — correct here, because the dense-LA kernels are rock-steady at p50.
  The contact solver tail is **high-variance** (p99 ≈ 10× p50): it is
  data-dependent contact work (the Go2 settles into ground contact over the
  first ~100 steps, after which the PGS solver does real iterations) plus
  forced-sync / 2-GPU-box jitter, **not** a measurement bug. p99 deltas on
  `solve_contact_rows` / `row_solver` will be noisy for #6 — weight p50 there.
- The earlier #3 short-window numbers (warmup 10 / 60 steps) reported the three
  dense-LA kernels as ≈ 80 % of the step because that window catches the
  **near-contact-free settling phase**, where `solve_contact_rows` p50 is only
  ≈ 54 µs (its work is already in the tail: mean 2351, p99 16229). This frozen
  baseline uses a 250-step warm-up so the measured window is **steady-state
  ground contact**, where the contact solver is the dominant p50 stage — exactly
  as the budget table predicts.

## Tooling

- `tools/perf/perf_harness.cpp` → built target **`nuka_perf_harness`** (under the
  CUDA-gated block in `tests/CMakeLists.txt`). A standalone CLI that drives the
  **real** built `gpu::BatchedArticulatedWorld` stepper over a chosen env count
  / step count and prints the engine `PerfRecorder` JSON (`schema_version` +
  `per_tag`, plus `gpu`/`n_envs`/`steps`/`warmup`/`determinism`) to **stdout**;
  all diagnostics go to stderr. It enables detail timing for the measured
  window and reads the GPU name via `cudaGetDeviceProperties`. These are
  **genuine measured engine timings**, not estimates.
- `tools/perf/run_perf_baseline.py` invokes `nuka_perf_harness` once per env
  count (`--harness <bin> --scene <usda> --envs <N> --steps <M>
  --determinism d1 --perf-json -`), sweeps `N ∈ {64, 256, 1024, 4096}`, and
  assembles the baseline JSON. It records the **device-reported** GPU name from
  the harness payload (not a hardcoded constant).
- `tools/perf/compare_perf_runs.py` diffs two baseline JSON files, matching
  sweep entries by `n_envs` and flagging per-tag regressions on a chosen metric
  (default `p50_us`).

### Reproduce the frozen baseline

```sh
export PATH="/opt/cuda-12.8-root/usr/local/cuda-12.8/bin:$PATH"
cmake --build build-cuda128 --target nuka_perf_harness -j
python3 tools/perf/run_perf_baseline.py \
    --harness build-cuda128/tests/nuka_perf_harness \
    --scene examples/scenes/go2_stand.usda \
    --envs 64,256,1024,4096 --steps 5000 --determinism d1 \
    --gpu-label "RTX 4000 Ada (dev/relative reference, not 4090/5080)" \
    --out out/perf/baseline_rtx4000ada_4096_frozen.json
```

The output path is an explicitly-frozen name; #6 must write its
post-optimization capture to a **different** file and diff against this one with
`compare_perf_runs.py`.
