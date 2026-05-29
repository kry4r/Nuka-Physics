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

| Stage | Canonical tag | Budget (µs) | RTX 4000 Ada baseline (p50 µs) |
| --- | --- | ---: | ---: |
| Contact generation (broadphase SAP + narrow-phase) | `contact_generation` | 150 | (TBD — filled by baseline-capture task) |
| Featherstone ABA (3 passes, warp-per-articulation) | `featherstone_aba` | 80 | (TBD — filled by baseline-capture task) |
| Row builder (per-env prefix-sum, no atomics) | `row_builder` | 50 | (TBD — filled by baseline-capture task) |
| Graph-coloring scheduler (cached unless topology changes) | `scheduler` | 20 | (TBD — filled by baseline-capture task) |
| Row solver (PGS ~10 iters) — dominant | `row_solver` | 500 | (TBD — filled by baseline-capture task) |
| Integrator (symplectic Euler) | `integrator` | 30 | (TBD — filled by baseline-capture task) |
| V2 invariant sampling (every 16 steps) | `v2_sampling` | 50 | (TBD — filled by baseline-capture task) |
| Buffer management | `buffer_mgmt` | 20 | (TBD — filled by baseline-capture task) |
| Headroom (stalls, launch overhead, jitter) | _(no tag)_ | 100 | (TBD — filled by baseline-capture task) |
| **Total** | `step_total` | **1000** | **(TBD — filled by baseline-capture task)** |

## Tooling

- `tools/perf/run_perf_baseline.py` drives a sweep over `N ∈ {64, 256, 1024, 4096}`
  and writes a baseline JSON (the `RTX 4000 Ada baseline (p50 µs)` column is
  populated from the captured `step_total` / per-tag `p50_us` values).
- `tools/perf/compare_perf_runs.py` diffs two baseline JSON files, matching
  sweep entries by `n_envs` and flagging per-tag regressions on a chosen metric
  (default `p50_us`).
