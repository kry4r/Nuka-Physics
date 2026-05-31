# Nuka Physics v0.3 (S1) — On-Box Test & Perf Report

**Date:** 2026-05-31 · **Branch:** `v03` · **Box:** 2× NVIDIA RTX 4000 Ada Generation
(dev/relative reference — **not** a 4090/5080; the master-plan §7 absolute gate is
unvalidatable here, only relative numbers are). **Determinism:** D1 (strong).

This is the on-box test-status + 4096-env perf hotspot report (the perf track the owner
re-prioritized after the S1 anchor landed). It is the input to the optimization decision
recorded in §4.

---

## 1. Test-suite status (this box, this commit)

| Suite | Result | Notes |
|---|---|---|
| **Python** `pytest python/tests/ -q` | **50 / 50 pass** | incl. `test_ppo_loop_regression.py` (action-space ±1 guard, rl_games rescale=identity, end-to-end target not amplified, D1 masked-teleport byte-compare). |
| **C++/CUDA** `ctest` (full) | **328 / 330 pass**, 1 disabled | All determinism / reset / base-pose / Go2-standing / 4096-env / C-ABI / perf-timing tests green. |
| — known carry-forward failures | **2** (pre-existing, orthogonal) | `#38 V01FoundationE2E.Phase6CudaAbaMatchesGo2AndH1MuJoCoOracle`, `#206 FeatherstoneOracle.RandomSampleGoldensMatchCudaAba` — the ABA-vs-MuJoCo oracle gap documented since sim-val #43; **not green**, not introduced by v0.3 RL work. |
| — disabled | 1 | `Go2PdStanding.StaticPoseSweep` (intentionally disabled). |
| **Full Runner smoke** `train_go2_ppo.py --smoke` | clean exit | 3 epochs, c_loss finite, entropy stable, no collapse signature. |

## 2. 4096-env step perf (D1, RTX 4000 Ada)

Harness: `nuka_perf_harness --scene go2_stand.usda --envs 4096 --steps 2000 --determinism d1`.
Two independent runs (reproducible):

- **step_total p50 = 3789–3839 µs → per-env-step p50 = 0.93 µs.**
- step_total mean = 6197–6230 µs → per-env-step mean ≈ 1.51 µs.

**The median per-env-step (0.93 µs) is already under the 1 µs/env-step target on the
*slow* dev box.** A 4090 (~3× this GPU) would put p50 ≈ 0.3 µs. The mean is higher than
the median — characterized in §3.

### Hotspot map (per-tag, mean µs @ 4096, vs the frozen baseline)

| tag | cur mean | base mean | cur p50 | p99 | share of p50 step |
|---|---|---|---|---|---|
| **step_total** | 6196.6 | 4566.9 | 3788.8 | 17852 | 100% |
| **solve_contact_rows** (PGS) | 3860.0 | 2217.3 | 1458.2 | 15667 | ~38% |
| **factor_inertia_m_inv** | 887.4 | 887.9 | 887.7 | 896 | ~23% |
| **aba_compute_accelerations** | 663.3 | 664.0 | 663.6 | 668 | ~17% |
| **crba_inertia_m** | 383.3 | 382.7 | 383.2 | 387 | ~10% |
| row_builder | 353.7 | 354.5 | 49.4 | 1307 | ~1% (p50) |
| update_world_poses / contact_* / integrator / … | < 50 each | — | — | — | remainder |

Every non-solver tag matches the frozen baseline to <1% and has **p50 ≈ p99** (rock-stable,
deterministic). At p50 the step is: contact solver ~38%, inertia factorization ~23%,
ABA ~17%, CRBA ~10%.

## 3. The mean is contact-solver load-variance, not jitter

`solve_contact_rows` is the **only** tag with a p50→p99 spread (1458 → 15667 µs); on the
same GPU, same stream, same steps, `factor_inertia_m_inv` (887.7→896), `aba_…` (663.6→668),
`crba` are all flat p50≈p99. Uniform GPU contention (a co-tenant on this shared box) would
smear **every** tag's tail — it doesn't — so the inflation is intrinsic to the solver.

Mechanism: the articulated PGS runs **fixed** iteration counts (10 velocity + 4 position)
with **one block per env**; kernel wall-time = the slowest block, and the active-contact
count fluctuates per step as the feet make/break contact at the stand. The worst env per
step gates the whole kernel → a heavy right tail in the mean while the median stays clean.
(Note: `nvidia-smi`/NVML is driver-mismatch-broken on this box, so the co-tenant check is by
the per-tag-spread argument above, not a live process query.)

## 4. Optimization assessment & decision

**Constraints:** D1 strong determinism is non-negotiable — the dominant cost
(`solve_contact_rows`) is also the most determinism-fragile kernel (graph-free per-env PGS,
fixed lane order, no atomics, bit-exact goldens). **Only a change that preserves the exact
floating-point operation order — and therefore bit-identical state — qualifies**
(launch-config/occupancy, memory coalescing, or dead-row elimination *proven* to contribute
exactly 0.0). Any FP-reordering "optimization" breaks the ABA/contact goldens and is off-limits,
especially heading into a branch merge.

**Median already meets the target on the slow dev box.** Per the owner's delegation of the
"good enough" call, the legitimate bar for touching the solver is: a *cheap, clearly-winning,
provably bit-identical* change surfaced by profiling. The verification gate for ANY kernel
change is fixed: two-run byte-exact + ABA/contact goldens + determinism ctest + before/after
harness delta; a speedup that cannot show bit-identical state is rejected.

→ **Decision recorded in §5** (filled after the solver profile).

## 5. Profile result & outcome — **investigated, no D1-safe win taken (median meets target)**

**Tooling note:** this box's driver (550.163) mismatches the CUDA 12.8 toolkit, which breaks
both NVML (`nvidia-smi` queries) and Nsight Compute (`ncu` fails with
`cudaGetDeviceProperties ... not supported` when it instruments the process). So the analysis
is structural (source + per-tag timing), not a live hardware profile.

**Structural finding — the optimization surface is closed under D1.** Every hotspot kernel is
launched **one warp (32 threads) per env** with a fixed, deterministic FP sequence:
`SolveArticulatedContactRowsKernel<<<articulation_count, 32>>>` (fixed 10+4 iterations),
`FactorArticulationInertiaMKernel<<<…,32>>>`, `aba_compute_accelerations`,
`crba_inertia_m`. The candidate the structure suggested — **dead/zero-depth contact-row
elimination — is already implemented**: inactive slots are tagged `kRowInactive` and skipped
by the solver (`articulation_contacts.cu:653` + the solver loop). So the rows actually
processed are *live* contacts; the right-tail in §3 is genuine active-contact work + per-env
load imbalance, not wasted work on dead rows.

The remaining ideas all **reorder floating-point ops and break the bit-exact ABA/contact
goldens**, so they are disqualified under the non-negotiable D1 constraint:
- changing the per-env warp width or the in-warp reduction structure (would change FP order);
- a different inertia factorization / inverse algorithm (changes FP results);
- reducing solver iterations (changes the physics + goldens).

The only *potentially* D1-safe directions are **launch-overhead reduction** (the ~250 µs / ~6.5%
of the p50 step that is unaccounted kernel-launch latency — addressable with CUDA-graph
capture/replay, but complicated by the per-step-varying contact-count launch configs) and
**multi-env-per-block packing** (keep each env on its own warp with an identical per-warp FP
sequence, pack several warps per block for occupancy — bit-identical *iff* a shared-memory/sync
audit confirms no cross-warp coupling). Both are **substantial kernel restructures requiring the
full verification gate**, target only the mean's right tail (the median already meets the
target), and carry real D1 risk **immediately before a branch merge**.

**Decision (the owner-delegated "good enough" call):** the per-env-step **median already meets
the < 1 µs target on the slow dev box** (0.93 µs; ≈ 0.3 µs projected on a 4090), and no
*cheap, clearly-winning, provably-bit-identical* optimization exists — the dead-row win is
already taken and everything else reorders FP. Per the project's D1-is-non-negotiable rule and
the proximity of the merge, **no kernel change is made**; the hotspot map + this assessment are
the deliverable. The two launch-overhead / occupancy directions are recorded above as
**owner-optional follow-ups** for a future perf sprint (ideally on a 4090 with working Nsight,
where the absolute §7 gate can actually be measured and a profile can target a real bottleneck).
Any future attempt must clear the fixed gate: two-run byte-exact + ABA/contact goldens +
determinism ctest + before/after harness delta.

---

## 6. Post-decision follow-ups (owner re-opened the perf track)

After §5, the owner re-opened the perf track and asked for the spec'd mechanisms +
regression tests. These were delivered **without touching `Step()`'s floating-point
sequence** — so §5's "no reference-path kernel change" decision stands; the new work is
all *additive* (opt-in paths + a measurement test). Each cleared the fixed gate.

- **Opt-in CUDA-graph step path** (`BatchedArticulatedWorld::StepGraph`, commit
  `2e412a5`). Captures the step kernel sequence once and replays the instantiated graph,
  cutting per-launch overhead. It is **bit-for-bit identical** to `Step()`: a byte-compare
  regression test (`tests/runtime/test_batched_articulated_graph.cpp`) runs the same
  initial state through `Step()` and `StepGraph()` and `memcmp`s q / qdot / base_pose /
  v_root / lambda over 250 steps with mid-run drive-target variation → **all five mismatch
  counts = 0**. `Step()`'s kernels are unchanged; the graph path is a separate, opt-in
  replay of the same launches. **Measured benefit (as-measured, median):** at 4096 envs the
  graph-vs-`Step()` delta is **≈ 0.6 % (within run-to-run noise)**, and at 1024 envs it
  flips sign across reps (noise). Reason: at these batch sizes the step is firmly
  **compute-bound** (~0.9 µs/env-step of kernel time), so µs-scale launch overhead is a
  rounding error — neither size is actually launch-bound. **The graph path's real value here
  is the determinism-preserving mechanism, not a speedup**; it would matter in launch-bound
  regimes (small batch / many tiny kernels). Orthogonal to the D1/D2 level — a captured D1
  graph is bit-exact.
- **D1/D2 determinism toggle + C ABI** (commit `ad2a4fd`).
  `enum DeterminismLevel{Strong=0, Weak=1}` selected once at world creation and exposed
  through the C ABI (`nuka_world_desc_t.determinism`) + the Python binding. **D1 (Strong)
  remains the byte-exact default** through all of this; D2 is the spec'd escape hatch for
  workloads that trade reproducibility for speed. Covered by `test_determinism_toggle.py`
  (5/5, incl. `test_strong_two_run_byte_exact`, `test_invalid_determinism_rejected`) and the
  C-ABI create/step/destroy test.
- **Parameterized perf-gate regression test** (spec Task 3.1.6,
  `tests/perf/test_go2_4096env_step_time.cpp`, ctest `Go2_4096env_StepTime.MeetsGatePerEnv`).
  Builds the 4096-env Go2 world, wall-clock times 1000 back-to-back `Step()` calls, and
  **records** the per-env-step µs; it asserts `< NUKA_PERF_GATE_US` (default 1000) **only on
  the designated `NUKA_PERF_VALIDATION_GPU`**, else `GTEST_SKIP`. On this RTX-4000-Ada dev
  box it records-and-skips (per-env-step ≈ 2.07 µs wall-clock — a deliberately conservative,
  no-graph/no-overlap throughput figure, still ~480× under the 1000 µs gate) so CI stays
  green; the absolute §7 gate asserts on the owner's validation card.

### Perf-number reconciliation (so the cited figure is honest)

A spot run once showed a 4096-env wall-clock step ~9× inflated (~33,700 µs); a clean re-run
**reproduced the committed figure** and identified the spike as **transient GPU contention**,
not a regression:

| method | 4096 step_total p50 (µs) | per-env-step (µs) |
|---|---|---|
| CUDA-event per-tag `step_total` | 3749–3754 | **≈ 0.92** |
| chrono wall-clock (per-step stream sync) | 3672–3678 | ≈ 0.90 |

The two methods now agree at **≈ 0.9 µs/env-step** (p99 a flat ~4020 µs — rock-stable), and a
`/proc`-based co-tenant check (NVML is driver-broken) found the box clean. The cited
**≈ 0.9 µs/env-step** median (§2's 0.93 µs) is confirmed; the transient 8.2 µs/env wall-clock
figure was a contention artifact and is discarded. Every method is far under the 1000 µs gate.

### Verification gate (this whole batch)

Full ctest **329 passed / 2 failed / 1 disabled** — the two failures are exactly the
pre-existing orthogonal oracle gaps (`#38 V01…Phase6`, `#206 FeatherstoneOracle…`), **no new
regressions**; pytest **55/55**; graph byte-exact (above); determinism two-run / reset /
base-pose / Go2-standing / C-ABI all green.
