# Nuka Physics v0.1 — Closure Report (Foundation Refactor)

> **Status:** **v0.1 CLOSED**
> **Date:** 2026-05-29
> **Branch:** `v01-foundation-refactor`
> **Closing commit:** `b3d60b8` — *"closes v0.1 phase 7: resolve owner goldens and C++20 wrapper gate"* (pushed to `origin`, `b9811e4..b3d60b8`)
> **Master plan reference:** §7 v0.1 exit criteria; Phase 7 is the v0.1 exit gate (`docs/plans/2026-05-28-v01-p07-c-abi-and-go2-stand-demo.md`)
> **🔒 Hard constraint upheld:** GPU-only simulation; no CPU physics simulation in production paths (master plan §5.6).

This report records the verification that closes the v0.1 *Foundation Refactor*. It supersedes the open blockers in
`docs/architecture/v01-p07-owner-handoff.md`.

---

## 1. v0.1 Exit-Criteria Verification (master plan §7)

All eight v0.1 exit criteria are green. Each was delivered by an earlier v0.1 phase; Phase 7 closed the final two
(C ABI + Go2 stand oracle demo) and this report confirms the strict gate over all of them.

| # | Master plan §7 exit criterion | Delivered by | Status |
| --- | --- | --- | :---: |
| 1 | PHI Stream refactored to non-owning view; PlatformContract de-singletonized | Phase 3 (`v01-p03-phi-refactor`) | ✅ |
| 2 | C ABI v0.1 working (create World, step, read state) | Phase 7 (`v01-p07`) | ✅ |
| 3 | CSR Universal Row format landed; Contact/Joint/Drive paths migrated through the new scheduler via diff-test bridge | Phase 5 (`v01-p05-csr-row-migration-scheduler`) | ✅ |
| 4 | Island / graph-coloring scheduler operational | Phase 5 | ✅ |
| 5 | V5 + V2 validation infrastructure operational | Phases 1, 4 (`v01-p01`, `v01-p04-v2-diagnostics`) | ✅ |
| 6 | Featherstone ABA forward dynamics CUDA-resident; advances Go2 correctly | Phase 6 (`v01-p06-featherstone-aba-cuda`) | ✅ |
| 7 | Codegen pipeline produces forward kernels for the four base row classes | Phase 2 (`v01-p02-row-ir-codegen-skeleton`) | ✅ |
| 8 | **Demo: Go2 stand 5 s, MJX oracle within tolerance** (literal v0.1 exit demonstration) | Phase 7 | ✅ |

---

## 2. Phase 7 Final Resolution — the last two strict-gate blockers

The owner handoff (`v01-p07-owner-handoff.md`) left exactly two owner-only blockers. Both are now resolved.

### 2.1 Owner golden artifacts (V1 oracle)

The three owner-approved candidate goldens are committed under `tests/oracle/golden/**` as **Git LFS pointers**. The
committed LFS object IDs match the SHA-256 fingerprints recorded in the handoff, confirming they are the approved
artifacts (and not regenerated):

| File | LFS oid (prefix) | Handoff SHA-256 (prefix) | Kind |
| --- | --- | --- | --- |
| `featherstone_go2_random_sample.bin` | `1b0fb3e3a4…` | `1b0fb3e3a4…` | random-qacc, 1000 samples, go2_mjx.xml |
| `featherstone_h1_random_sample.bin` | `0368ded305…` | `0368ded305…` | random-qacc, 1000 samples, h1.xml |
| `go2_stand_5s.bin` | `8630566e67…` | `8630566e67…` | joint-trajectory, 1200 samples, go2_stand.usda |

`tests/oracle/golden/**` is an AI-protected category (`ai-agent-boundary.md` §3); these files were added under explicit
owner direction using the owner-approved candidates.

### 2.2 C++20 `nuka::expected` wrapper contract

**Root cause:** libstdc++ exposes `std::expected` only at C++23 (gated on `__cpp_lib_expected`). Under the project's
`-std=c++20` build the `<expected>` header is *present but empty*, so `__has_include(<expected>)` is not a reliable
probe. Every locally-available compiler (GCC 14.3, 15.2, 10.5, 9.4) fails the raw `std::expected` probe in C++20 mode.

**Resolution (handoff "Owner Actions" option 3 — an `std::expected` shim/alternative, owner-directed):** the p07 C++20
wrapper *contract* is preserved — `nuka.hpp` still returns `expected<T, Error>` with the same `std::expected` semantics —
without switching the project to C++23.

- **New** `src/include/nuka/expected.hpp`: a feature-detected facility.
  - When `__cpp_lib_expected` is defined → `nuka::expected` / `nuka::unexpected` alias to `std::expected` / `std::unexpected`.
  - Otherwise (C++20) → a minimal fallback sufficient for the wrapper contract: **move-only** value types (Device/World
    have deleted copy ctors), an `expected<void, E>` specialization, and `unexpected<E>` with CTAD so
    `nuka::unexpected(err)` deduces.
- `nuka.hpp` now uses `nuka::expected` / `nuka::unexpected` throughout (public C++20 wrapper surface unchanged).
- The CMake probe was retargeted (`NUKA_HAS_STD_EXPECTED` → `NUKA_HAS_EXPECTED_WRAPPER`) to compile the actual
  `nuka::expected` contract. It is now true under C++20, which **enables the previously-skipped**
  `CppWrapper.RaiiCreateStepReadDestroy` GPU test to build and run against the fallback (ctest count 291 → 292).
- `tools/oracle/v01_exit_gate.py` now **compiles and runs** the `nuka::expected` wrapper contract under `-std=c++20`
  (success, `unexpected`/`error()`, and the `void` specialization), instead of probing raw `std::expected`.

---

## 3. Strict Exit-Gate Evidence

Command (auto-selects `.nuka-oracle-venv/bin/python`):

```bash
PATH=/root/.nuka-toolchain-gcc14/bin:/root/miniconda3/bin:/root/.local/bin:$PATH \
CC=/root/.nuka-toolchain-gcc14/bin/x86_64-conda-linux-gnu-gcc \
CXX=/root/.nuka-toolchain-gcc14/bin/x86_64-conda-linux-gnu-g++ \
python3 tools/oracle/v01_exit_gate.py --build-dir build-cuda128
```

Result: **`PASS: v0.1 p07 local exit gate is green`** — every check passed:

| Gate check | Result |
| --- | --- |
| C++20 `nuka::expected` wrapper (compile + run, `-std=c++20`) | ✅ PASS |
| MJX Python deps (jax, mujoco, mujoco.mjx) | ✅ PASS |
| Git LFS available | ✅ PASS |
| MuJoCo Menagerie assets (go2_mjx.xml, h1.xml) | ✅ PASS |
| 3× golden: LFS attributes + LFS pointer + present | ✅ PASS |
| Build (`cmake --build build-cuda128`) | ✅ PASS |
| ctest p07 filter (`CAbi\|CppWrapper\|FeatherstoneOracle\|Go2Stand`) | ✅ PASS |
| physics_smell lint | ✅ PASS |
| `find_package(nuka)` downstream install + build + run | ✅ PASS |
| Go2 engine trajectory vs MJX golden, tol 1e-4 | ✅ PASS, `max_abs = 3.58e-07` |

Supporting evidence:

- **`ctest --test-dir build-cuda128 --output-on-failure -j1` → 100% passed, 292/292** (includes the newly-live
  `CppWrapper.RaiiCreateStepReadDestroy`, test #43).
- **`tools/lint/physics_smell.py --time` → 0 violations.**
- **Go2 MJX trajectory diff:** `max_abs = 3.58e-07 < 1e-4` (well within the master plan §6 V1 tolerance).

---

## 4. Hard-Constraint Compliance (§5.6 GPU-only)

The Phase-7 closing change touches only the **C++20 host-side wrapper ergonomics** (`nuka::expected`), the CMake test
gate, and the exit-gate probe. No CPU physics-simulation path was introduced; the C ABI wrapper remains thin host-side
marshalling over GPU-resident state, consistent with §5.6. Production solver/integration logic remains CUDA-resident.

---

## 5. Retrospective

**What worked**
- The strict, runnable exit gate (`v01_exit_gate.py`) made "done" objective: every criterion is a command with a
  PASS/FAIL, and golden integrity is checked as LFS pointers (not regular blobs).
- Folding wrapper validation into an existing GPU e2e test (`CppWrapper.RaiiCreateStepReadDestroy`) — rather than a new
  narrow unit test — matches the project's validation philosophy (`agent.md`) and exercises the real move-only
  Device/World/BufferView path end-to-end.

**What was tricky / lessons**
- **C++20 vs C++23 `std::expected` divergence.** The standard-library split (header present, symbols absent in C++20)
  meant `__has_include` was a trap; the correct probe is the `__cpp_lib_expected` feature macro. Lesson captured in
  `expected.hpp` and both probes.
- **A silently-skipped test masked the gap.** Because the wrapper test was gated behind a `std::expected` compile-check
  that failed under C++20, it was skipped — so `nuka.hpp` was neither compiled nor tested under the project standard,
  while ctest still read "all green." Retargeting the gate to the real wrapper contract closed this blind spot.
- **Commit-message accuracy.** A pre-existing local commit (`615a7d4`, unpushed) described the `<expected>` resolution
  that was not yet in its tree (it contained only the goldens). The closing commit (`b3d60b8`) amends it so the message
  matches the actual diff (goldens **and** the wrapper fix). Lesson: verify `git show --stat` against the message before
  treating a closure commit as done.

**Risk register status (master plan §8, v0.1 items)**
- ✅ *Codegen IR over-engineering* — mitigated by the hard rule "v0.1 IR supports only the 4 base row classes" (§5.2).
- ⏳ *Self-written sparse solver delays* — N/A for v0.1; cuDSS remains default through v0.5 per the reserve.
- 🟡 *D1 strong determinism vs S1 perf bar* — not exercised in v0.1 (no perf gate yet); becomes active in v0.3 Phase 1.

---

## 6. Supersedes

`docs/architecture/v01-p07-owner-handoff.md` — all "Current Strict Gate Blockers" are resolved:
the C++20 `<expected>` blocker (via the `nuka::expected` C++20 fallback), the three missing goldens (committed as LFS),
and the Go2 MJX trajectory diff (now passing). The handoff's push blocker (workstation access / proxy "No route to
host") is also resolved — the closing commit pushed successfully via the project proxy.

---

## 7. Proposed Master-Plan Owner Edit-Log Entry (owner action required)

The master plan is AI-protected (`ai-agent-boundary.md` §1); per its §13 amendment process, the owner — not an AI
agent — adds the edit-log entry. Suggested line for the top of `docs/plans/2026-05-28-nuka-physics-master-plan.md`:

```
- 2026-05-29 – v0.1 (Foundation Refactor) closed. All §7 v0.1 exit criteria green; strict exit gate PASS (ctest 292/292, physics_smell 0, Go2 MJX max_abs 3.58e-07 < 1e-4). C++20 nuka::expected wrapper contract preserved via fallback shim. v0.3 (S1 sprint) may begin.
```

---

## 8. Next Steps

1. **v0.3 entry is now unblocked** (master plan §5.2: v0.1 exit criteria must all pass before v0.3 begins). See the
   consolidated entry plan: `docs/plans/2026-05-29-v03-entry-plan.md`.
2. **Quarterly external output (rhythm requirement, §5.1).** The p07 plan and §5.1 note a blog post / talk / preprint
   at the v0.1 → v0.3 boundary. It is *recommended but not yet produced*; it is an owner-rhythm item, not a code gate.
3. **No v0.3 implementation has started**, per scope discipline and the explicit instruction to close v0.1 first.
