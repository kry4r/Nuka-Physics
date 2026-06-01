# v0.1 Phase 7 Owner Handoff

This handoff records the remaining owner-only blockers for closing v0.1 p07.
AI agents must not move candidate golden files into `tests/oracle/golden/**`
or change the p07 C++ wrapper contract without owner approval.

## Current Strict Gate Blockers

The local strict gate command is:

```bash
PATH=/root/.nuka-toolchain-gcc14/bin:/root/miniconda3/bin:/root/.local/bin:$PATH \
CC=/root/.nuka-toolchain-gcc14/bin/x86_64-conda-linux-gnu-gcc \
CXX=/root/.nuka-toolchain-gcc14/bin/x86_64-conda-linux-gnu-g++ \
python3 tools/oracle/v01_exit_gate.py \
  --build-dir build-cuda128
```

The gate now auto-selects `.nuka-oracle-venv/bin/python` when that repository
local oracle environment exists. Passing `--python .nuka-oracle-venv/bin/python`
is still valid and equivalent.

As of this handoff, the gate is blocked by:

- `C++20 <expected>`: the available GCC 14.3 and GCC 15.2 libstdc++
  installations expose `std::expected` only in C++23 mode, while p07 requires a
  C++20 wrapper surface using `std::expected`.
- Missing owner-provided golden files:
  - `tests/oracle/golden/featherstone_go2_random_sample.bin`
  - `tests/oracle/golden/featherstone_h1_random_sample.bin`
  - `tests/oracle/golden/go2_stand_5s.bin`
- `Go2 MJX trajectory diff`: blocked by the missing
  `tests/oracle/golden/go2_stand_5s.bin`.

Build, p07 ctest, physics-smell lint, MJX Python deps, Git LFS, MuJoCo assets,
and downstream `find_package(nuka)` have passed in the strict gate.

Current local C++20 `<expected>` probe results:

| Compiler | Version | `-std=c++20 <expected>` |
| --- | --- | --- |
| `/root/.nuka-toolchain/bin/x86_64-conda-linux-gnu-g++` | GCC 15.2.0 | fails; `std::expected` is C++23-only |
| `/root/.nuka-toolchain-gcc14/bin/x86_64-conda-linux-gnu-g++` | GCC 14.3.0 | fails |
| `/usr/bin/g++` | GCC 9.4.0 | fails; no `-std=c++20` |
| `/usr/bin/g++-10` | GCC 10.5.0 | fails |

## Candidate Golden Files

Candidate files are staged outside protected storage:

| File | Size | SHA-256 |
| --- | ---: | --- |
| `/tmp/nuka_owner_candidates/featherstone_go2_random_sample.bin` | 208047 | `1b0fb3e3a453ac40e4aee13953943dd868004da6d71c5a1f1f40a9853860046f` |
| `/tmp/nuka_owner_candidates/featherstone_h1_random_sample.bin` | 320042 | `0368ded305da6ec3f61b66823dc65396a7328861a1a9939323878de0d488a14e` |
| `/tmp/nuka_owner_candidates/go2_stand_5s.bin` | 62450 | `8630566e67958545c102d3e8f0dcbf4112d1ca527edcbb682e99efd26ec6ddb3` |

Candidate headers:

| File | Kind | Samples | qpos | qvel | qacc | Model |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `featherstone_go2_random_sample.bin` | `random-qacc` | 1000 | 13 | 13 | 13 | `go2_mjx.xml` |
| `featherstone_h1_random_sample.bin` | `random-qacc` | 1000 | 20 | 20 | 20 | `h1.xml` |
| `go2_stand_5s.bin` | `joint-trajectory` | 1200 | 13 | 0 | 0 | `go2_stand.usda` |

The candidate shape checks pass with:

```bash
python3 tools/oracle/validate_golden_candidate.py \
  --path /tmp/nuka_owner_candidates/featherstone_go2_random_sample.bin \
  --kind random-qacc \
  --samples 1000 \
  --qpos-count 13 \
  --qvel-count 13 \
  --qacc-count 13 \
  --model-name go2_mjx.xml

python3 tools/oracle/validate_golden_candidate.py \
  --path /tmp/nuka_owner_candidates/featherstone_h1_random_sample.bin \
  --kind random-qacc \
  --samples 1000 \
  --qpos-count 20 \
  --qvel-count 20 \
  --qacc-count 20 \
  --model-name h1.xml

python3 tools/oracle/validate_golden_candidate.py \
  --path /tmp/nuka_owner_candidates/go2_stand_5s.bin \
  --kind joint-trajectory \
  --samples 1200 \
  --qpos-count 13 \
  --qvel-count 0 \
  --qacc-count 0 \
  --model-name go2_stand.usda
```

The owner preflight against CUDA ABA and the Go2 stand demo passes with:

```bash
NUKA_GOLDEN_DIR=/tmp/nuka_owner_candidates \
PATH=/root/miniconda3/bin:/root/.local/bin:$PATH \
ctest --test-dir build-cuda128 \
  -R 'FeatherstoneOracle|Go2Stand.OwnerGoldenTrajectoryMatchesWithinTolerance' \
  --output-on-failure -j1
```

## Owner Actions Required

1. Inspect and approve or reject the three candidate golden files.
2. If approved, add them to `tests/oracle/golden/**` through Git LFS.
3. Decide the p07 C++ wrapper toolchain contract:
   - provide a C++20 toolchain/standard library where `std::expected` compiles
     in `-std=c++20`, or
   - explicitly approve changing the public wrapper contract to C++23, or
   - explicitly approve an `std::expected` shim/alternative.
4. Re-run the strict gate command above.

## Push Status

Local branch `v01-foundation-refactor` is ready to push with all outgoing commit
messages containing `[skip ci]` and author `kry4r <nidhogxt@outlook.com>`.
The push is currently blocked by local workstation access rather than by the
repository contents:

- Direct HTTPS reaches GitHub but fails because the configured VS Code
  credential helper socket is unavailable and GitHub rejects anonymous write
  access.
- Temporary SSH push using local keys fails with `Permission denied
  (publickey)`, and the configured SSH agent socket cannot list identities.
- A temporary local HTTP proxy also failed to reach GitHub (`No route to
  host`).

v0.3 must not begin until the strict gate passes and v0.1 p07 is committed as
closed.
