# v0.1 Phase 7 Protected Change Proposal

This proposal records the owner-only changes required to close v0.1 p07.
It does not modify protected files. AI agents must not perform the steps in
this proposal without an explicit owner request.

## Protected Changes Requested

### 1. Approve and add oracle goldens through Git LFS

Strict v0.1 p07 requires these protected files:

- `tests/oracle/golden/featherstone_go2_random_sample.bin`
- `tests/oracle/golden/featherstone_h1_random_sample.bin`
- `tests/oracle/golden/go2_stand_5s.bin`

Current candidate files are outside protected storage:

| Candidate | Size | SHA-256 |
| --- | ---: | --- |
| `/tmp/nuka_owner_candidates/featherstone_go2_random_sample.bin` | 208047 | `1b0fb3e3a453ac40e4aee13953943dd868004da6d71c5a1f1f40a9853860046f` |
| `/tmp/nuka_owner_candidates/featherstone_h1_random_sample.bin` | 320042 | `0368ded305da6ec3f61b66823dc65396a7328861a1a9939323878de0d488a14e` |
| `/tmp/nuka_owner_candidates/go2_stand_5s.bin` | 62450 | `8630566e67958545c102d3e8f0dcbf4112d1ca527edcbb682e99efd26ec6ddb3` |

Owner preflight:

```bash
python3 tools/oracle/v01_owner_preflight.py \
  --candidate-dir /tmp/nuka_owner_candidates \
  --build-dir build-cuda128
```

That helper is read-only with respect to `tests/oracle/golden/**`. It validates
candidate SHA-256 values and shapes, runs the CUDA oracle preflight against
`NUKA_GOLDEN_DIR`, and reports the current C++20 `<expected>` toolchain status.
The equivalent manual commands are:

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

NUKA_GOLDEN_DIR=/tmp/nuka_owner_candidates \
PATH=/root/miniconda3/bin:/root/.local/bin:$PATH \
ctest --test-dir build-cuda128 \
  -R 'FeatherstoneOracle|Go2Stand.OwnerGoldenTrajectoryMatchesWithinTolerance' \
  --output-on-failure -j1
```

If the owner approves the candidates, the protected-file change should be
single-purpose and staged through Git LFS:

```bash
git lfs install --local
cp /tmp/nuka_owner_candidates/featherstone_go2_random_sample.bin \
  tests/oracle/golden/featherstone_go2_random_sample.bin
cp /tmp/nuka_owner_candidates/featherstone_h1_random_sample.bin \
  tests/oracle/golden/featherstone_h1_random_sample.bin
cp /tmp/nuka_owner_candidates/go2_stand_5s.bin \
  tests/oracle/golden/go2_stand_5s.bin
git add tests/oracle/golden/featherstone_go2_random_sample.bin \
  tests/oracle/golden/featherstone_h1_random_sample.bin \
  tests/oracle/golden/go2_stand_5s.bin
git lfs ls-files -- tests/oracle/golden
```

Before committing, verify that the staged blobs are Git LFS pointers:

```bash
git show :tests/oracle/golden/featherstone_go2_random_sample.bin
git show :tests/oracle/golden/featherstone_h1_random_sample.bin
git show :tests/oracle/golden/go2_stand_5s.bin
```

Each staged blob should contain `version https://git-lfs.github.com/spec/v1`,
`oid sha256:`, and `size`.

### 2. Decide the C++20 wrapper contract for `std::expected`

The p07 public C++ wrapper currently includes `<expected>` and returns
`std::expected`. The strict gate compiles this probe with `-std=c++20`, but the
available local standard libraries reject it.

Current local probes show:

- `/root/.nuka-toolchain/bin/x86_64-conda-linux-gnu-g++` GCC 15.2.0 rejects
  `std::expected` in `-std=c++20` and reports it as C++23-only.
- `/root/.nuka-toolchain-gcc14/bin/x86_64-conda-linux-gnu-g++` GCC 14.3.0
  rejects `std::expected` in `-std=c++20`.
- `/usr/bin/g++-10` has no `<expected>` header.

Owner decision required:

- provide a C++20 toolchain or standard library where `std::expected` compiles
  in `-std=c++20`, or
- approve changing the p07 wrapper contract to C++23, or
- approve an `std::expected` compatibility strategy.

AI agents should not silently choose one of these options because it changes
the public ABI/wrapper contract described by the p07 specification.

## Close-Gate Command

After the protected changes and toolchain decision are applied, run:

```bash
PATH=/root/.nuka-toolchain-gcc14/bin:/root/miniconda3/bin:/root/.local/bin:$PATH \
CC=/root/.nuka-toolchain-gcc14/bin/x86_64-conda-linux-gnu-gcc \
CXX=/root/.nuka-toolchain-gcc14/bin/x86_64-conda-linux-gnu-g++ \
python3 tools/oracle/v01_exit_gate.py \
  --build-dir build-cuda128
```

The gate auto-selects `.nuka-oracle-venv/bin/python` when present for the MJX
dependency check. Passing `--python .nuka-oracle-venv/bin/python` remains
equivalent.

Do not begin v0.3 until this strict gate passes and v0.1 p07 is committed
closed with `[skip ci]` in the commit message.
