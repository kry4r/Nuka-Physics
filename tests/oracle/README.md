# v0.1 Oracle Data

`tests/oracle/golden/**` is owner-protected. AI agents must not create or
modify the binary golden trajectory files there.

Golden files use a little-endian binary envelope:

- magic: `NUKAGOLD`
- `uint32 version` = `1`
- `uint32 kind`: `1` random q/qd/tau/qddot sample, `2` joint trajectory
- `uint32 sample_count`
- `uint32 qpos_count`
- `uint32 qvel_count`
- `uint32 qacc_count`
- `uint32 model_name_size`
- UTF-8 model name bytes
- float32 payload

For random qacc samples, each record is aligned to Nuka's fixed-root
articulation layout. The first slot is the fixed root link and remains zero;
the remaining slots are the MJX fixed-base joint coordinates:

```text
root_zero + qpos, root_zero + qvel, root_zero + tau, root_zero + qacc
```

For Go2 stand trajectories, each record currently matches the C ABI demo
trajectory payload:

```text
root_zero + qpos
```

Generate candidate random-sample files with:

```bash
python3 tools/oracle/generate_mjx_golden.py \
  --model .nuka-assets/mujoco_menagerie/unitree_go2/go2_mjx.xml \
  --mode random-qacc \
  --out /tmp/featherstone_go2_random_sample.bin
```

Generate the H1 random-sample file with:

```bash
python3 tools/oracle/generate_mjx_golden.py \
  --model .nuka-assets/mujoco_menagerie/unitree_h1/h1.xml \
  --mode random-qacc \
  --out /tmp/featherstone_h1_random_sample.bin
```

The random-sample generator removes the floating base and actuator blocks,
disables MJX constraint/contact/limit/equality/frictionloss terms, and evaluates
samples through a JIT-batched `mjx.forward` call. This matches the current CUDA
ABA oracle harness, which validates unconstrained joint-space Featherstone
dynamics rather than contact or joint-limit solve behavior.
Use `--batch-size N` only if the default all-samples batch is too large for the
local MJX/JAX runtime.

Validate a random-sample candidate before asking the owner to move it into the
protected golden directory:

```bash
python3 tools/oracle/validate_golden_candidate.py \
  --path /tmp/featherstone_go2_random_sample.bin \
  --kind random-qacc \
  --samples 1000 \
  --model-name go2_mjx.xml
```

Generate the 5 s Go2 stand trajectory with:

```bash
python3 tools/oracle/generate_mjx_golden.py \
  --model examples/scenes/go2_stand.usda \
  --mode stand-trajectory \
  --steps 1200 \
  --dt 0.004166666666666667 \
  --out /tmp/go2_stand_5s.bin
```

For ASCII USD stand scenes, the generator converts the authored v0.1 USD
articulation into a temporary fixed-base MJCF, then uses MJX forward dynamics
plus the same semi-implicit Euler update and explicit position-hold torque
contract as the C ABI demo path. The script writes through a temporary file
before atomically replacing the requested output path, so interrupted owner
runs do not leave a partial candidate at the target path. Prefer a CUDA-enabled
owner oracle environment for the approved golden run when available; the CPU
JAX path is acceptable for candidate generation if it finishes locally.

Validate the stand-trajectory candidate with:

```bash
python3 tools/oracle/validate_golden_candidate.py \
  --path /tmp/go2_stand_5s.bin \
  --kind joint-trajectory \
  --samples 1200 \
  --model-name go2_stand.usda
```

As of the current p07 implementation, the USD-derived stand candidate preflight
passes against the CUDA C ABI demo trajectory when generated from
`examples/scenes/go2_stand.usda`. AI agents still must not move
`/tmp/go2_stand_5s.bin` into `tests/oracle/golden/**`; the owner must inspect
and add the approved binary through Git LFS.

Diff two candidate trajectories with:

```bash
python3 tools/oracle/diff_trajectory.py \
  --actual /tmp/candidate.bin \
  --golden tests/oracle/golden/go2_stand_5s.bin \
  --tolerance 1e-4
```

Human owner flow:

1. Inspect candidate output.
2. Add approved binaries to `tests/oracle/golden/` through Git LFS.
3. Run the p07/v0.1 local close gate:

```bash
PATH=/root/.nuka-toolchain-gcc14/bin:/root/miniconda3/bin:/root/.local/bin:$PATH \
CC=/root/.nuka-toolchain-gcc14/bin/x86_64-conda-linux-gnu-gcc \
CXX=/root/.nuka-toolchain-gcc14/bin/x86_64-conda-linux-gnu-g++ \
python3 tools/oracle/v01_exit_gate.py \
  --build-dir build-cuda128 \
  --python .nuka-oracle-venv/bin/python
```

The gate is intentionally strict. It fails until the owner-provided goldens,
MJX Python environment, Git LFS, and `<expected>`-capable C++20 toolchain are
all available.

Before moving candidates into the protected golden directory, the owner can run
the C++ oracle tests against a separate candidate directory. The random-qacc
candidates must pass the CUDA ABA preflight before approval:

```bash
mkdir -p /tmp/nuka_owner_candidates
cp /tmp/featherstone_go2_random_sample.bin /tmp/nuka_owner_candidates/
cp /tmp/featherstone_h1_random_sample.bin /tmp/nuka_owner_candidates/
NUKA_GOLDEN_DIR=/tmp/nuka_owner_candidates \
  ctest --test-dir build-cuda128 \
  -R 'FeatherstoneOracle' \
  --output-on-failure -j1
```

The Go2 stand trajectory can be preflighted the same way after the owner
settles the model contract between `examples/scenes/go2_stand.usda` and the MJX
oracle source:

```bash
cp /tmp/go2_stand_5s.bin /tmp/nuka_owner_candidates/
NUKA_GOLDEN_DIR=/tmp/nuka_owner_candidates \
  ctest --test-dir build-cuda128 \
  -R 'Go2Stand.OwnerGoldenTrajectoryMatchesWithinTolerance' \
  --output-on-failure -j1
```

If that stand preflight reports a trajectory mismatch, do not approve the
candidate golden. It means the engine demo scene and the MJX oracle model are
not yet the same dynamics contract.

This override is only for owner preflight. The strict v0.1 gate still requires
the approved files at `tests/oracle/golden/**`.

On this workstation, downstream `find_package(nuka)` checks must use the
isolated CUDA 12.8 package root, not the system CUDA 11.x installation:

```bash
cmake -S /path/to/downstream -B /path/to/downstream/build \
  -DCMAKE_PREFIX_PATH=/path/to/nuka/install \
  -DCUDAToolkit_ROOT=/opt/cuda-12.8-root/usr/local/cuda-12.8
```
