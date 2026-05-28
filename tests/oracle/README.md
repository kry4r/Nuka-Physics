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
disables contact in MJX, and evaluates samples through a JIT-batched
`mjx.forward` call. This matches the current CUDA ABA oracle harness, which
validates joint-space Featherstone dynamics rather than contact solve behavior.
Use `--batch-size N` only if the default all-samples batch is too large for the
local MJX/JAX runtime.

Generate the 5 s Go2 stand trajectory with:

```bash
python3 tools/oracle/generate_mjx_golden.py \
  --model .nuka-assets/mujoco_menagerie/unitree_go2/go2_mjx.xml \
  --mode stand-trajectory \
  --steps 1200 \
  --dt 0.004166666666666667 \
  --out /tmp/go2_stand_5s.bin
```

The 5 s stand trajectory uses `mjx.step` and can be slow on a CPU-only JAX
runtime. Prefer a CUDA-enabled owner oracle environment for the approved golden
run.

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

On this workstation, downstream `find_package(nuka)` checks must use the
isolated CUDA 12.8 package root, not the system CUDA 11.x installation:

```bash
cmake -S /path/to/downstream -B /path/to/downstream/build \
  -DCMAKE_PREFIX_PATH=/path/to/nuka/install \
  -DCUDAToolkit_ROOT=/opt/cuda-12.8-root/usr/local/cuda-12.8
```
