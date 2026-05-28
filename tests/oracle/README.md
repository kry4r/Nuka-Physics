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

For random qacc samples, each record is:

```text
qpos[qpos_count], qvel[qvel_count], tau[qvel_count], qacc[qacc_count]
```

For Go2 stand trajectories, each record is:

```text
qpos[qpos_count], qvel[qvel_count], qacc[qacc_count]
```

Generate candidate random-sample files with:

```bash
python3 tools/oracle/generate_mjx_golden.py \
  --model .nuka-assets/mujoco_menagerie/unitree_go2/go2_mjx.xml \
  --mode random-qacc \
  --out /tmp/featherstone_go2_random_sample.bin
```

Generate the 5 s Go2 stand trajectory with:

```bash
python3 tools/oracle/generate_mjx_golden.py \
  --model .nuka-assets/mujoco_menagerie/unitree_go2/go2_mjx.xml \
  --mode stand-trajectory \
  --steps 1200 \
  --dt 0.004166666666666667 \
  --out /tmp/go2_stand_5s.bin
```

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
3. Run `ctest --test-dir build-cuda128 --output-on-failure`.
