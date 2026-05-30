# sim-val: Python drive harness (torch → C ABI → Nuka → readback)

`[sim-val #35]` — proves the **data path** works end to end from a Python torch
policy, through the Nuka C ABI, into the CUDA solver, and back out as readable
state. It is a *plumbing* validation, not control (see the caveat below).

## Files

- `nuka_cabi.py` — dependency-light (ctypes + numpy) binding of the public C
  ABI (`src/include/nuka/nuka.h`): version, device/world lifecycle, step,
  buffer views, and host↔device `cudaMemcpy`.
- `g1h1_drive_harness.py` — the diagnostic-ladder harness (L1..L7).

## How to run

```bash
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/opt/cuda-12.8-root/usr/local/cuda-12.8/lib64:\
/root/miniconda3/envs/nuka-v03/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
/root/miniconda3/envs/nuka-v03/bin/python examples/sim_val/g1h1_drive_harness.py
```

Exit code 0 = all ladder rungs PASS. The harness prints each rung PASS/FAIL with
concrete numbers and a final summary table.

`LD_LIBRARY_PATH` is needed so the process can resolve the CUDA 12.8 runtime
(`libcudart.so.12`, which `libnuka.so` links by rpath) and torch's own libs.
`libnuka.so` itself has no unresolved deps (verified with `ldd`); no rebuild is
required.

## ⚠️ This is PLUMBING, not Go2 control

g1/h1 are **different robots** (humanoids). Their policies were trained on a
humanoid observation/action space, **not** the Go2 quadruped. This harness only
borrows the policy's output *numbers* and maps them, by an **arbitrary,
documented** mapping, onto Go2's 12 leg-joint drive targets. The resulting
motion is meaningless for locomotion — the Go2 floating base simply collapses
(joint angles run large, e.g. |q| up to ~12 rad after 200 phys steps). That is
fine: the harness only asserts the *data path* is correct — values stay finite
and the run is bit-exactly deterministic. A real Go2 policy (trained later)
drops into the **same** harness for meaningful control.

## Buffer-view layout & gotchas

Views come from `nuka_world_get_buffer_view`. For Go2 the articulation has
`base_link_count = 13` (root link 0 + 12 actuated leg joints). All buffers are
**env-major**: env `e`'s slot for link `l` is element `e*13 + l`.

| Field (enum)                            | element_count | per-element            | notes |
|-----------------------------------------|---------------|------------------------|-------|
| `JOINT_POSITION` (2) — q                 | `env*13`      | 1 float                | slot 0 = root (not a joint), slots 1..12 = leg joints |
| `JOINT_VELOCITY` (3) — qd                | `env*13`      | 1 float                | same layout as q |
| `ARTICULATION_LINK_POSE` (1) — world pose| `env*13`      | 7 floats (stride 28 B) | env `e` base = element `e*13+0` |
| `DRIVE_TARGET` (6) — **WRITABLE** PD tgt  | `env*13`      | 1 float                | write in place; next `step` applies it. slot 0 = root no-op; slots 1..12 act |

**Quaternion order is `w,x,y,z` (w FIRST)** in `ARTICULATION_LINK_POSE`, *not*
the common `xyzw`. Each link pose element is a `math::Transform` =
`[px,py,pz, qw,qx,qy,qz]`, 7 contiguous floats, no padding.

**`ARTICULATION_LINK_POSE` has a one-step lag** vs `JOINT_POSITION`: FK is run
from the pre-integrate q, so after `step()` returns the pose reflects the
previous step's q. Benign here — L4 only needs to see the base *change*; L5/L7
use q/qd (no lag) for their assertions.

### Device↔host transfer

The view's `device_ptr` is a **CUDA device pointer**. We move bytes with
`cudaMemcpy` from the **same** `libcudart.so.12` that `libnuka.so` itself loads
(`/opt/cuda-12.8-root/.../lib64/libcudart.so.12`). The CUDA primary context on
device 0 is process-global, so a single runtime backs both the engine's
allocations and our copies. `cudaMemcpyHostToDevice = 1`, `DeviceToHost = 2`.
A write→step→read round-trip on `DRIVE_TARGET`/`JOINT_POSITION` confirms it.

## The diagnostic ladder

| Rung | Proves |
|------|--------|
| L1   | `libnuka.so` dlopens, `nuka_get_version` returns. |
| L2   | device + go2_float Go2 world create (64 envs); **L2b** 4096-env smoke. |
| L3   | q/qd/base-pose views: shapes `== env*13`, all finite; print env0 base pose. |
| L4   | env0 base pose **changes** under hold stepping → floating base + live FK (a fixed base would be a dead constant). |
| L5   | a single-joint ±0.2 drive-target perturbation moves that joint's **own** velocity sign-correctly at **1 step** → the write reaches the PD solver. |
| L6   | g1 (and h1) `motion.pt` load + forward pass in-process with the engine loaded. |
| L7   | 50 control steps × decim 4 closed loop (read → obs → policy → map → 12 targets → write → step); **no NaN/Inf**, **bit-exact determinism** across a repeated identical run, and **policy→solver effect** (env0 is policy-driven, envs 1..63 hold from the *same* cooked start, so env0 diverging proves the policy numbers actually reached the solver — not a silent no-op). |

### Why L5 is a *single-joint, N=1* test

Naïvely driving all 12 joints and asserting each q moves toward its target
fails for two real, physical reasons (both observed):
1. The floating base **sags** under the rest-pose PD gains (those gains were
   validated for the **fixed** base in `go2_stand.usda`, not the floating one),
   so over many steps gravity/contact swamp the small PD response.
2. With a free-floating base the inverse mass matrix `M⁻¹` is **dense**, so a
   target on joint *j* accelerates *other* joints too — joint-local
   "toward-target" is not even true in theory at multi-joint drive.

So L5 isolates cleanly: at **exactly one step** from the cooked rest pose, the
hold and perturbed worlds share identical q/qd/contact (common-mode, cancels);
the only difference is `Kp·delta` on the one driven joint. The diagonal of
`M⁻¹·Kp` is positive-definite, so that joint's **own** velocity differential has
the sign of `delta` — verified for all 12 joints, both signs, antisymmetric.

## Known gaps (relevant to the *later* real-Go2 obs, not this plumbing)

- **Base linear/angular velocity is NOT exposed by the C ABI.** The header notes
  it lives in the engine's `link_velocity[root]` but is "not currently exposed
  through this ABI." The real Go2 48-d observation needs base lin/ang vel
  (and projected gravity / commands). The harness's `build_obs` therefore packs
  only the 12 joint positions + 12 velocities (real readback) and **zero-pads**
  the rest of the g1 obs vector. This did **not** block plumbing — it only
  limits the realism of a future trained-Go2 observation. Exposing base velocity
  (e.g. a `NUKA_FIELD_LINK_VELOCITY`) is the follow-up for the real-obs goal.
- The rest-pose PD hold does not keep the *floating* base standing (it was tuned
  for the fixed base); expect the base to sag/collapse under any non-trivial
  drive. Harmless for data-path validation.
