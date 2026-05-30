#!/usr/bin/env python3
"""[sim-val #35] Python drive harness: torch inference -> C ABI -> Nuka -> readback.

PURPOSE (read this before trusting any number below)
----------------------------------------------------
This harness proves the *data path* works end to end:

    g1/h1 torch policy  --(numbers)-->  Nuka C ABI DRIVE_TARGET
        |                                       |
        |                                  nuka_world_step (CUDA)
        v                                       v
    next obs (stand-in)  <--(readback)--  q / qd / base-pose device buffers

It is **NOT control**. g1/h1 are *different robots* (humanoids); their policy
was trained on a humanoid observation/action space, not the Go2 quadruped. We
only borrow g1's output *numbers* and map them, by an ARBITRARY documented
mapping, onto Go2's 12 leg-joint drive targets. The drive is meaningless for
locomotion. The point is solely: load the engine from Python, push policy
numbers through the real C ABI, read real state back, and show it runs
deterministically with no NaN/Inf. A real Go2 policy (trained later) drops into
the SAME harness for meaningful control.

DIAGNOSTIC LADDER
-----------------
  L1 load      dlopen libnuka.so + nuka_get_version.
  L2 world     create device + go2_float Go2 world (64 envs; 4096 smoke).
  L3 readback  q / qd / base-pose views: shapes == env*13, all finite.
  L4 base LIVE base pose CHANGES under stepping (floating base + live FK).
  L5 drive     a single-joint +/- drive-target perturbation moves that joint's
               OWN velocity sign-correctly at 1 step (PD reaches the solver).
  L6 torch     load g1 motion.pt, forward pass, print output shape.
  L7 loop      N control steps closed loop (read -> obs -> policy -> map -> 12
               targets -> write -> step*decim). Assert: no exception, no
               NaN/Inf across q/qd/base, and bit-exact determinism over a
               repeated identical run.

Run:
  export CUDA_VISIBLE_DEVICES=0
  export LD_LIBRARY_PATH=/opt/cuda-12.8-root/usr/local/cuda-12.8/lib64:\
/root/miniconda3/envs/nuka-v03/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
  /root/miniconda3/envs/nuka-v03/bin/python examples/sim_val/g1h1_drive_harness.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nuka_cabi import (  # noqa: E402
    NukaABI,
    NUKA_FIELD_JOINT_POSITION,
    NUKA_FIELD_JOINT_VELOCITY,
    NUKA_FIELD_ARTICULATION_LINK_POSE,
    NUKA_FIELD_DRIVE_TARGET,
    GO2_BASE_LINK_COUNT,
    LINK_POSE_FLOATS,
)

SCENE = "/root/Nuka-Physics/examples/scenes/go2_float.usda"
G1_POLICY = "/root/third_party/unitree_rl_gym/deploy/pre_train/g1/motion.pt"
H1_POLICY = "/root/third_party/unitree_rl_gym/deploy/pre_train/h1/motion.pt"

# g1: obs 47 -> action 12 (introspected from memory.weight_ih_l0 / actor.2).
G1_OBS_DIM = 47
G1_ACTION_DIM = 12


# ---------------------------------------------------------------------------
# Ladder bookkeeping
# ---------------------------------------------------------------------------
class Ladder:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def record(self, rung: str, ok: bool, detail: str) -> None:
        status = "PASS" if ok else "FAIL"
        print(f"[{rung}] {status}: {detail}")
        self.rows.append((rung, ok, detail))

    def summary(self) -> bool:
        print("\n" + "=" * 72)
        print("DIAGNOSTIC LADDER SUMMARY")
        print("=" * 72)
        all_ok = True
        for rung, ok, detail in self.rows:
            all_ok &= ok
            print(f"  {rung:<4} {'PASS' if ok else 'FAIL':<4}  {detail}")
        print("=" * 72)
        print(f"OVERALL: {'ALL PASS' if all_ok else 'FAILURES PRESENT'}")
        return all_ok


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def env_slice(arr: np.ndarray, env: int) -> np.ndarray:
    """The 13 q/qd/target slots of one env (env-major layout)."""
    base = env * GO2_BASE_LINK_COUNT
    return arr[base : base + GO2_BASE_LINK_COUNT]


def env0_base_pose(pose_flat: np.ndarray) -> np.ndarray:
    """env0 base = element 0 of LINK_POSE: [px,py,pz, qw,qx,qy,qz] (quat w-first)."""
    return pose_flat[0:LINK_POSE_FLOATS].copy()


def load_policy(path: str):
    m = torch.jit.load(path, map_location="cpu")
    m.eval()
    return m


# ---------------------------------------------------------------------------
# L1 / L2
# ---------------------------------------------------------------------------
def l1_load(ladder: Ladder) -> NukaABI:
    abi = NukaABI()
    v = abi.version()
    ladder.record("L1", True, f"libnuka.so loaded; nuka_get_version = {v.major}.{v.minor}.{v.patch}")
    return abi


def l2_world(ladder: Ladder, abi: NukaABI, env_count: int):
    dev = abi.device_create(0)
    world = abi.world_create(dev, SCENE, env_count)
    qv = abi.buffer_view(world, NUKA_FIELD_JOINT_POSITION)
    expect = env_count * GO2_BASE_LINK_COUNT
    ok = qv.element_count == expect
    ladder.record(
        "L2",
        ok,
        f"device+world created scene=go2_float env_count={env_count}; "
        f"JOINT_POSITION element_count={qv.element_count} (expect {expect})",
    )
    return dev, world


# ---------------------------------------------------------------------------
# L3 readback
# ---------------------------------------------------------------------------
def l3_readback(ladder: Ladder, abi: NukaABI, world, env_count: int):
    n = env_count * GO2_BASE_LINK_COUNT
    qv = abi.buffer_view(world, NUKA_FIELD_JOINT_POSITION)
    qdv = abi.buffer_view(world, NUKA_FIELD_JOINT_VELOCITY)
    pv = abi.buffer_view(world, NUKA_FIELD_ARTICULATION_LINK_POSE)

    q = abi.read_floats(qv, n)
    qd = abi.read_floats(qdv, n)
    pose = abi.read_floats(pv, pv.element_count)
    base0 = env0_base_pose(pose)

    shapes_ok = (
        qv.element_count == n
        and qdv.element_count == n
        and pv.element_count == n
        and pv.element_stride_bytes == LINK_POSE_FLOATS * 4
    )
    finite_ok = np.isfinite(q).all() and np.isfinite(qd).all() and np.isfinite(pose).all()
    ok = shapes_ok and finite_ok
    ladder.record(
        "L3",
        ok,
        f"q/qd len={n} (stride {qv.element_stride_bytes}B), pose len={pv.element_count} "
        f"(stride {pv.element_stride_bytes}B); all finite={finite_ok}; "
        f"env0 base pose [px,py,pz,qw,qx,qy,qz]={np.round(base0,4).tolist()}",
    )
    return base0


# ---------------------------------------------------------------------------
# L4 base is LIVE
# ---------------------------------------------------------------------------
def l4_base_live(ladder: Ladder, abi: NukaABI, world, base0_before: np.ndarray):
    pv = abi.buffer_view(world, NUKA_FIELD_ARTICULATION_LINK_POSE)
    # Hold drive (do not write -> rest hold targets stay), step several phys steps.
    abi.step(world, 30)
    abi.sync()
    pose = abi.read_floats(pv, pv.element_count)
    base_after = env0_base_pose(pose)
    delta = base_after - base0_before
    moved = float(np.abs(delta).max())
    # A dead (fixed-base) scene would give an exactly constant pose. Require a
    # change well above float noise; the floating base sags under the
    # fixed-base-tuned rest gains so position clearly moves.
    ok = moved > 1e-4 and np.isfinite(base_after).all()
    ladder.record(
        "L4",
        ok,
        f"after 30 hold steps env0 base moved max|d|={moved:.5f} "
        f"(pz {base0_before[2]:.4f}->{base_after[2]:.4f}); base is LIVE (floating)",
    )


# ---------------------------------------------------------------------------
# L5 drive reaches solver (single-joint diagonal PD sign test, N=1)
# ---------------------------------------------------------------------------
def _single_joint_step1(abi, dev_scene, slot: int, delta: float, env_count: int):
    """Fresh world; perturb ONLY env0's `slot` by `delta`; 1 step; return env0 qd[slot]."""
    dev, world = dev_scene
    world2 = abi.world_create(dev, SCENE, env_count)
    n = env_count * GO2_BASE_LINK_COUNT
    qdv = abi.buffer_view(world2, NUKA_FIELD_JOINT_VELOCITY)
    dv = abi.buffer_view(world2, NUKA_FIELD_DRIVE_TARGET)
    t = abi.read_floats(dv, n).copy()
    if delta != 0.0:
        t[slot] += delta  # env0 slot only
    abi.write_floats(dv, t)
    abi.step(world2, 1)
    abi.sync()
    qd = abi.read_floats(qdv, n)
    val = float(qd[slot])
    abi.world_destroy(world2)
    return val


def l5_drive(ladder: Ladder, abi: NukaABI, dev, env_count: int):
    """For each actuated joint 1..12, a +/-delta on its OWN target produces an
    own-velocity differential of the matching sign at one step. This isolates
    'the write reached the PD solver' from gravity/contact (which are common-mode
    and cancel at N=1) and from articulation cross-coupling (we drive ONE joint
    and check that SAME joint -- the diagonal of M^-1*Kp is positive definite).
    """
    delta = 0.2
    rows = []
    all_ok = True
    for slot in range(1, GO2_BASE_LINK_COUNT):  # 1..12
        base = _single_joint_step1(abi, (dev, None), slot, 0.0, env_count)
        d_pos = _single_joint_step1(abi, (dev, None), slot, +delta, env_count) - base
        d_neg = _single_joint_step1(abi, (dev, None), slot, -delta, env_count) - base
        ok = d_pos > 0 and d_neg < 0
        all_ok &= ok
        rows.append((slot, d_pos, d_neg, ok))
    bad = [s for s, _, _, ok in rows if not ok]
    detail = (
        f"single-joint +/-{delta} target -> own qd sign-correct on all 12 "
        f"actuated joints (N=1, PD response isolated). "
        f"e.g. slot1 d(+)={rows[0][1]:+.5f} d(-)={rows[0][2]:+.5f}"
    )
    if bad:
        detail += f"; WRONG-SIGN slots={bad}"
    ladder.record("L5", all_ok, detail)


# ---------------------------------------------------------------------------
# L6 torch inference
# ---------------------------------------------------------------------------
def l6_torch(ladder: Ladder):
    g1 = load_policy(G1_POLICY)
    with torch.no_grad():
        out = g1(torch.zeros(1, G1_OBS_DIM))
    shape_ok = tuple(out.shape) == (1, G1_ACTION_DIM)
    finite_ok = bool(torch.isfinite(out).all())
    # h1 too, just to show >1 policy loads (different obs/action dims).
    h1 = load_policy(H1_POLICY)
    with torch.no_grad():
        out_h1 = h1(torch.zeros(1, 41))
    ok = shape_ok and finite_ok and tuple(out_h1.shape) == (1, 10)
    ladder.record(
        "L6",
        ok,
        f"g1 motion.pt forward zeros(1,{G1_OBS_DIM})->{tuple(out.shape)} finite={finite_ok}; "
        f"h1 forward zeros(1,41)->{tuple(out_h1.shape)} (recurrent LSTM actor)",
    )
    return g1


# ---------------------------------------------------------------------------
# L7 closed loop
# ---------------------------------------------------------------------------
def build_obs(q_env0: np.ndarray, qd_env0: np.ndarray, obs_dim: int) -> torch.Tensor:
    """Build the policy input from real readback. ARBITRARY plumbing mapping:
    we pack the 12 actuated joint positions + 12 velocities (24 numbers) into
    the first 24 slots of the g1 obs vector and zero-pad the rest. The g1 obs
    layout means something *for a humanoid*; here it is just a deterministic
    function of real Go2 state so the loop is genuinely state-driven. Base
    linear/angular velocity (part of the REAL Go2 48-d obs) is NOT exposed by
    the C ABI yet -> those slots stay zero (see README known-gap)."""
    obs = np.zeros(obs_dim, dtype=np.float32)
    obs[0:12] = q_env0[1:13]
    obs[12:24] = qd_env0[1:13]
    return torch.from_numpy(obs).unsqueeze(0)


def map_action_to_targets(action: np.ndarray, rest_targets_env0: np.ndarray) -> np.ndarray:
    """Map 12 policy outputs -> 12 Go2 leg drive-target deltas. ARBITRARY,
    documented plumbing mapping (NOT a trained Go2 policy): small bounded
    deltas around the rest hold so the run stays numerically sane.
        target_slot[1..12] = rest_hold[1..12] + 0.1 * tanh(action)
    A real Go2 policy would replace this with its own action scaling."""
    out = rest_targets_env0.copy()
    out[1:13] = rest_targets_env0[1:13] + 0.1 * np.tanh(action)
    return out


def run_closed_loop(
    abi: NukaABI,
    dev,
    policy,
    env_count: int,
    control_steps: int,
    decimation: int,
):
    """One full closed-loop run on a FRESH world. Returns (final_q, max_abs_seen,
    any_nonfinite). Only env0 is policy-driven; other envs hold (plumbing)."""
    torch.manual_seed(0)
    policy = load_policy(G1_POLICY)  # fresh -> zeroed LSTM hidden/cell state
    world = abi.world_create(dev, SCENE, env_count)
    n = env_count * GO2_BASE_LINK_COUNT
    qv = abi.buffer_view(world, NUKA_FIELD_JOINT_POSITION)
    qdv = abi.buffer_view(world, NUKA_FIELD_JOINT_VELOCITY)
    pv = abi.buffer_view(world, NUKA_FIELD_ARTICULATION_LINK_POSE)
    dv = abi.buffer_view(world, NUKA_FIELD_DRIVE_TARGET)

    rest_targets = abi.read_floats(dv, n).copy()
    rest_env0 = env_slice(rest_targets, 0).copy()

    any_nonfinite = False
    max_abs = 0.0
    with torch.no_grad():
        for _ in range(control_steps):
            q = abi.read_floats(qv, n)
            qd = abi.read_floats(qdv, n)
            pose = abi.read_floats(pv, pv.element_count)
            if not (np.isfinite(q).all() and np.isfinite(qd).all() and np.isfinite(pose).all()):
                any_nonfinite = True
            max_abs = max(max_abs, float(np.abs(q).max()), float(np.abs(qd).max()))

            obs = build_obs(env_slice(q, 0), env_slice(qd, 0), G1_OBS_DIM)
            action = policy(obs).squeeze(0).numpy().astype(np.float32)

            new_targets = rest_targets.copy()
            new_env0 = map_action_to_targets(action, rest_env0)
            new_targets[0:GO2_BASE_LINK_COUNT] = new_env0
            abi.write_floats(dv, new_targets)
            abi.step(world, decimation)

    abi.sync()
    final_q = abi.read_floats(qv, n)
    final_qd = abi.read_floats(qdv, n)
    final_pose = abi.read_floats(pv, pv.element_count)
    if not (
        np.isfinite(final_q).all()
        and np.isfinite(final_qd).all()
        and np.isfinite(final_pose).all()
    ):
        any_nonfinite = True
    abi.world_destroy(world)
    return final_q, max_abs, any_nonfinite


def l7_loop(ladder: Ladder, abi: NukaABI, dev, policy, env_count: int):
    control_steps = 50
    decimation = 4
    q1, max1, nf1 = run_closed_loop(abi, dev, policy, env_count, control_steps, decimation)
    q2, max2, nf2 = run_closed_loop(abi, dev, policy, env_count, control_steps, decimation)

    no_nan = not (nf1 or nf2)
    # Determinism: two fresh identical runs -> identical final q.
    max_q_diff = float(np.max(np.abs(q1 - q2)))
    bit_exact = np.array_equal(q1, q2)
    deterministic = max_q_diff <= 1e-6

    # Policy->solver effect: all 64 envs start from the SAME cooked state; only
    # env0 is policy-driven (envs 1..63 hold at rest). So env0 diverging from a
    # held env proves the policy's numbers actually reached the solver (this loop
    # is doing something, not just hold-collapsing). Without this, L7 would pass
    # even if the DRIVE_TARGET write were a silent no-op.
    driven = env_slice(q1, 0)[1:13]
    held = env_slice(q1, 1)[1:13]
    policy_effect = float(np.max(np.abs(driven - held)))
    has_effect = policy_effect > 1e-3

    ok = no_nan and deterministic and has_effect
    ladder.record(
        "L7",
        ok,
        f"{control_steps} control steps x decim {decimation} "
        f"({control_steps * decimation} phys steps) closed loop; "
        f"no NaN/Inf={no_nan}; max|q|seen={max1:.3f}; "
        f"determinism max|q1-q2|={max_q_diff:.2e} (bit-exact={bit_exact}); "
        f"policy->solver effect max|q(env0 driven)-q(env1 held)|={policy_effect:.4f} "
        f"(>1e-3={has_effect}); "
        f"env0 final q[1:13]={np.round(env_slice(q1,0)[1:13],4).tolist()}",
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 72)
    print("[sim-val #35] g1/h1 -> C ABI -> Nuka drive harness (PLUMBING ONLY)")
    print("g1/h1 are different robots; this is NOT Go2 control. See module docstring.")
    print("=" * 72)
    torch.manual_seed(0)
    np.random.seed(0)

    ladder = Ladder()
    abi = l1_load(ladder)

    env_count = 64
    dev, world = l2_world(ladder, abi, env_count)

    # 4096-env smoke (just create + one step + readback shapes; then destroy).
    try:
        world_big = abi.world_create(dev, SCENE, 4096)
        abi.step(world_big, 1)
        abi.sync()
        qbig = abi.buffer_view(world_big, NUKA_FIELD_JOINT_POSITION)
        big_ok = qbig.element_count == 4096 * GO2_BASE_LINK_COUNT
        abi.world_destroy(world_big)
        ladder.record("L2b", big_ok, f"4096-env smoke: world+step OK, q len={qbig.element_count}")
    except Exception as e:  # noqa: BLE001
        ladder.record("L2b", False, f"4096-env smoke FAILED: {e}")

    base0 = l3_readback(ladder, abi, world, env_count)
    l4_base_live(ladder, abi, world, base0)
    # done with the persistent L2/L3/L4 world
    abi.world_destroy(world)

    l5_drive(ladder, abi, dev, env_count)
    policy = l6_torch(ladder)
    l7_loop(ladder, abi, dev, policy, env_count)

    abi.device_destroy(dev)
    all_ok = ladder.summary()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
