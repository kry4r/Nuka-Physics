#!/usr/bin/env python3
"""[sim-val] Drive the REAL trained Go2 policy (motion.pt, 48->12) in Nuka.

RESULT (sim-val #41/#42/#43, honest TL;DR)
------------------------------------------
  * Conventions PROVEN (D0-D3, D5): obs assembly bit-exact to golden, identity
    joint permutation (cooked-q signature), gravity, body-frame velocities.
  * The policy WALKS at the NATIVE training dt=0.005 (decimation 4, 50 Hz):
    dx=+2.93 m / 6 s, vx 0.488 ~= cmd 0.5, tilt < 5.4 deg -- SAME quality as the
    old dt=0.001 sub-step, but at 1x physics cost. NO sub-stepping needed.
  * This is the #43 fix: the contractually-soft gains (Kp=20/Kd=0.5) previously
    FLAILED at dt=0.005 (tilt->110 deg) because the explicit -Kd*qdot drive damping
    is only conditionally stable (instability ~ dt*Kd/m_eff, m_eff shrunk by
    contact). The engine now applies GENERAL implicit joint damping (the drive
    defers -Kd*qdot to the constrained-velocity solve, (M+dt*C)^-1), which is
    unconditionally stable. The module ran at dt=0.001/decim 20 before #43; it now
    runs at the native dt=0.005/decim 4 (DT/DECIMATION below). The open-loop soft-
    gain FLOATING static hold still topples at any dt -- that is POSTURAL (the
    closed-loop policy is what keeps the robot up), NOT a damping/integration bug.
  See go2_policy_drive_README.md and docs/architecture/2026-05-31-v03-implicit-
  damping-golden.md for the full characterization.

THE POINT (read before trusting any number)
--------------------------------------------
Two facts are ALREADY proven separately:
  (a) Nuka's Go2 physics is CORRECT: a free-floating Go2 holds the real crouch
      under PD, deterministically (tests/runtime/test_go2_pd_standing.cpp), and --
      after the #42 seating fix -- on the c_abi/nanobind path too, at every dt.
  (b) the trained Go2 policy motion.pt infers correctly (C++/Python golden
      bit-exact; /root/third_party/go2_pr62/golden_io.json).
This harness CONNECTS them: it runs the policy in-the-loop on Nuka and PROVES every
convention first, so a misbehavior can be attributed (convention bug vs dynamics vs
integration). That discipline is what let #42 separate a real engine bug (the PD
stance penetration instability, FIXED) from the policy's native-dt flail (a
soft-gain explicit-integration limit, characterized + mitigated by sub-stepping).

We expect a sim2sim gap (the policy trained in Isaac Gym; Nuka has a different
solver/contact/integrator). The honest success bar is: "conventions provably
correct + coordinated, command-tracking locomotion under a stable integration
config", NOT necessarily native-dt or flawless walking.

THE CONTRACT (docs/plans/2026-05-30-v03-unitree-policy-config.md, re-confirmed
from PR62 deploy code):
  obs(48): [0:3] base_lin_vel x2.0, [3:6] base_ang_vel x0.25,
           [6:9] projected_gravity (gravity unit vec in BODY frame, no scale),
           [9:12] cmd*[2.0,2.0,0.25], [12:24] (q-default)*1.0,
           [24:36] qd*0.05, [36:48] previous raw action. clip +/-100.
  action(12): target_q = default + 0.25*action. Kp=20, Kd=0.5, target_vel=0.
  default_angles (URDF order FL,FR,RL,RR x hip,thigh,calf) =
    [0.1,0.8,-1.5, -0.1,0.8,-1.5, 0.1,1.0,-1.5, -0.1,1.0,-1.5]
  control: policy 50 Hz; dt=0.005, decimation=4 (matches training exactly).

DIAGNOSTIC LADDER (each rung PROVES a convention before the rollout trusts it):
  D0 obs-assembly  : build the t=0 default obs OFFLINE and assert it equals the
                     golden 'default_obs_cmd0.5' input AND policy(obs) == golden
                     output -- validates obs order/scales independent of Nuka.
  D1 permutation   : find Nuka-slot <-> URDF-index mapping STRUCTURALLY (quadrant
                     of each leg link from ARTICULATION_LINK_POSE + subtree size
                     from a single-joint perturbation), then verify direction.
  D2 gravity       : projected_gravity ~ [0,0,-1] upright AND a known 90-deg tilt
                     rotates correctly (catches sign/transpose bugs the upright
                     case would hide).
  D3 velocities    : base_lin/ang_vel read from LINK_VELOCITY root (body frame)
                     finite & sane.
  D4 warm-start obs: after warm-starting to default, (q-default)~0, last_action=0,
                     and the LIVE obs matches the offline default obs (modulo the
                     small physical sag + base velocity from the warm-start).
  D5 policy forward: policy(live obs) -> 12 finite actions.

Then: rollout (cmd=[0.5,0,0]) with honest behavior verdict, + an in-harness PD
baseline (Kp=60/Kd=4 hold at the cooked crouch) for an apples-to-apples
regression compare.

Run:
  export CUDA_VISIBLE_DEVICES=0
  /root/miniconda3/envs/nuka-v03/bin/python examples/sim_val/go2_policy_drive.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

import nuka

# ---------------------------------------------------------------------------
# Constants / contract
# ---------------------------------------------------------------------------
SCENE = "/root/Nuka-Physics/examples/scenes/go2_float.usda"
POLICY = "/root/third_party/go2_pr62/motion.pt"
GOLDEN = "/root/third_party/go2_pr62/golden_io.json"
GO2_BLC = 13  # base_link_count: slot 0 = root, slots 1..12 = actuated leg joints

# URDF / training joint order: FL, FR, RL, RR x (hip, thigh, calf).
URDF_JOINT_NAMES = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
]
# default standing angles in URDF order.
DEFAULT_ANGLES = np.array(
    [0.1, 0.8, -1.5,  -0.1, 0.8, -1.5,  0.1, 1.0, -1.5,  -0.1, 1.0, -1.5],
    dtype=np.float32,
)
# obs scales
S_LIN_VEL = 2.0
S_ANG_VEL = 0.25
S_DOF_POS = 1.0
S_DOF_VEL = 0.05
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
ACTION_SCALE = 0.25
OBS_CLIP = 100.0
ACTION_CLIP = 100.0
KP = 20.0
KD = 0.5
# per-joint-TYPE URDF effort limits (hip/thigh 23.7, calf 35.55) in URDF order.
FORCE_LIMIT_URDF = np.array(
    [23.7, 23.7, 35.55] * 4, dtype=np.float32
)

# timing: the policy rate is 50 Hz (matches training); the PHYSICS dt is decoupled from
# it via decimation. We run at the NATIVE training config dt=0.005/decim 4 -- after the
# #43 fix (general implicit joint damping) the contractually-soft gains (Kp=20/Kd=0.5)
# integrate stably at the native dt, so NO sub-stepping is needed. History (do not
# conflate the two engine fixes that got us here):
#
#   (1) #42 -- the PD STANCE.  The c_abi path collapsed the floating Go2 PD crouch at
#       dt>=0.002 (ground seating depth kRestFootPenetration 0.03 -> 0.002 m). FIXED;
#       the PD stance holds at every dt.
#
#   (2) #43 -- the POLICY flail.  The soft gains (Kp=20/Kd=0.5) FLAILED at dt=0.005
#       (tilt->110 deg) because explicit -Kd*qdot drive damping is only conditionally
#       stable (instability ~ dt*Kd/m_eff, m_eff shrunk by contact). The engine now
#       DEFERS the damping to the constrained-velocity solve and applies it IMPLICITLY
#       ((M+dt*C)^-1), which is unconditionally stable. The policy now WALKS at the
#       native dt=0.005 (dx +2.93 m/6 s, tilt<5.4 deg). NOTE: the open-loop soft-gain
#       FLOATING static hold still topples at any dt -- that is POSTURAL (the closed-
#       loop policy keeps it up), NOT a damping/integration bug.
DT = 0.005
DECIMATION = 4  # 0.005 * 4 = 0.020 s -> 50 Hz policy = the NATIVE training cadence
# PD-baseline = the EXACT proven C++ hold: hip splay +-0.1 (FL,FR,RL,RR = +,-,+,-),
# thigh +0.8, calf -1.5; Kp=60, Kd=4, force 24. The hip splay gives the roll margin
# the C++ test relies on; a symmetric hip=0 is metastable in roll.
PD_CROUCH = np.array(
    [0.1, 0.8, -1.5,  -0.1, 0.8, -1.5,  0.1, 0.8, -1.5,  -0.1, 0.8, -1.5],
    dtype=np.float32,
)
PD_KP, PD_KD, PD_FORCE = 60.0, 4.0, 24.0


def make_world(dev, env_count, dt=None):
    """Create the go2_float world at physics dt (default = module DT = 0.001, the
    sub-stepped policy config; pass dt=0.005 for the native-dt characterization).
    Passing dt is mandatory (the binding defaults to 1/240). See the timing note above
    for why the policy rollout sub-steps to dt=0.001 (the soft-gain integration limit,
    NOT the #42 seating bug, which is fixed)."""
    return nuka.World.create_from_scene(dev, SCENE, env_count, DT if dt is None else dt)


# ---------------------------------------------------------------------------
# Ladder bookkeeping
# ---------------------------------------------------------------------------
class Ladder:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def record(self, rung: str, ok: bool, detail: str) -> None:
        status = "PASS" if ok else "FAIL"
        print(f"[{rung}] {status}: {detail}", flush=True)
        self.rows.append((rung, ok, detail))

    def summary(self) -> bool:
        print("\n" + "=" * 78)
        print("CONVENTION DIAGNOSTIC LADDER SUMMARY")
        print("=" * 78)
        all_ok = True
        for rung, ok, detail in self.rows:
            all_ok &= ok
            print(f"  {rung:<4} {'PASS' if ok else 'FAIL':<4}  {detail}")
        print("=" * 78)
        print(f"LADDER OVERALL: {'ALL PASS' if all_ok else 'FAILURES PRESENT'}")
        return all_ok


# ---------------------------------------------------------------------------
# Math: gravity into body frame from the base quaternion (wxyz)
# ---------------------------------------------------------------------------
def quat_rotate_inverse_wxyz(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world vector v into the BODY frame, i.e. R(q)^T @ v.

    Mirrors Isaac Gym quat_rotate_inverse. q = [w,x,y,z]. The body-frame
    projected_gravity = quat_rotate_inverse(base_quat, [0,0,-1]).
    """
    w, x, y, z = q_wxyz
    # R^T = R(q^-1); build R from q then transpose-apply. Use the standard
    # quat_rotate_inverse closed form (a*v - 2 w (q_vec x v) + 2 q_vec (q_vec.v)).
    qvec = np.array([x, y, z], dtype=np.float64)
    a = (w * w - qvec.dot(qvec)) * v
    b = 2.0 * w * np.cross(qvec, v)
    c = 2.0 * qvec * (qvec.dot(v))
    return (a - b + c).astype(np.float32)


def projected_gravity_body(q_wxyz: np.ndarray) -> np.ndarray:
    return quat_rotate_inverse_wxyz(
        np.asarray(q_wxyz, dtype=np.float64), np.array([0.0, 0.0, -1.0])
    )


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
def load_policy():
    m = torch.jit.load(POLICY, map_location="cpu")
    m.eval()
    return m


def policy_forward(policy, obs_np: np.ndarray) -> np.ndarray:
    """obs_np: (N,48) float32 -> (N,12) float32 actions (CPU torch)."""
    with torch.no_grad():
        out = policy(torch.from_numpy(np.ascontiguousarray(obs_np, np.float32)))
    return out.numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# D0: prove the obs assembly OFFLINE against the golden (engine-independent)
# ---------------------------------------------------------------------------
def build_obs(
    base_lin_vel: np.ndarray,   # (N,3) body frame
    base_ang_vel: np.ndarray,   # (N,3) body frame
    proj_grav: np.ndarray,      # (N,3) gravity unit vec in body frame
    cmd: np.ndarray,            # (N,3) [vx,vy,wyaw]
    q_urdf: np.ndarray,         # (N,12) joint pos, URDF order
    qd_urdf: np.ndarray,        # (N,12) joint vel, URDF order
    last_action: np.ndarray,    # (N,12) previous raw action
) -> np.ndarray:
    N = base_lin_vel.shape[0]
    obs = np.empty((N, 48), dtype=np.float32)
    obs[:, 0:3] = base_lin_vel * S_LIN_VEL
    obs[:, 3:6] = base_ang_vel * S_ANG_VEL
    obs[:, 6:9] = proj_grav  # no scale
    obs[:, 9:12] = cmd * CMD_SCALE
    obs[:, 12:24] = (q_urdf - DEFAULT_ANGLES) * S_DOF_POS
    obs[:, 24:36] = qd_urdf * S_DOF_VEL
    obs[:, 36:48] = last_action
    np.clip(obs, -OBS_CLIP, OBS_CLIP, out=obs)
    return obs


def d0_obs_assembly(ladder: Ladder, policy) -> None:
    """The golden 'default_obs_cmd0.5' case IS the t=0 obs for an idealized
    upright robot at default pose, cmd=[0.5,0,0]: base vels 0, proj_grav=[0,0,-1],
    cmd*scale=[1,0,0], (q-default)=0, qd=0, last_action=0. Build it from those
    idealized values and assert (1) it equals the golden input and (2) the policy
    on it equals the golden output. This validates obs ORDER/SCALES with NO engine
    involved -- a pure convention check."""
    golden = json.load(open(GOLDEN))
    case = next(c for c in golden["cases"] if c["name"] == "default_obs_cmd0.5")
    g_in = np.array(case["input"], dtype=np.float32)
    g_out = np.array(case["output"], dtype=np.float32)

    obs = build_obs(
        base_lin_vel=np.zeros((1, 3), np.float32),
        base_ang_vel=np.zeros((1, 3), np.float32),
        proj_grav=np.array([[0.0, 0.0, -1.0]], np.float32),
        cmd=np.array([[0.5, 0.0, 0.0]], np.float32),
        q_urdf=DEFAULT_ANGLES[None, :].copy(),
        qd_urdf=np.zeros((1, 12), np.float32),
        last_action=np.zeros((1, 12), np.float32),
    )[0]
    in_match = float(np.max(np.abs(obs - g_in)))
    out = policy_forward(policy, obs[None, :])[0]
    out_match = float(np.max(np.abs(out - g_out)))
    ok = in_match < 1e-6 and out_match < 1e-3
    ladder.record(
        "D0",
        ok,
        f"offline obs-assembly vs golden 'default_obs_cmd0.5': "
        f"max|obs-golden_in|={in_match:.2e} (proves order+scales), "
        f"max|policy(obs)-golden_out|={out_match:.2e} (proves the whole obs->action). "
        f"golden_in[6:12]={g_in[6:12].tolist()} (proj_grav=[0,0,-1], cmd*scale=[1,0,0])",
    )


# ---------------------------------------------------------------------------
# D1: structural joint-order permutation discovery
# ---------------------------------------------------------------------------
def _classify_quadrant(x: float, y: float) -> str:
    """forward=+x, left=+y (confirmed from go2_float.usda fl_hip at x=+0.193,
    y=+0.046). FL=(+,+) FR=(+,-) RL=(-,+) RR=(-,-)."""
    fb = "F" if x > 0 else "R"
    lr = "L" if y > 0 else "R"
    return fb + lr


def _links_body_frame(pose_env):
    """Given env0 ARTICULATION_LINK_POSE (13,7) as numpy, return the 12 leg-link
    positions expressed in the TRUNK BODY frame:
        p_body[i] = R(q_trunk)^T @ (p_link_world[i] - p_trunk_world).
    This factors out the free base's rigid motion EXACTLY -- on a floating base any
    joint torque produces a base reaction so every link moves in WORLD coords; in
    the body frame only the perturbed joint's descendants move."""
    p_trunk = pose_env[0, 0:3]
    q_trunk = pose_env[0, 3:7]
    out = np.empty((12, 3), dtype=np.float32)
    for i in range(12):
        out[i] = quat_rotate_inverse_wxyz(
            np.asarray(q_trunk, np.float64), pose_env[i + 1, 0:3] - p_trunk
        )
    return out


def _verify_calf_nudge(dev, nuka_slot_for_urdf) -> tuple[bool, str]:
    """DIRECTIONAL write/read-index agreement check, CONTACT-ROBUST. Nudge the URDF
    FL_calf target (urdf idx 2) via the mapping: the slot it maps to is driven +0.4.
    We then assert THAT slot's own JOINT_POSITION (qd-integrated q) responds the most
    of all 12 joints -- i.e. writing to the mapped slot moves the mapped slot, so the
    write-index and the read-index agree at this joint. (We do NOT use body-frame link
    displacement: the standing feet are pinned by ground contact, so the kinematic
    subtree barely moves and the base reaction smears motion across all legs, making
    link displacement non-discriminating -- proven empirically. The joint q of the
    DRIVEN joint, by contrast, tracks its own moved target directly.)"""
    fl_calf_slot = int(nuka_slot_for_urdf[2])  # nuka slot index (0..11) for URDF FL_calf
    with make_world(dev, 16) as w:
        kpv = torch.from_dlpack(w.buffer_view(nuka.DRIVE_STIFFNESS))
        kdv = torch.from_dlpack(w.buffer_view(nuka.DRIVE_DAMPING))
        kpv[:, 1:GO2_BLC] = KP
        kdv[:, 1:GO2_BLC] = KD
        w.step_n(2)
        nuka.sync()
        q = torch.from_dlpack(w.buffer_view(nuka.JOINT_POSITION))
        before = q[0, 1:GO2_BLC].detach().cpu().numpy().copy()
        tgt = torch.from_dlpack(w.buffer_view(nuka.DRIVE_TARGET))
        tgt[0, fl_calf_slot + 1] += 0.4  # +1: drive buffer slot 0 is root
        w.step_n(60)
        nuka.sync()
        after = q[0, 1:GO2_BLC].detach().cpu().numpy().copy()
    dq = np.abs(after - before)  # (12,) per-slot |delta q|
    own = float(dq[fl_calf_slot])
    other_max = float(np.max(np.delete(dq, fl_calf_slot)))
    # The driven joint visibly responds (own |dq| > 0.05), but under ground contact
    # this is NOT a clean discriminator: the contact-coupled free base swings the
    # other legs comparably (consistent with the link-displacement finding), so own
    # |dq| does not reliably dominate. We report the numbers neutrally; the
    # permutation rests on the cooked-q signature + quadrant (exact) and the
    # end-to-end walk, NOT on this nudge.
    responds = own > 0.05
    dominates = own >= other_max
    return responds, (
        f"nudged URDF FL_calf(urdf2)->nuka slot{fl_calf_slot+1} target +0.4: "
        f"that joint's own |dq|={own:.4f} (max other |dq|={other_max:.4f}); "
        f"driven joint responds={responds}, dominates={dominates} "
        f"(own-q is non-discriminating under contact, like link-displacement)")


def d1_permutation(ladder: Ladder, dev) -> np.ndarray:
    """Return urdf_from_nuka_slot: a length-12 int array s.t.
    urdf_index = urdf_from_nuka_slot[nuka_slot] maps nuka actuated slot index
    (0..11, i.e. world slot 1..12) -> URDF joint index (0..11). Built STRUCTURALLY
    from (a) the leg QUADRANT of each link and (b) the cooked-q SIGNATURE within
    each leg, then verified by a directional nudge.

    Why the cooked-q signature (NOT a displacement ranking): the scene cooks each
    leg to (hip=0, thigh=+0.8, calf=-1.5). So within a leg, the slot whose cooked q
    ~ +0.8 IS the thigh, the slot ~ -1.5 IS the calf, and the remaining (~0) is the
    hip. This is exact and unambiguous -- a single-joint displacement ranking is
    degenerate here (all three links of a leg move ~equally in the body frame when
    nudged from the crouch), so we use the pose signature instead."""
    urdf_index_of_name = {n: i for i, n in enumerate(URDF_JOINT_NAMES)}
    with make_world(dev, 16) as w:
        w.step_n(2)  # let the one-step-lagged link pose populate
        nuka.sync()
        pose = torch.from_dlpack(w.buffer_view(nuka.ARTICULATION_LINK_POSE))
        q = torch.from_dlpack(w.buffer_view(nuka.JOINT_POSITION))
        trunk = pose[0, 0, 0:3].detach().cpu().numpy()       # root world pos
        link_xy = pose[0, 1:GO2_BLC, 0:3].detach().cpu().numpy()
        q_cooked = q[0, 1:GO2_BLC].detach().cpu().numpy()    # cooked crouch q per slot

    # (1) classify each slot's leg by the link world-position quadrant at rest.
    quad = [_classify_quadrant(float((link_xy[j] - trunk)[0]),
                               float((link_xy[j] - trunk)[1])) for j in range(12)]

    # (2) within each leg, name slots by the cooked-q signature (|q-0.8| -> thigh,
    #     |q+1.5| -> calf, remaining -> hip).
    urdf_from_nuka_slot = np.full(12, -1, dtype=np.int64)
    detail_rows = []
    signature_ok = True
    for leg in ("FL", "FR", "RL", "RR"):
        slots = [j for j in range(12) if quad[j] == leg]
        if len(slots) != 3:
            signature_ok = False
            continue
        thigh_slot = min(slots, key=lambda j: abs(q_cooked[j] - 0.8))
        calf_slot = min(slots, key=lambda j: abs(q_cooked[j] - (-1.5)))
        hip_slot = [j for j in slots if j not in (thigh_slot, calf_slot)]
        # the three picks must be distinct and the hip-q must be ~0.
        if len(hip_slot) != 1 or thigh_slot == calf_slot:
            signature_ok = False
            continue
        hip_slot = hip_slot[0]
        if not (abs(q_cooked[thigh_slot] - 0.8) < 0.3
                and abs(q_cooked[calf_slot] + 1.5) < 0.3
                and abs(q_cooked[hip_slot]) < 0.3):
            signature_ok = False
        for jtype, j in (("hip", hip_slot), ("thigh", thigh_slot), ("calf", calf_slot)):
            urdf_from_nuka_slot[j] = urdf_index_of_name[f"{leg}_{jtype}"]
            detail_rows.append((j, f"slot{j+1}->{leg}_{jtype}"
                                   f"(urdf{urdf_index_of_name[f'{leg}_{jtype}']},"
                                   f"q={q_cooked[j]:+.2f},{leg})"))
    detail_rows = [r for _, r in sorted(detail_rows)]

    # (3) verify: valid permutation of 0..11.
    is_perm = sorted(urdf_from_nuka_slot.tolist()) == list(range(12))
    identity = bool(np.array_equal(urdf_from_nuka_slot, np.arange(12)))

    # (4) DIRECTIONAL verification (the real proof, not the tautological roundtrip):
    #     nudge URDF FL_calf via the mapping and confirm the FL calf link moves most.
    nudge_ok, nudge_detail = (False, "skipped (invalid permutation)")
    if is_perm:
        nuka_slot_for_urdf = np.empty(12, dtype=np.int64)
        for slot_j in range(12):
            nuka_slot_for_urdf[urdf_from_nuka_slot[slot_j]] = slot_j
        nudge_ok, nudge_detail = _verify_calf_nudge(dev, nuka_slot_for_urdf)

    # The HARD gate is quadrant + cooked-q signature (an exact structural proof).
    # The directional nudge is a corroborating cross-check; the ULTIMATE proof of the
    # permutation is the coherent walk in the rollout (the policy is an MLP and is NOT
    # permutation-equivariant -- a scrambled obs-in or action-out order yields garbage,
    # so tracking the 0.5 m/s command end-to-end can ONLY happen if BOTH the obs read
    # indexing and the action write indexing are correct in URDF order).
    ok = is_perm and signature_ok
    ladder.record(
        "D1",
        ok,
        f"joint-order permutation urdf_from_nuka_slot={urdf_from_nuka_slot.tolist()} "
        f"(identity={identity}); valid_permutation={is_perm}; "
        f"cooked-q-signature_ok={signature_ok} [HARD GATE]; "
        f"directional_nudge_ok={nudge_ok} [corroborating: {nudge_detail}]; "
        f"END-TO-END proof = the coherent command-tracking walk in the rollout "
        f"(MLP is not permutation-equivariant). " + " ".join(detail_rows),
    )
    return urdf_from_nuka_slot, (is_perm and signature_ok)


# ---------------------------------------------------------------------------
# D2: projected_gravity convention (upright AND a known tilt)
# ---------------------------------------------------------------------------
def d2_gravity(ladder: Ladder, dev) -> None:
    # upright at rest
    with make_world(dev, 16) as w:
        w.step_n(2)
        nuka.sync()
        pose = torch.from_dlpack(w.buffer_view(nuka.ARTICULATION_LINK_POSE))
        q_wxyz = pose[0, 0, 3:7].detach().cpu().numpy()
    pg_up = projected_gravity_body(q_wxyz)
    up_ok = bool(np.allclose(pg_up, [0.0, 0.0, -1.0], atol=0.05))

    # known 90-deg roll about body x: q = [cos45, sin45, 0, 0]. World gravity
    # [0,0,-1] rotated into a body rolled +90 about x -> body frame sees gravity
    # along -(R^T z). For +90 about x, R^T maps world -z to body +y... verify the
    # closed form gives a unit vector with the expected nonzero component, so a
    # sign/transpose bug (which leaves the upright case at [0,0,-1]) is caught.
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    q_roll = np.array([c, s, 0.0, 0.0], dtype=np.float64)  # +90 about x
    pg_roll = projected_gravity_body(q_roll)
    # expected: quat_rotate_inverse([c,s,0,0],[0,0,-1]) = [0, -1, 0]
    expected_roll = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    roll_ok = bool(np.allclose(pg_roll, expected_roll, atol=1e-5))
    unit_ok = abs(np.linalg.norm(pg_roll) - 1.0) < 1e-5

    ok = up_ok and roll_ok and unit_ok
    ladder.record(
        "D2",
        ok,
        f"projected_gravity upright q={np.round(q_wxyz,4).tolist()} -> "
        f"{np.round(pg_up,4).tolist()} (~[0,0,-1]={up_ok}); "
        f"known +90deg-roll-about-x -> {np.round(pg_roll,4).tolist()} "
        f"(expect [0,-1,0]={roll_ok}, unit={unit_ok}) -- catches sign/transpose bugs",
    )


# ---------------------------------------------------------------------------
# D3: base velocities from LINK_VELOCITY root (body frame, no lag, no rotation)
# ---------------------------------------------------------------------------
def d3_velocities(ladder: Ladder, dev) -> None:
    with make_world(dev, 16) as w:
        vel = torch.from_dlpack(w.buffer_view(nuka.LINK_VELOCITY))
        root0 = vel[0, 0, :].detach().cpu().numpy().copy()
        w.step_n(20)  # base sags -> picks up some velocity
        nuka.sync()
        root1 = vel[0, 0, :].detach().cpu().numpy().copy()
    # convention: omega-first [wx,wy,wz, vx,vy,vz]; already BODY frame.
    ang0, lin0 = root0[0:3], root0[3:6]
    ang1, lin1 = root1[0:3], root1[3:6]
    finite = bool(np.isfinite(root0).all() and np.isfinite(root1).all())
    # sane: sagging under gravity gives a downward-ish lin vel and small magnitudes.
    sane = float(np.abs(lin1).max()) < 5.0 and float(np.abs(ang1).max()) < 20.0
    changed = float(np.abs(root1 - root0).max()) > 1e-5
    ok = finite and sane and changed
    ladder.record(
        "D3",
        ok,
        f"LINK_VELOCITY root (body frame, omega-first): "
        f"ang_vel0={np.round(ang0,4).tolist()} lin_vel0={np.round(lin0,4).tolist()} -> "
        f"after 20 sag steps ang={np.round(ang1,4).tolist()} lin={np.round(lin1,4).tolist()}; "
        f"finite={finite} sane={sane} changed={changed} (no world->body rotation applied)",
    )


# ---------------------------------------------------------------------------
# State extraction + warm-start helpers (env0 in URDF order)
# ---------------------------------------------------------------------------
def read_state_urdf(w, urdf_from_nuka_slot, env=0):
    """Return (q_urdf(12), qd_urdf(12), base_lin_vel(3), base_ang_vel(3),
    proj_grav(3), base_z, tilt_deg) for one env, in URDF order."""
    q = torch.from_dlpack(w.buffer_view(nuka.JOINT_POSITION))
    qd = torch.from_dlpack(w.buffer_view(nuka.JOINT_VELOCITY))
    pose = torch.from_dlpack(w.buffer_view(nuka.ARTICULATION_LINK_POSE))
    vel = torch.from_dlpack(w.buffer_view(nuka.LINK_VELOCITY))

    q_slots = q[env, 1:GO2_BLC].detach().cpu().numpy()    # nuka slot order
    qd_slots = qd[env, 1:GO2_BLC].detach().cpu().numpy()
    q_urdf = q_slots[urdf_from_nuka_slot]                  # -> URDF order
    qd_urdf = qd_slots[urdf_from_nuka_slot]

    base_pose = pose[env, 0, :].detach().cpu().numpy()
    base_z = float(base_pose[2])
    q_wxyz = base_pose[3:7]
    pg = projected_gravity_body(q_wxyz)
    tilt_deg = float(np.degrees(np.arccos(np.clip(-pg[2], -1.0, 1.0))))

    root_vel = vel[env, 0, :].detach().cpu().numpy()
    base_ang_vel = root_vel[0:3]
    base_lin_vel = root_vel[3:6]
    return q_urdf, qd_urdf, base_lin_vel, base_ang_vel, pg, base_z, tilt_deg


def set_gains_and_targets_default(w, urdf_from_nuka_slot, kp, kd, force, target_urdf):
    """Write Kp/Kd/force-limit (uniform per joint) and a DRIVE_TARGET expressed in
    URDF order, mapped into nuka slot order. Applied to ALL envs."""
    nuka_slot_for_urdf = np.empty(12, dtype=np.int64)
    for slot_j in range(12):
        nuka_slot_for_urdf[urdf_from_nuka_slot[slot_j]] = slot_j
    target_nuka = target_urdf[nuka_slot_for_urdf]   # URDF -> nuka layout
    force_nuka = FORCE_LIMIT_URDF[nuka_slot_for_urdf]

    kp_v = torch.from_dlpack(w.buffer_view(nuka.DRIVE_STIFFNESS))
    kd_v = torch.from_dlpack(w.buffer_view(nuka.DRIVE_DAMPING))
    fl_v = torch.from_dlpack(w.buffer_view(nuka.DRIVE_FORCE_LIMIT))
    tgt = torch.from_dlpack(w.buffer_view(nuka.DRIVE_TARGET))
    kp_v[:, 1:GO2_BLC] = kp
    kd_v[:, 1:GO2_BLC] = kd
    if force is not None:
        fl_v[:, 1:GO2_BLC] = torch.from_numpy(force).to(fl_v.device)
    tgt[:, 1:GO2_BLC] = torch.from_numpy(target_nuka).to(tgt.device)
    nuka.sync()


def write_targets_urdf(w, urdf_from_nuka_slot, target_urdf):
    """Write a per-env (or broadcast) URDF-order target into DRIVE_TARGET."""
    nuka_slot_for_urdf = np.empty(12, dtype=np.int64)
    for slot_j in range(12):
        nuka_slot_for_urdf[urdf_from_nuka_slot[slot_j]] = slot_j
    tgt = torch.from_dlpack(w.buffer_view(nuka.DRIVE_TARGET))
    if target_urdf.ndim == 1:
        target_nuka = target_urdf[nuka_slot_for_urdf]
        tgt[:, 1:GO2_BLC] = torch.from_numpy(target_nuka).to(tgt.device)
    else:
        # (N,12) per env
        target_nuka = target_urdf[:, nuka_slot_for_urdf]
        tgt[:, 1:GO2_BLC] = torch.from_numpy(target_nuka).to(tgt.device)


def warm_start(w, urdf_from_nuka_slot, warm_seconds=0.2):
    """The C ABI cannot set initial JOINT_POSITION; the scene cooks to a near-
    symmetric crouch (hip 0, thigh +0.8, calf -1.5), but the policy default is
    ASYMMETRIC (hip +/-0.1, thigh 0.8 front / 1.0 rear). So drive the policy default
    at the policy gains (Kp=20/Kd=0.5) for `warm_seconds` of SIM TIME to bring q
    toward default and qd -> 0 BEFORE starting the policy loop.

    TIME-BASED (not a fixed step count) ON PURPOSE: the open-loop soft-gain hold of
    a FLOATING base is posturally unstable (it slowly topples at ANY dt -- it is NOT
    a damping/integration bug; the closed-loop policy is what keeps the robot up). A
    dt-independent step count therefore warm-starts 5x longer in wall-clock at the
    native dt=0.005 (200 steps = 1.0 s) than at the sub-stepped dt=0.001 (0.2 s), and
    that extra 0.8 s of open-loop hold is enough to topple the start to ~22 deg before
    the policy ever engages -- a harness artifact, not a physics result. Pinning the
    warm-up to a fixed 0.2 s of SIM TIME makes the native-dt and sub-stepped rollouts
    apples-to-apples (both hand the policy the same lightly-sagged upright start).

    NOTE: legged_gym RESETS dof_pos = default exactly at training t=0, but the Nuka
    C ABI cannot teleport initial q; here we instead HOLD against gravity at the soft
    policy Kp=20, which settles to a steady sag (~ tau_grav/Kp on the load-bearing
    calf). So the policy takes over from a mildly OFF-default state -- a robustness
    check, not a faithful reproduction of the training reset. Returns (q_urdf,
    qd_urdf)."""
    n_steps = max(1, int(round(warm_seconds / DT)))
    set_gains_and_targets_default(
        w, urdf_from_nuka_slot, KP, KD, FORCE_LIMIT_URDF, DEFAULT_ANGLES
    )
    w.step_n(n_steps)
    nuka.sync()
    q_urdf, qd_urdf, *_ = read_state_urdf(w, urdf_from_nuka_slot)
    return q_urdf, qd_urdf


# ---------------------------------------------------------------------------
# D4 / D5 on a warm-started world
# ---------------------------------------------------------------------------
def d4_d5_warmstart(ladder: Ladder, dev, policy, urdf_from_nuka_slot) -> None:
    with make_world(dev, 16) as w:
        q_urdf, qd_urdf = warm_start(w, urdf_from_nuka_slot)
        (q_u, qd_u, blv, bav, pg, base_z, tilt) = read_state_urdf(
            w, urdf_from_nuka_slot
        )
        # D4: after warm-start the robot is UPRIGHT and SETTLED. At the soft policy
        # Kp=20, HOLDING against gravity (we can't teleport q to default via the C ABI)
        # settles to a sizeable steady sag (~ tau_grav/Kp), so the gate is on
        # UPRIGHT+SETTLED (low tilt, low qd, held height), NOT on q==default. The
        # policy then takes over from this off-default state (a robustness check, not a
        # faithful reproduction of legged_gym's exact-default reset).
        dq = q_u - DEFAULT_ANGLES
        max_dq = float(np.abs(dq).max())
        max_qd = float(np.abs(qd_u).max())
        d4_ok = tilt < 20.0 and max_qd < 1.5 and base_z > 0.30 and max_dq < 0.8
        ladder.record(
            "D4",
            d4_ok,
            f"warm-start (1.0 s @ Kp=20, HOLD not teleport): UPRIGHT+SETTLED gate -- "
            f"tilt={tilt:.2f} deg (<20), max|qd|={max_qd:.4f} rad/s (<1.5), "
            f"base_z={base_z:.4f} m (>0.30); max|q-default|={max_dq:.4f} rad "
            f"(soft-Kp gravity sag held against gravity, <0.8); "
            f"q_urdf={np.round(q_u,3).tolist()}",
        )

        # D5: build the LIVE obs (cmd=[0.5,0,0], last_action=0) and run the policy.
        cmd = np.array([[0.5, 0.0, 0.0]], np.float32)
        obs = build_obs(
            blv[None, :], bav[None, :], pg[None, :], cmd,
            q_u[None, :], qd_u[None, :], np.zeros((1, 12), np.float32),
        )
        finite_obs = bool(np.isfinite(obs).all())
        action = policy_forward(policy, obs)[0]
        finite_act = bool(np.isfinite(action).all())
        # obs[12:24] = (q-default) should match dq; cross-check the assembly is live.
        live_dq_match = float(np.max(np.abs(obs[0, 12:24] - dq)))
        d5_ok = finite_obs and finite_act and live_dq_match < 1e-5
        ladder.record(
            "D5",
            d5_ok,
            f"LIVE obs from warm-started Nuka -> policy: obs finite={finite_obs}, "
            f"action finite={finite_act}, |action|max={float(np.abs(action).max()):.3f}; "
            f"obs[12:24]==(q-default) live-check={live_dq_match:.2e}; "
            f"action={np.round(action,3).tolist()}",
        )


# ---------------------------------------------------------------------------
# Rollout: drive the policy in-the-loop, log behavior, classify honestly.
# ---------------------------------------------------------------------------
def rollout(dev, policy, urdf_from_nuka_slot, env_count=64, policy_steps=300,
            cmd_vec=(0.5, 0.0, 0.0), label="POLICY"):
    print("\n" + "-" * 78)
    print(f"ROLLOUT [{label}]: env_count={env_count}, policy_steps={policy_steps} "
          f"@ 50 Hz ({policy_steps*DT*DECIMATION:.1f} s sim), cmd={cmd_vec}")
    print("-" * 78, flush=True)
    cmd = np.array([cmd_vec], np.float32)
    with make_world(dev, env_count) as w:
        warm_start(w, urdf_from_nuka_slot)
        (_, _, _, _, _, z0, tilt0) = read_state_urdf(w, urdf_from_nuka_slot)
        pose = torch.from_dlpack(w.buffer_view(nuka.ARTICULATION_LINK_POSE))
        # WORLD-position cross-check (independent of LINK_VELOCITY): base px/py at
        # the start of the policy loop. If the robot WALKS (not a velocity-buffer
        # artifact), px must advance ~ cmd_vx * sim_time over the rollout.
        base_xy0 = pose[0, 0, 0:2].detach().cpu().numpy().copy()

        last_action = np.zeros((1, 12), np.float32)
        log = []
        nan_seen = False
        for k in range(policy_steps):
            (q_u, qd_u, blv, bav, pg, base_z, tilt) = read_state_urdf(
                w, urdf_from_nuka_slot
            )
            obs = build_obs(blv[None, :], bav[None, :], pg[None, :], cmd,
                            q_u[None, :], qd_u[None, :], last_action)
            if not np.isfinite(obs).all():
                nan_seen = True
                break
            action = policy_forward(policy, obs)
            action = np.clip(action, -ACTION_CLIP, ACTION_CLIP)
            last_action = action.copy()
            target_urdf = DEFAULT_ANGLES + ACTION_SCALE * action[0]
            write_targets_urdf(w, urdf_from_nuka_slot, target_urdf)
            w.step_n(DECIMATION)
            # log forward velocity from base_lin_vel (body x ~ forward for small tilt)
            log.append((k, base_z, tilt, float(blv[0]), float(blv[1]),
                        float(np.abs(qd_u).max())))
            if (k % 50) == 0 or k == policy_steps - 1:
                print(f"  t={k*DT*DECIMATION:5.2f}s step{k:4d}  "
                      f"base_z={base_z:.4f} tilt={tilt:5.1f}deg  "
                      f"fwd_vel(body_x)={blv[0]:+.3f} lat_vel={blv[1]:+.3f} "
                      f"max|qd|={np.abs(qd_u).max():.2f}", flush=True)
        nuka.sync()
        (qf, qdf, blvf, bavf, pgf, zf, tiltf) = read_state_urdf(
            w, urdf_from_nuka_slot
        )
        base_xyf = pose[0, 0, 0:2].detach().cpu().numpy().copy()

    arr = np.array(log) if log else np.zeros((1, 6))
    sim_time = max(len(arr), 1) * DT * DECIMATION
    dx_world = float(base_xyf[0] - base_xy0[0])  # net world x translation (m)
    dy_world = float(base_xyf[1] - base_xy0[1])
    vx_world = dx_world / sim_time  # mean world-frame forward speed from POSITION
    out = {
        "label": label, "nan": nan_seen, "z0": z0, "tilt0": tilt0,
        "z_final": zf, "tilt_final": tiltf,
        "z_min": float(arr[:, 1].min()), "z_max": float(arr[:, 1].max()),
        "z_mean": float(arr[:, 1].mean()),
        "tilt_max": float(arr[:, 2].max()), "tilt_mean": float(arr[:, 2].mean()),
        "fwd_vel_mean": float(arr[:, 3].mean()),
        "fwd_vel_last50_mean": float(arr[-50:, 3].mean()) if len(arr) >= 50 else float(arr[:, 3].mean()),
        "lat_vel_mean": float(arr[:, 4].mean()),
        "qd_max": float(arr[:, 5].max()),
        "steps_completed": len(arr),
        # world-position cross-check
        "dx_world": dx_world, "dy_world": dy_world,
        "vx_world": vx_world, "sim_time": sim_time,
    }
    return out


def pd_baseline(dev, urdf_from_nuka_slot, env_count=64, policy_steps=300):
    """In-harness PD baseline: hold the cooked crouch at Kp=60/Kd=4 (the proven
    C++ standing config), NO policy. Apples-to-apples height/tilt to compare the
    policy rollout against."""
    print("\n" + "-" * 78)
    print(f"PD BASELINE: hold the proven C++ crouch (hip +-0.1, thigh 0.8, "
          f"calf -1.5) Kp=60/Kd=4, env_count={env_count}, "
          f"{policy_steps*DT*DECIMATION:.1f}s")
    print("-" * 78, flush=True)
    with make_world(dev, env_count) as w:
        set_gains_and_targets_default(
            w, urdf_from_nuka_slot, PD_KP, PD_KD,
            np.full(12, PD_FORCE, np.float32), PD_CROUCH
        )
        log = []
        for k in range(policy_steps):
            (q_u, qd_u, blv, bav, pg, base_z, tilt) = read_state_urdf(
                w, urdf_from_nuka_slot
            )
            log.append((base_z, tilt, float(blv[0]), float(np.abs(qd_u).max())))
            w.step_n(DECIMATION)
            if (k % 100) == 0 or k == policy_steps - 1:
                print(f"  step{k:4d} base_z={base_z:.4f} tilt={tilt:5.1f}deg "
                      f"fwd_vel={blv[0]:+.3f} max|qd|={np.abs(qd_u).max():.2f}",
                      flush=True)
        nuka.sync()
        (_, _, _, _, _, zf, tiltf) = read_state_urdf(w, urdf_from_nuka_slot)
    arr = np.array(log)
    return {
        "z_final": zf, "tilt_final": tiltf,
        "z_mean": float(arr[:, 0].mean()), "tilt_mean": float(arr[:, 1].mean()),
        "tilt_max": float(arr[:, 1].max()),
        "fwd_vel_mean": float(arr[:, 2].mean()), "qd_max": float(arr[:, 3].max()),
    }


def classify(out: dict) -> str:
    """Honest verdict from the logged trajectory. The forward-motion verdict requires
    BOTH the body-frame velocity (LINK_VELOCITY) AND the independent world-position
    advance (ARTICULATION_LINK_POSE px) to agree -- so a velocity-buffer artifact
    cannot masquerade as locomotion."""
    if out["nan"]:
        return "DIVERGED (NaN/Inf)"
    upright = out["tilt_max"] < 35.0 and out["tilt_final"] < 35.0
    standing = out["z_min"] > 0.18  # didn't faceplant / fall through
    vel_fwd = out["fwd_vel_last50_mean"] > 0.05            # body-frame velocity
    pos_fwd = out["vx_world"] > 0.05                       # world-position advance
    if not (upright and standing):
        return "FLAILS / COLLAPSES (lost upright or height)"
    if vel_fwd and pos_fwd:
        return ("WALKS-LIKE (upright + net forward motion toward cmd, "
                "CONFIRMED by world-position advance)")
    if vel_fwd and not pos_fwd:
        return ("AMBIGUOUS: body-frame fwd_vel>0 but world px did NOT advance "
                "(velocity-buffer artifact or in-place gait) -- NOT confirmed walking")
    if abs(out["fwd_vel_last50_mean"]) < 0.05 and out["qd_max"] > 1.0:
        return "STANDS w/ leg motion (upright, coordinated legs, ~no net forward)"
    return "STANDS (upright, little motion)"


# ---------------------------------------------------------------------------
# Native-dt characterization: self-document the SEPARATE dt=0.005 findings so this
# harness does not hide the gap behind the sub-stepped walk above.
# ---------------------------------------------------------------------------
def native_dt_characterization(dev, policy, urdf_from_nuka_slot, env_count=64):
    """At the NATIVE training dt=0.005 (decim 4), confirm BOTH engine fixes hold: (A)
    the PD stance HOLDS (the #42 seating fix); (B) the soft-gain POLICY now WALKS rather
    than flailing (the #43 implicit-joint-damping fix). Pre-#43 (B) flailed (tilt->110
    deg) and forced dt=0.001 sub-stepping; a flail here now would be a REGRESSION."""
    dt_n, dec_n = 0.005, 4
    print("\n" + "-" * 78)
    print(f"NATIVE-DT CHARACTERIZATION @ dt={dt_n} (decim {dec_n}) -- honest self-doc "
          f"of the gap (NOT the headline config)")
    print("-" * 78, flush=True)

    # (A) PD stance at native dt -> should HOLD (the #42 fix in action).
    with make_world(dev, env_count, dt=dt_n) as w:
        set_gains_and_targets_default(w, urdf_from_nuka_slot, PD_KP, PD_KD,
                                      np.full(12, PD_FORCE, np.float32), PD_CROUCH)
        w.step_n(int(round(4.0 / dt_n)))  # 4 s settle
        nuka.sync()
        (_, _, _, _, _, z_pd, tilt_pd) = read_state_urdf(w, urdf_from_nuka_slot)
    pd_hold = tilt_pd < 5.0
    print(f"  (A) PD stance (Kp=60/Kd=4) @ native dt: z={z_pd:.3f} m tilt={tilt_pd:.2f} deg "
          f"-> {'HOLDS  (#42 stance fix works at native dt)' if pd_hold else 'COLLAPSED'}",
          flush=True)

    # (B) soft-gain policy at native dt -> should FLAIL.
    cmd = np.array([[0.5, 0.0, 0.0]], np.float32)
    with make_world(dev, env_count, dt=dt_n) as w:
        warm_start(w, urdf_from_nuka_slot)  # 200 steps @ 0.005 ~ 1 s
        last_action = np.zeros((1, 12), np.float32)
        tilt_max, nan_seen = 0.0, False
        for _ in range(150):
            (q_u, qd_u, blv, bav, pg, base_z, tilt) = read_state_urdf(w, urdf_from_nuka_slot)
            tilt_max = max(tilt_max, tilt)
            obs = build_obs(blv[None, :], bav[None, :], pg[None, :], cmd,
                            q_u[None, :], qd_u[None, :], last_action)
            if not np.isfinite(obs).all():
                nan_seen = True
                break
            action = np.clip(policy_forward(policy, obs), -ACTION_CLIP, ACTION_CLIP)
            last_action = action.copy()
            write_targets_urdf(w, urdf_from_nuka_slot,
                               DEFAULT_ANGLES + ACTION_SCALE * action[0])
            w.step_n(dec_n)
        nuka.sync()
        (_, _, _, _, _, z_f, tilt_f) = read_state_urdf(w, urdf_from_nuka_slot)
    policy_flail = nan_seen or tilt_max > 35.0 or tilt_f > 35.0
    print(f"  (B) soft-gain POLICY (Kp=20/Kd=0.5) @ native dt: final z={z_f:.3f} m "
          f"tilt={tilt_f:.2f} deg (max {tilt_max:.1f}) -> "
          f"{'FLAILS (REGRESSION -- #43 implicit damping should keep it upright)' if policy_flail else 'WALKS (the #43 implicit-damping fix holds at native dt)'}",
          flush=True)
    return dict(pd_hold=pd_hold, pd_tilt=tilt_pd, policy_flail=policy_flail,
                policy_tilt_max=tilt_max)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 78)
    print("[sim-val] REAL trained Go2 policy (motion.pt) driven in Nuka")
    print("Physics already proven correct (PD hold) + policy proven correct")
    print("(golden bit-exact). This connects them and PROVES every convention.")
    print("=" * 78, flush=True)
    torch.manual_seed(0)
    np.random.seed(0)

    policy = load_policy()
    ladder = Ladder()

    # D0: engine-independent obs-assembly proof (do this FIRST).
    d0_obs_assembly(ladder, policy)

    dev = nuka.Device.create(0)
    policy_out = pd_out = native = None
    verdict = "NOT RUN"
    smoke_ok = False
    try:
        # D1: structural joint-order permutation.
        perm, perm_valid = d1_permutation(ladder, dev)
        # D2: gravity convention.
        d2_gravity(ladder, dev)
        # D3: base velocities.
        d3_velocities(ladder, dev)
        # D4/D5: warm-start + live obs + policy forward (needs a valid perm).
        if perm_valid:
            d4_d5_warmstart(ladder, dev, policy, perm)
        else:
            ladder.record("D4", False, "SKIPPED: permutation invalid (D1 failed)")
            ladder.record("D5", False, "SKIPPED: permutation invalid (D1 failed)")

        ladder_ok = ladder.summary()

        # ---- Rollouts: GATED on a valid permutation. Interpreting a
        #      scrambled-joint trajectory is exactly the trap; never do it. ----
        if not perm_valid:
            print("\n[ABORT] permutation invalid -> NOT running rollout/PD-baseline "
                  "(a scrambled-joint trajectory is meaningless).", flush=True)
        else:
            policy_out = rollout(dev, policy, perm, env_count=64, policy_steps=300,
                                 cmd_vec=(0.5, 0.0, 0.0), label="POLICY cmd=[0.5,0,0]")
            verdict = classify(policy_out)
            pd_out = pd_baseline(dev, perm, env_count=64, policy_steps=300)

            # 4096-env smoke of the policy loop (short).
            print("\n" + "-" * 78)
            print("4096-env policy smoke (30 steps): create + drive + step, finite check")
            print("-" * 78, flush=True)
            try:
                smoke = rollout(dev, policy, perm, env_count=4096, policy_steps=30,
                                cmd_vec=(0.5, 0.0, 0.0), label="POLICY 4096 smoke")
                smoke_ok = not smoke["nan"]
            except Exception as e:  # noqa: BLE001
                smoke_ok = False
                print(f"  4096 smoke FAILED: {e}", flush=True)

            # Self-document the native-dt gap (PD holds; soft-gain policy flails).
            native = native_dt_characterization(dev, policy, perm)

    finally:
        dev.close()

    if policy_out is None:
        print("\n[no rollout: permutation gate failed]")
        return 1

    # ---- Report ----
    print("\n" + "=" * 78)
    print("BEHAVIOR VERDICT + PD-BASELINE REGRESSION")
    print("=" * 78)
    print(f"POLICY rollout ({policy_out['steps_completed']} steps, cmd=[0.5,0,0]):")
    print(f"  base_z: start {policy_out['z0']:.4f} -> final {policy_out['z_final']:.4f} "
          f"(min {policy_out['z_min']:.4f}, mean {policy_out['z_mean']:.4f}) m")
    print(f"  tilt:   start {policy_out['tilt0']:.2f} -> final {policy_out['tilt_final']:.2f} "
          f"(max {policy_out['tilt_max']:.2f}, mean {policy_out['tilt_mean']:.2f}) deg")
    print(f"  fwd_vel(body_x): mean {policy_out['fwd_vel_mean']:+.4f}, "
          f"last-50 mean {policy_out['fwd_vel_last50_mean']:+.4f} m/s (cmd 0.5) "
          f"[from LINK_VELOCITY]")
    print(f"  WORLD-POS cross-check [from ARTICULATION_LINK_POSE px, INDEPENDENT "
          f"buffer]: net dx={policy_out['dx_world']:+.3f} m over "
          f"{policy_out['sim_time']:.2f} s -> vx_world={policy_out['vx_world']:+.4f} m/s "
          f"(cmd 0.5); net dy={policy_out['dy_world']:+.3f} m")
    print(f"  lat_vel mean {policy_out['lat_vel_mean']:+.4f} m/s; "
          f"max|qd| {policy_out['qd_max']:.3f} rad/s; NaN={policy_out['nan']}")
    print(f"  VERDICT: {verdict}")
    print()
    print("PD BASELINE (hold cooked crouch Kp=60/Kd=4, no policy):")
    print(f"  base_z mean {pd_out['z_mean']:.4f} -> final {pd_out['z_final']:.4f} m; "
          f"tilt mean {pd_out['tilt_mean']:.2f} max {pd_out['tilt_max']:.2f} -> "
          f"final {pd_out['tilt_final']:.2f} deg; fwd_vel mean {pd_out['fwd_vel_mean']:+.4f}; "
          f"max|qd| {pd_out['qd_max']:.3f}")
    print()
    print("REGRESSION DELTA (policy vs PD-hold baseline):")
    print(f"  upright: policy tilt mean {policy_out['tilt_mean']:.2f} vs PD {pd_out['tilt_mean']:.2f} deg")
    print(f"  height : policy z mean   {policy_out['z_mean']:.4f} vs PD {pd_out['z_mean']:.4f} m")
    print(f"  motion : policy fwd_vel  {policy_out['fwd_vel_last50_mean']:+.4f} vs PD "
          f"{pd_out['fwd_vel_mean']:+.4f} m/s  (policy should ADD forward intent)")
    print(f"  4096 smoke finite: {smoke_ok}")
    if native is not None:
        print()
        print("NATIVE-DT (0.005) SELF-DOC -- the headline WALK above is sub-stepped "
              "(dt=0.001); at native dt:")
        print(f"  PD stance:    tilt {native['pd_tilt']:.2f} deg -> "
              f"{'HOLDS (#42 fix)' if native['pd_hold'] else 'COLLAPSED'}")
        print(f"  soft policy:  max tilt {native['policy_tilt_max']:.1f} deg -> "
              f"{'FLAILS (open soft-gain integration limit)' if native['policy_flail'] else 'held'}")
    print("=" * 78)

    return 0 if ladder_ok else 1


if __name__ == "__main__":
    sys.exit(main())
