"""On-GPU 48-dim Go2 observation builder + joint-order permutation.

This is the load-bearing convention layer the whole RL stack rests on. It ports
the EXACT legged_gym Go2 obs/action contract proven bit-exact in
``examples/sim_val/go2_policy_drive.py`` and documented in
``docs/plans/2026-05-30-v03-unitree-policy-config.md``, but composes everything
with **torch ops on the zero-copy DLPack CUDA views** -- no per-step numpy /
host round-trip (the sim_val harness uses a numpy/CPU path for a single-policy
diagnostic; at 4096 envs that dominates wall-clock, so it must NOT be copied).

The 48-dim obs (legged_gym order/scales), per env:

    [ 0: 3] base_lin_vel (body frame) * 2.0
    [ 3: 6] base_ang_vel (body frame) * 0.25
    [ 6: 9] projected_gravity (world [0,0,-1] rotated into body frame; unit)
    [ 9:12] command[:3] * [2.0, 2.0, 0.25]
    [12:24] (q_urdf - default_dof_pos) * 1.0       (joints in URDF order)
    [24:36] qd_urdf * 0.05                          (joints in URDF order)
    [36:48] last_action (previous raw clipped policy output)
    -> clip the whole vector to [-100, 100].

Action (12): target_q = default_dof_pos + 0.25 * action; Kp=20, Kd=0.5
(uniform), target_vel=0. Written to DRIVE_TARGET slots [1:] in URDF order via
the permutation.

THE SILENT KILLER -- joint order: Nuka enumerates the 12 actuated leg joints in
its own internal slot order (buffer slots 1..12); the policy expects the URDF
order [FL,FR,RL,RR x hip,thigh,calf]. We discover the Nuka-slot <-> URDF-index
permutation STRUCTURALLY (leg quadrant from the link world pose + cooked-q
signature within each leg) and apply it as a **device LongTensor index op**,
symmetric on obs-in (q/qd -> URDF order) and action-out (URDF -> Nuka slots).
"""

from __future__ import annotations

import numpy as np
import torch

import nuka

# ---------------------------------------------------------------------------
# Contract constants (URDF / training joint order: FL,FR,RL,RR x hip,thigh,calf)
# ---------------------------------------------------------------------------
GO2_BLC = 13          # base_link_count: slot 0 = root, slots 1..12 = leg joints
GO2_ACTION_DIM = 12   # actuated DOF
GO2_OBS_DIM = 48

URDF_JOINT_NAMES = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
]

# default standing angles in URDF order (legged_gym go2 default_joint_angles).
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

# PD gains (uniform across the 12 actuated joints) + per-joint-TYPE URDF effort
# limits (hip/thigh 23.7, calf 35.55) in URDF order.
KP = 20.0
KD = 0.5
FORCE_LIMIT_URDF = np.array([23.7, 23.7, 35.55] * 4, dtype=np.float32)


# ---------------------------------------------------------------------------
# Quaternion math (torch, GPU) -- gravity into the body frame
# ---------------------------------------------------------------------------
def quat_rotate_inverse_wxyz(q_wxyz: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate world vector(s) ``v`` into the BODY frame, i.e. ``R(q)^T @ v``.

    Mirrors Isaac Gym ``quat_rotate_inverse`` (the closed form
    ``a*v - 2 w (q_vec x v) + 2 q_vec (q_vec . v)``, note the MINUS on the cross
    term -- a sign/transpose error passes the upright case and only fails under
    tilt). ``q_wxyz`` is ``(..., 4)`` [w,x,y,z]; ``v`` is ``(..., 3)`` and
    broadcasts. Returns ``(..., 3)``. All-torch, runs on whatever device the
    inputs live on (cuda here).
    """
    w = q_wxyz[..., 0:1]
    qvec = q_wxyz[..., 1:4]
    a = (w * w - (qvec * qvec).sum(-1, keepdim=True)) * v
    b = 2.0 * w * torch.cross(qvec, v.expand_as(qvec), dim=-1)
    c = 2.0 * qvec * (qvec * v).sum(-1, keepdim=True)
    return a - b + c


def projected_gravity_body(base_quat_wxyz: torch.Tensor) -> torch.Tensor:
    """World gravity unit vector ``[0,0,-1]`` rotated into the body frame.

    ``base_quat_wxyz``: ``(N,4)`` root-link quaternion (w-first). Returns
    ``(N,3)`` on the same device.
    """
    g = torch.tensor(
        [0.0, 0.0, -1.0], dtype=base_quat_wxyz.dtype, device=base_quat_wxyz.device
    ).expand(base_quat_wxyz.shape[0], 3)
    return quat_rotate_inverse_wxyz(base_quat_wxyz, g)


# ---------------------------------------------------------------------------
# Structural joint-order permutation discovery
# ---------------------------------------------------------------------------
def _classify_quadrant(x: float, y: float) -> str:
    """forward=+x, left=+y (confirmed from go2_float.usda). FL=(+,+) FR=(+,-)
    RL=(-,+) RR=(-,-)."""
    fb = "F" if x > 0 else "R"
    lr = "L" if y > 0 else "R"
    return fb + lr


def discover_joint_permutation(device) -> np.ndarray:
    """Return ``urdf_from_nuka_slot`` (length-12 int64) s.t.
    ``urdf_index = urdf_from_nuka_slot[nuka_slot]`` maps a Nuka actuated slot
    index (0..11, i.e. buffer slot 1..12) -> URDF joint index (0..11).

    Built STRUCTURALLY (ported from ``go2_policy_drive.py``'s D1) from (a) the
    leg QUADRANT of each link's world position and (b) the cooked-q SIGNATURE
    within each leg (the scene cooks each leg to hip~0, thigh~+0.8/1.0,
    calf~-1.5, so within a leg the slot whose cooked q ~ +0.8/1.0 IS the thigh,
    ~ -1.5 IS the calf, remaining ~0 IS the hip -- exact and unambiguous).

    Runs on a THROWAWAY world (create -> step -> classify -> destroy) so it never
    touches the persistent env world's pristine creation pose: the leg-link
    quadrant needs ``step_n(2)`` to populate the one-step-lagged
    ARTICULATION_LINK_POSE non-root slots, but a freshly created env world's
    first ``reset()`` obs must read the un-stepped creation pose. Isolating
    discovery here keeps the two apart.

    Raises ``RuntimeError`` if the discovered mapping is not a valid permutation
    or the cooked-q signature is inconsistent (a structural mismatch the rest of
    the stack must NOT silently run past).
    """
    urdf_index_of_name = {n: i for i, n in enumerate(URDF_JOINT_NAMES)}
    scene = _go2_scene_path()
    with nuka.World.create_from_scene(device, scene, 4) as w:
        w.step_n(2)  # populate the one-step-lagged non-root link poses
        nuka.sync()
        pose = torch.from_dlpack(w.buffer_view(nuka.ARTICULATION_LINK_POSE))
        q = torch.from_dlpack(w.buffer_view(nuka.JOINT_POSITION))
        trunk = pose[0, 0, 0:3].detach().cpu().numpy()
        link_xyz = pose[0, 1:GO2_BLC, 0:3].detach().cpu().numpy()
        q_cooked = q[0, 1:GO2_BLC].detach().cpu().numpy()

    quad = [
        _classify_quadrant(float((link_xyz[j] - trunk)[0]),
                           float((link_xyz[j] - trunk)[1]))
        for j in range(12)
    ]

    urdf_from_nuka_slot = np.full(12, -1, dtype=np.int64)
    for leg in ("FL", "FR", "RL", "RR"):
        slots = [j for j in range(12) if quad[j] == leg]
        if len(slots) != 3:
            raise RuntimeError(
                f"joint permutation discovery: leg {leg} has {len(slots)} slots "
                f"(expected 3); quadrants={quad}"
            )
        thigh_slot = min(slots, key=lambda j: abs(q_cooked[j] - 0.9))
        calf_slot = min(slots, key=lambda j: abs(q_cooked[j] - (-1.5)))
        rest = [j for j in slots if j not in (thigh_slot, calf_slot)]
        if len(rest) != 1 or thigh_slot == calf_slot:
            raise RuntimeError(
                f"joint permutation discovery: leg {leg} cooked-q signature "
                f"degenerate (slots={slots}, q={q_cooked[slots]})"
            )
        hip_slot = rest[0]
        if not (abs(q_cooked[thigh_slot]) > 0.5
                and abs(q_cooked[calf_slot] + 1.5) < 0.4
                and abs(q_cooked[hip_slot]) < 0.4):
            raise RuntimeError(
                f"joint permutation discovery: leg {leg} cooked-q out of band "
                f"(hip={q_cooked[hip_slot]:.3f}, thigh={q_cooked[thigh_slot]:.3f}, "
                f"calf={q_cooked[calf_slot]:.3f})"
            )
        for jtype, j in (("hip", hip_slot), ("thigh", thigh_slot), ("calf", calf_slot)):
            urdf_from_nuka_slot[j] = urdf_index_of_name[f"{leg}_{jtype}"]

    if sorted(urdf_from_nuka_slot.tolist()) != list(range(12)):
        raise RuntimeError(
            f"joint permutation discovery: not a valid permutation of 0..11: "
            f"{urdf_from_nuka_slot.tolist()}"
        )
    return urdf_from_nuka_slot


def _go2_scene_path() -> str:
    # go2_locomotion.usda is a byte-identical copy of go2_float.usda (the proven
    # flat-ground floating-base Go2 scene) -- a dedicated, non-symlink locomotion
    # asset so the training task owns its scene. Identical content => identity
    # joint permutation => the golden obs-oracle / permutation tests stay green.
    return "/root/Nuka-Physics/examples/scenes/go2_locomotion.usda"


# ---------------------------------------------------------------------------
# Go2 obs/action machinery -- holds the GPU permutation + constant tensors.
# ---------------------------------------------------------------------------
class Go2ObsBuilder:
    """On-GPU composer of the 48-dim legged_gym Go2 obs + URDF<->Nuka action map.

    Construct once per world (the permutation is discovered on a throwaway world
    via :func:`discover_joint_permutation` and uploaded as device LongTensors).
    Every per-step call is pure torch on the engine's zero-copy DLPack views.
    """

    def __init__(self, device, world, *, dtype=torch.float32):
        self.world = world
        self.num_envs = world.env_count
        self._torch_device = torch.device("cuda")
        self.dtype = dtype

        # Discover the permutation on a throwaway world (does NOT step `world`).
        urdf_from_nuka_slot = discover_joint_permutation(device)
        self.urdf_from_nuka_slot_np = urdf_from_nuka_slot
        # nuka_slot_for_urdf is the inverse: nuka_slot = nuka_slot_for_urdf[urdf_idx]
        nuka_slot_for_urdf = np.empty(12, dtype=np.int64)
        for slot_j in range(12):
            nuka_slot_for_urdf[urdf_from_nuka_slot[slot_j]] = slot_j
        self.nuka_slot_for_urdf_np = nuka_slot_for_urdf

        # Device index tensors: obs maps Nuka-slot -> URDF order; action maps
        # a URDF-order vector -> Nuka-slot order.
        self.urdf_from_nuka_slot = torch.as_tensor(
            urdf_from_nuka_slot, dtype=torch.long, device=self._torch_device
        )
        self.nuka_slot_for_urdf = torch.as_tensor(
            nuka_slot_for_urdf, dtype=torch.long, device=self._torch_device
        )

        # Constant tensors on device.
        self.default_angles = torch.as_tensor(
            DEFAULT_ANGLES, dtype=dtype, device=self._torch_device
        )
        self.cmd_scale = torch.as_tensor(
            CMD_SCALE, dtype=dtype, device=self._torch_device
        )
        self.force_limit_urdf = torch.as_tensor(
            FORCE_LIMIT_URDF, dtype=dtype, device=self._torch_device
        )

        # Cache the zero-copy DLPack views (they alias the live engine buffers,
        # so re-reading them each step reflects the new state without re-export).
        self._q = torch.from_dlpack(world.buffer_view(nuka.JOINT_POSITION))
        self._qd = torch.from_dlpack(world.buffer_view(nuka.JOINT_VELOCITY))
        self._pose = torch.from_dlpack(world.buffer_view(nuka.ARTICULATION_LINK_POSE))
        self._vel = torch.from_dlpack(world.buffer_view(nuka.LINK_VELOCITY))
        self._tgt = torch.from_dlpack(world.buffer_view(nuka.DRIVE_TARGET))
        # AUTHORITATIVE per-env root pose (env_count, 7) == [px,py,pz, qw,qx,qy,qz].
        # Unlike ARTICULATION_LINK_POSE (root slot is FK-lagged one step), base_pose
        # is correct IMMEDIATELY after reset_envs -- so the just-reset envs read the
        # upright orientation here, not the stale fallen quat. Used ONLY for the
        # post-reset projected_gravity of done envs; the normal per-step obs keeps
        # reading the lagged (golden-pinned) ARTICULATION_LINK_POSE.
        self._base_pose = torch.from_dlpack(world.buffer_view(nuka.BASE_POSE))

    # -- gain setup (called once at construction / after reset, NOT per step) --
    def apply_pd_gains(self) -> None:
        """Write Kp=20 / Kd=0.5 (uniform) + the per-joint URDF force limits onto
        the 12 actuated joints (slots [1:]). Idempotent; cheap; call once."""
        kp = torch.from_dlpack(self.world.buffer_view(nuka.DRIVE_STIFFNESS))
        kd = torch.from_dlpack(self.world.buffer_view(nuka.DRIVE_DAMPING))
        fl = torch.from_dlpack(self.world.buffer_view(nuka.DRIVE_FORCE_LIMIT))
        kp[:, 1:GO2_BLC] = KP
        kd[:, 1:GO2_BLC] = KD
        # force limits in URDF order mapped into Nuka slots.
        fl[:, 1:GO2_BLC] = self.force_limit_urdf[self.nuka_slot_for_urdf]

    # -- per-step read helpers (URDF order, all on GPU) --
    def q_urdf(self) -> torch.Tensor:
        """(N,12) joint positions in URDF order."""
        return self._q[:, 1:GO2_BLC].index_select(1, self.urdf_from_nuka_slot)

    def qd_urdf(self) -> torch.Tensor:
        """(N,12) joint velocities in URDF order."""
        return self._qd[:, 1:GO2_BLC].index_select(1, self.urdf_from_nuka_slot)

    def base_quat_wxyz(self) -> torch.Tensor:
        """(N,4) root-link world quaternion (w-first)."""
        return self._pose[:, 0, 3:7]

    def base_pos(self) -> torch.Tensor:
        """(N,3) root-link world position."""
        return self._pose[:, 0, 0:3]

    def base_ang_vel(self) -> torch.Tensor:
        """(N,3) base angular velocity, ALREADY body frame (no world->body)."""
        return self._vel[:, 0, 0:3]

    def base_lin_vel(self) -> torch.Tensor:
        """(N,3) base linear velocity, ALREADY body frame (no world->body)."""
        return self._vel[:, 0, 3:6]

    def projected_gravity(self) -> torch.Tensor:
        """(N,3) world [0,0,-1] rotated into the body frame via the base quat."""
        return projected_gravity_body(self.base_quat_wxyz())

    def base_quat_wxyz_auth(self) -> torch.Tensor:
        """(N,4) root world quaternion (w-first) from the AUTHORITATIVE base_pose
        view -- correct immediately after reset_envs (no one-step FK lag). Use this
        for the post-reset orientation of just-reset envs (see env.py autoreset)."""
        return self._base_pose[:, 3:7]

    def projected_gravity_auth(self) -> torch.Tensor:
        """(N,3) projected gravity from the AUTHORITATIVE (un-lagged) base quat.

        Correct immediately after reset_envs, so the just-reset envs report the
        upright orientation (~[0,0,-1]) instead of the stale fallen quat that the
        FK-lagged ARTICULATION_LINK_POSE root slot still carries until the next
        Step. Used to patch obs[done, 6:9] in the autoreset path."""
        return projected_gravity_body(self.base_quat_wxyz_auth())

    # -- obs composition (the headline: all torch, zero numpy) --
    def compute_obs(
        self, command: torch.Tensor, last_action: torch.Tensor
    ) -> torch.Tensor:
        """Compose the 48-dim obs on GPU.

        ``command``: (N,3) [vx, vy, wyaw] on cuda. ``last_action``: (N,12)
        previous raw clipped action on cuda. Returns (N,48) cuda float32,
        clipped to [-100, 100]. No numpy, no host round-trip.
        """
        q_urdf = self.q_urdf()
        qd_urdf = self.qd_urdf()
        obs = torch.cat(
            (
                self.base_lin_vel() * S_LIN_VEL,
                self.base_ang_vel() * S_ANG_VEL,
                self.projected_gravity(),
                command * self.cmd_scale,
                (q_urdf - self.default_angles) * S_DOF_POS,
                qd_urdf * S_DOF_VEL,
                last_action,
            ),
            dim=-1,
        )
        return obs.clamp_(-OBS_CLIP, OBS_CLIP)

    # -- action -> DRIVE_TARGET (URDF order -> Nuka slots, on GPU) --
    def write_action(self, action: torch.Tensor) -> torch.Tensor:
        """Map a (N,12) URDF-order raw action to PD targets and write them into
        DRIVE_TARGET slots [1:] (in Nuka slot order). Returns the CLIPPED action
        (N,12) for storage as the next step's ``last_action``. All on GPU."""
        action = action.clamp(-ACTION_CLIP, ACTION_CLIP)
        target_urdf = self.default_angles + ACTION_SCALE * action  # (N,12) URDF
        # URDF order -> Nuka slot order. This is the EXACT inverse of the obs
        # read: obs uses ``q_slots[urdf_from_nuka_slot]`` (gather Nuka slots by the
        # slot->urdf map), so the action write uses the inverse index
        # ``nuka_slot_for_urdf`` -- i.e. ``target_urdf[:, nuka_slot_for_urdf]`` --
        # mirroring go2_policy_drive.py's proven ``write_targets_urdf``. (Using
        # ``urdf_from_nuka_slot`` here would be WRONG for any non-identity
        # permutation; for go2_float the permutation is identity so the two
        # coincide, but we drive the correct inverse so a future scene with a
        # scrambled joint order stays correct.)
        self._tgt[:, 1:GO2_BLC] = target_urdf.index_select(1, self.nuka_slot_for_urdf)
        return action

    def tilt_deg(self) -> torch.Tensor:
        """(N,) base tilt from upright in degrees (acos of -proj_grav_z)."""
        pg = self.projected_gravity()
        cos_up = (-pg[:, 2]).clamp(-1.0, 1.0)
        return torch.rad2deg(torch.arccos(cos_up))
