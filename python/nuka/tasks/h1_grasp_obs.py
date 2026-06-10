"""H1GraspEnv observation spec + on-host assembly (A3).

The grasp world is driven through the ``nuka.GraspWorld`` C-ABI binding, which
exports its per-step signals as HOST numpy arrays (``export_obs()`` +
``read_cup()`` -- one bulk device->host copy per buffer, NO per-env sync loop). So
unlike the Go2 env (zero-copy CUDA DLPack views), the grasp obs is assembled on the
HOST from those numpy dicts and then moved to CUDA in ONE ``torch.as_tensor`` round
trip per step (the task-sanctioned host round-trip -- the engine's own export is
host-resident, and the 2-DOF gripper obs is tiny).

PRIVILEGED state (NO vision): the full cup pose/vel relative to the (fixed) gripper
base + the finger joint state + the per-finger force-closure impulse + the fingertip
world positions + the last action. The cup position is expressed RELATIVE to the
gripper base so the policy sees the cup's offset from the pinch frame directly (the
base is fixed at (0,0,0.30), so this is a constant shift -- but it keeps the obs
frame-centered and ready for a movable base later).

OBS LAYOUT (dim = 27), in order:
   [ 0: 3)  cup_pos_rel      cup position - gripper_base (x,y,z)         3
   [ 3: 7)  cup_quat         cup orientation (w,x,y,z)                   4
   [ 7:10)  cup_lin_vel      cup linear velocity (vx,vy,vz)             3
   [10:13)  cup_ang_vel      cup angular velocity (wx,wy,wz)            3
   [13:15)  finger_q         the 2 prismatic finger joint positions     2
   [15:17)  finger_qdot      the 2 finger joint velocities              2
   [17:19)  finger_impulse   per-finger normal (squeeze) impulse        2
   [19:25)  fingertip_world  the 2 fingertip world positions (xyz x2)   6
   [25:27)  last_action      the previous step's clipped action         2
   total                                                                27
"""
from __future__ import annotations

import numpy as np

# The (fixed) gripper base world position the scene factory seeds
# (BuildGraspSceneBundle base_pos = {0,0,0.30}). The cup position is reported in
# world coords by read_cup; we subtract this to get the cup offset from the pinch
# frame. A constant for the fixed base (kept explicit for a future movable base).
GRIPPER_BASE_POS = np.array([0.0, 0.0, 0.30], dtype=np.float32)

# The fingertip catch plane Z (BuildGraspSceneBundle cup_center.z) -- the height a
# caught cup sits at. The cup-height reward + the drop termination are defined
# relative to this, NOT the (raised) start height.
CATCH_PLANE_Z = 0.20

# Obs sub-block sizes (must sum to H1_GRASP_OBS_DIM).
_BLOCKS = (
    ("cup_pos_rel", 3),
    ("cup_quat", 4),
    ("cup_lin_vel", 3),
    ("cup_ang_vel", 3),
    ("finger_q", 2),
    ("finger_qdot", 2),
    ("finger_impulse", 2),
    ("fingertip_world", 6),
    ("last_action", 2),
)
H1_GRASP_OBS_DIM = sum(d for _, d in _BLOCKS)  # 27
H1_GRASP_ACTION_DIM = 2

# A generous obs clip (the cup can free-fall to large negative Z / high vel before
# termination; clip keeps the network input bounded but does not distort the
# in-range signal).
OBS_CLIP = 50.0


def assemble_obs(obs_dict: dict, cup_dict: dict, last_action_np: np.ndarray) -> np.ndarray:
    """Build the (N, 27) float32 obs HOST array from the engine export dicts.

    Parameters
    ----------
    obs_dict : the ``GraspWorld.export_obs()`` dict -- keys q (N,2), qdot (N,2),
        fingertip_world_pos (N,6), finger_normal_impulse (N,2).
    cup_dict : the ``GraspWorld.read_cup()`` dict -- keys cup_pose (N,7), cup_vel
        (N,6) (+ cup_z/cup_vz/finger_contacts, unused here).
    last_action_np : (N,2) float32 host array, the previous step's clipped action.
    """
    cup_pose = cup_dict["cup_pose"]          # (N,7) [px,py,pz, qw,qx,qy,qz]
    cup_vel = cup_dict["cup_vel"]            # (N,6) [vx,vy,vz, wx,wy,wz]
    n = cup_pose.shape[0]
    out = np.empty((n, H1_GRASP_OBS_DIM), dtype=np.float32)
    out[:, 0:3] = cup_pose[:, 0:3] - GRIPPER_BASE_POS   # cup pos rel to base
    out[:, 3:7] = cup_pose[:, 3:7]                       # cup quat
    out[:, 7:10] = cup_vel[:, 0:3]                       # cup lin vel
    out[:, 10:13] = cup_vel[:, 3:6]                      # cup ang vel
    out[:, 13:15] = obs_dict["q"]                        # finger q
    out[:, 15:17] = obs_dict["qdot"]                     # finger qdot
    out[:, 17:19] = obs_dict["finger_normal_impulse"]    # per-finger impulse
    out[:, 19:25] = obs_dict["fingertip_world_pos"]      # 2 fingertip xyz
    out[:, 25:27] = last_action_np                       # last action
    np.clip(out, -OBS_CLIP, OBS_CLIP, out=out)
    return out
