"""Evaluate object motion relative to a gripper during supported transport."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def eef_poses_from_states(states):
    angle = np.linalg.norm(states[:, 3:6], axis=1)
    scale = np.divide(np.sin(angle/2), angle, out=np.full_like(angle, 0.5), where=angle > 1e-7)
    return np.column_stack((states[:, :3], np.cos(angle/2), states[:, 3:6]*scale[:, None]))


def transport_metrics(targets, eef_poses, grasp_mask, support_force, action_ticks, actions, dt):
    edges = np.diff(np.pad(np.asarray(grasp_mask, dtype=np.int8), (1, 1)))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    sustained = np.flatnonzero((ends-starts)*dt >= 0.1)
    if not len(sustained):
        return dict(transport_verified=False, max_transport_drift_m=None,
                    transport_begin_s=None, transport_end_s=None)
    begin = int(starts[sustained[0]])
    stop = len(targets)
    supported = np.flatnonzero((np.arange(len(targets)) > begin) & (support_force > 0.05))
    opened = action_ticks[(action_ticks > begin) & (actions[:, 6] < 0)]
    if len(supported):
        stop = min(stop, int(supported[0]))
    if len(opened):
        stop = min(stop, int(opened[0]))
    if stop <= begin:
        return dict(transport_verified=False, max_transport_drift_m=None,
                    transport_begin_s=begin*dt, transport_end_s=stop*dt)
    poses = np.asarray(eef_poses[begin:stop], dtype=np.float64)
    displacement = targets[begin:stop]-poses[:, :3]
    quaternion = poses[:, 3:]/np.linalg.norm(poses[:, 3:], axis=1, keepdims=True)
    vector = -quaternion[:, 1:]
    cross = 2*np.cross(vector, displacement)
    relative = displacement+quaternion[:, :1]*cross+np.cross(vector, cross)
    reference_count = min(len(relative), max(1, round(0.05/dt)))
    reference = np.mean(relative[:reference_count], axis=0)
    drift = float(np.linalg.norm(relative-reference, axis=1).max())
    return dict(transport_verified=bool(drift <= 0.005), max_transport_drift_m=drift,
                transport_begin_s=begin*dt, transport_end_s=stop*dt)


def audit_episode(run):
    data = np.load(run/"rollout.npz")
    targets, grippers, loads = data["target_positions"], data["gripper_positions"], data["contact_loads"]
    poses = data["eef_poses"] if "eef_poses" in data else eef_poses_from_states(data["states"])
    distance = np.linalg.norm(poses[:, :3]-targets, axis=1)
    grasp = np.all(grippers < 0.012, axis=1) & np.all(loads[:, :2] > 0.05, axis=1)
    grasp &= (targets[:, 2] > data["initial_target_position"][2]+0.04) & (distance < 0.09)
    result = transport_metrics(targets, poses, grasp, loads[:, 2], data["action_ticks"],
                               data["actions"], float(data["physics_dt"]))
    (run/"transport_audit.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    summary_path = run/"summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.setdefault("success_before_transport_audit", summary["success"])
    summary.update(result)
    summary["success"] = bool(summary["success_before_transport_audit"] and result["transport_verified"])
    summary_path.write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    print(json.dumps(audit_episode(parser.parse_args().run), indent=2))
