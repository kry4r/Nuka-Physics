#!/usr/bin/env python
"""nuka.UnionWorld smoke -- the G2 python substrate in ~60 lines.

Constructs the H1 whole-body UNION world (N=2), prints the scene/shape
metadata, drives 120 steps of the factory's reference HOLD choreography PD
(computed in PYTHON from the obs export -- the same seam a future H1UnifiedEnv
uses), and demonstrates the per-env seams (set_table_enabled, set_base_pose,
reset_envs). Run from the repo root in the nuka-v03 env, GPU lock held:

    python examples/training/h1_union_smoke.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
os.chdir(_REPO_ROOT)

import nuka  # noqa: E402


def pd_actions(w, obs, table):
    """The reference PD: tau[dof] = kp*(target - q[dof]) - kd*qdot[dof], clamped."""
    n, ad = w.env_count, w.action_dim
    q, qd = np.asarray(obs["q"]), np.asarray(obs["qdot"])
    a = np.zeros((n, ad), dtype=np.float32)
    dof = np.asarray(table["dof"], dtype=np.int64)
    u = (table["kp"][None, :] * (table["target"][None, :] - q[:, dof])
         - table["kd"][None, :] * qd[:, dof]).astype(np.float32)
    lim = np.asarray(table["tlim"], dtype=np.float32)
    mask = lim > 0.0
    u[:, mask] = np.clip(u[:, mask], -lim[mask], lim[mask])
    a[:, dof] = u
    return a.reshape(-1)


def main():
    mjcf = os.path.join(".nuka-assets", "newton_assets", "unitree_h1", "mjcf",
                        "h1_with_hand.xml")
    if not os.path.exists(mjcf):
        print("SKIP: assets not present")
        return 0
    with nuka.Device.create(0) as dev:
        w = nuka.UnionWorld.create(dev, env_count=2)
        info = w.scene_info()
        print(f"[smoke] dims: n={w.env_count} action_dim={w.action_dim} "
              f"base_dof={w.base_dof} fingertips={w.num_fingertips} "
              f"feet={w.num_feet} links={w.link_count}")
        print(f"[smoke] scene: table_z={info['table_height']:.4f} "
              f"cup_m={info['cup_mass']:.2f} total_m={info['total_mass']:.2f} "
              f"dt={info['dt']:.6f} place_found={info['place_found']}")
        print(f"[smoke] grip dof columns: {np.asarray(w.grip_dofs()).tolist()}")

        hold = w.drive_table("hold")
        obs0 = w.export_obs()
        print(f"[smoke] t=0 (settled snapshot): base_z={obs0['base_pose'][0, 2]:.4f} "
              f"cup_z={obs0['cup_pose'][0, 2]:.4f}")

        # env 1: lift the base 0.5 m (the airborne IC seam) -- its feet unload.
        pose = np.array(obs0["base_pose"][1], dtype=np.float32, copy=True)
        pose[2] += 0.5
        w.set_base_pose(1, pose)
        w.set_table_enabled(1, False)

        for s in range(120):
            obs = w.export_obs()
            w.step_with_actions(pd_actions(w, obs, hold))
        obs = w.export_obs()
        print(f"[smoke] t=120: env0 cup_z={obs['cup_pose'][0, 2]:.4f} "
              f"foot_rows={int(obs['foot_rows'][0])} "
              f"table_rows={int(obs['table_rows'][0])} | env1 (airborne, no table) "
              f"base_z={obs['base_pose'][1, 2]:.4f} foot_rows={int(obs['foot_rows'][1])}")
        assert int(obs["foot_rows"][0]) > 0, "env0 lost foot contact"
        assert np.all(np.isfinite(np.asarray(obs["q"]))), "non-finite q"

        w.reset_envs(np.array([1], dtype=np.uint32), seed=3)
        obs = w.export_obs()
        print(f"[smoke] after reset(1): env1 base_z={obs['base_pose'][1, 2]:.4f} "
              f"cup_z={obs['cup_pose'][1, 2]:.4f} table_enabled={w.table_enabled(1)}")
        w.destroy()
    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
