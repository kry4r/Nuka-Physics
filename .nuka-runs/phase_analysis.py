"""Correlate the policy's gripper phase with its distance to the bowl."""

import json
import sys

import numpy as np

run = sys.argv[1] if len(sys.argv) > 1 else "out/libero/pi05_gripper_fix"
z = np.load(f"{run}/rollout.npz")
summary = json.load(open(f"{run}/summary.json", encoding="utf-8"))

eef = z["eef_positions"]
bowl = z["initial_target_position"]
actions = z["actions"]
ticks = z["action_ticks"]
grip = z["gripper_positions"]

dist = np.linalg.norm(eef - bowl, axis=1)

print("action  tick   dist_to_bowl   eef_z    dz_cmd   grip_cmd  finger_open")
print("                                                (-1=open,+1=close)")
for i in range(0, len(actions), max(1, len(actions) // 30)):
    t = int(ticks[i])
    t = min(t, len(eef) - 1)
    print(
        "%6d  %5d   %.4f       %.3f   %+.3f   %+.3f    %.5f"
        % (i, t, dist[t], eef[t, 2], actions[i, 2], actions[i, 6], grip[t, 0])
    )

close_cmds = np.nonzero(actions[:, 6] > 0)[0]
open_cmds = np.nonzero(actions[:, 6] < 0)[0]
print("\nclose commands: %d / %d actions" % (len(close_cmds), len(actions)))
if len(close_cmds):
    first = int(close_cmds[0])
    t = min(int(ticks[first]), len(eef) - 1)
    print("first CLOSE at action %d (tick %d)" % (first, t))
    print("  distance to bowl then: %.4f m" % dist[t])
    print("  eef z then           : %.4f  (bowl z %.4f)" % (eef[t, 2], bowl[2]))
print("open commands : %d / %d actions" % (len(open_cmds), len(actions)))

print("\nclosest approach %.4f m at tick %d" % (dist.min(), int(np.argmin(dist))))
print("gripper dim: mean %+.3f  frac_close %.2f" % (
    actions[:, 6].mean(), float((actions[:, 6] > 0).mean())))
print("\nsummary success:", summary["success"])
print("grasp_candidate_ticks:", summary["grasp_candidate_ticks"])
