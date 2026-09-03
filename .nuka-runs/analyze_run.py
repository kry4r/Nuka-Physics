"""Compare commanded OSC targets against achieved end-effector motion."""

import sys

import numpy as np

run = sys.argv[1] if len(sys.argv) > 1 else "out/libero/pi05_gripper_fix"
z = np.load(f"{run}/rollout.npz")

eef = z["eef_positions"]
tgt = z["task_targets"]
bowl = z["initial_target_position"]
actions = z["actions"]

print("bowl:", np.round(bowl, 4).tolist())
print("ticks:", len(eef), " actions:", len(actions))

print("\n=== action stats (7D policy output) ===")
print("mean:", np.round(actions.mean(0), 4).tolist())
print("min :", np.round(actions.min(0), 4).tolist())
print("max :", np.round(actions.max(0), 4).tolist())
print("cumulative pos sum:", np.round(actions[:, :3].sum(0), 3).tolist())

print("\n=== OSC tracking: commanded target vs achieved eef ===")
print("tick      target(x,y,z)              eef(x,y,z)             lag(m)  dist_to_bowl")
for i in range(0, len(eef), max(1, len(eef) // 20)):
    lag = np.linalg.norm(tgt[i] - eef[i]) if len(tgt) else float("nan")
    dist = np.linalg.norm(eef[i] - bowl)
    print(
        "%5d  (%+.3f %+.3f %+.3f)  (%+.3f %+.3f %+.3f)  %.4f  %.4f"
        % (i, tgt[i][0], tgt[i][1], tgt[i][2], eef[i][0], eef[i][1], eef[i][2], lag, dist)
    )

lag_all = np.linalg.norm(tgt - eef, axis=1)
print("\nOSC lag: mean %.4f  max %.4f m" % (lag_all.mean(), lag_all.max()))

dist = np.linalg.norm(eef - bowl, axis=1)
best = int(np.argmin(dist))
print("closest approach: %.4f m at tick %d" % (dist[best], best))
print("  eef there:", np.round(eef[best], 4).tolist())
print("  bowl     :", np.round(bowl, 4).tolist())
print("  delta    :", np.round(eef[best] - bowl, 4).tolist())

print("\neef range: x[%.3f %.3f] y[%.3f %.3f] z[%.3f %.3f]" % (
    eef[:, 0].min(), eef[:, 0].max(),
    eef[:, 1].min(), eef[:, 1].max(),
    eef[:, 2].min(), eef[:, 2].max()))
print("target range: x[%.3f %.3f] y[%.3f %.3f] z[%.3f %.3f]" % (
    tgt[:, 0].min(), tgt[:, 0].max(),
    tgt[:, 1].min(), tgt[:, 1].max(),
    tgt[:, 2].min(), tgt[:, 2].max()))
