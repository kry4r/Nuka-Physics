# Nuka Physics v0.7 – Phase 16: H1 Grasp Cup + Place on Table (v0.7 Exit Gate)

> **Master plan reference:** §3 Round 13 (S2 entry demo) + §7 v0.7 exit criteria
> **Prerequisites:** v0.7 Phases 1–15 complete
> **Blocks:** v1.0 entry
> **Exit criteria gate:** **v0.7 CLOSE**
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

Final phase of v0.7. Produce the exit demo: **Unitree H1 humanoid robot grasps a rigid cup and places it on a table**. This is the literal proof that v0.7's infrastructure stack is operational:

- Featherstone H1 articulation (v0.1)
- Rigid contact via SDF + Newton (Phase 8)
- Sparse SDF cooker (Phase 7)
- V-HACD on H1 hand meshes (Phase 6)
- LBVH broadphase (Phase 4)
- RGB + depth + tactile + F/T sensors (Phase 14)
- Diff-sim through the grasp motion (v0.5 infrastructure carried forward)

The demo runs **single-env at 1× realtime** (S2 target), not 4096-env throughput. RL training of the grasp policy can come later; this phase is about the physics infrastructure working end-to-end.

## Tech Stack

- All v0.7 phase outputs
- Unitree H1 USD assets (existing import path)
- A pre-recorded grasp trajectory (PD-controlled) for the demo — RL training in a follow-up phase

## Files to Create

- `examples/scenes/h1_grasp_cup.usda` — H1 + table + cup scene
- `examples/scenes/h1_grasp_cup_with_cloth.usda` — variant with towel on table (for v1.0 H1 wring demo prep)
- `examples/demos/h1_grasp_demo.py` — pre-scripted demo runner
- `examples/demos/h1_grasp_demo.cpp` — C++ runner via C ABI
- `tests/regression/test_h1_grasp_5s_success.cpp` — automated regression
- `docs/architecture/h1-grasp-demo.md` — demo story / images / video links
- `docs/blog/2026-XX-XX-v07-launch.md` — quarterly external output

## Files to Modify

- `src/runtime/world_builder.cpp` — H1 articulation cooker handles complex hand meshes via V-HACD
- `examples/training/` — placeholder for future H1 grasp RL config

## Tasks

### Task 7.16.1 — H1 grasp scene authoring

`examples/scenes/h1_grasp_cup.usda`:

Scene contents:
- Unitree H1 humanoid (existing import path; ~28 joints).
- Table at known position.
- Cup (cylindrical rigid body, modest mass, friction high).
- Optional: lighting setup for RGB camera.
- Sensor attachments:
  - RGB + depth camera at H1's head.
  - Tactile sensors on each finger pad (H1 has dexterous-ish hands).
  - F/T sensor at each wrist (between hand and forearm).

Custom USD attributes:
```
nuka:soft:type = "rigid"      # cup
nuka:decompose = "auto"        # hand meshes get V-HACD
nuka:soft:stiffness = ...      # if cloth variant
```

### Task 7.16.2 — Pre-scripted grasp trajectory

For v0.7 we don't have a trained policy; use a hand-authored joint trajectory:

```python
# examples/demos/h1_grasp_demo.py

import nuka
import numpy as np

dev = nuka.Device(0)
world = nuka.World(dev, "examples/scenes/h1_grasp_cup.usda", num_envs=1)

# Phase 1: arm approach (1.5 s)
# Phase 2: hand close on cup (1.0 s)
# Phase 3: lift (1.0 s)
# Phase 4: move to drop position (1.5 s)
# Phase 5: place + release (1.0 s)

trajectory = build_pd_trajectory_for_h1(...)   # joint position targets over time

for t in range(int(6.0 / 0.0166667)):   # 6 s at 60 Hz
    actions = trajectory.action_at(t * 0.0166667)
    world.write_actions(actions)
    world.step()
    if t % 6 == 0:
        save_frame(world, f"out/h1_grasp_{t//6:04d}.ppm")
```

Trajectory designed by hand; adjustments made empirically based on simulation behavior.

### Task 7.16.3 — Tactile validation

During grasp, log per-finger tactile readings:
- Pre-contact: 0 normal force.
- Contact: positive normal force, ~ cup_weight × g / num_fingers.
- Grasp slip detection: if normal force drops below threshold, alarm.

Demo includes a tactile dashboard plot.

### Task 7.16.4 — Diff-sim through grasp

Run the grasp trajectory through `nuka.differentiable_step` (v0.5 infrastructure):

```python
tape = nuka.Tape(world, checkpoint_interval=50)
loss = 0.0
for t in range(...):
    obs = nuka.differentiable_step(world, tape, trajectory.action_at(t))
    # Loss: cup position at end should match target
    if t == final:
        target_cup_pos = torch.tensor([0.4, 0.2, 0.8], device="cuda")
        loss = (world.rigid_body_positions[cup_idx] - target_cup_pos).pow(2).sum()
loss.backward()
# Verify gradients flow back through all subsystems engaged in the grasp
```

This proves the v0.7 diff-sim path holds when soft+fluid+coupling are not engaged (cup is rigid). The cloth/fluid variants test those paths in v1.0.

### Task 7.16.5 — Sensor pipeline operational test

Demonstrate the full sensor pipeline:
- RGB image rendered each frame and saved.
- Depth image rendered.
- Tactile per-finger logged.
- F/T at wrist logged.

Output video + plots.

### Task 7.16.6 — V0.7 exit gate regression test

`tests/regression/test_h1_grasp_5s_success.cpp`:

```cpp
TEST(H1Grasp, FullPipelineRunsAndCupArrives) {
    auto dev = MakeDevice();
    auto world = MakeWorld(dev, "examples/scenes/h1_grasp_cup.usda", 1);

    auto trajectory = LoadTrajectory("examples/demos/h1_grasp_trajectory.bin");
    for (int t = 0; t < trajectory.steps; ++t) {
        ApplyActions(world, trajectory.action_at(t));
        world.Step();
    }

    auto cup_pos = ReadCupPosition(world);
    EXPECT_LT(distance(cup_pos, EXPECTED_FINAL_POSITION), 0.02f)
        << "Cup did not arrive at target";

    // Energy invariant (V2)
    EXPECT_LT(EnergyDrift(world, 0, trajectory.steps), 0.05f);

    // Determinism
    auto world2 = MakeWorld(dev, ...);
    ReplayTrajectory(world2, trajectory);
    EXPECT_TRUE(StateBitExact(world, world2));
}
```

### Task 7.16.7 — Quarterly external output

`docs/blog/2026-XX-XX-v07-launch.md`:

Suggested content:
- Recap of v0.7 deliverables (15 phases worth).
- Comparison: Isaac Lab can do grasping demos too, but emphasize:
  - Strong determinism throughout (bit-exact across runs)
  - Full diff-sim through the grasp
  - Self-written sparse solver + CUDA RT (no closed SDK dependencies)
- Demo video and tactile plots.
- Roadmap to v1.0: cloth wringing + water pouring — the cross-system coupling stress test.

## Validation

- Demo runs single-env end-to-end at ~real-time.
- Cup arrives at target position.
- Tactile readings make sense.
- Diff-sim backward computes finite gradients (no NaN / Inf).
- Determinism: bit-exact two runs.
- V2 invariants hold throughout.
- Video frames render correctly.

## Exit Criteria for Phase 16 = **v0.7 EXIT**

All master plan v0.7 exit criteria:

1. ✅ XPBD soft / cloth row classes operational; oracle vs Vellum (Phase 9 + 15).
2. ✅ PBF fluid + internal density rows operational; oracle vs Flex paper (Phase 10 + 15).
3. ✅ Sparse SDF cooker + Newton contact + analytical adjoint (Phase 7 + 8).
4. ✅ V-HACD convex decomposition (Phase 6).
5. ✅ SAP → LBVH broadphase + particle uniform grid + cross-system query (Phase 4 + 5).
6. ✅ RGB + depth + tactile + force/torque sensors (Phase 14).
7. ✅ **Demo: H1 grasp cup + place on table** (this phase).

Plus rhythm: ✅ Quarterly external output published.

When all checked: **v0.7 CLOSED**. Begin v1.0 Phase 1 (cross-system coupling stability tuning).

## What This Phase Does Not Do

- No H1 wringing towel (v1.0).
- No H1 pouring water (v1.0).
- No RL training of grasp policy (separate future work; teleop / scripted trajectory suffices here).
- No real H1 hardware (v3.0).
- No multi-env throughput training (S2 is 1× realtime, not throughput).
