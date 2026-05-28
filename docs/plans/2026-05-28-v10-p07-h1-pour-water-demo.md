# Nuka Physics v1.0 – Phase 7: H1 Pour Water Demo (Rigid + Fluid Coupling)

> **Master plan reference:** §3 Round 13 (S2 polish demos) + §7 v1.0 exit
> **Prerequisites:** v1.0 Phase 1 (stability), v0.7 Phase 10 (PBF), Phase 11 (coupling rows)
> **Blocks:** v1.0 Phase 9 (exit gate)
> **Exit criteria gate:** v1.0
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

H1 humanoid robot **pours water from one cup into another**. This demo exercises the most demanding rigid-fluid coupling: rigid container (cup, jug) constrains a body of fluid that responds to robot motion and pours out under gravity.

This is a tighter test of coupling stability than the v0.7 grasp demo: the fluid must remain in the container during transport, not leak, and pour correctly when the cup tips.

## Tech Stack

- All v0.7 + v1.0 infrastructure
- PBF fluid + RigidSDFContactRow + FeatherstoneSDFContactRow

## Files to Create

- `examples/scenes/h1_pour_water.usda` — H1 + jug + cup + water-filled jug
- `examples/demos/h1_pour_water.py`
- `examples/demos/h1_pour_water.cpp`
- `tests/regression/test_h1_pour_water_volume_conservation.cpp`
- `docs/architecture/h1-pour-demo.md`

## Tasks

### Task 10.7.1 — Scene authoring

`examples/scenes/h1_pour_water.usda`:

- H1 robot at standing pose, arms forward.
- Jug (rigid cylinder, narrow neck) in right hand; partially filled with water (PBF particles).
- Target cup (open cylinder) on table.
- Lighting + RGB camera at H1 head.

Cup capacity ~250 mL → ~25,000 particles at standard PBF resolution.

### Task 10.7.2 — Initial fluid placement

Fluid cooker (v0.7 Phase 10) samples PBF particles uniformly within jug interior at simulation start. Allow a few seconds of settling for hydrostatic equilibrium before scripted pour.

### Task 10.7.3 — Scripted pour trajectory

PD-controlled joint trajectory:
- Phase 1: Hold jug stable (1.5 s).
- Phase 2: Tilt wrist to pour angle (1.0 s, ~70° tilt).
- Phase 3: Maintain pour pose (2.0 s — water flows out).
- Phase 4: Return jug to upright (1.0 s).

Target: ~50% of jug's water ends in target cup; ~5% spills outside.

### Task 10.7.4 — Volume conservation regression test

```cpp
TEST(H1PourWater, VolumeConservedAndMostlyTargeted) {
    auto world = MakePourScene();
    RunDemoTrajectory(world);

    uint32_t total_particles_initial = CountParticlesInScene(world, /*after init*/);
    uint32_t total_particles_final = CountParticlesInScene(world, /*after demo*/);
    EXPECT_EQ(total_particles_initial, total_particles_final)
        << "Fluid particles lost during simulation";

    uint32_t particles_in_target_cup = CountParticlesIn(world, target_cup_aabb);
    float pour_fraction = float(particles_in_target_cup) / total_particles_initial;
    EXPECT_GT(pour_fraction, 0.4f) << "Less than 40% of water ended in target";

    uint32_t particles_outside_cups = CountParticlesOutside(world, jug_aabb, target_cup_aabb);
    float spill_fraction = float(particles_outside_cups) / total_particles_initial;
    EXPECT_LT(spill_fraction, 0.1f) << "More than 10% spilled";
}
```

### Task 10.7.5 — Stability check during transport

Before tilting, the jug is moved through space. Fluid must remain contained (rigid-fluid SDF coupling must work robustly).

```cpp
TEST(H1PourWater, FluidContainedDuringTransport) {
    // Move jug 10 cm laterally + tilt-free
    // Fluid surface remains within jug ±2cm
}
```

### Task 10.7.6 — Video capture + demo doc

Render the demo through CUDA RT pipeline (Phase 13 + sim2real if desired); produce video. Document in `docs/architecture/h1-pour-demo.md`.

### Task 10.7.7 — Performance

This is the most computationally intensive demo so far:
- 25K particles × density iteration × 240 Hz physics
- + RGB rendering @ 30 Hz
- + cross-system coupling

Target: 1× realtime on RTX 4090 (master plan S2 1× realtime target). Step time can be ~4 ms/step (vs S1 sub-ms — different target).

## Validation

- Volume conservation throughout.
- 40%+ of water reaches target cup.
- < 10% spills.
- Fluid contained during transport.
- Determinism: bit-exact two runs.
- 1× realtime on RTX 4090.

## Exit Criteria for v1.0 Phase 7

1. Scene + trajectory authored.
2. Pour demo runs end-to-end.
3. Volume conservation + pour fraction tests pass.
4. Transport stability test passes.
5. Video captured.
6. Performance target met.

## What This Phase Does Not Do

- No RL training of pour policy.
- No real-hardware (v3.0).
- No water surface tension specific tuning (PBF defaults assumed).
- No splash effects beyond what PBF naturally produces.
