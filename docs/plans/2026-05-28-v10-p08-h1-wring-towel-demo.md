# Nuka Physics v1.0 – Phase 8: H1 Wring Towel Demo (Rigid + Cloth + Fluid 3-Way Coupling)

> **Master plan reference:** §3 Round 13 (S2 polish demos) + §7 v1.0 exit
> **Prerequisites:** v1.0 Phase 1 (stability), v0.7 Phase 9 (XPBD), 10 (PBF), 11 (coupling rows)
> **Blocks:** v1.0 Phase 9 (exit gate)
> **Exit criteria gate:** v1.0
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

The ultimate v1.0 demo: H1 robot **wrings a wet towel**. This is **3-way coupling** — robot hands (Featherstone link) grip cloth (XPBD); cloth holds water (PBF particles via the cloth's interstices); twisting motion squeezes water out.

This stresses every coupling pair from master plan §3 Round 7. If it works stably and at 1× realtime, the v1.0 infrastructure is proven for S2.

## Tech Stack

- All v0.7 + v1.0 infrastructure
- Particularly: XPBD bend + distance, PBF + soft-fluid coupling, Featherstone-link-vs-cloth contact
- Sub-step orchestrator (v1.0 Phase 1) — likely needed for stability

## Files to Create

- `examples/scenes/h1_wring_towel.usda` — H1 + wet towel (cloth + embedded fluid particles)
- `examples/demos/h1_wring_towel.py`
- `examples/demos/h1_wring_towel.cpp`
- `tests/regression/test_h1_wring_water_extracted.cpp`
- `docs/architecture/h1-wring-demo.md`

## Tasks

### Task 10.8.1 — Scene authoring

`examples/scenes/h1_wring_towel.usda`:

- H1 robot, both hands grasping ends of a cloth.
- Cloth: 32×32 grid (1024 vertices), modeled with XPBD distance + bend constraints.
- Water: ~10,000 PBF particles distributed *within the cloth volume* (a thin slab geometry around the cloth mesh).
- Material params: cloth slightly hydrophilic (capillary-like interactions can be approximated via tighter coupling radius between cloth and fluid).

### Task 10.8.2 — Initial state — saturated cloth

At start, the cloth particles + fluid particles are co-located in a thin volumetric region. As the demo proceeds, twisting motion forces particles to migrate.

### Task 10.8.3 — Scripted wring trajectory

Both hands grasp cloth at opposite ends:
- Phase 1: Grip cloth (1.0 s).
- Phase 2: Twist hands in opposing directions (3.0 s — fluid begins to squeeze out).
- Phase 3: Hold twisted (2.0 s — finishing the squeeze).
- Phase 4: Untwist (2.0 s).

### Task 10.8.4 — Water extraction measurement

The demo's success metric: how much water ends below the cloth?

```cpp
TEST(H1Wring, WaterExtractedDuringTwist) {
    auto world = MakeWringScene();
    uint32_t initial_water_in_cloth = CountFluidParticlesInside(cloth_aabb);
    RunDemoTrajectory(world);
    uint32_t final_water_in_cloth = CountFluidParticlesInside(cloth_aabb);
    uint32_t water_extracted = initial_water_in_cloth - final_water_in_cloth;
    float extraction_fraction = float(water_extracted) / initial_water_in_cloth;
    EXPECT_GT(extraction_fraction, 0.3f) << "At least 30% water should be wrung out";
}
```

### Task 10.8.5 — Stability under deformation

The cloth undergoes large deformation (twisting). Verify:
- No cloth self-intersection (XPBD `TrianglePointContactRow`).
- No fluid particles tunneling through cloth.
- No NaN / blowup.

Likely requires sub-step orchestrator at 4× substeps (master plan §3 Round 13 risk).

### Task 10.8.6 — Determinism + diff-sim

```cpp
TEST(H1Wring, Determinism) {
    auto world1 = MakeWringScene();
    auto world2 = MakeWringScene();
    RunDemo(world1);
    RunDemo(world2);
    EXPECT_TRUE(StateBitExact(world1, world2));
}

TEST(H1Wring, DiffSimNoBlowup) {
    auto world = MakeWringScene();
    auto tape = MakeTape(world, /*K=*/50);
    /* ... forward 100 steps ... */
    auto loss = ComputeLoss(world);
    loss.backward();
    EXPECT_LT(MaxGradient(world), 1e4f);
}
```

### Task 10.8.7 — Performance budget

This is the most expensive demo. Per step on RTX 4090:
- 1024 cloth particles + their constraint network: ~150 µs
- 10K fluid particles + density iteration: ~400 µs
- Cross-system coupling rows: ~200 µs
- H1 Featherstone: ~50 µs
- Cloth-fluid coupling: ~150 µs
- Total: ~1 ms/step base × 4 substeps = 4 ms/step
- 1× realtime at 60 Hz = 16.67 ms budget — comfortable margin.

### Task 10.8.8 — Visual rendering

CUDA RT renders cloth + water + robot at 30 fps for the demo video.

### Task 10.8.9 — Demo doc

`docs/architecture/h1-wring-demo.md`: scenario design, parameter tuning notes (substep count, compliance values), failure modes encountered + fixes.

## Validation

- ≥ 30% water extracted during wring.
- No cloth self-intersection.
- No fluid leak / tunneling.
- Determinism passes.
- Diff-sim does not explode.
- 1× realtime on RTX 4090.

## Exit Criteria for v1.0 Phase 8

1. Scene + trajectory authored.
2. Wring demo runs end-to-end stably.
3. Water extraction test passes.
4. No-tunneling test passes.
5. Determinism + diff-sim health.
6. Performance target met.

## What This Phase Does Not Do

- No RL training of wring policy.
- No real-hardware.
- No actual moisture / drying physics (water just leaves cloth volume; doesn't model evaporation).
- No fluid-saturated-cloth specific simulation (capillary effects approximated, not modeled rigorously).
