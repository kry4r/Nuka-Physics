# Nuka Physics v0.7 – Phase 15: V1 Oracle Catalog Expansion (XPBD / PBF / SDF)

> **Master plan reference:** §3 Round 11 (V1 oracle table) + §6 Validation Architecture
> **Prerequisites:** v0.7 Phases 9 (XPBD), 10 (PBF), 8 (SDF)
> **Blocks:** v0.7 Phase 16 (H1 demo runs oracle-validated subsystems)
> **Exit criteria gate:** v0.7
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

Extend the V1 oracle infrastructure (established v0.1 Phase 7) with golden trajectories for the v0.7 subsystems:
- XPBD soft / cloth vs **Houdini Vellum**
- PBF fluid vs **NVIDIA Flex paper reconstruction** (already partial in Phase 10)
- SDF contact vs **MuJoCo 3.0 SDF plugin**

Each oracle is a fixed scene, deterministically replayed through the reference engine, with golden trajectory stored in Git LFS. CI regression tests compare per commit.

## Tech Stack

- Houdini Vellum (external; CI box must have Vellum-licensed Houdini or pre-baked golden files)
- NVIDIA Flex source code (for paper reproduction; may not be needed if pre-baked)
- MuJoCo 3.0 with SDF plugin (Python harness)
- Git LFS

## Files to Create

### Vellum oracle (cloth + soft tet)

- `tools/oracle/vellum/generate_cloth_drape.hou` — Houdini hipnc file
- `tools/oracle/vellum/generate_cloth_drape.py` — Python wrapper that runs Vellum
- `tests/oracle/golden/xpbd_cloth_drape_5s.bin` — Git LFS
- `tests/oracle/golden/xpbd_tet_compress.bin`
- `tests/oracle/golden/xpbd_cloth_self_collision.bin`
- `tests/oracle/test_xpbd_vellum_cloth_drape.cpp`
- `tests/oracle/test_xpbd_vellum_tet_compress.cpp`

### Flex paper fluid

- `tools/oracle/flex/ball_in_water.py` — generate / load Flex paper reconstruction
- `tests/oracle/golden/pbf_ball_in_water.bin`
- `tests/oracle/golden/pbf_dam_break.bin`
- `tests/oracle/test_pbf_flex_ball_in_water.cpp`
- `tests/oracle/test_pbf_flex_dam_break.cpp`

### MuJoCo 3.0 SDF oracle

- `tools/oracle/mujoco_sdf/generate_sdf_contact_cases.py`
- `tests/oracle/golden/sdf_contact_sphere_box.bin`
- `tests/oracle/golden/sdf_contact_thin_panel.bin`
- `tests/oracle/test_sdf_contact_mujoco_oracle.cpp`

### Documentation

- `tests/oracle/README.md` — extended

## Tasks

### Task 7.15.1 — Vellum cloth drape oracle

Cloth drape scenario:
- Square cloth (32x32 grid).
- Falls under gravity onto a sphere.
- Steady state after 5 s.

Generate Vellum golden trajectory:
- Run in Houdini Vellum with documented stiffness / damping params.
- Save per-frame vertex positions to binary (1200 frames @ 240 Hz × 1024 vertices × 12 bytes = ~14 MB → Git LFS).

Engine test:
- Load same cloth mesh + same params.
- Run 5 s in XPBD.
- Per-frame compare against golden; tolerance < 1% vertex position (master plan §6 V1 table).

### Task 7.15.2 — Vellum tet compression oracle

Tet-mesh ball under load:
- Sphere mesh (tet count ~1000) sitting on rigid plane.
- Load applied from above ramping to compression.
- Vellum reference: known equilibrium shape.
- Engine: same params; verify shape match.

### Task 7.15.3 — Vellum cloth self-collision

A cloth folded onto itself:
- 64x64 cloth dropped on a small box.
- Folds form.
- Self-collision should prevent penetration.

Verify: per-vertex distance to neighbors > thickness within tolerance.

### Task 7.15.4 — Flex paper fluid reconstruction

From Macklin & Müller 2014 Section 6:
- 16x16x16 = 4096 fluid particles in a box.
- Ball (radius = 4 × particle_h) released from above into water.
- Splash and settle.

Generate golden:
- Either reimplement the paper case in Flex (if Flex source available) OR
- Reconstruct from paper figures (less precise; use as qualitative oracle).

Tolerance per master plan: volume conservation ±2%, surface waveform qualitative.

### Task 7.15.5 — Dam break (classic CFD benchmark)

Standard test case for free-surface flow:
- Tank, water column at one end.
- Release water, observe propagation.
- Compare wave front timing + height to literature.

### Task 7.15.6 — MuJoCo SDF contact oracle

MuJoCo 3.0 has SDF collision via plugin. Generate test cases:
- Sphere-box contact (varies positions/orientations).
- Sphere-thin-panel contact (the hard case).
- Box-box contact.

For each, MuJoCo records contact point + normal + penetration. Engine computes same; comparison tolerance per master plan: normal angular error < 0.5°.

### Task 7.15.7 — V1 oracle harness integration

Each test file follows pattern from v0.1 Phase 7:

```cpp
TEST(XpbdClothDrape, MatchesVellumGoldenTrajectory) {
    auto ctx = MakeCtx();
    auto world = SetUpClothDrapeScene(ctx);
    auto golden = LoadGoldenBinary("tests/oracle/golden/xpbd_cloth_drape_5s.bin");

    for (int frame = 0; frame < 1200; ++frame) {
        nuka_world_step(world);
        auto state = DownloadClothVertices(world);
        for (uint32_t v = 0; v < state.size(); ++v) {
            float err = length(state[v] - golden[frame].vertices[v]);
            float rel_err = err / golden[frame].cloth_diagonal;
            EXPECT_LT(rel_err, 0.01f) << "frame " << frame << " vertex " << v;
        }
    }
}
```

CI runs the full oracle suite nightly; fast subset on every PR.

### Task 7.15.8 — Update validation strategy doc

`tests/oracle/README.md`:

- Lists all golden files (engine version + scene + tolerance).
- Documents the regeneration commands for each oracle.
- Notes the Git LFS size implications.

## Validation

- Each new oracle test passes within stated tolerance.
- Git LFS storage stays manageable (cumulative golden files < 1 GB).
- Regeneration scripts are reproducible (same scene + same engine → same golden).
- CI runs fast subset (one scene per subsystem) in < 60 s.

## Exit Criteria for v0.7 Phase 15

1. Vellum cloth drape oracle passes.
2. Vellum tet compression oracle passes.
3. Vellum cloth self-collision oracle passes.
4. Flex ball-in-water passes within volume tolerance.
5. Dam break passes qualitative check.
6. MuJoCo SDF contact (3 cases) passes within angular tolerance.
7. `tests/oracle/README.md` updated.
8. Git LFS for golden files works in CI.

## What This Phase Does Not Do

- No sim-to-real real-hardware oracle (v3.0).
- No diff-rendering oracle (v2.0).
- No regression on visual aesthetics (V4 demo suite handles that).
- Does not add new physics — purely validation infrastructure.
