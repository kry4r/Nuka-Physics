# M10 — H1 kitchen-grasp DEMO SCENE plan (kitchen + H1 beside counter + apple)

> From the kitchen-scene recon agent `afabe14d` (2026-06-15). Owner spec: H1 STANDS on the floor
> BESIDE a real kitchen counter and reaches to GRASP a nice object (apple preferred, cup fallback)
> on the counter. Grasp = RL-trained to convergence (separate track). This doc = the SCENE.

## Confirmed facts (smoke-tested)
- **Kitchen asset:** `.nuka-assets/kitchen/mjcf/kitchen.xml` (412KB, self-contained, NO `<include>`) + 47 OBJ
  meshes (5.6MB) under `.nuka-assets/kitchen/meshes/`. Robocasa kitchen. 327 bodies, 1608 geoms
  (1201 box + 364 cylinder + 49 mesh). 5 counters, cabinets, 6 walls, 2 floors, appliances.
- **Import WORKS:** `build-viewer/src/nuka_cook_scene kitchen.xml` → 327 bodies / 1608 shapes / 49
  visual_mesh_shapes / 144166 tris, EXIT 0 → 2.7MB .nks + 4.8MB .nka. Cup imports clean too.
  Gaps (benign): no `<texture>` (kitchen is flat rgba), OBJ loader drops vn/vt (render synthesizes normals).
- **Counter:** every counter top at world **z = 0.92m**. Best = `counter_1` slab 2.0m×0.65m centered at
  world **(1.7, -0.325, 0.92)**. Open floor on +y side = H1 standing space.
- **Compose:** fully supported — `.nks imports` block (`nks.cpp:856 ApplyImports`, attach_at) OR the
  python/C-ABI `Scene.load/compose/find/set_local/save` API. Mirror `h1_cup_table.nks`.
- **Render:** PBR per-instance (base_color+metallic+roughness), NO textures (fine — kitchen is flat rgba).
  Cooked subset .nka ~4.8MB (RAW-commit consistent w/ go2.nka 8MB / h1_visual.nka 88MB).

## ★ THE ONE REAL BLOCKER — primitive visual geoms don't render
A raw kitchen import renders ONLY the 49 appliance meshes; the **counters / cabinets / walls / floor are
PRIMITIVE box/cylinder geoms that render as NOTHING**. Traced: `nks::Save` routes a geom to a `.nka` MESH
chunk only if it has `mesh_vertices`; primitive visual geoms get an empty VisualMeshComponent → skipped by
`render_world.cpp`; and the primitive-tessellation fallback fires only when `loaded_visual_meshes==0`, but
the 49 appliance meshes make it >0 → fallback suppressed. **Net: floating appliances, invisible counter.**

## CONTROLLER DECISIONS (locked)
- **OBJECT = CUP (for now; apple deferred — owner 2026-06-15 "抓取对象先用杯子吧，苹果之后再说").** Use the
  in-repo cup (`.nuka-assets/newton_assets/manipulation_objects/cup/`, USD mesh `mesh.usd` + `model.usda`,
  ~6cm dia / ~8.5cm tall, the PROVEN RL grasp target). ★ The cup currently renders as a BARE convex HULL in
  h1_cup_table.nks (no visual mesh) → for the kitchen demo COOK the cup's REAL USD visual mesh into a
  renderable mesh + a ceramic PBR material (white/cream base_color, low metallic, mid-low roughness) so it
  looks like a real cup, not a faceted hull (cup.png texture won't render — no texture support — flat ceramic
  is fine). Collision stays the proven cup hull. (Apple = later: a squashed UV-sphere OBJ + red PBR; deferred.)
- **PRIMITIVE RENDER = option B (general engine fix).** Extend the cook/render so PRIMITIVE *visual* geoms
  (box/cylinder/sphere with contype=0) tessellate into renderable meshes (generalize the MakeBox/MakeSphere
  path onto the visual pipeline, or emit MESH chunks for primitive visual geoms in `nks::Save`). This is the
  directive-compliant GENERAL fix (any primitive-heavy scene renders), not a per-demo hack. CONSTRAINT: keep
  existing render goldens byte-identical (render_physics_parity etc.) — gate/guard so already-cooked gated
  scenes are unaffected; verify G2 memcmp=0. (Fallback = option C demo-local MakeBox slabs if B perturbs a gate.)
- **KITCHEN scope = render as much as renders cleanly + performs** on lavapipe; start with the full kitchen
  (327 bodies/1608 geoms) via option B, fall back to a tasteful SUBSET (counter_1 + back wall + floor + a few
  hero appliances) if perf/look needs it. Show the owner the assembled-scene STILL to approve before RL.

## Scene layout (world frame, z-up, grounded to kitchen floor z≈0)
- Kitchen = base scene (its `floor_room_g0` at z≈0 matches the engine implicit ground plane).
- **H1** standing on the floor in front of `counter_1`, facing −y: pelvis ≈ **(1.7, +0.55, 1.1)**, yaw 180°
  (hands reach toward −y / the counter). Tune the y-offset + reach against the actual arm length in a probe.
- **Apple** on the counter top ≈ **(1.7, -0.30, 0.92 + r)** (r = apple radius), centered on `counter_1`.
- Authorable as a `.nks` with an `imports` list {kitchen, h1, apple} + per-asset transforms (mirror
  `h1_cup_table.nks`, swapping the inline cube-table for the kitchen import).

## Plan (after the Go2 video review)
1. **Author the apple** (UV-sphere OBJ + red PBR material + sphere collider). Commit-ready asset.
2. **Option B primitive render** (general fix; verify existing render goldens byte-identical).
3. **Compose the kitchen-grasp scene .nks** (kitchen + H1 standing beside counter_1 + apple on counter).
4. **Render a STILL / short turntable of the ASSEMBLED scene** (H1 standing beside the counter, apple on
   it — no grasp yet) → SHOW THE OWNER to approve the look/layout BEFORE the multi-week RL grasp effort.
5. Then the RL grasp track (engine pre-work G-A..G-E + throughput gate + env + curriculum S1→S5 to convergence
   + policy-replay render) — per subagent-plans/m10-h1-grasp-forensics.md §4.

## Open for owner (low-stakes; proceeding on recommendations, will show results)
- Object: CUP for now (owner 2026-06-15); apple deferred. Cook the cup's real USD visual mesh + ceramic material.
- Full kitchen vs subset: proceeding with "as much as renders cleanly," showing the still for approval.
