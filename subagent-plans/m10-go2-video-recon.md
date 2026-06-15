# M10 / demo — Go2 beautiful-video recon (focused supplementary)

> Read-only recon agent `a5d621f4e1641f701` (opus, 2026-06-14), complementary to the main
> `wf_aeae1f0c-b20` recon. Focus = the Go2-specific path to a BEAUTIFUL rendered video (visual
> meshes + PBR), NOT the old "火柴人" (stick-figure / collision-primitive) style. Owner mandate
> (2026-06-14): BOTH H1 cup-grasp AND Go2 demos get beautiful PBR videos (see
> demo-homepage-readme-directives memory #6). Every claim grounded in a file read.

## Summary

A beautiful go2 PBR clip IS renderable offscreen on this box (lavapipe-CPU, no display, ffmpeg
present), but it is NOT a one-command path today. The two halves exist but were never joined:
1. **Visual-mesh PBR render path** — real + committed. `go2.nks`/`go2.nka` (124KB/8.2MB, cooked
   Jun 13) cook 33 visual-mesh instances from 16 real `.nka` MESH chunks (= the 16 source visual
   OBJs in `.nuka-assets/mujoco_menagerie/unitree_go2/assets/`, verified 1:1). The M8.5 T4b
   "no-stick-figure" suppression of collision proxies is LIVE in `render_world.cpp:367-421`.
2. **Trained walking policy** — `out/go2_policy/go2_walk_fixedcmd_best.pth` + `_randomcmd_ep250.pth`
   (2.3MB each, present), drives via the `nuka.World` DLPack `DRIVE_TARGET` interface (obs 48→
   action 12, validated dx +2.93 m/6 s in `examples/sim_val/go2_policy_drive.py`).

They live on opposite ends of one pipeline that was never wired together:
- The committed **Recorder** (`render_rollout.py` → `nuka.Recorder` → `nk::World`) renders visual
  meshes but steps the world with ZERO control (`SimSystem::Run` = bare `world.Step()`) → an
  unactuated floating-base go2 (`is_static:false`, no authored qpos) collapses under gravity, AND
  it renders without the beauty flags (`draw_ground`/`hero_framing` aren't even recorder fields).
- The committed **policy walk path** (`go2_demo_capture.py`) drives the real policy correctly but
  renders with the OLD stick-figure numpy rasterizer (`go2_demo_render.py`) on a collision-only
  scene (`go2_float.usda`).

**Walking does NOT require M10 RL** — the policy already exists + is Nuka-validated. M10 is RL
*regression* (retrain on unified world); a replay clip needs no M10.

**Control pick:** lowest-risk good-looking = scripted PD crouch-stance (option c, deterministic
hero product-shot). Best-looking = replay the trained policy (option a, real gait, no retrain) —
needs the G1 control hook. Recommend **c first to prove the pipe, then a for the flagship**.

## Go2 visual asset + no-stick-figure state (grounded)

- Committed + fresh: `examples/scenes/go2.nks` (124KB JSON-headed) + `examples/scenes/go2.nka`
  (8.2MB, magic `NKA1`).
- `go2.nka` = **16 MESH chunks** (fourcc 'MESH'); match the 16 source visual OBJs exactly.
  `go2.nks` declares **33 visual_mesh nodes** + metal/black/white/gray PBR palette
  (metallic/roughness/base_color) + 12-joint revolute articulation (same 12-DOF as go2_float.usda).
- **No-stick-figure (LIVE):** `src/render/render_world.cpp:367-421` — `BuildRenderWorld` counts
  `loaded_visual_meshes` (incremented only when a MESH ref resolves to triangles, line 410), then
  `if (loaded_visual_meshes == 0u)` gates the ENTIRE collision-primitive tessellation loop (line
  420). Once go2's 33 meshes load, the 23 collision proxies (incl. the 1m foot-box cubes) are
  suppressed → mesh-only. Black-robot NaN-normal fixed via `SmoothNormals`/`NormalsUnusable` in
  `FromNkaMesh` (commit d097557).
- **PBR path:** `src/render/raster/vulkan_raster_renderer.cpp` (Cook-Torrance GGX, `mesh_pbr.frag`,
  ACES tonemap, default 3-point studio rig when `world.lights` empty, ~lines 199-256). Same
  offscreen oracle `render_physics_parity` passes on lavapipe here.
- **Caveat:** hero still `/tmp/m8_5_go2_pbr.png` came from a TRANSIENT /tmp scratch exe, NOT
  committed. No committed gate emits it. The committed offscreen consumer is the Recorder, which
  does NOT set beauty flags. ⇒ any "beautiful go2" claim must be re-grounded by running the
  (to-be-wired) Recorder beauty path, not by pointing at that PNG.

## Go2 motion/control options

Unification is real: as of M9 T2 (`src/c_abi/world.cpp:4-13`), both `nuka.World.create_from_scene`
AND the Recorder cook `Scene → CookToModel → nk::World` (same world). Drive via
`FieldId::DriveTarget` (host `UploadField` or DLPack `buffer_view(DRIVE_TARGET)`).

- **(a) Policy replay** — `out/go2_policy/*.pth` present; logic in `go2_policy_drive.py`.
  BEST-looking (real gait), NO M10 retrain. **Risk MED** — needs G1 control hook (policy currently
  drives the non-rendered world); dt note: policy validated at dt=0.001 decim20 (or native 0.005
  decim4), recorder defaults 1/240.
- **(b) Recorded/scripted rollout** — `render_rollout.py` references nonexistent `go2_walk.nks`
  (stale; asset is `go2.nks`); `render_video.sh` pipes to stick-figure `go2_demo_render.py`.
  Feasible only after rewiring; **neither produces a beautiful clip as-is.**
- **(c) Scripted PD stance** — cook go2.nks, set fixed crouch DriveTarget (validated default joint
  angles), settle to clean stance, optional gentle bob/yaw. **Risk LOW, deterministic.** Standing
  not walking.
- **(d) Needs M10 RL?** NO.

**Pick: (a) policy replay flagship + (c) PD-stance de-risked fallback.** Both >> stick-figure (b).

## Render-rollout pipeline state

`Recorder.create` → cook go2.nks → nk::World + BuildRenderWorld →
`Simulation::EnableRendering(VulkanRasterRenderer, RasterOptions)` → per frame `Frame()`
(Input→`world.Step()`→TransformSync→Render) → `WritePpmP6` frame_%06u.ppm → `to_video` shells
`ffmpeg -framerate F -i frame_%06d.ppm -pix_fmt yuv420p -c:v libx264`.

- ffmpeg present (`/usr/bin/ffmpeg` 4.2.7), lavapipe ICD present. Offscreen runs with NO display
  (only the windowed viewer/present needs Xvfb).
- Asset side resolved (M8.5). Blockers: see gap list.
- Exact command once gaps close (Vulkan-ON ext installed):
  `python examples/demo/render_rollout.py --scene examples/scenes/go2.nks --frames 240 --fps 30 --width 1920 --height 1080`

## End-to-end go2-video pipeline (ordered)

1. Build a **Vulkan-ON python ext** (`_nuka_ext`) against build-viewer (flag ON, lavapipe), install
   into nuka-v03 so Recorder is real. (Never reconfigure build-cuda128.)
2. **Add a control hook** to the rendered loop (the crux): per-frame `DriveTarget` callback on
   Recorder/Simulation, OR a python driver owning `nuka.World.create_from_scene(go2.nks)` running
   the policy (a) or crouch (c) into `buffer_view(DRIVE_TARGET)` each step + render that world.
3. **Expose beauty + camera** through recorder C-ABI/python: add `draw_ground`/`hero_framing` to
   `nuka_recorder_desc_t`, set on `RasterOptions`, wire `use_camera_override`+eye/target/fov from
   python `Camera` (today dropped).
4. (Optional) re-cook reproducibility — committed asset already works; only re-cook if changing it.
5. **Render** N frames offscreen (lavapipe), PPM P6, with studio ground + 3-point rig + hero camera.
6. **Mux** ffmpeg PPM→h264 mp4 (committed `to_video` tail).
7. Eyeball a sample PNG (full GPU-color fidelity = manual on-NVIDIA check per m8-render-gates §5).

## Gap list

| # | Gap | Files | Risk | Recommendation |
|---|-----|-------|------|----------------|
| G1 | **No control script in rendered loop** — `SimSystem::Run` = bare `world.Step()`; recorder has no per-frame drive hook → floating go2 collapses. (Applies to H1 grasp too.) | `src/runtime/app/systems.cpp:152`, `src/c_abi/recorder.cpp:340-353`, `src/runtime/app/simulation.hpp` | HIGH (crux) | Add per-frame `DriveTarget` callback to Recorder/Simulation, OR python driver over `nuka.World` + offscreen render-of-current-world seam. Start opt-c PD-stance to prove pipe, then opt-a policy flagship. |
| G2 | **Beauty flags unreachable via recorder** — `draw_ground`/`hero_framing` not recorder fields; python hardcodes `use_camera_override=0`. Only the windowed viewer sets beauty. | `src/include/nuka/nuka_recorder.h:61-74`, `src/c_abi/recorder.cpp:278-299`, `python/src/nuka_ext.cpp:594-600`, `python/nuka/recorder.py` | MED | Add the two flags to desc; set on RasterOptions; wire camera-override from python Camera. Without it the clip is flat/dark, not a product-shot. |
| G3 | **Installed python ext is Vulkan-OFF** — `Recorder.create` → NOT_SUPPORTED(7) in nuka-v03 (verified live). | nuka-v03 `_nuka_ext*.so`, build-viewer, `src/CMakeLists.txt` | MED | Build `_nuka_ext` against build-viewer (flag ON), install to nuka-v03. build-viewer libnuka.so exists. |
| G4 | **Stale rollout filename** — `render_rollout.py` → `go2_walk.nks` (nonexistent), skips go2; `render_video.sh` uses stick-figure rasterizer. | `examples/demo/render_rollout.py:45`, `examples/demo/render_video.sh:71`, `examples/demo/go2_demo_render.py` | LOW | Point rollout at `examples/scenes/go2.nks`; replace the stick-figure render step with the Recorder path. |
| G5 | **No authored initial stance in go2.nks** — base `is_static:false`, no qpos → rest sags under gravity. | `examples/scenes/go2.nks` | LOW | Drive the validated crouch DriveTarget from frame 0; settle a few frames. Don't re-cook unless adding an authored stance. |
| G6 | **dt/decimation mismatch** — policy validated dt=0.001 (decim20) / native 0.005 (decim4); recorder defaults 1/240. | `src/c_abi/recorder.cpp:265`, `examples/sim_val/go2_policy_drive.py` | LOW | Pass dt=0.005 to recorder, apply policy every 4 sub-steps, mirroring `go2_demo_capture.py`. |

## Owner decisions

1. **Flagship motion: walking-policy replay (a) vs PD-stance hero (c)?** Recommend **c first (pipeline
   proof) then a (flagship)** — both reuse the same render plumbing; a only adds policy-in-loop.
2. **Where to add the control hook (G1)?** (i) generalize Recorder/Simulation with a per-frame
   `DriveTarget` callback vs (ii) drive a plain `nuka.World` in python + separate offscreen render.
   Recommend **(i)** — matches "imported scene data + per-scene control script", keeps Recorder as
   the single video tool. **NOTE: this same hook serves the H1 grasp video — shared infra.**
3. **Beauty flags through recorder (G2)?** Additive, D1-safe (default OFF). Recommend **yes**.
4. **Re-cook go2.nks for crouch stance?** Recommend **NOT** (drive it instead) — keep asset stable.

## Recommended clip spec

- Resolution 1920×1080 (recorder D6 default; 1280×720 for fast first pass, final 1080p).
- 30 fps ~8 s = 240 frames (policy 50 Hz; 30 fps = slight slow-mo, or 50 fps true real-time).
  Stance-only clip 4–6 s.
- Camera: 3/4 front-right-above, robot ~60–65% of frame, ~38° FOV (M8.5 T4b hero default), e.g.
  eye≈(2.4,-2.0,1.2), target≈(base, ~0.4 z), up=(0,0,1) — or `hero_framing=true` to auto-frame AABB.
- Dressing: `draw_ground=true` (dark studio disc), default 3-point rig (go2.nks authors no lights),
  bg {10,12,16} or a touch lighter.
- Codec: libx264, `-pix_fmt yuv420p`, `-crf 18–20`.
- Motion: opt-a → forward trot cmd≈[0.5,0,0]; opt-c → settle to crouch + gentle ±5° yaw/bob.

## ★ Cross-cutting insight (controller)

G1 (control hook), G2 (beauty flags), G3 (Vulkan-ON ext) are **shared infrastructure for BOTH the
Go2 video AND the H1 cup-grasp video**. Build them once, both demos consume them. This is exactly
the unified-world model: the Recorder is ONE general video tool; the difference between the go2 clip
and the h1 clip is the loaded scene (.nks) + the per-frame control script (policy/crouch vs grasp
choreography), never a special render path. Fold this into the main m10-recon.md demo-video track.
