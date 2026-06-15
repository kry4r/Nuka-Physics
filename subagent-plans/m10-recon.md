# M10 RECON + EXECUTION PLAN — H1 Cup-Grasp Demo Video (headline) + RL Regression Milestone + README

> Synthesis of 6 recon facets (M10-RL-SCOPE, DEMO-VIDEO-PIPELINE, RENDER-REALITY, STEPPING-API, ASSETS-LFS, README-HOMEPAGE).
> Ground state verified against code at `origin/master == d41af62` (HEAD `d41af62`). READ-ONLY recon; nothing built/edited/committed.
> Companion to `subagent-plans/m9-recon.md` / `m11-recon.md`. M10 is the LAST milestone.
> ★ Go2-video companion recon: `subagent-plans/m10-go2-video-recon.md` (the go2 beautiful-video path).

---

## 0. OWNER DECISIONS LOCKED + CO-HEADLINE (2026-06-15)

**Owner mandate update (2026-06-14):** the demo deliverable is TWO beautiful PBR videos — **H1 cup-grasp AND Go2** — both with visual meshes + PBR (NOT the old collision-primitive "火柴人" / stick-figure look). Go2 is a CO-HEADLINE, not an afterthought. See `demo-homepage-readme-directives` memory #6 + `m10-go2-video-recon.md`.

**Three owner decisions (AskUserQuestion 2026-06-15):**
- **Go2 motion = WALKING (policy replay).** Replay the trained `out/go2_policy/*.pth` gait. Mechanism: a python pre-pass loads the policy + rolls it out through `nuka.World` headless to DUMP a per-step `DriveTarget` trajectory to a file; the C++ video tool REPLAYS that trajectory + renders (torch stays out of C++). dt/decimation must mirror `go2_policy_drive.py` (native 0.005 / decim 4).
- **M10 RL scope = FULL.** All of T10–T18: go2 PPO short-run regression + greenfield H1-grasp RL env (smoke-learn / catch-eval rises, NOT multi-session converged) + throughput + SG-spec re-narration + carry-forward contracts. (OD-5 → option 1.)
- **H1 88 MB visual .nka = RAW COMMIT.** Cook the H1 visual meshes into a committed `.nka` (go2.nka precedent) so fresh clones reproduce the render. (OD-2 → option 1.) ⇒ **CR-3 / T2's "avoid the 88 MB" rationale is SUPERSEDED for reproducibility:** still use T2's renderer change as the draw mechanism if convenient, but the visual asset IS committed RAW so the demo does NOT depend on local `.nuka-assets/`. Physics stays on the FROZEN `h1_cup_table.nks` (union cook, D1-untouched); the committed visual `.nka` feeds ONLY the render-instance meshes (hand-bound, CR-2 option A). Do NOT edit the frozen `h1_cup_table.{nks,nka}` — the committed visual asset is a NEW file.

**Controller rulings locked (the rest of §6, resolved):** OD-1 = (a) scripted choreography (demo decoupled from RL); OD-3 = (d) GIF/WEBP inline + mp4 GitHub-Release link; OD-4 = (b) full grasp PERFORMANCE clip; OD-6 = (b)→(a) re-verify-then-close G0; OD-7 = copy logo to `docs/media/logo.png`, social-preview = manual owner step; OD-8 = controller drafts README copy/version, OWNER approves before the final push.

**★ Unified video TOOL ruling (controller, supersedes the H1-only framing of §2):** ONE general C++ video tool (build-viewer-gated, offscreen lavapipe, NO python-ext Vulkan rebuild — runs like the render gates) serves BOTH demos. Difference = the loaded scene (.nks) + the per-frame CONTROL SOURCE (H1: the proven `h1_grasp_lift` union-cook TableTorque choreography; Go2: a replayed policy `DriveTarget` trajectory). Same render+publish+PBR+encode tail. This is the unified-world model applied to tooling: one tool, scene-data + control-script, never a per-demo render path. The python Recorder path stays as named M10 cleanup (T6/T14), NOT the headline mechanism (CR-1: `recorder.cpp:248` cooks the generic single-artic cook → wrong topology + zero drive for H1).

**Revised phase order:** Phase A (H1 grasp video) → **Phase A2 (Go2 walking video)** → Phase B (README + push) → Phase C (M10 RL FULL) → Phase D (carry-forward) → M10-end ultracode review+fix.

---

## 1. EXECUTIVE SUMMARY

**Headline deliverable:** an H1 humanoid **cup-grasp DEMO rendered to an mp4 VIDEO** (approach → close → lift → sustained friction-only hold), produced **offscreen on THIS lavapipe-CPU box**, embedded into a streamlined+beautified `README.md`, and pushed to GitHub.

**Chosen end-to-end pipeline (lowest risk, all halves individually proven on this box):** a **new C++ demo exe** that reuses, verbatim, the two pieces that already work here — (a) the grasp physics from `tests/scenario/h1_grasp_lift.cpp` (`nks::Load(h1_cup_table.nks)` → `CookSceneToUnionTemplate` → `BuildNkUnionModel` → `nk::World`, with the per-step `TableTorque` PD drive over `drive_hold/drive_rest/drive_close` and the `TableEnabled<-0` lift trigger), and (b) the render+publish+encode tail from `tests/scenario/render_physics_parity.cpp` + `src/c_abi/recorder.cpp` (offscreen `VulkanRasterRenderer` on lavapipe → `WritePpmP6` → ffmpeg/libx264 mp4). The single genuinely new piece is **binding the union model's cup body + a few H1 link rows to `RenderInstance`s** (the union cook emits no `SceneMap`). This decouples the headline video from M10 RL entirely — the grasp is a **scripted choreography** (the per-scene control SCRIPT mandated by `[[unified-world-no-special-grasp-binding]]`), not a trained policy.

**Why NOT the existing python `render_rollout.py`/Recorder:** VERIFIED — `src/c_abi/recorder.cpp:248` cooks via `cook::CookToModel(scene,1)` (the GENERIC single-articulation cook that takes only the first articulation and treats the cup as a loose body), and `Simulation::Frame()` injects **zero** per-step drive. So the python recorder renders a **passive, wrong-topology, non-grasping** rollout today (the `render_rollout.py` docstring claiming H1 "END-TO-END" grasp is **stale/aspirational**). The working 51-DOF friction grasp exists ONLY on the C++ union cook. Reaching it from python needs M10-scale new C-ABI surface (union cook + torque drive-mode + name→dof map + `TableEnabled` write + rigid-body velocity reads) — out of scope for the deadline.

**M10-RL track (the broader milestone, NOT on the video's critical path):** per `docs/plans/2026-06-11-nk-core-platform-refactor.md` L503-505 the literal asks are (a) `train_go2_ppo.py` short run reaches prior reward trajectory; (b) `train_h1_grasp_ppo.py` smoke-learns (catch-eval rises) — **this file no longer exists** (M9 T8 `4fe7970` deleted the entire H1-grasp RL surface + its special-world C-ABI), so it is a **greenfield rebuild on the generic `nuka.World`**, NOT a port of the deleted fixed-base 2-finger `GraspWorld` env; (c) union throughput meets 100M env-steps ≤24h; (d) re-narrate `docs/specs/2026-06-10-h1-whole-body-rl-grasp-spec.md` into nk terms. Plus the M9→M10 carry-forward contracts (go2 PD-stance, per-link wrench, autoreset, base-pose, control-modes) re-asserted as nk-world tests.

---

## 2. THE CHOSEN DEMO-VIDEO PIPELINE (concrete)

```
examples/scenes/h1_cup_table.nks   (FROZEN — read only)
   │  scene::nks::Load  (ApplyImports re-imports h1_with_hand.xml → visual tris in mesh_vertices)
   ▼
scene::cook::CookSceneToUnionTemplate(scene, N=1)         ── the UNION cook (51-DOF H1 + cup + 30 fingertip spheres)
   ▼  CookedUnionScene { tmpl, cup_local_index, grip_dofs, drive_hold/rest/close (link+dof pre-resolved) }
coresident::BuildNkUnionModel(cooked.tmpl)  →  nk::Model (drive_mode=1 direct-TORQUE)
   ▼
nk::World  (union SolverConfig: vel_iters=64, fold_drive_damping=0)
   │
   │   PER FRAME s = 0..S:
   │     if s == kLiftAt(180):  UploadField(FieldId::TableEnabled, 0)     ← lift trigger
   │     tab = (s<10 ? drive_hold : s<100 ? drive_rest : drive_close)     ← phase select
   │     DownloadField(Q), DownloadField(Qdot)
   │     TableTorque(tab, q, qdot, kill_grip=false, &torque)              ← PD: u=kp*(tgt-q)-kd*qdot, clamp ±tlim
   │     UploadField(FieldId::DriveTarget, torque)                        ← drive_mode=1 → torque
   │     w.Step()
   │     ── PUBLISH FK → RENDER (the NEW glue) ──
   │     DownloadField(BodyPose[cup_local]) + LinkPose[h1 rows] + BasePose
   │     compose world_xform = fk * cached_visual_local  (per render_physics_parity:186-244)
   │     VulkanRasterRenderer::Render(rw, RasterOptions{draw_ground=ON, hero_framing=ON, use_camera_override=ON})
   │     WritePpmP6(frame_%06d.ppm)
   ▼
ffmpeg -framerate 30 -i frame_%06d.ppm -pix_fmt yuv420p -c:v libx264 out/h1_grasp.mp4
   ▼  (README) ffmpeg mp4 → looping GIF/WEBP (docs/media/) for inline embed + mp4 → GitHub Release link
```

- **Control path = C++ demo exe driving scripted choreography** (NOT python `h1_grasp_choreo`, NOT a trained policy). `h1_grasp_choreo.py` stays the python reference; it cannot drive a live world (no union cook / name→dof / per-step field IO exposed from python). The C++ `CookedUnionScene` already carries the resolved drive tables — use them directly; do NOT re-resolve via python.
- **Render = offscreen lavapipe raster, PBR beauty ON.** VERIFIED locally: `nuka_render_physics_parity_test` PASS on the real `h1_cup_table` scene (llvmpipe, 3 instances, G1/G2/G3 green, 2.7s, no display); `nuka_render_raster_smoke_test` PASS (D1 byte-identical). PBR stack (Cook-Torrance GGX + ACES + sRGB + 3-point studio rig) is wired always-on in `mesh_pbr.frag`; `draw_ground`/`hero_framing` default OFF and must be set in the exe via `RasterOptions`.
- **Arc / cadence (D3 ruling below):** APPROACH/REST (0–100, settled curl) → CLOSE (100–180) → LIFT (180) → HOLD (180–420, sustained friction-only carry). ~420 steps. Render EVERY step for an 8× slow-mo (420 frames @ 30fps ≈ 14 s, showcases grip/contact detail). **End on the stable hold; do NOT include BITE (a drop)** in the showcase clip — reserve BITE for a separate honesty clip. Camera pinned at the right-hand/cup via `RasterOptions.use_camera_override` (eye ≈ (2.6,-2.6,1.7), target ≈ (0,0,0.95)).

**Lowest-risk path & why:** the C++ exe reuses 100%-proven code (grasp drive AND render both individually green on this exact box) and adds only the RenderWorld↔union-row binding. The python path would render the wrong topology + no grasp and needs a large new public C-ABI surface. The asset story is solved WITHOUT committing 88 MB or editing frozen assets by rendering the **import-time visual triangles live from local `.nuka-assets/`** (renderer change, T2) — see Risk/Debt §5 and OWNER D-LFS.

---

## 3. PHASED TASK LIST

### PHASE A — DEMO VIDEO (THE HEADLINE — do FIRST, independent of RL)

| id | title | files touched | builds/wires | verification gate | dep |
|----|-------|---------------|--------------|-------------------|-----|
| **T1** | Lift the proven grasp driver into a shared, reusable header | NEW `src/scene/cook/union_drive.hpp` (or `examples/demo/h1_grasp_driver.hpp`); SOURCE: `tests/scenario/h1_grasp_lift.cpp:86-123,284-306` | Extract `TableTorque` + phase-window select (`kWindowEnd=10/kRestEnd=100/kLiftAt=180`) + `TableEnabled<-0` toggle into a callable `DriveStep(world, CookedUnionScene, step, kill_grip)` — verbatim PD law, no behavior change | builds in build-cuda128; `git grep TableTorque` resolves to the new header; `h1_grasp_lift.cpp` still PASSES if it adopts the header (else leave the test untouched and duplicate) | none |
| **T2** | Render import-time visual meshes (make the H1 VISIBLE) | `src/render/render_world.cpp:392-412` (the `vis.mesh.fourcc==NkaTagMesh()` early-return at :394); reads `rec.mesh_vertices` populated by `nks.cpp ApplyImports→LoadMjcf` | Materialize ApplyImports-loaded `VisualMeshComponent` triangles (`mesh_vertices`, no `.nka` MESH ref) into in-memory render meshes, so the FROZEN `h1_cup_table.nks` renders the H1 body from live local STL. Default-OFF-safe: only fires when `vis.mesh.Empty() && !mesh_vertices.empty()` | `nuka_render_physics_parity_test` STILL byte-identical (G2 memcmp=0) on the mesh-less light scene (new branch dormant there); a manual render of `h1_cup_table` shows H1 link geometry (non_bg pixel count jumps) | none |
| **T3** | Bind the UNION model's poses to RenderInstances (the real new work) | NEW exe `examples/demo/h1_grasp_video.cpp` (build-viewer-gated); `src/scene/cook/union_cook.hpp` (read `cup_local_index`/link order, no edit); `src/render/render_world.hpp` | Hand-build a `RenderWorld` (Decision D-RW option A): cup instance `pose_source = Body row cup_local`; H1 link instances bound to deterministic union link rows; reuse render_physics_parity publish (DownloadField LinkPose/BodyPose/BasePose → compose `fk*visual_local`) | exe builds under build-viewer; renders 1 frame with cup + H1 visible (non_bg > parity baseline) | T1, T2 |
| **T4** | Fuse drive + render into the per-frame demo loop | `examples/demo/h1_grasp_video.cpp` | The single fused loop: `DriveStep(s)` → `w.Step()` → publish FK → `Render(RasterOptions{draw_ground,hero_framing,use_camera_override})` → `WritePpmP6`. ~420 frames, arc per §2 | exe runs offscreen on lavapipe to PPM seq; visual sanity: cup approaches→closes→lifts→held (NOT static, NOT dropped) | T1, T3 |
| **T5** | Encode mp4 + derive inline GIF/WEBP | `examples/demo/h1_grasp_video.cpp` (shell ffmpeg) OR reuse `nuka_recorder_to_video`; `examples/demo/render_video.sh` | `ffmpeg -framerate 30 -i frame_%06d.ppm -pix_fmt yuv420p -c:v libx264 out/h1_grasp.mp4`; second pass mp4/frames → looping GIF or WEBP (≤5 MB, ~720–960px) for README inline | `out/h1_grasp.mp4` plays; GIF/WEBP < 5 MB renders inline | T4 |
| **T6** | (OPTIONAL, owner-gated) beauty knobs in the recorder C-ABI | `src/include/nuka/nuka_recorder.h` (+`draw_ground`/`hero_framing`/honor `use_camera_override`); `src/c_abi/recorder.cpp`; `python/src/nuka_ext.cpp` (drop hardcoded `use_camera_override=0`) | Plumb the three render-beauty fields so the python one-command path can also produce the hero look (NOT required for the C++ headline; enables the clean python path later) | render_physics_parity/raster_smoke STILL byte-identical (render-only opts, default-OFF preserves G2 red-line) | none (parallel) |

### PHASE B — README STREAMLINE + BEAUTIFY + PUBLISH (after the mp4 exists)

| id | title | files touched | builds/wires | verification gate | dep |
|----|-------|---------------|--------------|-------------------|-----|
| **T7** | Commit demo media | `docs/media/h1_grasp.gif` (or `.webp`); `out/h1_grasp.mp4` → GitHub Release asset (manual) | Inline-embeddable hero media mirroring `docs/media/go2_walk_4096env.gif` (2.1 MB precedent, raw not LFS) | GIF/WEBP ≤5 MB tracked, renders inline in a markdown preview | T5 |
| **T8** | Rewrite README (hero-video → badges → concise features → quickstart → architecture → docs) | `README.md`; badge sources `.github/workflows/{build,test,lint,ci}.yml`; move long diff-rollout block → `docs/examples/system_identification.md` | Re-anchor hero on the H1 grasp video; add unified-nk-solver + Scene→CookToModel + .nks/.nka + render-stack story; demote go2 gif to secondary; trim quickstart to <15-line hello-world; shields.io badges; target ~120–150 lines | every code snippet re-verified against the post-M9 python surface (`nuka.World.create_from_scene`/`Tape`/`autograd`); markdown renders; links resolve | T7 + OWNER copy/version |
| **T9** | Push to GitHub | git (LFS on PATH `/root/.nuka-toolchain-gcc14/bin`; proxy per `[[v03-git-push-procedure]]`) | Branch off master, commit demo exe + renderer change + media + README, push via proxy | `git push` succeeds through proxy; README hero renders on github.com | T8 |

### PHASE C — M10 RL REGRESSION (the broader milestone; shares Phase-A infra only via the cooked scenes)

| id | title | files touched | builds/wires | verification gate | dep |
|----|-------|---------------|--------------|-------------------|-----|
| **T10** | go2 PPO short-run reward-trajectory regression | `examples/training/train_go2_ppo.py`, `examples/training/go2_ppo_cfg.yaml`; `python/nuka/rl_games/vecenv.py` (`nuka_go2` survives) | Run a short PPO and confirm reward trajectory matches the prior envelope (plan ask a) | short run reaches prior reward envelope; deterministic on the generic `nuka.World` | scenes cooked (go2.nks present) |
| **T11** | SG-spec re-narration into nk terms (DOC-ONLY) | `docs/specs/2026-06-10-h1-whole-body-rl-grasp-spec.md` | Swap deleted-symbol anchors (`StepStandGrasp`/`BatchedUnifiedWorld`/18-DOF truncation) for `nuka.World` over Scene→CookToModel→nk::World; mark **G0 (DOF honesty) CLOSED** (51-DOF nk cook proven, `h1_grasp_lift` `ASSERT_EQ(dof_stride,51)`); re-point G1 to generic env-major batching; keep goal + honesty bars + G2-G5 verbatim | no golden/asset touch; goal unchanged; deleted symbols gone from the spec | none (doc) |
| **T12** | Greenfield H1-grasp RL env on the generic `nuka.World` | NEW `python/nuka/tasks/h1_grasp.py` + `h1_grasp_obs.py` + `h1_grasp_rewards.py`; `python/nuka/rl_games/vecenv.py` (+`nuka_h1_grasp` factory) | Mirror `NukaGymEnv` lifecycle on `create_from_scene`; obs from generic field exports; action via `DRIVE_TARGET` (DLPack zero-copy); reward = force-closure-gated terms re-derived from `LINK_CONTACT_WRENCH`/`CONTACT_FORCE`; reuse `h1_grasp_choreo.resolve(name_to_dof)` for column layout (the M8/M10 seam) | env constructs + reset/step_n/autoreset on `nuka.World`; obs/action shapes sane | T11; (needs name→dof C-ABI accessor — see T14) |
| **T13** | `train_h1_grasp_ppo.py` smoke-learn + catch-eval | NEW `examples/training/train_h1_grasp_ppo.py` + `h1_grasp_ppo_cfg.yaml` + `grasp_catch_eval.py` | Rebuild training/cfg/catch-rate observer on the new env; stage per SG-spec G3 (S1 hold→S2 lift→…) as config gates on ONE env (NOT per-stage worlds) | catch-eval rises over the smoke run (plan ask b) | T12 |
| **T14** | C-ABI name→dof accessor (enables T12 + the clean python demo path) | `src/c_abi/world.cpp` (currently DISCARDS `cooked.scene_map`); NEW `nuka_world_query_dof_by_name` over `SceneMap` + name→EntityId; `python/src/nuka_ext.cpp`; `python/nuka/__init__.py` | Surface the cooked link-name→dof map so `h1_grasp_choreo.resolve()` and the RL env have a real supplier (no hardcoded device indices) | python can resolve a known H1 link name → dof column; D1 unaffected (read-only accessor) | none (enables T12) |
| **T15** | Union throughput measurement (100M env-steps ≤24h bar) | NEW perf probe under `tests/perf/` (or a script) over the generic nk::World at N∈{32,256,1024} | Measure env-steps/sec on the env-major batched nk::World; check the 100M-steps≤24h bar (plan ask c) | measured throughput recorded; bar met or HONESTLY flagged RED with the number | none |

### PHASE D — M9→M10 CARRY-FORWARD CONTRACTS (re-assert as nk-world tests; see §4)

| id | title | files touched | builds/wires | verification gate | dep |
|----|-------|---------------|--------------|-------------------|-----|
| **T16** | go2 PD-stance behavioral gate on nk | NEW `tests/scenario/test_go2_pd_stance_nk.cpp` over cooked `go2.nks` via `nuka.World` | Re-assert crouch-hold height, Σλ=Mg/4 per foot, COM-inside-foot-polygon, no-tip, 4096-env clean+deterministic | gate PASSES on the unified nk world | go2.nks |
| **T17** | per-link contact-wrench readback on a seated go2 | NEW `tests/scenario/test_link_contact_wrench_nk.cpp` | Re-assert ΣF_z≈Mg per-link, `CONTACT_FORCE==lambda/dt`, wrench-arg-order, graph==wrench (field binary contract survives via `multi_env SingleEnvStillSupported`) | gate PASSES; cite `m9-t11-articulated-fold.md` L97-110 as contract source | go2.nks |
| **T18** | autoreset/base-pose/velocity-mode gates | NEW `tests/c_abi/test_batched_reset_nk.cpp` + `test_control_modes_nk.cpp`; rides `NukaGymEnv` (already does autoreset + un-lagged base_pose) | Re-assert autoreset/authority-isolation, un-lagged base pose after `ResetEnvs`, velocity(non-PD) mode + per-mode two-run byte-exact | gates PASS; cite `m9-t10-coverage-fold.md` L128-130/181-195 | none |

> **Phase independence:** A is fully independent of C/D (the headline ships first). C and D share only the cooked `go2.nks`/`h1_cup_table.nks` assets and the generic `nuka.World` substrate. T14 (name→dof) is the one shared seam between the demo's clean path and the RL env — built once, consumed by T12 and the future python demo.

---

## 4. M9 → M10 CARRY-FORWARD CONTRACTS

Source ledgers: `subagent-plans/m9-t11-articulated-fold.md` L97-110 and `subagent-plans/m9-t10-coverage-fold.md` L128-130 / L181-195. These were ACCEPTABLE-LOSS-to-M10 when M9 deleted the batched-only RL surface; the **generic foot-ground physics is already on nk** (`test_foot_ground_subsume`, `h1_grasp_lift`, `articulation_contact_rows`, `multi_env` 4096 D1) — only the go2-stance BEHAVIORAL envelope + the RL-surface contracts are owed.

| contract | what it asserts | original test (deleted) | re-assert in | substrate |
|----------|-----------------|-------------------------|--------------|-----------|
| **go2 PD-stance** | crouch-hold height tracking, Σλ=Mg/4 per foot, COM-inside-foot-polygon static balance, no-tip, 4096-env clean+deterministic | `tests/runtime/test_go2_pd_standing.cpp` | **T16** | cooked `go2.nks` → `nuka.World` |
| **per-link wrench** | ΣF_z≈Mg per-link on a seated stance, `CONTACT_FORCE==lambda/dt`, wrench-arg-order cross-check, graph==wrench | `tests/sensor/test_link_contact_wrench.cpp` (`Go2StanceBalancesWeight`/`D1`/`StepGraphMatchesStep`) | **T17** | seated go2 nk world (field contract survives via `multi_env SingleEnvStillSupported` resolving `NUKA_FIELD_LINK_CONTACT_WRENCH`) |
| **autoreset** | autoreset + authority-isolation, per-env independence | `tests/c_abi/test_batched_reset.cpp` | **T18** | `NukaGymEnv` (mechanism partly survives in python `env.py`; byte-exact GATE was deleted) |
| **base-pose** | un-lagged base pose after `ResetEnvs` (one-step-FK-lag boundary) | `tests/c_abi/test_base_pose_view.cpp` | **T18** | `NukaGymEnv` reads `NUKA_FIELD_BASE_POSE` for the gravity term |
| **control-modes** | velocity (non-PD) control mode + per-mode two-run byte-exact | `tests/c_abi/test_control_modes.cpp` | **T18** | `CONTROL_MODE_VELOCITY` two-run byte-exact on `nuka.World` (NOTE: C-ABI `world.cpp:144-146` currently accepts ONLY `PDPosition` — verify the velocity mode is wired before asserting, else flag) |

---

## 5. RISK / DEBT

- **[ASSET/RENDER — HIGH] H1 is INVISIBLE today.** `render_world.cpp:394` only draws a `VisualMeshComponent` whose mesh ref is a `.nka` MESH fourcc; an MJCF imported-at-Load visual geom has triangles in `mesh_vertices` but EMPTY `visual_mesh_ref`. The committed `h1_cup_table.nka` (21 KB) has ZERO MESH chunks (only 1 PMAS). So pointing the renderer at the frozen scene renders only the table box + 2 spheres (cup hull skipped, H1 invisible). **Mitigation = T2** (render import-time triangles live from local `.nuka-assets/`) — avoids the 88 MB commit AND the frozen-asset edit.
- **[ASSET/LFS — HIGH, OWNER] the ~88 MB h1 visual .nka.** `m8-render-gates.md:133-138` documents the deferral. `.nka` is NOT under LFS (`.gitattributes` LFS-tracks only `tests/oracle/golden/*.bin`); go2.nka (8.5 MB) is committed RAW. If fresh-clone reproducibility is required, the 88 MB asset must ship (RAW < GitHub 100 MB limit, or LFS-over-proxy which is UNVERIFIED here). T2 sidesteps this for THIS box. **OWNER decision D-LFS.**
- **[RENDER — NOT LOCALLY VERIFIABLE] honest scope boundary.** This box has CUDA GPU + only llvmpipe/lavapipe CPU Vulkan. NOT verifiable / NOT needed for the offscreen mp4: real-GPU PBR color fidelity (lavapipe approximates), swapchain present to a monitor, ImGui theme, CUDA↔Vulkan zero-copy interop (structurally impossible CUDA↔llvmpipe). State in the demo writeup that the mp4 is the offscreen-raster D1 oracle render on lavapipe — the gate-proven path.
- **[D1 / FROZEN ANCHORS — HARD]** `examples/scenes/h1_cup_table.{nks,nka}` are FROZEN — the demo reads them, never edits. Goldens NEVER regenerated; `tests/oracle/golden/**` owner-protected. T2/T6 are render-only and MUST keep `render_physics_parity`/`render_raster_smoke` byte-identical (G2 memcmp=0 is the red-line). The grasp drive in T1/T4 is byte-for-byte the proven `h1_grasp_lift` path — do not retune PD constants.
- **[STALE — cleanup] dead demo refs.** `examples/demo/h1_cup_demo_gate.py` references deleted binaries (`nuka_h1_bridge_spike_test`, `nuka_h1_dense_grasp_test`); `render_rollout.py` claims H1 grasp "END-TO-END" but cooks generically (recorder.cpp:248) — both should be corrected/flagged during Phase A/B.
- **[RL — HIGH] H1-grasp RL is greenfield.** M9 T8 (`4fe7970`) deleted the env/obs/reward/train/cfg/catch-eval/binding-tests (4153 net deletions) on the now-deleted `GraspWorld`/`UnionWorld` C-ABI. `train_h1_grasp_ppo.py` does NOT exist. T12/T13 rebuild from scratch on the generic world — do NOT port the fixed-base 2-finger synthetic gripper (wrong substrate under the highest directive).
- **[SEAM — MED] name→dof has no supplier.** `h1_grasp_choreo.resolve(name_to_dof)` is inert: `world.cpp` DISCARDS `cooked.scene_map`; no C-ABI/python accessor exists. T14 is the prerequisite for both the clean python demo path AND the RL env.
- **[LFS TOOLING] git-lfs 3.7.0 must be on PATH (`/root/.nuka-toolchain-gcc14/bin`) for ALL git ops; push via proxy per `[[v03-git-push-procedure]]`.**

---

## 6. §OWNER DECISIONS

**OD-1 — Demo driver: scripted choreography vs trained RL policy (drives D1-demo-vs-rl).**
The headline mp4 — scripted `h1_grasp_choreo`-style choreography (decouple from RL, ship now) OR gate on a trained M10 policy?
Options: (a) scripted choreography only; (b) trained RL policy; (c) both (ship scripted now, optionally re-render with policy later).
**Recommendation: (a) scripted choreography only.** The unified-world directive DEFINES behavior as scene-data + per-scene control SCRIPT; the grasp is already physics-proven on nk (`h1_grasp_lift`). Gating the headline on a multi-session RL run is unnecessary coupling. (Recon D1-demo-vs-rl + D-STEP-1 agree.)

**OD-2 — The ~88 MB h1 visual .nka: how to ship (or avoid shipping) the robot geometry (D-LFS-88MB).**
Options: (1) RAW commit (~88 MB < 100 MB, go2.nka precedent, bloats .git to ~106 MB); (2) LFS-track `examples/scenes/*.nka` scoped (lean clones, but LFS-over-proxy UNVERIFIED here, filter must not disturb go2.nka raw blob); (3) do NOT commit — render locally from `.nuka-assets/` (T2) and commit ONLY the small mp4/gif (fresh clones re-cook on demand); (4) decimate/LOD STL below 50 MB then RAW-commit.
**Recommendation: (3) for THIS deliverable** (T2 renderer change makes the frozen scene render the H1 from live local meshes; repo stays lean; frozen assets untouched; box renders offscreen). Escalate to (1) ONLY if owner requires fresh-clone reproducibility of the visual scene. Reserve (2) until a verified LFS-over-proxy dry-run.

**OD-3 — Homepage video hosting mechanism (VIDEO-HOST).**
Options: (a) commit looping GIF to `docs/media/` (go2-gif precedent, guaranteed inline, ~2–5 MB); (b) animated WEBP (smaller/sharper, slightly less universal); (c) upload mp4 to a GitHub Release/PR → user-images URL (native HTML5 player, best quality, manual upload); (d) do all.
**Recommendation: (d) do all** — compact GIF/WEBP inline hero (mirroring `docs/media/go2_walk_4096env.gif`) + Release-hosted full mp4 link. GitHub README markdown does NOT inline-embed a relative `.mp4`.

**OD-4 — Static held-cup beauty clip vs full grasp PERFORMANCE clip (D2 render-reality).**
Options: (a) static held-cup (beauty toggles only, no choreography wiring — fastest); (b) full approach→close→lift performance (the compelling story, needs T1/T4); (c) two-stage (static now, performance follow-on).
**Recommendation: (b) full performance.** With the C++ exe path (OD-1=a), the choreography IS the proven `h1_grasp_lift` drive — the performance clip is nearly the same cost as static and is the actual selling point. Fall back to (c) only under hard deadline pressure.

**OD-5 — RL scope vs demo scope for M10 (D2-grasp-env-greenfield).**
Options: (1) full M10 RL (T10–T15 greenfield H1-grasp env + go2 regression + throughput + spec); (2) reduced M10 (go2 reward-trajectory regression T10 + spec re-narration T11 + carry-forward T16-T18 only; defer the H1-grasp RL env); (3) port the deleted env (REJECTED — wrong substrate).
**Recommendation: owner adjudicates 1 vs 2.** The DEMO VIDEO needs neither. If M10 is the last milestone and the headline is the priority, (2) is a defensible reduced scope (the plan's literal asks a/c/d + carry-forward) with the H1-grasp RL env as a fast-follow. Recommend (1) only if owner wants a trained grasp this milestone.

**OD-6 — SG-spec G0 (18-DOF honesty) — mark closed or re-verify (D3-spec-renarration).**
Options: (a) mark closed (51-DOF nk cook is the evidence — `h1_grasp_lift ASSERT_EQ(dof_stride,51)`); (b) re-verify as a fresh nk DOF-honesty assertion first; (c) leave for owner.
**Recommendation: (b) then (a)** — cheaply re-verify (the `dof_above18_honesty` test survives standalone; `h1_grasp_lift` already steps 51-DOF), then mark closed. Honors the spec's honesty discipline.

**OD-7 — Logo + social preview (LOGO-SOURCE / SOCIAL-PREVIEW).**
Logo lives at the separate `kry4r/Nuka` repo (`assets/logo.png` per memory), NOT this repo. Copy into `docs/media/logo.png` (self-contained) vs hot-link vs ship without. Social-preview image is an owner-only GitHub Settings action.
**Recommendation:** copy logo into `docs/media/logo.png`; do NOT block the README rewrite on it (the hero video carries the page); flag social-preview as a manual owner step.

**OD-8 — README version string + marketing copy (README-COPY-AND-STATUS).**
The current README says "v0.5 — initial public release" (stale; predates the entire M0-M11 nk refactor). Version number, hero tagline, and status wording are owner calls.
**Recommendation:** recon supplies the architecture-accurate feature list + outline; OWNER supplies the version number, tagline, and status block before publish (T8 is gated on this).

---

## 7. §CONTROLLER RULINGS

These I resolve directly under strict-dominance + the highest directive (ONE general solver, no special-casing). Each preserves intent and dominates the alternative.

**CR-1 — Drive the headline demo from a C++ exe, NOT the python Recorder. (resolves D1-driver, D-STEP-1, D-STEP-2)**
Rationale: VERIFIED — `recorder.cpp:248` cooks `cook::CookToModel` (generic single-artic, loose cup, position-PD) and `Simulation::Frame()` injects zero drive ⇒ the python path renders a passive non-grasp of the wrong topology. The working 51-DOF friction grasp exists ONLY on the C++ union cook. The C++ exe reuses 100%-proven code (`h1_grasp_lift` drive + `render_physics_parity` render, both green on this box) and adds only the RenderWorld↔union binding. The python path needs M10-scale new C-ABI (union cook + torque mode + name→dof + `TableEnabled` + rigid-body velocity). Strictly dominates for the deadline; does NOT contradict the highest directive (the union cook is the GENERAL solver fed union scene data + a control script, not a special world type or solver branch — the deleted special worlds are gone and not reintroduced).

**CR-2 — RenderWorld↔union binding = option A (hand-build) for the video; option B (union cook emits SceneMap) is the named clean M10 path. (resolves D2-renderworld-union, D-RENDER-WIRING)**
Rationale: A is fast, low-risk, frozen-safe, and the cup grasp is the story; B is the principled unified-world end-state (cook emits `SceneMap` → reuse `BuildRenderWorld` unchanged) but is deeper cook work off the critical path. Ship A; name B as M10 cleanup.

**CR-3 — T2 (render import-time visual meshes) over committing 88 MB, for THIS box. (refines D-RENDER-WIRING (a))**
Rationale: T2 is the minimal, frozen-safe, repo-lean fix to the exact `render_world.cpp:394` gap that makes the H1 invisible; it generalizes (any imported robot renders without a cook step). Strictly dominates the asset commit for the local deliverable. The 88 MB LFS/RAW question is deferred to OWNER OD-2 (only triggered if fresh-clone reproducibility is mandated) — I do NOT pre-commit a large asset (irreversible repo bloat = de-risk, not silently choose).

**CR-4 — Cadence = render every step (8× slow-mo, ~420 frames, end on hold, BITE excluded). (resolves D3-cadence)**
Rationale: dt=1/240 vs 30 fps ⇒ every-step = 8× slow-mo that showcases grip/contact detail (the engine's selling point); ending on the sustained friction hold reads as success; including BITE (a drop) in the showcase would read as failure — reserve BITE for a separate honesty clip. Owner-overridable (OD-4) but this is the strict-dominant showcase choice.

**CR-5 — SG-spec re-narration is doc-only, no golden/asset touch; mark G0 closed after a cheap re-verify. (resolves D3-spec-renarration mechanism)**
Rationale: plan L505 explicitly scopes this as "改述 (re-narrate) to nk terms (goal unchanged)". G0's 18-DOF truncation landmine is already moot on the nk path (`h1_grasp_lift` steps full 51-DOF). Low-risk, plan-mandated. The CLOSED-vs-reverify nuance is surfaced to owner (OD-6) per honesty discipline.

**CR-6 — RL env is greenfield on `nuka.World`, NEVER a port of the deleted GraspWorld env. (resolves D2-grasp-env-greenfield, rejecting the port option)**
Rationale: the deleted env was a FIXED-BASE 2-finger SYNTHETIC gripper on the special `nuka.GraspWorld` — the wrong substrate under `[[unified-world-no-special-grasp-binding]]` (the HIGHEST directive). Reusing it would reintroduce a special-world dependency. Greenfield on the generic world (mirror `NukaGymEnv`, reuse the `h1_grasp_choreo` name→dof seam, re-derive obs/reward from generic field exports) is the only directive-compliant rebuild. The full-vs-reduced SCOPE of M10 RL remains an owner call (OD-5).

**CR-7 — Carry-forward contracts re-asserted as nk-world TESTS (T16-T18), not silently dropped.** 
Rationale: twice-documented as explicitly owed (`m9-t11`/`m9-t10` ledgers). They ride existing cooked assets (`go2.nks`) + the surviving `NukaGymEnv` mechanism. Re-asserting them on the unified nk world (not a special batched world) is directive-compliant and closes the M9 acceptable-loss debt.
