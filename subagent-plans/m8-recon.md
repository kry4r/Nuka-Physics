# M8 Recon Brief — Frame Loop + Offscreen 3D Raster Render + Recorder

> Synthesis of 5 parallel reader findings (plan-spec / existing-render / runtime-seams / cabi-recorder-parity / arch-directives),
> verified against the live tree on 2026-06-13. Plan = `docs/plans/2026-06-11-nk-core-platform-refactor.md` (referred to below as "plan Lxxx").
> This brief is self-contained: a controller can decompose M8 into subagents from this alone.

---

## 1. M8 Scope & the `render_physics_parity` Exit Gate

### 1.1 Scope (plan L488–493, §3.8 L334–336, §3.9 L338–356)
M8 = **"帧循环 + 离屏 3D 渲染 + Recorder"**. It builds, on top of the already-existing `nk::World`:
1. A **host-side frame loop** (`Simulation` + 4 systems `Input/Sim/TransformSync/Render` + `CommandQueue` + `PosePublisher`) under `src/runtime/app/` (net-new dir — confirmed absent).
2. A **1:1 Registry-driven `render::RenderWorld`** (`BuildRenderWorld(Registry, SceneMap)` once; `TransformSyncSystem` per frame).
3. A **real-triangle offscreen Vulkan PBR forward rasterizer** (`src/render/raster/vulkan_raster_renderer.*` + `shaders/{mesh.vert,mesh_pbr.frag}`) — net-new dir (confirmed absent). Reuses the existing **headless Vulkan init** only.
4. A **Recorder** C-ABI + python (`c_abi/recorder.cpp` + `nuka_recorder.h` + `python/nuka/recorder.py`) that captures frames and muxes an mp4.

M8 is **purely additive**: the ONLY deletion is the matchstick demo `examples/demo/go2_demo_render.py` (plan L492). All legacy stepping/render-seam deletion is **M9** (plan L499), NOT M8. This isolates M8 risk.

**M8 is RASTER ONLY — NOT the path-tracer/beauty tier.** The self-written CUDA RT path-tracer (`src/rt/`, 15 files, exists, untouched in M8) is wired to `RenderWorld` only in **M11** via `RenderWorldToTwoLevelScene` / `rt_adapter` / `RtBackendI` (plan L336, L371, L507–509).

### 1.2 The Exit Gate — `tests/scenario/render_physics_parity.cpp` (net-new)
**Pass criteria (plan L490 + L493), all three required:**

| # | Criterion | Concrete assertion | Source |
|---|-----------|--------------------|--------|
| G1 | **1:1 physics↔render** | Per frame, for EVERY entity that has BOTH a physics and a render component: `render_instance.world_xform == data.body_pose ∘ visual_local` (Transform compose equality, NOT a pixel check) | plan L336 verbatim "每帧对每个有双侧组件的实体断言 render_instance.world == data.body_pose ∘ visual_local" |
| G2 | **Two renders byte-identical (D1)** | `memcmp` of the two `VulkanOffscreenReport::pixels` vectors == 0 (render the same frame twice) | plan L490 "两次渲染 byte 相同" |
| G3 | **Non-background pixels > 0** | `EXPECT_GT(report.non_background_pixel_count, 0)` — the existing convention | plan L490 "非背景像素>0"; `VulkanOffscreenReport::non_background_pixel_count` exists |
| G4 | **Python one-command mp4** | One python command emits BOTH go2 AND h1 rollout mp4 **带视觉网格与材质** (with visual meshes AND PBR materials) | plan L493 |
| G5 | **Throughput-neutral** | Step throughput IDENTICAL when rendering is OFF (no perf regression vs M3/M4/M5 red-lines) | plan L493 "渲染关闭时步进吞吐零差异" |

**How it's run:** `render_physics_parity.cpp` is a gtest in `tests/scenario/`, asset-gated with `if(!AssetsAvailable()) GTEST_SKIP()` (mirror `tests/scenario/h1_union_parity.cpp:233`, asset check at :92–96). It MUST sit behind `if(NK_BUILD_VULKAN_VALIDATION)` (the guard for the only existing render tests, tests/CMakeLists.txt:3065) and link BOTH the nk physics libs AND `nuka_render`. In M9 (L500) it re-homes under the `nuka_render_test` executable; during M8 it is its own `add_executable` + `gtest_discover_tests` block. G4/G5 are driven by `examples/demo/render_rollout.py` (the one-command driver) + a render-off throughput comparison.

**Perf red-lines G5 must not break** (plan L515): union(H1+cup+table) N=1 ≤5ms/step; grasp batch ≥11k env-steps/s; Go2 4096 ≤~1µs/env-step; whole-step plan(graph) replayable + D1; hot path zero-alloc.

---

## 2. File Manifest (dependency / build order)

Legend: **NEW** = create from scratch · **MODIFY** = edit existing · **SEED** = existing, reused as a base/pattern · **SURVIVE** = existing dependency M8 reads read-only · **REPLACE** = existing, superseded by an M8 file but only DELETED in M9.

### Tier A — Upstream deps M8 consumes (all SURVIVE; exist by M8)
| Path | Status | Key symbols M8 needs |
|------|--------|----------------------|
| `src/nk/pipeline/world.hpp` / `.cpp` | SURVIVE | `World::Step/StepPlanned/Reset/DispatchOp/FieldPtr/EnvCount/GetData` — the thing `Simulation` steps; `Step()` is a single linear `BackendDispatch` over `Pipeline::Calls()` (world.cpp:292–307), NO frame loop |
| `src/nk/data/data.hpp` | SURVIVE | `Data::DownloadField(FieldId,dst,bytes,off)` (data.hpp:58) — the ONLY host-readback path; the pose_publisher SOURCE. Also `UploadField` (:54) |
| `src/nk/model/generated/field_ids.hpp` | SURVIVE | `FieldId::LinkPose` (:15, per:link persistent owner:data), `BasePose` (:16, per:env), `BodyPose` (:17, per:body) — what HostDownloadPublisher downloads |
| `src/scene/scene_map.hpp` | SURVIVE | `SceneMap::RefOf(entity)→CookedRef`, `EntityOfBody/EntityOfLink`. `CookedRef{body_row, joint_row, dof_index, shape_row, link_index, bp_group}` (scene_map.hpp:20–27) — **NO local-pose field** (see Risk R1) |
| `src/scene/ecs/registry.hpp` | SURVIVE | `Registry::ForEach<T>/Get/GetRenderMaterial` — closed-set component ECS; `BuildRenderWorld` reads it |
| `src/scene/ecs/components.hpp` | SURVIVE | `VisualMeshComponent{AssetRef mesh; uint32_t render_material_id}` (:73 — **no visual_local field**); `RenderMaterial{base_color[4],metallic,roughness,emissive[3],opacity, tex_albedo/normal/metallic_roughness}` (:58–69 — **full PBR already modeled**); `TransformComponent{math::Transform local}` (:30–31, "relative to PARENT NODE"); `CameraComponent.local_transform` (:151); `LightComponent.local_transform` (:160) |
| `src/scene/asset/nka.*` | SURVIVE | `.nka MESH` chunk (MeshLibrary source) + `TEXB` chunk (PBR texture source). Open Q: does it expose MESH triangle geometry to the host renderer today? |
| `src/include/nuka/nuka.h` | SURVIVE | `nuka_result_t`, `nuka_world_get_buffer_view`, `nuka_state_field_t` |
| `src/c_abi/internal.hpp` / `handle_table.hpp` / `error.cpp` | SURVIVE | `WorldRecord`, `HandleTable<Tag,T>`, `WorldTable()`, `MapExceptionToResult/RunNoThrow` — the per-module C-ABI pattern |

### Tier B — Render SEED / MODIFY / REPLACE (existing render stack)
| Path | Status | Notes |
|------|--------|-------|
| `src/render/vulkan_renderer.hpp` | SEED + MODIFY | `VulkanOffscreenOptions{width,height,view_scale,view_center,background}` + `VulkanOffscreenReport{pixels: vector<VulkanRgba8>, non_background_pixel_count}` (the G2/G3 surface). `RenderSceneVulkan` / `RenderDebugDrawListVulkan`. Headless init reused by the raster renderer |
| `src/render/vulkan_renderer.cpp` | MODIFY (L491) | Demote debug wireframe → overlay. Today: COMPUTE-only headless path — `CreateInstance` (apiVersion 1.0, NO surface/swapchain), `SelectComputeDevice`, single `vkCmdDispatch` of `debug_draw.comp`, `DownloadPixels` (vkMapMemory+memcpy, :797–809). **NO graphics pipeline.** Reuse = instance/device/queue/command-pool init ONLY (~120 LOC) |
| `src/render/shaders/debug_draw.comp` | SURVIVE | The ONLY existing shader (GLSL 450 COMPUTE, orthographic 2D outlines). Not the vert/frag M8 needs |
| `src/render/render_scene.hpp` / `.cpp` | REPLACE (DELETE in M9 L499) | `RenderScene{mesh_instances,materials,cameras,lights}` + `BuildRenderScene(SceneIR)` + `BodyWorldTransform`. `RenderMaterial` already PBR-shaped. Superseded by `render_world.*` |
| `src/scene/scene_pipeline.cpp` / `.hpp` | REPLACE (DELETE in M9 L499) | `ApplyRuntimeStateToCompiledScene` already writes `pose[body] * local` per step (scene_pipeline.cpp:50–98) — the PRIOR ART for TransformSync's 1:1 contract. Only consumer = its own test → low-risk to replace |
| `src/rt/` (15 files) | SURVIVE (untouched in M8) | Self-written CUDA two-level TLAS/BLAS path-tracer + `framebuffer.hpp` (6-AOV, D1). Wired to RenderWorld in **M11** only. Do NOT drive from M8 (Risk R7) |

### Tier C — M8 NET-NEW (create in build/dependency order)
Build order: render_world → raster renderer (+shaders) → runtime/app (command_queue → pose_publisher → systems → simulation) → recorder (C-ABI → python binding) → gate test + demo driver.

| # | Path | Status | Key symbols it MUST expose |
|---|------|--------|----------------------------|
| 1 | `src/render/render_world.hpp` / `.cpp` | NEW | `RenderWorld`; `RenderInstance{entity, mesh_id, world_xform, render_material_id}`; `MeshLibrary` (.nka MESH dedup); `BuildRenderWorld(Registry, SceneMap)`; material/camera/light tables |
| 2 | `src/render/raster/vulkan_raster_renderer.hpp` / `.cpp` | NEW | `VulkanRasterRenderer` with a `Render(...)→VulkanOffscreenReport`-shaped output. Real VkRenderPass + VkFramebuffer + VkGraphicsPipeline + depth buffer + vertex/index buffers + per-instance transform UBO/SSBO + PBR descriptor sets. ALL `RenderMaterial` fields effective |
| 3 | `src/render/raster/shaders/mesh.vert` | NEW | vertex shader (forward pass). Built via the same glslc custom_command pattern (src/CMakeLists.txt:131–150) |
| 4 | `src/render/raster/shaders/mesh_pbr.frag` | NEW | PBR fragment shader; base_color/metallic/roughness/emissive/opacity + 3 textures bind here |
| 5 | `src/runtime/app/command_queue.hpp` | NEW | `Command` (enum, anticipate M11 `MoveEntity` per L509); MPSC queue. Header-only |
| 6 | `src/runtime/app/pose_publisher.hpp` | NEW | `PosePublisher` (interface); `HostDownloadPublisher` decl. (`CudaVulkanInteropPublisher` is M11 — NOT M8) |
| 7 | `src/runtime/app/pose_publisher.cpp` | NEW | `HostDownloadPublisher` — `Data::DownloadField(LinkPose/BasePose/BodyPose)` for the selected env → SceneMap → RenderInstance |
| 8 | `src/runtime/app/systems.hpp` | NEW | `InputSystem`, `SimSystem`, `TransformSyncSystem`, `RenderSystem` |
| 9 | `src/runtime/app/systems.cpp` | NEW | `TransformSyncSystem::Run` — writes `data_pose ∘ visual_local` → `instance.world_xform` each frame |
| 10 | `src/runtime/app/simulation.hpp` | NEW | `Simulation`; Frame loop |
| 11 | `src/runtime/app/simulation.cpp` | NEW | `Simulation::Frame` — drains CommandQueue, steps `nk::World`, runs TransformSync + Render (MUST be bypassable when render off, G5) |
| 12 | `src/c_abi/recorder.cpp` | NEW | `RecorderRecord{WorldRecord*/nk::World*; camera; frame buffers}`, file-local `RecorderTable()`, `nuka_recorder_capture` (PPM P6 dump), `nuka_recorder_to_video` (ffmpeg shell). Mirror `diffsim.cpp` TapeRecord/TapeTable pattern |
| 13 | `src/include/nuka/nuka_recorder.h` | NEW | `nuka_recorder_handle`, `nuka_recorder_create/_capture/_to_video/_destroy`. MUST be added to `NUKA_PUBLIC_HEADERS` (src/CMakeLists.txt:117) |
| 14 | `python/nuka/recorder.py` | NEW | `Recorder(world, camera=...)`; `.capture('out/frames')`; `.to_video('out/run.mp4')` (plan L353 contract) |
| 15 | `tests/scenario/render_physics_parity.cpp` | NEW | `RenderPhysicsParity` — the G1–G3 gate (see §1.2) |
| 16 | `examples/demo/render_rollout.py` | NEW | one-command go2+h1 rollout→mp4 driver (G4) |

### Tier D — MODIFY existing (wiring)
| Path | Status | Change |
|------|--------|--------|
| `python/src/nuka_ext.cpp` | MODIFY (L491) | nanobind: bind Recorder into the python extension |
| `src/CMakeLists.txt` | MODIFY | Add `c_abi/recorder.cpp` to the `nuka` SHARED source list (currently :898–924, recorder absent); **add `nuka_render` to that target's link libs** (NOT currently linked — verified at :937 link block); add `nuka_recorder.h` to `NUKA_PUBLIC_HEADERS` (:117); add `mesh.vert`/`mesh_pbr.frag` glslc custom_commands |
| `tests/CMakeLists.txt` | MODIFY | Add `render_physics_parity` `add_executable` + `add_dependencies(... nuka_codegen)` + `NUKA_SOURCE_DIR` define + include `src` + link nk libs **AND** `nuka_render`, inside the `if(NK_BUILD_VULKAN_VALIDATION)` guard + `gtest_discover_tests(... WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})` |

### Tier E — DELETE
| Path | Status |
|------|--------|
| `examples/demo/go2_demo_render.py` | DELETE (L492) — matchstick numpy→PPM rasterizer, superseded by Recorder + render_rollout.py |

---

## 3. EXISTS vs NEW — the seams M8 builds on

### 3.1 Render seam — EXISTS but COMPUTE-ONLY 2D; the graphics pipeline is 100% NEW
- TWO unrelated render stacks today, sharing NO code: (A) `src/render/` = Vulkan **COMPUTE** wireframe/debug-draw offscreen (vulkan_renderer.cpp); (B) `src/rt/` = self-written CUDA path-tracer.
- (A) is **headless + compute-only**: `CreateInstance` apiVersion 1.0, NO surface/swapchain extensions; `SelectComputeDevice` picks first compute-queue + STORAGE_IMAGE device; render = one `vkCmdDispatch` of `debug_draw.comp` → `vkCmdCopyImageToBuffer` → host RGBA8 readback. **This IS the "too crude" renderer.**
- **NO graphics pipeline exists anywhere**: a tree-wide grep for `VkGraphicsPipeline/CreateGraphicsPipelines/VkSwapchain/vkCreateRenderPass/VkRenderPass/VkFramebuffer/.vert/VK_SHADER_STAGE_VERTEX` returns ZERO hits. The M8 raster renderer + mesh.vert/mesh_pbr.frag are from scratch; "reuse headless Vulkan init" (L490) is ONLY instance/device/queue/command-pool (~120 LOC).
- **NO PPM/PNG/MP4 encoder anywhere** in src/ (grep `P6/stbi_write/lodepng/png_write/WritePpm/VideoEncoder` empty). The ONLY pixel egress = `DownloadPixels` → `VulkanOffscreenReport::pixels`. Documented capture path (docs/architecture/headless-rendering.md): `pixels → binary PPM P6 ("P6 W H 255\n"+RGB) → ffmpeg -framerate 30 -pix_fmt yuv420p`. M8's Recorder implements PPM-write + ffmpeg-shell in C++ for the FIRST time.
- **NO CUDA↔Vulkan interop** (grep `external_memory/cudaExternalMemory/VK_KHR_external/vkGetMemoryFd` ZERO hits). Deferred to M11.
- Existing physics→render pose update already does the 1:1 math: `scene_pipeline.cpp:50–98` `ApplyRuntimeStateToCompiledScene` writes `pose[body] * local`. M8 formalizes this as `TransformSyncSystem` over Registry+SceneMap instead of SceneIR.

### 3.2 Runtime seam — NO frame loop exists; all 5 M8 components net-new
- `nk::World` has NO frame loop: `Step()` (world.cpp:292–307) is a single linear pass; cadence is caller-supplied. M8's `simulation.hpp` is the FIRST frame-loop abstraction over `nk::World`.
- Tree-wide grep for `command_queue|CommandQueue|pose_publisher|PosePublisher|render_world|RenderWorld|class Simulation|TransformSyncSystem|RenderSystem|class Recorder|HostDownloadPublisher` in `src/ python/` returns ONLY an unrelated `python/nuka/isaaclab_compat/sim.py` shim. `src/runtime/app/` and `src/render/raster/` confirmed absent.
- The pose_publisher SOURCE is HOST-DOWNLOAD, not an op: all readout ops in `phi/backend_cuda/ops/readout.cu` are device-side (D2D / kernels writing device fields) — there is NO host-readout op. Host path = `Data::DownloadField` (data.hpp:58).
- The ONLY pre-existing multi-step loop = `src/runtime/coresident/unified_coresident_stepper.cpp` (LEGACY, validated-not-wired, host-orchestrated, `DownloadWorldPoses` at :86). It is **NOT nk::World-based** and is DELETED wholesale in M9 (L499). Reference only for stage ordering, NOT a build target.

### 3.3 C-ABI seam — single SHARED lib, per-module record+table pattern
- Entire C-ABI = ONE `nuka` SHARED lib (src/CMakeLists.txt:898–924). `c_abi/` today: buffer/device/diffsim/error/grasp_world/handle_table/internal/noise/union_world/world — **no recorder**. `src/include/nuka/` today: expected/nuka.h/nuka.hpp/nuka_diffsim/nuka_grasp/nuka_noise/nuka_union — **no nuka_recorder.h**.
- Per-module pattern to mirror (`diffsim.cpp`): a `XRecord{WorldRecord* world; ...}` struct + file-local `static HandleTable<nuka_X_t,XRecord>& XTable()` + extern-C entrypoints that null-guard, look up `WorldTable().Get(world)`, wrap in try/catch → `MapExceptionToResult`.
- **`nuka` does NOT currently link `nuka_render`** (verified at the :937 link block). recorder.cpp pulling the raster renderer forces this link to be added — and creates a Vulkan-mandatory risk for libnuka (Risk R6 / Decision D5).

### 3.4 The CURRENT python frame driver is the LEGACY world, not nk::World
- `python world.step() → nuka_world_step → StepWorldGpu / StepBatchedWorld` (c_abi/world.cpp:667–685) over `BatchedArticulatedWorld` + `DownloadArticulationState/SyncHostToInstance`. The c_abi→nk::World cutover is **M9** (L498). M8 lands BEFORE the cutover → **what state does the M8 mp4 actually capture?** (Open Q OQ2 / Decision D2.)

---

## 4. Architectural Constraints (binding)

1. **Physics-PRIMARY, rendering-SECONDARY** (owner 拍板, plan §1 L35; renderer-architecture-directive.md). Rendering's only job is to showcase the physics sim. Borrow renderer *architecture patterns* (Mangifera skeleton = owner's own proto-design + AnKi render-graph + Filament PBR north-star + MuJoCo extract-decouple), NOT game-engine subsystems (no audio/gameplay/networking/editor/D3D12; Nuka = Vulkan+CUDA only). M8 needs only a forward PBR pass — do NOT over-scope a full auto-barrier render graph (Risk R8).

2. **Self-written, NO OptiX** (rt-self-written-vs-optix-decision.md; plan §3.10 L371 "自写不 OptiX 不变"). RT-cores accelerate traversal=SPEED not image quality; software RT is identical-quality + more deterministic (D1). OptiX is rejected (breaks D1 + differentiability + zero-closed-SDK pillars). M8's renderer is Vulkan graphics-API raster — fine under this ruling; any RT CUDA is M11 via `RtBackendI`.

3. **M8 = offscreen RASTER, NOT path-tracer.** Two-backend-one-extract: Vulkan RASTER for the realtime/offscreen viewport (D1 of pixels enforced ONLY by the G2 gate on the CI path) vs software CUDA PATH-TRACER for OFFLINE beauty (D1 required, M11). They share the `RenderWorld`/`RenderMaterial` inputs but are NOT required to produce identical shading output (Open Q OQ-shading).

4. **CUDA↔Vulkan interop deferred to M11.** M8 PosePublisher = `HostDownloadPublisher` (host-download the SELECTED env). `CudaVulkanInteropPublisher` (device-scatter; the "highest payoff" seam) is M11 (plan L336, L372, L509). `pose_publisher.hpp` MUST be a clean interface so the M11 impl drops in without M8 rework (Risk R5).

5. **Zero-CUDA-token lint red-line** covers `src/render/**` and `src/runtime/app/**` (plan §3.1 L155). NO `<<<>>>`, no `cudaX(`, no `<cuda_runtime>`, no `phi/backend_cuda` include. So render_world/raster/simulation/systems/pose_publisher are pure C++/Vulkan; device pose access goes ONLY through `Data::DownloadField` (host) or `nk::World::FieldPtr`. This is precisely WHY M8 pose-publish is host-download, not a CUDA scatter.

6. **SceneGraph ↔ RenderWorld coupling (the §3.8 1:1 model, plan L334–336):** the SAME ECS `Registry` hosts both physics and render system-VIEWS (plan §3.7 L302). `BuildRenderWorld(Registry, SceneMap)` builds ONCE (MeshLibrary = dedup .nka MESH load). Each frame `TransformSyncSystem` writes `Data` body/link poses (via SceneMap) into `instance.world_xform`. Identity flows through SceneMap bidirectionally. **SG spec is FROZEN through M10** (plan §6 L523) — M8 consumes §3.6 components read-only and MUST NOT alter component/SceneGraph contracts. `CameraComponent`/`LightComponent` reuse existing record field layouts unchanged.

7. **D1 (byte-exact) is project-wide** (plan §5 L517 "②D1 byte-exact——golden 红 = 退化, 永不再生成 golden"). On the render side, G2 enforces it (two renders byte-identical). This implicitly forbids ANY nondeterministic render input (time-seeded jitter, async readback races, driver-nondeterministic raster ordering).

8. **Owner tail-reorder:** execution = **M8 → M9 → M11 → comprehensive ultracode review → M10 (last)**. The plan file still lists M9/M10/M11 numerically (L495–509). This does NOT change M8 scope — it means the M11 RenderWorld consumers (RT adapter + interop publisher + viewer) arrive AFTER the review pass, so M8's RenderWorld API has no second consumer to validate it until late → **design RenderWorld for the M11 RT-consumer contract up front** (Risk R-design).

---

## 5. Decomposition Proposal (ultracode-first)

Execution policy = ultracode-first: parallelize wherever sound; reserve single-flight for the genuinely coupled spine. Below, each task is marked **[PAR]** (parallelizable, ultracode) or **[SEQ]** (coupled-sequential, single-flight). All tasks must obey the zero-CUDA-token lint on render/runtime files (Constraint 5) and the host-download-only pose path.

### Phase 0 — Decisions gate (controller, BEFORE any task)
Resolve D1–D6 (§6). T2/T7 are BLOCKED until D1 (visual_local) + D2 (pose source) are answered.

### Phase 1 — Parallel foundations (no cross-deps)
- **T1 [PAR] — `render_world.{hpp,cpp}`** (manifest #1). `RenderWorld`/`RenderInstance`/`MeshLibrary`/`BuildRenderWorld(Registry, SceneMap)`. Reads Registry (VisualMesh/RenderMaterial/Camera/Light) + .nka MESH dedup. Rationale: pure data structure over already-existing Registry+SceneMap+nka; no Vulkan, no nk::World step. The ONE design constraint: shape it for the M11 RT consumer (`RenderWorldToTwoLevelScene`). **Depends on D1** (where visual_local is stored — likely RenderWorld caches it at build time from TransformComponent chain).
- **T2 [PAR] — `command_queue.hpp`** (manifest #5). `Command` enum (anticipate M11 `MoveEntity`) + MPSC queue. Header-only, no consumer in M8 (InputSystem drains it as a stub). Rationale: fully isolated; trivial.
- **T3 [PAR] — Vulkan raster scaffolding** (manifest #2,#3,#4): `vulkan_raster_renderer.{hpp,cpp}` skeleton + `mesh.vert` + `mesh_pbr.frag` + CMake glslc wiring. Build the VkRenderPass/VkFramebuffer/graphics-pipeline/depth/vertex-index buffers/PBR descriptors, reusing `vulkan_renderer.cpp`'s instance/device/command-pool init. Rationale: the single largest net-new lift (Risk R-raster); independent of pose plumbing — can be developed against a static `RenderWorld` fixture. **Couples to T1's RenderWorld/RenderMaterial layout** → T1 must land its header first, then T3 proceeds in parallel against it. **Depends on D3** (mesh source: real triangles vs primitive fallback).

### Phase 2 — Frame-loop spine (coupled)
- **T4 [SEQ] — pose_publisher + systems + simulation** (manifest #6–#11). `PosePublisher` interface + `HostDownloadPublisher` (Data::DownloadField LinkPose/BasePose/BodyPose for selected env) → `TransformSyncSystem` (composes `data_pose ∘ visual_local` into instance.world_xform) → `Simulation::Frame` (drain queue, step nk::World, run TransformSync+Render, **bypass all of it when render off** for G5). Rationale: these four are a tight data-flow chain (publisher feeds TransformSync feeds Simulation), share the visual_local convention (Risk R1) and the render-off bypass (Risk R3/G5), and all hinge on D1+D2. Single-flight avoids convention drift across the seam. Consumes T1 (RenderWorld) + T2 (queue) + T3 (renderer).

### Phase 3 — Recorder + bindings (mostly parallel)
- **T5 [PAR] — C-ABI Recorder** (manifest #12,#13 + CMake): `recorder.cpp` (RecorderRecord/RecorderTable mirroring diffsim.cpp; PPM P6 capture + ffmpeg-shell to_video with a clean `NUKA_RESULT_*` on missing-encoder) + `nuka_recorder.h` + add to NUKA_PUBLIC_HEADERS + add `nuka_render` to the `nuka` link (gated per D5). Rationale: C-ABI module is a well-templated, isolated add; depends on T3 (renderer output) + T4 (frame/pose source) but its own surface is independent. **Depends on D2 (capture source) + D4 (env-selection API) + D5 (Vulkan-gating of libnuka).**
- **T6 [PAR] — python binding + driver** (manifest #14,#16 + nuka_ext): `recorder.py` + `nuka_ext.cpp` Recorder binding + `examples/demo/render_rollout.py` (G4). DELETE `go2_demo_render.py`. Rationale: thin wrapper over T5's C surface; parallel once T5's header is fixed.

### Phase 4 — Gate + wireframe demotion
- **T7 [SEQ] — `render_physics_parity.cpp` + CMake** (manifest #15 + tests/CMakeLists). The G1–G3 asserts (per-entity pose==pose∘local; two-render memcmp; non-bg>0), asset-gated SKIP, inside `NK_BUILD_VULKAN_VALIDATION`, linking nk libs + nuka_render. Rationale: the gate integrates T1+T3+T4 — must land after them and encodes the exact visual_local convention chosen in D1; single-flight to keep the assertion authoritative. **Depends on D1+D2+D6.**
- **T8 [PAR] — wireframe→overlay** (`vulkan_renderer.cpp` MODIFY, L491). Demote debug wireframe to an overlay. Rationale: localized edit to the existing file; D6 decides whether it's reused as an overlay subpass or kept standalone. Parallel with T5/T6/T7.

**Critical path:** D1+D2 (decisions) → T1 → T3 → T4 → T7. T5/T6/T8 parallel off the side. T2 fully independent.

---

## 6. Risks, Open Questions, and Controller Decisions Needed BEFORE Implementation

### 6.1 Decisions the controller MUST make first (ranked)
- **D1 — visual_local provenance (TOP, blocks T1/T4/T7).** The gate asserts `render_instance.world == data.body_pose ∘ visual_local`, but **`VisualMeshComponent` (components.hpp:73) has NO local-transform field, and `CookedRef` (scene_map.hpp:20–27) has NO local-pose field.** The per-node local lives in `TransformComponent.local` ("relative to the PARENT NODE", components.hpp:31). DECISION: is `visual_local` = the entity's `TransformComponent.local`, the composed SceneGraph chain from the link/body node down to the visual, a per-visual offset, or identity? And WHERE is it materialized — cached in `RenderWorld` at `BuildRenderWorld` time, or computed live in the publisher? A convention mismatch silently passes/fails the gate. **Recommendation:** cache the composed link/body-frame→visual-frame offset in RenderInstance at build time, so TransformSync is a single multiply `downloaded_pose ∘ cached_visual_local`.
- **D2 — Recorder/gate pose SOURCE (blocks T4/T5/T7).** M8 lands BEFORE the c_abi→nk::World cutover (M9, L498); python `world.step()` still drives the LEGACY world (c_abi/world.cpp:667–685). DECISION: does the M8 Recorder + parity gate read poses (a) directly from `nk::World::GetData().DownloadField(...)` (as h1_union_parity.cpp:156–167 does — clean, M9-proof), needing a minimal nk seam ahead of the full M9 switch, OR (b) from the legacy WorldRecord path (torn out in M9)? **Recommendation:** target nk::World/Registry directly to survive M9 with zero churn; resolve the M8/M9 python-surface sequencing with the owner.
- **D3 — mesh asset readiness (blocks T3 fidelity + G4).** A real raster render needs triangle meshes; the current render path has only primitive params. M8 RenderWorld.MeshLibrary depends on M2 importers having read visual geoms (contype=0 → VisualMeshComponent, rgba → RenderMaterial, plan L334) into cooked `.nka MESH` chunks for BOTH go2 AND h1. DECISION/verify: do go2 + h1_with_hand `.nks`/`.nka` already carry VisualMeshComponents + MESH chunks by M8, or must M8 fall back to rendering collision primitives (and does `src/scene/asset/nka.*` expose MESH triangle geometry to the host renderer)? If assets are missing, the go2-mp4 half of G4 is blocked by ASSETS, not M8 code.
- **D4 — env-selection API.** §3.8 says M8 host-downloads "the SELECTED env". `nk::World` Step/Reset are all-env; no env-selection plumbing exists. DECISION: is the rendered env fixed to env 0 for the gate, or must the env index be a `nuka_recorder.h` create/camera-desc parameter threaded → Simulation → publisher → DownloadField byte-offset? **Recommendation:** env-0 default + an optional recorder param; viewer env-select stays M11.
- **D5 — libnuka Vulkan-gating.** recorder.cpp pulling `nuka_render` into the always-built `nuka` SHARED lib would make Vulkan+glslc MANDATORY for EVERY libnuka build (renderer FATAL_ERRORs if glslc absent, src/CMakeLists.txt:132). DECISION: condition recorder.cpp/its nuka_render link on `NK_BUILD_VULKAN_VALIDATION` (or a new `NK_BUILD_RECORDER` flag), or accept mandatory Vulkan for libnuka? **Recommendation:** gate it — keep libnuka buildable on Vulkan-less configs.
- **D6 — render config + comparison surface.** The plan pins NO resolution/fps/codec — only "mp4" + visual-mesh+material + non-bg>0. DECISION: (a) pick resolution/fps/codec (homepage demo may want 1080p/30/h264 per demo-homepage-readme-directives — owner input?); (b) **does G2 "two renders byte-identical" compare RAW framebuffer pixels (robust D1) or encoded mp4 bytes (encoder-dependent, fragile)?** **Recommendation:** G2 compares `VulkanOffscreenReport::pixels` (raw), never mp4 bytes. (c) Is the wireframe overlay reused as a compute-overlay subpass or rewritten as a line-list graphics pass?

### 6.2 Risks (carry into implementation)
- **R-raster (effort) —** The real triangle raster + PBR forward pipeline (RenderPass/Framebuffer/GraphicsPipeline/depth/vertex-index/PBR descriptors + 2 shaders) is the biggest net-new lift; "reuse headless Vulkan init" (L490) is only ~120 LOC of init. The existing device is created at apiVersion 1.0 with NO features — the raster path likely needs its OWN device-creation tweaks (1.1+/descriptor-indexing for mesh/material tables), not verbatim reuse.
- **R1 (visual_local convention) —** see D1; the load-bearing under-specification. Also: `LinkPose` is the FK world pose of the LINK frame; the visual_local must be in the SAME convention the cook used, or the 1:1 assert silently breaks.
- **R2 (per-link vs per-entity) —** physics poses are per-LINK/body in Data; `RenderInstance` is per-ENTITY. SceneMap resolves entity→(link_index/body_row); an off-by-one in link indexing silently passes/fails G1.
- **R3 / G5 (throughput-neutrality) —** the frame loop + pose download + render MUST be fully bypassed (not merely no-op'd) when render is off, or `Data::DownloadField`'s D2H copy / a stream sync perturbs StepPlanned graph timing and regresses the perf red-lines.
- **R-determinism / G2 —** a real raster pipeline adds depth-test tie-breaks / MSAA / driver-dependent rasterization ordering. Across ICDs (lavapipe CPU vs NVIDIA) byte-exactness is unverified → the gate must pin a deterministic pipeline config and run on the CI GPU; the renderer must control all nondeterministic inputs.
- **R5 (interop forward-compat) —** if M8 hardcodes single-env/host-staging assumptions the M11 device-scatter `CudaVulkanInteropPublisher` can't satisfy, the interop port costs more. Keep `PosePublisher` a clean interface.
- **R6 (ffmpeg dependency) —** `to_video` shells ffmpeg; headless-rendering.md explicitly says "confirm ffmpeg on PATH; if absent, any PPM→MP4 encoder works". Undeclared in plan §6 / nuka-v03 conda env. Recorder must return a clean `NUKA_RESULT_*` on missing-encoder, not crash; G4 depends on ffmpeg availability.
- **R7 (RT-debt trap) —** the `src/rt/*.cu` files are CUDA-only and NOT yet behind `RtBackendI` (that's M11, plan §3.10). If M8 drives the RT path directly for the mp4, it creates exactly the "naked .cu on the engine side" debt the owner forbids. **M8 must drive the Vulkan RASTER path for the gate/deliverable; leave RT wiring to M11.**
- **R8 (over-scope) —** Mangifera's render-graph is a SKELETON; AnKi/Filament finish the hard parts. M8 needs only a forward PBR pass — do NOT build a full auto-barrier render graph now (primacy ruling caps this).
- **R-design (late validation) —** owner reorder means M11's RT adapter + interop are RenderWorld's only second consumers, landing AFTER the review. Any RenderWorld API mistake won't be exercised until then → design for the M11 RT contract up front.
- **R-particle-snapshot (inherited debt) —** the D1 snapshot/Reset only covers articulation+body state (M7 T1); PARTICLE snapshot is still owed with named consumer "M8 RL reset". If M8's frame loop/reset touches particle worlds, the gap surfaces here (otherwise inert for the render gate).
- **R-M9-churn —** M8 lands before M9 deletes `render_scene.{hpp,cpp}` + `scene_pipeline.*` and repoints c_abi → nk::World. Build M8 against the NEW RenderWorld + nk::World/Registry (not legacy `RenderScene`/`WorldRecord` internals) to minimize M9 tear-out.

### 6.3 Open Questions (genuine unknowns, not resolvable from the plan)
- **OQ1** — Where exactly does `visual_local` live / get composed? (= D1; the §3.6 component layout has no such field.)
- **OQ2** — What state does the M8 mp4 actually capture given the legacy-vs-nk python world split? (= D2.)
- **OQ3** — Do go2 + h1_with_hand assets already yield VisualMeshComponents + `.nka MESH` chunks by M8, and does `scene/asset/nka.*` expose MESH triangle geometry to the host renderer? (= D3.)
- **OQ4** — Build env: is there a real Vulkan ICD for a triangle GRAPHICS pipeline (go2_demo_render.py header notes lavapipe works for COMPUTE) adequate/performant for the M8 raster + PBR forward pass + the CI parity gate? Unverified.
- **OQ5** — Is the M8 video deliverable produced by the RASTER path alone (RT adapter is M11), and is G2 byte-identity asserted on raster output or could it reuse the RT framebuffer's already-D1 property? (Plan ties the gate to raster; raster cross-ICD determinism unverified.) (= D6b + R7.)
- **OQ-shading** — Must `mesh_pbr.frag` match the offline path-tracer's PBR/GGX model for cross-backend consistency, or are raster-PBR (approximate realtime) and RT-PBR (ground-truth beauty) allowed to diverge? (Two-backend-one-extract shares inputs, does not mandate identical output.)
- **OQ-overlay** — Is the existing `debug_draw.comp` compute impl reused as an overlay subpass, or rewritten as a line-list graphics pass? (= D6c.)
- **OQ-coexist** — During M8, do RenderScene (for the surviving scene_pipeline/its test) and the new RenderWorld coexist, or does M8 already retarget that test? (M9 deletes RenderScene; M8 cutover boundary unstated.)

---

*Verified facts cited above against the live tree (2026-06-13): `src/render/raster` and `src/runtime/app` absent; `c_abi/` has no recorder; `src/include/nuka/` has no nuka_recorder.h; `nuka` SHARED does not link nuka_render; `Data::DownloadField` at data.hpp:58; pose FieldIds at field_ids.hpp:15–17; `CookedRef` has no local-pose field (scene_map.hpp:20–27); `VisualMeshComponent` has no visual_local (components.hpp:73); `RenderMaterial` carries full PBR (components.hpp:58–69); glslc shader pattern at src/CMakeLists.txt:131–150; Vulkan test guard `NK_BUILD_VULKAN_VALIDATION` at tests/CMakeLists.txt:3065.*

---

## 7. Controller Decisions D1–D6 (RESOLVED 2026-06-13, autonomous — engineering defaults, no owner architecture fork)

- **D1 (visual_local):** `visual_local` = the composed SceneGraph transform from the physics node frame (the link/body the entity's physics state lives on) down to the visual node, **cached in `RenderInstance` at `BuildRenderWorld` time**. TransformSync per frame = a single multiply `downloaded_pose ∘ cached_visual_local`. (Where the visual IS the physics node, visual_local = its `TransformComponent.local` chain to that node.) The gate (G1) asserts exactly this compose.
- **D2 (pose source):** parity GATE + Recorder read poses from **`nk::World::GetData().DownloadField(LinkPose/BasePose/BodyPose)` directly** (M9-proof, mirrors h1_union_parity.cpp:156–167), NOT the legacy WorldRecord. The C++ exit gate (G1–G3,G5) is fully nk::World-based. For the python mp4 (G4): drive via the nk::World-backed c_abi (union_world exposes the H1 nk path post-M7-T6); if go2's nk-world python surface needs M9 plumbing, the **go2-half of G4 is a NAMED, scoped deferral to M9** — the true exit gate does not depend on it.
- **D3 (mesh assets):** render real `.nka MESH` triangles where present; **fall back to rendering collision primitives** for any asset lacking a VisualMeshComponent/MESH chunk (gate never blocked by asset gaps). T1 investigates + reports actual go2/h1 readiness; go2-mp4 FIDELITY is asset-gated, not code-gated.
- **D4 (env-selection):** render **env 0 by default** + optional env-index recorder param; multi-env viewer select stays M11.
- **D5 (libnuka Vulkan-gating):** recorder.cpp + its `nuka_render` link are **gated** (NK_BUILD_VULKAN_VALIDATION or a Vulkan-availability flag) so **libnuka stays buildable on Vulkan-less configs**. Constraint fixed; flag name = implementer's choice.
- **D6 (render config + comparison):** (a) default **1920×1080 @ 30fps, h264/yuv420p**, configurable via recorder params (homepage-beauty = M11 path-tracer / v1.0, not M8 raster). (b) **G2 compares RAW framebuffer pixels** (`VulkanOffscreenReport::pixels` memcmp), NEVER encoded mp4 bytes. (c) wireframe→overlay form = implementer's choice (T8).

**Execution strategy:** M8's from-scratch, build-coupled Vulkan + frame-loop spine is single-flight Opus 4.8 max (genuine "can't cleanly parallelize" — shared build dir + integration-link coupling; owner's "不能使用ultracode就用max" carve-out), in dependency order T1→T3→T4→T5/T6→T7/T8, controller integration-builds + commits per phase. ultracode is used for the END-OF-M8 adversarial review (+ any clean parallel burst). T3 owns an OQ4 graphics-pipeline de-risk probe FIRST.
