# Renderer Architecture Research — a "Big Renderer" for Nuka-Physics

> **Status:** research / reference (controller-authored 2026-06-03 from a source-backed read-only research pass). NOT a committed plan — a reference for the owner to **converge + iterate AFTER v0.7 closes**. Does NOT block in-flight v0.7 (G1 / p15 / p16).
> **Owner directive (2026-06-03, 中文):** "之后对于 vulkan 渲染器，现在还是太简陋，后续需要加入一个大的渲染器来管理，[kry4r/Mangifera; godlikepanos/anki-3d-engine] 可以参考这些渲染引擎；渲染只是这个 nuka-physics 的一个部分，只是用来展示物理仿真的效果，需要以物理仿真为主。去调研下这些，同时加入记忆，等 v0.7 结束之后我会详细的规划并迭代后续的需求。"
> See memory `[[renderer-architecture-directive]]`, `[[rt-self-written-vs-optix-decision]]`, `[[sim2sim-focus-completeness-beauty]]`, and the post-v0.7 roadmap `docs/plans/2026-06-03-post-v07-roadmap.md` (W2 render + W3-U1 viewport).

## 0. Scope & primacy framing (load-bearing)

Nuka-Physics is a GPU-resident, byte-exact-deterministic (D1), differentiable **physics** engine. **Rendering is a SECONDARY subsystem whose only job is to SHOWCASE the physics. Physics simulation remains the PRIMARY focus.** The goal here is to extract renderer *architecture patterns* worth borrowing — **not** to build a game engine. Everything below is sized for that restraint. Post-v0.7; the owner will plan the detail later.

**Key finding up front:** `kry4r/Mangifera` is the **owner's OWN repository** (the Nuka-Physics git owner is `kry4r`). So this is not "two peer references" — it is **the owner's own renderer proto-design (Mangifera) measured against the mature exemplar (AnKi)**, plus two better-fit references (Filament, MuJoCo). The synthesis collapses to: *Mangifera's skeleton is the plan; AnKi + Filament are how to finish it correctly; MuJoCo is the physics-integration spine; a CUDA↔Vulkan interop layer is the piece Nuka uniquely needs.*

## 1. Existing Nuka renderer pieces (baseline, read-only)

| Piece | Files | What it is |
|---|---|---|
| `src/render/` | `vulkan_renderer.cpp` (~1.2k LOC), `render_scene.{hpp,cpp}`, `shaders/` | Crude self-written Vulkan **offscreen** renderer. Public surface ≈ probe + debug-draw (lines/spheres/capsules/boxes/AABBs/contact-points) + offscreen-to-pixel-buffer (`ProbeVulkanRenderer`, `RenderDebugDrawListVulkan`, `RenderSceneVulkan`). **No swapchain, no frame loop, no render graph, no real material system** → confirms the owner's "太简陋". |
| `src/render/render_scene.hpp` | — | Already a **decoupled render-facing scene view**: `RenderScene` = flat arrays of `RenderMeshInstance` / `RenderMaterial` (base_color/roughness/metallic — **already PBR-shaped**) / `RenderCamera` / `RenderLight` / `DebugProxy`, built by `BuildRenderScene(SceneIR&)`. Nuka's nascent **extract** layer — the right seed. |
| `src/rt/` | two-level TLAS/BLAS CUDA ray tracer (p12/p13/p14b + G1) | Self-written CUDA path: BLAS-once / TLAS-per-frame, p13 leaves (triangle/sphere/sparse-SDF), GGX+Lambert + shadows, 6-AOV framebuffer. **D1-deterministic by construction** (atomic-free, total-order tie-break). The **offline-beauty + sensor** backend. |

Nuka already has the *seed* of the right shape: a `RenderScene` extract struct + two backends (Vulkan raster, CUDA RT). **What's missing is the orchestration layer above them.**

## 2. Engine 1 — `kry4r/Mangifera` (owner's own proto-design)

**Identity.** C++ / GLSL / CMake, Vulkan-first, early (a refactor in progress, not finished). **README intent (verbatim):** *"Mangifera is a Vulkan-first rendering engine refactor that is moving from a monolithic renderer toward a renderer-centered core with explicit frame orchestration, graph-friendly feature passes, and headless simulation entry points."* — almost word-for-word the architecture this research recommends; the owner has already drafted it.

**Layout / architecture:**
- `graphics/` — the **RHI**. Clean textbook tree (`desingn.md`): `RHI → Render-Resource{Buffer, Texture, Sampler, Shader} · Pipeline-State{PipelineLayout, Graphics+ComputePipeline} · Command-Execution{CommandBuffer, CommandQueue} · Synchronization{Fence, Semaphore, Barrier} · Render-Pass{RenderPass, Framebuffer, Swapchain} · Core{Device}`; `backends/vulkan`, `capabilities/`.
- `app/render_core/` — `Frame_Context`, `Render_Scene`, **`Render_Graph`** (+`render_pass_node`), `Frame_Pipeline`, **`render_scene_extractor`**, `render_targets`, `run_mode`, **`sensor_output`**.
- `app/render_features/passes/` — `depth_prepass`, **`sensor_export_pass`**, `post/`.
- `app/headless/` — headless batch bootstrap (`--headless --frames N`).

**Maturity caveat:** the render graph is a **skeleton, not a real barrier-tracking graph** — `Render_Graph` exposes `add_pass(...)` + `compile() const -> std::vector<std::string>` (returns pass-name ordering, i.e. scheduling-by-declaration scaffolding, NOT automatic transient aliasing / barrier insertion). The right skeleton with the hard parts unfilled.

**Why it matters to Nuka:** Mangifera already encodes Nuka's exact sim-showcase concerns — `render_scene_extractor` (sim↔render decouple), `sensor_output` / `sensor_export_pass` (sensor-aware rendering, mirrors `src/rt` sensor AOVs), `--headless --frames N` (batch/offline, mirrors Nuka's offline-beauty path). None of this is in a game engine; it is the owner's physics-sim-specific intent already written down.

## 3. Engine 2 — `godlikepanos/anki-3d-engine` (mature exemplar)

**Identity.** C++ / HLSL / Python / CMake. **BSD, ~1.6k★, mature, actively developed.** APIs: **Vulkan + D3D12**. Platforms: Linux/Windows/Android. Self-described: *"Vulkan and D3D12, modern renderer, scripting, physics and more"* — a full game engine; **most subsystems are out of scope** (see §6).

**RHI (`AnKi/Gr/`) — worth studying.** Clean multi-backend RHI: `GrManager`, `CommandBuffer`, `Buffer`, `Texture`, `Sampler`, `Shader`+`ShaderProgram`, `Fence`, `AccelerationStructure` (RT), `Occlusion/Pipeline/TimestampQuery`, `GrUpscaler`, `RenderGraph`. Backends in `Vulkan/`, `D3D/`, shared `BackendCommon/`; all objects derive `GrObject`. This is the textbook RHI shape Mangifera's tree is converging toward.

**Render graph (the real thing Mangifera lacks).** Per the dev blog: *"Every pass informs about the resources it will consume and produce … the render graph inspects the dependencies to find the order … batching as many passes together as possible to minimize pipeline barriers."* It does a **per-frame topological sort**, **automatic pipeline-barrier insertion** (example frame → "14 barriers total"), **conditional passes** (shadows/probes that don't always run), and **transient render-target** management. **The single most valuable AnKi pattern for taming Nuka's crude Vulkan path.**

**Bindless / GPU-driven (analogous to Nuka's GPU-resident physics).** Three parts — **Unified Geometry Buffer** (all vtx/idx of the scene), **GPU-Scene buffer** (transforms/lights/probes persistent on-GPU), **bindless texture set**. Intent: *"persistent data in the GPU … eliminating CPU↔GPU roundtrips."* Maps directly onto the CUDA↔Vulkan interop opportunity (§5.3).

**Techniques** (clustered deferred, ESM shadows, probe diffuse GI + cubemap specular, clustered/volumetric lighting+fog, SSAO, TAA/bloom/motion-blur/tonemap, RT via `AccelerationStructure`) — **far more than Nuka needs**; borrow the *structure*, cherry-pick 2-3 techniques.

## 4. Better-fit references (added during research)

### 4a. Google **Filament** — the "renderer, not a game engine" north star
`google/filament` — C++, **Apache-2.0, ~18k★**, Vulkan/GL ES/Metal/WebGL. The closest match to what Nuka wants: a *standalone real-time PBR renderer*, explicitly **not** a game engine (no ECS/gameplay/audio). Borrow:
- A **`Renderer` that owns a `FrameGraph`** (passes declare reads/writes; graph culls unused passes, aliases transient resources, inserts barriers — same pattern as AnKi).
- A **`Driver` interface + command stream** RHI for multi-backend — the clean RHI boundary to copy conceptually.
- A focused, well-documented **PBR material model** (the *"Physically Based Rendering in Filament"* doc is gold-standard). Nuka's `RenderMaterial` is already roughness/metallic — Filament shows how to finish it without bloat.

**Takeaway:** Filament proves a clean FrameGraph + RHI + PBR can exist *without* game-engine baggage. The architectural north star for "render well without becoming an engine."

### 4b. **MuJoCo** — gold-standard physics→renderer decoupling (key for §5)
`google-deepmind/mujoco` — visualization as a textbook 3-layer split:
1. **Abstract visualization** — reads `mjModel`+`mjData`, fills `mjvScene` with abstract `mjvGeom`. **Platform-agnostic; no graphics lib required.**
2. **Rendering** — `mjvScene` → OpenGL via `mjr_render()` (`mjrContext` owns GPU resources).
3. **Interactive** — Simulate app / passive viewer.
Crucially: *"users can integrate another rendering engine … by bypassing the native OpenGL renderer while still using the abstract visualizer."* **Exactly the seam Nuka needs:** one abstract extract (`mjvScene` ≈ Nuka `RenderScene` ≈ Mangifera `render_scene_extractor`) feeding *multiple* backends (Vulkan viewport + CUDA path-tracer).

### 4c. **NVIDIA Newton / Isaac Sim** — already Nuka's v1.0 demo reference
GPU-resident robotics sims feeding a high-end renderer (Omniverse RTX / USD). Lesson (not architecture-to-clone): the industry pattern for GPU sims is **share GPU buffers with the renderer (GPU-resident scene)** rather than CPU round-trip — reinforces §5.3.

## 5. How a physics engine should integrate with a renderer (the core question)

Four patterns, priority order. Not generic "physics writes transforms into a scene graph."

**(1) Extract / snapshot decoupling.** The renderer reads an *immutable per-frame snapshot*, never live sim state. Physics owns authoritative state; once per render frame an *extractor* produces a read-only `RenderScene` (MuJoCo `mjvScene`; Mangifera `render_scene_extractor`). **Nuka already has this** (`BuildRenderScene`). Keep/harden it; never let the renderer touch live physics buffers. Also enables sim/render on different threads/cadences.

**(2) Fixed physics dt vs variable render rate → interpolation.** Nuka physics runs fixed **dt = 0.005**; a 60 fps viewport won't line up with ticks → must **lerp position + slerp quaternion between the two latest snapshots** or the viewport judders. This interpolation is the principled reason the realtime **viewport can be non-deterministic** (a *display-time* blend), fully consistent with Nuka's D1 boundary (**D1 required only on the sensor/diff CUDA-RT path, not the viewer**). The offline path-tracer renders *exact* snapshots and stays D1.

**(3) ★ CUDA ↔ Vulkan external-memory interop (zero-copy) — highest-value, Nuka-unique.** Physics state already lives in CUDA device buffers; a naive renderer would download→re-upload every frame. Instead, share GPU memory directly: allocate the buffer in **Vulkan with an export flag** (`VK_KHR_external_memory`), export an OS handle (fd/NT), **import into CUDA** via the external-resource API, map it (*"changes in CUDA visible in Vulkan and conversely"*). Synchronize the producer/consumer handoff with an **exported timeline/binary semaphore** (no CPU stall). → **physics positions/instance transforms become the renderer's instance buffer with no copy.** The Nuka analog of AnKi's GPU-scene/bindless and Isaac/Newton's GPU-resident scene. Effort **L**; the pattern-(1) CPU-copy extract is the fallback until it lands.

**(4) Two consumers of one extract, orchestrated differently.** The extract feeds **two backends**: the Vulkan *raster* viewport (real-time, interpolated, non-deterministic, render-graph-orchestrated) and the CUDA *path-tracer* (offline beauty + sensors, exact snapshot, D1, simple linear driver — **do NOT impose the render-graph on the RT path**; its two-level TLAS/BLAS is already its own scheduler).

## 6. Synthesis & recommended architectural shape

### Comparison

| Concern | Mangifera (owner's proto) | AnKi (mature) | Filament (best-fit) | → Nuka borrows |
|---|---|---|---|---|
| Scope | Renderer-only, sim-aware | Full game engine | Renderer-only | Renderer-only |
| RHI | Clean tree, partial | `Gr/` multi-backend, complete | `Driver`/`DriverAPI` | **Mature Mangifera's tree, AnKi/Filament shape** |
| Render graph | **Skeleton** (`compile()→strings`) | **Real** (auto-barrier/topo-sort/batch) | **Real** (FrameGraph, transient alias) | **Fill Mangifera's graph w/ AnKi/Filament algo** |
| Extract layer | `render_scene_extractor` ✓ | `Scene`→renderer | scene→FrameGraph | **Already in Nuka; harden** |
| Sensor/headless | `sensor_output`, `--headless` ✓ | — | offline render | **Keep — maps to `src/rt` sensors + offline** |
| Physics seam | implicit | `Physics`/`Collision` inside engine | external | **MuJoCo 3-layer + CUDA↔Vulkan interop** |

**Bottom line:** Mangifera and AnKi are not alternatives — **Mangifera *is the plan*, AnKi (and Filament) are *how to finish it correctly*.**

### Recommended layered shape

```
              physics (CUDA, fixed dt=0.005, GPU-resident)   ← PRIMARY, unchanged
                       │
 (1) SCENE-EXTRACT layer │  immutable per-frame RenderScene snapshot
     = harden BuildRenderScene │  + (2) interpolation for viewport, exact for offline
                       ├───────────────────────────────┐
                       ▼                                ▼
 VULKAN RASTER VIEWPORT (real-time 60fps)        CUDA PATH-TRACER (offline beauty + sensors)
 non-deterministic, interpolated                 D1-deterministic, exact snapshot
 ┌──────────────────────────────────┐            = src/rt as-is (two-level TLAS/BLAS)
 │ RENDER-GRAPH layer (NEW)         │            orchestrated by a SIMPLE linear driver
 │  passes declare read/write,      │            — NO render graph here
 │  auto-barrier + transient alloc  │
 │  (fill Mangifera skel w/ AnKi)   │
 ├──────────────────────────────────┤
 │ MATERIAL+SHADER layer            │   PBR (Filament model); reuse RenderMaterial
 ├──────────────────────────────────┤
 │ RHI layer (matures src/render)   │   Mangifera tree: Device/CmdBuf/Buffer/Texture/
 │                                  │   Sampler/Pipeline/Sync/Swapchain (single Vulkan
 │                                  │   backend; NO D3D12 — that's AnKi bloat)
 └──────────────────────────────────┘
                       ▲
   (3) CUDA↔Vulkan external-memory interop binds physics buffers
       directly into the RHI instance buffers (zero-copy)
```

**Relation to existing pieces:**
- `src/render/` (crude Vulkan) → **grows into the RHI + raster-viewport + render-graph** stack. Keep its `RenderScene` extract; replace offscreen-only with a real swapchain viewport + the graph above it.
- `src/rt/` (CUDA RT) → **stays as the offline-beauty / sensor backend, unchanged**, a *separate backend* under the same extract layer. **Do not graph-ify it.**

### Effort sizing (post-v0.7; owner plans later)

| Borrowed piece | Source pattern | Effort |
|---|---|---|
| Harden the extract/snapshot seam (already exists) | Mangifera `render_scene_extractor` / MuJoCo `mjvScene` | **S** |
| Viewport interpolation (lerp/slerp between snapshots) | physics-render-rate decoupling | **S** |
| Proper RHI (Device/CmdBuf/Buffer/Texture/Sampler/Pipeline/Sync/Swapchain) | Mangifera tree / AnKi `Gr` / Filament `Driver` | **M** |
| Render-graph w/ auto-barriers + transient aliasing | Fill Mangifera `Render_Graph` using AnKi/Filament FrameGraph algo | **M–L** |
| PBR material/shader system | Filament material model | **M** |
| CUDA↔Vulkan external-memory + semaphore interop (zero-copy) | `VK_KHR_external_memory` + CUDA interop | **L** (highest payoff; CPU-copy extract is the fallback) |

### Explicit NOT-build list (rendering is secondary)

Do **not** import from AnKi (or any game engine): **ECS / gameplay / scene-script, audio, networking, editor/UI tooling, asset-streaming pipelines, animation/skeletal rigs beyond demo needs, the D3D12 backend** (Nuka = Vulkan+CUDA only), and **don't over-build the GI/shadow/post stack** — pick the 2-3 techniques a physics demo actually shows (e.g. soft shadows + tonemap + simple AO), not AnKi's full clustered-deferred-GI-volumetric suite. The render graph belongs **only** over the Vulkan side, not the CUDA RT path.

## 7. Sources

- `github.com/kry4r/Mangifera` — README, `graphics/desingn.md` RHI tree, `app/render_core/`, `app/render_features/passes/` (owner's own proto-design)
- `github.com/godlikepanos/anki-3d-engine` + dev blog *Anatomy of a frame in AnKi*, *GPU driven rendering in AnKi*
- `github.com/google/filament` + Filament docs + *PBR in Filament*
- MuJoCo Visualization docs (`mjvScene`/`mjr_render`/`mjvGeom`) + DeepWiki MuJoCo Visualization
- CUDA Programming Guide (Graphics Interop), `VK_KHR_external_memory` (Khronos registry), Vulkan-CUDA memory interoperability notes
- Local (read-only): `src/render/{vulkan_renderer.hpp, render_scene.hpp}`, `src/rt/`
