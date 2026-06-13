# Code Context

## Files Retrieved
1. `src/rt/two_level_render.hpp` (lines 1-112) - core self-written TLAS/BLAS API, D1 contract, mesh/instance/material types.
2. `src/rt/framebuffer.hpp` (lines 1-73) - AOV layout and color/depth/normal/albedo/uv/prim buffers to serialize/check.
3. `src/rt/material.hpp` (lines 1-48) - bounded material/light structs for p13/two-level shading.
4. `src/rt/camera.hpp` (lines 1-121) - deterministic pinhole camera and `BuildPinhole` entry point.
5. `tests/rt/test_two_level_render.cpp` (lines 334-655) - canonical two-level scene construction, transform-tracking, and D1 memcmp gates.
6. `src/CMakeLists.txt` (lines 598-635) - `nuka_rt_gpu` target wiring and CUDA `--fmad=false` requirement.
7. `tests/CMakeLists.txt` (lines 1187-1298) - RT gtest executable patterns and `-ffp-contract=off` target setup.
8. `src/scene/scene_ir.hpp` (lines 30-122) - `CollisionShapeRecord` mesh payload, materials, and cameras available to converter.
9. `src/import/mesh_file_loader.hpp` (lines 1-62) - self-written STL/OBJ loader and D1 file-order guarantees.
10. `src/import/mjcf_importer.cpp` (lines 143-164, 384-544) - MJCF `type="mesh" -> TriMesh`, meshdir resolution, STL/OBJ load into SceneIR.
11. `src/render/render_scene.hpp` (lines 15-80) and `src/render/render_scene.cpp` (lines 25-96) - existing render-facing scene omits triangle arrays; converter should not rely on it for RT meshes.
12. `.nuka-assets/newton_assets/unitree_h1/mjcf/h1_with_hand.xml` (lines 1-78, 168-178, 200-207) - H1 STL assets and duplicate visual/collision mesh geoms.
13. `docs/plans/2026-06-03-v07-p16-asset-pipeline-reshape.md` (lines 54-59) - prior RT-video spine: pose snapshots -> `rt::TwoLevelScene` -> PPM -> ffmpeg.
14. `src/apps/debug_shell/main.cpp` (lines 50-83) - CLI argument convention for existing scene demo executable.
15. `src/apps/debug_shell/headless_debug_renderer.cpp` (lines 174-185) - existing binary PPM writer shape.
16. `docs/architecture/headless-rendering.md` (lines 17-23, 50-58) - documented PPM -> ffmpeg headless path.
17. `examples/demo/render_video.sh` (lines 70-92) - existing PPM frame encoding and sample PNG conversion convention.
18. `tools/scripts/plot_invariants.py` (lines 167-241) - pure-stdlib PNG chunk writer pattern if a no-ffmpeg PNG option is required.
19. `src/include/nuka/nuka.h` (lines 126-190, 286-306), `src/c_abi/buffer.cpp` (lines 24-58), `examples/demo/go2_demo_capture.py` (lines 68-116), `examples/sim_val/g1h1_drive_harness.py` (lines 145-186, 307-322) - pose-dump data contract and examples of reading `ARTICULATION_LINK_POSE`.

## Key Code

```cpp
// src/rt/two_level_render.hpp:53-75
struct BlasMesh {
    std::vector<TrianglePrim> triangles;
    std::vector<SpherePrim> spheres;
    std::vector<SdfPrim> sdfs;
};
struct Instance {
    uint32_t blas_id = 0u;
    math::Transform transform;
    uint32_t material_id = 0u;
};
struct TwoLevelScene {
    std::vector<BlasMesh> meshes;
    std::vector<Instance> instances;
    std::vector<Material> materials;
    Light light;
    AmbientTerm ambient;
};
```

```cpp
// src/rt/two_level_render.hpp:98-109
TwoLevelSceneDevice BuildTwoLevelScene(const TwoLevelScene& scene);
Framebuffer RenderFrame(TwoLevelSceneDevice& device,
                        const TwoLevelScene& scene,
                        const PinholeCamera& camera);
```

```cpp
// src/scene/scene_ir.hpp:51-52
std::vector<float>    mesh_vertices;   // x,y,z triples (source mesh)
std::vector<uint32_t> mesh_indices;    // triangle indices (source mesh)
```

```cpp
// src/import/mesh_file_loader.hpp:27-34
struct MeshGeometry {
    std::vector<float>    vertices;
    std::vector<uint32_t> indices;
    std::size_t VertexCount() const { return vertices.size() / 3; }
    std::size_t TriangleCount() const { return indices.size() / 3; }
};
```

## Architecture

- The RT path is already self-written and suitable for TASK 2d: `nuka_rt_gpu` compiles `rt/bvh_ray_traversal.cu`, `rt/scene_render.cu`, and `rt/two_level_render.cu` with CUDA `--fmad=false` (`src/CMakeLists.txt:616-624`). Tests pair host oracles with `-ffp-contract=off` (`tests/CMakeLists.txt:1218-1219`, `1254-1255`).
- `RenderFrame` expects immutable unique BLAS meshes plus mutable per-frame instances. Build BLAS once with `BuildTwoLevelScene`, then update `scene.instances[i].transform` each frame and call `RenderFrame`; tests already prove this tracking (`tests/rt/test_two_level_render.cpp:587-617`).
- The framebuffer is host-side float AOVs plus uint prim IDs. D1 is normally a bytewise `memcmp` over all six AOVs (`tests/rt/test_two_level_render.cpp:622-655`). For exported screenshots/video, quantize `color` deterministically to RGB8 after `RenderFrame` and write P6 PPM.
- `SceneIR` now has enough raw mesh storage for imported MJCF/STL and USD/USDC mesh shapes: `CollisionShapeRecord::mesh_vertices/mesh_indices`. MJCF mesh assets are resolved with `<compiler meshdir>` and loaded via self-written `LoadMeshFile` (`src/import/mjcf_importer.cpp:384-420`, `526-543`). USD mesh geometry is also populated by importer code, but TASK 2d should not add production OpenUSD.
- Important caveat: `render::RenderScene` does **not** carry mesh vertices/indices (`src/render/render_scene.hpp:15-25`, `src/render/render_scene.cpp:45-55`), so an RT converter should read `scene::SceneIR` directly or add a new sidecar converter type; do not route through `render::BuildRenderScene` if STL visuals are required.
- H1 visuals: `h1_with_hand.xml` declares STL assets for body/hand links (`.nuka-assets/.../h1_with_hand.xml:18-70`) and visual/collision mesh geoms (`lines 76-77`, `170-178`, `201-207`). Because the MJCF importer loads every TriMesh geom into `mesh_vertices/mesh_indices`, SceneIR has enough triangle data for H1 STL visuals, though duplicated visual+collision geoms should be deduped by geometry identity/shape name/body if the converter wants fewer render instances.
- Existing pose-dump bridge: `ARTICULATION_LINK_POSE` is env-major, one `math::Transform` per link, `[px,py,pz,qw,qx,qy,qz]`, 7 floats/28 bytes (`src/c_abi/buffer.cpp:24-36`). Existing Python capture uses this exact buffer into frame arrays (`examples/demo/go2_demo_capture.py:103-116`). For final 2b wiring, consume the pose dump as the authoritative per-frame link transform source.

## Implementation Plan

1. Add an additive RT conversion helper, preferably under `src/rt/` (for reusable C++ tests/tools) or `src/apps/rt_render_demo/` if kept app-local: `SceneIrToTwoLevelScene(const scene::SceneIR&, ConverterOptions)`. It should scan `SceneIR::Shapes()`, skip non-renderable collision-only duplicates if requested, turn each TriMesh shape with non-empty `mesh_vertices/mesh_indices` into an `rt::BlasMesh`, and map materials from `MaterialRecord` to `rt::Material`.
2. Mesh conversion details: for each index triplet, read `mesh_vertices[3*i + 0..2]` into `rt::TrianglePrim{v0,v1,v2,0}` in mesh-local coordinates. Keep shape-local transform in the instance transform (`body_world * shape.local_transform`) rather than baking vertices, so BLAS stays reusable and per-frame pose updates only mutate `Instance::transform`.
3. Dedup strategy: use a deterministic key over triangle source (shape mesh pointer/id plus `mesh_indices`/`mesh_vertices` content hash, or simpler first pass: one BLAS per shape). One-BLAS-per-shape is safest and still under caps for H1; dedup can be additive if render performance needs it.
4. Primitive fallback for smoke: add internal helpers that generate box/table/cup-proxy triangle meshes without Bullet/MuJoCo/OpenUSD. This enables a standalone smoke before 2b: table plane/box + cup cylinder/box proxy + a few H1/STL shapes if assets exist, all rendered through RT.
5. Add an executable frame loop modeled on existing CLI conventions: `nuka_rt_frame_demo <scene.xml|scene.usda|...> <frames-dir> [width height] [max_frames] [pose_dump]`. For standalone smoke, no pose dump is required: load/import SceneIR or construct programmatic scene, build BLAS once, render one or N frames while applying a deterministic transform animation.
6. PPM/PNG: write PPM P6 in C++ first (`P6\nW H\n255\n` + RGB bytes), matching `DebugRasterImage::WritePpm` and docs. Clamp/quantize `rt::Framebuffer::color` with a fixed rule, e.g. `uint8_t(round(clamp(c,0,1)*255))`. For PNG, prefer optional postprocess via existing `ffmpeg -i frame.ppm sample.png`; only add a tiny stdlib/zlib PNG writer if the task explicitly requires PNG without external tools (pattern exists in `tools/scripts/plot_invariants.py`).
7. Deterministic D1 frame check: add a smoke/gtest or app `--check-d1` mode that renders the same frame twice after a single `BuildTwoLevelScene` and asserts `memcmp` over all six raw AOV buffers before RGB8 quantization. Also compare the emitted PPM bytes for one frame to catch serialization nondeterminism.
8. Standalone smoke before 2b: create a minimal `tests/rt/test_sceneir_two_level_converter.cpp` or app smoke that uses a synthetic SceneIR with one TriMesh quad/table, one material, one camera, and a deterministic light. It should build, render, assert non-background pixels/hits, write a temp PPM if app-level, and run D1. If asset-gated H1 smoke is added, skip honestly when `.nuka-assets/newton_assets/unitree_h1/mjcf/h1_with_hand.xml` is missing.
9. Final wiring to 2b pose dump: after the 2b grasp/pose-dump exists, load the same H1 SceneIR, build a stable mapping from `BodyId`/body name to pose-dump link index, then for each frame set `Instance::transform = link_world_pose[body_id] * shape.local_transform`. Keep BLAS/materials fixed, rebuild TLAS per `RenderFrame`, write `frame_%05d.ppm`, then use `ffmpeg` for MP4 and sample PNGs.
10. CMake wiring: put converter code in a small library linked by both tests and app, e.g. `nuka_rt_scene_converter` (depends on `nuka_scene`, `nuka_import`, `nuka_rt_gpu` only if it exposes render helpers; pure converter can avoid CUDA). For tests under `if(CMAKE_CUDA_COMPILER)`, follow `nuka_rt_two_level_test`: include `${CMAKE_SOURCE_DIR}` and `${CMAKE_SOURCE_DIR}/src`, link `nuka_rt_gpu`, `nuka_collision_gpu`, `nuka_cooker` if using SDF fixtures, `nuka_phi`, and `GTest::gtest_main`, and set `-ffp-contract=off` on any TU doing host/device float parity.

## Risks / Gaps

- SceneIR has collision-shape meshes, not a distinct visual mesh layer. For H1 this likely suffices because visual and collision mesh geoms both reference the STL, but duplicate visual/collision entries may render doubled surfaces unless filtered.
- `RenderScene` drops triangle payloads, so reusing it directly would silently lose H1 STL visuals.
- MJCF importer stores rgba only through named materials; inline `rgba` on geoms is not surfaced in `CollisionShapeRecord`, so H1 hand geoms without named material may fall back to default material color unless converter adds a deterministic heuristic.
- Cameras in SceneIR are authored transforms/FOV only; for smoke render it may be simpler and more deterministic to use `rt::BuildPinhole` with an app-selected camera until final camera authoring is decided.
- Pose-dump/link mapping is the biggest integration risk. The C ABI exposes slot order, but final 2b dump must include enough metadata (body/link names or stable cooked order) to map H1 SceneIR bodies to pose rows without guessing.
- Large duplicated H1 meshes could approach runtime/memory limits if every visual+collision duplicate becomes a separate BLAS. The prim cap itself is probably fine (4096 instances, 2^20 prims/BLAS), but first implementation should print shape/triangle/instance counts.
- Do not introduce production OpenUSD/Bullet/MuJoCo rendering paths. Use existing importers only to populate SceneIR, and the existing self-written RT path for rendering.

## Validation Commands

- Build existing RT gates: `cmake --build build-cuda128 --target nuka_rt_scene_render_test nuka_rt_two_level_test -j`
- Run existing RT gates: `ctest --test-dir build-cuda128 -R 'RtSceneRender|RtTwoLevelRender' --output-on-failure`
- Build importer mesh tests: `cmake --build build-cuda128 --target nuka_import_test -j`
- Run importer mesh tests: `./build-cuda128/tests/nuka_import_test --gtest_filter='MeshFileLoader.*:MjcfMeshWiring.*' --gtest_color=no`
- After adding converter smoke: `cmake --build build-cuda128 --target nuka_rt_sceneir_converter_test -j` then `./build-cuda128/tests/nuka_rt_sceneir_converter_test --gtest_color=no`
- After adding app smoke: `cmake --build build-cuda128 --target nuka_rt_frame_demo -j` then `./build-cuda128/src/nuka_rt_frame_demo --smoke out/rt_smoke/frames 320 180 2 --check-d1`
- Encode frames: `ffmpeg -y -framerate 30 -i out/rt_smoke/frames/frame_%05d.ppm -c:v libx264 -pix_fmt yuv420p out/rt_smoke/smoke.mp4`

## Start Here

Open `src/rt/two_level_render.hpp` first, because TASK 2d should build around `rt::TwoLevelScene`, `BuildTwoLevelScene`, and `RenderFrame` rather than adding a new renderer. Then open `src/import/mjcf_importer.cpp` around mesh loading to confirm the H1 STL triangles already flow into `SceneIR`.
