# Code Context

## Files Retrieved
1. `tests/scene/test_scene_compose.cpp` (lines 1-400) - existing pure SceneIR compose tests: helper style, count assertions, placement semantics, D1 checks.
2. `tests/CMakeLists.txt` (lines 287-304) - current `nuka_scene_test` target block and link libraries.
3. `src/scene/scene_ir.hpp` (lines 1-250) - SceneIR record fields, mutators/accessors, counts, and bulk accessors needed to author table/cup/H1 assertions.
4. `src/scene/scene_compose.hpp` (lines 1-45) - `Compose` contract: append/remap ids, root placement, optional name prefix, purity/D1.
5. `src/scene/cooker.hpp` (lines 1-14) - `CookScene(const SceneIR&)` entry point.
6. `src/scene/cooked_blob.hpp` (lines 1-288) - cooked table/count fields for post-compose cook assertions.
7. `src/import/mjcf_importer.hpp` (lines 1-13) - `LoadMjcf(path)` returns `SceneIR`.
8. `src/import/usd_importer.hpp` (lines 1-15) - `LoadUsd(path)` returns `SceneIR`.
9. `tests/coresident/test_h1_dense_grasp.cpp` (lines 81-210) - asset constants, availability check, cup USD loading/cooking pattern, H1 MJCF loading helper.
10. `tests/coresident/test_h1_dense_grasp.cpp` (lines 234-353) - dense grasp finger/body names and minimal reachability-related constants/helpers.
11. `tests/coresident/test_h1_dense_grasp.cpp` (lines 456-566) - shallow/caging reachability thresholds and search predicates to avoid copying wholesale.

## Key Code

`Compose` is the core API to exercise:

```cpp
SceneIR Compose(const SceneIR& base, const SceneIR& addon,
                const math::Transform& placement,
                const std::string& addon_name_prefix = "");
```

Important contract from `src/scene/scene_compose.hpp`:
- Copies `base`, appends all `addon` records, offsets ids by base counts.
- Preserves invalid sentinels.
- Applies `placement * original_local_transform` only to addon root bodies.
- Optional `addon_name_prefix` is prepended to appended named records.
- Pure and deterministic.

SceneIR authoring primitives from `src/scene/scene_ir.hpp`:
- `AddRigidBody(RigidBodyRecord)` / `AddRigidBody(std::string)`.
- `AddMaterial(MaterialRecord)`.
- `AddCollisionShape(CollisionShapeRecord)`.
- Counts: `RigidBodyCount`, `JointCount`, `ShapeCount`, `MaterialCount`, etc.
- Accessors: `GetBody`, `GetShape`, `GetBodyMut`, `GetShapeMut`, `Bodies`, `Shapes`.

For a table helper, author a tiny static body + box shape:

```cpp
SceneIR MakeTableScene() {
    SceneIR table;
    RigidBodyRecord body;
    body.name = "table";
    body.parent_id = kInvalidBody;
    body.is_static = true;
    body.mass = 0.0f;
    body.local_transform = Transform{Vec3{0.0f, 0.0f, 0.0f}, Quat::Identity()};
    const BodyId table_body = table.AddRigidBody(std::move(body));

    MaterialRecord mat;
    mat.name = "table_mat";
    mat.friction_mu = 1.0f;
    const MaterialId table_mat = table.AddMaterial(std::move(mat));

    CollisionShapeRecord shape;
    shape.name = "table_top";
    shape.body_id = table_body;
    shape.material_id = table_mat;
    shape.type = ShapeType::Box;
    shape.half_extents = Vec3{0.6f, 0.4f, 0.025f};
    table.AddCollisionShape(std::move(shape));
    return table;
}
```

Adjust table dimensions/height in the final test as needed, but keep it programmatic SceneIR; do not add generated/golden assets.

Asset paths from `test_h1_dense_grasp.cpp`:
- H1 MJCF: `.nuka-assets/newton_assets/unitree_h1/mjcf/h1_with_hand.xml`.
- Cup USD currently used there: `.nuka-assets/newton_assets/manipulation_objects/cup/model.usda`.
- Task asks for `model_large` cup; search likely sibling path under `.nuka-assets/newton_assets/manipulation_objects/cup/` or use exact task-known path if available. Gate with `std::filesystem::exists` and `GTEST_SKIP()` if absent.

Cup importer/cook pattern from dense test:

```cpp
auto cup = nuka::import::LoadUsd(kCupUsda);
for (size_t i = 0; i < cup.ShapeCount(); ++i) {
    auto& shape = cup.GetShapeMut(static_cast<nuka::scene::ShapeId>(i));
    if (!shape.mesh_vertices.empty()) {
        shape.decompose_mode = nuka::scene::DecomposeMode::Skip;
    }
}
const auto cup_blob = nuka::scene::CookScene(cup);
```

Use that only if the test needs a single hull or stable cook cost. For a pure scene-compose pipeline smoke, it may be enough to load the USD and compose/cook once.

## Architecture

Recommended implementation shape:
1. Add a new scene test translation unit, e.g. `tests/scene/test_scene_compose_h1_cup_table.cpp`, rather than growing `test_scene_compose.cpp` with asset-gated integration logic. This keeps the existing unit-level compose tests pure and fast.
2. The test should be asset-gated:
   - `if (!std::filesystem::exists(kH1Mjcf) || !std::filesystem::exists(kCupLargeUsd)) GTEST_SKIP();`
   - Work from `${CMAKE_SOURCE_DIR}` because `nuka_scene_test` already has `WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}`.
3. Load H1 via `nuka::import::LoadMjcf(kH1Mjcf)`.
4. Load large cup via `nuka::import::LoadUsd(kCupLargeUsd)`.
5. Build table via local `MakeTableScene()` in the new test file.
6. Compose in deterministic fixed order:
   - `SceneIR scene = MakeTableScene();`
   - `scene = Compose(scene, h1, h1_placement, "h1/");`
   - `scene = Compose(scene, cup, cup_placement, "cup/");`
7. Count assertions should compare sums from the three source scenes, not magic numbers:
   - body/joint/shape/material/sensor/camera/light/actuator counts equal table + H1 + cup counts.
8. Cook assertion should call `CookScene(scene)` and assert cooked top-level counts mirror SceneIR counts where applicable:
   - `blob.body_count == scene.RigidBodyCount()`
   - `blob.joint_count == scene.JointCount()`
   - `blob.shape_count == scene.ShapeCount()`
   - `blob.material_count == scene.MaterialCount()`
   - plus table static mass/pose if easy.
9. D1 check should compose/load twice in the same process and compare stable observable fields:
   - counts equal;
   - appended root/body names equal;
   - selected local transforms equal bitwise or with small epsilon;
   - cooked counts equal.
   Avoid comparing every mesh vertex unless needed; that risks large brittle assertions.
10. Cup-on-table check can be purely geometric from SceneIR/cooked transforms:
   - table top world z = `table_body.local_transform.position.z + shape.local_transform.position.z + table_half_extents.z` if shape local is identity.
   - place cup root so its local/root position z is at or slightly above table top.
   - If cup mesh vertices are available, compute min local z across cup shapes' `mesh_vertices`, then set `cup_placement.position.z = table_top_z - min_cup_z + clearance`.
   - Assert cup root/world placement has z >= table top, and/or cooked cup root pose has expected z.
11. Reachability should be a light sanity check, not dense grasp copy:
   - Reuse only the body-name constants from dense test as strings for known right-hand links (`right_hand_link`, `R_index_proximal`, etc.).
   - Find relevant H1 bodies by name in composed SceneIR with `h1/` prefix.
   - Assert the cup center is within a broad distance band from `right_hand_link` or a simple hand anchor, e.g. not kilometers away and plausibly near tabletop. Since SceneIR has local hierarchy but no FK in scene tests, avoid importing articulation helpers unless creating a coresident/runtime test.
   - If stronger reachability is needed, keep it analytic: place cup near the H1 hand nominal area and assert named hand bodies exist after prefixing; do not copy `ForwardKinematics`, dense sphere chains, or caging search.

Concrete CMake wiring options:

Preferred: add the new file to existing scene target:

```cmake
add_executable(nuka_scene_test
    scene/test_scene_ir.cpp
    scene/test_scene_compose.cpp
    scene/test_scene_cooker.cpp
    scene/test_scene_pipeline.cpp
    scene/test_contact_metadata_cook.cpp
    scene/test_filter_precedence.cpp
    scene/test_scene_compose_h1_cup_table.cpp
)
```

No new link libraries are needed for this option: `nuka_scene_test` already links `nuka_scene_pipeline`, `nuka_import`, and `GTest::gtest_main`, and `CookScene`, `LoadMjcf`, `LoadUsd`, and `Compose` are already reachable in existing scene tests.

Separate target option only if the test is slow/asset-heavy and the parent wants isolated naming:

```cmake
add_executable(nuka_scene_h1_cup_table_test
    scene/test_scene_compose_h1_cup_table.cpp
)
target_link_libraries(nuka_scene_h1_cup_table_test
    PRIVATE
        nuka_scene_pipeline
        nuka_import
        GTest::gtest_main
)
gtest_discover_tests(nuka_scene_h1_cup_table_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
```

Prefer the existing `nuka_scene_test` target unless runtime cost becomes significant; the test is asset-gated and read-only.

## Start Here

Open `tests/scene/test_scene_compose.cpp` first. It already shows the exact local helper/test style for `SceneIR`, `Compose`, sum-count assertions, placement semantics, and D1 assertions. Then add the asset-gated integration test as a new file under `tests/scene/` and wire it into `tests/CMakeLists.txt` at the scene target block.

## Supervisor coordination

Constraints and risks:
- Do not modify generated/goldens or production steppers; this task can be entirely additive in tests/CMake.
- Keep code/comments English.
- Parent commits; do not commit.
- `model_large` exact path was not in the inspected files. Confirm by targeted file search before implementation, or use the task-provided exact path if the parent has one.
- SceneIR alone does not provide articulated FK. Do not copy dense grasp machinery into scene tests; use a simple named-body/prefix and broad placement sanity check unless the task is explicitly broadened to coresident/runtime.
- If CMake target isolation is desired for asset-gated/slower behavior, use the separate target option above; otherwise adding one source to `nuka_scene_test` is minimal and consistent.
