# Nuka Physics Engine P0 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the greenfield CUDA-first rigid-body and articulation physics engine foundation, plus scene compilation, basic sensors, and a native debug shell for single-scene validation with batched-world extension points.

**Architecture:** The implementation starts with core infrastructure, then builds canonical scene compilation, PHI, runtime containers, rigid simulation, collision, constraints, articulation, sensors, and debug tooling in that order. Every runtime-facing feature is compiled from authoring formats into canonical IR and then cooked into device-friendly runtime blobs. The hot path lives in CUDA-oriented data layouts while host code owns orchestration, compilation, tooling, and verification.

**Tech Stack:** C++20, CUDA, Vulkan, CMake, GoogleTest, Python 3.11, Dear ImGui, tinyxml2, USD SDK, optional pybind11 for tooling bindings.

**Execution Notes:** Use `@superpowers:test-driven-development` and `@superpowers:verification-before-completion` during execution. If the repository is still not a Git repo when execution starts, run Task 1 first to initialize it before attempting task commits.

---

### Task 1: Repository Bootstrap

**Files:**
- Create: `CMakeLists.txt`
- Create: `cmake/PreventInSourceBuilds.cmake`
- Create: `cmake/Warnings.cmake`
- Create: `src/CMakeLists.txt`
- Create: `tests/CMakeLists.txt`
- Create: `.gitignore`
- Create: `tests/smoke/test_build_smoke.cpp`
- Create: `src/core/version.hpp`
- Create: `src/core/version.cpp`

**Step 1: Write the failing test**

```cpp
#include <gtest/gtest.h>
#include "core/version.hpp"

TEST(BuildSmoke, EngineVersionStringPresent) {
    EXPECT_STREQ(nuka::core::EngineVersion(), "0.1.0-dev");
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
git init
cmake -S . -B build -DNK_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure -R BuildSmoke
```

Expected: configuration succeeds only after build files exist, and the test fails with a missing symbol or missing target before `version.cpp` is added.

**Step 3: Write minimal implementation**

```cpp
#pragma once

namespace nuka::core {
const char* EngineVersion();
}
```

```cpp
#include "core/version.hpp"

namespace nuka::core {
const char* EngineVersion() { return "0.1.0-dev"; }
}
```

**Step 4: Run test to verify it passes**

Run:

```bash
cmake -S . -B build -DNK_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure -R BuildSmoke
```

Expected: `BuildSmoke` passes.

**Step 5: Commit**

```bash
git add CMakeLists.txt cmake src tests .gitignore
git commit -m "build: bootstrap CMake project and test harness"
```

### Task 2: Math and Spatial Algebra Core

**Files:**
- Create: `src/math/vec3.hpp`
- Create: `src/math/quat.hpp`
- Create: `src/math/transform.hpp`
- Create: `src/math/spatial.hpp`
- Create: `tests/math/test_spatial_math.cpp`
- Modify: `src/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

**Step 1: Write the failing test**

```cpp
#include <gtest/gtest.h>
#include "math/quat.hpp"
#include "math/transform.hpp"

TEST(SpatialMath, RotateAxisAngleVector) {
    const auto q = nuka::math::Quat::FromAxisAngle({0.0f, 0.0f, 1.0f}, 3.1415926f * 0.5f);
    const auto v = q.Rotate({1.0f, 0.0f, 0.0f});
    EXPECT_NEAR(v.x, 0.0f, 1e-4f);
    EXPECT_NEAR(v.y, 1.0f, 1e-4f);
    EXPECT_NEAR(v.z, 0.0f, 1e-4f);
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build
ctest --test-dir build --output-on-failure -R SpatialMath
```

Expected: compile failure because math headers and types do not exist yet.

**Step 3: Write minimal implementation**

```cpp
struct Vec3 {
    float x, y, z;
};

struct Quat {
    float w, x, y, z;
    static Quat FromAxisAngle(Vec3 axis, float angle);
    Vec3 Rotate(Vec3 v) const;
};
```

**Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build
ctest --test-dir build --output-on-failure -R SpatialMath
```

Expected: `SpatialMath` passes.

**Step 5: Commit**

```bash
git add src/math tests/math src/CMakeLists.txt tests/CMakeLists.txt
git commit -m "math: add core vector, quaternion, and transform types"
```

### Task 3: PHI v1 Surface

**Files:**
- Create: `src/phi/device.hpp`
- Create: `src/phi/buffer.hpp`
- Create: `src/phi/stream.hpp`
- Create: `src/phi/capabilities.hpp`
- Create: `src/phi/backend_cuda/device.cu`
- Create: `src/phi/backend_cuda/buffer.cu`
- Create: `tests/phi/test_capabilities.cpp`
- Modify: `src/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

**Step 1: Write the failing test**

```cpp
#include <gtest/gtest.h>
#include "phi/capabilities.hpp"

TEST(PhiCapabilities, ReportsBasicRuntimeProperties) {
    const auto caps = nuka::phi::QueryCapabilities();
    EXPECT_GT(caps.warp_size, 0);
    EXPECT_GE(caps.max_threads_per_block, 32);
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build
ctest --test-dir build --output-on-failure -R PhiCapabilities
```

Expected: missing PHI headers or missing implementation.

**Step 3: Write minimal implementation**

```cpp
struct Capabilities {
    int warp_size;
    int max_threads_per_block;
};

Capabilities QueryCapabilities();
```

```cpp
Capabilities QueryCapabilities() {
    return {32, 1024};
}
```

**Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build
ctest --test-dir build --output-on-failure -R PhiCapabilities
```

Expected: `PhiCapabilities` passes on a CUDA-capable machine and can be stubbed in a host-only CI mode if needed.

**Step 5: Commit**

```bash
git add src/phi tests/phi src/CMakeLists.txt tests/CMakeLists.txt
git commit -m "phi: add initial device, stream, buffer, and capability interfaces"
```

### Task 4: Canonical Scene IR

**Files:**
- Create: `src/scene/canonical_types.hpp`
- Create: `src/scene/scene_ir.hpp`
- Create: `src/scene/scene_ir.cpp`
- Create: `tests/scene/test_scene_ir.cpp`
- Modify: `src/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

**Step 1: Write the failing test**

```cpp
#include <gtest/gtest.h>
#include "scene/scene_ir.hpp"

TEST(SceneIr, SupportsRigidBodyAndJointRecords) {
    nuka::scene::SceneIR scene;
    const auto body = scene.AddRigidBody("base_link");
    const auto joint = scene.AddJoint("joint0", body, body);
    EXPECT_EQ(scene.RigidBodyCount(), 1u);
    EXPECT_EQ(scene.JointCount(), 1u);
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build
ctest --test-dir build --output-on-failure -R SceneIr
```

Expected: missing scene IR types.

**Step 3: Write minimal implementation**

```cpp
class SceneIR {
public:
    using BodyId = uint32_t;
    using JointId = uint32_t;
    BodyId AddRigidBody(std::string name);
    JointId AddJoint(std::string name, BodyId parent, BodyId child);
    size_t RigidBodyCount() const;
    size_t JointCount() const;
};
```

**Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build
ctest --test-dir build --output-on-failure -R SceneIr
```

Expected: `SceneIr` passes.

**Step 5: Commit**

```bash
git add src/scene tests/scene src/CMakeLists.txt tests/CMakeLists.txt
git commit -m "scene: add canonical scene IR core types"
```

### Task 5: Runtime Blob Cooking

**Files:**
- Create: `src/scene/cooked_blob.hpp`
- Create: `src/scene/cooker.hpp`
- Create: `src/scene/cooker.cpp`
- Create: `tests/scene/test_scene_cooker.cpp`
- Modify: `src/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

**Step 1: Write the failing test**

```cpp
#include <gtest/gtest.h>
#include "scene/cooker.hpp"

TEST(SceneCooker, EmitsBodyTableForMinimalScene) {
    nuka::scene::SceneIR scene;
    scene.AddRigidBody("root");
    const auto blob = nuka::scene::CookScene(scene);
    EXPECT_EQ(blob.body_count, 1u);
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build
ctest --test-dir build --output-on-failure -R SceneCooker
```

Expected: missing cooker implementation.

**Step 3: Write minimal implementation**

```cpp
struct CookedBlob {
    uint32_t body_count;
    uint32_t joint_count;
};

CookedBlob CookScene(const SceneIR& scene);
```

**Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build
ctest --test-dir build --output-on-failure -R SceneCooker
```

Expected: `SceneCooker` passes.

**Step 5: Commit**

```bash
git add src/scene tests/scene src/CMakeLists.txt tests/CMakeLists.txt
git commit -m "scene: add first cooked runtime blob path"
```

### Task 6: MJCF and URDF Importer Skeleton

**Files:**
- Create: `src/import/mjcf_importer.hpp`
- Create: `src/import/mjcf_importer.cpp`
- Create: `src/import/urdf_importer.hpp`
- Create: `src/import/urdf_importer.cpp`
- Create: `tests/import/test_mjcf_importer.cpp`
- Create: `tests/import/test_urdf_importer.cpp`
- Create: `tests/data/minimal_arm.xml`
- Create: `tests/data/minimal_arm.urdf`
- Modify: `src/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

**Step 1: Write the failing test**

```cpp
#include <gtest/gtest.h>
#include "import/mjcf_importer.hpp"

TEST(MjcfImporter, LoadsMinimalBodyAndJoint) {
    const auto scene = nuka::import::LoadMjcf("tests/data/minimal_arm.xml");
    EXPECT_EQ(scene.RigidBodyCount(), 2u);
    EXPECT_EQ(scene.JointCount(), 1u);
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build
ctest --test-dir build --output-on-failure -R "MjcfImporter|UrdfImporter"
```

Expected: parser symbols missing or fixture data not recognized.

**Step 3: Write minimal implementation**

```cpp
nuka::scene::SceneIR LoadMjcf(const std::string& path);
nuka::scene::SceneIR LoadUrdf(const std::string& path);
```

Implementation goal: parse one body tree, one revolute joint, and inertial records into canonical IR using `tinyxml2`.

**Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build
ctest --test-dir build --output-on-failure -R "MjcfImporter|UrdfImporter"
```

Expected: both importer tests pass for minimal fixtures.

**Step 5: Commit**

```bash
git add src/import tests/import tests/data src/CMakeLists.txt tests/CMakeLists.txt
git commit -m "import: add minimal MJCF and URDF compiler frontends"
```

### Task 7: USD Importer Skeleton

**Files:**
- Create: `src/import/usd_importer.hpp`
- Create: `src/import/usd_importer.cpp`
- Create: `tests/import/test_usd_importer.cpp`
- Create: `tests/data/minimal_scene.usda`
- Modify: `src/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

**Step 1: Write the failing test**

```cpp
#include <gtest/gtest.h>
#include "import/usd_importer.hpp"

TEST(UsdImporter, LoadsUsdPhysicsRigidBodyAndJoint) {
    const auto scene = nuka::import::LoadUsd("tests/data/minimal_scene.usda");
    EXPECT_EQ(scene.RigidBodyCount(), 2u);
    EXPECT_EQ(scene.JointCount(), 1u);
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build
ctest --test-dir build --output-on-failure -R UsdImporter
```

Expected: build fails until USD SDK integration and importer scaffolding are added.

**Step 3: Write minimal implementation**

```cpp
nuka::scene::SceneIR LoadUsd(const std::string& path);
```

Implementation goal: map a minimal `UsdPhysicsRigidBodyAPI`, collider, and joint into canonical IR. Defer materials, sensors, and extensions to later tasks.

**Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build
ctest --test-dir build --output-on-failure -R UsdImporter
```

Expected: `UsdImporter` passes against the minimal USDA fixture.

**Step 5: Commit**

```bash
git add src/import tests/import tests/data src/CMakeLists.txt tests/CMakeLists.txt
git commit -m "import: add minimal OpenUSD compiler frontend"
```

### Task 8: Runtime Containers

**Files:**
- Create: `src/runtime/world_template.hpp`
- Create: `src/runtime/world_instance.hpp`
- Create: `src/runtime/batch_context.hpp`
- Create: `src/runtime/world_builder.cpp`
- Create: `tests/runtime/test_world_build.cpp`
- Modify: `src/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

**Step 1: Write the failing test**

```cpp
#include <gtest/gtest.h>
#include "runtime/world_template.hpp"

TEST(WorldBuild, BuildsTemplateAndSingleInstanceFromBlob) {
    nuka::scene::SceneIR scene;
    scene.AddRigidBody("root");
    const auto blob = nuka::scene::CookScene(scene);
    auto world = nuka::runtime::BuildWorld(blob);
    EXPECT_EQ(world.template_view.body_count, 1u);
    EXPECT_EQ(world.batch.instance_count, 1u);
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build
ctest --test-dir build --output-on-failure -R WorldBuild
```

Expected: build fails because runtime containers do not exist.

**Step 3: Write minimal implementation**

```cpp
struct WorldTemplate { uint32_t body_count; };
struct WorldInstance { uint32_t body_count; };
struct BatchContext { uint32_t instance_count; };
```

**Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build
ctest --test-dir build --output-on-failure -R WorldBuild
```

Expected: `WorldBuild` passes.

**Step 5: Commit**

```bash
git add src/runtime tests/runtime src/CMakeLists.txt tests/CMakeLists.txt
git commit -m "runtime: add world template, instance, and batch containers"
```

### Task 9: Rigid State and Integration

**Files:**
- Create: `src/runtime/rigid/body_state.hpp`
- Create: `src/runtime/rigid/integrator.cu`
- Create: `src/runtime/rigid/integrator.hpp`
- Create: `tests/runtime/test_rigid_integration.cpp`
- Modify: `src/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

**Step 1: Write the failing test**

```cpp
#include <gtest/gtest.h>
#include "runtime/rigid/integrator.hpp"

TEST(RigidIntegration, GravityAdvancesVelocity) {
    nuka::runtime::rigid::BodyState body{};
    body.inv_mass = 1.0f;
    body.linear_velocity = {0.0f, 0.0f, 0.0f};
    nuka::runtime::rigid::IntegrateGravity(body, {0.0f, -9.81f, 0.0f}, 0.1f);
    EXPECT_NEAR(body.linear_velocity.y, -0.981f, 1e-4f);
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build
ctest --test-dir build --output-on-failure -R RigidIntegration
```

Expected: missing body state or integrator API.

**Step 3: Write minimal implementation**

```cpp
struct BodyState {
    float inv_mass;
    nuka::math::Vec3 linear_velocity;
};

void IntegrateGravity(BodyState& body, nuka::math::Vec3 gravity, float dt);
```

**Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build
ctest --test-dir build --output-on-failure -R RigidIntegration
```

Expected: `RigidIntegration` passes.

**Step 5: Commit**

```bash
git add src/runtime/rigid tests/runtime src/CMakeLists.txt tests/CMakeLists.txt
git commit -m "runtime: add first rigid-body state and integration path"
```

### Task 10: Geometry Queries and Broadphase

**Files:**
- Create: `src/collision/aabb.hpp`
- Create: `src/collision/static_bvh.hpp`
- Create: `src/collision/dynamic_broadphase.cu`
- Create: `src/collision/query.hpp`
- Create: `tests/collision/test_raycast.cpp`
- Create: `tests/collision/test_broadphase.cpp`
- Modify: `src/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

**Step 1: Write the failing test**

```cpp
#include <gtest/gtest.h>
#include "collision/query.hpp"

TEST(SceneQuery, RayHitsGroundPlane) {
    const auto hit = nuka::collision::RaycastPlane({0.0f, 1.0f, 0.0f}, {0.0f, -1.0f, 0.0f});
    ASSERT_TRUE(hit.has_value());
    EXPECT_NEAR(hit->distance, 1.0f, 1e-4f);
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build
ctest --test-dir build --output-on-failure -R "SceneQuery|Broadphase"
```

Expected: collision query interfaces are missing.

**Step 3: Write minimal implementation**

```cpp
struct RayHit { float distance; };
std::optional<RayHit> RaycastPlane(nuka::math::Vec3 origin, nuka::math::Vec3 direction);
```

Implementation goal: add CPU-checkable plane raycast plus the first static BVH and dynamic broadphase data paths.

**Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build
ctest --test-dir build --output-on-failure -R "SceneQuery|Broadphase"
```

Expected: query and broadphase tests pass.

**Step 5: Commit**

```bash
git add src/collision tests/collision src/CMakeLists.txt tests/CMakeLists.txt
git commit -m "collision: add first geometry queries and broadphase scaffolding"
```

### Task 11: Contact Manifolds and Constraint Blocks

**Files:**
- Create: `src/constraint/contact_manifold.hpp`
- Create: `src/constraint/constraint_block.hpp`
- Create: `src/constraint/contact_builder.cpp`
- Create: `tests/constraint/test_contact_manifold.cpp`
- Create: `tests/constraint/test_constraint_block.cpp`
- Modify: `src/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

**Step 1: Write the failing test**

```cpp
#include <gtest/gtest.h>
#include "constraint/constraint_block.hpp"

TEST(ConstraintBlock, ContactManifoldCreatesNormalAndFrictionRows) {
    const auto block = nuka::constraint::BuildContactBlock(2);
    EXPECT_EQ(block.row_count, 3u);
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build
ctest --test-dir build --output-on-failure -R "ConstraintBlock|ContactManifold"
```

Expected: missing manifold or constraint block implementation.

**Step 3: Write minimal implementation**

```cpp
struct ConstraintBlock {
    uint32_t row_count;
    uint32_t body_a;
    uint32_t body_b;
};

ConstraintBlock BuildContactBlock(uint32_t contact_points);
```

**Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build
ctest --test-dir build --output-on-failure -R "ConstraintBlock|ContactManifold"
```

Expected: both tests pass.

**Step 5: Commit**

```bash
git add src/constraint tests/constraint src/CMakeLists.txt tests/CMakeLists.txt
git commit -m "constraint: add manifold cache and block representation"
```

### Task 12: Iterative Rigid Solver

**Files:**
- Create: `src/solver/rigid_solver.hpp`
- Create: `src/solver/rigid_solver.cu`
- Create: `tests/solver/test_joint_limit.cpp`
- Create: `tests/solver/test_contact_resting.cpp`
- Modify: `src/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

**Step 1: Write the failing test**

```cpp
#include <gtest/gtest.h>
#include "solver/rigid_solver.hpp"

TEST(RigidSolver, RestingContactPreventsFurtherPenetration) {
    auto result = nuka::solver::SolveUnitGroundContact();
    EXPECT_LE(result.max_penetration, 1e-3f);
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build
ctest --test-dir build --output-on-failure -R "RigidSolver|JointLimit|ContactResting"
```

Expected: solver interfaces are missing.

**Step 3: Write minimal implementation**

```cpp
struct SolveResult {
    float max_penetration;
};

SolveResult SolveUnitGroundContact();
```

Implementation goal: solve one normal contact plus friction rows using a block iterative loop and return the penetration metric.

**Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build
ctest --test-dir build --output-on-failure -R "RigidSolver|JointLimit|ContactResting"
```

Expected: solver tests pass.

**Step 5: Commit**

```bash
git add src/solver tests/solver src/CMakeLists.txt tests/CMakeLists.txt
git commit -m "solver: add first iterative rigid constraint solver"
```

### Task 13: Articulation and Drives

**Files:**
- Create: `src/runtime/articulation/articulation_graph.hpp`
- Create: `src/runtime/articulation/joint_drive.hpp`
- Create: `src/runtime/articulation/joint_constraints.cpp`
- Create: `tests/runtime/test_revolute_joint.cpp`
- Create: `tests/runtime/test_joint_drive.cpp`
- Modify: `src/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

**Step 1: Write the failing test**

```cpp
#include <gtest/gtest.h>
#include "runtime/articulation/joint_drive.hpp"

TEST(JointDrive, VelocityDriveMovesTowardTarget) {
    auto result = nuka::runtime::articulation::SimulateVelocityDriveStep();
    EXPECT_GT(result.joint_velocity, 0.0f);
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build
ctest --test-dir build --output-on-failure -R "RevoluteJoint|JointDrive"
```

Expected: articulation symbols are missing.

**Step 3: Write minimal implementation**

```cpp
struct DriveStepResult {
    float joint_velocity;
};

DriveStepResult SimulateVelocityDriveStep();
```

Implementation goal: represent a revolute joint as a constraint block and apply a simple velocity target drive.

**Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build
ctest --test-dir build --output-on-failure -R "RevoluteJoint|JointDrive"
```

Expected: both articulation tests pass.

**Step 5: Commit**

```bash
git add src/runtime/articulation tests/runtime src/CMakeLists.txt tests/CMakeLists.txt
git commit -m "articulation: add maximal-coordinate joint graph and drive path"
```

### Task 14: Sensor Graph and Packets

**Files:**
- Create: `src/sensor/sensor_packet.hpp`
- Create: `src/sensor/sensor_graph.hpp`
- Create: `src/sensor/state_sensor.cpp`
- Create: `src/sensor/ray_sensor.cu`
- Create: `tests/sensor/test_imu_sensor.cpp`
- Create: `tests/sensor/test_lidar_sensor.cpp`
- Modify: `src/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

**Step 1: Write the failing test**

```cpp
#include <gtest/gtest.h>
#include "sensor/sensor_graph.hpp"

TEST(SensorGraph, ImuPacketContainsLinearAcceleration) {
    auto packet = nuka::sensor::BuildTestImuPacket();
    EXPECT_TRUE(packet.has_linear_acceleration);
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build
ctest --test-dir build --output-on-failure -R "SensorGraph|ImuSensor|LidarSensor"
```

Expected: sensor interfaces are missing.

**Step 3: Write minimal implementation**

```cpp
struct SensorPacket {
    bool has_linear_acceleration;
};

SensorPacket BuildTestImuPacket();
```

Implementation goal: add one state sensor packet path and one ray-based range sensor path using shared query kernels.

**Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build
ctest --test-dir build --output-on-failure -R "SensorGraph|ImuSensor|LidarSensor"
```

Expected: sensor tests pass.

**Step 5: Commit**

```bash
git add src/sensor tests/sensor src/CMakeLists.txt tests/CMakeLists.txt
git commit -m "sensor: add sensor graph, packets, and first state and ray sensors"
```

### Task 15: Debug Shell

**Files:**
- Create: `apps/debug_shell/main.cpp`
- Create: `apps/debug_shell/app.hpp`
- Create: `apps/debug_shell/app.cpp`
- Create: `apps/debug_shell/render_debug.cpp`
- Create: `tests/apps/test_debug_draw_list.cpp`
- Modify: `CMakeLists.txt`
- Modify: `src/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

**Step 1: Write the failing test**

```cpp
#include <gtest/gtest.h>
#include "apps/debug_shell/app.hpp"

TEST(DebugDraw, ProducesContactOverlayCommands) {
    const auto commands = nuka::app::BuildContactOverlayCommandCount(4);
    EXPECT_EQ(commands, 4u);
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build
ctest --test-dir build --output-on-failure -R DebugDraw
```

Expected: debug shell scaffolding does not exist.

**Step 3: Write minimal implementation**

```cpp
uint32_t BuildContactOverlayCommandCount(uint32_t contacts);
```

Implementation goal: create a native app target with a Vulkan production
renderer and ImGui controls that can render contact, AABB, and joint-frame
overlays. The headless PPM rasterizer remains only a CI/reference artifact path.

**Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build
ctest --test-dir build --output-on-failure -R DebugDraw
```

Expected: debug draw unit test passes and the shell app builds successfully.

**Step 5: Commit**

```bash
git add apps/debug_shell tests/apps CMakeLists.txt src/CMakeLists.txt tests/CMakeLists.txt
git commit -m "app: add native debug shell and overlay pipeline"
```

### Task 16: BatchContext Expansion Hooks

**Files:**
- Create: `src/runtime/batch_scheduler.hpp`
- Create: `src/runtime/batch_scheduler.cpp`
- Create: `tests/runtime/test_batch_context.cpp`
- Modify: `src/runtime/batch_context.hpp`
- Modify: `src/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

**Step 1: Write the failing test**

```cpp
#include <gtest/gtest.h>
#include "runtime/batch_context.hpp"

TEST(BatchContext, SupportsTemplateReuseAcrossInstances) {
    auto batch = nuka::runtime::BuildTestBatch(3);
    EXPECT_EQ(batch.instance_count, 3u);
    EXPECT_EQ(batch.template_count, 1u);
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build
ctest --test-dir build --output-on-failure -R BatchContext
```

Expected: current batch context only supports a single instance or missing scheduler types.

**Step 3: Write minimal implementation**

```cpp
struct BatchContext {
    uint32_t template_count;
    uint32_t instance_count;
};

BatchContext BuildTestBatch(uint32_t instance_count);
```

**Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build
ctest --test-dir build --output-on-failure -R BatchContext
```

Expected: `BatchContext` passes and the scheduler compiles.

**Step 5: Commit**

```bash
git add src/runtime tests/runtime src/CMakeLists.txt tests/CMakeLists.txt
git commit -m "runtime: add batched-instance scheduling extension hooks"
```

### Task 17: Regression Scenes and Reference Checks

**Files:**
- Create: `tests/regression/test_free_fall.cpp`
- Create: `tests/regression/test_rest_stack.cpp`
- Create: `tests/regression/test_two_link_arm.cpp`
- Create: `tests/perf/test_step_timing.cpp`
- Create: `tools/reference/compare_reference.py`
- Create: `docs/testing/p0-regression-matrix.md`
- Modify: `tests/CMakeLists.txt`

**Step 1: Write the failing test**

```cpp
#include <gtest/gtest.h>

TEST(FreeFallRegression, EnergyDriftStaysBounded) {
    const float drift = RunFreeFallRegression();
    EXPECT_LT(drift, 1e-2f);
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build
ctest --test-dir build --output-on-failure -R "Regression|FreeFall|RestStack|TwoLinkArm|StepTiming"
```

Expected: helper runners and baselines are missing.

**Step 3: Write minimal implementation**

```cpp
float RunFreeFallRegression();
```

Implementation goal: add three baseline scenes plus a Python helper that can compare exported CSV metrics against reference simulator runs when those references are available.

**Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build
ctest --test-dir build --output-on-failure -R "Regression|FreeFall|RestStack|TwoLinkArm|StepTiming"
python tools/reference/compare_reference.py --help
```

Expected: regression and perf tests pass, and the comparison tool prints CLI help.

**Step 5: Commit**

```bash
git add tests/regression tests/perf tools/reference docs/testing tests/CMakeLists.txt
git commit -m "test: add regression matrix, perf probes, and reference comparison tooling"
```

### Task 18: Documentation and CI Guardrails

**Files:**
- Create: `docs/architecture/runtime-overview.md`
- Create: `docs/architecture/scene-compiler.md`
- Create: `.github/workflows/build.yml`
- Create: `.github/workflows/test.yml`
- Create: `scripts/configure.ps1`
- Create: `scripts/run-tests.ps1`
- Modify: `README.md`

**Step 1: Write the failing test**

```text
No code test for this task. The failing condition is operational: CI files and operator docs do not exist, so the project cannot be reproduced consistently by a new engineer.
```

**Step 2: Run validation to verify the gap**

Run:

```bash
Get-ChildItem .github/workflows
Get-Content README.md
```

Expected: missing workflow files and missing operator instructions.

**Step 3: Write minimal implementation**

```yaml
name: build
on: [push, pull_request]
jobs:
  build:
    runs-on: windows-latest
```

Implementation goal: add CI entry points for configure, build, and targeted test execution, plus architecture docs for the runtime and scene compiler.

**Step 4: Run validation to verify it passes**

Run:

```bash
Get-ChildItem .github/workflows
Get-Content docs/architecture/runtime-overview.md
Get-Content scripts/run-tests.ps1
```

Expected: files exist and contain actionable instructions.

**Step 5: Commit**

```bash
git add .github docs scripts README.md
git commit -m "docs: add architecture docs and CI guardrails"
```

## Implementation Order Summary

1. Finish Tasks 1 through 5 before writing any solver kernels that depend on unstable interfaces.
2. Do not start Task 12 before Task 11 stabilizes the constraint-block representation.
3. Do not start Task 14 before Task 10 establishes query kernels that sensors can reuse.
4. Treat Task 16 as an architectural hook task, not as a full batched runtime implementation.
5. Defer any soft-body or fluid work until the full P0 regression matrix is stable.

## Exit Criteria for P0

P0 is complete only when all of the following are true:

1. A single-scene robot articulation with rigid contacts runs end-to-end.
2. Broadphase, narrowphase, contact manifolds, and the iterative rigid solver all pass regression tests.
3. At least one MJCF scene, one URDF scene, and one USD scene compile into canonical IR and cooked runtime blobs.
4. IMU-style state packets and ray-based depth or lidar packets are generated from the same runtime state.
5. The native debug shell can visualize bodies, joints, contacts, and rays.
6. BatchContext supports template reuse and multi-instance metadata, even if full batched solving remains deferred.
7. CI can configure, build, and run the essential test matrix reproducibly.
