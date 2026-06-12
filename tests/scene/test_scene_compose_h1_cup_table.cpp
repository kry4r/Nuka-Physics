// ---------------------------------------------------------------------------
// H1 cup demo kitchen scene compose gate.
// ---------------------------------------------------------------------------
// Compose the imported kitchen MJCF scene, the H1 MJCF, and the USD cup into one
// SceneIR. The cup support is a real imported kitchen counter collision top; no
// programmatic/simple table scaffold is allowed in this gate.
// ---------------------------------------------------------------------------

#include "import/mjcf_importer.hpp"
#include "import/usd_importer.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "scene/canonical_types.hpp"
#include "scene/cooked_blob.hpp"
#include "scene/cooker.hpp"
#include "scene/scene_compose.hpp"
#include "scene/scene_ir.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;
using namespace nuka::scene;

namespace {

const std::string kKitchenMjcf = ".nuka-assets/kitchen/mjcf/kitchen.xml";
const std::string kH1Mjcf =
    ".nuka-assets/newton_assets/unitree_h1/mjcf/h1_with_hand.xml";
const std::string kCupLargeUsd =
    ".nuka-assets/newton_assets/manipulation_objects/cup/model_large.usda";

constexpr float kCupClearance = 0.001f;

const std::string kH1Prefix = "h1/";
const std::string kCupPrefix = "cup/";
const std::string kRightHandLink = "right_hand_link";
const std::string kCupRootBody = "Model_body";
const std::string kPreferredSupportShape = "counter_1_main_group_top_1";
const std::string kPreferredSupportBody = "counter_1_main_group_main";

bool AssetsAvailable() {
    return std::filesystem::exists(kKitchenMjcf) &&
           std::filesystem::exists(kH1Mjcf) &&
           std::filesystem::exists(kCupLargeUsd);
}

bool Contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

int FindBody(const SceneIR& scene, const std::string& name) {
    for (size_t i = 0; i < scene.RigidBodyCount(); ++i) {
        if (scene.GetBody(static_cast<BodyId>(i)).name == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

int FindShape(const SceneIR& scene, const std::string& name) {
    for (size_t i = 0; i < scene.ShapeCount(); ++i) {
        if (scene.GetShape(static_cast<ShapeId>(i)).name == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

Transform WorldOf(const SceneIR& scene, int body) {
    std::vector<int> chain;
    int current = body;
    while (current >= 0) {
        chain.push_back(current);
        const BodyId parent = scene.GetBody(static_cast<BodyId>(current)).parent_id;
        current = (parent == kInvalidBody) ? -1 : static_cast<int>(parent);
    }

    Transform world = Transform::Identity();
    for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
        world = world * scene.GetBody(static_cast<BodyId>(*it)).local_transform;
    }
    return world;
}

SceneIR LoadKitchenScene() {
    SceneIR scene = nuka::import::LoadMjcf(kKitchenMjcf);
    for (size_t i = 0; i < scene.ShapeCount(); ++i) {
        auto& shape = scene.GetShapeMut(static_cast<ShapeId>(i));
        shape.mesh_vertices.clear();
        shape.mesh_indices.clear();
    }
    return scene;
}

SceneIR LoadH1ClearedMesh() {
    SceneIR scene = nuka::import::LoadMjcf(kH1Mjcf);
    for (size_t i = 0; i < scene.ShapeCount(); ++i) {
        auto& shape = scene.GetShapeMut(static_cast<ShapeId>(i));
        shape.mesh_vertices.clear();
        shape.mesh_indices.clear();
    }
    return scene;
}

SceneIR LoadCupSingleHull() {
    SceneIR scene = nuka::import::LoadUsd(kCupLargeUsd);
    for (size_t i = 0; i < scene.ShapeCount(); ++i) {
        auto& shape = scene.GetShapeMut(static_cast<ShapeId>(i));
        if (!shape.mesh_vertices.empty()) {
            shape.decompose_mode = DecomposeMode::Skip;
        }
    }
    return scene;
}

float CupRootRelativeMinZ(const SceneIR& cup) {
    const int root = FindBody(cup, kCupRootBody);
    if (root < 0) {
        throw std::runtime_error("cup root body not found: " + kCupRootBody);
    }

    const Transform root_inv = WorldOf(cup, root).Inverse();
    float min_z = std::numeric_limits<float>::infinity();
    for (size_t shape_i = 0; shape_i < cup.ShapeCount(); ++shape_i) {
        const auto& shape = cup.GetShape(static_cast<ShapeId>(shape_i));
        if (shape.body_id == kInvalidBody || shape.mesh_vertices.empty()) {
            continue;
        }
        const Transform shape_world =
            WorldOf(cup, static_cast<int>(shape.body_id)) * shape.local_transform;
        for (size_t vertex = 0; vertex + 2 < shape.mesh_vertices.size(); vertex += 3) {
            const Vec3 local{shape.mesh_vertices[vertex + 0],
                             shape.mesh_vertices[vertex + 1],
                             shape.mesh_vertices[vertex + 2]};
            const Vec3 root_local = root_inv.TransformPoint(shape_world.TransformPoint(local));
            min_z = std::min(min_z, root_local.z);
        }
    }

    if (!std::isfinite(min_z)) {
        throw std::runtime_error("cup mesh has no vertices for placement");
    }
    return min_z;
}

struct SupportSurface {
    ShapeId shape_id = kInvalidShape;
    BodyId body_id = kInvalidBody;
    std::string body_name;
    std::string shape_name;
    Transform world = Transform::Identity();
    Vec3 half_extents = {};
    Vec3 center = {};
    float top_z = 0.0f;
    float area = 0.0f;
};

std::array<Vec3, 8> BoxCorners(Vec3 half_extents) {
    const float x = half_extents.x;
    const float y = half_extents.y;
    const float z = half_extents.z;
    return {{{-x, -y, -z}, { x, -y, -z}, { x,  y, -z}, {-x,  y, -z},
             {-x, -y,  z}, { x, -y,  z}, { x,  y,  z}, {-x,  y,  z}}};
}

SupportSurface BuildSupportSurface(const SceneIR& scene, ShapeId shape_id) {
    const auto& shape = scene.GetShape(shape_id);
    if (shape.type != ShapeType::Box || shape.body_id == kInvalidBody) {
        throw std::runtime_error("support shape must be a box with a body");
    }

    SupportSurface surface;
    surface.shape_id = shape_id;
    surface.body_id = shape.body_id;
    surface.body_name = scene.GetBody(shape.body_id).name;
    surface.shape_name = shape.name;
    surface.world = WorldOf(scene, static_cast<int>(shape.body_id)) * shape.local_transform;
    surface.half_extents = shape.half_extents;
    surface.center = surface.world.TransformPoint(Vec3::Zero());
    surface.area = 4.0f * shape.half_extents.x * shape.half_extents.y;
    surface.top_z = -std::numeric_limits<float>::infinity();
    for (const Vec3& corner : BoxCorners(shape.half_extents)) {
        surface.top_z = std::max(surface.top_z, surface.world.TransformPoint(corner).z);
    }
    return surface;
}

bool IsKitchenCounterSupportCandidate(const SceneIR& scene, ShapeId shape_id) {
    const auto& shape = scene.GetShape(shape_id);
    if (shape.type != ShapeType::Box || shape.body_id == kInvalidBody) {
        return false;
    }
    if (shape.contype == 0u || shape.conaffinity == 0u) {
        return false;
    }
    if (!Contains(shape.name, "counter") || !Contains(shape.name, "top") ||
        Contains(shape.name, "visual")) {
        return false;
    }
    const SupportSurface surface = BuildSupportSurface(scene, shape_id);
    return surface.area > 0.05f && surface.top_z > 0.70f && surface.top_z < 1.20f;
}

SupportSurface FindKitchenCounterSupport(const SceneIR& kitchen) {
    const int preferred = FindShape(kitchen, kPreferredSupportShape);
    if (preferred >= 0 &&
        IsKitchenCounterSupportCandidate(kitchen, static_cast<ShapeId>(preferred))) {
        return BuildSupportSurface(kitchen, static_cast<ShapeId>(preferred));
    }

    bool found = false;
    SupportSurface best;
    for (size_t i = 0; i < kitchen.ShapeCount(); ++i) {
        const auto shape_id = static_cast<ShapeId>(i);
        if (!IsKitchenCounterSupportCandidate(kitchen, shape_id)) {
            continue;
        }
        const SupportSurface candidate = BuildSupportSurface(kitchen, shape_id);
        if (!found || candidate.area > best.area ||
            (candidate.area == best.area && candidate.shape_name < best.shape_name)) {
            best = candidate;
            found = true;
        }
    }
    if (!found) {
        throw std::runtime_error("no imported kitchen counter support shape found");
    }
    return best;
}

Transform MakeH1Placement(const SceneIR& h1, const Vec3& hand_target) {
    const int right_hand = FindBody(h1, kRightHandLink);
    if (right_hand < 0) {
        throw std::runtime_error("H1 body not found: " + kRightHandLink);
    }
    const Vec3 rest_hand = WorldOf(h1, right_hand).position;
    return Transform{hand_target - rest_hand, Quat::Identity()};
}

struct Composed {
    SceneIR scene;
    Transform h1_placement;
    Transform cup_placement;
    SupportSurface support;
    float cup_root_relative_min_z = 0.0f;
};

Composed ComposeAll() {
    Composed out;

    const SceneIR kitchen = LoadKitchenScene();
    const SceneIR h1 = LoadH1ClearedMesh();
    const SceneIR cup = LoadCupSingleHull();

    out.support = FindKitchenCounterSupport(kitchen);
    out.cup_root_relative_min_z = CupRootRelativeMinZ(cup);

    const float front_offset =
        std::min(0.18f, std::max(0.0f, out.support.half_extents.y - 0.08f));
    const Vec3 cup_xy = out.support.world.TransformPoint({0.0f, -front_offset, 0.0f});
    out.cup_placement = Transform{
        Vec3{cup_xy.x, cup_xy.y,
             out.support.top_z - out.cup_root_relative_min_z + kCupClearance},
        Quat::Identity()};

    const Vec3 cup_root = out.cup_placement.position;
    const Vec3 hand_target{cup_root.x, cup_root.y, cup_root.z + 0.12f};
    out.h1_placement = MakeH1Placement(h1, hand_target);

    SceneIR scene = kitchen;
    scene = Compose(scene, h1, out.h1_placement, kH1Prefix);
    scene = Compose(scene, cup, out.cup_placement, kCupPrefix);
    out.scene = std::move(scene);
    return out;
}

}  // namespace

TEST(SceneComposeH1CupKitchen, CountsAddAcrossThreeSources) {
    if (!AssetsAvailable()) GTEST_SKIP() << "kitchen / H1 / cup assets not present";

    const SceneIR kitchen = LoadKitchenScene();
    const SceneIR h1 = LoadH1ClearedMesh();
    const SceneIR cup = LoadCupSingleHull();
    const Composed composed = ComposeAll();
    const SceneIR& out = composed.scene;

    EXPECT_EQ(out.RigidBodyCount(),
              kitchen.RigidBodyCount() + h1.RigidBodyCount() + cup.RigidBodyCount());
    EXPECT_EQ(out.JointCount(), kitchen.JointCount() + h1.JointCount() + cup.JointCount());
    EXPECT_EQ(out.ShapeCount(), kitchen.ShapeCount() + h1.ShapeCount() + cup.ShapeCount());
    EXPECT_EQ(out.MaterialCount(),
              kitchen.MaterialCount() + h1.MaterialCount() + cup.MaterialCount());
    EXPECT_EQ(out.SensorCount(),
              kitchen.SensorCount() + h1.SensorCount() + cup.SensorCount());
    EXPECT_EQ(out.CameraCount(),
              kitchen.CameraCount() + h1.CameraCount() + cup.CameraCount());
    EXPECT_EQ(out.LightCount(), kitchen.LightCount() + h1.LightCount() + cup.LightCount());
    EXPECT_EQ(out.ActuatorCount(),
              kitchen.ActuatorCount() + h1.ActuatorCount() + cup.ActuatorCount());
}

TEST(SceneComposeH1CupKitchen, UsesImportedKitchenCounterSupport) {
    if (!AssetsAvailable()) GTEST_SKIP() << "kitchen / H1 / cup assets not present";

    const Composed composed = ComposeAll();
    const SceneIR& out = composed.scene;

    EXPECT_GE(FindBody(out, kPreferredSupportBody), 0)
        << "composed scene must contain the real kitchen counter support";
    EXPECT_GE(FindShape(out, composed.support.shape_name), 0)
        << "composed scene must contain the selected imported counter top shape";
    EXPECT_EQ(composed.support.shape_name, kPreferredSupportShape);
    EXPECT_LT(FindBody(out, "table"), 0)
        << "H1 cup demo must not use the old programmatic simple table scaffold";
    EXPECT_NE(composed.support.shape_name, "table_top");
}

TEST(SceneComposeH1CupKitchen, ComposedSceneCooksWithMirroredCounts) {
    if (!AssetsAvailable()) GTEST_SKIP() << "kitchen / H1 / cup assets not present";

    const Composed composed = ComposeAll();
    const SceneIR& out = composed.scene;

    const CookedBlob blob = CookScene(out);

    EXPECT_EQ(blob.body_count, static_cast<uint32_t>(out.RigidBodyCount()));
    EXPECT_EQ(blob.joint_count, static_cast<uint32_t>(out.JointCount()));
    EXPECT_EQ(blob.shape_count, static_cast<uint32_t>(out.ShapeCount()));
    EXPECT_EQ(blob.material_count, static_cast<uint32_t>(out.MaterialCount()));
    EXPECT_EQ(blob.sensor_count, static_cast<uint32_t>(out.SensorCount()));
    EXPECT_EQ(blob.camera_count, static_cast<uint32_t>(out.CameraCount()));
    EXPECT_EQ(blob.light_count, static_cast<uint32_t>(out.LightCount()));
    EXPECT_EQ(blob.actuator_count, static_cast<uint32_t>(out.ActuatorCount()));
    EXPECT_GE(blob.convex_geometry.Count(), 1u);
}

TEST(SceneComposeH1CupKitchen, CupRestsOnKitchenCounterTop) {
    if (!AssetsAvailable()) GTEST_SKIP() << "kitchen / H1 / cup assets not present";

    const Composed composed = ComposeAll();
    const SceneIR& out = composed.scene;

    const int cup_body = FindBody(out, kCupPrefix + kCupRootBody);
    ASSERT_GE(cup_body, 0) << "composed scene must contain " << kCupPrefix + kCupRootBody;

    const Transform cup_world = WorldOf(out, cup_body);
    const Vec3 cup_bottom =
        cup_world.TransformPoint({0.0f, 0.0f, composed.cup_root_relative_min_z});
    const Vec3 cup_in_support =
        composed.support.world.Inverse().TransformPoint(cup_world.position);

    EXPECT_NEAR(cup_bottom.z, composed.support.top_z + kCupClearance, 1e-4f)
        << "cup bottom z=" << cup_bottom.z
        << " kitchen counter top z=" << composed.support.top_z;
    EXPECT_GE(cup_bottom.z, composed.support.top_z - 1e-4f)
        << "cup must not penetrate the kitchen counter";
    EXPECT_LE(std::fabs(cup_in_support.x), composed.support.half_extents.x - 0.02f);
    EXPECT_LE(std::fabs(cup_in_support.y), composed.support.half_extents.y - 0.02f);
}

TEST(SceneComposeH1CupKitchen, CupWithinRightHandReachBand) {
    if (!AssetsAvailable()) GTEST_SKIP() << "kitchen / H1 / cup assets not present";

    const Composed composed = ComposeAll();
    const SceneIR& out = composed.scene;

    for (const std::string& name : {std::string("right_shoulder_pitch_link"),
                                    std::string("right_elbow_link"),
                                    kRightHandLink}) {
        EXPECT_GE(FindBody(out, kH1Prefix + name), 0)
            << "composed scene must contain " << kH1Prefix + name;
    }

    const int hand = FindBody(out, kH1Prefix + kRightHandLink);
    const int cup_body = FindBody(out, kCupPrefix + kCupRootBody);
    ASSERT_GE(hand, 0);
    ASSERT_GE(cup_body, 0);

    const Vec3 hand_world = WorldOf(out, hand).position;
    const Vec3 cup_world = WorldOf(out, cup_body).position;
    const float dist = (hand_world - cup_world).Length();

    EXPECT_LT(dist, 0.5f) << "right hand must be within a plausible reach of the cup"
                          << " (dist=" << dist << " m)";
    EXPECT_GT(dist, 0.0f);
}

TEST(SceneComposeH1CupKitchen, DeterministicAcrossRuns) {
    if (!AssetsAvailable()) GTEST_SKIP() << "kitchen / H1 / cup assets not present";

    const Composed a = ComposeAll();
    const Composed b = ComposeAll();
    const SceneIR& sa = a.scene;
    const SceneIR& sb = b.scene;

    ASSERT_EQ(sa.RigidBodyCount(), sb.RigidBodyCount());
    ASSERT_EQ(sa.JointCount(), sb.JointCount());
    ASSERT_EQ(sa.ShapeCount(), sb.ShapeCount());
    ASSERT_EQ(sa.MaterialCount(), sb.MaterialCount());
    ASSERT_EQ(sa.ActuatorCount(), sb.ActuatorCount());
    EXPECT_EQ(a.support.shape_name, b.support.shape_name);
    EXPECT_EQ(a.support.body_name, b.support.body_name);
    EXPECT_EQ(a.support.top_z, b.support.top_z);

    for (size_t i = 0; i < sa.RigidBodyCount(); ++i) {
        const RigidBodyRecord& ra = sa.GetBody(static_cast<BodyId>(i));
        const RigidBodyRecord& rb = sb.GetBody(static_cast<BodyId>(i));
        EXPECT_EQ(ra.name, rb.name);
        EXPECT_EQ(ra.parent_id, rb.parent_id);
        EXPECT_EQ(ra.local_transform.position.x, rb.local_transform.position.x);
        EXPECT_EQ(ra.local_transform.position.y, rb.local_transform.position.y);
        EXPECT_EQ(ra.local_transform.position.z, rb.local_transform.position.z);
        EXPECT_EQ(ra.local_transform.rotation.w, rb.local_transform.rotation.w);
        EXPECT_EQ(ra.local_transform.rotation.x, rb.local_transform.rotation.x);
        EXPECT_EQ(ra.local_transform.rotation.y, rb.local_transform.rotation.y);
        EXPECT_EQ(ra.local_transform.rotation.z, rb.local_transform.rotation.z);
    }

    for (size_t i = 0; i < sa.JointCount(); ++i) {
        EXPECT_EQ(sa.GetJoint(static_cast<JointId>(i)).parent_body,
                  sb.GetJoint(static_cast<JointId>(i)).parent_body);
        EXPECT_EQ(sa.GetJoint(static_cast<JointId>(i)).child_body,
                  sb.GetJoint(static_cast<JointId>(i)).child_body);
    }

    const CookedBlob ba = CookScene(sa);
    const CookedBlob bb = CookScene(sb);
    EXPECT_EQ(ba.body_count, bb.body_count);
    EXPECT_EQ(ba.joint_count, bb.joint_count);
    EXPECT_EQ(ba.shape_count, bb.shape_count);
    EXPECT_EQ(ba.material_count, bb.material_count);
    EXPECT_EQ(ba.convex_geometry.Count(), bb.convex_geometry.Count());
}
