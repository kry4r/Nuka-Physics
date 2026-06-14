// ---------------------------------------------------------------------------
// H1 cup demo kitchen scene — the FULL .nks pipeline gate (M7 T5b).
// ---------------------------------------------------------------------------
// Compose the imported kitchen MJCF scene, the H1 MJCF, and the USD cup into one
// SceneIR (the cup support is a REAL imported kitchen counter collision top — no
// programmatic table scaffold), seat the cup on the counter with the SHARED
// cook::FindRestPlacement (T2) drop-to-rest primitive, then route the composed
// scene through the WHOLE own-format pipeline:
//
//     Compose -> Save(.nks + .nka) -> [re-Save byte-identity] -> Load
//             -> CookToModel -> nk::World -> Step.
//
// The gate asserts the scene authors, persists (byte-identically on re-save),
// reloads (preserving the cook counts + the cup-on-counter rest placement),
// cooks to an nk::Model with the right capacities, and steps without error.
//
// KITCHEN-SCALE COOK GAP (flagged, not silently dropped): the 412 KB kitchen
// MJCF composes to ~374 bodies / ~1700 shapes / ~113 joints. The generic
// CookToModel takes only the FIRST cooked articulation (cook_to_model.cpp: "the
// FIRST articulation, the single-robot scene shape"), so the whole-body 51-DOF
// H1 is NOT cooked as one coherent articulation here (it cooks to a small
// articulation + hundreds of free rigid bodies). The cook+World+Step is
// therefore scoped to "builds + steps a few frames + capacities mirror the
// records"; the full H1-under-kitchen union grasp world is the dedicated
// h1_cup_table.nks union path (M7 T4 union_cook / h1_grasp_lift), NOT this
// kitchen-compose gate. This gate proves the GENERIC .nks pipeline end-to-end on
// a real composed scene; the per-cell grasp physics live in the union gate.
// ---------------------------------------------------------------------------

#include "import/mjcf_importer.hpp"
#include "import/usd_importer.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/pipeline/world.hpp"
#include "phi/backend.hpp"
#include "phi/scoped_device_guard.hpp"
#include "scene/canonical_types.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/cook/placement.hpp"
#include "scene/cooked_blob.hpp"
#include "scene/cooker.hpp"
#include "scene/format/json.hpp"
#include "scene/format/nks.hpp"
#include "scene/scene_compose.hpp"
#include "scene/scene_ir.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;
using namespace nuka::scene;
namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace cook = nuka::scene::cook;
namespace fs = std::filesystem;

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
    Transform cup_placement;     // the cup ROOT world transform after rest-seat.
    SupportSurface support;      // resolved against the COMPOSED scene.
    ShapeId composed_support_shape = kInvalidShape;
    BodyId cup_body = kInvalidBody;
    float cup_root_relative_min_z = 0.0f;
};

// Compose kitchen + H1 + cup, then SEAT the cup on the imported counter top with
// the shared cook::FindRestPlacement (T2) drop-to-rest. The cup is composed in
// parked, then its record local_transform is set to the rest placement (a ROOT
// body, so the placement IS its world transform). The H1 hand is placed within
// reach of the seated cup. Deterministic (pure geometry; no RNG).
Composed ComposeAll() {
    Composed out;

    const SceneIR kitchen = LoadKitchenScene();
    const SceneIR h1 = LoadH1ClearedMesh();
    const SceneIR cup = LoadCupSingleHull();

    // Resolve the counter top + the cup geometry on the standalone scenes (so the
    // reach/placement targets are known before composing).
    const SupportSurface kitchen_support = FindKitchenCounterSupport(kitchen);
    out.cup_root_relative_min_z = CupRootRelativeMinZ(cup);

    // XY target: the front-of-counter offset (same as the legacy gate).
    const float front_offset =
        std::min(0.18f, std::max(0.0f, kitchen_support.half_extents.y - 0.08f));
    const Vec3 cup_xy =
        kitchen_support.world.TransformPoint({0.0f, -front_offset, 0.0f});

    // Hand target above the cup seat (z resolved from the counter top + cup geom).
    const float cup_root_z =
        kitchen_support.top_z - out.cup_root_relative_min_z + kCupClearance;
    const Vec3 hand_target{cup_xy.x, cup_xy.y, cup_root_z + 0.12f};
    out.h1_placement = MakeH1Placement(h1, hand_target);

    // Compose: kitchen base + H1 (placed) + cup (parked; seated below).
    SceneIR scene = kitchen;
    scene = Compose(scene, h1, out.h1_placement, kH1Prefix);
    scene = Compose(scene, cup, Transform::Identity(), kCupPrefix);

    // Resolve the counter support + the cup body against the COMPOSED scene.
    out.support = FindKitchenCounterSupport(scene);
    out.composed_support_shape = out.support.shape_id;
    const int cup_body = FindBody(scene, kCupPrefix + kCupRootBody);
    if (cup_body < 0) {
        throw std::runtime_error("composed scene missing cup root " + kCupRootBody);
    }
    out.cup_body = static_cast<BodyId>(cup_body);

    // SEAT the cup with the shared rest-placement primitive (replaces the legacy
    // in-file drop-to-rest): the cup bottom rests at counter_top + clearance, at
    // the front-of-counter XY target. The cup is a ROOT body so the returned root
    // transform is applied directly as its record local_transform.
    out.cup_placement = cook::FindRestPlacement(
        scene, out.cup_body, out.composed_support_shape, cup_xy, kCupClearance);
    scene.GetBodyMut(out.cup_body).local_transform = out.cup_placement;

    out.scene = std::move(scene);
    return out;
}

// A fresh temp dir per run (hermetic). Mirrors scene_roundtrip's TempDir.
class TempDir {
public:
    TempDir() {
        char tmpl[] = "/tmp/nks_compose_h1_cup_XXXXXX";
        char* p = ::mkdtemp(tmpl);
        if (!p) throw std::runtime_error("mkdtemp failed");
        path_ = p;
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    // A fresh sub-dir (created) so the SAME scene saved twice has an identical
    // embedded sibling .nka BASENAME (byte-identity is determinism, not path).
    fs::path SubFile(const std::string& subdir, const std::string& name) const {
        const fs::path d = path_ / subdir;
        std::error_code ec;
        fs::create_directories(d, ec);
        return d / name;
    }

private:
    fs::path path_;
};

std::vector<uint8_t> ReadBytes(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)),
                                std::istreambuf_iterator<char>());
}

// The cup root world transform + its bottom z, recomputed from a (loaded) scene.
struct CupSeat {
    Transform root_world = Transform::Identity();
    float bottom_z = 0.0f;
};
CupSeat CupSeatOf(const SceneIR& scene, float cup_root_relative_min_z) {
    const int cup_body = FindBody(scene, kCupPrefix + kCupRootBody);
    CupSeat seat;
    if (cup_body < 0) return seat;
    seat.root_world = WorldOf(scene, cup_body);
    seat.bottom_z =
        seat.root_world.TransformPoint({0.0f, 0.0f, cup_root_relative_min_z}).z;
    return seat;
}

// --- shared CUDA backend (outlives the World dtors; freed at process exit) -----
struct Backend {
    nphi::Device* dev = nullptr;
    nphi::Backend* backend = nullptr;
};
Backend GetBackend() {
    static Backend b = [] {
        Backend r;
        // BUF-14: InitBestDevice() sets the active CUDA device for this process.
        r.dev = nphi::InitBestDevice();
        if (r.dev) r.backend = nphi::DeviceInitBackend(r.dev, nullptr);
        return r;
    }();
    return b;
}

}  // namespace

// ===========================================================================
// (1) SceneIR compose — counts add across the three imported sources.
// ===========================================================================
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

// ===========================================================================
// (2) Rest placement — the cup is seated on the imported counter top with the
//     SHARED cook::FindRestPlacement (T2) primitive (replaces the in-file
//     drop-to-rest): cup bottom rests at counter_top + clearance, within bounds.
// ===========================================================================
TEST(SceneComposeH1CupKitchen, CupRestsOnKitchenCounterTop) {
    if (!AssetsAvailable()) GTEST_SKIP() << "kitchen / H1 / cup assets not present";

    const Composed composed = ComposeAll();
    const SceneIR& out = composed.scene;

    const int cup_body = FindBody(out, kCupPrefix + kCupRootBody);
    ASSERT_GE(cup_body, 0) << "composed scene must contain " << kCupPrefix + kCupRootBody;

    const CupSeat seat = CupSeatOf(out, composed.cup_root_relative_min_z);
    const Vec3 cup_in_support =
        composed.support.world.Inverse().TransformPoint(seat.root_world.position);

    EXPECT_NEAR(seat.bottom_z, composed.support.top_z + kCupClearance, 1e-4f)
        << "cup bottom z=" << seat.bottom_z
        << " kitchen counter top z=" << composed.support.top_z;
    EXPECT_GE(seat.bottom_z, composed.support.top_z - 1e-4f)
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

// ===========================================================================
// (3) THE .nks PIPELINE — Save(.nks+.nka) -> re-Save byte-identity -> Load.
//     The composed+seated scene authors, persists byte-identically on a second
//     save, and reloads preserving the cook counts + the cup rest placement.
// ===========================================================================
TEST(SceneComposeH1CupKitchen, NksRoundTripsByteIdenticalAndPreservesCounts) {
    if (!AssetsAvailable()) GTEST_SKIP() << "kitchen / H1 / cup assets not present";

    const Composed composed = ComposeAll();
    const SceneIR& orig = composed.scene;
    TempDir tmp;

    // Same stem in two sub-dirs so the embedded .nka basename matches across the
    // two saves (so byte-identity is a pure determinism check).
    const fs::path nks1 = tmp.SubFile("run1", "h1_cup_kitchen.nks");
    nks::Save(orig, nks1.string());

    // Format-shape pin: the saved JSON carries the §3.7 nested "tree" + split
    // material sections (no legacy flat record arrays).
    {
        std::ifstream in(nks1, std::ios::binary);
        const std::string text((std::istreambuf_iterator<char>(in)),
                               std::istreambuf_iterator<char>());
        const json::Value root = json::Value::Parse(text);
        EXPECT_TRUE(root.Has("tree")) << "saved .nks lacks the \"tree\" section";
        EXPECT_TRUE(root.Has("physics_materials"));
        EXPECT_FALSE(root.Has("bodies")) << "flat \"bodies\" array must be gone";
    }

    const SceneIR loaded = nks::Load(nks1.string());

    // Re-save the reloaded scene: byte-identical .nks + .nka (determinism).
    const fs::path nks2 = tmp.SubFile("run2", "h1_cup_kitchen.nks");
    nks::Save(loaded, nks2.string());
    EXPECT_EQ(ReadBytes(nks1), ReadBytes(nks2)) << ".nks not byte-identical on re-save";
    EXPECT_EQ(ReadBytes(tmp.SubFile("run1", "h1_cup_kitchen.nka")),
              ReadBytes(tmp.SubFile("run2", "h1_cup_kitchen.nka")))
        << ".nka not byte-identical on re-save";

    // The reload preserves the record counts and the cooked counts.
    EXPECT_EQ(orig.RigidBodyCount(), loaded.RigidBodyCount());
    EXPECT_EQ(orig.JointCount(), loaded.JointCount());
    EXPECT_EQ(orig.ShapeCount(), loaded.ShapeCount());
    EXPECT_EQ(orig.MaterialCount(), loaded.MaterialCount());
    EXPECT_EQ(orig.ActuatorCount(), loaded.ActuatorCount());

    const CookedBlob co = CookScene(orig);
    const CookedBlob cl = CookScene(loaded);
    EXPECT_EQ(co.body_count, cl.body_count);
    EXPECT_EQ(co.joint_count, cl.joint_count);
    EXPECT_EQ(co.shape_count, cl.shape_count);
    EXPECT_EQ(co.convex_geometry.Count(), cl.convex_geometry.Count());

    // The cup rest placement survives the .nks round-trip (the seated cup bottom
    // is still at counter_top + clearance after Save -> Load).
    const SupportSurface loaded_support = FindKitchenCounterSupport(loaded);
    const CupSeat seat = CupSeatOf(loaded, composed.cup_root_relative_min_z);
    EXPECT_NEAR(seat.bottom_z, loaded_support.top_z + kCupClearance, 1e-4f)
        << "cup rest placement not preserved through the .nks round-trip";
}

// ===========================================================================
// (4) COOK + STEP — Load(.nks) -> CookToModel -> nk::World -> Step. Scoped to
//     "builds + steps + capacities mirror the records" (see the KITCHEN-SCALE
//     COOK GAP banner at the top of this file). The full H1-under-kitchen union
//     grasp world is the dedicated h1_cup_table.nks union path, NOT this gate.
// ===========================================================================
TEST(SceneComposeH1CupKitchen, CooksToWorldAndStepsThroughNks) {
    if (!AssetsAvailable()) GTEST_SKIP() << "kitchen / H1 / cup assets not present";
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";

    const Composed composed = ComposeAll();
    TempDir tmp;
    const fs::path nks = tmp.SubFile("cook", "h1_cup_kitchen.nks");
    nks::Save(composed.scene, nks.string());

    // The cook+step runs on the RELOADED scene (the whole .nks pipeline).
    const SceneIR loaded = nks::Load(nks.string());
    cook::CookToModelResult cooked = cook::CookToModel(loaded, /*env_count=*/1);
    nk::Model& model = cooked.model;
    const nk::ModelCapacities& cap = model.capacities;

    // Capacities mirror the records: one body row per rigid body.
    EXPECT_EQ(cap.bodies_per_env, static_cast<uint32_t>(loaded.RigidBodyCount()));
    EXPECT_GT(cap.bodies_per_env, 0u);

    // KITCHEN-SCALE COOK GAP — FLAGGED (not silently dropped): the generic cook
    // takes only the FIRST cooked articulation, so the whole-body 51-DOF H1 is
    // NOT one coherent articulation here. Surface the cooked vs expected DOFs so
    // the gap is visible in the gate log (the union h1_grasp_lift gate owns the
    // real 51-DOF grasp world). NON-FATAL: this gate proves the GENERIC pipeline.
    std::printf("[compose-cook GAP] cooked articulation links=%u dofs=%u (the H1 "
                "alone is 51-DOF); movable free body_init rows seeded under the "
                "single-articulation cook=%zu of %u bodies. The full H1-under-"
                "kitchen union grasp world is the dedicated h1_cup_table.nks union "
                "path, not this kitchen-compose gate.\n",
                cap.links_per_env, cap.dofs_per_env,
                [&] {
                    size_t n = 0;
                    for (const auto& bi : model.body_init)
                        if (bi.inv_mass > 0.0f) ++n;
                    return n;
                }(),
                cap.bodies_per_env);

    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[0] = 0.0f;
    cfg.gravity[1] = 0.0f;
    cfg.gravity[2] = -9.81f;

    Backend b = GetBackend();
    nk::World world(std::move(model), 1u, b.dev, b.backend, cfg);
    ASSERT_TRUE(world.Ready()) << "the composed kitchen+H1+cup scene did not cook "
                                  "to a Ready nk::World";

    // Step a few frames: the cooked world steps without a device error.
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(world.Step().AllOk())
            << "Step " << i << " failed on the composed-kitchen world";
    }
}

// ===========================================================================
// (5) Determinism — two composes produce identical records + cook counts.
// ===========================================================================
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
