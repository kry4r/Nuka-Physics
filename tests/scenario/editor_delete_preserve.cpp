// ---------------------------------------------------------------------------
// Editor delete-preservation gate.
//
// The data-destruction trap this pins down: deleting ONE spawned free body must
// never drop unrelated records. Covers the general SceneIR::RemoveBodySubtree
// remap (pure host), the live spawn->delete->re-cook->Save round-trip on a real
// jointed scene (go2 + script + exclude pair), the /script node delete, the
// reparent-onto-a-shape guard, and the transactional re-cook failure path.
// ---------------------------------------------------------------------------

#include "phi/backend.hpp"
#include <atomic>
#include <chrono>
#include "runtime/app/simulation.hpp"
#include "runtime/app/viewer/editor_edits.hpp"
#include "runtime/app/viewer/editor_scene.hpp"
#include "scene/format/nks.hpp"
#include "scene/graph/scene_graph.hpp"
#include "scene/scene_ir.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace viewer = nuka::runtime::app::viewer;
using namespace nuka::scene;

namespace {

constexpr float kDt = 1.0f / 240.0f;

class TempDir {
public:
    TempDir() {
        static std::atomic<unsigned long long> seq{0ull};
        const auto stamp =
            std::chrono::steady_clock::now().time_since_epoch().count();
        fs::path candidate = fs::temp_directory_path() /
            ("nuka_test_" + std::to_string(stamp) + "_" +
             std::to_string(seq++));
        if (!fs::create_directory(candidate)) {
            throw std::runtime_error("temp dir create failed");
        }
        path_ = candidate;
    }
    ~TempDir() { std::error_code ec; fs::remove_all(path_, ec); }
    fs::path File(const std::string& n) const { return path_ / n; }
private:
    fs::path path_;
};

std::vector<uint8_t> FileBytes(const fs::path& p) {
    std::FILE* f = std::fopen(p.string().c_str(), "rb");
    if (!f) return {};
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(n > 0 ? static_cast<size_t>(n) : 0u);
    if (!buf.empty()) buf.resize(std::fread(buf.data(), 1, buf.size(), f));
    std::fclose(f);
    return buf;
}

// A reference scene exercising every record family that references bodies:
// an articulated pair (rig/base + rig/arm, joint + actuator + sensor + camera +
// light + IC), a free prop, exclude/contact pairs, media-free, one script.
SceneIR BuildReference() {
    SceneIR s;
    MaterialRecord m;
    m.name = "mat";
    const MaterialId mid = s.AddMaterial(m);

    RigidBodyRecord base;
    base.name = "rig/base";
    const BodyId b_base = s.AddRigidBody(base);
    RigidBodyRecord arm;
    arm.name = "arm";
    arm.parent_id = b_base;
    const BodyId b_arm = s.AddRigidBody(arm);
    RigidBodyRecord prop;
    prop.name = "prop";
    const BodyId b_prop = s.AddRigidBody(prop);

    CollisionShapeRecord cs;
    cs.material_id = mid;
    cs.body_id = b_base; cs.name = "base_geom"; s.AddCollisionShape(cs);
    cs.body_id = b_arm;  cs.name = "arm_geom";  const ShapeId sh_arm = s.AddCollisionShape(cs);
    cs.body_id = b_prop; cs.name = "prop_geom"; const ShapeId sh_prop = s.AddCollisionShape(cs);

    JointRecord j;
    j.name = "elbow";
    j.parent_body = b_base;
    j.child_body = b_arm;
    const JointId jid = s.AddJoint(j);

    ActuatorRecord a;
    a.name = "elbow_act";
    a.joint_id = jid;
    s.AddActuator(a);

    SensorDesc imu;
    imu.name = "arm_imu";
    imu.mount = MountFrame::Body;
    imu.mount_index = b_arm;
    s.AddSensor(imu);

    CameraRecord cam;
    cam.name = "arm_cam";
    cam.attached_body = b_arm;
    s.AddCamera(cam);
    LightRecord light;
    light.name = "prop_light";
    light.attached_body = b_prop;
    s.AddLight(light);

    s.AddExcludePair(b_base, b_arm);
    s.AddExcludePair(b_arm, b_prop);
    ContactPairOverride cp;
    cp.geom1 = sh_arm;
    cp.geom2 = sh_prop;
    s.AddContactPair(cp);

    ScriptRecord sr;
    sr.stable_id = 7;
    sr.source = "x = 1";
    s.AddScript(sr);

    ArticulationInitialState ic;
    ic.qpos = {0.1f, 0.2f};
    s.InitialStateMut()["rig"] = ic;
    return s;
}

}  // namespace

// Deleting the free prop keeps the articulation intact: joint / actuator /
// sensor / camera / IC survive; only prop-referencing records drop, remapped.
TEST(RemoveBodySubtree, UnrelatedRecordsSurviveWithRemap) {
    SceneIR s = BuildReference();
    const size_t bodies0 = s.RigidBodyCount();

    ASSERT_TRUE(s.RemoveBodySubtree(2 /*prop*/));

    EXPECT_EQ(s.RigidBodyCount(), bodies0 - 1u);
    ASSERT_EQ(s.JointCount(), 1u) << "joint must survive an unrelated delete";
    EXPECT_EQ(s.GetJoint(0).parent_body, 0u);
    EXPECT_EQ(s.GetJoint(0).child_body, 1u);
    ASSERT_EQ(s.ActuatorCount(), 1u);
    EXPECT_EQ(s.GetActuator(0).joint_id, 0u);
    ASSERT_EQ(s.SensorCount(), 1u);
    EXPECT_EQ(s.GetSensor(0).mount_index, 1u);
    EXPECT_EQ(s.CameraCount(), 1u);
    EXPECT_EQ(s.LightCount(), 0u) << "the prop's light belongs to the subtree";
    ASSERT_EQ(s.ExcludePairs().size(), 1u) << "only the prop-touching pair drops";
    EXPECT_EQ(s.ExcludePairs()[0].first, 0u);
    EXPECT_EQ(s.ExcludePairs()[0].second, 1u);
    EXPECT_EQ(s.ContactPairs().size(), 0u) << "the prop-shape override must drop";
    EXPECT_EQ(s.ScriptCount(), 1u) << "scripts are body-independent";
    EXPECT_EQ(s.InitialState().count("rig"), 1u) << "untouched IC must persist";
    ASSERT_EQ(s.ShapeCount(), 2u);
    EXPECT_EQ(s.GetShape(0).body_id, 0u);
    EXPECT_EQ(s.GetShape(1).body_id, 1u);
}

// Deleting the articulation root takes its joints / actuator / sensor / camera /
// touched IC with it and remaps the surviving prop's references.
TEST(RemoveBodySubtree, SubtreeReferencesGoWithIt) {
    SceneIR s = BuildReference();

    ASSERT_TRUE(s.RemoveBodySubtree(0 /*rig/base -> arm follows*/));

    ASSERT_EQ(s.RigidBodyCount(), 1u);
    EXPECT_EQ(s.GetBody(0).name, "prop");
    EXPECT_EQ(s.JointCount(), 0u);
    EXPECT_EQ(s.ActuatorCount(), 0u);
    EXPECT_EQ(s.SensorCount(), 0u);
    EXPECT_EQ(s.CameraCount(), 0u);
    ASSERT_EQ(s.LightCount(), 1u);
    EXPECT_EQ(s.GetLight(0).attached_body, 0u) << "prop light must remap";
    EXPECT_EQ(s.ExcludePairs().size(), 0u);
    EXPECT_EQ(s.ContactPairs().size(), 0u);
    EXPECT_EQ(s.InitialState().count("rig"), 0u) << "touched IC must drop";
    ASSERT_EQ(s.ShapeCount(), 1u);
    EXPECT_EQ(s.GetShape(0).body_id, 0u);
    EXPECT_EQ(s.ScriptCount(), 1u);
}

// ===========================================================================
// Live gates -- cook go2.nks on the phi device (SKIP when none).
// ===========================================================================

class EditorDeleteLive : public ::testing::Test {
protected:
    void SetUp() override {
        dev_ = nuka::phi::InitBestDevice();
        if (!dev_) GTEST_SKIP() << "no phi device";
        backend_ = nuka::phi::DeviceInitBackend(dev_, nullptr);
        if (!backend_) GTEST_SKIP() << "no phi backend";
        if (!fs::exists(kScene)) GTEST_SKIP() << "go2.nks missing";
        tmp_ = std::make_unique<TempDir>();
        es_ = viewer::LoadEditorScene(kScene, dev_, backend_, kDt);
        ASSERT_NE(es_, nullptr) << "cook/load failed";
    }
    const char* kScene = "examples/scenes/go2.nks";
    nuka::phi::Device*  dev_ = nullptr;
    nuka::phi::Backend* backend_ = nullptr;
    std::unique_ptr<TempDir> tmp_;
    std::unique_ptr<viewer::EditorScene> es_;
};

// THE review repro: on a jointed scene carrying a script + an exclude pair,
// spawn a box, delete it, re-cook -- every unrelated record must survive, and
// the Save of the result must round-trip byte-identically.
TEST_F(EditorDeleteLive, SpawnDeleteKeepsJointsScriptsExcludes) {
    // Author the extra record families the trap destroyed.
    es_->scene.AddExcludePair(0, 1);
    ScriptRecord sr;
    sr.stable_id = 42;
    sr.source = "y = 2";
    es_->scene.AddScript(sr);

    const size_t joints0  = es_->scene.JointCount();
    const size_t acts0    = es_->scene.ActuatorCount();
    const size_t sens0    = es_->scene.SensorCount();
    const size_t excl0    = es_->scene.ExcludePairs().size();
    const size_t scripts0 = es_->scene.ScriptCount();
    const size_t bodies0  = es_->scene.RigidBodyCount();
    const uint32_t rows0  = es_->caps.bodies_per_env;
    ASSERT_GT(joints0, 0u) << "go2 must be jointed for this gate to bite";

    nuka::math::Transform placement;
    placement.position = nuka::math::Vec3{0.0f, 0.0f, 1.0f};
    const std::string path = viewer::SpawnPrimitive(*es_, viewer::PrimitiveKind::Box,
                                                    placement, "");
    ASSERT_FALSE(path.empty());
    ASSERT_TRUE(viewer::RecookEditorScene(*es_, dev_, backend_, kDt));
    ASSERT_EQ(es_->scene.RigidBodyCount(), bodies0 + 1u);

    // A root-level node's parent path is "" (the tree root), so success is
    // asserted through the record count, exactly as the viewer detects `changed`.
    viewer::DeleteSubtree(*es_, path);
    ASSERT_EQ(es_->scene.RigidBodyCount(), bodies0) << "delete declined";
    ASSERT_TRUE(viewer::RecookEditorScene(*es_, dev_, backend_, kDt));

    EXPECT_EQ(es_->scene.JointCount(), joints0) << "DATA LOSS: joints dropped";
    EXPECT_EQ(es_->scene.ActuatorCount(), acts0) << "DATA LOSS: actuators dropped";
    EXPECT_EQ(es_->scene.SensorCount(), sens0) << "DATA LOSS: sensors dropped";
    EXPECT_EQ(es_->scene.ExcludePairs().size(), excl0) << "DATA LOSS: excludes dropped";
    EXPECT_EQ(es_->scene.ScriptCount(), scripts0) << "DATA LOSS: scripts dropped";
    EXPECT_EQ(es_->caps.bodies_per_env, rows0) << "cooked rows did not return";

    // Save round-trip: save -> load -> save again must be byte-identical (the two
    // saves use the SAME basename in sibling dirs -- the JSON embeds the .nka name).
    const fs::path d1 = tmp_->File("a"), d2 = tmp_->File("b");
    fs::create_directories(d1);
    fs::create_directories(d2);
    const fs::path s1 = d1 / "scene.nks";
    nks::Save(es_->scene, s1.string());
    const SceneIR reloaded = nks::Load(s1.string());
    EXPECT_EQ(reloaded.JointCount(), joints0);
    EXPECT_EQ(reloaded.ScriptCount(), scripts0);
    EXPECT_EQ(reloaded.ExcludePairs().size(), excl0);
    const fs::path s2 = d2 / "scene.nks";
    nks::Save(reloaded, s2.string());
    EXPECT_EQ(FileBytes(s1), FileBytes(s2)) << "Save round-trip not byte-identical";
    EXPECT_EQ(FileBytes(d1 / "scene.nka"), FileBytes(d2 / "scene.nka"))
        << ".nka sibling round-trip not byte-identical";
}

// Deleting a /script node drops exactly that record; bodies / joints untouched.
TEST_F(EditorDeleteLive, ScriptNodeDeletes) {
    ScriptRecord sr;
    sr.stable_id = 9;
    sr.source = "z = 3";
    const ScriptId sid = es_->scene.AddScript(sr);
    const size_t bodies0 = es_->scene.RigidBodyCount();
    const size_t joints0 = es_->scene.JointCount();

    const auto node = es_->scene.Ecs().NodeOf(es_->scene.EntityOfScript(sid));
    ASSERT_NE(node, nullptr);
    const std::string path = es_->scene.Tree().PathOf(node);
    viewer::DeleteSubtree(*es_, path);

    EXPECT_EQ(es_->scene.ScriptCount(), 0u) << "script record must drop";
    EXPECT_EQ(es_->scene.RigidBodyCount(), bodies0);
    EXPECT_EQ(es_->scene.JointCount(), joints0);
}

// Reparenting onto a shape node is declined (no phantom same-named group).
TEST_F(EditorDeleteLive, ReparentOntoShapeDeclined) {
    nuka::math::Transform placement;
    placement.position = nuka::math::Vec3{0.0f, 0.0f, 1.0f};
    const std::string box = viewer::SpawnPrimitive(*es_, viewer::PrimitiveKind::Box,
                                                   placement, "");
    ASSERT_FALSE(box.empty());
    // Any shape-backed node path: the spawned box's own colliding geom.
    const std::string shape_path = box + "/box_geom";
    ASSERT_NE(es_->scene.Tree().NodeOf(shape_path), nullptr);

    const std::string prop = viewer::SpawnPrimitive(*es_, viewer::PrimitiveKind::Sphere,
                                                    placement, "");
    ASSERT_FALSE(prop.empty());
    EXPECT_EQ(viewer::ReparentNode(*es_, prop, shape_path), "")
        << "reparent onto a shape node must be declined";
}

// The transactional re-cook: a failing rebuild (null device -> World never Ready)
// must leave the OLD world + sim alive and steppable, never a dangling null.
TEST_F(EditorDeleteLive, FailedRecookKeepsOldWorld) {
    nuka::nk::World* world_before = es_->world.get();
    nuka::runtime::app::Simulation* sim_before = es_->sim.get();
    ASSERT_TRUE(es_->world->Ready());

    ASSERT_FALSE(viewer::RecookEditorScene(*es_, nullptr, nullptr, kDt))
        << "a null device must fail the rebuild";

    EXPECT_EQ(es_->world.get(), world_before) << "old world must be retained";
    EXPECT_EQ(es_->sim.get(), sim_before) << "old sim must be retained";
    ASSERT_NE(es_->world.get(), nullptr);
    EXPECT_TRUE(es_->world->Ready());
    EXPECT_TRUE(es_->sim->FramePublish(nullptr, /*do_step=*/true))
        << "the retained world must still step healthily";
}
