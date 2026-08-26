// ---------------------------------------------------------------------------
// Editor undo/redo state round-trip gate.
//
// For EACH of the six editor_edits kinds (transform, material, rename, add,
// delete, reparent) this cooks a real EditorScene, applies the edit + records the
// reversible op, then asserts apply -> undo restores BOTH the authoritative SceneIR
// (FNV over the persisted .nks bytes -- exactly what a later Save writes) AND the
// live RenderWorld (field-equality), and redo re-reaches the edited state. The
// edited != original guard makes it SENSITIVE: a neutered revert leaves the edited
// digest and the round-trip assertion fails.
//
// HOST-ONLY: cooks on the phi device (SKIP when none); no Vulkan / window.
// ---------------------------------------------------------------------------

#include "phi/backend.hpp"
#include <atomic>
#include <chrono>
#include "render/render_world.hpp"
#include "runtime/app/simulation.hpp"
#include "runtime/app/viewer/editor_edits.hpp"
#include "runtime/app/viewer/editor_scene.hpp"
#include "runtime/app/viewer/editor_undo.hpp"
#include "scene/format/nks.hpp"
#include "scene/graph/scene_graph.hpp"
#include "scene/scene_ir.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace viewer = nuka::runtime::app::viewer;
using nuka::scene::SceneIR;

namespace {

constexpr float kDt = 1.0f / 240.0f;

uint64_t FnvBytes(const void* data, size_t n, uint64_t h = 1469598103934665603ull) {
    const auto* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}
uint64_t FnvF(float f, uint64_t h) { return FnvBytes(&f, sizeof(f), h); }
uint64_t FnvU(uint32_t v, uint64_t h) { return FnvBytes(&v, sizeof(v), h); }

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

// The AUTHORITY digest: FNV over the bytes a Save persists -- the exact thing that
// must round-trip so a later Save writes the reverted state.
uint64_t SceneFnv(const SceneIR& s, const fs::path& tmp) {
    nuka::scene::nks::Save(s, tmp.string());
    std::FILE* f = std::fopen(tmp.string().c_str(), "rb");
    if (!f) return 0u;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(n > 0 ? static_cast<size_t>(n) : 0u);
    if (!buf.empty()) { size_t rd = std::fread(buf.data(), 1, buf.size(), f); buf.resize(rd); }
    std::fclose(f);
    return FnvBytes(buf.data(), buf.size());
}

// The LIVE render digest: per-instance pose + mesh + material slot, plus the full
// material field set + counts. Order is deterministic across a re-cook, so it is a
// faithful "did undo restore the visible/live state" signal (entity ids excluded:
// they re-project and carry no visible meaning).
uint64_t RenderFnv(const nuka::render::RenderWorld& w) {
    uint64_t h = 1469598103934665603ull;
    h = FnvU(w.InstanceCount(), h);
    h = FnvU(w.MaterialCount(), h);
    for (const auto& inst : w.instances) {
        const auto& t = inst.world_xform;
        h = FnvF(t.position.x, h); h = FnvF(t.position.y, h); h = FnvF(t.position.z, h);
        h = FnvF(t.rotation.w, h); h = FnvF(t.rotation.x, h);
        h = FnvF(t.rotation.y, h); h = FnvF(t.rotation.z, h);
        h = FnvU(inst.mesh_id, h);
        h = FnvU(inst.render_material_id, h);
    }
    for (uint32_t i = 0; i < w.MaterialCount(); ++i) {
        const auto& m = w.materials[i];
        for (float c : m.base_color) h = FnvF(c, h);
        h = FnvF(m.roughness, h); h = FnvF(m.metallic, h);
        for (float e : m.emissive) h = FnvF(e, h);
        h = FnvF(m.transmission, h); h = FnvF(m.ior, h); h = FnvF(m.sheen, h);
    }
    return h;
}

// A clean editable scene: a grey ground, a static VISUAL "redbox" (skin shape +
// red material -> owns a render instance + editable material), and a movable
// "meshprop" carrying a collision hull -- the delete / reparent subject.
SceneIR BuildDemo() {
    using namespace nuka::scene;
    SceneIR s;
    MaterialRecord red; red.name = "demo_red";
    red.base_color = {0.85f, 0.13f, 0.13f}; red.roughness = 0.45f;
    const MaterialId red_id = s.AddMaterial(red);
    MaterialRecord gnd; gnd.name = "demo_ground"; gnd.base_color = {0.46f, 0.48f, 0.52f};
    const MaterialId gnd_id = s.AddMaterial(gnd);

    RigidBodyRecord gb; gb.name = "ground"; gb.is_static = true;
    const BodyId gbid = s.AddRigidBody(gb);
    CollisionShapeRecord gs; gs.body_id = gbid; gs.type = ShapeType::Box;
    gs.half_extents = {3.0f, 3.0f, 0.1f}; gs.material_id = gnd_id; gs.name = "gnd_geom";
    s.AddCollisionShape(gs);

    RigidBodyRecord bb; bb.name = "redbox"; bb.is_static = true;
    bb.local_transform.position = {0.0f, 0.0f, 0.6f};
    const BodyId bbid = s.AddRigidBody(bb);
    CollisionShapeRecord vis; vis.body_id = bbid; vis.name = "skin";
    vis.type = ShapeType::Box; vis.half_extents = {0.4f, 0.4f, 0.4f};
    vis.material_id = red_id; vis.contype = 0; vis.conaffinity = 0;
    s.AddCollisionShape(vis);

    RigidBodyRecord mp; mp.name = "meshprop"; mp.mass = 1.0f;
    mp.local_transform.position = {1.2f, 0.0f, 0.5f};
    const BodyId mpid = s.AddRigidBody(mp);
    CollisionShapeRecord ms; ms.body_id = mpid; ms.name = "hull";
    ms.type = ShapeType::ConvexHull; ms.decompose_mode = DecomposeMode::Skip;
    ms.mesh_vertices = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                        0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    ms.mesh_indices = {0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3};
    s.AddCollisionShape(ms);
    return s;
}

nuka::scene::EntityId EntForPath(const viewer::EditorScene& es, const std::string& p) {
    if (const auto n = es.scene.Tree().NodeOf(p)) return n->entity;
    return nuka::scene::kInvalidEntity;
}

// A fixture that stands up one cooked EditorScene the whole suite edits.
class EditorUndo : public ::testing::Test {
protected:
    void SetUp() override {
        dev_ = nuka::phi::InitBestDevice();
        if (!dev_) GTEST_SKIP() << "no phi device";
        backend_ = nuka::phi::DeviceInitBackend(dev_, nullptr);
        if (!backend_) GTEST_SKIP() << "no phi backend";
        tmp_ = std::make_unique<TempDir>();
        const std::string src = tmp_->File("demo.nks").string();
        nuka::scene::nks::Save(BuildDemo(), src);
        es_ = viewer::LoadEditorScene(src, dev_, backend_, kDt);
        ASSERT_NE(es_, nullptr) << "cook/load failed";
        idx_.Build(es_->scene);
    }

    uint64_t SceneFnvNow() { return SceneFnv(es_->scene, tmp_->File("probe.nks")); }
    uint64_t RenderFnvNow() { return RenderFnv(es_->sim->GetRenderWorld()); }
    const nuka::render::RenderWorld& Rw() { return es_->sim->GetRenderWorld(); }

    nuka::phi::Device*  dev_ = nullptr;
    nuka::phi::Backend* backend_ = nullptr;
    std::unique_ptr<TempDir> tmp_;
    std::unique_ptr<viewer::EditorScene> es_;
    viewer::EntityRecordIndex idx_;
    viewer::EditStack stack_;
};

}  // namespace

// Transform edit on the static "redbox/skin" instance: undo restores the record +
// the live RenderWorld pose exactly; redo re-applies it.
TEST_F(EditorUndo, TransformRoundTrip) {
    const std::string path = "redbox/skin";
    const nuka::scene::EntityId e = EntForPath(*es_, path);
    ASSERT_NE(e, nuka::scene::kInvalidEntity);
    // redbox is a movable Body pose-source: the teleport rides the MoveEntity queue,
    // so a publish (drain + republish from Data) lands it in the live RenderWorld.
    auto publish = [&] { es_->sim->FramePublish(nullptr, /*do_step=*/false); };
    publish();  // settle the baseline render poses from Data
    const nuka::render::RenderInstance* inst = viewer::InstanceOfEntity(Rw(), e);
    ASSERT_NE(inst, nullptr);

    const uint64_t s0 = SceneFnvNow(), r0 = RenderFnvNow();
    nuka::math::Transform before = inst->world_xform;
    nuka::math::Transform after = before;
    after.position.x += 0.55f; after.position.z -= 0.30f;

    viewer::EditOp op = viewer::MakeTransformOp(es_.get(), path, before, after);
    op.apply(); publish();            // land the edit (live)
    stack_.Record(op);
    const uint64_t s1 = SceneFnvNow(), r1 = RenderFnvNow();
    ASSERT_NE(s1, s0) << "transform edit did not change the authority";
    ASSERT_NE(r1, r0) << "transform edit did not change the render world";

    ASSERT_TRUE(stack_.Undo()); publish();
    EXPECT_EQ(SceneFnvNow(), s0) << "undo did not restore the authority";
    EXPECT_EQ(RenderFnvNow(), r0) << "undo did not restore the render pose";

    ASSERT_TRUE(stack_.Redo()); publish();
    EXPECT_EQ(SceneFnvNow(), s1);
    EXPECT_EQ(RenderFnvNow(), r1);
}

// Material edit on "redbox/skin": undo restores the MaterialRecord + the live
// render material slot; redo re-applies it.
TEST_F(EditorUndo, MaterialRoundTrip) {
    const std::string path = "redbox/skin";
    const nuka::scene::EntityId e = EntForPath(*es_, path);
    ASSERT_NE(e, nuka::scene::kInvalidEntity);
    const viewer::MaterialBinding mb = viewer::ResolveMaterial(*es_, idx_, e);
    ASSERT_TRUE(mb.valid);
    nuka::scene::RenderMaterial before = Rw().materials[mb.render_slot];
    nuka::scene::RenderMaterial after = before;
    after.base_color[0] = 0.10f; after.base_color[1] = 0.80f; after.base_color[2] = 0.30f;
    after.roughness = 0.18f; after.metallic = 0.90f; after.emissive[0] = 0.4f;

    const uint64_t s0 = SceneFnvNow(), r0 = RenderFnvNow();
    viewer::EditOp op = viewer::MakeMaterialOp(es_.get(), path, before, after);
    op.apply();
    stack_.Record(op);
    const uint64_t s1 = SceneFnvNow(), r1 = RenderFnvNow();
    ASSERT_NE(s1, s0) << "material edit did not change the authority";
    ASSERT_NE(r1, r0) << "material edit did not change the render world";

    ASSERT_TRUE(stack_.Undo());
    EXPECT_EQ(SceneFnvNow(), s0) << "undo did not restore the material record";
    EXPECT_EQ(RenderFnvNow(), r0) << "undo did not restore the render material";

    ASSERT_TRUE(stack_.Redo());
    EXPECT_EQ(SceneFnvNow(), s1);
    EXPECT_EQ(RenderFnvNow(), r1);
}

// Structural helper: snapshot before, run a forward mutator + re-cook, snapshot
// after, record one structural op. Returns the recorded op (already applied).
viewer::EditOp RecordStructural(viewer::EditorScene* es, viewer::EntityRecordIndex* idx,
                                nuka::phi::Device* dev, nuka::phi::Backend* backend,
                                const std::function<void()>& forward) {
    SceneIR before = es->scene;
    forward();
    EXPECT_TRUE(viewer::RecookEditorScene(*es, dev, backend, kDt));
    idx->Build(es->scene);
    SceneIR after = es->scene;
    return viewer::MakeStructuralOp(es, idx, dev, backend, kDt, before, after);
}

TEST_F(EditorUndo, RenameRoundTrip) {
    const uint64_t s0 = SceneFnvNow(), r0 = RenderFnvNow();
    viewer::EditOp op = RecordStructural(es_.get(), &idx_, dev_, backend_, [&] {
        viewer::RenameNode(*es_, "redbox/skin", "crate");
    });
    stack_.Record(op);
    const uint64_t s1 = SceneFnvNow();
    ASSERT_NE(s1, s0) << "rename did not change the authority";

    ASSERT_TRUE(stack_.Undo());
    EXPECT_EQ(SceneFnvNow(), s0) << "undo did not restore the name";
    EXPECT_EQ(RenderFnvNow(), r0) << "undo did not restore the render world";
    ASSERT_TRUE(stack_.Redo());
    EXPECT_EQ(SceneFnvNow(), s1);
}

TEST_F(EditorUndo, AddBoxRoundTrip) {
    const uint64_t s0 = SceneFnvNow(), r0 = RenderFnvNow();
    const uint32_t bodies0 = es_->scene.RigidBodyCount();
    viewer::EditOp op = RecordStructural(es_.get(), &idx_, dev_, backend_, [&] {
        viewer::AddBoxChild(*es_, "");  // a unit box under the scene root
    });
    stack_.Record(op);
    ASSERT_EQ(es_->scene.RigidBodyCount(), bodies0 + 1u);
    const uint64_t s1 = SceneFnvNow(), r1 = RenderFnvNow();
    ASSERT_NE(s1, s0); ASSERT_NE(r1, r0) << "add did not change the render world";

    ASSERT_TRUE(stack_.Undo());
    EXPECT_EQ(es_->scene.RigidBodyCount(), bodies0) << "undo did not remove the added body";
    EXPECT_EQ(SceneFnvNow(), s0);
    EXPECT_EQ(RenderFnvNow(), r0) << "undo did not restore the render world";
    ASSERT_TRUE(stack_.Redo());
    EXPECT_EQ(es_->scene.RigidBodyCount(), bodies0 + 1u);
    EXPECT_EQ(RenderFnvNow(), r1);
}

// Delete the "meshprop" subtree; undo must bring the body AND its collision hull
// back (the structural snapshot restores the whole subtree).
TEST_F(EditorUndo, DeleteRoundTrip) {
    auto has_meshprop = [&] {
        for (nuka::scene::BodyId i = 0; i < es_->scene.RigidBodyCount(); ++i)
            if (es_->scene.GetBody(i).name == "meshprop") return true;
        return false;
    };
    auto has_hull = [&] {
        for (nuka::scene::ShapeId i = 0; i < es_->scene.ShapeCount(); ++i)
            if (es_->scene.GetShape(i).name == "hull") return true;
        return false;
    };
    ASSERT_TRUE(has_meshprop()); ASSERT_TRUE(has_hull());
    const uint64_t s0 = SceneFnvNow(), r0 = RenderFnvNow();

    viewer::EditOp op = RecordStructural(es_.get(), &idx_, dev_, backend_, [&] {
        viewer::DeleteSubtree(*es_, "meshprop");
    });
    stack_.Record(op);
    ASSERT_FALSE(has_meshprop()) << "delete did not remove the body";
    ASSERT_FALSE(has_hull()) << "delete did not remove the hull";
    const uint64_t s1 = SceneFnvNow();
    ASSERT_NE(s1, s0);

    ASSERT_TRUE(stack_.Undo());
    EXPECT_TRUE(has_meshprop()) << "undo did not restore the deleted body";
    EXPECT_TRUE(has_hull()) << "undo did not restore the deleted hull";
    EXPECT_EQ(SceneFnvNow(), s0) << "undo did not restore the authority";
    EXPECT_EQ(RenderFnvNow(), r0) << "undo did not restore the render world";
    ASSERT_TRUE(stack_.Redo());
    EXPECT_FALSE(has_meshprop());
    EXPECT_EQ(SceneFnvNow(), s1);
}

// Reparent "meshprop" under "redbox"; undo must restore BOTH the old parent and
// the original world pose (ReparentNode re-expresses the local transform).
TEST_F(EditorUndo, ReparentRoundTrip) {
    auto meshprop = [&]() -> const nuka::scene::RigidBodyRecord* {
        for (nuka::scene::BodyId i = 0; i < es_->scene.RigidBodyCount(); ++i)
            if (es_->scene.GetBody(i).name.find("meshprop") != std::string::npos)
                return &es_->scene.GetBody(i);
        return nullptr;
    };
    const nuka::scene::RigidBodyRecord* mp0 = meshprop();
    ASSERT_NE(mp0, nullptr);
    const nuka::scene::BodyId parent0 = mp0->parent_id;
    const nuka::math::Vec3 pos0 = mp0->local_transform.position;
    const uint64_t s0 = SceneFnvNow(), r0 = RenderFnvNow();

    viewer::EditOp op = RecordStructural(es_.get(), &idx_, dev_, backend_, [&] {
        const std::string np = viewer::ReparentNode(*es_, "meshprop", "redbox");
        EXPECT_FALSE(np.empty()) << "reparent declined";
    });
    stack_.Record(op);
    const nuka::scene::RigidBodyRecord* mp1 = meshprop();
    ASSERT_NE(mp1, nullptr);
    ASSERT_NE(mp1->parent_id, parent0) << "reparent did not change the parent";
    const uint64_t s1 = SceneFnvNow();
    ASSERT_NE(s1, s0);

    ASSERT_TRUE(stack_.Undo());
    const nuka::scene::RigidBodyRecord* mp2 = meshprop();
    ASSERT_NE(mp2, nullptr);
    EXPECT_EQ(mp2->parent_id, parent0) << "undo did not restore the old parent";
    EXPECT_FLOAT_EQ(mp2->local_transform.position.x, pos0.x) << "undo did not restore the pose";
    EXPECT_FLOAT_EQ(mp2->local_transform.position.y, pos0.y);
    EXPECT_FLOAT_EQ(mp2->local_transform.position.z, pos0.z);
    EXPECT_EQ(SceneFnvNow(), s0);
    EXPECT_EQ(RenderFnvNow(), r0) << "undo did not restore the render world";
    ASSERT_TRUE(stack_.Redo());
    EXPECT_EQ(SceneFnvNow(), s1);
}

// The stack mechanics: a new edit after an undo truncates the redo branch, and
// CanUndo / CanRedo track depth.
TEST_F(EditorUndo, RedoTruncatedByNewEdit) {
    const std::string path = "redbox/skin";
    const nuka::render::RenderInstance* inst =
        viewer::InstanceOfEntity(Rw(), EntForPath(*es_, path));
    ASSERT_NE(inst, nullptr);
    nuka::math::Transform a = inst->world_xform, b = a; b.position.x += 0.2f;
    nuka::math::Transform c = a; c.position.y += 0.4f;

    viewer::EditOp op1 = viewer::MakeTransformOp(es_.get(), path, a, b);
    op1.apply(); stack_.Record(op1);
    EXPECT_TRUE(stack_.CanUndo()); EXPECT_FALSE(stack_.CanRedo());
    ASSERT_TRUE(stack_.Undo());
    EXPECT_FALSE(stack_.CanUndo()); EXPECT_TRUE(stack_.CanRedo());

    // A fresh edit clears the redo branch.
    viewer::EditOp op2 = viewer::MakeTransformOp(es_.get(), path, a, c);
    op2.apply(); stack_.Record(op2);
    EXPECT_TRUE(stack_.CanUndo());
    EXPECT_FALSE(stack_.CanRedo()) << "a new edit did not truncate the redo branch";
}
