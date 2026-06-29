// ---------------------------------------------------------------------------
// Editor primitive-spawn round-trip gate.
//
// For EACH spawnable primitive (box / sphere / capsule / plane) this cooks a real
// EditorScene, spawns the body through the general SpawnPrimitive mutator + records
// the reversible structural op, then asserts:
//   (1) the authority grew by one body + two shapes (a colliding geom + a visual
//       twin) + a material, with the right static/dynamic flag and contype split;
//   (2) the re-cook produced a new render instance (it renders) and, for a dynamic
//       body, a new movable collidable row (caps.bodies_per_env);
//   (3) Save -> reload survives the spawned records AND the persisted bytes;
//   (4) undo restores the authority + the live RenderWorld EXACTLY, redo re-reaches
//       the spawned state.
// The before != after digest guard makes it SENSITIVE: a neutered spawn / revert
// leaves the wrong digest and an assertion fails.
//
// HOST-ONLY: cooks on the phi device (SKIP when none); no Vulkan / window.
// ---------------------------------------------------------------------------

#include "phi/backend.hpp"
#include "render/render_world.hpp"
#include "runtime/app/simulation.hpp"
#include "runtime/app/viewer/editor_edits.hpp"
#include "runtime/app/viewer/editor_scene.hpp"
#include "runtime/app/viewer/editor_undo.hpp"
#include "runtime/app/viewer/primitive_kind.hpp"
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
using nuka::scene::SceneIR;

namespace {

constexpr float kDt = 1.0f / 240.0f;

uint64_t FnvBytes(const void* data, size_t n, uint64_t h = 1469598103934665603ull) {
    const auto* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

class TempDir {
public:
    TempDir() {
        char tmpl[] = "/tmp/editor_spawn_XXXXXX";
        char* p = ::mkdtemp(tmpl);
        if (!p) throw std::runtime_error("mkdtemp failed");
        path_ = p;
    }
    ~TempDir() { std::error_code ec; fs::remove_all(path_, ec); }
    fs::path File(const std::string& n) const { return path_ / n; }
private:
    fs::path path_;
};

// The AUTHORITY digest: FNV over the bytes a Save persists -- the exact thing a later
// Save writes, so an undo / a reload must reproduce it.
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

bool HasBody(const SceneIR& s, const std::string& name, bool* is_static = nullptr) {
    for (nuka::scene::BodyId i = 0; i < s.RigidBodyCount(); ++i)
        if (s.GetBody(i).name == name) {
            if (is_static) *is_static = s.GetBody(i).is_static;
            return true;
        }
    return false;
}
// A shape's contype by name (-1 when absent): the colliding geom keeps contype 1,
// the visual twin carries contype 0.
int ShapeContype(const SceneIR& s, const std::string& name) {
    for (nuka::scene::ShapeId i = 0; i < s.ShapeCount(); ++i)
        if (s.GetShape(i).name == name) return static_cast<int>(s.GetShape(i).contype);
    return -1;
}
bool HasMaterial(const SceneIR& s, const std::string& name) {
    for (nuka::scene::MaterialId i = 0; i < s.MaterialCount(); ++i)
        if (s.GetMaterial(i).name == name) return true;
    return false;
}

// The render instance for a node at `path`, or nullptr.
const nuka::render::RenderInstance* InstanceAt(const viewer::EditorScene& es,
                                               const std::string& path) {
    const auto node = es.scene.Tree().NodeOf(path);
    if (!node) return nullptr;
    return viewer::InstanceOfEntity(es.sim->GetRenderWorld(), node->entity);
}
// A body's authored mass by name (-1 when absent): dynamic kinds carry mass, the
// static plane is massless.
float BodyMass(const SceneIR& s, const std::string& name) {
    for (nuka::scene::BodyId i = 0; i < s.RigidBodyCount(); ++i)
        if (s.GetBody(i).name == name) return s.GetBody(i).mass;
    return -1.0f;
}

// A clean editable scene: a static ground box, a static VISUAL redbox (a primitive
// VisualMeshComponent), and a dynamic meshprop carrying a collision hull -- enough
// that the cook already emits movable + render rows for the deltas to be meaningful.
SceneIR BuildDemo() {
    using namespace nuka::scene;
    SceneIR s;
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
    vis.contype = 0; vis.conaffinity = 0;
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

struct SpawnCase {
    viewer::PrimitiveKind kind;
    const char*           tag;
    bool                  dynamic;   // box/sphere/capsule are dynamic; plane is static
};

class EditorSpawn : public ::testing::TestWithParam<SpawnCase> {
protected:
    void SetUp() override {
        dev_ = nuka::phi::InitBestDevice();
        if (!dev_) GTEST_SKIP() << "no phi device";
        backend_ = nuka::phi::DeviceInitBackend(dev_, nullptr);
        if (!backend_) GTEST_SKIP() << "no phi backend";
        tmp_ = std::make_unique<TempDir>();
        nuka::scene::nks::Save(BuildDemo(), tmp_->File("demo.nks").string());
        es_ = viewer::LoadEditorScene(tmp_->File("demo.nks").string(), dev_, backend_, kDt);
        ASSERT_NE(es_, nullptr) << "cook/load failed";
        idx_.Build(es_->scene);
    }

    uint64_t SceneFnvNow() { return SceneFnv(es_->scene, tmp_->File("probe.nks")); }
    uint32_t Instances()   { return es_->sim->GetRenderWorld().InstanceCount(); }

    nuka::phi::Device*  dev_ = nullptr;
    nuka::phi::Backend* backend_ = nullptr;
    std::unique_ptr<TempDir> tmp_;
    std::unique_ptr<viewer::EditorScene> es_;
    viewer::EntityRecordIndex idx_;
    viewer::EditStack stack_;
};

}  // namespace

TEST_P(EditorSpawn, RoundTrip) {
    const SpawnCase c = GetParam();
    const bool want_static = !c.dynamic;
    const std::string geom   = std::string(c.tag) + "_geom";
    const std::string visual = std::string(c.tag) + "_visual";
    const std::string mat    = std::string(c.tag) + "_mat";

    const uint32_t b0  = es_->scene.RigidBodyCount();
    const uint32_t sh0 = es_->scene.ShapeCount();
    const uint32_t mt0 = es_->scene.MaterialCount();
    const uint32_t bp0 = es_->caps.bodies_per_env;
    const uint32_t i0  = Instances();
    const uint64_t f0  = SceneFnvNow();

    // Spawn through the general mutator + record the structural op (snapshot before /
    // after + re-cook), exactly as the viewer does.
    SceneIR before = es_->scene;
    nuka::math::Transform placement;
    placement.position = nuka::math::Vec3{0.0f, 0.0f, 1.0f};
    const std::string path = viewer::SpawnPrimitive(*es_, c.kind, placement, "");
    ASSERT_FALSE(path.empty()) << "spawn returned no path";
    ASSERT_TRUE(viewer::RecookEditorScene(*es_, dev_, backend_, kDt)) << "re-cook failed";
    idx_.Build(es_->scene);
    SceneIR after = es_->scene;
    stack_.Record(viewer::MakeStructuralOp(es_.get(), &idx_, dev_, backend_, kDt,
                                           std::move(before), std::move(after)));

    // (1) authority grew correctly: one body + two shapes + one material.
    EXPECT_EQ(es_->scene.RigidBodyCount(), b0 + 1u);
    EXPECT_EQ(es_->scene.ShapeCount(), sh0 + 2u) << "expected a colliding geom + a visual twin";
    EXPECT_EQ(es_->scene.MaterialCount(), mt0 + 1u);
    bool spawned_static = c.dynamic;
    EXPECT_TRUE(HasBody(es_->scene, c.tag, &spawned_static)) << "spawned body missing";
    EXPECT_EQ(spawned_static, want_static) << "wrong static/dynamic flag";
    EXPECT_EQ(ShapeContype(es_->scene, geom), 1) << "the geom must collide";
    EXPECT_EQ(ShapeContype(es_->scene, visual), 0) << "the visual twin must be non-colliding";
    EXPECT_TRUE(HasMaterial(es_->scene, mat));

    // (2) it cooked: a new collidable body row + a new render instance appeared (the
    // visual twin renders). Static/dynamic is authored honestly -- the plane is
    // massless, the others carry mass.
    EXPECT_EQ(es_->caps.bodies_per_env, bp0 + 1u) << "no new collidable body row";
    EXPECT_GT(Instances(), i0) << "spawn produced no render instance";
    EXPECT_NE(InstanceAt(*es_, path + "/" + visual), nullptr)
        << "spawned body has no visual render instance";
    EXPECT_EQ(BodyMass(es_->scene, c.tag) > 0.0f, c.dynamic) << "wrong mass for the kind";

    const uint64_t f1 = SceneFnvNow();
    const uint32_t i1 = Instances();
    ASSERT_NE(f1, f0) << "spawn did not change the authority";

    // (3) Save -> reload: the spawned body + its two shapes + material survive (the
    // .nks structural round-trip; the canonical editor round-trip gate).
    const fs::path saved = tmp_->File("spawned.nks");
    nuka::scene::nks::Save(es_->scene, saved.string());
    const SceneIR reloaded = nuka::scene::nks::Load(saved.string());
    EXPECT_EQ(reloaded.RigidBodyCount(), b0 + 1u) << "reload lost a body";
    EXPECT_EQ(reloaded.ShapeCount(), sh0 + 2u) << "reload lost a shape";
    bool rs = c.dynamic;
    EXPECT_TRUE(HasBody(reloaded, c.tag, &rs)) << "spawned body did not survive reload";
    EXPECT_EQ(rs, want_static);
    EXPECT_EQ(ShapeContype(reloaded, geom), 1);
    EXPECT_EQ(ShapeContype(reloaded, visual), 0);
    EXPECT_TRUE(HasMaterial(reloaded, mat));

    // (4) undo restores the authority + the live render world EXACTLY; redo re-reaches.
    ASSERT_TRUE(stack_.Undo());
    EXPECT_EQ(es_->scene.RigidBodyCount(), b0) << "undo did not remove the spawned body";
    EXPECT_EQ(es_->scene.ShapeCount(), sh0) << "undo did not remove the spawned shapes";
    EXPECT_EQ(es_->caps.bodies_per_env, bp0) << "undo did not free the body row";
    EXPECT_EQ(Instances(), i0) << "undo did not restore the render world";
    EXPECT_EQ(SceneFnvNow(), f0) << "undo did not restore the authority";

    ASSERT_TRUE(stack_.Redo());
    EXPECT_EQ(es_->scene.RigidBodyCount(), b0 + 1u);
    EXPECT_EQ(Instances(), i1) << "redo did not restore the render world";
    EXPECT_EQ(SceneFnvNow(), f1) << "redo did not re-reach the spawned authority";
}

INSTANTIATE_TEST_SUITE_P(
    Kinds, EditorSpawn,
    ::testing::Values(
        SpawnCase{viewer::PrimitiveKind::Box,     "box",     true},
        SpawnCase{viewer::PrimitiveKind::Sphere,  "sphere",  true},
        SpawnCase{viewer::PrimitiveKind::Capsule, "capsule", true},
        SpawnCase{viewer::PrimitiveKind::Plane,   "plane",   false}),
    [](const ::testing::TestParamInfo<SpawnCase>& info) {
        return std::string(info.param.tag);
    });
