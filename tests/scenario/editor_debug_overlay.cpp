// ---------------------------------------------------------------------------
// Read-only physics debug-overlay data-path gate.
//
// Cooks a real EditorScene, evolves it, and asserts the DebugOverlay emits exactly
// one collider proxy per cooked PRIMITIVE shape -- posed at the shape's live frame
// (BodyPose[i] for a body-attached shape, world frame for implicit ground) and
// coloured by body kind (green dynamic / magenta static) -- and one contact marker
// per active solver contact point. A self-contained two-box scene (static slab +
// dynamic box, plus the cook's implicit ground plane) pins both colours + a known
// box/box contact; go2 / go2_field add real-robot scale + heightfield foot contacts.
//
// SENSITIVE: proxy poses are compared to a FRESH BodyPose download after stepping
// (so they cannot be the authored rest pose); the contact count is cross-checked
// against the INDEPENDENT per-env ContactCount watermark (not the per-slot
// enumeration the overlay itself uses); toggling the overlay off must return the
// RenderWorld to its exact real instance set.
//
// HOST-ONLY: cooks on the phi device (SKIP when none); no Vulkan / window.
// ---------------------------------------------------------------------------

#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "phi/backend.hpp"
#include "render/render_world.hpp"
#include "runtime/app/viewer/debug_overlay.hpp"
#include "runtime/app/viewer/editor_scene.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace viewer = nuka::runtime::app::viewer;
namespace render = nuka::render;
using nuka::nk::FieldId;
using nuka::math::Transform;
using nuka::math::Vec3;

namespace {

constexpr float kDt  = 1.0f / 240.0f;
constexpr float kTol = 1.0e-5f;  // proxy pose == downloaded BodyPose: byte-for-byte

bool IsPrimitiveKind(uint32_t k) { return k <= 3u; }  // sphere/capsule/box/plane

class TempDir {
public:
    TempDir() {
        char tmpl[] = "/tmp/editor_overlay_XXXXXX";
        char* p = ::mkdtemp(tmpl);
        if (!p) throw std::runtime_error("mkdtemp failed");
        path_ = p;
    }
    ~TempDir() { std::error_code ec; fs::remove_all(path_, ec); }
    std::string File(const std::string& n) const { return (path_ / n).string(); }
private:
    fs::path path_;
};

std::string Author(const TempDir& tmp, const std::string& name, const std::string& json) {
    const std::string p = tmp.File(name);
    std::FILE* f = std::fopen(p.c_str(), "wb");
    if (!f) throw std::runtime_error("author open failed");
    std::fwrite(json.data(), 1, json.size(), f);
    std::fclose(f);
    return p;
}

// One box body node (free root). is_static slabs carry inv_mass 0; the collision
// block mirrors the validated .nks schema.
std::string BoxBody(const std::string& name, float z, float hx, float hy, float hz,
                    bool is_static) {
    char buf[2048];
    std::snprintf(buf, sizeof(buf), R"({
      "name": "%s",
      "children": [{
        "name": "%s/body",
        "transform": { "pos": [0.0,0.0,%f], "quat": [1.0,0.0,0.0,0.0] },
        "rigid_body": {
          "parent_id": 4294967295,
          "inertial": { "pos": [0.0,0.0,0.0], "quat": [1.0,0.0,0.0,0.0] },
          "mass": 1.0, "inertia": [1.0,1.0,1.0], "is_static": %s
        },
        "children": [{
          "name": "%s/collision",
          "collision_shape": {
            "type": "box",
            "local": { "pos": [0.0,0.0,0.0], "quat": [1.0,0.0,0.0,0.0] },
            "half_extents": [%f,%f,%f],
            "radius": 0.1, "half_height": 0.1,
            "material_id": 0, "contype": 1, "conaffinity": 1, "collision_group": 0,
            "solref": [0.02,1.0], "solimp": [0.9,0.95,0.001,0.5,2.0],
            "friction_mu": 0.8, "priority": 0, "solmix": 1.0,
            "margin": 0.0, "gap": 0.0, "condim": 3,
            "decompose_mode": "auto", "decompose_max_pieces": 32
          },
          "children": []
        }]
      }]
    })", name.c_str(), name.c_str(), z, is_static ? "true" : "false",
        name.c_str(), hx, hy, hz);
    return buf;
}

// Two-box scene: a large static slab (top at z=0.1) under a small dynamic box that
// drops a hair and rests on it. The cook also adds an implicit ground plane at z=0,
// so the colliders are slab(static) + box(dynamic) + ground-plane(static).
std::string TwoBoxScene() {
    return std::string(R"({
  "nks_version": 1,
  "physics_materials": { "m": { "static_friction": 0.8, "dynamic_friction": 0.8 } },
  "render_materials": { "m": { "base_color": [0.8,0.8,0.8,1.0], "metallic": 0.0, "roughness": 0.5 } },
  "tree": [
)") + BoxBody("slab", 0.0f, 1.0f, 1.0f, 0.1f, /*is_static=*/true) + ",\n" +
        BoxBody("box", 0.32f, 0.2f, 0.2f, 0.2f, /*is_static=*/false) +
        R"(
  ],
  "media": [],
  "sensors": []
})";
}

float TformDiff(const Transform& a, const Transform& b) {
    float m = 0.0f;
    m = std::max(m, std::fabs(a.position.x - b.position.x));
    m = std::max(m, std::fabs(a.position.y - b.position.y));
    m = std::max(m, std::fabs(a.position.z - b.position.z));
    m = std::max(m, std::fabs(a.rotation.w - b.rotation.w));
    m = std::max(m, std::fabs(a.rotation.x - b.rotation.x));
    m = std::max(m, std::fabs(a.rotation.y - b.rotation.y));
    m = std::max(m, std::fabs(a.rotation.z - b.rotation.z));
    return m;
}

// The seam's view of one cooked shape: is it a drawn primitive, is it static, and
// where does its proxy belong. Built from the SAME fields the overlay reads.
struct Expected { bool primitive = false; bool is_static = false; Transform pose; };

std::vector<Expected> ExpectedColliders(nuka::nk::World& w) {
    const auto& rows = w.GetModel().shape_table_rows;
    const auto& b2l = w.GetModel().body_to_link;                    // host; empty w/o articulation
    const auto& lgl = w.GetModel().articulation.link_geom_local;    // host; per template link
    const uint32_t bodies = w.GetModel().capacities.bodies_per_env;
    const uint32_t links  = w.GetModel().capacities.links_per_env;
    std::vector<Transform> poses(bodies ? bodies : 1u, Transform::Identity());
    std::vector<Transform> lposes(links ? links : 1u, Transform::Identity());
    std::vector<float> invm(bodies ? bodies : 1u, 0.0f);
    if (bodies > 0u) {
        w.GetData().DownloadField(FieldId::BodyPose, poses.data(),
                                  static_cast<uint64_t>(bodies) * sizeof(Transform), 0);
        w.GetData().DownloadField(FieldId::BodyInvMass, invm.data(),
                                  static_cast<uint64_t>(bodies) * sizeof(float), 0);
    }
    if (links > 0u) {
        w.GetData().DownloadField(FieldId::LinkPose, lposes.data(),
                                  static_cast<uint64_t>(links) * sizeof(Transform), 0);
    }
    std::vector<Expected> out;
    out.reserve(rows.size());
    for (uint32_t i = 0; i < rows.size(); ++i) {
        Expected e;
        e.primitive = IsPrimitiveKind(rows[i].kind);
        const bool has_body = (i < bodies);
        const bool is_link = has_body && i < b2l.size() && b2l[i] != ~uint32_t(0);
        e.is_static = has_body ? (!is_link && invm[i] == 0.0f) : true;
        // A link reads LinkPose o link_geom_local; a free body reads BodyPose.
        if (is_link) {
            const uint32_t l = b2l[i];
            if (l < lposes.size()) e.pose = lposes[l];
            if (l < lgl.size()) e.pose = e.pose * lgl[l];
        } else if (has_body) {
            e.pose = poses[i];
        }
        out.push_back(e);
    }
    return out;
}

bool IsGreen(const float c[4])   { return c[1] > 0.6f && c[0] < 0.4f && c[2] < 0.4f; }
bool IsMagenta(const float c[4]) { return c[0] > 0.6f && c[2] > 0.6f && c[1] < 0.4f; }

class EditorOverlay : public ::testing::Test {
protected:
    void SetUp() override {
        dev_ = nuka::phi::InitBestDevice();
        if (!dev_) GTEST_SKIP() << "no phi device";
        backend_ = nuka::phi::DeviceInitBackend(dev_, nullptr);
        if (!backend_) GTEST_SKIP() << "no phi backend";
        tmp_ = std::make_unique<TempDir>();
    }
    nuka::phi::Device*  dev_ = nullptr;
    nuka::phi::Backend* backend_ = nullptr;
    std::unique_ptr<TempDir> tmp_;
};

// Colliders: one proxy per primitive shape, posed at its live frame, coloured by
// kind. The two-box scene pins slab(magenta) + box(green) + ground-plane(magenta).
TEST_F(EditorOverlay, CollidersMatchCookedPrimitives) {
    const std::string scene = Author(*tmp_, "twobox.nks", TwoBoxScene());
    auto es = viewer::LoadEditorScene(scene, dev_, backend_, kDt);
    ASSERT_NE(es, nullptr) << "two-box cook failed";
    nuka::nk::World& w = *es->world;

    for (int i = 0; i < 150; ++i) w.Step();  // box drops + settles on the slab

    const std::vector<Expected> exp = ExpectedColliders(w);
    uint32_t prims = 0u, statics = 0u, dyns = 0u;
    for (const Expected& e : exp)
        if (e.primitive) { ++prims; (e.is_static ? statics : dyns)++; }
    std::printf("[overlay] two-box: shapes=%zu primitives=%u static=%u dynamic=%u\n",
                exp.size(), prims, statics, dyns);
    std::fflush(stdout);
    ASSERT_EQ(prims, 3u);    // slab + box + implicit ground plane
    ASSERT_EQ(statics, 2u);  // slab + ground plane
    ASSERT_EQ(dyns, 1u);     // the falling box

    render::RenderWorld& rw = es->sim->GetRenderWorld();
    const size_t real = rw.instances.size();
    viewer::DebugOverlay overlay;
    overlay.Rebuild(w, rw, 0u, /*colliders=*/true, /*contacts=*/false);

    EXPECT_EQ(overlay.LastColliderCount(), prims);
    EXPECT_EQ(overlay.LastSkippedShapes(), 0u);  // every shape is a primitive
    EXPECT_EQ(rw.instances.size(), real) << "the real instance set is never touched";
    ASSERT_EQ(rw.debug_instances.size(), prims) << "one proxy per collider in the debug channel";

    // The proxies are in cooked-primitive order; each must sit at its expected
    // frame and carry the kind-correct colour.
    uint32_t green = 0u, magenta = 0u, moved = 0u;
    size_t at = 0u;
    for (size_t i = 0; i < exp.size() && at < rw.debug_instances.size(); ++i) {
        if (!exp[i].primitive) continue;
        const render::RenderInstance& inst = rw.debug_instances[at++];
        EXPECT_EQ(inst.entity.index, ~uint32_t(0))  // sentinel: pick / tree never see it
            << "debug proxy must carry the non-entity sentinel";
        EXPECT_LT(TformDiff(inst.world_xform, exp[i].pose), kTol)
            << "proxy " << i << " not at its live frame";
        ASSERT_LT(inst.render_material_id, rw.materials.size());
        const float* c = rw.materials[inst.render_material_id].base_color;
        if (exp[i].is_static) { EXPECT_TRUE(IsMagenta(c)) << "static must be magenta"; ++magenta; }
        else                  { EXPECT_TRUE(IsGreen(c))   << "dynamic must be green";  ++green; }
        if (!exp[i].is_static && std::fabs(exp[i].pose.position.z - 0.32f) > 1.0e-3f)
            ++moved;  // the dynamic box left its authored z -> proxy tracks the LIVE pose
    }
    EXPECT_EQ(magenta, 2u);
    EXPECT_EQ(green, 1u);
    EXPECT_EQ(moved, 1u) << "dynamic box did not evolve -- liveness not exercised";
}

// Contacts: one marker per active solver contact point, at the buffer positions,
// count-matched to the INDEPENDENT per-env ContactCount watermark.
TEST_F(EditorOverlay, ContactsMatchSolverBuffer) {
    const std::string scene = Author(*tmp_, "twobox.nks", TwoBoxScene());
    auto es = viewer::LoadEditorScene(scene, dev_, backend_, kDt);
    ASSERT_NE(es, nullptr);
    nuka::nk::World& w = *es->world;

    for (int i = 0; i < 200; ++i) w.Step();  // settle so the contact manifold is live

    uint32_t cc = 0u;
    w.GetData().DownloadField(FieldId::ContactCount, &cc, sizeof(uint32_t), 0);

    // Independently enumerate the per-slot manifold the overlay reads.
    const uint32_t max_c = w.GetModel().capacities.max_contacts_per_env;
    std::vector<uint32_t> ucount(max_c, 0u);
    std::vector<Vec3> upoint(static_cast<size_t>(max_c) * 4u, Vec3{0, 0, 0});
    w.GetData().DownloadField(FieldId::UcontactCount, ucount.data(),
                              static_cast<uint64_t>(max_c) * sizeof(uint32_t), 0);
    w.GetData().DownloadField(FieldId::UcontactPoint, upoint.data(),
                              static_cast<uint64_t>(max_c) * 4u * sizeof(Vec3), 0);
    uint32_t summed = 0u;
    for (uint32_t s = 0; s < max_c; ++s) summed += std::min<uint32_t>(ucount[s], 4u);
    std::printf("[overlay] two-box contacts: ContactCount=%u sum(Ucontact)=%u\n", cc, summed);
    std::fflush(stdout);
    ASSERT_GT(cc, 0u) << "the box must rest on the slab (contacts present)";

    render::RenderWorld& rw = es->sim->GetRenderWorld();
    const size_t real = rw.instances.size();
    viewer::DebugOverlay overlay;
    overlay.Rebuild(w, rw, 0u, /*colliders=*/false, /*contacts=*/true);

    EXPECT_EQ(overlay.LastContactCount(), cc)
        << "marker count must match the solver's contact watermark";
    EXPECT_EQ(overlay.LastContactCount(), summed);
    EXPECT_EQ(rw.instances.size(), real) << "the real instance set is never touched";
    ASSERT_EQ(rw.debug_instances.size(), overlay.LastContactCount());

    // Every marker sits at a real manifold point and near the slab/box interface
    // (slab top z=0.1; markers within a small penetration band of it).
    for (size_t k = 0; k < rw.debug_instances.size(); ++k) {
        const Vec3 p = rw.debug_instances[k].world_xform.position;
        bool found = false;
        for (uint32_t s = 0; s < max_c && !found; ++s)
            for (uint32_t e = 0; e < std::min<uint32_t>(ucount[s], 4u); ++e) {
                const Vec3& q = upoint[static_cast<size_t>(s) * 4u + e];
                if (std::fabs(p.x - q.x) < kTol && std::fabs(p.y - q.y) < kTol &&
                    std::fabs(p.z - q.z) < kTol) found = true;
            }
        EXPECT_TRUE(found) << "marker not at any contact-buffer point";
        EXPECT_NEAR(p.z, 0.1f, 0.05f) << "marker not near the box/slab interface";
    }
}

// Off-by-default + clear: toggling the overlay off must restore the EXACT real
// instance set (GATE-B: the default path leaves the RenderWorld untouched).
TEST_F(EditorOverlay, OverlayOffRestoresRealInstances) {
    const std::string scene = Author(*tmp_, "twobox.nks", TwoBoxScene());
    auto es = viewer::LoadEditorScene(scene, dev_, backend_, kDt);
    ASSERT_NE(es, nullptr);
    nuka::nk::World& w = *es->world;
    for (int i = 0; i < 150; ++i) w.Step();

    render::RenderWorld& rw = es->sim->GetRenderWorld();
    const size_t real = rw.instances.size();
    viewer::DebugOverlay overlay;

    overlay.Rebuild(w, rw, 0u, false, false);             // off: nothing emitted
    EXPECT_EQ(rw.instances.size(), real);
    EXPECT_TRUE(rw.debug_instances.empty());
    overlay.Rebuild(w, rw, 0u, true, true);               // on: the debug channel grows
    EXPECT_EQ(rw.instances.size(), real) << "the real instance set is never touched";
    EXPECT_GT(rw.debug_instances.size(), 0u);
    overlay.Rebuild(w, rw, 0u, false, false);             // off again: clears the channel
    EXPECT_EQ(rw.instances.size(), real);
    EXPECT_TRUE(rw.debug_instances.empty()) << "debug channel not cleared on toggle-off";
}

// Real-robot scale: a go2 cooks to many DYNAMIC primitive link colliders (green),
// each aligned to its live link BodyPose. Any static collider must be the cook's
// implicit ground (shape index >= bodies) -- never a misclassified articulation link.
TEST_F(EditorOverlay, Go2ManyCollidersAligned) {
    const std::string scene = "examples/scenes/go2.nks";
    if (!fs::exists(scene)) GTEST_SKIP() << "go2.nks missing";
    auto es = viewer::LoadEditorScene(scene, dev_, backend_, kDt);
    ASSERT_NE(es, nullptr);
    nuka::nk::World& w = *es->world;
    for (int i = 0; i < 60; ++i) w.Step();

    const uint32_t bodies = w.GetModel().capacities.bodies_per_env;
    const std::vector<Expected> exp = ExpectedColliders(w);
    uint32_t prims = 0u, statics = 0u;
    for (const Expected& e : exp) if (e.primitive) { ++prims; if (e.is_static) ++statics; }
    ASSERT_GT(prims, 10u) << "a go2 has many link colliders";

    render::RenderWorld& rw = es->sim->GetRenderWorld();
    const size_t real = rw.instances.size();
    viewer::DebugOverlay overlay;
    overlay.Rebuild(w, rw, 0u, true, false);
    EXPECT_EQ(overlay.LastColliderCount(), prims);
    EXPECT_EQ(rw.instances.size(), real) << "the real instance set is never touched";
    ASSERT_EQ(rw.debug_instances.size(), prims);

    size_t at = 0u;
    float worst = 0.0f;
    uint32_t green = 0u;
    Vec3 lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f};
    for (size_t i = 0; i < exp.size() && at < rw.debug_instances.size(); ++i) {
        if (!exp[i].primitive) continue;
        const render::RenderInstance& inst = rw.debug_instances[at++];
        worst = std::max(worst, TformDiff(inst.world_xform, exp[i].pose));
        const float* c = rw.materials[inst.render_material_id].base_color;
        if (exp[i].is_static) {
            EXPECT_GE(i, bodies)
                << "a go2 static collider must be implicit ground, not a misclassified link";
            EXPECT_TRUE(IsMagenta(c));
        } else {
            ++green;
            EXPECT_TRUE(IsGreen(c)) << "a go2 link collider must be dynamic (green)";
            const Vec3 p = inst.world_xform.position;
            lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y); lo.z = std::min(lo.z, p.z);
            hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y); hi.z = std::max(hi.z, p.z);
        }
    }
    // The link colliders must SPAN the dog's body, not collapse to one point -- a
    // regression to the BodyPose-zero bug would pile every link proxy at the origin.
    const float spread =
        std::sqrt((hi.x - lo.x) * (hi.x - lo.x) + (hi.y - lo.y) * (hi.y - lo.y) +
                  (hi.z - lo.z) * (hi.z - lo.z));
    std::printf("[overlay] go2 colliders=%u static=%u green=%u worst pose residual=%.2e spread=%.3f\n",
                prims, statics, green, worst, spread);
    std::fflush(stdout);
    EXPECT_LT(worst, kTol) << "go2 proxies must align to their live link poses";
    EXPECT_GT(spread, 0.3f) << "link colliders collapsed to a point (BodyPose-zero regression)";
    EXPECT_EQ(green, prims - statics) << "every dynamic primitive collider is green";
    EXPECT_GT(green, 10u) << "the dog's link colliders are dynamic";
}

// Heightfield foot contacts on the general path: a go2 on terrain produces contacts
// (and the heightfield itself is honestly SKIPPED, not drawn as a primitive).
TEST_F(EditorOverlay, Go2FieldFootContacts) {
    const std::string scene = "examples/scenes/go2_field.nks";
    if (!fs::exists(scene)) GTEST_SKIP() << "go2_field.nks missing";
    auto es = viewer::LoadEditorScene(scene, dev_, backend_, kDt);
    ASSERT_NE(es, nullptr);
    nuka::nk::World& w = *es->world;
    for (int i = 0; i < 120; ++i) w.Step();  // feet settle onto the terrain

    uint32_t cc = 0u;
    w.GetData().DownloadField(FieldId::ContactCount, &cc, sizeof(uint32_t), 0);

    render::RenderWorld& rw = es->sim->GetRenderWorld();
    viewer::DebugOverlay overlay;
    overlay.Rebuild(w, rw, 0u, true, true);
    std::printf("[overlay] go2_field: ContactCount=%u markers=%u colliders=%u skipped=%u\n",
                cc, overlay.LastContactCount(), overlay.LastColliderCount(),
                overlay.LastSkippedShapes());
    std::fflush(stdout);
    EXPECT_EQ(overlay.LastContactCount(), cc);
    EXPECT_GT(overlay.LastSkippedShapes(), 0u) << "the heightfield must be skipped, not drawn";
}

}  // namespace
