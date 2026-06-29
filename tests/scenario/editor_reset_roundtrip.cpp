// ---------------------------------------------------------------------------
// Editor true-physics-Reset round-trip gate.
//
// Cooks a real EditorScene, records the authored initial physics state, OVERWRITES
// a live drive target (the Drive panel's edit), STEPS the world so the state
// genuinely evolves (asserts it DID), then calls the single reset entry point
// ResetEditorScenePhysics and asserts the world returns to the authored qpos / root
// pose / particle positions EXACTLY, with qdot / particle velocities zeroed and the
// drive targets re-seeded to authored. The "evolved" guard makes it SENSITIVE: a
// no-op reset, or one that restores only a subset (pose-but-not-velocity, joints-but-
// not-root, leaves the drive edit), fails a specific assertion.
//
// ONE general path coverage: the SAME reset is exercised across the demo media that
// span every restore family -- soft (soft_ball.nks), cloth, fluid, and articulation /
// multi-dog (go2.nks) -- because World::Reset round-trips articulation + body +
// particle slices uniformly.
//
// HOST-ONLY: cooks on the phi device (SKIP when none); no Vulkan / window.
// ---------------------------------------------------------------------------

#include "nk/model/generated/field_ids.hpp"
#include "phi/backend.hpp"
#include "runtime/app/simulation.hpp"
#include "runtime/app/viewer/editor_scene.hpp"
#include "scene/format/nks.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace viewer = nuka::runtime::app::viewer;
using nuka::nk::FieldId;
using nuka::math::Transform;
using nuka::math::Vec3;

namespace {

constexpr float kDt    = 1.0f / 240.0f;
constexpr int   kSteps = 160;     // enough for a Go2 to sag / particles to fall
constexpr float kMoved = 1.0e-3f; // the state must have evolved by at least this
constexpr float kTol   = 1.0e-4f; // a snapshot restore is a D2D copy -> ~exact

class TempDir {
public:
    TempDir() {
        char tmpl[] = "/tmp/editor_reset_XXXXXX";
        char* p = ::mkdtemp(tmpl);
        if (!p) throw std::runtime_error("mkdtemp failed");
        path_ = p;
    }
    ~TempDir() { std::error_code ec; fs::remove_all(path_, ec); }
    std::string File(const std::string& n) const { return (path_ / n).string(); }
private:
    fs::path path_;
};

// Download env 0's slice of a float field (count scalars from offset 0).
std::vector<float> DlF(nuka::nk::World& w, FieldId id, uint32_t count) {
    std::vector<float> out(count, 0.0f);
    if (count > 0u)
        w.GetData().DownloadField(id, out.data(),
                                  static_cast<uint64_t>(count) * sizeof(float), 0);
    return out;
}
std::vector<Transform> DlT(nuka::nk::World& w, FieldId id, uint32_t count) {
    std::vector<Transform> out(count);
    if (count > 0u)
        w.GetData().DownloadField(id, out.data(),
                                  static_cast<uint64_t>(count) * sizeof(Transform), 0);
    return out;
}
std::vector<Vec3> DlV(nuka::nk::World& w, FieldId id, uint32_t count) {
    std::vector<Vec3> out(count);
    if (count > 0u)
        w.GetData().DownloadField(id, out.data(),
                                  static_cast<uint64_t>(count) * sizeof(Vec3), 0);
    return out;
}

float MaxAbs(const std::vector<float>& v) {
    float m = 0.0f; for (float x : v) m = std::max(m, std::fabs(x)); return m;
}
float MaxAbsV(const std::vector<Vec3>& v) {
    float m = 0.0f;
    for (const Vec3& p : v)
        m = std::max(m, std::max(std::fabs(p.x), std::max(std::fabs(p.y), std::fabs(p.z))));
    return m;
}
float Diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f; for (size_t i = 0; i < a.size() && i < b.size(); ++i)
        m = std::max(m, std::fabs(a[i] - b[i])); return m;
}
float DiffV(const std::vector<Vec3>& a, const std::vector<Vec3>& b) {
    float m = 0.0f; for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        m = std::max(m, std::fabs(a[i].x - b[i].x));
        m = std::max(m, std::fabs(a[i].y - b[i].y));
        m = std::max(m, std::fabs(a[i].z - b[i].z));
    } return m;
}
float DiffT(const std::vector<Transform>& a, const std::vector<Transform>& b) {
    float m = 0.0f; for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        const Transform& x = a[i]; const Transform& y = b[i];
        m = std::max(m, std::fabs(x.position.x - y.position.x));
        m = std::max(m, std::fabs(x.position.y - y.position.y));
        m = std::max(m, std::fabs(x.position.z - y.position.z));
        m = std::max(m, std::fabs(x.rotation.w - y.rotation.w));
        m = std::max(m, std::fabs(x.rotation.x - y.rotation.x));
        m = std::max(m, std::fabs(x.rotation.y - y.rotation.y));
        m = std::max(m, std::fabs(x.rotation.z - y.rotation.z));
    } return m;
}

// Cook `nks_path`, evolve it, reset it, and assert the authored physics state returns.
// `dev`/`backend` are a live phi device. Skips (not fails) when the cook yields nothing
// to evolve so a missing optional asset never red-flags the suite.
void CheckResetRoundTrip(nuka::phi::Device* dev, nuka::phi::Backend* backend,
                         const std::string& nks_path) {
    auto es = viewer::LoadEditorScene(nks_path, dev, backend, kDt);
    ASSERT_NE(es, nullptr) << "load/cook failed: " << nks_path;
    nuka::nk::World& w = *es->world;
    const uint32_t L = es->caps.links_per_env;
    const uint32_t K = es->caps.articulations_per_env;
    const uint32_t P = es->caps.particles_per_env;
    ASSERT_TRUE(L > 0u || P > 0u)
        << "scene has neither articulation nor particles (nothing to reset): " << nks_path;

    // Authored initial state (== the construction-time snapshot), env 0.
    const std::vector<float>     q0    = DlF(w, FieldId::Q, L);
    const std::vector<float>     drv0  = DlF(w, FieldId::DriveTarget, L);
    const std::vector<Transform> base0 = DlT(w, FieldId::BasePose, K);
    const std::vector<Vec3>      pos0  = DlV(w, FieldId::ParticlePos, P);

    // Simulate a live Drive-panel edit: overwrite the drive targets (env 0) so the
    // reset has a non-trivial drive state to restore.
    if (L > 0u && !drv0.empty()) {
        std::vector<float> drvp = drv0;
        for (float& x : drvp) x += 0.37f;
        w.GetData().UploadField(FieldId::DriveTarget, drvp.data(),
                                static_cast<uint64_t>(L) * sizeof(float), 0);
    }

    // Evolve: gravity sags the robot / drops the particles.
    for (int i = 0; i < kSteps; ++i) w.Step();

    const std::vector<float>     q1    = DlF(w, FieldId::Q, L);
    const std::vector<Transform> base1 = DlT(w, FieldId::BasePose, K);
    const std::vector<Vec3>      pos1  = DlV(w, FieldId::ParticlePos, P);
    const float moved = std::max({Diff(q0, q1), DiffT(base0, base1), DiffV(pos0, pos1)});
    ASSERT_GT(moved, kMoved) << "state did not evolve under stepping: " << nks_path;

    // The single reset entry point the transport button + script on_reset both call.
    ASSERT_TRUE(viewer::ResetEditorScenePhysics(*es)) << "reset failed: " << nks_path;

    const std::vector<float>     q2    = DlF(w, FieldId::Q, L);
    const std::vector<float>     qd2   = DlF(w, FieldId::Qdot, L);
    const std::vector<float>     drv2  = DlF(w, FieldId::DriveTarget, L);
    const std::vector<Transform> base2 = DlT(w, FieldId::BasePose, K);
    const std::vector<Vec3>      pos2  = DlV(w, FieldId::ParticlePos, P);
    const std::vector<Vec3>      vel2  = DlV(w, FieldId::ParticleVel, P);

    EXPECT_LT(Diff(q0, q2), kTol)     << "qpos not restored to authored: " << nks_path;
    EXPECT_LT(DiffT(base0, base2), kTol) << "root pose not restored: " << nks_path;
    EXPECT_LT(MaxAbs(qd2), kTol)      << "qdot not zeroed: " << nks_path;
    EXPECT_LT(Diff(drv0, drv2), kTol) << "drive targets not re-seeded to authored: " << nks_path;
    EXPECT_LT(DiffV(pos0, pos2), kTol) << "particle positions not restored: " << nks_path;
    EXPECT_LT(MaxAbsV(vel2), kTol)    << "particle velocities not zeroed: " << nks_path;
}

class EditorReset : public ::testing::Test {
protected:
    void SetUp() override {
        dev_ = nuka::phi::InitBestDevice();
        if (!dev_) GTEST_SKIP() << "no phi device";
        backend_ = nuka::phi::DeviceInitBackend(dev_, nullptr);
        if (!backend_) GTEST_SKIP() << "no phi backend";
        tmp_ = std::make_unique<TempDir>();
    }
    std::string Author(const std::string& name, const std::string& json) {
        const std::string p = tmp_->File(name);
        std::FILE* f = std::fopen(p.c_str(), "wb");
        if (!f) throw std::runtime_error("author open failed");
        std::fwrite(json.data(), 1, json.size(), f);
        std::fclose(f);
        return p;
    }
    nuka::phi::Device*  dev_ = nullptr;
    nuka::phi::Backend* backend_ = nullptr;
    std::unique_ptr<TempDir> tmp_;
};

// Articulation + multi-dog: qpos + floating-base root + drive targets all restore.
TEST_F(EditorReset, Go2Articulation) {
    const std::string scene = "examples/scenes/go2.nks";
    if (!fs::exists(scene)) GTEST_SKIP() << "go2.nks missing";
    CheckResetRoundTrip(dev_, backend_, scene);
}

// Soft body (XPBD tet): particle positions + velocities restore.
TEST_F(EditorReset, SoftTet) {
    const std::string scene = "examples/scenes/soft_ball.nks";
    if (!fs::exists(scene)) GTEST_SKIP() << "soft_ball.nks missing";
    CheckResetRoundTrip(dev_, backend_, scene);
}

// Cloth (XPBD grid): the free drape falls, then its particle state restores.
TEST_F(EditorReset, Cloth) {
    const std::string scene = Author("cloth.nks", R"({
  "nks_version": 1,
  "physics_materials": {},
  "render_materials": {},
  "tree": [],
  "media": [
    { "name": "drape", "kind": "cloth", "method": "xpbd",
      "cloth_grid": { "nx": 8, "ny": 8, "spacing": 0.05, "origin": [0.0, 0.0, 0.9], "free": true },
      "xpbd": { "particle_mass": 0.012, "friction": 1.0, "bend_alpha": 0.09, "iters": 20 } }
  ]
})");
    CheckResetRoundTrip(dev_, backend_, scene);
}

// Fluid (PBF box): the pool of particles falls, then its particle state restores.
TEST_F(EditorReset, Fluid) {
    const std::string scene = Author("fluid.nks", R"({
  "nks_version": 1,
  "physics_materials": {},
  "render_materials": {},
  "tree": [],
  "media": [
    { "name": "pool", "kind": "fluid", "method": "pbf",
      "fluid_box": { "min": [-0.1, -0.1, 0.5], "max": [0.1, 0.1, 0.7], "spacing": 0.04 },
      "pbf": { "rest_density": 1000.0, "support_scale": 2.0, "iters": 4, "friction": 0.1 } }
  ]
})");
    CheckResetRoundTrip(dev_, backend_, scene);
}

}  // namespace
