// ---------------------------------------------------------------------------
// M8 T4 — render_frame_loop_smoke: the frame-loop SPINE smoke (PosePublisher +
// systems + Simulation), on a real cooked .nks scene.
//
// This verifies the coupled frame loop independently of the full
// render_physics_parity gate (T7), and splits its assertions by build config so
// the spine is exercised in the DEFAULT (Vulkan-off) build:
//
//   * G5 (render OFF) — NON-gated, runs in the default build. Two nk::Worlds are
//     cooked from the SAME scene; one is driven N frames through
//     Simulation::Frame() with rendering DISABLED, the other N times through a
//     bare world.Step() loop. Their final device state is downloaded and memcmp'd
//     BYTE-IDENTICAL. This proves the render-off Frame() path issues ZERO extra
//     device work (no DownloadField, no draw) vs a plain step loop — the G5
//     throughput-neutral bypass is a true bypass, not a no-op.
//
//   * G1 (render ON) — behind NK_BUILD_VULKAN_VALIDATION. A few Frame()s with a
//     real VulkanRasterRenderer attached; for sampled instances we independently
//     DownloadField the pose row and assert
//         instance.world_xform == downloaded_pose * cached_visual_local
//     (the exact publisher contract, Decision D1) bit-for-bit. Plus the stored
//     report's non_background_pixel_count > 0.
//
// Asset-gated: SKIP when the .nks / its imported MJCF+USD assets are absent.
//
// World construction reuses the canonical nk path (cf. h1_union_parity.cpp /
// h1_grasp_lift.cpp): Load(.nks) -> CookToModel -> nk::World. The RenderWorld is
// built from the SAME cook's Registry + SceneMap so its instances are bound to
// the very rows the World's Data fields carry (the G1 contract is then exact).
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <vector>

#include "math/transform.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "phi/scoped_device_guard.hpp"
#include "render/render_world.hpp"
#include "runtime/app/pose_publisher.hpp"
#include "runtime/app/simulation.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/format/nks.hpp"
#include "scene/scene_ir.hpp"

#ifdef NK_BUILD_VULKAN_VALIDATION
#include "render/raster/vulkan_raster_renderer.hpp"
#endif

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace app = nuka::runtime::app;
namespace render = nuka::render;
namespace cook = nuka::scene::cook;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr const char* kNksPath = "examples/scenes/h1_cup_table.nks";
constexpr const char* kH1Mjcf =
    ".nuka-assets/newton_assets/unitree_h1/mjcf/h1_with_hand.xml";
constexpr const char* kCupUsd =
    ".nuka-assets/newton_assets/manipulation_objects/cup/model_large.usda";

bool AssetsAvailable() {
    return std::filesystem::exists(kNksPath) &&
           std::filesystem::exists(kH1Mjcf) &&
           std::filesystem::exists(kCupUsd);
}

// Load the scene and STRIP shape mesh geometry so the generic CookToModel skips
// V-HACD (the heavy part). The spine smoke validates pose-publish + frame-loop
// flow, NOT collision-hull fidelity: the link/body pose Data fields the publisher
// reads are produced from the articulation/body rows regardless of collision
// decomposition, so clearing meshes leaves the G1/G5 surface intact while making
// the cook fast. (Same trick the compose gate uses: GetShapeMut + mesh clear.)
nuka::scene::SceneIR LoadLightScene() {
    nuka::scene::SceneIR scene = nuka::scene::nks::Load(kNksPath);
    for (size_t i = 0; i < scene.ShapeCount(); ++i) {
        auto& shape = scene.GetShapeMut(static_cast<nuka::scene::ShapeId>(i));
        shape.mesh_vertices.clear();
        shape.mesh_indices.clear();
    }
    return scene;
}

struct Backend {
    nphi::Device* dev = nullptr;
    nphi::Backend* backend = nullptr;
    bool ok() const { return dev != nullptr && backend != nullptr; }
};
Backend GetBackend() {
    static Backend b = [] {
        Backend r;
        r.dev = nphi::InitBestDevice();
        if (r.dev) r.backend = nphi::DeviceInitBackend(r.dev, nullptr);
        return r;
    }();
    return b;
}

nk::Pipeline::SolverConfig DefaultCfg() {
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[0] = 0.0f;
    cfg.gravity[1] = 0.0f;
    cfg.gravity[2] = -9.81f;
    return cfg;
}

// Download the WHOLE persistent live-state surface for env 0 (the fields a step
// mutates) so two worlds can be byte-compared. We compare the pose + velocity
// fields that carry the integrated state.
struct StateBlob {
    std::vector<Transform> link_pose;
    std::vector<Transform> body_pose;
    Transform base_pose = Transform::Identity();
    std::vector<float> q;
    std::vector<float> qdot;
};

StateBlob DownloadState(nk::World& w) {
    const nk::ModelCapacities& caps = w.GetModel().capacities;
    StateBlob s;
    const uint64_t kTf = sizeof(Transform);
    if (caps.links_per_env > 0u) {
        s.link_pose.resize(caps.links_per_env);
        EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::LinkPose,
                                              s.link_pose.data(),
                                              caps.links_per_env * kTf, 0));
        s.q.resize(caps.links_per_env);
        s.qdot.resize(caps.links_per_env);
        EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::Q, s.q.data(),
                                              caps.links_per_env * sizeof(float),
                                              0));
        EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::Qdot, s.qdot.data(),
                                              caps.links_per_env * sizeof(float),
                                              0));
    }
    if (caps.bodies_per_env > 0u) {
        s.body_pose.resize(caps.bodies_per_env);
        EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::BodyPose,
                                              s.body_pose.data(),
                                              caps.bodies_per_env * kTf, 0));
    }
    EXPECT_TRUE(w.GetData().DownloadField(nk::FieldId::BasePose, &s.base_pose, kTf,
                                          0));
    return s;
}

int CmpState(const StateBlob& a, const StateBlob& b) {
    if (a.link_pose.size() != b.link_pose.size()) return 1;
    if (a.body_pose.size() != b.body_pose.size()) return 2;
    if (!a.link_pose.empty() &&
        std::memcmp(a.link_pose.data(), b.link_pose.data(),
                    a.link_pose.size() * sizeof(Transform)) != 0)
        return 3;
    if (!a.body_pose.empty() &&
        std::memcmp(a.body_pose.data(), b.body_pose.data(),
                    a.body_pose.size() * sizeof(Transform)) != 0)
        return 4;
    if (std::memcmp(&a.base_pose, &b.base_pose, sizeof(Transform)) != 0) return 5;
    if (!a.q.empty() &&
        std::memcmp(a.q.data(), b.q.data(), a.q.size() * sizeof(float)) != 0)
        return 6;
    if (!a.qdot.empty() &&
        std::memcmp(a.qdot.data(), b.qdot.data(),
                    a.qdot.size() * sizeof(float)) != 0)
        return 7;
    return 0;
}

}  // namespace

// ===========================================================================
// G5 — render-OFF Frame() == bare Step() loop, byte-identical (NON-gated:
// runs in the DEFAULT build). Proves the render-off branch is a true bypass.
// ===========================================================================
TEST(RenderFrameLoopSmoke, RenderOffFrameMatchesBareStepByteIdentical) {
    if (!AssetsAvailable())
        GTEST_SKIP() << "h1_cup_table.nks / H1 / cup assets not present";
    Backend b = GetBackend();
    if (!b.ok()) GTEST_SKIP() << "no CUDA backend";

    const nuka::scene::SceneIR scene = LoadLightScene();
    const nk::Pipeline::SolverConfig cfg = DefaultCfg();
    constexpr uint32_t kFrames = 16u;

    // World A — driven through Simulation::Frame() with rendering DISABLED.
    StateBlob state_a;
    {
        cook::CookToModelResult cooked = cook::CookToModel(scene, 1);
        render::RenderWorld rw =
            render::BuildRenderWorld(scene.Ecs(), cooked.scene_map);
        nk::World w(std::move(cooked.model), 1u, b.dev, b.backend, cfg);
        ASSERT_TRUE(w.Ready());
        app::HostDownloadPublisher publisher;
        app::Simulation sim(w, publisher, std::move(rw));
        ASSERT_FALSE(sim.RenderEnabled());  // default = OFF (G5 bypass armed).
        for (uint32_t f = 0; f < kFrames; ++f) {
            EXPECT_TRUE(sim.Frame()) << "render-off Frame() reported an unhealthy step @"
                                     << f;
            EXPECT_FALSE(sim.LastFrameRendered());  // no draw on the bypass path.
        }
        state_a = DownloadState(w);
    }

    // World B — the SAME model cooked again, stepped via bare world.Step().
    StateBlob state_b;
    {
        cook::CookToModelResult cooked = cook::CookToModel(scene, 1);
        nk::World w(std::move(cooked.model), 1u, b.dev, b.backend, cfg);
        ASSERT_TRUE(w.Ready());
        for (uint32_t f = 0; f < kFrames; ++f) {
            EXPECT_TRUE(w.Step().AllOk()) << "bare Step() unhealthy @" << f;
        }
        state_b = DownloadState(w);
    }

    const int cmp = CmpState(state_a, state_b);
    std::printf("[FRAME-LOOP G5] render-off Frame() vs bare Step() over %u frames: "
                "state memcmp=%d (0 == byte-identical)\n",
                kFrames, cmp);
    EXPECT_EQ(cmp, 0)
        << "render-off Simulation::Frame() perturbed the step path (cmp=" << cmp
        << ") — the G5 bypass is NOT byte-neutral";
}

// ===========================================================================
// G1 — render-ON publish contract: world_xform == downloaded_pose ∘ visual_local
// (behind NK_BUILD_VULKAN_VALIDATION — needs a real Vulkan raster device).
// ===========================================================================
#ifdef NK_BUILD_VULKAN_VALIDATION
TEST(RenderFrameLoopSmoke, RenderOnPublishContractAndNonBackgroundPixels) {
    if (!AssetsAvailable())
        GTEST_SKIP() << "h1_cup_table.nks / H1 / cup assets not present";
    Backend b = GetBackend();
    if (!b.ok()) GTEST_SKIP() << "no CUDA backend";

    std::unique_ptr<render::VulkanRasterRenderer> renderer;
    try {
        renderer = std::make_unique<render::VulkanRasterRenderer>();
    } catch (const std::exception& e) {
        GTEST_SKIP() << "no Vulkan graphics device: " << e.what();
    }

    const nuka::scene::SceneIR scene = LoadLightScene();
    const nk::Pipeline::SolverConfig cfg = DefaultCfg();

    cook::CookToModelResult cooked = cook::CookToModel(scene, 1);
    render::RenderWorld rw =
        render::BuildRenderWorld(scene.Ecs(), cooked.scene_map);
    ASSERT_GT(rw.InstanceCount(), 0u);

    nk::World w(std::move(cooked.model), 1u, b.dev, b.backend, cfg);
    ASSERT_TRUE(w.Ready());

    app::HostDownloadPublisher publisher;
    app::Simulation sim(w, publisher, std::move(rw));
    render::RasterOptions opts;
    opts.width = 640;
    opts.height = 360;
    sim.EnableRendering(renderer.get(), opts);
    ASSERT_TRUE(sim.RenderEnabled());

    for (int f = 0; f < 4; ++f) {
        ASSERT_TRUE(sim.Frame());
        ASSERT_TRUE(sim.LastFrameRendered());
    }

    // -- G1 contract: independently re-download the pose row for sampled
    //    instances and assert world_xform == downloaded_pose * cached_visual_local.
    const nk::ModelCapacities& caps = w.GetModel().capacities;
    const uint64_t kTf = sizeof(Transform);
    const render::RenderWorld& live = sim.GetRenderWorld();
    uint32_t checked = 0u;
    double max_resid = 0.0;
    for (const render::RenderInstance& inst : live.instances) {
        Transform fk;
        bool got = false;
        if (inst.pose_source.kind == render::PoseSource::Kind::Link &&
            caps.links_per_env > 0u && inst.pose_source.row < caps.links_per_env) {
            ASSERT_TRUE(w.GetData().DownloadField(nk::FieldId::LinkPose, &fk, kTf,
                                                  inst.pose_source.row * kTf));
            got = true;
        } else if (inst.pose_source.kind == render::PoseSource::Kind::Body &&
                   caps.bodies_per_env > 0u &&
                   inst.pose_source.row < caps.bodies_per_env) {
            ASSERT_TRUE(w.GetData().DownloadField(nk::FieldId::BodyPose, &fk, kTf,
                                                  inst.pose_source.row * kTf));
            got = true;
        }
        if (!got) continue;  // Static instance: keeps bind pose, not a G1 row.
        const Transform expect = fk * inst.cached_visual_local;
        const double dp = (inst.world_xform.position - expect.position).Length();
        const Vec3 dq{inst.world_xform.rotation.x - expect.rotation.x,
                      inst.world_xform.rotation.y - expect.rotation.y,
                      inst.world_xform.rotation.z - expect.rotation.z};
        const double dqw =
            std::fabs(inst.world_xform.rotation.w - expect.rotation.w);
        max_resid = std::max({max_resid, dp, (double)dq.Length(), dqw});
        EXPECT_EQ(std::memcmp(&inst.world_xform, &expect, sizeof(Transform)), 0)
            << "G1 compose mismatch on instance row " << inst.pose_source.row;
        if (++checked >= 16u) break;
    }
    std::printf("[FRAME-LOOP G1] publish contract checked on %u instances, "
                "max residual=%.3e (bit-exact expected)\n",
                checked, max_resid);
    EXPECT_GT(checked, 0u) << "no Link/Body-bound instance found to check G1";

    const render::VulkanOffscreenReport& rep = sim.LatestReport();
    std::printf("[FRAME-LOOP G3] non_background_pixel_count=%zu (%ux%u)\n",
                rep.non_background_pixel_count, rep.width, rep.height);
    EXPECT_GT(rep.non_background_pixel_count, 0u)
        << "render produced an all-background frame";
}
#endif  // NK_BUILD_VULKAN_VALIDATION
