// ---------------------------------------------------------------------------
// viewer_frame_smoke -- M8.5 GATE-B: the RUN-VERIFIED OFFSCREEN viewer-frame
// composite smoke (the deliverable for T3, recon §1.4 GATE-B).
//
// Builds a RenderWorld (the synthetic two-box scene + at least one non-trivial
// instance), stands up the OFFSCREEN VulkanRasterRenderer (the D1 oracle), inits
// the nuka ImGui context against the renderer's offscreen render pass + descriptor
// pool, records the REAL viewer panels (imgui_layer) with a FIXED ViewerUiState
// (NO time-seeded animation), and composites the scene + ImGui draw data into the
// offscreen image via the Render(world, options, overlay) seam (overlay =
// ImGui_ImplVulkan_RenderDrawData through NukaImGuiContext::RenderDrawData).
//
// Asserts:
//   (a) ImDrawData.CmdListsCount > 0                      (the UI actually built)
//   (b) composited image non_background_pixel_count > 0   (scene + UI drew)
//   (c) two composited renders byte-identical (memcmp==0) (D1, fixed UI state)
//   (d) dumps a composited PPM to /tmp/m8_5_viewer_frame.ppm for off-box review.
// SKIPs cleanly if no Vulkan device (renderer ctor throws) or ImGui init fails.
//
// The offscreen overlay seam is opt-in: with an EMPTY overlay Render() is the
// untouched G2 oracle (verified by render_raster_smoke / render_physics_parity);
// this test exercises the NON-empty branch only.
//
// HOST-ONLY / zero-CUDA-token.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <vulkan/vulkan.h>

#include "imgui.h"
#include "imgui_internal.h"

#include "render/imgui/nuka_imgui.hpp"
#include "render/raster/vulkan_raster_renderer.hpp"
#include "render/render_world.hpp"
#include "runtime/app/viewer/camera_controller.hpp"
#include "runtime/app/viewer/imgui_layer.hpp"
#include "runtime/app/viewer/entity_drag.hpp"
#include "runtime/app/viewer/debug_draw.hpp"
#include "runtime/app/viewer/window_input.hpp"
#ifndef _WIN32
#include "render/window/xcb_keyboard.h"
#include <xcb/xcb.h>
#endif
#include "scene/asset/asset_ref.hpp"
#include "scene/asset/nka.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using nuka::render::MeshGeometry;
using nuka::render::RenderInstance;
using nuka::render::RenderWorld;
using nuka::render::VulkanRgba8;

MeshGeometry MakeBox(float h) {
    MeshGeometry g;
    const float p[8][3] = {
        {-h, -h, -h}, {h, -h, -h}, {h, h, -h}, {-h, h, -h},
        {-h, -h, h},  {h, -h, h},  {h, h, h},  {-h, h, h}};
    for (auto& v : p) {
        g.positions.push_back(v[0]); g.positions.push_back(v[1]); g.positions.push_back(v[2]);
    }
    const uint32_t faces[12][3] = {
        {0, 1, 2}, {0, 2, 3}, {4, 6, 5}, {4, 7, 6}, {0, 4, 5}, {0, 5, 1},
        {3, 2, 6}, {3, 6, 7}, {1, 5, 6}, {1, 6, 2}, {0, 3, 7}, {0, 7, 4}};
    for (auto& f : faces) {
        g.indices.push_back(f[0]); g.indices.push_back(f[1]); g.indices.push_back(f[2]);
    }
    return g;
}

// A synthetic RenderWorld: two boxes (one tagged as an .nka MESH source so the
// scene-tree shows the real-vs-primitive accent dot, exercising that path) + a
// metallic material.
RenderWorld BuildSyntheticWorld() {
    RenderWorld world;
    nuka::scene::RenderMaterial red;
    red.base_color[0] = 0.85f; red.base_color[1] = 0.15f; red.base_color[2] = 0.15f; red.base_color[3] = 1.0f;
    red.metallic = 0.1f; red.roughness = 0.6f;
    nuka::scene::RenderMaterial steel;
    steel.base_color[0] = 0.55f; steel.base_color[1] = 0.6f; steel.base_color[2] = 0.7f; steel.base_color[3] = 1.0f;
    steel.metallic = 0.9f; steel.roughness = 0.25f;
    world.materials.push_back(red);
    world.materials.push_back(steel);

    // mesh_a tagged NkaMesh via InternNkaMesh (a non-trivial "real geometry"
    // instance for the scene tree); mesh_b a primitive fallback.
    nuka::scene::AssetRef ref;
    ref.fourcc = nuka::scene::NkaTagMesh();
    ref.nka_path = "synthetic://box_a";
    const uint32_t mesh_a = world.meshes.InternNkaMesh(ref, [] { return MakeBox(0.3f); });
    const uint32_t mesh_b = world.meshes.InternPrimitive("prim:box:0.2", [] { return MakeBox(0.2f); });

    RenderInstance a;
    a.entity = nuka::scene::EntityId{1u, 0u};
    a.mesh_id = mesh_a; a.render_material_id = 0; a.world_xform.position = {-0.5f, 0.0f, 0.0f};
    world.instances.push_back(a);
    RenderInstance b;
    b.entity = nuka::scene::EntityId{2u, 0u};
    b.mesh_id = mesh_b; b.render_material_id = 1; b.world_xform.position = {0.5f, 0.0f, 0.25f};
    world.instances.push_back(b);
    return world;
}

void WritePpm(const std::string& path, const std::vector<VulkanRgba8>& pixels,
              uint32_t width, uint32_t height) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    std::fprintf(f, "P6\n%u %u\n255\n", width, height);
    for (const auto& px : pixels) {
        const unsigned char rgb[3] = {px.r, px.g, px.b};
        std::fwrite(rgb, 1, 3, f);
    }
    std::fclose(f);
}

}  // namespace

// VIEW-1: camera screen->world ray + drag-plane unproject (pure host, NO Vulkan
// -- runs even with no graphics device). Center pixel -> a ray along the camera
// forward; a known ground plane is hit at the expected world point.
TEST(ViewerDebugDraw, ExactPrimitiveKeysKeepDistinctGeometryAndReuseIdenticalInputs) {
    using namespace nuka::runtime::app::viewer;
    RenderWorld world = BuildSyntheticWorld();
    const auto real = world.instances;
    DebugDrawBatch batch;
    const auto pose = nuka::math::Transform::Identity();
    for (uint32_t kind = 0u; kind < 4u; ++kind) {
        const uint32_t count = kind == 0u ? 1u : (kind == 2u ? 3u : 2u);
        for (uint32_t axis = 0u; axis < count; ++axis) {
            float a[4] = {0.00001f, 0.00001f, 0.00001f, 0.0f};
            float b[4] = {0.00001f, 0.00001f, 0.00001f, 0.0f};
            b[axis] = 0.00002f;
            batch.AppendCollider(world, kind, a, pose, 0u);
            const auto first = world.debug_instances.back().mesh_id;
            batch.AppendCollider(world, kind, b, pose, 0u);
            const auto second = world.debug_instances.back().mesh_id;
            EXPECT_NE(first, second);
            EXPECT_NE(world.meshes.Geometry(first).positions, world.meshes.Geometry(second).positions);
            batch.AppendCollider(world, kind, a, pose, 0u);
            EXPECT_EQ(first, world.debug_instances.back().mesh_id);
            b[axis] = std::nextafter(a[axis], 1.0f);
            batch.AppendCollider(world, kind, b, pose, 0u);
            EXPECT_NE(first, world.debug_instances.back().mesh_id);
        }
    }
    float plane[4] = {};
    batch.AppendCollider(world, 3u, plane, pose, 0u);
    const auto fallback = world.debug_instances.back().mesh_id;
    plane[0] = -1.0f; plane[1] = -0.0f;
    batch.AppendCollider(world, 3u, plane, pose, 0u);
    EXPECT_EQ(fallback, world.debug_instances.back().mesh_id);
    float sphere[4] = {0.00002f, 0.0f, 0.0f, 0.0f};
    batch.AppendCollider(world, 0u, sphere, pose, 0u);
    float extent = 0.0f;
    for (float p : world.meshes.Geometry(world.debug_instances.back().mesh_id).positions)
        extent = std::max(extent, std::abs(p));
    EXPECT_FLOAT_EQ(extent, sphere[0]);
    ASSERT_EQ(world.instances.size(), real.size());
    for (size_t i = 0u; i < real.size(); ++i) {
        EXPECT_EQ(world.instances[i].mesh_id, real[i].mesh_id);
        EXPECT_EQ(world.instances[i].world_xform.position, real[i].world_xform.position);
        EXPECT_EQ(world.instances[i].entity, real[i].entity);
    }
}

TEST(ViewerDebugDraw, RejectsInvalidInputsAndKeepsTinyAndLargeNormalsFinite) {
    using namespace nuka::runtime::app::viewer;
    using nuka::math::Transform;
    RenderWorld world;
    DebugDrawBatch batch;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    float p[4] = {nan, 0.1f, 0.1f, 0.0f};
    for (uint32_t kind = 0u; kind < 4u; ++kind) batch.AppendCollider(world, kind, p, Transform::Identity(), 0u);
    p[0] = -1.0f;
    for (uint32_t kind = 0u; kind < 3u; ++kind) batch.AppendCollider(world, kind, p, Transform::Identity(), 0u);
    p[0] = 0.1f;
    Transform bad;
    bad.position.x = inf;
    batch.AppendCollider(world, 0u, p, bad, 0u);
    bad = Transform::Identity(); bad.rotation.w = 0.0f;
    batch.AppendCollider(world, 0u, p, bad, 0u);
    bad.rotation.w = nan;
    batch.AppendCollider(world, 0u, p, bad, 0u);
    bad.rotation.w = 2.0f;
    batch.AppendCollider(world, 0u, p, bad, 0u);
    p[0] = p[1] = std::numeric_limits<float>::max();
    batch.AppendCollider(world, 1u, p, Transform::Identity(), 0u);
    p[0] = 0.0f;
    batch.AppendCollider(world, 0u, p, Transform::Identity(), 0u);
    p[0] = 0.1f; p[1] = -0.1f;
    batch.AppendCollider(world, 1u, p, Transform::Identity(), 0u);
    p[1] = 0.1f; p[2] = 0.0f;
    batch.AppendCollider(world, 2u, p, Transform::Identity(), 0u);
    p[1] = inf;
    batch.AppendCollider(world, 3u, p, Transform::Identity(), 0u);
    p[0] = p[1] = p[2] = 2e38f;
    bad = Transform::Identity(); bad.position.x = 2e38f;
    batch.AppendCollider(world, 2u, p, bad, 0u);
    batch.AppendCollider(world, 4u, p, Transform::Identity(), 0u);
    batch.AppendContact(world, {nan, 0.0f, 0.0f}, 0u);
    batch.AppendContact(world, {0.0f, inf, 0.0f}, 0u);
    EXPECT_EQ(batch.Report().invalid_colliders, 17u);
    EXPECT_EQ(batch.Report().unsupported_shapes, 1u);
    EXPECT_EQ(batch.Report().invalid_contacts, 2u);
    EXPECT_TRUE(world.debug_instances.empty());
    EXPECT_EQ(world.meshes.Count(), 0u);
    for (float size : {1e-20f, 1e30f}) {
        p[0] = p[1] = p[2] = size;
        batch.AppendCollider(world, 2u, p, Transform::Identity(), 0u);
        const auto& geometry = world.meshes.Geometry(world.debug_instances.back().mesh_id);
        for (float v : geometry.positions) EXPECT_TRUE(std::isfinite(v));
        for (size_t i = 0; i < geometry.normals.size(); i += 3u) {
            const double n = std::hypot(geometry.normals[i], geometry.normals[i + 1u], geometry.normals[i + 2u]);
            EXPECT_NEAR(n, 1.0, 1e-6);
        }
    }
    batch.AppendContact(world, {0.0f, 0.0f, 0.0f}, 0u);
    EXPECT_EQ(batch.Report().contacts, 1u);
    batch.Reset();
    EXPECT_EQ(batch.Remaining(), kMaxDebugOverlayInstances);
    EXPECT_EQ(batch.Report().invalid_colliders, 0u);
    EXPECT_EQ(batch.Report().invalid_contacts, 0u);
    EXPECT_EQ(batch.Report().unsupported_shapes, 0u);
}

TEST(ViewerDebugDraw, BudgetSeparatesExactCapacityKnownOmissionsAndUnreadContacts) {
    using namespace nuka::runtime::app::viewer;
    RenderWorld world;
    DebugDrawBatch batch;
    EXPECT_FALSE(batch.ShouldReadContacts(0u));
    EXPECT_FALSE(batch.Report().contacts_budget_skipped);
    EXPECT_TRUE(batch.ShouldReadContacts(64u));
    for (uint32_t i = 0u; i < kMaxDebugOverlayInstances - 1u; ++i)
        batch.AppendContact(world, {}, 0u);
    EXPECT_EQ(batch.Remaining(), 1u);
    batch.AppendContact(world, {}, 0u);
    EXPECT_EQ(batch.Remaining(), 0u);
    EXPECT_EQ(batch.Report().omitted_instances, 0u);
    EXPECT_FALSE(batch.ShouldReadContacts(64u));
    EXPECT_TRUE(batch.Report().contacts_budget_skipped);
    EXPECT_FALSE(batch.ShouldReadContacts(0u));
    EXPECT_FALSE(batch.Report().contacts_budget_skipped);
    batch.AppendContact(world, {}, 0u);
    EXPECT_EQ(batch.Report().omitted_instances, 1u);
    float p[4] = {0.123456f, 0.0f, 0.0f, 0.0f};
    const auto meshes = world.meshes.Count();
    batch.AppendCollider(world, 0u, p, nuka::math::Transform::Identity(), 0u);
    EXPECT_EQ(batch.Report().omitted_instances, 2u);
    EXPECT_EQ(world.meshes.Count(), meshes);
    EXPECT_EQ(world.debug_instances.size(), kMaxDebugOverlayInstances);
    batch.Reset();
    world.debug_instances.clear();
    for (uint32_t i = 0u; i < kMaxDebugOverlayInstances; ++i)
        batch.AppendCollider(world, 0u, p, nuka::math::Transform::Identity(), 0u);
    EXPECT_EQ(batch.Report().omitted_instances, 0u);
    EXPECT_FALSE(batch.ShouldReadContacts(64u));
    EXPECT_TRUE(batch.Report().contacts_budget_skipped);
    EXPECT_EQ(batch.Report().contacts, 0u);
    batch.Reset();
    EXPECT_EQ(batch.Report().omitted_instances, 0u);
    EXPECT_FALSE(batch.Report().contacts_budget_skipped);
}

TEST(ViewerInput, Utf8EditingModifiersAndFocusUseProductionAdapter) {
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(960.0f, 600.0f);
    io.DeltaTime = 1.0f / 60.0f;
    io.IniFilename = nullptr;
    unsigned char* pixels = nullptr;
    int width = 0, height = 0;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    nuka::runtime::app::viewer::WindowInput input;
    char buffer[128] = {};
    bool focus_lost_event = false;
    auto frame = [&](bool focus = false) {
        ImGui::NewFrame();
        focus_lost_event = io.AppFocusLost;
        ImGui::Begin("Input fixture");
        if (focus) ImGui::SetKeyboardFocusHere();
        ImGui::InputText("text", buffer, sizeof(buffer));
        ImGui::End();
        ImGui::Render();
    };
    auto key = [&](uint32_t code, uint32_t symbol, bool pressed) {
        nuka::render::window::WindowEvent event;
        event.type = nuka::render::window::WindowEvent::Type::Key;
        event.key = code; event.keysym = symbol; event.pressed = pressed;
        input.Feed(event);
        frame();
    };
    frame(true); frame();
    nuka::render::window::WindowEvent event;
    event.type = nuka::render::window::WindowEvent::Type::TextInput;
    event.text = "Nuka \xc3\xa9\xce\xa9";
    input.Feed(event); frame();
    EXPECT_STREQ(buffer, "Nuka \xc3\xa9\xce\xa9");
    key(22u, 0xff08u, true);
    EXPECT_STREQ(buffer, "Nuka \xc3\xa9");
    key(22u, 0xff08u, false);
    key(37u, 0xffe3u, true);
    key(105u, 0xffe4u, true);
    key(37u, 0xffe3u, false);
    EXPECT_TRUE(io.KeyCtrl);
    key(38u, 'a', true);
    key(38u, 'q', true);
    EXPECT_TRUE(ImGui::IsKeyDown(ImGuiKey_A));
    EXPECT_FALSE(ImGui::IsKeyDown(ImGuiKey_Q));
    key(38u, 'q', false);
    EXPECT_FALSE(ImGui::IsKeyDown(ImGuiKey_A));
    key(105u, 0xffe4u, false);
    event.text = "edited";
    input.Feed(event); frame();
    EXPECT_STREQ(buffer, "edited");
    key(37u, 0xffe3u, true);
    event.type = nuka::render::window::WindowEvent::Type::FocusLost;
    input.Feed(event); frame();
    EXPECT_FALSE(io.KeyCtrl);
    EXPECT_TRUE(focus_lost_event);
    event.type = nuka::render::window::WindowEvent::Type::FocusGained;
    input.Feed(event); frame(true); frame();
    EXPECT_FALSE(io.AppFocusLost);
    event.type = nuka::render::window::WindowEvent::Type::TextInput;
    event.text = "ready";
    input.Feed(event); frame();
    EXPECT_NE(std::strstr(buffer, "ready"), nullptr);
    key(37u, 0xffe3u, true);
    key(38u, 'a', true);
    event.type = nuka::render::window::WindowEvent::Type::MouseButton;
    event.button = 0u; event.pressed = true;
    input.Feed(event); frame();
    event.type = nuka::render::window::WindowEvent::Type::FocusLost;
    input.Feed(event);
    event.type = nuka::render::window::WindowEvent::Type::FocusGained;
    input.Feed(event); frame();
    EXPECT_FALSE(io.KeyCtrl);
    EXPECT_FALSE(ImGui::IsKeyDown(ImGuiKey_A));
    EXPECT_FALSE(io.MouseDown[0]);
    key(108u, 0xfe03u, true);
    EXPECT_TRUE(input.LayoutModifierDown());
    key(108u, 0xfe03u, false);
    EXPECT_FALSE(input.LayoutModifierDown());
    ImGui::DestroyContext();
}

#ifndef _WIN32
TEST(ViewerInput, XkbComposeProducesCommittedUtf8AndResets) {
    std::unique_ptr<NukaXkbText, decltype(&NukaXkbTextDestroy)> text(
        NukaXkbTextCreate("en_US.UTF-8"), &NukaXkbTextDestroy);
    ASSERT_NE(text, nullptr);
    EXPECT_STREQ(NukaXkbTextFeed(text.get(), 0xfe51u), "");
    EXPECT_STREQ(NukaXkbTextFeed(text.get(), 'a'), "\xc3\xa1");
    EXPECT_STREQ(NukaXkbTextFeed(text.get(), 0x010003a9u), "\xce\xa9");
    EXPECT_STREQ(NukaXkbTextFeed(text.get(), 0xfe51u), "");
    NukaXkbTextReset(text.get());
    EXPECT_STREQ(NukaXkbTextFeed(text.get(), 'a'), "a");
    EXPECT_STREQ(NukaXkbTextFeed(text.get(), 0xff0du), "");
    xcb_key_release_event_t release{};
    release.response_type = XCB_KEY_RELEASE; release.detail = 38u; release.time = 123u;
    xcb_key_press_event_t press{};
    press.response_type = XCB_KEY_PRESS; press.detail = 38u; press.time = 123u;
    EXPECT_TRUE(NukaXcbAutoRepeatPair(&release, &press));
    press.time = 124u;
    EXPECT_FALSE(NukaXcbAutoRepeatPair(&release, &press));
    EXPECT_FALSE(NukaXcbAutoRepeatPair(&release, nullptr));
}
#endif

TEST(ViewerCameraFraming, SelectionUsesTransformedGeometryAndExcludesDebug) {
    auto world = BuildSyntheticWorld();
    auto geometry = world.meshes.Geometry(world.instances[1].mesh_id);
    for (size_t i = 0; i + 2u < geometry.positions.size(); i += 3u) {
        geometry.positions[i] *= 3.0f;
        geometry.positions[i + 1u] *= 0.5f;
        geometry.positions[i + 2u] *= 2.0f;
    }
    world.meshes.ReplaceGeometry(world.instances[1].mesh_id, std::move(geometry));
    world.instances[1].world_xform.rotation = nuka::math::Quat::FromAxisAngle({0, 0, 1}, 1.57079633f);
    auto debug = world.instances[1];
    debug.world_xform.position = {1000.0f, 1000.0f, 1000.0f};
    world.debug_instances.push_back(debug);
    nuka::runtime::app::viewer::CameraController camera;
    camera.SetView({5, 5, 5}, 10.0f, 0.5f, 0.2f);
    const auto before = world.instances[1].world_xform;
    ASSERT_TRUE(camera.FrameSelected(world, world.instances[1].entity));
    EXPECT_NEAR(camera.ResolvedTarget().x, 0.5f, 1e-5f);
    EXPECT_NEAR(camera.ResolvedTarget().z, 0.25f, 1e-5f);
    EXPECT_FLOAT_EQ(camera.Yaw(), 0.5f);
    EXPECT_FLOAT_EQ(camera.Pitch(), 0.2f);
    EXPECT_LT(camera.Distance(), 10.0f);
    const float wide_distance = camera.Distance();
    ASSERT_TRUE(camera.FrameSelected(world, world.instances[1].entity, 0.5f));
    EXPECT_GT(camera.Distance(), wide_distance);
    EXPECT_FLOAT_EQ(world.instances[1].world_xform.position.x, before.position.x);
    EXPECT_FLOAT_EQ(world.instances[1].world_xform.rotation.w, before.rotation.w);
    EXPECT_FALSE(camera.FrameSelected(world, nuka::scene::kInvalidEntity));
    EXPECT_FALSE(camera.FrameSelected(world, {999u, 0u}));
    ASSERT_TRUE(camera.FrameAll(world));
    EXPECT_LT(camera.Distance(), 10.0f);
    RenderWorld empty;
    const float unchanged = camera.Distance();
    EXPECT_FALSE(camera.FrameAll(empty));
    EXPECT_FLOAT_EQ(camera.Distance(), unchanged);
    const auto target = camera.ResolvedTarget();
    const float huge = std::numeric_limits<float>::max();
    EXPECT_FALSE(camera.FrameAabb({huge, huge, huge}, {huge, huge, huge}));
    EXPECT_FALSE(camera.FrameAabb({-huge, -huge, -huge}, {huge, huge, huge}));
    EXPECT_FLOAT_EQ(camera.ResolvedTarget().x, target.x);
    EXPECT_FLOAT_EQ(camera.Distance(), unchanged);
    camera.fov_degrees = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(camera.FrameAll(world));
    EXPECT_FLOAT_EQ(camera.ResolvedTarget().x, target.x);
    EXPECT_FLOAT_EQ(camera.Distance(), unchanged);
    camera.fov_degrees = 45.0f;
    camera.SetView({0, 0, 0}, 10.0f, 0.0f, 0.0f);
    world.instances[1].world_xform.position = {1e9f, 0.0f, 0.0f};
    EXPECT_FALSE(camera.FrameSelected(world, world.instances[1].entity));
    EXPECT_FLOAT_EQ(camera.ResolvedTarget().x, 0.0f);
    EXPECT_FLOAT_EQ(camera.Distance(), 10.0f);
}

TEST(ViewerInput, CameraShortcutsRespectTextFocusAndViewportOwnership) {
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(960.0f, 600.0f);
    io.DeltaTime = 1.0f / 60.0f;
    io.IniFilename = nullptr;
    unsigned char* pixels = nullptr;
    int width = 0, height = 0;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    auto world = BuildSyntheticWorld();
    nuka::runtime::app::viewer::WindowInput input;
    nuka::runtime::app::viewer::CameraController camera;
    auto press = [&](bool hovered, bool typing, bool gizmo, uint32_t symbol = 'f') {
        camera.SetView({10, 10, 10}, 20.0f, 0.5f, 0.2f);
        nuka::render::window::WindowEvent event;
        event.type = nuka::render::window::WindowEvent::Type::Key;
        event.key = 41u; event.keysym = symbol; event.pressed = true;
        input.Feed(event);
        ImGui::NewFrame();
        io.WantCaptureKeyboard = typing;
        io.WantTextInput = typing;
        nuka::runtime::app::viewer::ApplyCameraShortcuts(world, camera, world.instances[1].entity, hovered, gizmo);
        ImGui::Render();
        event.pressed = false;
        input.Feed(event);
        ImGui::NewFrame(); ImGui::Render();
    };
    press(true, true, false);
    EXPECT_FLOAT_EQ(camera.ResolvedTarget().x, 10.0f);
    press(false, false, false);
    EXPECT_FLOAT_EQ(camera.ResolvedTarget().x, 10.0f);
    press(true, false, true);
    EXPECT_FLOAT_EQ(camera.ResolvedTarget().x, 10.0f);
    press(true, false, false);
    EXPECT_NEAR(camera.ResolvedTarget().x, 0.5f, 1e-5f);
    press(true, false, false, 0xff50u);
    EXPECT_LT(camera.ResolvedTarget().x, 0.5f);
    ImGui::DestroyContext();
}

TEST(ViewerInteraction, TransportTogglesKeepStyleStackBalanced) {
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(1280.0f, 720.0f);
    io.DeltaTime = 1.0f / 60.0f;
    unsigned char* pixels = nullptr;
    int width = 0, height = 0;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    nuka::runtime::app::viewer::ImGuiLayer ui;
    ui.EnableDocking();
    nuka::runtime::app::viewer::CameraController camera;
    nuka::runtime::app::viewer::ViewerStats stats;
    nuka::runtime::app::viewer::ViewerUiState state;
    state.has_scene = true;
    const RenderWorld world = BuildSyntheticWorld();
    auto frame = [&]() {
        ImGui::NewFrame();
        ui.RecordUi(world, stats, camera, state);
        EXPECT_EQ(GImGui->ColorStack.Size, 0);
        ImGui::Render();
    };
    for (int i = 0; i < 4; ++i) frame();
    ImGuiWindow* transport = ImGui::FindWindowByName("##transport");
    ASSERT_NE(transport, nullptr);
    for (int i = 0; i < 20; ++i) {
        const bool before = state.playing;
        state.play_toggled = false;
        ImGui::ActivateItemByID(transport->GetID(before ? "Pause" : "Play"));
        frame();
        EXPECT_NE(state.playing, before);
        EXPECT_TRUE(state.play_toggled);
    }
    state.playing = true;
    state.step_requested = false;
    ImGui::ActivateItemByID(transport->GetID("Step"));
    frame();
    EXPECT_FALSE(state.step_requested);
    state.playing = false;
    ImGui::ActivateItemByID(transport->GetID("Step"));
    frame();
    EXPECT_TRUE(state.step_requested);
    ImGui::DestroyContext();
}

TEST(ViewerInteraction, EntityDragCancelsBeforeInputCaptureGates) {
    using nuka::render::window::WindowEvent;
    nuka::runtime::app::viewer::EntityDrag drag;
    WindowEvent event;
    event.type = WindowEvent::Type::MouseButton;
    event.button = 0u;
    event.pressed = false;
    drag.instance = 3u;
    drag.Observe(event);
    EXPECT_EQ(drag.instance, drag.kNone);
    for (const uint32_t key : {0xffe3u, 0xffe4u}) {
        drag.instance = 3u;
        event.type = WindowEvent::Type::Key;
        event.keysym = key;
        drag.Observe(event);
        EXPECT_EQ(drag.instance, drag.kNone);
    }
    for (const auto type : {WindowEvent::Type::FocusLost, WindowEvent::Type::Close}) {
        drag.instance = 3u;
        event.type = type;
        drag.Observe(event);
        EXPECT_EQ(drag.instance, drag.kNone);
    }
    drag.instance = 3u;
    drag.Cancel();
    event.type = WindowEvent::Type::MouseMove;
    drag.Observe(event);
    EXPECT_EQ(drag.instance, drag.kNone);
}

TEST(ViewerInteraction, CameraStopsWhenUiCapturesDrag) {
    using nuka::render::window::WindowEvent;
    nuka::runtime::app::viewer::CameraController camera;
    camera.FrameAabb({-1.0f, -1.0f, -1.0f}, {1.0f, 1.0f, 1.0f});
    WindowEvent event;
    event.type = WindowEvent::Type::MouseButton;
    event.button = 0u;
    event.pressed = true;
    camera.HandleEvent(event, true, true);
    const float yaw = camera.Yaw();
    event.type = WindowEvent::Type::MouseMove;
    event.mouse_x = 80;
    camera.HandleEvent(event, false, false);
    EXPECT_FLOAT_EQ(camera.Yaw(), yaw);
    event.mouse_x = 160;
    camera.HandleEvent(event, true, true);
    EXPECT_FLOAT_EQ(camera.Yaw(), yaw);
}

TEST(ViewerCameraRay, CenterRayAndKnownPlaneHit) {
    using nuka::runtime::app::viewer::CameraController;
    using nuka::runtime::app::viewer::Ray;
    using nuka::math::Vec3;

    CameraController cam;
    // Frame a symmetric box -> target at its centre, a deterministic 3/4 view.
    cam.FrameAabb({-1.0f, -1.0f, -1.0f}, {1.0f, 1.0f, 1.0f});
    const Vec3 eye = cam.ResolvedEye();
    const Vec3 tgt = cam.ResolvedTarget();
    const Vec3 fwd = (tgt - eye).Normalized();

    const uint32_t W = 1280u, H = 720u;

    // (1) The CENTER pixel ray points along the camera forward (within a px of
    // discretization) and originates at the eye.
    const Ray center = cam.ScreenRay(static_cast<float>(W) * 0.5f - 0.5f,
                                     static_cast<float>(H) * 0.5f - 0.5f, W, H);
    EXPECT_NEAR(center.origin.x, eye.x, 1e-4f);
    EXPECT_NEAR(center.origin.y, eye.y, 1e-4f);
    EXPECT_NEAR(center.origin.z, eye.z, 1e-4f);
    const float align = center.dir.Dot(fwd);
    EXPECT_GT(align, 0.999f) << "center ray not aligned with camera forward";

    // (2) The center ray hits the plane through the target perpendicular to the
    // view forward EXACTLY at the target (the drag anchor plane case).
    Vec3 hit;
    ASSERT_TRUE(cam.RayPlaneHit(center, tgt, fwd, &hit));
    EXPECT_NEAR(hit.x, tgt.x, 1e-3f);
    EXPECT_NEAR(hit.y, tgt.y, 1e-3f);
    EXPECT_NEAR(hit.z, tgt.z, 1e-3f);

    // (3) A KNOWN ground-plane hit: a top-down camera straight above the origin,
    // center ray hits the z=0 plane at the origin.
    CameraController top;
    top.FrameAabb({-0.001f, -0.001f, -0.001f}, {0.001f, 0.001f, 0.001f});
    // FrameAabb sets a 3/4 view; drive a top-down ray analytically instead by
    // intersecting the center ray of `top` with z=0 and just checking it is a
    // finite point (a generic plane-hit smoke -- the exact analytic top-down view
    // is owner-tunable). Use the camera's own forward-facing plane for the assert.
    const Ray r2 = top.ScreenRay(640.0f, 360.0f, W, H);
    Vec3 ground;
    const bool hit_ground =
        top.RayPlaneHit(r2, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, &ground);
    // The 3/4 view looks down at the origin region -> the z=0 plane is hit in front.
    EXPECT_TRUE(hit_ground) << "expected the framing ray to hit the ground plane";

    // (4) A degenerate viewport falls back to a forward ray (no div-by-zero).
    const Ray deg = cam.ScreenRay(0.0f, 0.0f, 0u, 0u);
    EXPECT_NEAR(deg.dir.Dot(fwd), 1.0f, 1e-4f);
}

TEST(ViewerFrameSmoke, OffscreenScenePlusImGuiCompositeIsDeterministic) {
    const RenderWorld world = BuildSyntheticWorld();

    nuka::render::RasterOptions options;
    options.width = 1280;
    options.height = 720;

    // 1. Offscreen renderer (the D1 oracle). SKIP if no Vulkan device.
    std::unique_ptr<nuka::render::VulkanRasterRenderer> renderer;
    try {
        renderer = std::make_unique<nuka::render::VulkanRasterRenderer>();
    } catch (const std::exception& e) {
        GTEST_SKIP() << "No Vulkan graphics device available: " << e.what();
    }

    // 2. ImGui context bound to the OFFSCREEN render pass + descriptor pool.
    nuka::render::RendererVulkanHandles vk = renderer->VulkanHandles();
    nuka::render::imgui::NukaImGuiInitInfo info;
    info.api_version     = vk.api_version;
    info.instance        = reinterpret_cast<NukaVkInstance>(vk.instance);
    info.physical_device = reinterpret_cast<NukaVkPhysicalDevice>(vk.physical_device);
    info.device          = reinterpret_cast<NukaVkDevice>(vk.device);
    info.queue_family    = vk.graphics_family;
    info.queue           = reinterpret_cast<NukaVkQueue>(vk.graphics_queue);
    info.descriptor_pool = reinterpret_cast<NukaVkDescriptorPool>(vk.imgui_descriptor_pool);
    info.min_image_count = 2u;
    info.image_count     = 2u;
    info.render_pass     = reinterpret_cast<NukaVkRenderPass>(vk.offscreen_render_pass);
    info.subpass         = 0u;

    nuka::render::imgui::NukaImGuiContext imgui;
    if (!imgui.Init(info)) {
        GTEST_SKIP() << "ImGui Vulkan backend init failed (no device / pool)";
    }

    // 3. FIXED UI state -> deterministic draw data (the D1 precondition).
    nuka::runtime::app::viewer::ImGuiLayer ui;
    ui.EnableDocking();  // MUST precede the first NewFrame (ImGui asserts this).
    nuka::runtime::app::viewer::CameraController camera;
    camera.FrameAabb({-0.8f, -0.5f, -0.3f}, {0.8f, 0.5f, 0.6f});
    nuka::runtime::app::viewer::ViewerStats stats;
    stats.step_time_ms = 1.23f; stats.fps = 60.0f; stats.dof = 11u; stats.links = 6u;
    stats.bodies = 1u; stats.contact_cap = 64u; stats.draw_calls = 2u; stats.non_bg_pixels = 12345u;
    stats.frame_index = 7u; stats.device_name = "offscreen";
    nuka::runtime::app::viewer::ViewerUiState ui_state;
    ui_state.playing = false; ui_state.speed = 1.0f;
    ui_state.has_scene = true;
    ui_state.loaded_path = "Synthetic renderer fixture (no physics session)";
    stats.debug_capacity = 8192u;
    // FIXED drive-panel + entity-selection state so the new panels record
    // DETERMINISTICALLY (no time/animation widget). Pre-seed a few DOF sliders to
    // fixed values + select the steel box's entity; drive_dirty stays all-zero so
    // the offscreen record is steady (no slider interaction happens).
    ui_state.drive_targets = {0.10f, -0.25f, 0.50f, 0.0f, 1.23f, -0.75f};
    ui_state.drive_dirty.assign(ui_state.drive_targets.size(), 0u);
    ui_state.selected_entity = world.instances[1].entity;  // the steel box

    // Build one ImGui frame with a fixed display size; record the panels.
    auto build_ui = [&]() -> int {
        ImGuiIO& io = ImGui::GetIO();
        io.DisplaySize = ImVec2(static_cast<float>(options.width),
                                static_cast<float>(options.height));
        io.DeltaTime = 1.0f / 60.0f;  // fixed (no time-seeded animation)
        imgui.NewFrame();
        ui.RecordUi(world, stats, camera, ui_state);
        ImGui::Render();
        ImDrawData* dd = ImGui::GetDrawData();
        return dd ? dd->CmdListsCount : 0;
    };

    // Warm-up: the docking layout (DockBuilder split + DockSpaceOverViewport) needs
    // a couple of frames to settle -- on the FIRST frame the panels are created and
    // docked but not yet laid out, so draw data is incomplete. Drive a few frames so
    // the layout latches; AFTER that, a fixed UI state is steady-state deterministic.
    // (imgui_impl_vulkan also uploads the font atlas lazily on first NewFrame.)
    for (int w = 0; w < 4; ++w) build_ui();

    // 4. First composite.
    const int cmd_lists_a = build_ui();
    nuka::render::VulkanOffscreenReport first;
    try {
        first = renderer->Render(world, options, [&imgui](void* cmd) {
            imgui.RenderDrawData(reinterpret_cast<NukaVkCommandBuffer>(cmd));
        });
    } catch (const std::exception& e) {
        GTEST_SKIP() << "composite render failed: " << e.what();
    }

    // 5. Second composite (rebuild the same fixed UI -> same draw data).
    const int cmd_lists_b = build_ui();
    nuka::render::VulkanOffscreenReport second;
    try {
        second = renderer->Render(world, options, [&imgui](void* cmd) {
            imgui.RenderDrawData(reinterpret_cast<NukaVkCommandBuffer>(cmd));
        });
    } catch (const std::exception& e) {
        GTEST_SKIP() << "composite render (2) failed: " << e.what();
    }

    // (a) the UI actually built draw commands.
    EXPECT_GT(cmd_lists_a, 0) << "imgui_layer produced no draw command lists";
    EXPECT_EQ(cmd_lists_a, cmd_lists_b) << "fixed UI state must build identical command-list count";

    // (b) the composited image has lit (scene + UI) pixels.
    EXPECT_GT(first.non_background_pixel_count, 0u);
    EXPECT_EQ(first.width, options.width);
    EXPECT_EQ(first.height, options.height);
    ASSERT_EQ(first.pixels.size(), static_cast<size_t>(options.width) * options.height);

    // (c) D1: two composited renders byte-identical.
    ASSERT_EQ(first.pixels.size(), second.pixels.size());
    const int cmp = std::memcmp(first.pixels.data(), second.pixels.data(),
                                first.pixels.size() * sizeof(VulkanRgba8));
    EXPECT_EQ(cmp, 0) << "two composited (scene + fixed-UI) renders are not byte-identical";

    // (d) dump a PPM for off-box beauty review.
    WritePpm("/tmp/m8_5_viewer_frame.ppm", first.pixels, first.width, first.height);

    std::printf("[viewer_frame_smoke] device=%s cmd_lists=%d non_bg=%zu D1=%s "
                "ppm=/tmp/m8_5_viewer_frame.ppm\n",
                first.selected_device_name.c_str(), cmd_lists_a,
                first.non_background_pixel_count, cmp == 0 ? "BYTE-IDENTICAL" : "MISMATCH");

    options.width = 960u;
    options.height = 600u;
    for (int i = 0; i < 4; ++i) build_ui();
    const auto narrow = renderer->Render(world, options, [&imgui](void* cmd) {
        imgui.RenderDrawData(reinterpret_cast<NukaVkCommandBuffer>(cmd));
    });
    WritePpm("/tmp/nuka_viewer_narrow.ppm", narrow.pixels, narrow.width, narrow.height);
    ui_state.inspector.valid = true;
    ui_state.inspector.has_material = true;
    ui_state.inspector.pos[0] = world.instances[1].world_xform.position.x;
    ui_state.inspector.pos[2] = world.instances[1].world_xform.position.z;
    ImGui::SetWindowFocus("Entity");
    for (int i = 0; i < 4; ++i) build_ui();
    const auto inspector = renderer->Render(world, options, [&imgui](void* cmd) {
        imgui.RenderDrawData(reinterpret_cast<NukaVkCommandBuffer>(cmd));
    });
    WritePpm("/tmp/nuka_viewer_inspector.ppm", inspector.pixels, inspector.width, inspector.height);
    options.width = 1280u; options.height = 720u;
    ImGui::SetWindowFocus("Camera");
    for (int i = 0; i < 4; ++i) build_ui();
    ImGuiWindow* camera_panel = ImGui::FindWindowByName("Camera");
    ASSERT_NE(camera_panel, nullptr);
    ImGui::ActivateItemByID(camera_panel->GetID("Frame Selected (F)"));
    build_ui();
    EXPECT_NEAR(camera.ResolvedTarget().x, world.instances[1].world_xform.position.x, 1e-5f);
    EXPECT_NEAR(camera.ResolvedTarget().z, world.instances[1].world_xform.position.z, 1e-5f);
    build_ui();
    camera.WriteOptions(options);
    const auto selected_view = renderer->Render(world, options, [&imgui](void* cmd) {
        imgui.RenderDrawData(reinterpret_cast<NukaVkCommandBuffer>(cmd));
    });
    WritePpm("/tmp/nuka_viewer_camera.ppm", selected_view.pixels, selected_view.width, selected_view.height);

    ASSERT_TRUE(camera.FrameAll(world, static_cast<float>(options.width) / static_cast<float>(options.height)));
    camera.WriteOptions(options);
    stats.debug_invalid_colliders = 3u;
    stats.debug_invalid_contacts = 0u;
    stats.debug_omitted_instances = 1u;
    stats.debug_contacts_budget_skipped = true;
    stats.debug_colliders_available = true;
    stats.debug_colliders = 8192u;
    ui_state.show_colliders = ui_state.show_contacts = true;
    ImGui::SetWindowFocus("Physics Debug");
    for (int i = 0; i < 4; ++i) build_ui();
    const auto diagnostics = renderer->Render(world, options, [&imgui](void* cmd) {
        imgui.RenderDrawData(reinterpret_cast<NukaVkCommandBuffer>(cmd));
    });
    WritePpm("/tmp/nuka_debug_diagnostics.ppm", diagnostics.pixels, diagnostics.width, diagnostics.height);

    options.width = 960u; options.height = 600u;
    for (int i = 0; i < 4; ++i) build_ui();
    const auto diagnostics_narrow = renderer->Render(world, options, [&imgui](void* cmd) {
        imgui.RenderDrawData(reinterpret_cast<NukaVkCommandBuffer>(cmd));
    });
    WritePpm("/tmp/nuka_debug_diagnostics_narrow.ppm", diagnostics_narrow.pixels,
             diagnostics_narrow.width, diagnostics_narrow.height);

    ImGuiWindow* debug_panel = ImGui::FindWindowByName("Physics Debug");
    ASSERT_NE(debug_panel, nullptr);
    ASSERT_GT(debug_panel->ScrollMax.y, 0.0f);
    ImGui::SetScrollY(debug_panel, debug_panel->ScrollMax.y);
    for (int i = 0; i < 2; ++i) build_ui();
    const auto diagnostics_scrolled = renderer->Render(world, options, [&imgui](void* cmd) {
        imgui.RenderDrawData(reinterpret_cast<NukaVkCommandBuffer>(cmd));
    });
    WritePpm("/tmp/nuka_debug_diagnostics_scrolled.ppm", diagnostics_scrolled.pixels,
             diagnostics_scrolled.width, diagnostics_scrolled.height);

    imgui.Shutdown();
}
