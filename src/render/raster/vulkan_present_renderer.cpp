// ---------------------------------------------------------------------------
// nuka::render::PresentRenderer implementation (M8.5 T2).
//
// The realtime swapchain present path. SEPARATE from the offscreen renderer (the
// G2 oracle stays pristine): it BORROWS a present-capable VulkanRasterRenderer for
// the device/queue/handles, then owns its own swapchain + present render pass +
// pipeline + framebuffers + per-frame sync, and runs acquire -> draw -> present.
//
// Defensive surface negotiation (lavapipe + 1.2.131 loader): formats / present
// modes / capabilities are QUERIED, never hardcoded. The matrix/camera math
// mirrors the offscreen renderer's helpers (intentionally duplicated file-local
// so this sibling shares NO mutable state with the offscreen path).
//
// HOST-ONLY / zero-CUDA-token: pure C++ / Vulkan. No CUDA tokens.
// ---------------------------------------------------------------------------

#include "render/raster/vulkan_present_renderer.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace nuka::render {

namespace {

#ifndef NUKA_RASTER_MESH_VERT_SPV
#define NUKA_RASTER_MESH_VERT_SPV "mesh.vert.spv"
#endif
#ifndef NUKA_RASTER_MESH_FRAG_SPV
#define NUKA_RASTER_MESH_FRAG_SPV "mesh_pbr.frag.spv"
#endif
#ifndef NUKA_RASTER_MESH_INSTANCED_VERT_SPV
#define NUKA_RASTER_MESH_INSTANCED_VERT_SPV "mesh_instanced.vert.spv"
#endif

constexpr VkFormat   kDepthFormat = VK_FORMAT_D32_SFLOAT;
constexpr uint32_t   kFramesInFlight = 2u;

void CheckVk(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with VkResult " +
                                 std::to_string(static_cast<int>(result)));
    }
}

std::vector<char> ReadBinaryFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Failed to open Vulkan shader: " + path);
    const std::streamsize size = file.tellg();
    if (size <= 0) throw std::runtime_error("Vulkan shader is empty: " + path);
    file.seekg(0, std::ios::beg);
    std::vector<char> bytes(static_cast<size_t>(size));
    if (!file.read(bytes.data(), size)) {
        throw std::runtime_error("Failed to read Vulkan shader: " + path);
    }
    return bytes;
}

uint32_t FindMemoryType(VkPhysicalDevice physical_device, uint32_t type_filter,
                        VkMemoryPropertyFlags required_flags) {
    VkPhysicalDeviceMemoryProperties memory_properties{};
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
    for (uint32_t index = 0; index < memory_properties.memoryTypeCount; ++index) {
        const bool type_matches = (type_filter & (1u << index)) != 0u;
        const bool flags_match =
            (memory_properties.memoryTypes[index].propertyFlags & required_flags) == required_flags;
        if (type_matches && flags_match) return index;
    }
    throw std::runtime_error("No compatible Vulkan memory type found (present)");
}

// -- column-major 4x4 (matches GLSL mat4 + the offscreen push block) ----------
struct Mat4 {
    std::array<float, 16> m{};
    static Mat4 Identity() {
        Mat4 r;
        r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
        return r;
    }
};

Mat4 Multiply(const Mat4& a, const Mat4& b) {
    Mat4 r;
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) sum += a.m[k * 4 + row] * b.m[col * 4 + k];
            r.m[col * 4 + row] = sum;
        }
    }
    return r;
}

Mat4 TransformToMatrix(const math::Transform& t) {
    const math::Quat& qn = t.rotation;
    const float x = qn.x, y = qn.y, z = qn.z, w = qn.w;
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, xz = x * z, yz = y * z;
    const float wx = w * x, wy = w * y, wz = w * z;
    Mat4 r = Mat4::Identity();
    r.m[0] = 1.0f - 2.0f * (yy + zz);
    r.m[1] = 2.0f * (xy + wz);
    r.m[2] = 2.0f * (xz - wy);
    r.m[4] = 2.0f * (xy - wz);
    r.m[5] = 1.0f - 2.0f * (xx + zz);
    r.m[6] = 2.0f * (yz + wx);
    r.m[8] = 2.0f * (xz + wy);
    r.m[9] = 2.0f * (yz - wx);
    r.m[10] = 1.0f - 2.0f * (xx + yy);
    r.m[12] = t.position.x;
    r.m[13] = t.position.y;
    r.m[14] = t.position.z;
    return r;
}

Mat4 LookAt(const math::Vec3& eye, const math::Vec3& target, const math::Vec3& up) {
    math::Vec3 f = (target - eye).Normalized();
    math::Vec3 s = f.Cross(up).Normalized();
    math::Vec3 u = s.Cross(f);
    Mat4 r = Mat4::Identity();
    r.m[0] = s.x;  r.m[4] = s.y;  r.m[8]  = s.z;
    r.m[1] = u.x;  r.m[5] = u.y;  r.m[9]  = u.z;
    r.m[2] = -f.x; r.m[6] = -f.y; r.m[10] = -f.z;
    r.m[12] = -s.Dot(eye);
    r.m[13] = -u.Dot(eye);
    r.m[14] = f.Dot(eye);
    return r;
}

Mat4 Perspective(float fov_y_radians, float aspect, float z_near, float z_far) {
    const float t = std::tan(fov_y_radians * 0.5f);
    Mat4 r;
    r.m[0] = 1.0f / (aspect * t);
    r.m[5] = -1.0f / t;
    r.m[10] = z_far / (z_near - z_far);
    r.m[11] = -1.0f;
    r.m[14] = (z_near * z_far) / (z_near - z_far);
    return r;
}

// M8.5 T4: mirrors the offscreen renderer's PushBlock + SceneUbo (same shaders).
struct PushBlock {
    float model[16];
    float base_color[4];
    float mr[4];        // x metallic, y roughness, z opacity, w pad
    float emissive[4];  // rgb emissive, a pad
};
static_assert(sizeof(PushBlock) == 112, "PushBlock must stay <=128 push-constant bytes");

constexpr int kMaxUboLights = 8;
struct GpuLight {
    float direction[4];
    float position[4];
    float color[4];
};
struct SceneUbo {
    float    view_proj[16];
    float    camera_pos[4];
    float    ambient[4];
    float    ambient_ground[4];
    int32_t  counts[4];
    GpuLight lights[kMaxUboLights];
};
static_assert(sizeof(SceneUbo) == 512, "SceneUbo must match the std140 shader block");

// Build the per-frame SceneUbo (camera + light rig). Same default 3-point rig +
// hemispheric ambient as the offscreen renderer (kept in sync deliberately).
SceneUbo BuildSceneUbo(const RenderWorld& world, const Mat4& view_proj,
                       const math::Vec3& eye) {
    SceneUbo ubo{};
    std::memcpy(ubo.view_proj, view_proj.m.data(), sizeof(ubo.view_proj));
    ubo.camera_pos[0] = eye.x; ubo.camera_pos[1] = eye.y;
    ubo.camera_pos[2] = eye.z; ubo.camera_pos[3] = 1.0f;
    ubo.ambient[0] = 0.20f; ubo.ambient[1] = 0.23f; ubo.ambient[2] = 0.28f; ubo.ambient[3] = 0.0f;
    ubo.ambient_ground[0] = 0.09f; ubo.ambient_ground[1] = 0.08f;
    ubo.ambient_ground[2] = 0.07f; ubo.ambient_ground[3] = 0.0f;
    auto set_dir = [](GpuLight& l, const math::Vec3& dir, float r, float g, float b) {
        const math::Vec3 d = dir.Normalized();
        l.direction[0] = d.x; l.direction[1] = d.y; l.direction[2] = d.z; l.direction[3] = 1.0f;
        l.position[0] = l.position[1] = l.position[2] = l.position[3] = 0.0f;
        l.color[0] = r; l.color[1] = g; l.color[2] = b; l.color[3] = 0.0f;
    };
    int n = 0;
    if (!world.lights.empty()) {
        for (const RenderLight& rl : world.lights) {
            if (n >= kMaxUboLights) break;
            GpuLight& l = ubo.lights[n];
            const math::Vec3 col = rl.color * rl.intensity;
            if (rl.type == scene::LightComponent::Type::Directional) {
                const math::Vec3 fwd = rl.world_xform.TransformDirection({0.0f, 0.0f, -1.0f});
                set_dir(l, fwd * -1.0f, col.x, col.y, col.z);
            } else {
                const math::Vec3 p = rl.world_xform.position;
                l.direction[0] = l.direction[1] = l.direction[2] = 0.0f; l.direction[3] = 0.0f;
                l.position[0] = p.x; l.position[1] = p.y; l.position[2] = p.z; l.position[3] = 0.0f;
                l.color[0] = col.x; l.color[1] = col.y; l.color[2] = col.z; l.color[3] = 0.0f;
            }
            ++n;
        }
    } else {
        set_dir(ubo.lights[n++], {0.5f, -0.6f, 0.75f}, 3.2f, 3.1f, 2.9f);
        set_dir(ubo.lights[n++], {-0.7f, 0.3f, 0.4f}, 0.8f, 0.9f, 1.1f);
        set_dir(ubo.lights[n++], {0.1f, 0.8f, 0.5f}, 1.0f, 0.95f, 0.85f);
    }
    ubo.counts[0] = n;
    return ubo;
}

std::vector<float> SynthesizeFlatNormals(const std::vector<float>& positions,
                                         const std::vector<uint32_t>& indices) {
    std::vector<float> normals(positions.size(), 0.0f);
    auto at = [&](uint32_t v, int c) { return positions[static_cast<size_t>(v) * 3 + c]; };
    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        const uint32_t a = indices[i], b = indices[i + 1], c = indices[i + 2];
        const math::Vec3 pa{at(a, 0), at(a, 1), at(a, 2)};
        const math::Vec3 pb{at(b, 0), at(b, 1), at(b, 2)};
        const math::Vec3 pc{at(c, 0), at(c, 1), at(c, 2)};
        math::Vec3 n = (pb - pa).Cross(pc - pa);
        const float len = n.Length();
        if (len > 0.0f) n = n * (1.0f / len);
        for (uint32_t v : {a, b, c}) {
            normals[static_cast<size_t>(v) * 3 + 0] += n.x;
            normals[static_cast<size_t>(v) * 3 + 1] += n.y;
            normals[static_cast<size_t>(v) * 3 + 2] += n.z;
        }
    }
    for (size_t v = 0; v + 2 < normals.size(); v += 3) {
        math::Vec3 n{normals[v], normals[v + 1], normals[v + 2]};
        const float len = n.Length();
        if (len > 0.0f) {
            normals[v] = n.x / len; normals[v + 1] = n.y / len; normals[v + 2] = n.z / len;
        } else {
            normals[v] = 0.0f; normals[v + 1] = 0.0f; normals[v + 2] = 1.0f;
        }
    }
    return normals;
}

struct GpuBuffer {
    VkBuffer       buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
};

// Camera resolution mirrors the offscreen renderer (so present and offscreen
// frame the same scene identically -- the offscreen path stays the visual oracle).
struct ResolvedCamera {
    math::Vec3 eye{0.0f, 0.0f, 0.0f};
    math::Vec3 target{0.0f, 0.0f, 0.0f};
    math::Vec3 up{0.0f, 0.0f, 1.0f};
    float fov_degrees = 45.0f;
    float near_clip = 0.05f;
    float far_clip = 1000.0f;
};

ResolvedCamera ResolveCamera(const RenderWorld& world, const RasterOptions& options,
                             const math::Vec3& aabb_min, const math::Vec3& aabb_max,
                             bool has_geometry) {
    ResolvedCamera cam;
    if (options.use_camera_override) {
        cam.eye = options.camera_eye;  cam.target = options.camera_target;
        cam.up = options.camera_up;    cam.fov_degrees = options.camera_fov_degrees;
        cam.near_clip = options.camera_near; cam.far_clip = options.camera_far;
        return cam;
    }
    if (world.CameraCount() > 0u) {
        const RenderCamera& rc = world.cameras[0];
        cam.eye = rc.world_xform.position;
        cam.target = rc.world_xform.TransformPoint({0.0f, 0.0f, -1.0f});
        cam.up = rc.world_xform.TransformDirection({0.0f, 1.0f, 0.0f});
        cam.fov_degrees = rc.vertical_fov_degrees;
        cam.near_clip = rc.near_clip; cam.far_clip = rc.far_clip;
        return cam;
    }
    math::Vec3 center{0.0f, 0.0f, 0.0f};
    float radius = 1.0f;
    if (has_geometry) {
        center = (aabb_min + aabb_max) * 0.5f;
        radius = std::max((aabb_max - aabb_min).Length() * 0.5f, 1e-3f);
    }
    const math::Vec3 dir = math::Vec3{1.0f, -1.0f, 0.6f}.Normalized();
    const float distance = radius * 3.0f + 0.5f;
    cam.eye = center + dir * distance;
    cam.target = center;
    cam.up = {0.0f, 0.0f, 1.0f};
    cam.fov_degrees = 45.0f;
    cam.near_clip = std::max(distance * 0.01f, 1e-3f);
    cam.far_clip = distance * 10.0f + radius * 10.0f;
    return cam;
}

}  // namespace

// ---------------------------------------------------------------------------
// PresentRenderer::Impl
// ---------------------------------------------------------------------------
struct PresentRenderer::Impl {
    std::unique_ptr<window::WindowSurface> window_surface;  // owns the platform window + VkSurfaceKHR
    std::unique_ptr<VulkanRasterRenderer>  renderer;        // present-capable; owns device/queue
    RendererVulkanHandles                  vk;              // borrowed handles from `renderer`

    VkSurfaceKHR    surface = VK_NULL_HANDLE;   // borrowed (window_surface owns destroy)
    VkSwapchainKHR  swapchain = VK_NULL_HANDLE;
    VkFormat        surface_format = VK_FORMAT_UNDEFINED;
    VkColorSpaceKHR color_space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
    VkExtent2D      extent{0u, 0u};
    uint32_t        min_image_count = 2u;

    std::vector<VkImage>       images;
    std::vector<VkImageView>   image_views;
    std::vector<VkFramebuffer> framebuffers;

    VkImage        depth_image = VK_NULL_HANDLE;
    VkDeviceMemory depth_memory = VK_NULL_HANDLE;
    VkImageView    depth_view = VK_NULL_HANDLE;

    VkRenderPass     present_pass = VK_NULL_HANDLE;
    VkDescriptorSetLayout scene_set_layout = VK_NULL_HANDLE;  // set 0: per-frame SceneUbo
    VkDescriptorPool scene_pool = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline       pipeline = VK_NULL_HANDLE;
    VkShaderModule   vert_module = VK_NULL_HANDLE;
    VkShaderModule   frag_module = VK_NULL_HANDLE;
    VkCommandPool    command_pool = VK_NULL_HANDLE;

    // M11 INT-F1: the OPT-IN instanced pipeline (mesh_instanced.vert) reading the
    // per-instance model matrix from the interop SSBO at set 1. Built lazily by
    // SetInteropTransforms (and rebuilt on swapchain recreate) ONLY when the viewer
    // installs a live SSBO descriptor set. The SSBO layout + set are BORROWED from
    // the InteropTransformSsbo (the viewer owns them); we never destroy them.
    VkShaderModule        instanced_vert_module = VK_NULL_HANDLE;  // freed in ~Impl
    VkPipelineLayout      instanced_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline            instanced_pipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout ssbo_set_layout = VK_NULL_HANDLE;  // borrowed (not owned)
    VkDescriptorSet       ssbo_set = VK_NULL_HANDLE;         // borrowed (not owned)
    bool                  interop_draw = false;  // bind the instanced pipeline this frame

    // Per-frame-in-flight SceneUbo buffer + descriptor set (one per frame so the
    // in-flight frame's uniform is not overwritten by the next).
    struct UboSlot { VkBuffer buffer = VK_NULL_HANDLE; VkDeviceMemory memory = VK_NULL_HANDLE;
                     VkDescriptorSet set = VK_NULL_HANDLE; };
    std::array<UboSlot, kFramesInFlight> ubo_slots{};

    // Per-frame-in-flight sync + command buffers.
    std::array<VkSemaphore, kFramesInFlight>     image_available{};
    std::array<VkSemaphore, kFramesInFlight>     render_finished{};
    std::array<VkFence, kFramesInFlight>         in_flight{};
    std::array<VkCommandBuffer, kFramesInFlight> command_buffers{};
    uint32_t current_frame = 0u;

    // Per-swapchain-image: a fence pointer tracking which in-flight frame last
    // used it (avoid presenting into an image still being read).
    std::vector<VkFence> images_in_flight;

    PresentReport report;
    bool should_close = false;

    VkDevice         device = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkQueue          graphics_queue = VK_NULL_HANDLE;
    VkQueue          present_queue = VK_NULL_HANDLE;

    explicit Impl(std::unique_ptr<window::WindowSurface> surf)
        : window_surface(std::move(surf)) {
        if (!window_surface) {
            throw std::runtime_error("PresentRenderer: no window surface (no display / no ext)");
        }
        report.backend_name = window_surface->BackendName();

        // STEP 1: create a present-capable renderer. We need a VkInstance with the
        // surface extensions BEFORE we can create the VkSurfaceKHR -- and the
        // device's present-queue check needs the surface. We resolve the
        // chicken-and-egg by: (a) constructing the renderer present_capable WITHOUT
        // a probe surface (enables instance/device surface+swapchain ext, picks a
        // graphics+swapchain device), (b) creating the surface on that instance,
        // (c) verifying present support on the chosen queue family. lavapipe's
        // single graphics family also supports present, so a re-pick is unneeded;
        // we assert support and throw if (improbably) absent.
        renderer = std::make_unique<VulkanRasterRenderer>(
            RendererConfig{/*present_capable=*/true, /*surface=*/nullptr});
        vk = renderer->VulkanHandles();
        device = reinterpret_cast<VkDevice>(vk.device);
        physical = reinterpret_cast<VkPhysicalDevice>(vk.physical_device);
        graphics_queue = reinterpret_cast<VkQueue>(vk.graphics_queue);
        present_queue = reinterpret_cast<VkQueue>(vk.present_queue);
        report.device_name = renderer->DeviceName();

        // STEP 2: surface.
        auto raw = window_surface->CreateSurface(
            reinterpret_cast<window::WindowVkInstance>(vk.instance));
        if (raw == nullptr) {
            throw std::runtime_error("PresentRenderer: vkCreateSurface failed");
        }
        surface = reinterpret_cast<VkSurfaceKHR>(raw);

        // STEP 3: present support on the graphics family.
        VkBool32 present_support = VK_FALSE;
        CheckVk(vkGetPhysicalDeviceSurfaceSupportKHR(physical, vk.graphics_family, surface,
                                                     &present_support),
                "vkGetPhysicalDeviceSurfaceSupportKHR");
        if (present_support != VK_TRUE) {
            throw std::runtime_error(
                "PresentRenderer: the graphics queue family does not support present on this surface");
        }

        CreateCommandPool();
        // Shader modules are extent/format-independent -> create ONCE here (not in
        // CreatePipeline, which reruns on every swapchain recreate). They are freed
        // only in ~Impl. This avoids orphaning 2 modules per resize on the
        // long-running viewer (CreatePipeline used to recreate them unconditionally
        // while DestroySwapchainDependents never freed them).
        CreateShaderModules();
        NegotiateSurface();
        CreateSwapchain();
        CreatePresentRenderPass();
        CreateDepthResources();
        CreateFramebuffers();
        CreatePipeline();
        CreateSceneDescriptors();
        CreateSyncObjects();
        AllocateCommandBuffers();

        report.swapchain_image_count = static_cast<uint32_t>(images.size());
        report.surface_format = static_cast<int32_t>(surface_format);
        report.present_mode = static_cast<int32_t>(present_mode);
        report.width = extent.width;
        report.height = extent.height;
    }

    ~Impl() {
        if (device != VK_NULL_HANDLE) vkDeviceWaitIdle(device);
        DestroySwapchainDependents();
        for (uint32_t i = 0; i < kFramesInFlight; ++i) {
            if (image_available[i] != VK_NULL_HANDLE) vkDestroySemaphore(device, image_available[i], nullptr);
            if (render_finished[i] != VK_NULL_HANDLE) vkDestroySemaphore(device, render_finished[i], nullptr);
            if (in_flight[i] != VK_NULL_HANDLE) vkDestroyFence(device, in_flight[i], nullptr);
        }
        for (auto& slot : ubo_slots) {
            if (slot.buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, slot.buffer, nullptr);
            if (slot.memory != VK_NULL_HANDLE) vkFreeMemory(device, slot.memory, nullptr);
        }
        if (scene_pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, scene_pool, nullptr);
        if (scene_set_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, scene_set_layout, nullptr);
        if (vert_module != VK_NULL_HANDLE) vkDestroyShaderModule(device, vert_module, nullptr);
        if (frag_module != VK_NULL_HANDLE) vkDestroyShaderModule(device, frag_module, nullptr);
        if (instanced_vert_module != VK_NULL_HANDLE) vkDestroyShaderModule(device, instanced_vert_module, nullptr);
        if (command_pool != VK_NULL_HANDLE) vkDestroyCommandPool(device, command_pool, nullptr);
        // ORDERED teardown (the VkSurfaceKHR is owned by window_surface but was
        // created against the renderer's VkInstance): the surface MUST be destroyed
        // while that instance is still alive. Default member destruction would
        // destroy `renderer` (the instance owner, declared 2nd) BEFORE
        // `window_surface` (declared 1st) -> destroying the surface against a freed
        // instance (a teardown segfault). So we reset them explicitly in the
        // correct order here: window_surface (frees VkSurfaceKHR) FIRST, then the
        // renderer (frees VkDevice + VkInstance).
        window_surface.reset();
        renderer.reset();
    }

    void CreateCommandPool() {
        VkCommandPoolCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        info.queueFamilyIndex = vk.graphics_family;
        CheckVk(vkCreateCommandPool(device, &info, nullptr, &command_pool),
                "vkCreateCommandPool(present)");
    }

    // Defensive: query formats + present modes + capabilities; pick conservatively.
    void NegotiateSurface() {
        uint32_t format_count = 0;
        CheckVk(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &format_count, nullptr),
                "vkGetPhysicalDeviceSurfaceFormatsKHR(count)");
        if (format_count == 0u) {
            throw std::runtime_error("PresentRenderer: surface advertises no formats");
        }
        std::vector<VkSurfaceFormatKHR> formats(format_count);
        CheckVk(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &format_count, formats.data()),
                "vkGetPhysicalDeviceSurfaceFormatsKHR(list)");

        // Preference order: B8G8R8A8/R8G8B8A8 UNORM FIRST, then the _SRGB picks,
        // then whatever the driver lists first (or the wildcard if it returns
        // VK_FORMAT_UNDEFINED meaning "any"). UNORM is preferred deliberately: the
        // SHARED mesh_pbr.frag ALREADY applies LinearToSrgb manually (matching the
        // offscreen UNORM oracle), so a true _SRGB swapchain would double-encode
        // (washed-out present). Picking UNORM keeps the shader's manual encode the
        // single authoritative sRGB step -> present matches the offscreen byte
        // semantics. The color_space stays VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
        // (display expects sRGB-encoded bytes, which the shader produces).
        auto pick = [&](VkFormat fmt, VkColorSpaceKHR space) -> bool {
            for (const auto& f : formats) {
                if (f.format == fmt && f.colorSpace == space) {
                    surface_format = f.format;
                    color_space = f.colorSpace;
                    return true;
                }
            }
            return false;
        };
        const VkColorSpaceKHR srgb = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
        if (formats.size() == 1 && formats[0].format == VK_FORMAT_UNDEFINED) {
            surface_format = VK_FORMAT_B8G8R8A8_UNORM;
            color_space = srgb;
        } else if (!pick(VK_FORMAT_B8G8R8A8_UNORM, srgb) &&
                   !pick(VK_FORMAT_R8G8B8A8_UNORM, srgb) &&
                   !pick(VK_FORMAT_B8G8R8A8_SRGB, srgb) &&
                   !pick(VK_FORMAT_R8G8B8A8_SRGB, srgb)) {
            surface_format = formats[0].format;
            color_space = formats[0].colorSpace;
        }

        // Present mode: FIFO is guaranteed present; prefer it (vsync, no tearing).
        uint32_t mode_count = 0;
        CheckVk(vkGetPhysicalDeviceSurfacePresentModesKHR(physical, surface, &mode_count, nullptr),
                "vkGetPhysicalDeviceSurfacePresentModesKHR(count)");
        std::vector<VkPresentModeKHR> modes(mode_count);
        if (mode_count != 0u) {
            CheckVk(vkGetPhysicalDeviceSurfacePresentModesKHR(physical, surface, &mode_count,
                                                              modes.data()),
                    "vkGetPhysicalDeviceSurfacePresentModesKHR(list)");
        }
        present_mode = VK_PRESENT_MODE_FIFO_KHR;  // always supported
        // (FIFO chosen deliberately; mailbox/immediate left for a future toggle.)
    }

    VkExtent2D ChooseExtent(const VkSurfaceCapabilitiesKHR& caps) {
        if (caps.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return caps.currentExtent;
        }
        VkExtent2D actual{window_surface->Width(), window_surface->Height()};
        actual.width = std::clamp(actual.width, caps.minImageExtent.width, caps.maxImageExtent.width);
        actual.height = std::clamp(actual.height, caps.minImageExtent.height, caps.maxImageExtent.height);
        if (actual.width == 0u) actual.width = std::max(1u, caps.minImageExtent.width);
        if (actual.height == 0u) actual.height = std::max(1u, caps.minImageExtent.height);
        return actual;
    }

    void CreateSwapchain() {
        VkSurfaceCapabilitiesKHR caps{};
        CheckVk(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface, &caps),
                "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
        extent = ChooseExtent(caps);

        uint32_t image_count = caps.minImageCount + 1u;
        if (caps.maxImageCount > 0u && image_count > caps.maxImageCount) {
            image_count = caps.maxImageCount;
        }
        image_count = std::max(image_count, caps.minImageCount);
        min_image_count = caps.minImageCount;

        VkSwapchainCreateInfoKHR info{};
        info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        info.surface = surface;
        info.minImageCount = image_count;
        info.imageFormat = surface_format;
        info.imageColorSpace = color_space;
        info.imageExtent = extent;
        info.imageArrayLayers = 1u;
        info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;  // single graphics+present family
        info.preTransform = caps.currentTransform;
        info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        info.presentMode = present_mode;
        info.clipped = VK_TRUE;
        info.oldSwapchain = VK_NULL_HANDLE;
        CheckVk(vkCreateSwapchainKHR(device, &info, nullptr, &swapchain), "vkCreateSwapchainKHR");

        uint32_t count = 0;
        CheckVk(vkGetSwapchainImagesKHR(device, swapchain, &count, nullptr),
                "vkGetSwapchainImagesKHR(count)");
        images.resize(count);
        CheckVk(vkGetSwapchainImagesKHR(device, swapchain, &count, images.data()),
                "vkGetSwapchainImagesKHR(list)");

        image_views.resize(count);
        for (uint32_t i = 0; i < count; ++i) {
            VkImageViewCreateInfo view_info{};
            view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            view_info.image = images[i];
            view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view_info.format = surface_format;
            view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            view_info.subresourceRange.levelCount = 1u;
            view_info.subresourceRange.layerCount = 1u;
            CheckVk(vkCreateImageView(device, &view_info, nullptr, &image_views[i]),
                    "vkCreateImageView(swapchain)");
        }
        images_in_flight.assign(count, VK_NULL_HANDLE);
    }

    void CreatePresentRenderPass() {
        std::array<VkAttachmentDescription, 2> attachments{};
        attachments[0].format = surface_format;
        attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;  // <- the present split

        attachments[1].format = kDepthFormat;
        attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference color_ref{0u, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference depth_ref{1u, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1u;
        subpass.pColorAttachments = &color_ref;
        subpass.pDepthStencilAttachment = &depth_ref;

        // Two dependencies: one for the color/depth acquire->write transition, one
        // for the present (color attachment -> present src). Keeps the swapchain
        // image layout transitions correct against the acquire semaphore.
        std::array<VkSubpassDependency, 2> deps{};
        deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        deps[0].dstSubpass = 0u;
        deps[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                               VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                               VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        deps[0].srcAccessMask = 0u;
        deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        deps[1].srcSubpass = 0u;
        deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        deps[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        deps[1].dstAccessMask = 0u;

        VkRenderPassCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        info.attachmentCount = static_cast<uint32_t>(attachments.size());
        info.pAttachments = attachments.data();
        info.subpassCount = 1u;
        info.pSubpasses = &subpass;
        info.dependencyCount = static_cast<uint32_t>(deps.size());
        info.pDependencies = deps.data();
        CheckVk(vkCreateRenderPass(device, &info, nullptr, &present_pass),
                "vkCreateRenderPass(present)");
    }

    void CreateDepthResources() {
        VkImageCreateInfo image_info{};
        image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image_info.imageType = VK_IMAGE_TYPE_2D;
        image_info.format = kDepthFormat;
        image_info.extent = {extent.width, extent.height, 1u};
        image_info.mipLevels = 1u;
        image_info.arrayLayers = 1u;
        image_info.samples = VK_SAMPLE_COUNT_1_BIT;
        image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        image_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        CheckVk(vkCreateImage(device, &image_info, nullptr, &depth_image), "vkCreateImage(depth)");

        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(device, depth_image, &req);
        VkMemoryAllocateInfo alloc{};
        alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc.allocationSize = req.size;
        alloc.memoryTypeIndex = FindMemoryType(physical, req.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        CheckVk(vkAllocateMemory(device, &alloc, nullptr, &depth_memory), "vkAllocateMemory(depth)");
        CheckVk(vkBindImageMemory(device, depth_image, depth_memory, 0), "vkBindImageMemory(depth)");

        VkImageViewCreateInfo view_info{};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image = depth_image;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = kDepthFormat;
        view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        view_info.subresourceRange.levelCount = 1u;
        view_info.subresourceRange.layerCount = 1u;
        CheckVk(vkCreateImageView(device, &view_info, nullptr, &depth_view), "vkCreateImageView(depth)");
    }

    void CreateFramebuffers() {
        framebuffers.resize(image_views.size());
        for (size_t i = 0; i < image_views.size(); ++i) {
            std::array<VkImageView, 2> views{image_views[i], depth_view};
            VkFramebufferCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            info.renderPass = present_pass;
            info.attachmentCount = static_cast<uint32_t>(views.size());
            info.pAttachments = views.data();
            info.width = extent.width;
            info.height = extent.height;
            info.layers = 1u;
            CheckVk(vkCreateFramebuffer(device, &info, nullptr, &framebuffers[i]),
                    "vkCreateFramebuffer(present)");
        }
    }

    VkShaderModule CreateShaderModule(const std::string& path) {
        const auto bytes = ReadBinaryFile(path);
        if ((bytes.size() % sizeof(uint32_t)) != 0u) {
            throw std::runtime_error("Vulkan shader bytecode has invalid size: " + path);
        }
        VkShaderModuleCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        info.codeSize = bytes.size();
        info.pCode = reinterpret_cast<const uint32_t*>(bytes.data());
        VkShaderModule module = VK_NULL_HANDLE;
        CheckVk(vkCreateShaderModule(device, &info, nullptr, &module), "vkCreateShaderModule(present)");
        return module;
    }

    // Create the (extent/format-independent) vert+frag modules once at ctor time.
    // Guarded so it is a no-op if somehow re-entered. Destroyed in ~Impl.
    void CreateShaderModules() {
        if (vert_module == VK_NULL_HANDLE) {
            vert_module = CreateShaderModule(NUKA_RASTER_MESH_VERT_SPV);
        }
        if (frag_module == VK_NULL_HANDLE) {
            frag_module = CreateShaderModule(NUKA_RASTER_MESH_FRAG_SPV);
        }
    }

    void CreatePipeline() {
        // Same shaders + push-block layout as the offscreen pipeline, but built
        // against the PRESENT render pass (different attachment format -> a
        // separate, render-pass-compatible pipeline). The shader modules are
        // created once (CreateShaderModules at ctor) and REUSED across swapchain
        // recreates -> no per-resize module leak.

        // Set 0, binding 0: per-frame SceneUbo (camera + light rig). Same layout
        // as the offscreen renderer (shared shaders).
        VkDescriptorSetLayoutBinding ubo_binding{};
        ubo_binding.binding = 0u;
        ubo_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        ubo_binding.descriptorCount = 1u;
        ubo_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo set_info{};
        set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        set_info.bindingCount = 1u;
        set_info.pBindings = &ubo_binding;
        CheckVk(vkCreateDescriptorSetLayout(device, &set_info, nullptr, &scene_set_layout),
                "vkCreateDescriptorSetLayout(present-scene)");

        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        push_range.offset = 0u;
        push_range.size = sizeof(PushBlock);
        VkPipelineLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout_info.setLayoutCount = 1u;
        layout_info.pSetLayouts = &scene_set_layout;
        layout_info.pushConstantRangeCount = 1u;
        layout_info.pPushConstantRanges = &push_range;
        CheckVk(vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout),
                "vkCreatePipelineLayout(present)");

        std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vert_module;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = frag_module;
        stages[1].pName = "main";

        std::array<VkVertexInputBindingDescription, 2> bindings{};
        bindings[0].binding = 0u; bindings[0].stride = sizeof(float) * 3u;
        bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        bindings[1].binding = 1u; bindings[1].stride = sizeof(float) * 3u;
        bindings[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        std::array<VkVertexInputAttributeDescription, 2> attrs{};
        attrs[0].location = 0u; attrs[0].binding = 0u;
        attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT; attrs[0].offset = 0u;
        attrs[1].location = 1u; attrs[1].binding = 1u;
        attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT; attrs[1].offset = 0u;
        VkPipelineVertexInputStateCreateInfo vertex_input{};
        vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertex_input.vertexBindingDescriptionCount = static_cast<uint32_t>(bindings.size());
        vertex_input.pVertexBindingDescriptions = bindings.data();
        vertex_input.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size());
        vertex_input.pVertexAttributeDescriptions = attrs.data();

        VkPipelineInputAssemblyStateCreateInfo input_assembly{};
        input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo viewport_state{};
        viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport_state.viewportCount = 1u;
        viewport_state.scissorCount = 1u;

        VkPipelineRasterizationStateCreateInfo rasterization{};
        rasterization.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterization.polygonMode = VK_POLYGON_MODE_FILL;
        rasterization.cullMode = VK_CULL_MODE_NONE;
        rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterization.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo multisample{};
        multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depth_stencil{};
        depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depth_stencil.depthTestEnable = VK_TRUE;
        depth_stencil.depthWriteEnable = VK_TRUE;
        depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineColorBlendAttachmentState blend_attachment{};
        blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        blend_attachment.blendEnable = VK_FALSE;
        VkPipelineColorBlendStateCreateInfo color_blend{};
        color_blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        color_blend.attachmentCount = 1u;
        color_blend.pAttachments = &blend_attachment;

        std::array<VkDynamicState, 2> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT,
                                                     VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic_state{};
        dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic_state.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
        dynamic_state.pDynamicStates = dynamic_states.data();

        VkGraphicsPipelineCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        info.stageCount = static_cast<uint32_t>(stages.size());
        info.pStages = stages.data();
        info.pVertexInputState = &vertex_input;
        info.pInputAssemblyState = &input_assembly;
        info.pViewportState = &viewport_state;
        info.pRasterizationState = &rasterization;
        info.pMultisampleState = &multisample;
        info.pDepthStencilState = &depth_stencil;
        info.pColorBlendState = &color_blend;
        info.pDynamicState = &dynamic_state;
        info.layout = pipeline_layout;
        info.renderPass = present_pass;
        info.subpass = 0u;
        CheckVk(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1u, &info, nullptr, &pipeline),
                "vkCreateGraphicsPipelines(present)");
    }

    // INT-F1: build the OPT-IN instanced pipeline (mesh_instanced.vert reads the
    // per-instance model matrix from the interop SSBO at set 1 by gl_InstanceIndex).
    // Requires scene_set_layout (set 0) + the borrowed ssbo_set_layout (set 1) to be
    // valid -> called by SetInteropTransforms after CreatePipeline, and by
    // RecreateSwapchain when interop is active. Idempotent-safe: destroys any prior
    // instanced pipeline/layout first. Shares the offscreen frag (mesh_pbr.frag) +
    // the same PushBlock + the same fixed-function state as the default pipeline, so
    // the ONLY difference is the model-matrix source (SSBO vs push constant).
    void CreateInstancedPipeline() {
        if (ssbo_set_layout == VK_NULL_HANDLE || scene_set_layout == VK_NULL_HANDLE) return;
        if (instanced_vert_module == VK_NULL_HANDLE) {
            instanced_vert_module = CreateShaderModule(NUKA_RASTER_MESH_INSTANCED_VERT_SPV);
        }
        if (instanced_pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, instanced_pipeline, nullptr); instanced_pipeline = VK_NULL_HANDLE;
        }
        if (instanced_pipeline_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device, instanced_pipeline_layout, nullptr);
            instanced_pipeline_layout = VK_NULL_HANDLE;
        }

        // Pipeline layout: set 0 = SceneUbo, set 1 = the interop transform SSBO.
        const std::array<VkDescriptorSetLayout, 2> set_layouts{scene_set_layout, ssbo_set_layout};
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        push_range.offset = 0u;
        push_range.size = sizeof(PushBlock);
        VkPipelineLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout_info.setLayoutCount = static_cast<uint32_t>(set_layouts.size());
        layout_info.pSetLayouts = set_layouts.data();
        layout_info.pushConstantRangeCount = 1u;
        layout_info.pPushConstantRanges = &push_range;
        CheckVk(vkCreatePipelineLayout(device, &layout_info, nullptr, &instanced_pipeline_layout),
                "vkCreatePipelineLayout(present-instanced)");

        std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = instanced_vert_module;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = frag_module;
        stages[1].pName = "main";

        std::array<VkVertexInputBindingDescription, 2> bindings{};
        bindings[0].binding = 0u; bindings[0].stride = sizeof(float) * 3u;
        bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        bindings[1].binding = 1u; bindings[1].stride = sizeof(float) * 3u;
        bindings[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        std::array<VkVertexInputAttributeDescription, 2> attrs{};
        attrs[0].location = 0u; attrs[0].binding = 0u;
        attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT; attrs[0].offset = 0u;
        attrs[1].location = 1u; attrs[1].binding = 1u;
        attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT; attrs[1].offset = 0u;
        VkPipelineVertexInputStateCreateInfo vertex_input{};
        vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertex_input.vertexBindingDescriptionCount = static_cast<uint32_t>(bindings.size());
        vertex_input.pVertexBindingDescriptions = bindings.data();
        vertex_input.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size());
        vertex_input.pVertexAttributeDescriptions = attrs.data();

        VkPipelineInputAssemblyStateCreateInfo input_assembly{};
        input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo viewport_state{};
        viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport_state.viewportCount = 1u;
        viewport_state.scissorCount = 1u;

        VkPipelineRasterizationStateCreateInfo rasterization{};
        rasterization.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterization.polygonMode = VK_POLYGON_MODE_FILL;
        rasterization.cullMode = VK_CULL_MODE_NONE;
        rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterization.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo multisample{};
        multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depth_stencil{};
        depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depth_stencil.depthTestEnable = VK_TRUE;
        depth_stencil.depthWriteEnable = VK_TRUE;
        depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineColorBlendAttachmentState blend_attachment{};
        blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        blend_attachment.blendEnable = VK_FALSE;
        VkPipelineColorBlendStateCreateInfo color_blend{};
        color_blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        color_blend.attachmentCount = 1u;
        color_blend.pAttachments = &blend_attachment;

        std::array<VkDynamicState, 2> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT,
                                                     VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic_state{};
        dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic_state.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
        dynamic_state.pDynamicStates = dynamic_states.data();

        VkGraphicsPipelineCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        info.stageCount = static_cast<uint32_t>(stages.size());
        info.pStages = stages.data();
        info.pVertexInputState = &vertex_input;
        info.pInputAssemblyState = &input_assembly;
        info.pViewportState = &viewport_state;
        info.pRasterizationState = &rasterization;
        info.pMultisampleState = &multisample;
        info.pDepthStencilState = &depth_stencil;
        info.pColorBlendState = &color_blend;
        info.pDynamicState = &dynamic_state;
        info.layout = instanced_pipeline_layout;
        info.renderPass = present_pass;
        info.subpass = 0u;
        CheckVk(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1u, &info, nullptr, &instanced_pipeline),
                "vkCreateGraphicsPipelines(present-instanced)");
    }

    // Persistent per-frame-in-flight SceneUbo buffers + descriptor sets.
    void CreateSceneDescriptors() {
        const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kFramesInFlight};
        VkDescriptorPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.maxSets = kFramesInFlight;
        pool_info.poolSizeCount = 1u;
        pool_info.pPoolSizes = &size;
        CheckVk(vkCreateDescriptorPool(device, &pool_info, nullptr, &scene_pool),
                "vkCreateDescriptorPool(present-scene)");
        for (uint32_t i = 0; i < kFramesInFlight; ++i) {
            UboSlot& slot = ubo_slots[i];
            const GpuBuffer b = CreateHostBuffer(sizeof(SceneUbo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
            slot.buffer = b.buffer;
            slot.memory = b.memory;
            VkDescriptorSetAllocateInfo alloc{};
            alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            alloc.descriptorPool = scene_pool;
            alloc.descriptorSetCount = 1u;
            alloc.pSetLayouts = &scene_set_layout;
            CheckVk(vkAllocateDescriptorSets(device, &alloc, &slot.set),
                    "vkAllocateDescriptorSets(present-scene)");
            VkDescriptorBufferInfo buffer_info{slot.buffer, 0u, sizeof(SceneUbo)};
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = slot.set;
            write.dstBinding = 0u;
            write.descriptorCount = 1u;
            write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            write.pBufferInfo = &buffer_info;
            vkUpdateDescriptorSets(device, 1u, &write, 0u, nullptr);
        }
    }

    void CreateSyncObjects() {
        VkSemaphoreCreateInfo sem_info{};
        sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;  // first wait passes
        for (uint32_t i = 0; i < kFramesInFlight; ++i) {
            CheckVk(vkCreateSemaphore(device, &sem_info, nullptr, &image_available[i]),
                    "vkCreateSemaphore(imageAvailable)");
            CheckVk(vkCreateSemaphore(device, &sem_info, nullptr, &render_finished[i]),
                    "vkCreateSemaphore(renderFinished)");
            CheckVk(vkCreateFence(device, &fence_info, nullptr, &in_flight[i]),
                    "vkCreateFence(inFlight)");
        }
    }

    void AllocateCommandBuffers() {
        VkCommandBufferAllocateInfo alloc{};
        alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc.commandPool = command_pool;
        alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc.commandBufferCount = kFramesInFlight;
        CheckVk(vkAllocateCommandBuffers(device, &alloc, command_buffers.data()),
                "vkAllocateCommandBuffers(present)");
    }

    void DestroySwapchainDependents() {
        for (VkFramebuffer fb : framebuffers) {
            if (fb != VK_NULL_HANDLE) vkDestroyFramebuffer(device, fb, nullptr);
        }
        framebuffers.clear();
        if (depth_view != VK_NULL_HANDLE) { vkDestroyImageView(device, depth_view, nullptr); depth_view = VK_NULL_HANDLE; }
        if (depth_image != VK_NULL_HANDLE) { vkDestroyImage(device, depth_image, nullptr); depth_image = VK_NULL_HANDLE; }
        if (depth_memory != VK_NULL_HANDLE) { vkFreeMemory(device, depth_memory, nullptr); depth_memory = VK_NULL_HANDLE; }
        if (pipeline != VK_NULL_HANDLE) { vkDestroyPipeline(device, pipeline, nullptr); pipeline = VK_NULL_HANDLE; }
        if (pipeline_layout != VK_NULL_HANDLE) { vkDestroyPipelineLayout(device, pipeline_layout, nullptr); pipeline_layout = VK_NULL_HANDLE; }
        // INT-F1: the instanced pipeline + its layout are render-pass-dependent too
        // (built against present_pass) -> destroy here so a swapchain recreate rebuilds
        // them. The instanced_vert_module + the BORROWED ssbo_set/layout survive.
        if (instanced_pipeline != VK_NULL_HANDLE) { vkDestroyPipeline(device, instanced_pipeline, nullptr); instanced_pipeline = VK_NULL_HANDLE; }
        if (instanced_pipeline_layout != VK_NULL_HANDLE) { vkDestroyPipelineLayout(device, instanced_pipeline_layout, nullptr); instanced_pipeline_layout = VK_NULL_HANDLE; }
        // scene_set_layout is (re)created by CreatePipeline; destroy it here so a
        // swapchain recreate does not leak it. The persistent scene_pool + ubo_slots
        // sets stay valid (a byte-identical re-created layout remains bind-compatible).
        if (scene_set_layout != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(device, scene_set_layout, nullptr); scene_set_layout = VK_NULL_HANDLE; }
        if (present_pass != VK_NULL_HANDLE) { vkDestroyRenderPass(device, present_pass, nullptr); present_pass = VK_NULL_HANDLE; }
        for (VkImageView v : image_views) {
            if (v != VK_NULL_HANDLE) vkDestroyImageView(device, v, nullptr);
        }
        image_views.clear();
        images.clear();
        if (swapchain != VK_NULL_HANDLE) { vkDestroySwapchainKHR(device, swapchain, nullptr); swapchain = VK_NULL_HANDLE; }
    }

    void RecreateSwapchain() {
        vkDeviceWaitIdle(device);
        DestroySwapchainDependents();
        NegotiateSurface();
        CreateSwapchain();
        CreatePresentRenderPass();
        CreateDepthResources();
        CreateFramebuffers();
        CreatePipeline();
        // INT-F1: rebuild the instanced pipeline against the NEW present pass when
        // interop is active (DestroySwapchainDependents tore the old one down).
        if (interop_draw && ssbo_set_layout != VK_NULL_HANDLE) {
            CreateInstancedPipeline();
        }
        report.swapchain_image_count = static_cast<uint32_t>(images.size());
        report.width = extent.width;
        report.height = extent.height;
    }

    // -- per-frame transient geometry upload (mirrors the offscreen draw) ------
    GpuBuffer CreateHostBuffer(VkDeviceSize size, VkBufferUsageFlags usage) {
        GpuBuffer res;
        VkBufferCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size = size;
        info.usage = usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        CheckVk(vkCreateBuffer(device, &info, nullptr, &res.buffer), "vkCreateBuffer(present)");
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(device, res.buffer, &req);
        VkMemoryAllocateInfo alloc{};
        alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc.allocationSize = req.size;
        alloc.memoryTypeIndex = FindMemoryType(physical, req.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        CheckVk(vkAllocateMemory(device, &alloc, nullptr, &res.memory), "vkAllocateMemory(present buf)");
        CheckVk(vkBindBufferMemory(device, res.buffer, res.memory, 0), "vkBindBufferMemory(present)");
        return res;
    }

    void Upload(const GpuBuffer& buf, const void* data, size_t size) {
        if (size == 0u) return;
        void* mapped = nullptr;
        CheckVk(vkMapMemory(device, buf.memory, 0, static_cast<VkDeviceSize>(size), 0, &mapped),
                "vkMapMemory(present upload)");
        std::memcpy(mapped, data, size);
        vkUnmapMemory(device, buf.memory);
    }

    void Destroy(GpuBuffer& buf) {
        if (buf.buffer != VK_NULL_HANDLE) { vkDestroyBuffer(device, buf.buffer, nullptr); buf.buffer = VK_NULL_HANDLE; }
        if (buf.memory != VK_NULL_HANDLE) { vkFreeMemory(device, buf.memory, nullptr); buf.memory = VK_NULL_HANDLE; }
    }

    struct InstanceDraw {
        GpuBuffer pos, nrm, idx;
        uint32_t  index_count = 0;
        Mat4      model;
        float     base_color[4] = {0.8f, 0.8f, 0.8f, 1.0f};
        float     metallic = 0.0f;
        float     roughness = 0.5f;
        float     opacity = 1.0f;
        float     emissive[3] = {0.0f, 0.0f, 0.0f};
        // INT-F1: the ORIGINAL index into world.instances (== the scatter SSBO row).
        // Used as firstInstance so gl_InstanceIndex selects the right SSBO mat4 on
        // the instanced path (the draw loop skips geometry-less instances, so the
        // compacted draw index would not match the SSBO row).
        uint32_t  instance_row = 0;
    };

    PresentFrameResult DrawFrame(const RenderWorld& world, const RasterOptions& options,
                                 const OverlayRecordFn& overlay) {
        const uint32_t frame = current_frame;
        CheckVk(vkWaitForFences(device, 1u, &in_flight[frame], VK_TRUE,
                                std::numeric_limits<uint64_t>::max()),
                "vkWaitForFences");

        uint32_t image_index = 0;
        const VkResult acquire = vkAcquireNextImageKHR(
            device, swapchain, std::numeric_limits<uint64_t>::max(),
            image_available[frame], VK_NULL_HANDLE, &image_index);
        if (acquire == VK_ERROR_OUT_OF_DATE_KHR) {
            RecreateSwapchain();
            return PresentFrameResult::Recreated;
        }
        if (acquire != VK_SUCCESS && acquire != VK_SUBOPTIMAL_KHR) {
            return PresentFrameResult::Error;
        }

        // If a prior frame is still using this image, wait on its fence.
        if (images_in_flight[image_index] != VK_NULL_HANDLE) {
            vkWaitForFences(device, 1u, &images_in_flight[image_index], VK_TRUE,
                            std::numeric_limits<uint64_t>::max());
        }
        images_in_flight[image_index] = in_flight[frame];

        // -- build per-instance geometry + scene AABB (same as the offscreen path).
        std::vector<InstanceDraw> draws;
        draws.reserve(world.instances.size());
        math::Vec3 aabb_min{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                            std::numeric_limits<float>::max()};
        math::Vec3 aabb_max{-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
                            -std::numeric_limits<float>::max()};
        bool has_geometry = false;
        const uint32_t instance_total = static_cast<uint32_t>(world.instances.size());
        for (uint32_t inst_idx = 0; inst_idx < instance_total; ++inst_idx) {
            const RenderInstance& inst = world.instances[inst_idx];
            if (inst.mesh_id == kNoId || inst.mesh_id >= world.meshes.Count()) continue;
            const MeshGeometry& geo = world.meshes.Geometry(inst.mesh_id);
            if (geo.Empty()) continue;
            InstanceDraw draw;
            draw.instance_row = inst_idx;
            draw.model = TransformToMatrix(inst.world_xform);
            if (inst.render_material_id != kNoId &&
                inst.render_material_id < world.materials.size()) {
                const scene::RenderMaterial& mat = world.materials[inst.render_material_id];
                draw.base_color[0] = mat.base_color[0];
                draw.base_color[1] = mat.base_color[1];
                draw.base_color[2] = mat.base_color[2];
                draw.base_color[3] = mat.base_color[3];
                draw.metallic = mat.metallic;
                draw.roughness = mat.roughness;
                draw.opacity = mat.opacity;
                draw.emissive[0] = mat.emissive[0];
                draw.emissive[1] = mat.emissive[1];
                draw.emissive[2] = mat.emissive[2];
            }
            const VkDeviceSize pos_bytes = geo.positions.size() * sizeof(float);
            draw.pos = CreateHostBuffer(pos_bytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            Upload(draw.pos, geo.positions.data(), pos_bytes);
            std::vector<float> normals;
            const std::vector<float>* normal_src = &geo.normals;
            if (geo.normals.size() != geo.positions.size()) {
                normals = SynthesizeFlatNormals(geo.positions, geo.indices);
                normal_src = &normals;
            }
            const VkDeviceSize nrm_bytes = normal_src->size() * sizeof(float);
            draw.nrm = CreateHostBuffer(nrm_bytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            Upload(draw.nrm, normal_src->data(), nrm_bytes);
            const VkDeviceSize idx_bytes = geo.indices.size() * sizeof(uint32_t);
            draw.idx = CreateHostBuffer(idx_bytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
            Upload(draw.idx, geo.indices.data(), idx_bytes);
            draw.index_count = static_cast<uint32_t>(geo.indices.size());
            for (size_t v = 0; v + 2 < geo.positions.size(); v += 3) {
                const math::Vec3 local{geo.positions[v], geo.positions[v + 1], geo.positions[v + 2]};
                const math::Vec3 wp = inst.world_xform.TransformPoint(local);
                aabb_min.x = std::min(aabb_min.x, wp.x); aabb_min.y = std::min(aabb_min.y, wp.y);
                aabb_min.z = std::min(aabb_min.z, wp.z); aabb_max.x = std::max(aabb_max.x, wp.x);
                aabb_max.y = std::max(aabb_max.y, wp.y); aabb_max.z = std::max(aabb_max.z, wp.z);
                has_geometry = true;
            }
            draws.push_back(std::move(draw));
        }

        const ResolvedCamera cam = ResolveCamera(world, options, aabb_min, aabb_max, has_geometry);
        const float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
        const float fov_y = cam.fov_degrees * 3.14159265358979323846f / 180.0f;
        const Mat4 view = LookAt(cam.eye, cam.target, cam.up);
        const Mat4 proj = Perspective(fov_y, aspect, cam.near_clip, cam.far_clip);
        const Mat4 view_proj = Multiply(proj, view);

        // Upload this frame's SceneUbo (camera + light rig) into the frame slot.
        const SceneUbo scene_ubo = BuildSceneUbo(world, view_proj, cam.eye);
        {
            void* mapped = nullptr;
            CheckVk(vkMapMemory(device, ubo_slots[frame].memory, 0, sizeof(SceneUbo), 0, &mapped),
                    "vkMapMemory(present-scene-ubo)");
            std::memcpy(mapped, &scene_ubo, sizeof(SceneUbo));
            vkUnmapMemory(device, ubo_slots[frame].memory);
        }

        // -- record ----------------------------------------------------------
        VkCommandBuffer cmd = command_buffers[frame];
        CheckVk(vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer");
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        CheckVk(vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer(present)");

        std::array<VkClearValue, 2> clears{};
        clears[0].color = {{static_cast<float>(options.background.r) / 255.0f,
                            static_cast<float>(options.background.g) / 255.0f,
                            static_cast<float>(options.background.b) / 255.0f,
                            static_cast<float>(options.background.a) / 255.0f}};
        clears[1].depthStencil = {1.0f, 0u};
        VkRenderPassBeginInfo rp{};
        rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp.renderPass = present_pass;
        rp.framebuffer = framebuffers[image_index];
        rp.renderArea = {{0, 0}, extent};
        rp.clearValueCount = static_cast<uint32_t>(clears.size());
        rp.pClearValues = clears.data();
        vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

        // INT-F1: pick the pipeline + layout for this frame. The instanced path is
        // used ONLY when interop was installed AND the instanced pipeline built
        // successfully -- otherwise the DEFAULT per-draw push-constant pipeline (the
        // unchanged D1 path) draws, so a non-interop present is byte-identical.
        const bool use_instanced = interop_draw && instanced_pipeline != VK_NULL_HANDLE &&
                                   ssbo_set != VK_NULL_HANDLE;
        const VkPipeline       active_pipeline = use_instanced ? instanced_pipeline : pipeline;
        const VkPipelineLayout active_layout =
            use_instanced ? instanced_pipeline_layout : pipeline_layout;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, active_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, active_layout,
                                0u, 1u, &ubo_slots[frame].set, 0u, nullptr);
        if (use_instanced) {
            // set 1: the device-local interop transform SSBO (the scatter's output).
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, active_layout,
                                    1u, 1u, &ssbo_set, 0u, nullptr);
        }
        VkViewport viewport{};
        viewport.x = 0.0f; viewport.y = 0.0f;
        viewport.width = static_cast<float>(extent.width);
        viewport.height = static_cast<float>(extent.height);
        viewport.minDepth = 0.0f; viewport.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0u, 1u, &viewport);
        VkRect2D scissor{{0, 0}, extent};
        vkCmdSetScissor(cmd, 0u, 1u, &scissor);

        for (const InstanceDraw& draw : draws) {
            PushBlock push{};
            // On the instanced path the model matrix comes from the SSBO (the shader
            // ignores push.model); we still fill it harmlessly. Material always rides
            // the push constant on BOTH paths (materials are not device physics state).
            std::memcpy(push.model, draw.model.m.data(), sizeof(push.model));
            std::memcpy(push.base_color, draw.base_color, sizeof(push.base_color));
            push.mr[0] = draw.metallic; push.mr[1] = draw.roughness;
            push.mr[2] = draw.opacity;  push.mr[3] = 0.0f;
            push.emissive[0] = draw.emissive[0]; push.emissive[1] = draw.emissive[1];
            push.emissive[2] = draw.emissive[2]; push.emissive[3] = 0.0f;
            vkCmdPushConstants(cmd, active_layout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0u,
                               sizeof(PushBlock), &push);
            std::array<VkBuffer, 2> vbuffers{draw.pos.buffer, draw.nrm.buffer};
            std::array<VkDeviceSize, 2> offsets{0u, 0u};
            vkCmdBindVertexBuffers(cmd, 0u, 2u, vbuffers.data(), offsets.data());
            vkCmdBindIndexBuffer(cmd, draw.idx.buffer, 0u, VK_INDEX_TYPE_UINT32);
            // INT-F1: on the instanced path, firstInstance == the SSBO row so
            // gl_InstanceIndex picks the right scattered transform (the draw loop
            // skipped geometry-less instances, so the SSBO row != compacted index).
            const uint32_t first_instance = use_instanced ? draw.instance_row : 0u;
            vkCmdDrawIndexed(cmd, draw.index_count, 1u, 0u, 0, first_instance);
        }

        // Overlay (ImGui) records INSIDE the pass, after the scene.
        if (overlay) {
            overlay(reinterpret_cast<void*>(cmd));
        }

        vkCmdEndRenderPass(cmd);
        CheckVk(vkEndCommandBuffer(cmd), "vkEndCommandBuffer(present)");

        // -- submit ----------------------------------------------------------
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        submit.waitSemaphoreCount = 1u;
        submit.pWaitSemaphores = &image_available[frame];
        submit.pWaitDstStageMask = &wait_stage;
        submit.commandBufferCount = 1u;
        submit.pCommandBuffers = &cmd;
        submit.signalSemaphoreCount = 1u;
        // NOTE: render_finished is indexed by frame-in-flight, NOT by swapchain
        // image. That is only safe because this frame's per-frame fence wait
        // (vkWaitForFences at the top of DrawFrame + the free-wait below)
        // fully serializes each frame slot's submit->present before the slot is
        // reused, so a render_finished[frame] can never be in two presents at
        // once. A proper per-image render_finished[] (one per swapchain image,
        // signalled by the present-image producer) is the correct fix and is
        // deferred (M9 present-render-finished per-image variant).
        submit.pSignalSemaphores = &render_finished[frame];
        CheckVk(vkResetFences(device, 1u, &in_flight[frame]), "vkResetFences");
        CheckVk(vkQueueSubmit(graphics_queue, 1u, &submit, in_flight[frame]), "vkQueueSubmit(present)");

        // -- present ---------------------------------------------------------
        VkPresentInfoKHR present{};
        present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present.waitSemaphoreCount = 1u;
        present.pWaitSemaphores = &render_finished[frame];
        present.swapchainCount = 1u;
        present.pSwapchains = &swapchain;
        present.pImageIndices = &image_index;
        const VkResult present_result = vkQueuePresentKHR(present_queue, &present);

        // The transient per-frame geometry can be freed once the fence (set on
        // submit) is reached -- but to keep this simple + correct we wait on the
        // device-side fence for THIS frame before the NEXT use of it (the
        // vkWaitForFences at the top), and free here after a queue wait scoped to
        // the just-submitted work. For a showcase viewer the per-frame waitidle on
        // the in-flight fence is acceptable; we instead free after the present by
        // waiting this frame's fence (bounded, only this submit).
        CheckVk(vkWaitForFences(device, 1u, &in_flight[frame], VK_TRUE,
                                std::numeric_limits<uint64_t>::max()),
                "vkWaitForFences(free)");
        for (InstanceDraw& draw : draws) {
            Destroy(draw.pos); Destroy(draw.nrm); Destroy(draw.idx);
        }

        current_frame = (current_frame + 1u) % kFramesInFlight;
        ++report.frames_presented;

        if (present_result == VK_ERROR_OUT_OF_DATE_KHR || present_result == VK_SUBOPTIMAL_KHR) {
            RecreateSwapchain();
            return PresentFrameResult::Recreated;
        }
        if (present_result != VK_SUCCESS) {
            return PresentFrameResult::Error;
        }
        return PresentFrameResult::Presented;
    }
};

// ---------------------------------------------------------------------------
PresentRenderer::PresentRenderer(std::unique_ptr<window::WindowSurface> surface)
    : impl_(std::make_unique<Impl>(std::move(surface))) {}
PresentRenderer::~PresentRenderer() = default;
PresentRenderer::PresentRenderer(PresentRenderer&&) noexcept = default;
PresentRenderer& PresentRenderer::operator=(PresentRenderer&&) noexcept = default;

PresentFrameResult PresentRenderer::DrawFrame(const RenderWorld& world,
                                              const RasterOptions& options,
                                              const OverlayRecordFn& overlay) {
    return impl_->DrawFrame(world, options, overlay);
}

RendererVulkanHandles PresentRenderer::VulkanHandles() {
    RendererVulkanHandles h = impl_->vk;
    // Override the render pass with the PRESENT pass (ImGui must draw into it) +
    // refresh the image counts from the live swapchain.
    h.offscreen_render_pass = reinterpret_cast<NkVkRenderPass>(impl_->present_pass);
    return h;
}

NkVkRenderPass PresentRenderer::PresentRenderPass() const {
    return reinterpret_cast<NkVkRenderPass>(impl_->present_pass);
}

uint32_t PresentRenderer::SwapchainImageCount() const {
    return static_cast<uint32_t>(impl_->images.size());
}
uint32_t PresentRenderer::MinImageCount() const { return impl_->min_image_count; }

const PresentReport& PresentRenderer::Report() const { return impl_->report; }
const std::string& PresentRenderer::DeviceName() const { return impl_->report.device_name; }

void PresentRenderer::PollEvents(std::vector<window::WindowEvent>& out) {
    impl_->window_surface->PollEvents(out);
    for (const auto& ev : out) {
        if (ev.type == window::WindowEvent::Type::Close) impl_->should_close = true;
    }
}

bool PresentRenderer::ShouldClose() const { return impl_->should_close; }

void PresentRenderer::WaitIdle() {
    if (impl_->device != VK_NULL_HANDLE) vkDeviceWaitIdle(impl_->device);
}

void PresentRenderer::SetInteropTransforms(NkVkDescriptorSetLayout ssbo_set_layout,
                                           NkVkDescriptorSet ssbo_set) {
    // Disable when either handle is null -> revert to the default per-draw pipeline.
    if (ssbo_set_layout == nullptr || ssbo_set == nullptr) {
        impl_->interop_draw = false;
        impl_->ssbo_set_layout = VK_NULL_HANDLE;
        impl_->ssbo_set = VK_NULL_HANDLE;
        return;
    }
    impl_->ssbo_set_layout = reinterpret_cast<VkDescriptorSetLayout>(ssbo_set_layout);
    impl_->ssbo_set = reinterpret_cast<VkDescriptorSet>(ssbo_set);
    // Build the instanced pipeline against the current present pass. If it fails to
    // build for any reason, CheckVk throws; we let it propagate (the caller knows
    // interop is unsupported and can fall back). On success the next DrawFrame draws
    // the instanced path.
    impl_->CreateInstancedPipeline();
    impl_->interop_draw = (impl_->instanced_pipeline != VK_NULL_HANDLE);
}

bool PresentRenderer::InteropDrawActive() const {
    return impl_->interop_draw && impl_->instanced_pipeline != VK_NULL_HANDLE;
}

}  // namespace nuka::render
