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
#include "render/raster/present_surface.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
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
#ifndef NUKA_PRESENT_SHADER_DIR
#define NUKA_PRESENT_SHADER_DIR "."
#endif

constexpr VkFormat   kDepthFormat = VK_FORMAT_D32_SFLOAT;
constexpr VkFormat   kIdFormat = VK_FORMAT_R32_UINT;
constexpr uint32_t   kFramesInFlight = 2u;
// An id-pass texel: the instance row + 1 (0 is empty space) under the selected and hovered bits.
constexpr uint32_t   kIdSelected = 0x80000000u, kIdHovered = 0x40000000u, kIdMask = 0x3FFFFFFFu;
// The Residual and Contact views' clay albedo.
constexpr float      kClay = 0.72f;
// A see-through instance's opacity over its material's.
constexpr float      kSeeThroughOpacity = 0.35f;

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

// Mirrors the offscreen renderer's PushBlock + SceneUbo byte-for-byte (the
// mesh.vert / mesh_pbr.frag shaders are shared, so the layouts MUST agree).
struct PushBlock {
    float model[16];
    float base_color[4];
    float mr[4];            // x metallic, y roughness, z opacity, w pad
    float emissive[4];      // rgb emissive, a pad
    float ground_shadow[4]; // xy contact centre, z radius (0=off), w strength
};
static_assert(sizeof(PushBlock) == 128, "PushBlock must match the shared push range");

// outline.frag's push block.
struct OutlinePush {
    float   selected[4];
    float   hovered[4];
    int32_t rect[4];
};
static_assert(sizeof(OutlinePush) == 48, "OutlinePush must match outline.frag");

// A stable mid-bright colour per material id for the Material view.
void MaterialColor(uint32_t material_id, float out[4]) {
    uint32_t h = (material_id + 1u) * 0x9E3779B1u;
    h ^= h >> 16; h *= 0x85EBCA6Bu; h ^= h >> 13;
    for (int c = 0; c < 3; ++c) out[c] = 0.25f + 0.7f * static_cast<float>((h >> (8 * c)) & 0xFFu) / 255.0f;
    out[3] = 1.0f;
}

// Sizes SceneUbo.lights[]; MUST equal the GLSL NUKA_MAX_LIGHTS in
// render/raster/shaders/light_limits.glsl (the single shader-side source).
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
    float    fog[4];               // rgb haze, w density (0 => fog OFF)
    GpuLight lights[kMaxUboLights];
    float    light_view_proj[16];  // sun world->clip (shadows; unused here)
    float    shadow_params[4];     // x strength (0 => OFF), y bias, z texel, w pad
};
static_assert(sizeof(SceneUbo) == 608, "SceneUbo must match the std140 shader block");

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
        // The UBO/shader light array is fixed at kMaxUboLights (must match the GLSL
        // NUKA_MAX_LIGHTS). Warn ONCE on truncation rather than silently dropping.
        if (world.lights.size() > static_cast<size_t>(kMaxUboLights)) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                std::fprintf(stderr,
                             "[nuka_render] scene has %zu lights; only the first %d "
                             "fit the UBO (kMaxUboLights) -- the rest are dropped\n",
                             world.lights.size(), kMaxUboLights);
            }
        }
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

struct Image {
    VkImage        image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView    view = VK_NULL_HANDLE;
};

// A cheap O(1) content signature for a mesh: stream sizes + storage addresses + a
// few sampled values. A scene swap or an in-place ReplaceGeometry (which re-allocs
// the vectors) yields a new signature -> the device-local cache rebuilds; identical
// geometry hashes equal and is reused. Never 0 (0 marks an unused cache slot).
uint64_t MeshSignature(const MeshGeometry& geo) {
    uint64_t h = 1469598103934665603ull;  // FNV-1a offset basis
    auto mix = [&h](uint64_t v) { h ^= v; h *= 1099511628211ull; };
    mix(geo.positions.size());
    mix(geo.normals.size());
    mix(geo.colors.size());
    mix(geo.indices.size());
    mix(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(geo.positions.data())));
    mix(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(geo.colors.data())));
    mix(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(geo.indices.data())));
    auto sample = [&](const std::vector<float>& v, size_t i) {
        if (i < v.size()) { uint32_t b = 0; std::memcpy(&b, &v[i], 4); mix(b); }
    };
    if (!geo.positions.empty()) {
        sample(geo.positions, 0);
        sample(geo.positions, geo.positions.size() / 2);
        sample(geo.positions, geo.positions.size() - 1);
    }
    if (!geo.colors.empty()) {
        sample(geo.colors, 0);
        sample(geo.colors, geo.colors.size() / 2);
        sample(geo.colors, geo.colors.size() - 1);
    }
    if (!geo.indices.empty()) { mix(geo.indices.front()); mix(geo.indices.back()); }
    return h | 1ull;
}

// Write a host-mapped 8-bit swapchain readback as a binary PPM (P6) -- the reliable
// on-screen capture, bypassing DWM/WGC (which can return a stale flip-model thumbnail).
void WritePpm(const std::string& path, const uint8_t* pixels, uint32_t w, uint32_t h,
              bool bgr_swap) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "[present] capture: cannot open %s\n", path.c_str());
        return;
    }
    f << "P6\n" << w << ' ' << h << "\n255\n";
    std::vector<uint8_t> row(static_cast<size_t>(w) * 3u);
    for (uint32_t y = 0; y < h; ++y) {
        const uint8_t* src = pixels + static_cast<size_t>(y) * w * 4u;
        for (uint32_t x = 0; x < w; ++x) {
            const uint8_t* p = src + static_cast<size_t>(x) * 4u;
            row[x * 3 + 0] = bgr_swap ? p[2] : p[0];
            row[x * 3 + 1] = p[1];
            row[x * 3 + 2] = bgr_swap ? p[0] : p[2];
        }
        f.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(row.size()));
    }
}

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

    Image          depth;

    // set 0, binding 1 (the shared mesh_pbr.frag's uShadowMap): a persistent 1x1
    // dummy depth texture. Shadows are OFF in the present path (shadow_params.x==0)
    // so it is never sampled; it exists only to COMPLETE the descriptor set the
    // shared fragment shader statically declares. Extent-independent -> created once.
    Image          shadow_dummy;
    VkSampler      shadow_sampler = VK_NULL_HANDLE;

    // The id pass renders kId* texels at swapchain size with its own depth, for the selection
    // outline and Pick; outline.frag samples it through outline_set.
    Image            id_image, id_depth;
    VkRenderPass     id_pass = VK_NULL_HANDLE;
    VkFramebuffer    id_framebuffer = VK_NULL_HANDLE;
    VkSampler        id_sampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout outline_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool outline_pool = VK_NULL_HANDLE;
    VkDescriptorSet  outline_set = VK_NULL_HANDLE;
    VkPipelineLayout outline_layout = VK_NULL_HANDLE;

    VkRenderPass     present_pass = VK_NULL_HANDLE;
    VkRenderPass     overlay_pass = VK_NULL_HANDLE;  // present_pass loading the drawn scene back
    VkDescriptorSetLayout scene_set_layout = VK_NULL_HANDLE;  // set 0: per-frame SceneUbo
    VkDescriptorPool scene_pool = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline       pipeline = VK_NULL_HANDLE;
    // See-through debug overlay pipeline: shares pipeline_layout + the shared
    // shaders, but draws RenderWorld::debug_instances with depth-test off (overlays
    // through the scene). Wireframe when the device offers fillModeNonSolid, else a
    // translucent fill. Built/destroyed with `pipeline` (render-pass dependent).
    VkPipeline       debug_pipeline = VK_NULL_HANDLE;
    // RenderWorld::ghost_instances: a translucent fill tested against the scene's depth
    // without writing it, built and destroyed with debug_pipeline.
    VkPipeline       ghost_pipeline = VK_NULL_HANDLE;
    // The Wireframe view's edges, the id pass and the outline, rebuilt with `pipeline`.
    VkPipeline       wire_pipeline = VK_NULL_HANDLE;
    VkPipeline       id_pipeline = VK_NULL_HANDLE;
    VkPipeline       outline_pipeline = VK_NULL_HANDLE;
    bool             wireframe_capable_ = false;
    VkShaderModule   vert_module = VK_NULL_HANDLE;
    VkShaderModule   frag_module = VK_NULL_HANDLE;
    VkShaderModule   id_frag_module = VK_NULL_HANDLE;
    VkShaderModule   fullscreen_vert_module = VK_NULL_HANDLE;
    VkShaderModule   outline_frag_module = VK_NULL_HANDLE;
    VkCommandPool    command_pool = VK_NULL_HANDLE;

    // Instanced pipeline reading model matrices from a pose slot at set 1; built by
    // SetInteropTransforms and rebuilt on swapchain recreate.
    VkShaderModule        instanced_vert_module = VK_NULL_HANDLE;  // freed in ~Impl
    VkPipelineLayout      instanced_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline            instanced_pipeline = VK_NULL_HANDLE;
    VkPipeline            wire_instanced_pipeline = VK_NULL_HANDLE;
    VkPipeline            id_instanced_pipeline = VK_NULL_HANDLE;
    VkPipeline            ghost_instanced_pipeline = VK_NULL_HANDLE;  // see-through instances
    VkDescriptorSetLayout ssbo_set_layout = VK_NULL_HANDLE;  // borrowed (not owned)

    // Per-frame-in-flight SceneUbo buffer + descriptor set (one per frame so the
    // in-flight frame's uniform is not overwritten by the next); the last one serves Pick.
    struct UboSlot { VkBuffer buffer = VK_NULL_HANDLE; VkDeviceMemory memory = VK_NULL_HANDLE;
                     VkDescriptorSet set = VK_NULL_HANDLE; };
    // One slot per frame in flight and view, then Pick's.
    std::array<UboSlot, kFramesInFlight * kMaxSceneViews + 1u> ubo_slots{};

    // Per-frame-in-flight sync + command buffers. render_finished is PER SWAPCHAIN
    // IMAGE (indexed by image_index), not per frame-in-flight: acquire returns an
    // image only after its prior present consumed that image's semaphore, so signaling
    // it again is safe without a per-frame present stall.
    std::array<VkSemaphore, kFramesInFlight>     image_available{};
    std::vector<VkSemaphore>                     render_finished;
    std::array<VkFence, kFramesInFlight>         in_flight{};
    std::array<VkCommandBuffer, kFramesInFlight> command_buffers{};
    uint32_t current_frame = 0u;

    // Device-local geometry cached per mesh_id, uploaded once and reused every frame.
    // `signature` is the content signature it was built from; a mismatch (scene swap
    // or in-place replace) rebuilds the slot. The buffers are NOT swapchain-dependent,
    // so a resize never re-uploads geometry.
    struct MeshCacheEntry {
        uint64_t  signature = 0;  // 0 = unused slot
        GpuBuffer pos, nrm, col, idx;
        uint32_t  index_count = 0;
    };
    std::vector<MeshCacheEntry> mesh_cache;

    // Per-swapchain-image: a fence pointer tracking which in-flight frame last
    // used it (avoid presenting into an image still being read).
    std::vector<VkFence> images_in_flight;

    PresentReport report;
    bool should_close = false;
    bool recreate_requested = false;

    // Optional swapchain readback: when draw_counter_ reaches capture_frame_, the
    // presented image is copied to a host buffer and written to capture_path_ (PPM).
    int         capture_frame_ = -1;
    std::string capture_path_;
    // The same for the scene viewport alone, read before the overlay draws over it.
    int         scene_capture_frame_ = -1;
    std::string scene_capture_path_;
    uint64_t    draw_counter_ = 0;

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
        renderer = std::make_unique<VulkanRasterRenderer>(RendererConfig{
            /*present_capable=*/true, /*surface=*/nullptr, window_surface->RequiredInstanceExtensions()});
        vk = renderer->VulkanHandles();
        device = reinterpret_cast<VkDevice>(vk.device);
        physical = reinterpret_cast<VkPhysicalDevice>(vk.physical_device);
        graphics_queue = reinterpret_cast<VkQueue>(vk.graphics_queue);
        present_queue = reinterpret_cast<VkQueue>(vk.present_queue);
        report.device_name = renderer->DeviceName();

        // The present device enables fillModeNonSolid when the physical device offers
        // it (see VulkanRasterRenderer::CreateDevice), so a supported bit here means
        // POLYGON_MODE_LINE is usable; otherwise the debug pass falls back to fill.
        VkPhysicalDeviceFeatures supported_features{};
        vkGetPhysicalDeviceFeatures(physical, &supported_features);
        wireframe_capable_ = supported_features.fillModeNonSolid == VK_TRUE;

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
        CreateShadowDummy();
        CreateOutlineLayout();
        NegotiateSurface();
        CreateSwapchain();
        CreatePresentRenderPass();
        CreateDepthResources();
        CreateIdResources();
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
        DestroyMeshCache();
        DestroySwapchainDependents();  // frees the per-image render_finished semaphores
        for (uint32_t i = 0; i < kFramesInFlight; ++i) {
            if (image_available[i] != VK_NULL_HANDLE) vkDestroySemaphore(device, image_available[i], nullptr);
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
        for (VkShaderModule m : {id_frag_module, fullscreen_vert_module, outline_frag_module})
            vkDestroyShaderModule(device, m, nullptr);
        vkDestroyPipelineLayout(device, outline_layout, nullptr);
        vkDestroyDescriptorPool(device, outline_pool, nullptr);
        vkDestroyDescriptorSetLayout(device, outline_set_layout, nullptr);
        vkDestroySampler(device, id_sampler, nullptr);
        if (shadow_sampler != VK_NULL_HANDLE) vkDestroySampler(device, shadow_sampler, nullptr);
        DestroyImage(shadow_dummy);
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
        return detail::ChooseSwapchainExtent(caps, window_surface->Width(), window_surface->Height());
    }

    void CreateSwapchain() {
        VkSurfaceCapabilitiesKHR caps{};
        CheckVk(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface, &caps),
                "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
        extent = ChooseExtent(caps);
        if (extent.width == 0u || extent.height == 0u) {
            throw std::runtime_error("PresentRenderer: cannot create a zero-sized swapchain");
        }

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
        info.imageUsage = detail::SwapchainImageUsage(caps.supportedUsageFlags);
        info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;  // single graphics+present family
        info.preTransform = caps.currentTransform;
        info.compositeAlpha = detail::ChooseCompositeAlpha(caps.supportedCompositeAlpha);
        info.presentMode = present_mode;
        info.clipped = VK_TRUE;
        info.oldSwapchain = VK_NULL_HANDLE;
        report.capture_supported = detail::CanCaptureSwapchain(info.imageUsage, surface_format);
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

        // One render-finished semaphore per swapchain image (see the member note).
        render_finished.resize(count);
        VkSemaphoreCreateInfo sem_info{};
        sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        for (uint32_t i = 0; i < count; ++i) {
            CheckVk(vkCreateSemaphore(device, &sem_info, nullptr, &render_finished[i]),
                    "vkCreateSemaphore(renderFinished-per-image)");
        }
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
        attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        attachments[0].initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        CheckVk(vkCreateRenderPass(device, &info, nullptr, &overlay_pass),
                "vkCreateRenderPass(overlay)");
    }

    void CreateDepthResources() {
        depth = MakeImage(kDepthFormat, extent, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                          VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    // A device-local single-mip 2D image with a view over `aspect`.
    Image MakeImage(VkFormat format, VkExtent2D size, VkImageUsageFlags usage, VkImageAspectFlags aspect) {
        Image out;
        VkImageCreateInfo image_info{};
        image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image_info.imageType = VK_IMAGE_TYPE_2D;
        image_info.format = format;
        image_info.extent = {size.width, size.height, 1u};
        image_info.mipLevels = 1u;
        image_info.arrayLayers = 1u;
        image_info.samples = VK_SAMPLE_COUNT_1_BIT;
        image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        image_info.usage = usage;
        image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        CheckVk(vkCreateImage(device, &image_info, nullptr, &out.image), "vkCreateImage(present)");

        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(device, out.image, &req);
        VkMemoryAllocateInfo alloc{};
        alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc.allocationSize = req.size;
        alloc.memoryTypeIndex = FindMemoryType(physical, req.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        CheckVk(vkAllocateMemory(device, &alloc, nullptr, &out.memory), "vkAllocateMemory(present image)");
        CheckVk(vkBindImageMemory(device, out.image, out.memory, 0), "vkBindImageMemory(present)");

        VkImageViewCreateInfo view_info{};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image = out.image;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = format;
        view_info.subresourceRange.aspectMask = aspect;
        view_info.subresourceRange.levelCount = 1u;
        view_info.subresourceRange.layerCount = 1u;
        CheckVk(vkCreateImageView(device, &view_info, nullptr, &out.view), "vkCreateImageView(present)");
        return out;
    }

    void DestroyImage(Image& img) {
        vkDestroyImageView(device, img.view, nullptr);
        vkDestroyImage(device, img.image, nullptr);
        vkFreeMemory(device, img.memory, nullptr);
        img = {};
    }

    // The id pass's targets at swapchain size, its pass and framebuffer, and outline_set
    // pointed at the new id image.
    void CreateIdResources() {
        id_image = MakeImage(kIdFormat, extent,
                             VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                 VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                             VK_IMAGE_ASPECT_COLOR_BIT);
        id_depth = MakeImage(kDepthFormat, extent, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                             VK_IMAGE_ASPECT_DEPTH_BIT);

        std::array<VkAttachmentDescription, 2> attachments{};
        attachments[0].format = kIdFormat;
        attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        attachments[1] = attachments[0];
        attachments[1].format = kDepthFormat;
        attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        VkAttachmentReference color_ref{0u, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference depth_ref{1u, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1u;
        subpass.pColorAttachments = &color_ref;
        subpass.pDepthStencilAttachment = &depth_ref;
        // The previous frame's outline sampling, Pick's copy and depth writes finish before this
        // pass writes; its ids are visible to the outline and the copy after it.
        std::array<VkSubpassDependency, 2> deps{};
        deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        deps[0].dstSubpass = 0u;
        deps[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
                               VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                               VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        deps[0].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        deps[1].srcSubpass = 0u;
        deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        deps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
        deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        VkRenderPassCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        info.attachmentCount = static_cast<uint32_t>(attachments.size());
        info.pAttachments = attachments.data();
        info.subpassCount = 1u;
        info.pSubpasses = &subpass;
        info.dependencyCount = static_cast<uint32_t>(deps.size());
        info.pDependencies = deps.data();
        CheckVk(vkCreateRenderPass(device, &info, nullptr, &id_pass), "vkCreateRenderPass(id)");

        const std::array<VkImageView, 2> views{id_image.view, id_depth.view};
        VkFramebufferCreateInfo fb{};
        fb.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb.renderPass = id_pass;
        fb.attachmentCount = static_cast<uint32_t>(views.size());
        fb.pAttachments = views.data();
        fb.width = extent.width;
        fb.height = extent.height;
        fb.layers = 1u;
        CheckVk(vkCreateFramebuffer(device, &fb, nullptr, &id_framebuffer), "vkCreateFramebuffer(id)");

        VkDescriptorImageInfo image_info{id_sampler, id_image.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = outline_set;
        write.dstBinding = 0u;
        write.descriptorCount = 1u;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = &image_info;
        vkUpdateDescriptorSets(device, 1u, &write, 0u, nullptr);
    }

    // The outline pass's id sampler, its one descriptor set and its pipeline layout.
    void CreateOutlineLayout() {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = VK_FILTER_NEAREST;
        si.minFilter = VK_FILTER_NEAREST;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        CheckVk(vkCreateSampler(device, &si, nullptr, &id_sampler), "vkCreateSampler(id)");

        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0u;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        binding.descriptorCount = 1u;
        binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo set_info{};
        set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        set_info.bindingCount = 1u;
        set_info.pBindings = &binding;
        CheckVk(vkCreateDescriptorSetLayout(device, &set_info, nullptr, &outline_set_layout),
                "vkCreateDescriptorSetLayout(outline)");

        const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1u};
        VkDescriptorPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.maxSets = 1u;
        pool_info.poolSizeCount = 1u;
        pool_info.pPoolSizes = &size;
        CheckVk(vkCreateDescriptorPool(device, &pool_info, nullptr, &outline_pool),
                "vkCreateDescriptorPool(outline)");
        VkDescriptorSetAllocateInfo alloc{};
        alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool = outline_pool;
        alloc.descriptorSetCount = 1u;
        alloc.pSetLayouts = &outline_set_layout;
        CheckVk(vkAllocateDescriptorSets(device, &alloc, &outline_set), "vkAllocateDescriptorSets(outline)");

        VkPushConstantRange push_range{VK_SHADER_STAGE_FRAGMENT_BIT, 0u, sizeof(OutlinePush)};
        VkPipelineLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout_info.setLayoutCount = 1u;
        layout_info.pSetLayouts = &outline_set_layout;
        layout_info.pushConstantRangeCount = 1u;
        layout_info.pPushConstantRanges = &push_range;
        CheckVk(vkCreatePipelineLayout(device, &layout_info, nullptr, &outline_layout),
                "vkCreatePipelineLayout(outline)");
    }

    void CreateFramebuffers() {
        framebuffers.resize(image_views.size());
        for (size_t i = 0; i < image_views.size(); ++i) {
            std::array<VkImageView, 2> views{image_views[i], depth.view};
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
        const std::string dir = NUKA_PRESENT_SHADER_DIR;
        id_frag_module = CreateShaderModule(dir + "/mesh_id.frag.spv");
        fullscreen_vert_module = CreateShaderModule(dir + "/fullscreen.vert.spv");
        outline_frag_module = CreateShaderModule(dir + "/outline.frag.spv");
    }

    // 1x1 dummy depth texture + sampler bound at set 0 binding 1 so the shared
    // mesh_pbr.frag's uShadowMap is always a valid descriptor (shadows OFF here,
    // so it is never read). Created once; the layout transition uses a one-shot
    // command buffer. Mirrors the offscreen renderer's EnsureShadowSampler.
    void CreateShadowDummy() {
        shadow_dummy = MakeImage(kDepthFormat, {1u, 1u},
                                 VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                 VK_IMAGE_ASPECT_DEPTH_BIT);

        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = VK_FILTER_NEAREST;
        si.minFilter = VK_FILTER_NEAREST;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        si.maxLod = 0.0f;
        CheckVk(vkCreateSampler(device, &si, nullptr, &shadow_sampler),
                "vkCreateSampler(present-shadow)");

        // One-shot UNDEFINED -> DEPTH_STENCIL_READ_ONLY transition so sampling is legal.
        VkCommandBuffer cmd = BeginOneShot();
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = shadow_dummy.image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        barrier.subresourceRange.levelCount = 1u;
        barrier.subresourceRange.layerCount = 1u;
        barrier.srcAccessMask = 0u;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
                             1, &barrier);
        EndOneShotAndWait(cmd);
    }

    // The fixed-function state the present pipelines differ in.
    struct PipelineSpec {
        VkShaderModule   vert = VK_NULL_HANDLE;
        VkShaderModule   frag = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkRenderPass     pass = VK_NULL_HANDLE;
        bool             mesh = true;  // the three float3 vertex streams; false draws a bare triangle
        VkPolygonMode    polygon = VK_POLYGON_MODE_FILL;
        VkCullModeFlags  cull = VK_CULL_MODE_NONE;
        bool             depth_test = true;
        bool             depth_write = true;
        bool             blend = false;  // straight-alpha over
    };

    void CreatePipeline() {
        // Same shaders + push-block layout as the offscreen pipeline, but built
        // against the PRESENT render pass (different attachment format -> a
        // separate, render-pass-compatible pipeline). The shader modules are
        // created once (CreateShaderModules at ctor) and REUSED across swapchain
        // recreates -> no per-resize module leak.

        // Set 0: binding 0 = per-frame SceneUbo (camera + light rig); binding 1 =
        // the shadow map sampler. BOTH match the offscreen renderer's layout (the
        // shared mesh_pbr.frag statically declares the sampler at binding 1, so the
        // present set must provide it even though shadows are OFF here).
        std::array<VkDescriptorSetLayoutBinding, 2> set_bindings{};
        set_bindings[0].binding = 0u;
        set_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        set_bindings[0].descriptorCount = 1u;
        set_bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        set_bindings[1].binding = 1u;
        set_bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        set_bindings[1].descriptorCount = 1u;
        set_bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo set_info{};
        set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        set_info.bindingCount = static_cast<uint32_t>(set_bindings.size());
        set_info.pBindings = set_bindings.data();
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

        const PipelineSpec scene{vert_module, frag_module, pipeline_layout, present_pass};
        pipeline = MakePipeline(scene, "vkCreateGraphicsPipelines(present)");
        PipelineSpec wire = scene;
        wire.polygon = WirePolygon();
        wire_pipeline = MakePipeline(wire, "vkCreateGraphicsPipelines(present-wire)");
        PipelineSpec id = scene;
        id.frag = id_frag_module;
        id.pass = id_pass;
        id_pipeline = MakePipeline(id, "vkCreateGraphicsPipelines(present-id)");

        // The debug overlay never tests depth, so collider proxies read THROUGH the visual
        // meshes, and draws wireframe, with a translucent fill where fillModeNonSolid is
        // absent. The ghost overlay is a blended fill hidden behind the scene.
        PipelineSpec debug = wire;
        debug.depth_test = debug.depth_write = false;
        debug.blend = !wireframe_capable_;
        debug_pipeline = MakePipeline(debug, "vkCreateGraphicsPipelines(present-debug)");
        PipelineSpec ghost = scene;
        ghost.cull = VK_CULL_MODE_BACK_BIT;
        ghost.depth_write = false;
        ghost.blend = true;
        ghost_pipeline = MakePipeline(ghost, "vkCreateGraphicsPipelines(present-ghost)");

        PipelineSpec outline{fullscreen_vert_module, outline_frag_module, outline_layout, present_pass};
        outline.mesh = false;
        outline.depth_test = outline.depth_write = false;
        outline.blend = true;
        outline_pipeline = MakePipeline(outline, "vkCreateGraphicsPipelines(present-outline)");
    }

    // POLYGON_MODE_LINE where the device enables fillModeNonSolid, else a fill.
    VkPolygonMode WirePolygon() const {
        return wireframe_capable_ ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
    }

    VkPipeline MakePipeline(const PipelineSpec& spec, const char* what) {
        std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = spec.vert;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = spec.frag;
        stages[1].pName = "main";

        // Position, normal and colour, one float3 stream each.
        std::array<VkVertexInputBindingDescription, 3> bindings{};
        std::array<VkVertexInputAttributeDescription, 3> attrs{};
        for (uint32_t i = 0; i < 3u; ++i) {
            bindings[i] = {i, sizeof(float) * 3u, VK_VERTEX_INPUT_RATE_VERTEX};
            attrs[i] = {i, i, VK_FORMAT_R32G32B32_SFLOAT, 0u};
        }
        VkPipelineVertexInputStateCreateInfo vertex_input{};
        vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertex_input.vertexBindingDescriptionCount = spec.mesh ? 3u : 0u;
        vertex_input.pVertexBindingDescriptions = bindings.data();
        vertex_input.vertexAttributeDescriptionCount = spec.mesh ? 3u : 0u;
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
        rasterization.polygonMode = spec.polygon;
        rasterization.cullMode = spec.cull;
        rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterization.lineWidth = 1.0f;  // 1.0 needs no wideLines feature

        VkPipelineMultisampleStateCreateInfo multisample{};
        multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depth_stencil{};
        depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depth_stencil.depthTestEnable = spec.depth_test ? VK_TRUE : VK_FALSE;
        depth_stencil.depthWriteEnable = spec.depth_write ? VK_TRUE : VK_FALSE;
        depth_stencil.depthCompareOp = spec.depth_test ? VK_COMPARE_OP_LESS_OR_EQUAL : VK_COMPARE_OP_ALWAYS;

        VkPipelineColorBlendAttachmentState blend_attachment{};
        blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        blend_attachment.blendEnable = spec.blend ? VK_TRUE : VK_FALSE;
        blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
        blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;
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
        info.layout = spec.layout;
        info.renderPass = spec.pass;
        info.subpass = 0u;
        VkPipeline out = VK_NULL_HANDLE;
        CheckVk(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1u, &info, nullptr, &out), what);
        return out;
    }

    void DestroyPipelines(std::initializer_list<VkPipeline*> pipelines) {
        for (VkPipeline* p : pipelines) {
            vkDestroyPipeline(device, *p, nullptr);
            *p = VK_NULL_HANDLE;
        }
    }

    // INT-F1: build the OPT-IN instanced pipelines (mesh_instanced.vert reads the
    // per-instance model matrix from the interop SSBO at set 1 by gl_InstanceIndex).
    // Requires scene_set_layout (set 0) + the borrowed ssbo_set_layout (set 1) to be
    // valid -> called by SetInteropTransforms after CreatePipeline, and by
    // RecreateSwapchain when interop is active. They mirror the scene, wire and id
    // pipelines; the ONLY difference is the model-matrix source (SSBO vs push constant).
    void CreateInstancedPipeline() {
        if (ssbo_set_layout == VK_NULL_HANDLE || scene_set_layout == VK_NULL_HANDLE) return;
        if (instanced_vert_module == VK_NULL_HANDLE) {
            instanced_vert_module = CreateShaderModule(NUKA_RASTER_MESH_INSTANCED_VERT_SPV);
        }
        DestroyInstancedPipelines();

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

        PipelineSpec spec{instanced_vert_module, frag_module, instanced_pipeline_layout, present_pass};
        instanced_pipeline = MakePipeline(spec, "vkCreateGraphicsPipelines(present-instanced)");
        spec.polygon = WirePolygon();
        wire_instanced_pipeline = MakePipeline(spec, "vkCreateGraphicsPipelines(present-instanced-wire)");
        spec.polygon = VK_POLYGON_MODE_FILL;
        PipelineSpec ghost = spec;
        ghost.cull = VK_CULL_MODE_BACK_BIT;
        ghost.depth_write = false;
        ghost.blend = true;
        ghost_instanced_pipeline = MakePipeline(ghost, "vkCreateGraphicsPipelines(present-instanced-ghost)");
        spec.frag = id_frag_module;
        spec.pass = id_pass;
        id_instanced_pipeline = MakePipeline(spec, "vkCreateGraphicsPipelines(present-instanced-id)");
    }

    void DestroyInstancedPipelines() {
        DestroyPipelines({&instanced_pipeline, &wire_instanced_pipeline, &id_instanced_pipeline, &ghost_instanced_pipeline});
        vkDestroyPipelineLayout(device, instanced_pipeline_layout, nullptr);
        instanced_pipeline_layout = VK_NULL_HANDLE;
    }

    // Persistent per-frame-in-flight SceneUbo buffers + descriptor sets.
    void CreateSceneDescriptors() {
        const uint32_t slot_count = static_cast<uint32_t>(ubo_slots.size());
        const std::array<VkDescriptorPoolSize, 2> sizes{{
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, slot_count},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, slot_count}}};
        VkDescriptorPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.maxSets = slot_count;
        pool_info.poolSizeCount = static_cast<uint32_t>(sizes.size());
        pool_info.pPoolSizes = sizes.data();
        CheckVk(vkCreateDescriptorPool(device, &pool_info, nullptr, &scene_pool),
                "vkCreateDescriptorPool(present-scene)");
        for (UboSlot& slot : ubo_slots) {
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
            VkDescriptorImageInfo shadow_info{};
            shadow_info.sampler = shadow_sampler;
            shadow_info.imageView = shadow_dummy.view;
            shadow_info.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            std::array<VkWriteDescriptorSet, 2> writes{};
            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = slot.set;
            writes[0].dstBinding = 0u;
            writes[0].descriptorCount = 1u;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[0].pBufferInfo = &buffer_info;
            writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet = slot.set;
            writes[1].dstBinding = 1u;
            writes[1].descriptorCount = 1u;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[1].pImageInfo = &shadow_info;
            vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(),
                                   0u, nullptr);
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
        for (VkSemaphore s : render_finished) {
            if (s != VK_NULL_HANDLE) vkDestroySemaphore(device, s, nullptr);
        }
        render_finished.clear();
        for (VkFramebuffer fb : framebuffers) {
            if (fb != VK_NULL_HANDLE) vkDestroyFramebuffer(device, fb, nullptr);
        }
        framebuffers.clear();
        DestroyImage(depth);
        vkDestroyFramebuffer(device, id_framebuffer, nullptr);
        id_framebuffer = VK_NULL_HANDLE;
        DestroyImage(id_image);
        DestroyImage(id_depth);
        // Every pipeline is built against present_pass or id_pass; destroy them before their
        // layouts so a swapchain recreate rebuilds them all.
        DestroyPipelines({&pipeline, &wire_pipeline, &id_pipeline, &outline_pipeline, &debug_pipeline,
                          &ghost_pipeline});
        if (pipeline_layout != VK_NULL_HANDLE) { vkDestroyPipelineLayout(device, pipeline_layout, nullptr); pipeline_layout = VK_NULL_HANDLE; }
        // INT-F1: the instanced pipelines + their layout go too; the instanced_vert_module
        // + the BORROWED ssbo_set/layout survive.
        DestroyInstancedPipelines();
        // scene_set_layout is (re)created by CreatePipeline; destroy it here so a
        // swapchain recreate does not leak it. The persistent scene_pool + ubo_slots
        // sets stay valid (a byte-identical re-created layout remains bind-compatible).
        if (scene_set_layout != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(device, scene_set_layout, nullptr); scene_set_layout = VK_NULL_HANDLE; }
        if (present_pass != VK_NULL_HANDLE) { vkDestroyRenderPass(device, present_pass, nullptr); present_pass = VK_NULL_HANDLE; }
        if (overlay_pass != VK_NULL_HANDLE) { vkDestroyRenderPass(device, overlay_pass, nullptr); overlay_pass = VK_NULL_HANDLE; }
        vkDestroyRenderPass(device, id_pass, nullptr);
        id_pass = VK_NULL_HANDLE;
        for (VkImageView v : image_views) {
            if (v != VK_NULL_HANDLE) vkDestroyImageView(device, v, nullptr);
        }
        image_views.clear();
        images.clear();
        if (swapchain != VK_NULL_HANDLE) { vkDestroySwapchainKHR(device, swapchain, nullptr); swapchain = VK_NULL_HANDLE; }
    }

    bool RecreateSwapchain() {
        recreate_requested = true;
        VkSurfaceCapabilitiesKHR caps{};
        CheckVk(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface, &caps),
                "vkGetPhysicalDeviceSurfaceCapabilitiesKHR(recreate)");
        const VkExtent2D next_extent = ChooseExtent(caps);
        if (next_extent.width == 0u || next_extent.height == 0u) return false;
        CheckVk(vkDeviceWaitIdle(device), "vkDeviceWaitIdle(recreate)");
        DestroySwapchainDependents();
        NegotiateSurface();
        CreateSwapchain();
        CreatePresentRenderPass();
        CreateDepthResources();
        CreateIdResources();
        CreateFramebuffers();
        CreatePipeline();
        // DestroySwapchainDependents tore the instanced pipeline down with the old pass.
        if (ssbo_set_layout != VK_NULL_HANDLE) CreateInstancedPipeline();
        report.swapchain_image_count = static_cast<uint32_t>(images.size());
        report.surface_format = static_cast<int32_t>(surface_format);
        report.present_mode = static_cast<int32_t>(present_mode);
        report.width = extent.width;
        report.height = extent.height;
        recreate_requested = false;
        return true;
    }

    // -- host-visible buffer (per-frame SceneUbo, readback, staging) -----------
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

    // A device-local buffer (TRANSFER_DST so a staging copy can fill it). Filled once
    // via a one-shot transfer; reused every frame -> no per-frame upload.
    GpuBuffer CreateDeviceLocalBuffer(VkDeviceSize size, VkBufferUsageFlags usage) {
        GpuBuffer res;
        VkBufferCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size = size;
        info.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        CheckVk(vkCreateBuffer(device, &info, nullptr, &res.buffer), "vkCreateBuffer(present-device)");
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(device, res.buffer, &req);
        VkMemoryAllocateInfo alloc{};
        alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc.allocationSize = req.size;
        alloc.memoryTypeIndex = FindMemoryType(physical, req.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        CheckVk(vkAllocateMemory(device, &alloc, nullptr, &res.memory),
                "vkAllocateMemory(present-device)");
        CheckVk(vkBindBufferMemory(device, res.buffer, res.memory, 0),
                "vkBindBufferMemory(present-device)");
        return res;
    }

    VkCommandBuffer BeginOneShot() {
        VkCommandBufferAllocateInfo ca{};
        ca.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ca.commandPool = command_pool;
        ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ca.commandBufferCount = 1u;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        CheckVk(vkAllocateCommandBuffers(device, &ca, &cmd), "vkAllocateCommandBuffers(present-oneshot)");
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        CheckVk(vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer(present-oneshot)");
        return cmd;
    }

    // Submits a one-shot buffer and waits for the queue; with `pose` its vertex shading
    // first waits for the pose slot's timeline value.
    void EndOneShotAndWait(VkCommandBuffer cmd, const PresentInterop* pose = nullptr) {
        CheckVk(vkEndCommandBuffer(cmd), "vkEndCommandBuffer(present-oneshot)");
        const VkSemaphore timeline = pose != nullptr ? reinterpret_cast<VkSemaphore>(pose->timeline) : VK_NULL_HANDLE;
        const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;
        VkTimelineSemaphoreSubmitInfo timeline_info{};
        timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timeline_info.waitSemaphoreValueCount = 1u;
        timeline_info.pWaitSemaphoreValues = pose != nullptr ? &pose->ready_value : nullptr;
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.pNext = pose != nullptr ? &timeline_info : nullptr;
        submit.waitSemaphoreCount = pose != nullptr ? 1u : 0u;
        submit.pWaitSemaphores = &timeline;
        submit.pWaitDstStageMask = &wait_stage;
        submit.commandBufferCount = 1u;
        submit.pCommandBuffers = &cmd;
        CheckVk(vkQueueSubmit(graphics_queue, 1u, &submit, VK_NULL_HANDLE), "vkQueueSubmit(present-oneshot)");
        CheckVk(vkQueueWaitIdle(graphics_queue), "vkQueueWaitIdle(present-oneshot)");
        vkFreeCommandBuffers(device, command_pool, 1u, &cmd);
    }

    // Bring the device-local geometry cache in sync with `world`'s mesh table: keep it
    // sized to the table, (re)upload any referenced mesh whose content signature
    // changed (scene swap / in-place replace), and free dropped slots. Steady state =
    // a signature check per referenced mesh and NO GPU work. Any actual mutation first
    // waits the device idle so no in-flight frame still references a freed buffer.
    void SyncMeshCache(const RenderWorld& world) {
        const size_t mesh_count = world.meshes.Count();
        std::vector<uint32_t> rebuild;
        std::vector<uint64_t> rebuild_sig;
        std::vector<char> seen(mesh_count, 0);
        auto consider = [&](const RenderInstance& inst) {
            const uint32_t id = inst.mesh_id;
            if (id == kNoId || id >= mesh_count || seen[id]) return;
            const MeshGeometry& geo = world.meshes.Geometry(id);
            if (geo.Empty()) return;
            seen[id] = 1;
            const uint64_t sig = MeshSignature(geo);
            const bool current = (id < mesh_cache.size() && mesh_cache[id].signature == sig);
            if (!current) { rebuild.push_back(id); rebuild_sig.push_back(sig); }
        };
        for (const RenderInstance& inst : world.instances) consider(inst);
        // The debug overlay's proxy meshes live in the same library; upload them too.
        // Empty when overlays are off -> the non-overlay cache is unchanged.
        for (const RenderInstance& inst : world.debug_instances) consider(inst);
        for (const RenderInstance& inst : world.extra_instances) consider(inst);
        const bool shrink = mesh_cache.size() > mesh_count;
        if (rebuild.empty() && !shrink) return;

        vkDeviceWaitIdle(device);
        if (shrink) {
            for (size_t i = mesh_count; i < mesh_cache.size(); ++i) {
                Destroy(mesh_cache[i].pos); Destroy(mesh_cache[i].nrm); Destroy(mesh_cache[i].col);
                Destroy(mesh_cache[i].idx);
            }
        }
        mesh_cache.resize(mesh_count);
        if (rebuild.empty()) return;

        std::vector<GpuBuffer> staging;
        staging.reserve(rebuild.size() * 4u);
        VkCommandBuffer cmd = BeginOneShot();
        for (size_t k = 0; k < rebuild.size(); ++k) {
            const uint32_t id = rebuild[k];
            MeshCacheEntry& e = mesh_cache[id];
            Destroy(e.pos); Destroy(e.nrm); Destroy(e.col); Destroy(e.idx);
            const MeshGeometry& geo = world.meshes.Geometry(id);
            std::vector<float> synth;
            const std::vector<float>* nrm_src = &geo.normals;
            if (geo.normals.size() != geo.positions.size()) {
                synth = SynthesizeFlatNormals(geo.positions, geo.indices);
                nrm_src = &synth;
            }
            std::vector<float> white;
            const std::vector<float>* col_src = &geo.colors;
            if (geo.colors.size() != geo.positions.size()) {
                white.assign(geo.positions.size(), 1.0f);
                col_src = &white;
            }
            const VkDeviceSize pos_bytes = geo.positions.size() * sizeof(float);
            const VkDeviceSize nrm_bytes = nrm_src->size() * sizeof(float);
            const VkDeviceSize idx_bytes = geo.indices.size() * sizeof(uint32_t);
            e.pos = CreateDeviceLocalBuffer(pos_bytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            e.nrm = CreateDeviceLocalBuffer(nrm_bytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            e.col = CreateDeviceLocalBuffer(pos_bytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            e.idx = CreateDeviceLocalBuffer(idx_bytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
            e.index_count = static_cast<uint32_t>(geo.indices.size());
            auto stage_copy = [&](const void* data, VkDeviceSize n, const GpuBuffer& dst) {
                GpuBuffer s = CreateHostBuffer(n, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
                Upload(s, data, static_cast<size_t>(n));
                VkBufferCopy region{0u, 0u, n};
                vkCmdCopyBuffer(cmd, s.buffer, dst.buffer, 1u, &region);
                staging.push_back(s);
            };
            stage_copy(geo.positions.data(), pos_bytes, e.pos);
            stage_copy(nrm_src->data(), nrm_bytes, e.nrm);
            stage_copy(col_src->data(), pos_bytes, e.col);
            stage_copy(geo.indices.data(), idx_bytes, e.idx);
            e.signature = rebuild_sig[k];
        }
        EndOneShotAndWait(cmd);
        for (GpuBuffer& s : staging) Destroy(s);
    }

    void DestroyMeshCache() {
        for (MeshCacheEntry& e : mesh_cache) {
            Destroy(e.pos); Destroy(e.nrm); Destroy(e.col); Destroy(e.idx);
        }
        mesh_cache.clear();
    }

    struct InstanceDraw {
        VkBuffer  pos = VK_NULL_HANDLE, nrm = VK_NULL_HANDLE, col = VK_NULL_HANDLE, idx = VK_NULL_HANDLE;  // cache-owned
        uint32_t  index_count = 0;
        Mat4      model;
        float     base_color[4] = {0.8f, 0.8f, 0.8f, 1.0f};
        float     metallic = 0.0f;
        float     roughness = 0.5f;
        float     opacity = 1.0f;
        float     emissive[3] = {0.0f, 0.0f, 0.0f};
        float     shade = 0.0f;  // the view mode mesh_view.frag reads from emissive.a
        // INT-F1: the ORIGINAL index into world.instances (== the scatter SSBO row).
        // Used as firstInstance so gl_InstanceIndex selects the right SSBO mat4 on
        // the instanced path (the draw loop skips geometry-less instances, so the
        // compacted draw index would not match the SSBO row).
        uint32_t  instance_row = 0;
    };

    // Copies `area` of a presentable swapchain image into `out`, then moves the image to `after`.
    static void RecordReadback(VkCommandBuffer cmd, VkImage image, const VkRect2D& area, VkBuffer out,
                               VkImageLayout after) {
        VkImageMemoryBarrier to_src{};
        to_src.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        to_src.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        to_src.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        to_src.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_src.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_src.image = image;
        to_src.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u};
        to_src.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        to_src.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                             1, &to_src);
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1u;
        region.imageOffset = {area.offset.x, area.offset.y, 0};
        region.imageExtent = {area.extent.width, area.extent.height, 1u};
        vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, out, 1u, &region);
        VkImageMemoryBarrier back = to_src;
        back.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        back.newLayout = after;
        back.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        back.dstAccessMask = 0u;
        VkPipelineStageFlags next_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        if (after == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
            back.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            next_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        }
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, next_stage, 0, 0, nullptr, 0, nullptr,
                             1, &back);
    }

    // Writes a finished readback as a PPM and frees its buffer.
    void FinishReadback(GpuBuffer& buf, uint32_t width, uint32_t height, const std::string& path) {
        void* mapped = nullptr;
        if (vkMapMemory(device, buf.memory, 0, static_cast<VkDeviceSize>(width) * height * 4u, 0,
                        &mapped) == VK_SUCCESS) {
            const bool bgr_swap = (surface_format == VK_FORMAT_B8G8R8A8_UNORM ||
                                   surface_format == VK_FORMAT_B8G8R8A8_SRGB);
            WritePpm(path, static_cast<const uint8_t*>(mapped), width, height, bgr_swap);
            vkUnmapMemory(device, buf.memory);
            std::fprintf(stderr, "[present] captured frame %llu -> %s (%ux%u)\n",
                         static_cast<unsigned long long>(draw_counter_), path.c_str(), width, height);
        }
        Destroy(buf);
    }

    // Rewrites a draw's material for the view mode; mesh_view.frag reads the mode from emissive.a.
    static void ApplyViewMode(ViewMode mode, uint32_t material_id, InstanceDraw& draw) {
        switch (mode) {
        case ViewMode::Lit:
            break;
        case ViewMode::Unlit:
        case ViewMode::Wireframe:
            draw.shade = 1.0f;
            break;
        case ViewMode::Normals:
            draw.shade = 2.0f;
            break;
        case ViewMode::Material:
            draw.shade = 3.0f;
            if (material_id == kNoId) std::fill(draw.base_color, draw.base_color + 3, 0.5f);
            else MaterialColor(material_id, draw.base_color);
            break;
        case ViewMode::Residual:
        case ViewMode::Contact:
            std::fill(draw.base_color, draw.base_color + 3, kClay);
            draw.base_color[3] = 1.0f;
            draw.metallic = 0.0f;
            draw.roughness = 0.8f;
            draw.opacity = 1.0f;
            std::fill(draw.emissive, draw.emissive + 3, 0.0f);
            break;
        }
    }

    // One view's draws, framebuffer rect and SceneUbo.
    struct PreparedView {
        std::vector<InstanceDraw> draws;
        std::vector<InstanceDraw> env_draws;  // env_instances, rows after the instances
        SceneViewport rect;
        SceneUbo ubo{};
    };

    PreparedView Prepare(const RenderWorld& world, const RasterOptions& options) {
        // Bring the device-local geometry cache up to date (uploads only on a
        // geometry-set change). Per-frame work below is just transforms + materials.
        SyncMeshCache(world);

        // -- build per-instance draws from the cached geometry + scene AABB.
        PreparedView view;
        view.draws.reserve(world.instances.size());
        math::Vec3 aabb_min{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                            std::numeric_limits<float>::max()};
        math::Vec3 aabb_max{-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
                            -std::numeric_limits<float>::max()};
        bool has_geometry = false;
        // The world AABB is consumed ONLY by the auto-frame camera (no override, no
        // scene camera); skip the per-vertex sweep when a camera is supplied.
        const bool needs_aabb = !options.use_camera_override && world.CameraCount() == 0u;
        const uint32_t instance_total = static_cast<uint32_t>(world.instances.size());
        const auto make_draw = [&](const RenderInstance& inst, uint32_t row, InstanceDraw& draw) {
            if (inst.mesh_id == kNoId || inst.mesh_id >= world.meshes.Count()) return false;
            if (world.meshes.Geometry(inst.mesh_id).Empty()) return false;
            const MeshCacheEntry& cached = mesh_cache[inst.mesh_id];
            if (cached.index_count == 0u) return false;  // not cached (defensive)
            draw.instance_row = row;
            draw.model = TransformToMatrix(inst.world_xform);
            draw.pos = cached.pos.buffer;
            draw.nrm = cached.nrm.buffer;
            draw.col = cached.col.buffer;
            draw.idx = cached.idx.buffer;
            draw.index_count = cached.index_count;
            const bool has_material = inst.render_material_id != kNoId &&
                                      inst.render_material_id < world.materials.size();
            if (has_material) {
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
            ApplyViewMode(options.view_mode, has_material ? inst.render_material_id : kNoId, draw);
            return true;
        };
        for (uint32_t inst_idx = 0; inst_idx < instance_total; ++inst_idx) {
            const RenderInstance& inst = world.instances[inst_idx];
            InstanceDraw draw;
            if (!make_draw(inst, inst_idx, draw)) continue;
            if (needs_aabb) {
                const MeshGeometry& geo = world.meshes.Geometry(inst.mesh_id);
                for (size_t v = 0; v + 2 < geo.positions.size(); v += 3) {
                    const math::Vec3 local{geo.positions[v], geo.positions[v + 1], geo.positions[v + 2]};
                    const math::Vec3 wp = inst.world_xform.TransformPoint(local);
                    aabb_min.x = std::min(aabb_min.x, wp.x); aabb_min.y = std::min(aabb_min.y, wp.y);
                    aabb_min.z = std::min(aabb_min.z, wp.z); aabb_max.x = std::max(aabb_max.x, wp.x);
                    aabb_max.y = std::max(aabb_max.y, wp.y); aabb_max.z = std::max(aabb_max.z, wp.z);
                    has_geometry = true;
                }
            }
            view.draws.push_back(std::move(draw));
        }
        for (size_t j = 0; j < world.env_instances.size(); ++j) {
            InstanceDraw draw;
            if (make_draw(world.env_instances[j], instance_total + static_cast<uint32_t>(j), draw))
                view.env_draws.push_back(std::move(draw));
        }

        const ResolvedCamera cam = ResolveCamera(world, options, aabb_min, aabb_max, has_geometry);
        view.rect = ResolveSceneViewport(options.scene_viewport, extent.width, extent.height);
        const float aspect = view.rect.Aspect();
        const float fov_y = cam.fov_degrees * 3.14159265358979323846f / 180.0f;
        const Mat4 view_matrix = LookAt(cam.eye, cam.target, cam.up);
        const Mat4 proj = Perspective(fov_y, aspect, cam.near_clip, cam.far_clip);
        view.ubo = BuildSceneUbo(world, Multiply(proj, view_matrix), cam.eye);
        return view;
    }

    void UploadUbo(const UboSlot& slot, const SceneUbo& ubo) {
        Upload(GpuBuffer{slot.buffer, slot.memory}, &ubo, sizeof(SceneUbo));
    }

    static VkRect2D ScissorOf(const SceneViewport& rect) {
        return {{static_cast<int32_t>(rect.x), static_cast<int32_t>(rect.y)}, {rect.width, rect.height}};
    }

    // The smallest rect holding both; an empty one adds nothing.
    static SceneViewport Bounds(const SceneViewport& a, const SceneViewport& b) {
        if (a.Empty()) return b;
        if (b.Empty()) return a;
        const uint32_t x = std::min(a.x, b.x), y = std::min(a.y, b.y);
        return {true, x, y, std::max(a.x + a.width, b.x + b.width) - x, std::max(a.y + a.height, b.y + b.height) - y};
    }

    static void SetViewport(VkCommandBuffer cmd, const SceneViewport& rect) {
        VkViewport viewport{};
        viewport.x = static_cast<float>(rect.x); viewport.y = static_cast<float>(rect.y);
        viewport.width = static_cast<float>(std::max(rect.width, 1u));
        viewport.height = static_cast<float>(std::max(rect.height, 1u));
        viewport.minDepth = 0.0f; viewport.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0u, 1u, &viewport);
        const VkRect2D scissor = ScissorOf(rect);
        vkCmdSetScissor(cmd, 0u, 1u, &scissor);
    }

    // What a RecordDraws call draws: the scene leaves see-through instances to their own blended
    // pass, and the id pass writes each instance's kId* texel, skipping unpickable ones.
    enum class DrawPass : uint8_t { Scene, SeeThrough, Id };

    void RecordDraws(VkCommandBuffer cmd, const PreparedView& view, const std::vector<InstanceDraw>& draws,
                     VkPipeline pipe, VkPipelineLayout layout, VkDescriptorSet ubo_set, const PresentInterop* pose,
                     const std::vector<uint8_t>& flags, DrawPass pass) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0u, 1u, &ubo_set, 0u, nullptr);
        if (pose != nullptr) {
            const VkDescriptorSet slot_set = reinterpret_cast<VkDescriptorSet>(pose->set);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 1u, 1u, &slot_set, 0u, nullptr);
        }
        SetViewport(cmd, view.rect);
        for (const InstanceDraw& draw : draws) {
            const uint8_t f = draw.instance_row < flags.size() ? flags[draw.instance_row] : uint8_t{0};
            const bool see_through = (f & kInstanceSeeThrough) != 0;
            if (pass == DrawPass::Id ? (f & kInstanceUnpickable) != 0 : see_through != (pass == DrawPass::SeeThrough))
                continue;
            PushBlock push{};
            // On the instanced path the model matrix comes from the SSBO (the shader
            // ignores push.model); we still fill it harmlessly. Material always rides
            // the push constant on BOTH paths (materials are not device physics state).
            std::memcpy(push.model, draw.model.m.data(), sizeof(push.model));
            std::memcpy(push.base_color, draw.base_color, sizeof(push.base_color));
            push.mr[0] = draw.metallic; push.mr[1] = draw.roughness;
            push.mr[2] = pass == DrawPass::SeeThrough ? draw.opacity * kSeeThroughOpacity : draw.opacity;
            push.mr[3] = 0.0f;
            push.emissive[0] = draw.emissive[0]; push.emissive[1] = draw.emissive[1];
            push.emissive[2] = draw.emissive[2]; push.emissive[3] = draw.shade;
            if (pass == DrawPass::Id) {
                const uint32_t id = (draw.instance_row + 1u) | ((f & kInstanceSelected) != 0 ? kIdSelected : 0u) |
                                    ((f & kInstanceHovered) != 0 ? kIdHovered : 0u);
                std::memcpy(push.ground_shadow, &id, sizeof(id));
            }
            vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0u,
                               sizeof(PushBlock), &push);
            std::array<VkBuffer, 3> vbuffers{draw.pos, draw.nrm, draw.col};
            std::array<VkDeviceSize, 3> offsets{0u, 0u, 0u};
            vkCmdBindVertexBuffers(cmd, 0u, 3u, vbuffers.data(), offsets.data());
            vkCmdBindIndexBuffer(cmd, draw.idx, 0u, VK_INDEX_TYPE_UINT32);
            // INT-F1: on the instanced path, firstInstance == the SSBO row so
            // gl_InstanceIndex picks the right scattered transform (the draw loop
            // skipped geometry-less instances, so the SSBO row != compacted index).
            vkCmdDrawIndexed(cmd, draw.index_count, 1u, 0u, 0, pose != nullptr ? draw.instance_row : 0u);
        }
    }

    // The id pass over every view, each with its own set; it leaves id_image in SHADER_READ_ONLY_OPTIMAL.
    void RecordIdPass(VkCommandBuffer cmd, const std::vector<PreparedView>& views,
                      const std::vector<VkDescriptorSet>& sets, const std::vector<uint8_t>& flags,
                      const PresentInterop* pose) {
        std::array<VkClearValue, 2> clears{};
        clears[1].depthStencil = {1.0f, 0u};
        VkRenderPassBeginInfo rp{};
        rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp.renderPass = id_pass;
        rp.framebuffer = id_framebuffer;
        rp.renderArea = {{0, 0}, extent};
        rp.clearValueCount = static_cast<uint32_t>(clears.size());
        rp.pClearValues = clears.data();
        vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
        for (size_t i = 0; i < views.size(); ++i) {
            if (views[i].rect.Empty()) continue;
            if (pose != nullptr) {
                RecordDraws(cmd, views[i], views[i].draws, id_instanced_pipeline, instanced_pipeline_layout, sets[i],
                            pose, flags, DrawPass::Id);
            } else {
                RecordDraws(cmd, views[i], views[i].draws, id_pipeline, pipeline_layout, sets[i], nullptr, flags,
                            DrawPass::Id);
            }
            if (!views[i].env_draws.empty())
                RecordDraws(cmd, views[i], views[i].env_draws, id_pipeline, pipeline_layout, sets[i], nullptr, flags,
                            DrawPass::Id);
        }
        vkCmdEndRenderPass(cmd);
    }

    void RecordOutline(VkCommandBuffer cmd, const SceneViewport& rect, const RasterOptions& options) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, outline_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, outline_layout, 0u, 1u, &outline_set,
                                0u, nullptr);
        SetViewport(cmd, rect);
        OutlinePush push{};
        std::memcpy(push.selected, options.outline_color, sizeof(push.selected));
        std::memcpy(push.hovered, options.hover_color, sizeof(push.hovered));
        push.rect[0] = static_cast<int32_t>(rect.x);
        push.rect[1] = static_cast<int32_t>(rect.y);
        push.rect[2] = static_cast<int32_t>(rect.width);
        push.rect[3] = static_cast<int32_t>(rect.height);
        vkCmdPushConstants(cmd, outline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0u, sizeof(push), &push);
        vkCmdDraw(cmd, 3u, 1u, 0u, 0u);
    }

    static bool Marked(const std::vector<uint8_t>& flags) {
        return std::any_of(flags.begin(), flags.end(),
                           [](uint8_t f) { return (f & (kInstanceSelected | kInstanceHovered)) != 0; });
    }

    PresentFrameResult DrawFrame(const RenderWorld& world, const std::vector<RasterOptions>& views,
                                 const OverlayRecordFn& overlay, const PresentInterop* interop) {
        if (window_surface->Width() == 0u || window_surface->Height() == 0u) {
            recreate_requested = true;
            return PresentFrameResult::Suspended;
        }
        if (recreate_requested) {
            return RecreateSwapchain() ? PresentFrameResult::Recreated : PresentFrameResult::Suspended;
        }
        if (capture_frame_ >= 0 && !report.capture_supported) {
            throw std::runtime_error("PresentRenderer: this surface does not support RGBA8 capture");
        }
        const uint32_t frame = current_frame;
        CheckVk(vkWaitForFences(device, 1u, &in_flight[frame], VK_TRUE,
                                std::numeric_limits<uint64_t>::max()),
                "vkWaitForFences");

        uint32_t image_index = 0;
        const VkResult acquire = vkAcquireNextImageKHR(
            device, swapchain, std::numeric_limits<uint64_t>::max(),
            image_available[frame], VK_NULL_HANDLE, &image_index);
        if (acquire == VK_ERROR_OUT_OF_DATE_KHR) {
            return RecreateSwapchain() ? PresentFrameResult::Recreated : PresentFrameResult::Suspended;
        }
        if (acquire != VK_SUCCESS && acquire != VK_SUBOPTIMAL_KHR) {
            return PresentFrameResult::Error;
        }

        // If a prior frame is still using this image, wait on its fence.
        if (images_in_flight[image_index] != VK_NULL_HANDLE) {
            CheckVk(vkWaitForFences(device, 1u, &images_in_flight[image_index], VK_TRUE,
                                   std::numeric_limits<uint64_t>::max()),
                    "vkWaitForFences(image)");
        }
        images_in_flight[image_index] = in_flight[frame];

        // The first view's background, flags and outline colours apply to the frame.
        const RasterOptions& options = views.front();
        const size_t view_count = std::min<size_t>(views.size(), kMaxSceneViews);
        std::vector<PreparedView> prepared;
        std::vector<VkDescriptorSet> sets;
        SceneViewport shot{};
        for (size_t i = 0; i < view_count; ++i) {
            prepared.push_back(Prepare(world, views[i]));
            const UboSlot& slot = ubo_slots[frame * kMaxSceneViews + i];
            UploadUbo(slot, prepared.back().ubo);
            sets.push_back(slot.set);
            shot = Bounds(shot, prepared.back().rect);
        }

        const bool do_capture =
            (capture_frame_ >= 0 && draw_counter_ == static_cast<uint64_t>(capture_frame_));
        GpuBuffer capture_buf{};
        if (do_capture) {
            capture_buf = CreateHostBuffer(
                static_cast<VkDeviceSize>(extent.width) * extent.height * 4u,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        }
        const bool do_scene_capture = scene_capture_frame_ >= 0 && !shot.Empty() &&
                                      draw_counter_ == static_cast<uint64_t>(scene_capture_frame_);
        GpuBuffer scene_capture_buf{};
        if (do_scene_capture) {
            scene_capture_buf = CreateHostBuffer(
                static_cast<VkDeviceSize>(shot.width) * shot.height * 4u,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        }

        // -- record ----------------------------------------------------------
        VkCommandBuffer cmd = command_buffers[frame];
        CheckVk(vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer");
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        CheckVk(vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer(present)");

        // A frame with a pose slot draws instanced from it; otherwise the per-draw
        // push-constant pipelines draw the host transforms.
        const PresentInterop* pose =
            interop != nullptr && instanced_pipeline != VK_NULL_HANDLE ? interop : nullptr;
        // The id pass feeds the selection outline and runs only while something is marked.
        const bool outline = !shot.Empty() && Marked(options.instance_flags);
        if (outline) RecordIdPass(cmd, prepared, sets, options.instance_flags, pose);
        const bool see_through = std::any_of(options.instance_flags.begin(), options.instance_flags.end(),
                                             [](uint8_t f) { return (f & kInstanceSeeThrough) != 0; });

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

        // The overlays draw AFTER the opaque scene: the ghost behind it, then the debug
        // overlay (collider wireframes + contact markers) THROUGH it. Model rides the push
        // constant (no interop SSBO) and the view's viewport stays bound; an empty list is skipped.
        const auto draw_overlay = [&](const std::vector<RenderInstance>& overlay_list, VkPipeline overlay_pipeline,
                                      VkDescriptorSet ubo_set) {
            if (overlay_list.empty() || overlay_pipeline == VK_NULL_HANDLE) return;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, overlay_pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                    0u, 1u, &ubo_set, 0u, nullptr);
            for (const RenderInstance& inst : overlay_list) {
                if (inst.mesh_id == kNoId || inst.mesh_id >= world.meshes.Count()) continue;
                if (inst.mesh_id >= mesh_cache.size()) continue;
                const MeshCacheEntry& cached = mesh_cache[inst.mesh_id];
                if (cached.index_count == 0u) continue;
                PushBlock push{};
                const Mat4 model = TransformToMatrix(inst.world_xform);
                std::memcpy(push.model, model.m.data(), sizeof(push.model));
                push.base_color[0] = 0.8f; push.base_color[1] = 0.8f;
                push.base_color[2] = 0.8f; push.base_color[3] = 1.0f;
                push.mr[0] = 0.0f; push.mr[1] = 1.0f; push.mr[2] = 1.0f; push.mr[3] = 0.0f;
                if (inst.render_material_id != kNoId &&
                    inst.render_material_id < world.materials.size()) {
                    const scene::RenderMaterial& mat = world.materials[inst.render_material_id];
                    std::memcpy(push.base_color, mat.base_color, sizeof(push.base_color));
                    push.mr[0] = mat.metallic; push.mr[1] = mat.roughness; push.mr[2] = mat.opacity;
                    push.emissive[0] = mat.emissive[0]; push.emissive[1] = mat.emissive[1];
                    push.emissive[2] = mat.emissive[2];
                }
                vkCmdPushConstants(cmd, pipeline_layout,
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0u,
                                   sizeof(PushBlock), &push);
                std::array<VkBuffer, 3> vbuffers{cached.pos.buffer, cached.nrm.buffer, cached.col.buffer};
                std::array<VkDeviceSize, 3> offsets{0u, 0u, 0u};
                vkCmdBindVertexBuffers(cmd, 0u, 3u, vbuffers.data(), offsets.data());
                vkCmdBindIndexBuffer(cmd, cached.idx.buffer, 0u, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(cmd, cached.index_count, 1u, 0u, 0, 0u);
            }
        };
        for (size_t i = 0; i < prepared.size(); ++i) {
            const PreparedView& view = prepared[i];
            if (view.rect.Empty()) continue;
            const bool wire = views[i].view_mode == ViewMode::Wireframe;
            if (pose != nullptr) {
                RecordDraws(cmd, view, view.draws, wire ? wire_instanced_pipeline : instanced_pipeline,
                            instanced_pipeline_layout, sets[i], pose, options.instance_flags, DrawPass::Scene);
            } else {
                RecordDraws(cmd, view, view.draws, wire ? wire_pipeline : pipeline, pipeline_layout, sets[i], nullptr,
                            options.instance_flags, DrawPass::Scene);
            }
            if (!view.env_draws.empty())
                RecordDraws(cmd, view, view.env_draws, wire ? wire_pipeline : pipeline, pipeline_layout, sets[i],
                            nullptr, options.instance_flags, DrawPass::Scene);
            draw_overlay(world.extra_instances, wire ? wire_pipeline : pipeline, sets[i]);
            if (see_through && pose != nullptr) {
                RecordDraws(cmd, view, view.draws, ghost_instanced_pipeline, instanced_pipeline_layout, sets[i], pose,
                            options.instance_flags, DrawPass::SeeThrough);
            } else if (see_through) {
                RecordDraws(cmd, view, view.draws, ghost_pipeline, pipeline_layout, sets[i], nullptr,
                            options.instance_flags, DrawPass::SeeThrough);
            }
            draw_overlay(world.ghost_instances, ghost_pipeline, sets[i]);
            draw_overlay(world.debug_instances, debug_pipeline, sets[i]);
        }

        // The scene alone is read between two passes; the second loads it back for the overlay.
        if (do_scene_capture) {
            vkCmdEndRenderPass(cmd);
            RecordReadback(cmd, images[image_index], ScissorOf(shot), scene_capture_buf.buffer,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            rp.renderPass = overlay_pass;
            vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
        }

        // The outline goes after the scene capture, so recorded frames stay clean.
        if (outline) {
            for (const PreparedView& view : prepared) {
                if (!view.rect.Empty()) RecordOutline(cmd, view.rect, options);
            }
        }

        // Overlay (ImGui) records INSIDE the pass, after the scene.
        if (overlay) {
            overlay(reinterpret_cast<void*>(cmd));
        }

        vkCmdEndRenderPass(cmd);

        // Reliable capture: copy the just-presented image (now PRESENT_SRC) into a
        // host buffer, then restore PRESENT_SRC so the present still succeeds.
        if (do_capture) {
            RecordReadback(cmd, images[image_index], VkRect2D{{0, 0}, extent}, capture_buf.buffer,
                           VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        }
        CheckVk(vkEndCommandBuffer(cmd), "vkEndCommandBuffer(present)");

        // -- submit ----------------------------------------------------------
        // An instanced frame also waits, before vertex shading, for the CUDA writer's
        // timeline value that completed its pose slot.
        const std::array<VkSemaphore, 2> wait_semaphores{
            image_available[frame],
            pose != nullptr ? reinterpret_cast<VkSemaphore>(pose->timeline) : VK_NULL_HANDLE};
        const std::array<VkPipelineStageFlags, 2> wait_stages{VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                                              VK_PIPELINE_STAGE_VERTEX_SHADER_BIT};
        const std::array<uint64_t, 2> wait_values{0u, pose != nullptr ? pose->ready_value : 0u};
        VkTimelineSemaphoreSubmitInfo timeline_info{};
        timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timeline_info.waitSemaphoreValueCount = 2u;
        timeline_info.pWaitSemaphoreValues = wait_values.data();
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.pNext = pose != nullptr ? &timeline_info : nullptr;
        submit.waitSemaphoreCount = pose != nullptr ? 2u : 1u;
        submit.pWaitSemaphores = wait_semaphores.data();
        submit.pWaitDstStageMask = wait_stages.data();
        submit.commandBufferCount = 1u;
        submit.pCommandBuffers = &cmd;
        submit.signalSemaphoreCount = 1u;
        // render_finished is indexed by swapchain IMAGE: the present that displayed an
        // image must release it before acquire can return it again, so re-signaling
        // render_finished[image_index] is safe with no per-frame present stall.
        submit.pSignalSemaphores = &render_finished[image_index];
        CheckVk(vkResetFences(device, 1u, &in_flight[frame]), "vkResetFences");
        CheckVk(vkQueueSubmit(graphics_queue, 1u, &submit, in_flight[frame]), "vkQueueSubmit(present)");

        // -- present ---------------------------------------------------------
        VkPresentInfoKHR present{};
        present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present.waitSemaphoreCount = 1u;
        present.pWaitSemaphores = &render_finished[image_index];
        present.swapchainCount = 1u;
        present.pSwapchains = &swapchain;
        present.pImageIndices = &image_index;
        const VkResult present_result = vkQueuePresentKHR(present_queue, &present);

        // Geometry is cached + persistent, so there is nothing transient to free and
        // the frame pipelines (the top-of-frame fence wait bounds frames-in-flight).
        // The readback path is the one exception: the copy lives in THIS command
        // buffer, so wait its fence before mapping the host-visible capture buffer.
        if (do_capture || do_scene_capture) {
            CheckVk(vkWaitForFences(device, 1u, &in_flight[frame], VK_TRUE,
                                    std::numeric_limits<uint64_t>::max()),
                    "vkWaitForFences(capture)");
        }
        if (do_capture) FinishReadback(capture_buf, extent.width, extent.height, capture_path_);
        if (do_scene_capture)
            FinishReadback(scene_capture_buf, shot.width, shot.height, scene_capture_path_);

        current_frame = (current_frame + 1u) % kFramesInFlight;
        ++draw_counter_;
        if (present_result == VK_SUCCESS || present_result == VK_SUBOPTIMAL_KHR) {
            ++report.frames_presented;
        }
        if (present_result == VK_ERROR_OUT_OF_DATE_KHR || present_result == VK_SUBOPTIMAL_KHR ||
            (present_result == VK_SUCCESS && acquire == VK_SUBOPTIMAL_KHR)) {
            return RecreateSwapchain() ? PresentFrameResult::Recreated : PresentFrameResult::Suspended;
        }
        if (present_result != VK_SUCCESS) return PresentFrameResult::Error;
        return PresentFrameResult::Presented;
    }

    PickResult Pick(const RenderWorld& world, const RasterOptions& options, const SceneViewport& region,
                    const PresentInterop* interop) {
        PickResult result;
        const PreparedView view = Prepare(world, options);
        const uint32_t x0 = std::max(region.x, view.rect.x), y0 = std::max(region.y, view.rect.y);
        const uint32_t x1 = std::min(region.x + region.width, view.rect.x + view.rect.width);
        const uint32_t y1 = std::min(region.y + region.height, view.rect.y + view.rect.height);
        if (x1 <= x0 || y1 <= y0) return result;
        const uint32_t w = x1 - x0, h = y1 - y0;
        const UboSlot& slot = ubo_slots.back();
        UploadUbo(slot, view.ubo);
        GpuBuffer out = CreateHostBuffer(static_cast<VkDeviceSize>(w) * h * 4u, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        const PresentInterop* pose =
            interop != nullptr && id_instanced_pipeline != VK_NULL_HANDLE ? interop : nullptr;

        VkCommandBuffer cmd = BeginOneShot();
        RecordIdPass(cmd, {view}, {slot.set}, options.instance_flags, pose);
        VkImageMemoryBarrier to_src{};
        to_src.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        to_src.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        to_src.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        to_src.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_src.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_src.image = id_image.image;
        to_src.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u};
        to_src.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        to_src.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &to_src);
        VkBufferImageCopy copy{};
        copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.imageSubresource.layerCount = 1u;
        copy.imageOffset = {static_cast<int32_t>(x0), static_cast<int32_t>(y0), 0};
        copy.imageExtent = {w, h, 1u};
        vkCmdCopyImageToBuffer(cmd, id_image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, out.buffer, 1u, &copy);
        EndOneShotAndWait(cmd, pose);

        void* mapped = nullptr;
        CheckVk(vkMapMemory(device, out.memory, 0, VK_WHOLE_SIZE, 0, &mapped), "vkMapMemory(pick)");
        const uint32_t* ids = static_cast<const uint32_t*>(mapped);
        const uint32_t center = ids[(h / 2u) * w + w / 2u] & kIdMask;
        if (center != 0u) result.center = center - 1u;
        std::vector<char> seen(world.instances.size() + world.env_instances.size(), 0);
        for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
            const uint32_t id = ids[i] & kIdMask;
            if (id == 0u || seen[id - 1u] != 0) continue;
            seen[id - 1u] = 1;
            result.instances.push_back(id - 1u);
        }
        vkUnmapMemory(device, out.memory);
        Destroy(out);
        std::sort(result.instances.begin(), result.instances.end());
        return result;
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
                                              const OverlayRecordFn& overlay,
                                              const PresentInterop* interop) {
    return impl_->DrawFrame(world, std::vector<RasterOptions>{options}, overlay, interop);
}

PresentFrameResult PresentRenderer::DrawViews(const RenderWorld& world, const std::vector<RasterOptions>& views,
                                              const OverlayRecordFn& overlay, const PresentInterop* interop) {
    return impl_->DrawFrame(world, views, overlay, interop);
}

PickResult PresentRenderer::Pick(const RenderWorld& world, const RasterOptions& options,
                                 const SceneViewport& region, const PresentInterop* interop) {
    return impl_->Pick(world, options, region, interop);
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
uint32_t PresentRenderer::FramesInFlight() const { return kFramesInFlight; }
uint64_t PresentRenderer::SubmittedFrames() const { return impl_->draw_counter_; }

const PresentReport& PresentRenderer::Report() const { return impl_->report; }
const std::string& PresentRenderer::DeviceName() const { return impl_->report.device_name; }

void PresentRenderer::PollEvents(std::vector<window::WindowEvent>& out) {
    const size_t begin = out.size();
    impl_->window_surface->PollEvents(out);
    for (size_t i = begin; i < out.size(); ++i) {
        const auto& ev = out[i];
        if (ev.type == window::WindowEvent::Type::Close) impl_->should_close = true;
        if (ev.type == window::WindowEvent::Type::Resize) impl_->recreate_requested = true;
    }
}

bool PresentRenderer::ShouldClose() const { return impl_->should_close; }

void PresentRenderer::WaitIdle() {
    if (impl_->device != VK_NULL_HANDLE) vkDeviceWaitIdle(impl_->device);
}

void PresentRenderer::SetCaptureFrame(int frame_index, std::string ppm_path) {
    impl_->capture_frame_ = frame_index;
    impl_->capture_path_ = std::move(ppm_path);
}

void PresentRenderer::SetSceneCaptureFrame(int frame_index, std::string ppm_path) {
    impl_->scene_capture_frame_ = frame_index;
    impl_->scene_capture_path_ = std::move(ppm_path);
}

void PresentRenderer::SetInteropTransforms(NkVkDescriptorSetLayout ssbo_set_layout) {
    Impl& d = *impl_;
    d.DestroyInstancedPipelines();
    d.ssbo_set_layout = reinterpret_cast<VkDescriptorSetLayout>(ssbo_set_layout);
    if (d.ssbo_set_layout != VK_NULL_HANDLE) d.CreateInstancedPipeline();
}

}  // namespace nuka::render
