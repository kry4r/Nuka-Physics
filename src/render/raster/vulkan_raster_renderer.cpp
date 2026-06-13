// ---------------------------------------------------------------------------
// nuka::render::VulkanRasterRenderer implementation (M8 T3a).
//
// A real offscreen forward GRAPHICS pipeline. Reuses the init PATTERN of the
// compute path (vulkan_renderer.cpp): CreateInstance / select physical device /
// CreateDevice / CreateCommandPool -- here EXTENDED with a graphics queue +
// depth attachment + render pass + framebuffer + a VkGraphicsPipeline
// (mesh.vert / mesh_pbr.frag) + per-frame vertex/index buffers.
//
// Zero-CUDA-token (recon §4.5): pure C++/Vulkan, no CUDA tokens of any kind.
// ---------------------------------------------------------------------------

#include "render/raster/vulkan_raster_renderer.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace nuka::render {

namespace {

#ifndef NUKA_RASTER_MESH_VERT_SPV
#define NUKA_RASTER_MESH_VERT_SPV "mesh.vert.spv"
#endif
#ifndef NUKA_RASTER_MESH_FRAG_SPV
#define NUKA_RASTER_MESH_FRAG_SPV "mesh_pbr.frag.spv"
#endif

constexpr VkFormat kColorFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;

void CheckVk(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) +
                                 " failed with VkResult " +
                                 std::to_string(static_cast<int>(result)));
    }
}

std::vector<char> ReadBinaryFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open Vulkan shader: " + path);
    }
    const std::streamsize size = file.tellg();
    if (size <= 0) {
        throw std::runtime_error("Vulkan shader is empty: " + path);
    }
    file.seekg(0, std::ios::beg);
    std::vector<char> bytes(static_cast<size_t>(size));
    if (!file.read(bytes.data(), size)) {
        throw std::runtime_error("Failed to read Vulkan shader: " + path);
    }
    return bytes;
}

uint32_t FindMemoryType(VkPhysicalDevice physical_device,
                        uint32_t type_filter,
                        VkMemoryPropertyFlags required_flags) {
    VkPhysicalDeviceMemoryProperties memory_properties{};
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
    for (uint32_t index = 0; index < memory_properties.memoryTypeCount; ++index) {
        const bool type_matches = (type_filter & (1u << index)) != 0u;
        const bool flags_match =
            (memory_properties.memoryTypes[index].propertyFlags & required_flags)
            == required_flags;
        if (type_matches && flags_match) {
            return index;
        }
    }
    throw std::runtime_error("No compatible Vulkan memory type found");
}

// -- a tiny column-major 4x4 matrix (GLSL std140 / VkPushConstant friendly) ---
// Stored column-major: m[col*4 + row]. Matches GLSL mat4 memory layout so the
// push-constant block in mesh.vert reads it directly.
struct Mat4 {
    std::array<float, 16> m{};
    static Mat4 Identity() {
        Mat4 r;
        r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
        return r;
    }
};

Mat4 Multiply(const Mat4& a, const Mat4& b) {
    // a * b, both column-major.
    Mat4 r;
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) {
                sum += a.m[k * 4 + row] * b.m[col * 4 + k];
            }
            r.m[col * 4 + row] = sum;
        }
    }
    return r;
}

// Object->world matrix from a rigid Transform (rotation quaternion + position).
Mat4 TransformToMatrix(const math::Transform& t) {
    const math::Quat& qn = t.rotation;  // assumed normalized (cook output)
    const float x = qn.x, y = qn.y, z = qn.z, w = qn.w;
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, xz = x * z, yz = y * z;
    const float wx = w * x, wy = w * y, wz = w * z;
    Mat4 r = Mat4::Identity();
    // column 0
    r.m[0] = 1.0f - 2.0f * (yy + zz);
    r.m[1] = 2.0f * (xy + wz);
    r.m[2] = 2.0f * (xz - wy);
    // column 1
    r.m[4] = 2.0f * (xy - wz);
    r.m[5] = 1.0f - 2.0f * (xx + zz);
    r.m[6] = 2.0f * (yz + wx);
    // column 2
    r.m[8] = 2.0f * (xz + wy);
    r.m[9] = 2.0f * (yz - wx);
    r.m[10] = 1.0f - 2.0f * (xx + yy);
    // column 3 (translation)
    r.m[12] = t.position.x;
    r.m[13] = t.position.y;
    r.m[14] = t.position.z;
    return r;
}

// Right-handed look-at producing a view matrix (world -> camera, +z forward in
// view space is -forward). Column-major.
Mat4 LookAt(const math::Vec3& eye, const math::Vec3& target, const math::Vec3& up) {
    math::Vec3 f = (target - eye).Normalized();    // forward
    math::Vec3 s = f.Cross(up).Normalized();       // right
    math::Vec3 u = s.Cross(f);                     // true up
    Mat4 r = Mat4::Identity();
    r.m[0] = s.x;  r.m[4] = s.y;  r.m[8]  = s.z;
    r.m[1] = u.x;  r.m[5] = u.y;  r.m[9]  = u.z;
    r.m[2] = -f.x; r.m[6] = -f.y; r.m[10] = -f.z;
    r.m[12] = -s.Dot(eye);
    r.m[13] = -u.Dot(eye);
    r.m[14] = f.Dot(eye);
    return r;
}

// Vulkan-clip perspective (right-handed, depth 0..1, y points down in clip so we
// flip y to keep world-up = screen-up). Column-major.
Mat4 Perspective(float fov_y_radians, float aspect, float z_near, float z_far) {
    const float t = std::tan(fov_y_radians * 0.5f);
    Mat4 r;
    r.m[0] = 1.0f / (aspect * t);
    r.m[5] = -1.0f / t;  // negate to flip y for Vulkan clip space
    r.m[10] = z_far / (z_near - z_far);
    r.m[11] = -1.0f;
    r.m[14] = (z_near * z_far) / (z_near - z_far);
    return r;
}

struct PushBlock {
    float mvp[16];
    float model[16];
    float base_color[4];
};

// -- flat-normal synthesis for a mesh whose normals stream is empty ----------
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
            normals[v] = n.x / len;
            normals[v + 1] = n.y / len;
            normals[v + 2] = n.z / len;
        } else {
            normals[v] = 0.0f;
            normals[v + 1] = 0.0f;
            normals[v + 2] = 1.0f;
        }
    }
    return normals;
}

size_t CountNonBackground(const std::vector<VulkanRgba8>& pixels,
                          const VulkanRgba8& background) {
    size_t count = 0;
    for (const auto& pixel : pixels) {
        if (pixel.r != background.r || pixel.g != background.g ||
            pixel.b != background.b || pixel.a != background.a) {
            ++count;
        }
    }
    return count;
}

// GPU buffer / image wrappers (file-local so per-frame draw helpers can name them).
struct GpuBuffer {
    VkBuffer       buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
};
struct GpuImage {
    VkImage        image  = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView    view   = VK_NULL_HANDLE;
};

}  // namespace

// ---------------------------------------------------------------------------
// Impl -- owns persistent Vulkan state (device + render pass + pipeline) and the
// per-frame draw path.
// ---------------------------------------------------------------------------
struct VulkanRasterRenderer::Impl {
    // M8.5 T2 present configuration. present_capable=false reproduces the M8
    // offscreen renderer bit-for-bit (zero extra ext, graphics-only queue).
    bool             present_capable = false;
    VkSurfaceKHR     probe_surface  = VK_NULL_HANDLE;  // borrowed (NOT owned)

    VkInstance       instance       = VK_NULL_HANDLE;
    VkPhysicalDevice physical       = VK_NULL_HANDLE;
    uint32_t         queue_family   = 0;          // graphics (== present when present_capable)
    uint32_t         present_family = 0;          // present queue family (== queue_family here)
    VkPhysicalDeviceProperties props{};
    VkDevice         device         = VK_NULL_HANDLE;
    VkQueue          queue          = VK_NULL_HANDLE;
    VkQueue          present_queue  = VK_NULL_HANDLE;  // == queue (shared family)
    VkCommandPool    command_pool   = VK_NULL_HANDLE;
    VkRenderPass     render_pass    = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline       pipeline       = VK_NULL_HANDLE;
    VkShaderModule   vert_module    = VK_NULL_HANDLE;
    VkShaderModule   frag_module    = VK_NULL_HANDLE;
    VkDescriptorPool imgui_pool     = VK_NULL_HANDLE;  // lazily created (M8.5 ImGui seam)
    std::string      device_name;

    explicit Impl(const RendererConfig& config)
        : present_capable(config.present_capable),
          probe_surface(reinterpret_cast<VkSurfaceKHR>(config.surface)) {
        CreateInstance();
        SelectGraphicsDevice();
        device_name = props.deviceName;
        CreateDevice();
        CreateCommandPool();
        CreateRenderPass();
        CreatePipeline();
    }

    ~Impl() {
        if (device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
        }
        if (imgui_pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, imgui_pool, nullptr);
        if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, pipeline, nullptr);
        if (pipeline_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        if (vert_module != VK_NULL_HANDLE) vkDestroyShaderModule(device, vert_module, nullptr);
        if (frag_module != VK_NULL_HANDLE) vkDestroyShaderModule(device, frag_module, nullptr);
        if (render_pass != VK_NULL_HANDLE) vkDestroyRenderPass(device, render_pass, nullptr);
        if (command_pool != VK_NULL_HANDLE) vkDestroyCommandPool(device, command_pool, nullptr);
        if (device != VK_NULL_HANDLE) vkDestroyDevice(device, nullptr);
        if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, nullptr);
    }

    void CreateInstance() {
        VkApplicationInfo app_info{};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "nuka-physics";
        app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
        app_info.pEngineName = "nuka";
        app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
        app_info.apiVersion = VK_API_VERSION_1_1;

        VkInstanceCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.pApplicationInfo = &app_info;

        // OFFSCREEN: zero instance extensions (the G2 oracle, unchanged).
        // PRESENT:   the surface + xcb-surface instance extensions ONLY when a
        // present path is requested. We do NOT enable VK_KHR_xlib_surface (the
        // xcb backend is the chosen D1 windowing path); the headless-surface ext
        // is enabled OPPORTUNISTICALLY only if the loader advertises it (it does
        // NOT on the 1.2.131 loader here -- recon 1.3), so the same binary can use
        // it on a modern CI box. Missing exts are simply not requested.
        std::vector<const char*> instance_exts;
        if (present_capable) {
            const bool has_surface     = InstanceExtensionAvailable("VK_KHR_surface");
            const bool has_xcb         = InstanceExtensionAvailable("VK_KHR_xcb_surface");
            const bool has_headless    = InstanceExtensionAvailable("VK_EXT_headless_surface");
            if (has_surface)  instance_exts.push_back("VK_KHR_surface");
            if (has_xcb)      instance_exts.push_back("VK_KHR_xcb_surface");
            if (has_headless) instance_exts.push_back("VK_EXT_headless_surface");
        }
        create_info.enabledExtensionCount = static_cast<uint32_t>(instance_exts.size());
        create_info.ppEnabledExtensionNames =
            instance_exts.empty() ? nullptr : instance_exts.data();
        CheckVk(vkCreateInstance(&create_info, nullptr, &instance), "vkCreateInstance");
    }

    static bool InstanceExtensionAvailable(const char* name) {
        uint32_t count = 0;
        if (vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr) != VK_SUCCESS ||
            count == 0u) {
            return false;
        }
        std::vector<VkExtensionProperties> available(count);
        if (vkEnumerateInstanceExtensionProperties(nullptr, &count, available.data()) != VK_SUCCESS) {
            return false;
        }
        for (const auto& ext : available) {
            if (std::strcmp(ext.extensionName, name) == 0) return true;
        }
        return false;
    }

    static bool DeviceExtensionAvailable(VkPhysicalDevice candidate, const char* name) {
        uint32_t count = 0;
        if (vkEnumerateDeviceExtensionProperties(candidate, nullptr, &count, nullptr) != VK_SUCCESS ||
            count == 0u) {
            return false;
        }
        std::vector<VkExtensionProperties> available(count);
        if (vkEnumerateDeviceExtensionProperties(candidate, nullptr, &count, available.data()) !=
            VK_SUCCESS) {
            return false;
        }
        for (const auto& ext : available) {
            if (std::strcmp(ext.extensionName, name) == 0) return true;
        }
        return false;
    }

    void SelectGraphicsDevice() {
        uint32_t count = 0;
        CheckVk(vkEnumeratePhysicalDevices(instance, &count, nullptr),
                "vkEnumeratePhysicalDevices(count)");
        if (count == 0u) {
            throw std::runtime_error("No Vulkan physical device is available");
        }
        std::vector<VkPhysicalDevice> devices(count);
        CheckVk(vkEnumeratePhysicalDevices(instance, &count, devices.data()),
                "vkEnumeratePhysicalDevices(list)");
        for (const auto device_candidate : devices) {
            // Require the depth + color formats as render-pass attachments.
            VkFormatProperties depth_props{};
            vkGetPhysicalDeviceFormatProperties(device_candidate, kDepthFormat, &depth_props);
            if ((depth_props.optimalTilingFeatures &
                 VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) == 0u) {
                continue;
            }
            // PRESENT: a present-capable device must also expose the swapchain
            // device extension (offscreen never queries this -> unchanged).
            if (present_capable &&
                !DeviceExtensionAvailable(device_candidate, "VK_KHR_swapchain")) {
                continue;
            }
            uint32_t family_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(device_candidate, &family_count, nullptr);
            std::vector<VkQueueFamilyProperties> families(family_count);
            vkGetPhysicalDeviceQueueFamilyProperties(device_candidate, &family_count, families.data());
            for (uint32_t family = 0; family < family_count; ++family) {
                if ((families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0u) {
                    continue;
                }
                // OFFSCREEN: first graphics queue wins (M8 behaviour, unchanged).
                // PRESENT with a probe surface: the family must ALSO support
                // present on that surface (the SelectGraphicsDevice present-bug
                // seam the recon flagged -- now closed via the surface support
                // query). PRESENT without a surface (e.g. an early probe): accept
                // the graphics family (the surface is verified later at swapchain
                // negotiation).
                if (present_capable && probe_surface != VK_NULL_HANDLE) {
                    VkBool32 present_support = VK_FALSE;
                    if (vkGetPhysicalDeviceSurfaceSupportKHR(
                            device_candidate, family, probe_surface, &present_support) !=
                            VK_SUCCESS ||
                        present_support != VK_TRUE) {
                        continue;
                    }
                    present_family = family;
                }
                physical = device_candidate;
                queue_family = family;
                present_family = family;  // single shared graphics+present family
                vkGetPhysicalDeviceProperties(device_candidate, &props);
                return;
            }
        }
        if (present_capable) {
            throw std::runtime_error(
                "No Vulkan physical device exposes a graphics+present queue (swapchain)");
        }
        throw std::runtime_error("No Vulkan physical device exposes a graphics queue");
    }

    void CreateDevice() {
        constexpr float kQueuePriority = 1.0f;
        VkDeviceQueueCreateInfo queue_info{};
        queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = queue_family;
        queue_info.queueCount = 1u;
        queue_info.pQueuePriorities = &kQueuePriority;

        VkPhysicalDeviceFeatures features{};
        VkDeviceCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        create_info.queueCreateInfoCount = 1u;
        create_info.pQueueCreateInfos = &queue_info;
        create_info.pEnabledFeatures = &features;

        // OFFSCREEN: zero device extensions (unchanged). PRESENT: the swapchain
        // device extension (verified available in SelectGraphicsDevice).
        std::vector<const char*> device_exts;
        if (present_capable) {
            device_exts.push_back("VK_KHR_swapchain");
        }
        create_info.enabledExtensionCount = static_cast<uint32_t>(device_exts.size());
        create_info.ppEnabledExtensionNames =
            device_exts.empty() ? nullptr : device_exts.data();

        CheckVk(vkCreateDevice(physical, &create_info, nullptr, &device), "vkCreateDevice");
        vkGetDeviceQueue(device, queue_family, 0u, &queue);
        // Single shared graphics+present family -> same queue handle.
        vkGetDeviceQueue(device, present_family, 0u, &present_queue);
    }

    // Lazily create the FREE_DESCRIPTOR_SET_BIT descriptor pool ImGui needs. The
    // offscreen path never calls this, so its allocation footprint is unchanged.
    VkDescriptorPool EnsureImGuiPool() {
        if (imgui_pool != VK_NULL_HANDLE) return imgui_pool;
        // A generous pool sized like the upstream imgui example (combined-image
        // samplers dominate; the rest are spare headroom).
        const std::array<VkDescriptorPoolSize, 11> sizes{{
            {VK_DESCRIPTOR_TYPE_SAMPLER, 64u},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 256u},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 64u},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 64u},
            {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 64u},
            {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 64u},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 64u},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64u},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 64u},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 64u},
            {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 64u},
        }};
        VkDescriptorPoolCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        info.maxSets = 256u;
        info.poolSizeCount = static_cast<uint32_t>(sizes.size());
        info.pPoolSizes = sizes.data();
        CheckVk(vkCreateDescriptorPool(device, &info, nullptr, &imgui_pool),
                "vkCreateDescriptorPool(imgui)");
        return imgui_pool;
    }

    void CreateCommandPool() {
        VkCommandPoolCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        create_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        create_info.queueFamilyIndex = queue_family;
        CheckVk(vkCreateCommandPool(device, &create_info, nullptr, &command_pool),
                "vkCreateCommandPool");
    }

    void CreateRenderPass() {
        std::array<VkAttachmentDescription, 2> attachments{};
        attachments[0].format = kColorFormat;
        attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[0].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

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

        VkRenderPassCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        create_info.attachmentCount = static_cast<uint32_t>(attachments.size());
        create_info.pAttachments = attachments.data();
        create_info.subpassCount = 1u;
        create_info.pSubpasses = &subpass;
        CheckVk(vkCreateRenderPass(device, &create_info, nullptr, &render_pass),
                "vkCreateRenderPass");
    }

    VkShaderModule CreateShaderModule(const std::string& path) {
        const auto bytes = ReadBinaryFile(path);
        if ((bytes.size() % sizeof(uint32_t)) != 0u) {
            throw std::runtime_error("Vulkan shader bytecode has invalid size: " + path);
        }
        VkShaderModuleCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        create_info.codeSize = bytes.size();
        create_info.pCode = reinterpret_cast<const uint32_t*>(bytes.data());
        VkShaderModule module = VK_NULL_HANDLE;
        CheckVk(vkCreateShaderModule(device, &create_info, nullptr, &module),
                "vkCreateShaderModule");
        return module;
    }

    void CreatePipeline() {
        vert_module = CreateShaderModule(NUKA_RASTER_MESH_VERT_SPV);
        frag_module = CreateShaderModule(NUKA_RASTER_MESH_FRAG_SPV);

        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        push_range.offset = 0u;
        push_range.size = sizeof(PushBlock);

        VkPipelineLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout_info.pushConstantRangeCount = 1u;
        layout_info.pPushConstantRanges = &push_range;
        CheckVk(vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout),
                "vkCreatePipelineLayout");

        std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vert_module;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = frag_module;
        stages[1].pName = "main";

        // Vertex layout: location 0 = position (vec3) from binding 0,
        //                location 1 = normal   (vec3) from binding 1.
        std::array<VkVertexInputBindingDescription, 2> bindings{};
        bindings[0].binding = 0u;
        bindings[0].stride = sizeof(float) * 3u;
        bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        bindings[1].binding = 1u;
        bindings[1].stride = sizeof(float) * 3u;
        bindings[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        std::array<VkVertexInputAttributeDescription, 2> attrs{};
        attrs[0].location = 0u;
        attrs[0].binding = 0u;
        attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[0].offset = 0u;
        attrs[1].location = 1u;
        attrs[1].binding = 1u;
        attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[1].offset = 0u;

        VkPipelineVertexInputStateCreateInfo vertex_input{};
        vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertex_input.vertexBindingDescriptionCount = static_cast<uint32_t>(bindings.size());
        vertex_input.pVertexBindingDescriptions = bindings.data();
        vertex_input.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size());
        vertex_input.pVertexAttributeDescriptions = attrs.data();

        VkPipelineInputAssemblyStateCreateInfo input_assembly{};
        input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        // Viewport/scissor are dynamic so one pipeline serves any render size.
        VkPipelineViewportStateCreateInfo viewport_state{};
        viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport_state.viewportCount = 1u;
        viewport_state.scissorCount = 1u;

        VkPipelineRasterizationStateCreateInfo rasterization{};
        rasterization.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterization.polygonMode = VK_POLYGON_MODE_FILL;
        rasterization.cullMode = VK_CULL_MODE_NONE;  // robust for primitive fallback
        rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterization.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo multisample{};
        multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;  // no MSAA -> deterministic

        VkPipelineDepthStencilStateCreateInfo depth_stencil{};
        depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depth_stencil.depthTestEnable = VK_TRUE;
        depth_stencil.depthWriteEnable = VK_TRUE;
        depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineColorBlendAttachmentState blend_attachment{};
        blend_attachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        blend_attachment.blendEnable = VK_FALSE;
        VkPipelineColorBlendStateCreateInfo color_blend{};
        color_blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        color_blend.attachmentCount = 1u;
        color_blend.pAttachments = &blend_attachment;

        std::array<VkDynamicState, 2> dynamic_states{
            VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic_state{};
        dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic_state.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
        dynamic_state.pDynamicStates = dynamic_states.data();

        VkGraphicsPipelineCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        create_info.stageCount = static_cast<uint32_t>(stages.size());
        create_info.pStages = stages.data();
        create_info.pVertexInputState = &vertex_input;
        create_info.pInputAssemblyState = &input_assembly;
        create_info.pViewportState = &viewport_state;
        create_info.pRasterizationState = &rasterization;
        create_info.pMultisampleState = &multisample;
        create_info.pDepthStencilState = &depth_stencil;
        create_info.pColorBlendState = &color_blend;
        create_info.pDynamicState = &dynamic_state;
        create_info.layout = pipeline_layout;
        create_info.renderPass = render_pass;
        create_info.subpass = 0u;
        CheckVk(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1u, &create_info,
                                          nullptr, &pipeline),
                "vkCreateGraphicsPipelines");
    }

    // -- resource helpers ----------------------------------------------------
    GpuBuffer CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                           VkMemoryPropertyFlags memory_flags) {
        GpuBuffer resource;
        VkBufferCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        create_info.size = size;
        create_info.usage = usage;
        create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        CheckVk(vkCreateBuffer(device, &create_info, nullptr, &resource.buffer), "vkCreateBuffer");
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device, resource.buffer, &requirements);
        VkMemoryAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = requirements.size;
        alloc_info.memoryTypeIndex =
            FindMemoryType(physical, requirements.memoryTypeBits, memory_flags);
        CheckVk(vkAllocateMemory(device, &alloc_info, nullptr, &resource.memory),
                "vkAllocateMemory(buffer)");
        CheckVk(vkBindBufferMemory(device, resource.buffer, resource.memory, 0),
                "vkBindBufferMemory");
        return resource;
    }

    void UploadBuffer(const GpuBuffer& buffer, const void* data, size_t size) {
        if (size == 0u) return;
        void* mapped = nullptr;
        CheckVk(vkMapMemory(device, buffer.memory, 0, static_cast<VkDeviceSize>(size), 0, &mapped),
                "vkMapMemory(upload)");
        std::memcpy(mapped, data, size);
        vkUnmapMemory(device, buffer.memory);
    }

    void DestroyBuffer(GpuBuffer& resource) {
        if (resource.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, resource.buffer, nullptr);
            resource.buffer = VK_NULL_HANDLE;
        }
        if (resource.memory != VK_NULL_HANDLE) {
            vkFreeMemory(device, resource.memory, nullptr);
            resource.memory = VK_NULL_HANDLE;
        }
    }

    GpuImage CreateImage(uint32_t width, uint32_t height, VkFormat format,
                         VkImageUsageFlags usage, VkImageAspectFlags aspect) {
        GpuImage resource;
        VkImageCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        create_info.imageType = VK_IMAGE_TYPE_2D;
        create_info.format = format;
        create_info.extent = {width, height, 1u};
        create_info.mipLevels = 1u;
        create_info.arrayLayers = 1u;
        create_info.samples = VK_SAMPLE_COUNT_1_BIT;
        create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        create_info.usage = usage;
        create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        CheckVk(vkCreateImage(device, &create_info, nullptr, &resource.image), "vkCreateImage");
        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(device, resource.image, &requirements);
        VkMemoryAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = requirements.size;
        alloc_info.memoryTypeIndex =
            FindMemoryType(physical, requirements.memoryTypeBits,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        CheckVk(vkAllocateMemory(device, &alloc_info, nullptr, &resource.memory),
                "vkAllocateMemory(image)");
        CheckVk(vkBindImageMemory(device, resource.image, resource.memory, 0), "vkBindImageMemory");
        VkImageViewCreateInfo view_info{};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image = resource.image;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = format;
        view_info.subresourceRange.aspectMask = aspect;
        view_info.subresourceRange.levelCount = 1u;
        view_info.subresourceRange.layerCount = 1u;
        CheckVk(vkCreateImageView(device, &view_info, nullptr, &resource.view), "vkCreateImageView");
        return resource;
    }

    void DestroyImage(GpuImage& resource) {
        if (resource.view != VK_NULL_HANDLE) {
            vkDestroyImageView(device, resource.view, nullptr);
            resource.view = VK_NULL_HANDLE;
        }
        if (resource.image != VK_NULL_HANDLE) {
            vkDestroyImage(device, resource.image, nullptr);
            resource.image = VK_NULL_HANDLE;
        }
        if (resource.memory != VK_NULL_HANDLE) {
            vkFreeMemory(device, resource.memory, nullptr);
            resource.memory = VK_NULL_HANDLE;
        }
    }
};

// ---------------------------------------------------------------------------
// Per-draw geometry packed for one instance (object-space positions + normals).
// ---------------------------------------------------------------------------
namespace {

struct InstanceDraw {
    GpuBuffer vertex_pos;
    GpuBuffer vertex_nrm;
    GpuBuffer index;
    uint32_t index_count = 0;
    Mat4 model;
    float base_color[4] = {0.8f, 0.8f, 0.8f, 1.0f};
};

// Resolve the camera (eye/target/up/fov) for this render. Priority:
// 1) explicit override; 2) RenderWorld.cameras[0]; 3) auto-frame the scene AABB.
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
        cam.eye = options.camera_eye;
        cam.target = options.camera_target;
        cam.up = options.camera_up;
        cam.fov_degrees = options.camera_fov_degrees;
        cam.near_clip = options.camera_near;
        cam.far_clip = options.camera_far;
        return cam;
    }
    if (world.CameraCount() > 0u) {
        const RenderCamera& rc = world.cameras[0];
        // The camera looks down its local -Z (Vulkan/GL convention); up is +Y.
        cam.eye = rc.world_xform.position;
        cam.target = rc.world_xform.TransformPoint({0.0f, 0.0f, -1.0f});
        cam.up = rc.world_xform.TransformDirection({0.0f, 1.0f, 0.0f});
        cam.fov_degrees = rc.vertical_fov_degrees;
        cam.near_clip = rc.near_clip;
        cam.far_clip = rc.far_clip;
        return cam;
    }
    // Auto-frame: look at the AABB center from an offset along (+1,-1,+0.6),
    // distance scaled to the AABB diagonal so the whole scene fits.
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
// Public type forwarding (Impl is private; define member fns here).
// ---------------------------------------------------------------------------
VulkanRasterRenderer::VulkanRasterRenderer()
    : impl_(std::make_unique<Impl>(RendererConfig{})) {}
VulkanRasterRenderer::VulkanRasterRenderer(const RendererConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}
VulkanRasterRenderer::~VulkanRasterRenderer() = default;
VulkanRasterRenderer::VulkanRasterRenderer(VulkanRasterRenderer&&) noexcept = default;
VulkanRasterRenderer& VulkanRasterRenderer::operator=(VulkanRasterRenderer&&) noexcept = default;

const std::string& VulkanRasterRenderer::DeviceName() const { return impl_->device_name; }

RendererVulkanHandles VulkanRasterRenderer::VulkanHandles() {
    Impl& d = *impl_;
    RendererVulkanHandles handles;
    handles.instance        = reinterpret_cast<NkVkInstance>(d.instance);
    handles.physical_device = reinterpret_cast<NkVkPhysicalDevice>(d.physical);
    handles.device          = reinterpret_cast<NkVkDevice>(d.device);
    handles.graphics_family = d.queue_family;
    handles.present_family  = d.present_family;
    handles.graphics_queue  = reinterpret_cast<NkVkQueue>(d.queue);
    handles.present_queue   = reinterpret_cast<NkVkQueue>(d.present_queue);
    handles.offscreen_render_pass = reinterpret_cast<NkVkRenderPass>(d.render_pass);
    handles.imgui_descriptor_pool = reinterpret_cast<NkVkDescriptorPool>(d.EnsureImGuiPool());
    handles.api_version     = VK_API_VERSION_1_1;
    return handles;
}

VulkanOffscreenReport VulkanRasterRenderer::Render(const RenderWorld& world,
                                                   const RasterOptions& options) {
    if (options.width == 0u || options.height == 0u) {
        throw std::runtime_error("Vulkan raster render dimensions must be non-zero");
    }
    Impl& d = *impl_;

    // -- 1. Build per-instance geometry + accumulate the scene AABB ----------
    std::vector<InstanceDraw> draws;
    draws.reserve(world.instances.size());
    math::Vec3 aabb_min{std::numeric_limits<float>::max(),
                        std::numeric_limits<float>::max(),
                        std::numeric_limits<float>::max()};
    math::Vec3 aabb_max{-std::numeric_limits<float>::max(),
                        -std::numeric_limits<float>::max(),
                        -std::numeric_limits<float>::max()};
    bool has_geometry = false;

    for (const RenderInstance& inst : world.instances) {
        if (inst.mesh_id == kNoId || inst.mesh_id >= world.meshes.Count()) {
            continue;
        }
        const MeshGeometry& geo = world.meshes.Geometry(inst.mesh_id);
        if (geo.Empty()) {
            continue;
        }
        InstanceDraw draw;
        draw.model = TransformToMatrix(inst.world_xform);

        // Material base color (default if unresolved).
        if (inst.render_material_id != kNoId &&
            inst.render_material_id < world.materials.size()) {
            const scene::RenderMaterial& mat = world.materials[inst.render_material_id];
            draw.base_color[0] = mat.base_color[0];
            draw.base_color[1] = mat.base_color[1];
            draw.base_color[2] = mat.base_color[2];
            draw.base_color[3] = mat.base_color[3];
        }

        // Positions.
        const VkDeviceSize pos_bytes = geo.positions.size() * sizeof(float);
        draw.vertex_pos = d.CreateBuffer(
            pos_bytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        d.UploadBuffer(draw.vertex_pos, geo.positions.data(), pos_bytes);

        // Normals (synthesize flat if absent).
        std::vector<float> normals;
        const std::vector<float>* normal_src = &geo.normals;
        if (geo.normals.size() != geo.positions.size()) {
            normals = SynthesizeFlatNormals(geo.positions, geo.indices);
            normal_src = &normals;
        }
        const VkDeviceSize nrm_bytes = normal_src->size() * sizeof(float);
        draw.vertex_nrm = d.CreateBuffer(
            nrm_bytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        d.UploadBuffer(draw.vertex_nrm, normal_src->data(), nrm_bytes);

        // Indices.
        const VkDeviceSize idx_bytes = geo.indices.size() * sizeof(uint32_t);
        draw.index = d.CreateBuffer(
            idx_bytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        d.UploadBuffer(draw.index, geo.indices.data(), idx_bytes);
        draw.index_count = static_cast<uint32_t>(geo.indices.size());

        // World-space AABB from the transformed vertices.
        for (size_t v = 0; v + 2 < geo.positions.size(); v += 3) {
            const math::Vec3 local{geo.positions[v], geo.positions[v + 1], geo.positions[v + 2]};
            const math::Vec3 wp = inst.world_xform.TransformPoint(local);
            aabb_min.x = std::min(aabb_min.x, wp.x);
            aabb_min.y = std::min(aabb_min.y, wp.y);
            aabb_min.z = std::min(aabb_min.z, wp.z);
            aabb_max.x = std::max(aabb_max.x, wp.x);
            aabb_max.y = std::max(aabb_max.y, wp.y);
            aabb_max.z = std::max(aabb_max.z, wp.z);
            has_geometry = true;
        }
        draws.push_back(std::move(draw));
    }

    // -- 2. Resolve camera + build view-projection --------------------------
    const ResolvedCamera cam =
        ResolveCamera(world, options, aabb_min, aabb_max, has_geometry);
    const float aspect = static_cast<float>(options.width) / static_cast<float>(options.height);
    const float fov_y = cam.fov_degrees * 3.14159265358979323846f / 180.0f;
    const Mat4 view = LookAt(cam.eye, cam.target, cam.up);
    const Mat4 proj = Perspective(fov_y, aspect, cam.near_clip, cam.far_clip);
    const Mat4 view_proj = Multiply(proj, view);

    // -- 3. Offscreen color + depth targets + framebuffer -------------------
    GpuImage color = d.CreateImage(
        options.width, options.height, kColorFormat,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT);
    GpuImage depth = d.CreateImage(
        options.width, options.height, kDepthFormat,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);

    std::array<VkImageView, 2> fb_views{color.view, depth.view};
    VkFramebufferCreateInfo fb_info{};
    fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb_info.renderPass = d.render_pass;
    fb_info.attachmentCount = static_cast<uint32_t>(fb_views.size());
    fb_info.pAttachments = fb_views.data();
    fb_info.width = options.width;
    fb_info.height = options.height;
    fb_info.layers = 1u;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    CheckVk(vkCreateFramebuffer(d.device, &fb_info, nullptr, &framebuffer), "vkCreateFramebuffer");

    const VkDeviceSize readback_bytes =
        static_cast<VkDeviceSize>(options.width) * options.height * sizeof(VulkanRgba8);
    GpuBuffer readback = d.CreateBuffer(
        readback_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // -- 4. Record + submit one render pass ---------------------------------
    VkCommandBufferAllocateInfo cmd_alloc{};
    cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc.commandPool = d.command_pool;
    cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc.commandBufferCount = 1u;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    CheckVk(vkAllocateCommandBuffers(d.device, &cmd_alloc, &cmd), "vkAllocateCommandBuffers");

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CheckVk(vkBeginCommandBuffer(cmd, &begin_info), "vkBeginCommandBuffer");

    std::array<VkClearValue, 2> clears{};
    clears[0].color = {{static_cast<float>(options.background.r) / 255.0f,
                        static_cast<float>(options.background.g) / 255.0f,
                        static_cast<float>(options.background.b) / 255.0f,
                        static_cast<float>(options.background.a) / 255.0f}};
    clears[1].depthStencil = {1.0f, 0u};

    VkRenderPassBeginInfo rp_begin{};
    rp_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp_begin.renderPass = d.render_pass;
    rp_begin.framebuffer = framebuffer;
    rp_begin.renderArea = {{0, 0}, {options.width, options.height}};
    rp_begin.clearValueCount = static_cast<uint32_t>(clears.size());
    rp_begin.pClearValues = clears.data();
    vkCmdBeginRenderPass(cmd, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, d.pipeline);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(options.width);
    viewport.height = static_cast<float>(options.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0u, 1u, &viewport);
    VkRect2D scissor{{0, 0}, {options.width, options.height}};
    vkCmdSetScissor(cmd, 0u, 1u, &scissor);

    for (const InstanceDraw& draw : draws) {
        PushBlock push{};
        const Mat4 mvp = Multiply(view_proj, draw.model);
        std::memcpy(push.mvp, mvp.m.data(), sizeof(push.mvp));
        std::memcpy(push.model, draw.model.m.data(), sizeof(push.model));
        std::memcpy(push.base_color, draw.base_color, sizeof(push.base_color));
        vkCmdPushConstants(cmd, d.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0u,
                           sizeof(PushBlock), &push);

        std::array<VkBuffer, 2> vbuffers{draw.vertex_pos.buffer, draw.vertex_nrm.buffer};
        std::array<VkDeviceSize, 2> offsets{0u, 0u};
        vkCmdBindVertexBuffers(cmd, 0u, 2u, vbuffers.data(), offsets.data());
        vkCmdBindIndexBuffer(cmd, draw.index.buffer, 0u, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, draw.index_count, 1u, 0u, 0, 0u);
    }

    vkCmdEndRenderPass(cmd);

    VkBufferImageCopy copy_region{};
    copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy_region.imageSubresource.layerCount = 1u;
    copy_region.imageExtent = {options.width, options.height, 1u};
    vkCmdCopyImageToBuffer(cmd, color.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           readback.buffer, 1u, &copy_region);

    CheckVk(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1u;
    submit.pCommandBuffers = &cmd;
    CheckVk(vkQueueSubmit(d.queue, 1u, &submit, VK_NULL_HANDLE), "vkQueueSubmit");
    CheckVk(vkQueueWaitIdle(d.queue), "vkQueueWaitIdle");
    vkFreeCommandBuffers(d.device, d.command_pool, 1u, &cmd);

    // -- 5. Read back pixels ------------------------------------------------
    std::vector<VulkanRgba8> pixels(static_cast<size_t>(options.width) * options.height);
    void* mapped = nullptr;
    CheckVk(vkMapMemory(d.device, readback.memory, 0, readback_bytes, 0, &mapped),
            "vkMapMemory(readback)");
    std::memcpy(pixels.data(), mapped, static_cast<size_t>(readback_bytes));
    vkUnmapMemory(d.device, readback.memory);

    // -- 6. Teardown per-frame resources ------------------------------------
    d.DestroyBuffer(readback);
    vkDestroyFramebuffer(d.device, framebuffer, nullptr);
    d.DestroyImage(depth);
    d.DestroyImage(color);
    for (InstanceDraw& draw : draws) {
        d.DestroyBuffer(draw.vertex_pos);
        d.DestroyBuffer(draw.vertex_nrm);
        d.DestroyBuffer(draw.index);
    }

    VulkanOffscreenReport report;
    report.backend = RenderBackend::Vulkan;
    report.production_backend = true;
    report.width = options.width;
    report.height = options.height;
    report.command_count = static_cast<uint32_t>(draws.size());
    report.physical_device_count = 1u;
    report.selected_device_name = d.device_name;
    report.non_background_pixel_count = CountNonBackground(pixels, options.background);
    report.pixels = std::move(pixels);
    return report;
}

VulkanOffscreenReport RenderWorldVulkanRaster(const RenderWorld& world,
                                              const RasterOptions& options) {
    VulkanRasterRenderer renderer;
    return renderer.Render(world, options);
}

}  // namespace nuka::render
