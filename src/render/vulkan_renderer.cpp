// ---------------------------------------------------------------------------
// nuka::render::vulkan_renderer implementation
// ---------------------------------------------------------------------------

#include "render/vulkan_renderer.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nuka::render {

namespace {

#ifndef NUKA_VULKAN_DEBUG_DRAW_SPV
#define NUKA_VULKAN_DEBUG_DRAW_SPV "debug_draw.comp.spv"
#endif

constexpr VkFormat kOffscreenFormat = VK_FORMAT_R8G8B8A8_UNORM;

struct HostDebugCommand {
    uint32_t type = 0;
    uint32_t color = 0xFFFFFFFFu;
    uint32_t pad0 = 0;
    uint32_t pad1 = 0;
    float position[3] = {};
    float radius = 0.0f;
    float end[3] = {};
    float half_height = 0.0f;
    float size[3] = {};
    float pad2 = 0.0f;
};

struct PushConstants {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t command_count = 0;
    float view_scale = 1.0f;
    float view_center[4] = {};
    float background[4] = {};
};

struct QueueSelection {
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    uint32_t queue_family_index = 0;
    VkPhysicalDeviceProperties properties{};
};

struct BufferResource {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
};

struct ImageResource {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
};

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

bool SupportsStorageImage(VkPhysicalDevice device) {
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(device, kOffscreenFormat, &properties);
    return (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0u;
}

QueueSelection SelectComputeDevice(VkInstance instance) {
    uint32_t device_count = 0;
    CheckVk(vkEnumeratePhysicalDevices(instance, &device_count, nullptr),
            "vkEnumeratePhysicalDevices(count)");
    if (device_count == 0u) {
        throw std::runtime_error("No Vulkan physical device is available");
    }

    std::vector<VkPhysicalDevice> devices(device_count);
    CheckVk(vkEnumeratePhysicalDevices(instance, &device_count, devices.data()),
            "vkEnumeratePhysicalDevices(list)");

    for (const auto device : devices) {
        if (!SupportsStorageImage(device)) {
            continue;
        }

        uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &family_count, families.data());

        for (uint32_t family = 0; family < family_count; ++family) {
            if ((families[family].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0u) {
                QueueSelection selection;
                selection.physical_device = device;
                selection.queue_family_index = family;
                vkGetPhysicalDeviceProperties(device, &selection.properties);
                return selection;
            }
        }
    }

    throw std::runtime_error("No Vulkan physical device supports compute storage images");
}

VkInstance CreateInstance() {
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "nuka-physics";
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.pEngineName = "nuka";
    app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    VkInstance instance = VK_NULL_HANDLE;
    CheckVk(vkCreateInstance(&create_info, nullptr, &instance), "vkCreateInstance");
    return instance;
}

HostDebugCommand ToHostCommand(const VulkanDebugDrawCommand& command) {
    HostDebugCommand out;
    out.type = static_cast<uint32_t>(command.type);
    out.color = command.color;
    out.position[0] = command.position.x;
    out.position[1] = command.position.y;
    out.position[2] = command.position.z;
    out.radius = command.radius;
    out.end[0] = command.end.x;
    out.end[1] = command.end.y;
    out.end[2] = command.end.z;
    out.half_height = command.half_height;
    out.size[0] = command.size.x;
    out.size[1] = command.size.y;
    out.size[2] = command.size.z;
    return out;
}

float Clamp01(float value) {
    return std::clamp(value, 0.0f, 1.0f);
}

uint8_t FloatColorToByte(float value) {
    return static_cast<uint8_t>(std::round(Clamp01(value) * 255.0f));
}

uint32_t PackRgba8(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    return (static_cast<uint32_t>(r) << 24u) |
        (static_cast<uint32_t>(g) << 16u) |
        (static_cast<uint32_t>(b) << 8u) |
        static_cast<uint32_t>(a);
}

uint32_t MaterialColorToU32(const RenderMaterial& material) {
    return PackRgba8(FloatColorToByte(material.base_color.x),
                     FloatColorToByte(material.base_color.y),
                     FloatColorToByte(material.base_color.z),
                     FloatColorToByte(material.alpha));
}

const RenderMaterial* FindMaterial(const RenderScene& scene, scene::MaterialId material_id) {
    const auto it = std::find_if(
        scene.materials.begin(),
        scene.materials.end(),
        [material_id](const RenderMaterial& material) {
            return material.material_id == material_id;
        });
    return it != scene.materials.end() ? &(*it) : nullptr;
}

uint32_t MeshColor(const RenderScene& scene, const RenderMeshInstance& mesh) {
    if (const auto* material = FindMaterial(scene, mesh.material_id)) {
        return MaterialColorToU32(*material);
    }
    return 0xFFFFFFFFu;
}

math::Vec3 BoxLikeHalfExtents(const RenderMeshInstance& mesh) {
    if (mesh.shape_type == scene::ShapeType::Plane) {
        return {std::max(mesh.half_extents.x, 0.5f),
                std::max(mesh.half_extents.y, 0.5f),
                0.0f};
    }
    return mesh.half_extents;
}

VulkanDebugDrawCommand ToVulkanCommand(const RenderScene& scene,
                                       const RenderMeshInstance& mesh) {
    VulkanDebugDrawCommand command;
    command.position = mesh.world_transform.position;
    command.color = MeshColor(scene, mesh);

    switch (mesh.shape_type) {
    case scene::ShapeType::Sphere:
        command.type = VulkanDebugDrawCommandType::Sphere;
        command.radius = mesh.radius;
        break;
    case scene::ShapeType::Capsule:
        command.type = VulkanDebugDrawCommandType::Capsule;
        command.radius = mesh.radius;
        command.half_height = mesh.half_height;
        command.end = mesh.world_transform.TransformDirection(math::Vec3::UnitZ());
        break;
    case scene::ShapeType::Box:
    case scene::ShapeType::Plane:
    case scene::ShapeType::ConvexHull:
    case scene::ShapeType::TriMesh:
    case scene::ShapeType::HeightField:
        command.type = VulkanDebugDrawCommandType::Box;
        command.size = BoxLikeHalfExtents(mesh);
        break;
    }

    return command;
}

std::vector<VulkanDebugDrawCommand> BuildRenderSceneCommands(const RenderScene& scene) {
    std::vector<VulkanDebugDrawCommand> commands;
    commands.reserve(scene.mesh_instances.size());
    for (const auto& mesh : scene.mesh_instances) {
        commands.push_back(ToVulkanCommand(scene, mesh));
    }
    return commands;
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

class VulkanOffscreenRenderer {
public:
    VulkanOffscreenRenderer()
        : instance_(CreateInstance())
        , selection_(SelectComputeDevice(instance_)) {
        CreateDevice();
        CreateCommandPool();
    }

    ~VulkanOffscreenRenderer() {
        if (device_ != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device_);
        }
        if (command_pool_ != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device_, command_pool_, nullptr);
        }
        if (device_ != VK_NULL_HANDLE) {
            vkDestroyDevice(device_, nullptr);
        }
        if (instance_ != VK_NULL_HANDLE) {
            vkDestroyInstance(instance_, nullptr);
        }
    }

    VulkanOffscreenRenderer(const VulkanOffscreenRenderer&) = delete;
    VulkanOffscreenRenderer& operator=(const VulkanOffscreenRenderer&) = delete;

    VulkanOffscreenReport Render(const std::vector<VulkanDebugDrawCommand>& commands,
                                 const VulkanOffscreenOptions& options) {
        if (options.width == 0u || options.height == 0u) {
            throw std::runtime_error("Vulkan offscreen render dimensions must be non-zero");
        }

        const auto host_commands = BuildHostCommands(commands);
        BufferResource command_buffer = CreateBuffer(
            host_commands.size() * sizeof(HostDebugCommand),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        UploadBuffer(command_buffer, host_commands.data(),
                     host_commands.size() * sizeof(HostDebugCommand));

        ImageResource image = CreateStorageImage(options.width, options.height);
        BufferResource readback_buffer = CreateBuffer(
            static_cast<VkDeviceSize>(options.width) * options.height * sizeof(VulkanRgba8),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        VkDescriptorSetLayout descriptor_layout = CreateDescriptorSetLayout();
        VkDescriptorPool descriptor_pool = CreateDescriptorPool();
        VkDescriptorSet descriptor_set =
            AllocateDescriptorSet(descriptor_pool, descriptor_layout);
        UpdateDescriptorSet(descriptor_set, image.view, command_buffer.buffer);

        VkShaderModule shader = CreateShaderModule(NUKA_VULKAN_DEBUG_DRAW_SPV);
        VkPipelineLayout pipeline_layout = CreatePipelineLayout(descriptor_layout);
        VkPipeline pipeline = CreateComputePipeline(shader, pipeline_layout);

        RecordAndSubmit(options,
                        static_cast<uint32_t>(commands.size()),
                        descriptor_set,
                        pipeline_layout,
                        pipeline,
                        image.image,
                        readback_buffer.buffer);

        auto pixels = DownloadPixels(readback_buffer, options.width, options.height);

        vkDestroyPipeline(device_, pipeline, nullptr);
        vkDestroyPipelineLayout(device_, pipeline_layout, nullptr);
        vkDestroyShaderModule(device_, shader, nullptr);
        vkDestroyDescriptorPool(device_, descriptor_pool, nullptr);
        vkDestroyDescriptorSetLayout(device_, descriptor_layout, nullptr);
        DestroyImage(image);
        DestroyBuffer(readback_buffer);
        DestroyBuffer(command_buffer);

        VulkanOffscreenReport report;
        report.backend = RenderBackend::Vulkan;
        report.production_backend = true;
        report.width = options.width;
        report.height = options.height;
        report.command_count = static_cast<uint32_t>(commands.size());
        report.physical_device_count = PhysicalDeviceCount();
        report.selected_device_name = selection_.properties.deviceName;
        report.non_background_pixel_count = CountNonBackground(pixels, options.background);
        report.pixels = std::move(pixels);
        return report;
    }

private:
    void CreateDevice() {
        constexpr float kQueuePriority = 1.0f;

        VkDeviceQueueCreateInfo queue_info{};
        queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = selection_.queue_family_index;
        queue_info.queueCount = 1u;
        queue_info.pQueuePriorities = &kQueuePriority;

        VkPhysicalDeviceFeatures features{};

        VkDeviceCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        create_info.queueCreateInfoCount = 1u;
        create_info.pQueueCreateInfos = &queue_info;
        create_info.pEnabledFeatures = &features;

        CheckVk(vkCreateDevice(selection_.physical_device, &create_info, nullptr, &device_),
                "vkCreateDevice");
        vkGetDeviceQueue(device_, selection_.queue_family_index, 0u, &queue_);
    }

    void CreateCommandPool() {
        VkCommandPoolCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        create_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        create_info.queueFamilyIndex = selection_.queue_family_index;
        CheckVk(vkCreateCommandPool(device_, &create_info, nullptr, &command_pool_),
                "vkCreateCommandPool");
    }

    std::vector<HostDebugCommand> BuildHostCommands(
        const std::vector<VulkanDebugDrawCommand>& commands) const {
        std::vector<HostDebugCommand> result;
        result.reserve(std::max<size_t>(commands.size(), 1u));
        for (const auto& command : commands) {
            result.push_back(ToHostCommand(command));
        }
        if (result.empty()) {
            result.push_back(HostDebugCommand{});
        }
        return result;
    }

    BufferResource CreateBuffer(VkDeviceSize size,
                                VkBufferUsageFlags usage,
                                VkMemoryPropertyFlags memory_flags) const {
        BufferResource resource;

        VkBufferCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        create_info.size = size;
        create_info.usage = usage;
        create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        CheckVk(vkCreateBuffer(device_, &create_info, nullptr, &resource.buffer),
                "vkCreateBuffer");

        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device_, resource.buffer, &requirements);

        VkMemoryAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = requirements.size;
        alloc_info.memoryTypeIndex =
            FindMemoryType(selection_.physical_device,
                           requirements.memoryTypeBits,
                           memory_flags);
        CheckVk(vkAllocateMemory(device_, &alloc_info, nullptr, &resource.memory),
                "vkAllocateMemory(buffer)");
        CheckVk(vkBindBufferMemory(device_, resource.buffer, resource.memory, 0),
                "vkBindBufferMemory");
        return resource;
    }

    void UploadBuffer(const BufferResource& buffer, const void* data, size_t size) const {
        void* mapped = nullptr;
        CheckVk(vkMapMemory(device_, buffer.memory, 0, static_cast<VkDeviceSize>(size), 0, &mapped),
                "vkMapMemory(upload)");
        std::memcpy(mapped, data, size);
        vkUnmapMemory(device_, buffer.memory);
    }

    ImageResource CreateStorageImage(uint32_t width, uint32_t height) const {
        ImageResource resource;

        VkImageCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        create_info.imageType = VK_IMAGE_TYPE_2D;
        create_info.format = kOffscreenFormat;
        create_info.extent = {width, height, 1u};
        create_info.mipLevels = 1u;
        create_info.arrayLayers = 1u;
        create_info.samples = VK_SAMPLE_COUNT_1_BIT;
        create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        create_info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        CheckVk(vkCreateImage(device_, &create_info, nullptr, &resource.image),
                "vkCreateImage");

        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(device_, resource.image, &requirements);

        VkMemoryAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = requirements.size;
        alloc_info.memoryTypeIndex =
            FindMemoryType(selection_.physical_device,
                           requirements.memoryTypeBits,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        CheckVk(vkAllocateMemory(device_, &alloc_info, nullptr, &resource.memory),
                "vkAllocateMemory(image)");
        CheckVk(vkBindImageMemory(device_, resource.image, resource.memory, 0),
                "vkBindImageMemory");

        VkImageViewCreateInfo view_info{};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image = resource.image;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = kOffscreenFormat;
        view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view_info.subresourceRange.baseMipLevel = 0u;
        view_info.subresourceRange.levelCount = 1u;
        view_info.subresourceRange.baseArrayLayer = 0u;
        view_info.subresourceRange.layerCount = 1u;
        CheckVk(vkCreateImageView(device_, &view_info, nullptr, &resource.view),
                "vkCreateImageView");
        return resource;
    }

    VkDescriptorSetLayout CreateDescriptorSetLayout() const {
        std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
        bindings[0].binding = 0u;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[0].descriptorCount = 1u;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        bindings[1].binding = 1u;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[1].descriptorCount = 1u;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        create_info.bindingCount = static_cast<uint32_t>(bindings.size());
        create_info.pBindings = bindings.data();

        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        CheckVk(vkCreateDescriptorSetLayout(device_, &create_info, nullptr, &layout),
                "vkCreateDescriptorSetLayout");
        return layout;
    }

    VkDescriptorPool CreateDescriptorPool() const {
        std::array<VkDescriptorPoolSize, 2> sizes{};
        sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        sizes[0].descriptorCount = 1u;
        sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        sizes[1].descriptorCount = 1u;

        VkDescriptorPoolCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        create_info.maxSets = 1u;
        create_info.poolSizeCount = static_cast<uint32_t>(sizes.size());
        create_info.pPoolSizes = sizes.data();

        VkDescriptorPool pool = VK_NULL_HANDLE;
        CheckVk(vkCreateDescriptorPool(device_, &create_info, nullptr, &pool),
                "vkCreateDescriptorPool");
        return pool;
    }

    VkDescriptorSet AllocateDescriptorSet(VkDescriptorPool pool,
                                          VkDescriptorSetLayout layout) const {
        VkDescriptorSetAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool = pool;
        alloc_info.descriptorSetCount = 1u;
        alloc_info.pSetLayouts = &layout;

        VkDescriptorSet set = VK_NULL_HANDLE;
        CheckVk(vkAllocateDescriptorSets(device_, &alloc_info, &set),
                "vkAllocateDescriptorSets");
        return set;
    }

    void UpdateDescriptorSet(VkDescriptorSet set,
                             VkImageView image_view,
                             VkBuffer command_buffer) const {
        VkDescriptorImageInfo image_info{};
        image_info.imageView = image_view;
        image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorBufferInfo buffer_info{};
        buffer_info.buffer = command_buffer;
        buffer_info.offset = 0u;
        buffer_info.range = VK_WHOLE_SIZE;

        std::array<VkWriteDescriptorSet, 2> writes{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = set;
        writes[0].dstBinding = 0u;
        writes[0].descriptorCount = 1u;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[0].pImageInfo = &image_info;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = set;
        writes[1].dstBinding = 1u;
        writes[1].descriptorCount = 1u;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &buffer_info;

        vkUpdateDescriptorSets(device_,
                               static_cast<uint32_t>(writes.size()),
                               writes.data(),
                               0u,
                               nullptr);
    }

    VkShaderModule CreateShaderModule(const std::string& path) const {
        const auto bytes = ReadBinaryFile(path);
        if ((bytes.size() % sizeof(uint32_t)) != 0u) {
            throw std::runtime_error("Vulkan shader bytecode has invalid size: " + path);
        }

        VkShaderModuleCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        create_info.codeSize = bytes.size();
        create_info.pCode = reinterpret_cast<const uint32_t*>(bytes.data());

        VkShaderModule module = VK_NULL_HANDLE;
        CheckVk(vkCreateShaderModule(device_, &create_info, nullptr, &module),
                "vkCreateShaderModule");
        return module;
    }

    VkPipelineLayout CreatePipelineLayout(VkDescriptorSetLayout descriptor_layout) const {
        VkPushConstantRange push_constants{};
        push_constants.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_constants.offset = 0u;
        push_constants.size = sizeof(PushConstants);

        VkPipelineLayoutCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        create_info.setLayoutCount = 1u;
        create_info.pSetLayouts = &descriptor_layout;
        create_info.pushConstantRangeCount = 1u;
        create_info.pPushConstantRanges = &push_constants;

        VkPipelineLayout layout = VK_NULL_HANDLE;
        CheckVk(vkCreatePipelineLayout(device_, &create_info, nullptr, &layout),
                "vkCreatePipelineLayout");
        return layout;
    }

    VkPipeline CreateComputePipeline(VkShaderModule shader,
                                     VkPipelineLayout layout) const {
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = shader;
        stage.pName = "main";

        VkComputePipelineCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        create_info.stage = stage;
        create_info.layout = layout;

        VkPipeline pipeline = VK_NULL_HANDLE;
        CheckVk(vkCreateComputePipelines(device_,
                                         VK_NULL_HANDLE,
                                         1u,
                                         &create_info,
                                         nullptr,
                                         &pipeline),
                "vkCreateComputePipelines");
        return pipeline;
    }

    PushConstants BuildPushConstants(const VulkanOffscreenOptions& options,
                                     uint32_t command_count) const {
        PushConstants push;
        push.width = options.width;
        push.height = options.height;
        push.command_count = command_count;
        push.view_scale = options.view_scale;
        push.view_center[0] = options.view_center.x;
        push.view_center[1] = options.view_center.y;
        push.view_center[2] = options.view_center.z;
        push.view_center[3] = 0.0f;
        push.background[0] = static_cast<float>(options.background.r) / 255.0f;
        push.background[1] = static_cast<float>(options.background.g) / 255.0f;
        push.background[2] = static_cast<float>(options.background.b) / 255.0f;
        push.background[3] = static_cast<float>(options.background.a) / 255.0f;
        return push;
    }

    VkCommandBuffer AllocateCommandBuffer() const {
        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = command_pool_;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 1u;

        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        CheckVk(vkAllocateCommandBuffers(device_, &alloc_info, &command_buffer),
                "vkAllocateCommandBuffers");
        return command_buffer;
    }

    void RecordAndSubmit(const VulkanOffscreenOptions& options,
                         uint32_t command_count,
                         VkDescriptorSet descriptor_set,
                         VkPipelineLayout pipeline_layout,
                         VkPipeline pipeline,
                         VkImage image,
                         VkBuffer readback_buffer) const {
        VkCommandBuffer command_buffer = AllocateCommandBuffer();

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        CheckVk(vkBeginCommandBuffer(command_buffer, &begin_info), "vkBeginCommandBuffer");

        VkImageMemoryBarrier to_general{};
        to_general.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        to_general.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        to_general.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        to_general.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_general.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_general.image = image;
        to_general.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        to_general.subresourceRange.levelCount = 1u;
        to_general.subresourceRange.layerCount = 1u;
        to_general.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(command_buffer,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0u,
                             0u,
                             nullptr,
                             0u,
                             nullptr,
                             1u,
                             &to_general);

        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(command_buffer,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline_layout,
                                0u,
                                1u,
                                &descriptor_set,
                                0u,
                                nullptr);

        const auto push = BuildPushConstants(options, command_count);
        vkCmdPushConstants(command_buffer,
                           pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0u,
                           sizeof(PushConstants),
                           &push);

        vkCmdDispatch(command_buffer,
                      (options.width + 15u) / 16u,
                      (options.height + 15u) / 16u,
                      1u);

        VkImageMemoryBarrier to_transfer{};
        to_transfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        to_transfer.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        to_transfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        to_transfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_transfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_transfer.image = image;
        to_transfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        to_transfer.subresourceRange.levelCount = 1u;
        to_transfer.subresourceRange.layerCount = 1u;
        to_transfer.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        to_transfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(command_buffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0u,
                             0u,
                             nullptr,
                             0u,
                             nullptr,
                             1u,
                             &to_transfer);

        VkBufferImageCopy copy_region{};
        copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy_region.imageSubresource.layerCount = 1u;
        copy_region.imageExtent = {options.width, options.height, 1u};
        vkCmdCopyImageToBuffer(command_buffer,
                               image,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               readback_buffer,
                               1u,
                               &copy_region);

        CheckVk(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1u;
        submit_info.pCommandBuffers = &command_buffer;
        CheckVk(vkQueueSubmit(queue_, 1u, &submit_info, VK_NULL_HANDLE), "vkQueueSubmit");
        CheckVk(vkQueueWaitIdle(queue_), "vkQueueWaitIdle");

        vkFreeCommandBuffers(device_, command_pool_, 1u, &command_buffer);
    }

    std::vector<VulkanRgba8> DownloadPixels(const BufferResource& buffer,
                                            uint32_t width,
                                            uint32_t height) const {
        const size_t byte_count = static_cast<size_t>(width) * height * sizeof(VulkanRgba8);
        std::vector<VulkanRgba8> pixels(static_cast<size_t>(width) * height);

        void* mapped = nullptr;
        CheckVk(vkMapMemory(device_, buffer.memory, 0,
                            static_cast<VkDeviceSize>(byte_count), 0, &mapped),
                "vkMapMemory(readback)");
        std::memcpy(pixels.data(), mapped, byte_count);
        vkUnmapMemory(device_, buffer.memory);
        return pixels;
    }

    uint32_t PhysicalDeviceCount() const {
        uint32_t count = 0;
        CheckVk(vkEnumeratePhysicalDevices(instance_, &count, nullptr),
                "vkEnumeratePhysicalDevices(report count)");
        return count;
    }

    void DestroyBuffer(BufferResource& resource) const {
        if (resource.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, resource.buffer, nullptr);
            resource.buffer = VK_NULL_HANDLE;
        }
        if (resource.memory != VK_NULL_HANDLE) {
            vkFreeMemory(device_, resource.memory, nullptr);
            resource.memory = VK_NULL_HANDLE;
        }
    }

    void DestroyImage(ImageResource& resource) const {
        if (resource.view != VK_NULL_HANDLE) {
            vkDestroyImageView(device_, resource.view, nullptr);
            resource.view = VK_NULL_HANDLE;
        }
        if (resource.image != VK_NULL_HANDLE) {
            vkDestroyImage(device_, resource.image, nullptr);
            resource.image = VK_NULL_HANDLE;
        }
        if (resource.memory != VK_NULL_HANDLE) {
            vkFreeMemory(device_, resource.memory, nullptr);
            resource.memory = VK_NULL_HANDLE;
        }
    }

    VkInstance instance_ = VK_NULL_HANDLE;
    QueueSelection selection_;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
};

} // namespace

VulkanRendererReport ProbeVulkanRenderer() {
    uint32_t loader_api_version = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion(&loader_api_version) != VK_SUCCESS) {
        loader_api_version = VK_API_VERSION_1_0;
    }

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "nuka-physics";
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.pEngineName = "nuka";
    app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    VkInstance instance = VK_NULL_HANDLE;
    CheckVk(vkCreateInstance(&create_info, nullptr, &instance), "vkCreateInstance");

    uint32_t device_count = 0;
    CheckVk(vkEnumeratePhysicalDevices(instance, &device_count, nullptr),
            "vkEnumeratePhysicalDevices(count)");

    std::vector<VkPhysicalDevice> devices(device_count);
    if (device_count > 0) {
        CheckVk(vkEnumeratePhysicalDevices(instance, &device_count, devices.data()),
                "vkEnumeratePhysicalDevices(list)");
    }

    VulkanRendererReport report;
    report.backend = RenderBackend::Vulkan;
    report.production_backend = true;
    report.instance_api_version_major = VK_VERSION_MAJOR(loader_api_version);
    report.instance_api_version_minor = VK_VERSION_MINOR(loader_api_version);
    report.instance_api_version_patch = VK_VERSION_PATCH(loader_api_version);
    report.physical_device_count = device_count;

    if (!devices.empty()) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(devices.front(), &properties);
        report.selected_device_name = properties.deviceName;
    }

    vkDestroyInstance(instance, nullptr);
    return report;
}

VulkanOffscreenReport RenderDebugDrawListVulkan(
    const std::vector<VulkanDebugDrawCommand>& commands,
    const VulkanOffscreenOptions& options) {
    VulkanOffscreenRenderer renderer;
    return renderer.Render(commands, options);
}

VulkanOffscreenReport RenderSceneVulkan(
    const RenderScene& scene,
    const VulkanOffscreenOptions& options) {
    return RenderDebugDrawListVulkan(BuildRenderSceneCommands(scene), options);
}

} // namespace nuka::render
