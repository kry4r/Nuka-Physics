#pragma once

#include "nuka/nuka.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <memory>
#include <new>
#include <string_view>
#include <utility>

namespace nuka {

class Error {
public:
    explicit Error(nuka_result_t code) noexcept : code_(code) {}

    nuka_result_t code() const noexcept { return code_; }
    const char* message() const noexcept { return nuka_result_message(code_); }

private:
    nuka_result_t code_;
};

class Device {
public:
    static std::expected<Device, Error> Create(uint32_t gpu_index,
                                               cudaStream_t stream = nullptr) {
        nuka_device_desc_t desc{};
        desc.gpu_index = gpu_index;
        desc.cuda_stream = stream;
        desc.backend_selection_layer_enabled = 1u;

        nuka_device_handle handle = nullptr;
        const nuka_result_t result = nuka_device_create(&desc, &handle);
        if (result != NUKA_RESULT_OK) {
            return std::unexpected(Error(result));
        }
        return Device(handle);
    }

    Device() = default;
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    Device(Device&& other) noexcept : h_(std::exchange(other.h_, nullptr)) {}

    Device& operator=(Device&& other) noexcept {
        if (this != &other) {
            Reset();
            h_ = std::exchange(other.h_, nullptr);
        }
        return *this;
    }

    ~Device() noexcept { Reset(); }

    nuka_device_handle raw() const noexcept { return h_; }

private:
    explicit Device(nuka_device_handle handle) noexcept : h_(handle) {}

    void Reset() noexcept {
        if (h_ != nullptr) {
            nuka_device_destroy(h_);
            h_ = nullptr;
        }
    }

    nuka_device_handle h_ = nullptr;
};

class World {
public:
    struct BufferView {
        void* device_ptr = nullptr;
        size_t element_count = 0u;
        uint32_t stride = 0u;
        uint8_t dtype = 0u;
    };

    static std::expected<World, Error> CreateFromScene(const Device& device,
                                                       std::string_view path,
                                                       uint32_t env_count,
                                                       float dt) {
        auto path_storage = std::unique_ptr<char[]>(new (std::nothrow) char[path.size() + 1u]);
        if (!path_storage) {
            return std::unexpected(Error(NUKA_RESULT_OUT_OF_MEMORY));
        }
        if (!path.empty()) {
            std::memcpy(path_storage.get(), path.data(), path.size());
        }
        path_storage[path.size()] = '\0';

        nuka_world_desc_t desc{};
        desc.scene_path = path_storage.get();
        desc.env_count = env_count;
        desc.fixed_dt = dt;

        nuka_world_handle handle = nullptr;
        const nuka_result_t result =
            nuka_world_create_from_scene(device.raw(), &desc, &handle);
        if (result != NUKA_RESULT_OK) {
            return std::unexpected(Error(result));
        }
        return World(handle);
    }

    World() = default;
    World(const World&) = delete;
    World& operator=(const World&) = delete;

    World(World&& other) noexcept : h_(std::exchange(other.h_, nullptr)) {}

    World& operator=(World&& other) noexcept {
        if (this != &other) {
            Reset();
            h_ = std::exchange(other.h_, nullptr);
        }
        return *this;
    }

    ~World() noexcept { Reset(); }

    nuka_world_handle raw() const noexcept { return h_; }

    std::expected<void, Error> Step() noexcept {
        return StepN(1u);
    }

    std::expected<void, Error> StepN(uint32_t step_count) noexcept {
        const nuka_result_t result = nuka_world_step_n(h_, step_count);
        if (result != NUKA_RESULT_OK) {
            return std::unexpected(Error(result));
        }
        return std::expected<void, Error>{};
    }

    std::expected<BufferView, Error> GetBufferView(nuka_state_field_t field) const noexcept {
        nuka_buffer_view_t view{};
        const nuka_result_t result = nuka_world_get_buffer_view(h_, field, &view);
        if (result != NUKA_RESULT_OK) {
            return std::unexpected(Error(result));
        }
        return BufferView{
            view.device_ptr,
            view.element_count,
            view.element_stride_bytes,
            view.dtype,
        };
    }

private:
    explicit World(nuka_world_handle handle) noexcept : h_(handle) {}

    void Reset() noexcept {
        if (h_ != nullptr) {
            nuka_world_destroy(h_);
            h_ = nullptr;
        }
    }

    nuka_world_handle h_ = nullptr;
};

} // namespace nuka
