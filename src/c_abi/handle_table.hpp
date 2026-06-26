#pragma once

#include "nuka/nuka.h"
#include "nuka/nuka_scene.h"  // nuka_scene_t (the built-scene handle tag)

#include <cstdint>
#include <mutex>
#include <memory>
#include <unordered_map>

namespace nuka::c_abi {

template <typename Tag, typename T>
class HandleTable {
public:
    using Handle = Tag*;

    Handle Insert(std::unique_ptr<T> value) {
        if (!value) {
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        const uintptr_t id = next_id_++;
        values_.emplace(id, std::move(value));
        return reinterpret_cast<Handle>(id);
    }

    T* Get(Handle handle) {
        if (handle == nullptr) {
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = values_.find(reinterpret_cast<uintptr_t>(handle));
        if (it == values_.end()) {
            return nullptr;
        }
        return it->second.get();
    }

    std::unique_ptr<T> Remove(Handle handle) {
        if (handle == nullptr) {
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = values_.find(reinterpret_cast<uintptr_t>(handle));
        if (it == values_.end()) {
            return nullptr;
        }
        auto value = std::move(it->second);
        values_.erase(it);
        return value;
    }

private:
    std::mutex mutex_;
    uintptr_t next_id_ = 1u;
    std::unordered_map<uintptr_t, std::unique_ptr<T>> values_;
};

struct DeviceRecord;
struct WorldRecord;
struct SceneRecord;

HandleTable<nuka_device_t, DeviceRecord>& DeviceTable();
HandleTable<nuka_world_t, WorldRecord>& WorldTable();
// The built-scene authoring handle table (shared so the scene-builder TU + the
// built-scene world create reach the SAME SceneRecord the load/edit surface owns).
HandleTable<nuka_scene_t, SceneRecord>& SceneTable();

nuka_result_t MapExceptionToResult(const std::exception& error) noexcept;
nuka_result_t RunNoThrow(void (*fn)(void*), void* user) noexcept;

} // namespace nuka::c_abi
