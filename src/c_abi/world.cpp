#include "c_abi/handle_table.hpp"
#include "c_abi/internal.hpp"

#include "import/mjcf_importer.hpp"
#include "import/urdf_importer.hpp"
#include "import/usd_importer.hpp"
#include "runtime/articulation/featherstone_aba.hpp"
#include "scene/cooker.hpp"

#include <algorithm>
#include <exception>
#include <filesystem>
#include <memory>
#include <string>

namespace nuka::c_abi {

namespace {

bool LoadSceneByExtension(const char* scene_path, scene::SceneIR* out_scene) {
    if (out_scene == nullptr) {
        return false;
    }
    const std::filesystem::path path(scene_path);
    const std::string extension = path.extension().string();
    if (extension == ".xml" || extension == ".mjcf") {
        *out_scene = import::LoadMjcf(scene_path);
        return true;
    }
    if (extension == ".urdf") {
        *out_scene = import::LoadUrdf(scene_path);
        return true;
    }
    if (extension == ".usd" || extension == ".usda") {
        *out_scene = import::LoadUsd(scene_path);
        return true;
    }
    return false;
}

void SyncHostToInstance(const runtime::articulation::ArticulationHostState& host,
                        runtime::WorldInstance* instance) {
    if (instance == nullptr) {
        return;
    }
    instance->articulation_q = host.q;
    instance->articulation_qdot = host.qdot;
    instance->articulation_qddot = host.qddot;
    instance->articulation_tau = host.tau;
}

void DownloadDeviceStateToInstance(WorldRecord& record) {
    runtime::articulation::DownloadArticulationState(record.articulation_device,
                                                     &record.articulation_host);
    SyncHostToInstance(record.articulation_host, &record.world.instance);
}

nuka_result_t StepWorldGpu(WorldRecord& record, uint32_t step_count) {
    if (step_count == 0u) {
        return NUKA_RESULT_OK;
    }
    if (record.device == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    if (record.world.template_view.articulations.empty()) {
        return NUKA_RESULT_NOT_SUPPORTED;
    }

    if (record.articulation_device.Empty()) {
        record.articulation_device =
            runtime::articulation::UploadArticulationState(record.device->context,
                                                           record.articulation_host);
    }

    for (uint32_t step = 0u; step < step_count; ++step) {
        runtime::articulation::FeatherstoneAba::ComputeAccelerations(
            record.device->context,
            record.articulation_device.View(),
            record.step_options.gravity.z);
        runtime::articulation::FeatherstoneAba::Integrate(record.device->context,
                                                          record.articulation_device.View(),
                                                          record.step_options.dt);
    }
    record.device->context.stream.Synchronize();
    DownloadDeviceStateToInstance(record);
    return NUKA_RESULT_OK;
}

} // namespace

nuka_result_t RefreshWorldBuffers(WorldRecord& record) noexcept {
    try {
        if (!record.world.instance.articulation_q.empty()) {
            record.joint_position_buffer =
                phi::Buffer(record.world.instance.articulation_q.size() * sizeof(float),
                            phi::MemoryKind::Device);
            record.joint_position_buffer.CopyFromHost(
                record.world.instance.articulation_q.data(),
                record.world.instance.articulation_q.size() * sizeof(float));
        }
        if (!record.world.instance.articulation_qdot.empty()) {
            record.joint_velocity_buffer =
                phi::Buffer(record.world.instance.articulation_qdot.size() * sizeof(float),
                            phi::MemoryKind::Device);
            record.joint_velocity_buffer.CopyFromHost(
                record.world.instance.articulation_qdot.data(),
                record.world.instance.articulation_qdot.size() * sizeof(float));
        }
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

} // namespace nuka::c_abi

extern "C" {

nuka_result_t nuka_world_create_from_scene(nuka_device_handle device,
                                           const nuka_world_desc_t* desc,
                                           nuka_world_handle* out) {
    if (out == nullptr) {
        return NUKA_RESULT_INVALID_ARG;
    }
    *out = nullptr;
    if (desc == nullptr || desc->scene_path == nullptr || desc->fixed_dt <= 0.0f) {
        return NUKA_RESULT_INVALID_ARG;
    }
    if (desc->env_count != 1u) {
        return NUKA_RESULT_NOT_SUPPORTED;
    }

    auto* device_record = nuka::c_abi::DeviceTable().Get(device);
    if (device_record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }

    try {
        nuka::scene::SceneIR scene;
        if (!nuka::c_abi::LoadSceneByExtension(desc->scene_path, &scene)) {
            return NUKA_RESULT_NOT_SUPPORTED;
        }
        const auto blob = nuka::scene::CookScene(scene);
        auto record = std::make_unique<nuka::c_abi::WorldRecord>();
        record->device = device_record;
        record->world = nuka::runtime::BuildWorld(blob);
        record->step_options.dt = desc->fixed_dt;
        record->step_options.gravity = {0.0f, 0.0f, -9.81f};
        record->step_options.enable_contacts = false;
        record->step_options.solver_position_iterations = 0u;
        record->articulation_host =
            nuka::runtime::articulation::BuildArticulationHostState(
                record->world.template_view.articulations,
                record->world.template_view.body_table);
        nuka::c_abi::SyncHostToInstance(record->articulation_host, &record->world.instance);
        if (!record->world.template_view.articulations.empty()) {
            record->articulation_device =
                nuka::runtime::articulation::UploadArticulationState(
                    record->device->context,
                    record->articulation_host);
        }
        const nuka_result_t refresh_result = nuka::c_abi::RefreshWorldBuffers(*record);
        if (refresh_result != NUKA_RESULT_OK) {
            return refresh_result;
        }
        *out = nuka::c_abi::WorldTable().Insert(std::move(record));
        return *out == nullptr ? NUKA_RESULT_INTERNAL : NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

void nuka_world_destroy(nuka_world_handle world) {
    (void)nuka::c_abi::WorldTable().Remove(world);
}

nuka_result_t nuka_world_step(nuka_world_handle world) {
    return nuka_world_step_n(world, 1u);
}

nuka_result_t nuka_world_step_n(nuka_world_handle world, uint32_t step_count) {
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }

    try {
        const nuka_result_t step_result = nuka::c_abi::StepWorldGpu(*record, step_count);
        if (step_result != NUKA_RESULT_OK) {
            return step_result;
        }
        return nuka::c_abi::RefreshWorldBuffers(*record);
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_world_get_last_invariant_violations(nuka_world_handle world,
                                                       uint32_t* out_count,
                                                       void* out_array,
                                                       uint32_t array_capacity) {
    if (out_count == nullptr) {
        return NUKA_RESULT_INVALID_ARG;
    }
    if (nuka::c_abi::WorldTable().Get(world) == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    (void)out_array;
    (void)array_capacity;
    *out_count = 0u;
    return NUKA_RESULT_OK;
}

} // extern "C"
