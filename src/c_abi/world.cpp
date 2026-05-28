#include "c_abi/handle_table.hpp"
#include "c_abi/internal.hpp"

#include "import/mjcf_importer.hpp"
#include "import/urdf_importer.hpp"
#include "import/usd_importer.hpp"
#include "runtime/articulation/featherstone_aba.hpp"
#include "scene/cooker.hpp"

#include <algorithm>
#include <cmath>
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

void SampleInvariants(WorldRecord& record, uint32_t step_index) {
    record.last_invariant_violations.clear();
    const core::diagnostics::InvariantWorldView view{
        &record.world.template_view,
        &record.world.instance,
        nullptr,
        record.step_options.gravity,
    };
    record.invariant_sampler.Sample(view,
                                    step_index,
                                    &record.last_invariant_violations);
}

float ClampDriveLimit(float limit) {
    return limit > 0.0f ? limit : 0.0f;
}

float DefaultDriveDamping(float stiffness) {
    return 2.0f * std::sqrt(std::max(stiffness, 0.0f));
}

void BuildHoldDriveTargets(WorldRecord& record) {
    const auto link_count = record.articulation_host.TotalLinkCount();
    record.drive_targets_host.assign(link_count, 0.0f);
    record.drive_stiffness_host.assign(link_count, 0.0f);
    record.drive_damping_host.assign(link_count, 0.0f);
    record.drive_force_limits_host.assign(link_count, 0.0f);
    if (link_count == 0u) {
        return;
    }

    record.drive_targets_host = record.articulation_host.q;

    const auto& actuator_table = record.world.template_view.actuator_table;
    const auto& joint_table = record.world.template_view.joint_table;
    for (uint32_t actuator = 0u;
         actuator < record.world.template_view.actuator_count;
         ++actuator) {
        if (actuator >= actuator_table.joint_ids.size() ||
            actuator >= actuator_table.types.size() ||
            actuator_table.types[actuator] != scene::ActuatorType::Position) {
            continue;
        }
        const scene::JointId joint = actuator_table.joint_ids[actuator];
        if (joint >= joint_table.child_bodies.size()) {
            continue;
        }
        const scene::BodyId child_body = joint_table.child_bodies[joint];
        auto link_it = std::find(record.articulation_host.link_body.begin(),
                                 record.articulation_host.link_body.end(),
                                 child_body);
        if (link_it == record.articulation_host.link_body.end()) {
            continue;
        }
        const uint32_t link = static_cast<uint32_t>(
            std::distance(record.articulation_host.link_body.begin(), link_it));
        if (link >= link_count ||
            record.articulation_host.joint_type[link] ==
                runtime::articulation::ArticulationJointType::Fixed) {
            continue;
        }
        const float gain = actuator < actuator_table.gains.size()
            ? std::max(actuator_table.gains[actuator], 0.0f)
            : 0.0f;
        const float force_limit = actuator < actuator_table.force_limits.size()
            ? ClampDriveLimit(actuator_table.force_limits[actuator])
            : 0.0f;
        if (gain > 0.0f) {
            record.drive_stiffness_host[link] = gain;
            record.drive_damping_host[link] = DefaultDriveDamping(gain);
        }
        if (force_limit > 0.0f) {
            record.drive_force_limits_host[link] = force_limit;
        }
    }
}

void UploadHoldDriveTargets(WorldRecord& record) {
    const auto upload = [](const std::vector<float>& values, phi::Buffer* out) {
        if (values.empty()) {
            *out = phi::Buffer();
            return;
        }
        *out = phi::Buffer(values.size() * sizeof(float), phi::MemoryKind::Device);
        out->CopyFromHost(values.data(), values.size() * sizeof(float));
    };
    upload(record.drive_targets_host, &record.drive_targets_device);
    upload(record.drive_stiffness_host, &record.drive_stiffness_device);
    upload(record.drive_damping_host, &record.drive_damping_device);
    upload(record.drive_force_limits_host, &record.drive_force_limits_device);
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
        runtime::articulation::FeatherstoneAba::ApplyPositionDrives(
            record.device->context,
            record.articulation_device.View(),
            static_cast<const float*>(record.drive_targets_device.Data()),
            static_cast<const float*>(record.drive_stiffness_device.Data()),
            static_cast<const float*>(record.drive_damping_device.Data()),
            static_cast<const float*>(record.drive_force_limits_device.Data()));
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
    record.simulated_step_count += step_count;
    SampleInvariants(record, record.simulated_step_count);
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
        nuka::c_abi::BuildHoldDriveTargets(*record);
        nuka::c_abi::UploadHoldDriveTargets(*record);
        nuka::c_abi::SampleInvariants(*record, record->simulated_step_count);
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
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    if (array_capacity > 0u && out_array == nullptr) {
        return NUKA_RESULT_INVALID_ARG;
    }

    const auto& violations = record->last_invariant_violations;
    *out_count = static_cast<uint32_t>(violations.size());
    if (out_array == nullptr || array_capacity == 0u || violations.empty()) {
        return NUKA_RESULT_OK;
    }

    const uint32_t copy_count =
        std::min(array_capacity, static_cast<uint32_t>(violations.size()));
    auto* out = static_cast<nuka_invariant_violation_t*>(out_array);
    for (uint32_t index = 0u; index < copy_count; ++index) {
        const auto& violation = violations[index];
        out[index].invariant = static_cast<uint32_t>(violation.which);
        out[index].step = violation.step_index;
        out[index].env_id = violation.env_id;
        out[index].value = violation.value;
        out[index].threshold = violation.threshold;
    }
    return NUKA_RESULT_OK;
}

} // extern "C"
