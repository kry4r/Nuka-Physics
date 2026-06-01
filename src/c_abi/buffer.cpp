#include "c_abi/handle_table.hpp"
#include "c_abi/internal.hpp"

#include <exception>

extern "C" {

nuka_result_t nuka_world_get_buffer_view(nuka_world_handle world,
                                         nuka_state_field_t field,
                                         nuka_buffer_view_t* out) {
    if (out == nullptr) {
        return NUKA_RESULT_INVALID_ARG;
    }
    out->device_ptr = nullptr;
    out->element_count = 0u;
    out->element_stride_bytes = 0u;
    out->dtype = 0u;

    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }

    try {
        // --- Multi-env batched path (env_count > 1) -----------------------
        // Serve the batched device buffers zero-copy (env-major). All of q/qdot/
        // drive-targets have length env_count * base_link_count and the IDENTICAL
        // env-major layout (index env*base_link_count + link). ARTICULATION_LINK_POSE
        // exposes the live per-link world pose (math::Transform, 7 floats). No host
        // round-trip / RefreshWorldBuffers on this arm.
        //
        // Layout reference (env-major; base_link_count = state.total_link_count /
        // env_count; the articulation ROOT/base is local link 0, so env e's base is
        // element e*base_link_count). math::Transform = {Vec3 position(px,py,pz),
        // Quat rotation(qw,qx,qy,qz)} == 7 contiguous floats / 28 bytes, NO padding;
        // NOTE the quaternion order is w,x,y,z (w FIRST), not the common xyzw.
        if (record->batched != nullptr) {
            auto& batched = *record->batched;
            if (field == NUKA_FIELD_JOINT_POSITION) {
                const auto state = batched.View();
                out->device_ptr = state.q;
                out->element_count = state.total_link_count;
                out->element_stride_bytes = sizeof(float);
                out->dtype = 0u;
                return NUKA_RESULT_OK;
            }
            if (field == NUKA_FIELD_JOINT_VELOCITY) {
                const auto state = batched.View();
                out->device_ptr = state.qdot;
                out->element_count = state.total_link_count;
                out->element_stride_bytes = sizeof(float);
                out->dtype = 0u;
                return NUKA_RESULT_OK;
            }
            if (field == NUKA_FIELD_ARTICULATION_LINK_POSE) {
                // Live per-link WORLD pose, refreshed each Step by the batched
                // pipeline's UpdateWorldLinkPoses (stage 4). element_count ==
                // env_count * base_link_count; the base/root of env e is element
                // e*base_link_count (root == local link 0). Each element is a
                // math::Transform = 7 floats [px,py,pz, qw,qx,qy,qz] (quat w-first).
                // CAVEAT: UpdateWorldLinkPoses runs from the PRE-integrate q (stage
                // 4) while q itself is advanced at stage 11 with no later FK, so
                // after Step() returns link_pose reflects the PREVIOUS step's q
                // (one-step lag vs JOINT_POSITION). Before the first Step it is the
                // cooked rest pose (not the assembled FK pose). Both are intrinsic
                // to the validated T6 step and intentionally not "fixed" here.
                const auto state = batched.View();
                out->device_ptr = state.link_pose;
                out->element_count = state.total_link_count;
                out->element_stride_bytes =
                    static_cast<uint32_t>(sizeof(nuka::math::Transform));
                out->dtype = 0u;
                return NUKA_RESULT_OK;
            }
            if (field == NUKA_FIELD_BASE_POSE) {
                // AUTHORITATIVE per-env floating-base root world pose: the
                // internal base_pose[articulation] buffer the integrator advances
                // (stage 11) and reset_envs restores from the creation snapshot.
                // element_count == articulation_count == env_count (ONE Transform
                // per env, NOT per link); each element is a math::Transform == 7
                // floats [px,py,pz, qw,qx,qy,qz] (quat w-first), the SAME element
                // layout as ARTICULATION_LINK_POSE's ROOT slot. UNLIKE that field
                // it carries NO one-step FK lag: it is correct after Step() and --
                // the property RL autoreset depends on -- correct IMMEDIATELY after
                // reset_envs with no further step (the lagged ARTICULATION_LINK_POSE
                // still shows the pre-reset/fallen pose until the next Step's FK).
                const auto state = batched.View();
                out->device_ptr = state.base_pose;
                out->element_count = state.articulation_count;
                out->element_stride_bytes =
                    static_cast<uint32_t>(sizeof(nuka::math::Transform));
                out->dtype = 0u;
                return NUKA_RESULT_OK;
            }
            if (field == NUKA_FIELD_DRIVE_TARGET) {
                // WRITABLE view aliasing the live per-env PD drive-target buffer
                // that batched_step_params.drive_targets points at. Writing it in
                // place (device-to-device or host-to-device) is picked up by the
                // NEXT nuka_world_step. Same layout as JOINT_POSITION: float[
                // env_count*base_link_count], env-major. Only actuated links (those
                // with non-zero drive stiffness) act on their target.
                if (record->batched_drive_targets_device.Size() == 0u) {
                    return NUKA_RESULT_NOT_SUPPORTED;
                }
                out->device_ptr = record->batched_drive_targets_device.Data();
                out->element_count =
                    record->batched_drive_targets_device.Size() / sizeof(float);
                out->element_stride_bytes = sizeof(float);
                out->dtype = 0u;
                return NUKA_RESULT_OK;
            }
            if (field == NUKA_FIELD_LINK_VELOCITY) {
                // READ-only per-link spatial velocity, env-major, SAME indexing
                // as ARTICULATION_LINK_POSE (element_count == env_count*
                // base_link_count; env e's root == element e*base_link_count).
                // Each element is a LinkSpatialVel == 6 floats [wx,wy,wz, vx,vy,vz]
                // (omega-first). The ROOT slot is the live floating-base spatial
                // velocity in the ROOT-LINK BODY frame (already body-local; written
                // post-integrate AND post-contact each Step, so no link_pose-style
                // lag). Non-root slots are Featherstone per-link-local velocities
                // and are not the policy observation (see nuka.h).
                const auto state = batched.View();
                out->device_ptr = state.link_velocity;
                out->element_count = state.total_link_count;
                out->element_stride_bytes =
                    static_cast<uint32_t>(
                        sizeof(nuka::runtime::articulation::LinkSpatialVel));
                out->dtype = 0u;
                return NUKA_RESULT_OK;
            }
            if (field == NUKA_FIELD_DRIVE_STIFFNESS ||
                field == NUKA_FIELD_DRIVE_DAMPING ||
                field == NUKA_FIELD_DRIVE_FORCE_LIMIT) {
                // WRITABLE views aliasing the live per-env PD GAIN buffers that
                // batched_step_params.{drive_stiffness,drive_damping,
                // drive_force_limits} point at. Writing them in place is picked up
                // by the NEXT nuka_world_step. Same layout as DRIVE_TARGET:
                // float[env_count*base_link_count], env-major. Only actuated links
                // (slots 1..12) use their gains; root/slot-0 is a no-op.
                const nuka::phi::Buffer& gains =
                    (field == NUKA_FIELD_DRIVE_STIFFNESS)
                        ? record->batched_drive_stiffness_device
                        : (field == NUKA_FIELD_DRIVE_DAMPING)
                              ? record->batched_drive_damping_device
                              : record->batched_drive_force_limits_device;
                if (gains.Size() == 0u) {
                    return NUKA_RESULT_NOT_SUPPORTED;
                }
                out->device_ptr = const_cast<void*>(gains.Data());
                out->element_count = gains.Size() / sizeof(float);
                out->element_stride_bytes = sizeof(float);
                out->dtype = 0u;
                return NUKA_RESULT_OK;
            }
            if (field == NUKA_FIELD_TORQUE_INPUT ||
                field == NUKA_FIELD_VELOCITY_TARGET ||
                field == NUKA_FIELD_ACTUATOR_NOLOAD_SPEED) {
                // v0.5 C-fwd: WRITABLE views aliasing the batched world's owned,
                // always-allocated control-input buffers (per-link, env-major;
                // SAME layout as DRIVE_TARGET: float[env_count*base_link_count]).
                // Writing them in place is picked up by the NEXT nuka_world_step
                // WHEN the world's control_mode selects the matching mode
                // (TORQUE_INPUT -> Torque/Actuator, VELOCITY_TARGET -> Velocity,
                // ACTUATOR_NOLOAD_SPEED -> Actuator); otherwise the buffers exist
                // but are never read. Allocated for every batched world so the view
                // always succeeds.
                const nuka::phi::Buffer& buf =
                    (field == NUKA_FIELD_TORQUE_INPUT)
                        ? batched.TorqueInputBuffer()
                        : (field == NUKA_FIELD_VELOCITY_TARGET)
                              ? batched.VelocityTargetBuffer()
                              : batched.ActuatorNoloadSpeedBuffer();
                if (buf.Size() == 0u) {
                    return NUKA_RESULT_NOT_SUPPORTED;
                }
                out->device_ptr = const_cast<void*>(buf.Data());
                out->element_count = buf.Size() / sizeof(float);
                out->element_stride_bytes = sizeof(float);
                out->dtype = 0u;
                return NUKA_RESULT_OK;
            }
            if (field == NUKA_FIELD_TASK_TARGET) {
                // v0.5 C-fwd slice 3: WRITABLE per-ENV (NOT per-link) Osc task
                // target. element_count == env_count, element_stride_bytes ==
                // 3*sizeof(float) (a {x,y,z} world target per env). Aliases the
                // batched world's owned, always-allocated buffer (write IN PLACE;
                // picked up by the NEXT step WHEN control_mode == Osc, else exists
                // but is never read). Separate branch from the per-link control
                // fields because the layout/stride differ.
                const nuka::phi::Buffer& buf = batched.TaskTargetBuffer();
                if (buf.Size() == 0u) {
                    return NUKA_RESULT_NOT_SUPPORTED;
                }
                out->device_ptr = const_cast<void*>(buf.Data());
                out->element_count = buf.Size() / (3u * sizeof(float));
                out->element_stride_bytes = 3u * sizeof(float);
                out->dtype = 0u;
                return NUKA_RESULT_OK;
            }
            if (field == NUKA_FIELD_CONTACT_POINTS) {
                // Vec3 (3 floats) per contact slot; slot_count == env_count *
                // kMaxFootContactsPerEnv. Inactive slots are zero (see T2 doc).
                const nuka::phi::Buffer& points = batched.ContactPointBuffer();
                out->device_ptr = const_cast<void*>(points.Data());
                out->element_count = batched.SlotCount();
                out->element_stride_bytes =
                    static_cast<uint32_t>(points.Size() / batched.SlotCount());
                out->dtype = 0u;
                return NUKA_RESULT_OK;
            }
            return NUKA_RESULT_NOT_SUPPORTED;
        }

        const nuka_result_t refresh_result = nuka::c_abi::RefreshWorldBuffers(*record);
        if (refresh_result != NUKA_RESULT_OK) {
            return refresh_result;
        }

        if (field == NUKA_FIELD_JOINT_POSITION) {
            out->device_ptr = record->joint_position_buffer.Data();
            out->element_count = record->joint_position_buffer.Size() / sizeof(float);
            out->element_stride_bytes = sizeof(float);
            out->dtype = 0u;
            return NUKA_RESULT_OK;
        }
        if (field == NUKA_FIELD_JOINT_VELOCITY) {
            out->device_ptr = record->joint_velocity_buffer.Data();
            out->element_count = record->joint_velocity_buffer.Size() / sizeof(float);
            out->element_stride_bytes = sizeof(float);
            out->dtype = 0u;
            return NUKA_RESULT_OK;
        }
        return NUKA_RESULT_NOT_SUPPORTED;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

} // extern "C"
