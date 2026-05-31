#include "c_abi/handle_table.hpp"
#include "c_abi/internal.hpp"

#include "import/mjcf_importer.hpp"
#include "import/urdf_importer.hpp"
#include "import/usd_importer.hpp"
#include "runtime/articulation/articulation_contacts.hpp"
#include "runtime/articulation/featherstone_aba.hpp"
#include "runtime/gpu/batched_articulated_world.hpp"
#include "scene/cooker.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <vector>

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

// --- Multi-env batched articulated path (p01-F T7) ------------------------

// Small penetration (m) baked into the derived ground height so the rest-pose
// feet engage the ground with contacts ACTIVE from the first step. The detection
// convention is depth = (ground_height + radius) - foot_world_z, so a contact
// fires only for depth > 0; ground = min_foot_bottom_z + kRestFootPenetration
// places the lowest foot kRestFootPenetration metres below the surface.
//
// Value = 0.002 m, matching the C++ standing test's SeatGround (test_go2_pd_standing
// .cpp). This was 0.03 m; sim-val #42 ISOLATED the seating depth as the cause of a
// floating-base PD-stance instability on this very stepper, via a clean A/B (only
// this constant changed; same gains/pose/dt): the floating Go2 PD crouch COLLAPSES at
// 0.03 m for dt>=0.002 (surviving only at dt=0.001) but HOLDS at 0.002 m for EVERY dt
// incl. the natural training dt 0.005 (probe: max tilt <0.8 deg / 6 s). The causal
// variable (penetration depth) is isolated; the MECHANISM is only inferred (the
// deeper initial seat plausibly injects a larger startup contact-correction
// transient) and NOT fully pinned -- the dt-dependence of the 0.03 m collapse is
// non-monotonic -- so we anchor to the empirical result + the validated C++ seat, not
// to a mechanism. (T6 only proved the contact solve RESOLVES a deep >0.1 m
// penetration toward zero over a few steps; it never validated long-horizon
// floating-base PD stance stability, so "T6-validated" did not license 0.03 m here.)
// 0.002 m still fires the lowest-foot contact (depth 0.002 > 0); uniform scalar =>
// D1 determinism preserved.
constexpr float kRestFootPenetration = 0.002f;

// Finite Baumgarte velocity cap (m/s) for the batched normal-row positional bias.
// Matches the T6 production choice: bounds the one-step velocity transient at
// large penetrations without affecting the steady-state seat. Uniform scalar =>
// preserves D1 determinism.
constexpr float kBatchedBaumgarteMaxVelocity = 3.0f;

// Derive the per-foot shapes (base-relative link index + sphere center offset +
// radius) from the cooked scene shape table, REUSING the T2/T6 derivation: every
// Sphere shape whose owning body maps to an articulation link is a foot. The
// resulting FootShape table is in BASE (single-replica) link indices; the
// detection kernel turns calf_local_link into a per-replica global link.
std::vector<runtime::articulation::FootShape> DeriveFeet(const WorldRecord& record) {
    std::vector<runtime::articulation::FootShape> feet;
    const auto& shapes = record.world.template_view.shape_table;
    const auto& host = record.articulation_host;
    const uint32_t link_count = host.TotalLinkCount();
    for (uint32_t shape = 0u; shape < shapes.types.size(); ++shape) {
        if (shapes.types[shape] != scene::ShapeType::Sphere) {
            continue;
        }
        if (shape >= shapes.body_ids.size()) {
            continue;
        }
        const uint32_t body = shapes.body_ids[shape];
        uint32_t calf_link = ~0u;
        for (uint32_t link = 0u; link < link_count; ++link) {
            if (host.link_body[link] == body) {
                calf_link = link;
                break;
            }
        }
        if (calf_link == ~0u) {
            continue;
        }
        runtime::articulation::FootShape foot;
        foot.calf_local_link = calf_link;
        foot.local_offset = shape < shapes.local_transforms.size()
            ? shapes.local_transforms[shape].position
            : math::Vec3::Zero();
        foot.radius = shape < shapes.radii.size() ? shapes.radii[shape] : 0.0f;
        feet.push_back(foot);
    }
    return feet;
}

// Derive the ground height so the rest-pose feet rest on / just penetrate the
// ground. The cooked host link_pose is NOT the assembled world pose -- the
// floating-base root and joint chain are only folded together by forward
// kinematics. We therefore derive the foot world position from the SAME source
// the contact detection uses: UpdateWorldLinkPoses on the rest (q=0) base device
// state produces the per-link world REST pose, the foot world center of foot f is
//   foot_world = world_pose[calf_f].TransformPoint(local_offset_f),
// its lowest point is foot_world.z - radius_f, and we seat the ground at
//   ground = min_f(foot_world.z - radius_f) + kRestFootPenetration
// so the lowest foot starts kRestFootPenetration metres below the surface and
// contacts are active immediately. Returns false if no feet were found.
bool DeriveGroundHeight(WorldRecord& record,
                        const std::vector<runtime::articulation::FootShape>& feet,
                        float* out_ground_height) {
    if (feet.empty() || record.device == nullptr ||
        record.articulation_device.Empty()) {
        return false;
    }
    const auto& host = record.articulation_host;
    const uint32_t link_count = host.TotalLinkCount();

    // FK the rest base state into world poses (same kernel the batched step's
    // contact detection runs on its per-step world poses).
    phi::Buffer world_pose_buf(static_cast<size_t>(link_count) * sizeof(math::Transform),
                               phi::MemoryKind::Device);
    runtime::articulation::UpdateWorldLinkPoses(
        record.device->context,
        record.articulation_device.View(),
        static_cast<math::Transform*>(world_pose_buf.Data()));
    record.device->context.stream.Synchronize();
    std::vector<math::Transform> world_pose(link_count);
    world_pose_buf.CopyToHost(world_pose.data(), world_pose.size() * sizeof(math::Transform));

    float min_foot_bottom = std::numeric_limits<float>::infinity();
    bool any = false;
    for (const auto& foot : feet) {
        if (foot.calf_local_link >= link_count) {
            continue;
        }
        const math::Transform& rest_pose = world_pose[foot.calf_local_link];
        const math::Vec3 foot_world = rest_pose.TransformPoint(foot.local_offset);
        const float bottom = foot_world.z - foot.radius;
        min_foot_bottom = std::fmin(min_foot_bottom, bottom);
        any = true;
    }
    if (!any) {
        return false;
    }
    *out_ground_height = min_foot_bottom + kRestFootPenetration;
    return true;
}

// Tile the base-length hold-drive host vectors across env_count envs (link-major,
// the convention BatchedArticulatedStepParams / ApplyPositionDrives expect) and
// upload to a device buffer. base.size() == base_link_count.
phi::Buffer UploadReplicatedDrive(const std::vector<float>& base, uint32_t env_count) {
    if (base.empty() || env_count == 0u) {
        return phi::Buffer();
    }
    const size_t per = base.size();
    std::vector<float> tiled(per * env_count);
    for (uint32_t env = 0u; env < env_count; ++env) {
        for (size_t i = 0u; i < per; ++i) {
            tiled[env * per + i] = base[i];
        }
    }
    phi::Buffer out(tiled.size() * sizeof(float), phi::MemoryKind::Device);
    out.CopyFromHost(tiled.data(), tiled.size() * sizeof(float));
    return out;
}

nuka_result_t StepBatchedWorld(WorldRecord& record, uint32_t step_count) {
    if (step_count == 0u) {
        return NUKA_RESULT_OK;
    }
    if (record.device == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    if (record.batched == nullptr) {
        return NUKA_RESULT_NOT_SUPPORTED;
    }
    for (uint32_t step = 0u; step < step_count; ++step) {
        record.batched->Step(record.batched_step_params);
    }
    record.device->context.stream.Synchronize();
    record.simulated_step_count += step_count;
    return NUKA_RESULT_OK;
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

    // Lazily allocate the single-env implicit joint-damping scratch (M, M^-1,
    // composite) on the first step, mirroring the lazy articulation_device
    // upload above. Reused for every step. The single-env path has exactly one
    // articulation (articulation index 0); max_dof is its DOF count == the M
    // tile stride. Guarded so a zero-DOF / empty articulation degrades to the
    // joint_damping==nullptr (no-damping) path rather than allocating zero-size.
    if (record.single_env_max_dof == 0u) {
        const uint32_t max_dof =
            runtime::articulation::ArticulationDofCount(record.articulation_host, 0u);
        if (max_dof > 0u) {
            const uint32_t articulation_count =
                record.articulation_host.ArticulationCount();
            const uint32_t total_link_count =
                record.articulation_host.TotalLinkCount();
            const size_t tile =
                static_cast<size_t>(max_dof) * max_dof;
            record.single_env_inertia_m = phi::Buffer(
                static_cast<size_t>(articulation_count) * tile * sizeof(float),
                phi::MemoryKind::Device);
            record.single_env_inertia_m_inv = phi::Buffer(
                static_cast<size_t>(articulation_count) * tile * sizeof(float),
                phi::MemoryKind::Device);
            record.single_env_composite = phi::Buffer(
                static_cast<size_t>(total_link_count) *
                    sizeof(runtime::articulation::LinkSpatialInertia),
                phi::MemoryKind::Device);
            record.single_env_max_dof = max_dof;
        }
    }

    const uint32_t max_dof = record.single_env_max_dof;
    // drive_damping (the deferred PD Kd = 2*sqrt(Kp), per link) is the per-DOF c_j
    // for the general implicit joint damping. nullptr when there is no scratch
    // (zero-DOF articulation) -> the implicit-damping kernels degrade to no-op.
    const float* const drive_damping_device =
        (max_dof > 0u)
            ? static_cast<const float*>(record.drive_damping_device.Data())
            : nullptr;

    for (uint32_t step = 0u; step < step_count; ++step) {
        // (1) Position drives with the -Kd*qdot velocity damping DEFERRED: the
        // drive emits only the Kp*(target-q) stiffness torque. The damping is
        // applied IMPLICITLY in stage (7) below, the same general implicit joint
        // damping the batched contacts path uses (Option B unification).
        runtime::articulation::FeatherstoneAba::ApplyPositionDrives(
            record.device->context,
            record.articulation_device.View(),
            static_cast<const float*>(record.drive_targets_device.Data()),
            static_cast<const float*>(record.drive_stiffness_device.Data()),
            static_cast<const float*>(record.drive_damping_device.Data()),
            static_cast<const float*>(record.drive_force_limits_device.Data()),
            /*defer_velocity_damping=*/true);
        // (2) ABA accelerations -> qddot.
        runtime::articulation::FeatherstoneAba::ComputeAccelerations(
            record.device->context,
            record.articulation_device.View(),
            record.step_options.gravity.z);
        // (3) T8a: floating-base velocity integrate before the joint integrate
        // (uses the same pre-step acceleration). No-op for fixed/kinematic roots,
        // so the fixed-base single-env scene is unaffected.
        runtime::articulation::FeatherstoneAba::IntegrateFloatingBaseVelocity(
            record.device->context,
            record.articulation_device.View(),
            record.step_options.dt,
            record.step_options.gravity.z);
        // (4) Joint velocity integrate qdot += qddot*dt (SPLIT from the combined
        // Integrate, so the implicit damping sits between velocity and position).
        runtime::articulation::FeatherstoneAba::IntegrateVelocity(
            record.device->context,
            record.articulation_device.View(),
            record.step_options.dt);
        if (max_dof > 0u) {
            // (5) CRBA joint-space inertia M with dt*C folded into the joint
            // diagonals so the factored inverse below is (M + dt*C)^-1.
            runtime::articulation::ComputeArticulationInertiaM(
                record.device->context,
                record.articulation_device.View(),
                max_dof,
                static_cast<runtime::articulation::LinkSpatialInertia*>(
                    record.single_env_composite.Data()),
                static_cast<float*>(record.single_env_inertia_m.Data()),
                drive_damping_device,
                record.step_options.dt);
            // (6) Factor M -> (M + dt*C)^-1.
            runtime::articulation::FactorArticulationInertiaM(
                record.device->context,
                record.articulation_device.View(),
                max_dof,
                static_cast<const float*>(record.single_env_inertia_m.Data()),
                static_cast<float*>(record.single_env_inertia_m_inv.Data()));
            // (7) Apply the general implicit joint damping in place:
            // qdot -= dt*(M + dt*C)^-1 * (C*qdot). Same scheme as the batched
            // contacts path, just without any contact rows.
            runtime::articulation::ApplyImplicitJointDamping(
                record.device->context,
                record.articulation_device.View(),
                static_cast<const float*>(record.single_env_inertia_m_inv.Data()),
                drive_damping_device,
                max_dof,
                record.step_options.dt);
        }
        // (8) Joint position integrate q += qdot*dt (SPLIT from Integrate).
        runtime::articulation::FeatherstoneAba::IntegratePosition(
            record.device->context,
            record.articulation_device.View(),
            record.step_options.dt);
        // (9) T8a: floating-base pose integrate after the joint integrate, using
        // the just-updated base spatial velocity. No-op for fixed/kinematic roots.
        runtime::articulation::FeatherstoneAba::IntegrateFloatingBasePose(
            record.device->context,
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
    if (desc->env_count == 0u) {
        return NUKA_RESULT_INVALID_ARG;
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
        record->env_count = desc->env_count;

        // --- Multi-env: build the batched articulated-with-contacts world. ---
        // env_count == 1 keeps the single-env oracle path above byte-for-byte;
        // env_count > 1 additionally constructs the T6 batched world and routes
        // stepping/readback to it (the single-env articulation_device is left
        // built but unused for the batched path -- it is cheap and keeps the
        // common setup uniform).
        if (desc->env_count > 1u) {
            if (record->world.template_view.articulations.empty()) {
                // Only articulated scenes have a batched path.
                return NUKA_RESULT_NOT_SUPPORTED;
            }
            const auto feet = nuka::c_abi::DeriveFeet(*record);
            float ground_height = 0.0f;
            if (!nuka::c_abi::DeriveGroundHeight(*record, feet, &ground_height)) {
                return NUKA_RESULT_NOT_SUPPORTED;
            }
            const uint32_t base_link_count = record->articulation_host.TotalLinkCount();
            const uint32_t max_dof =
                nuka::runtime::articulation::ArticulationDofCount(
                    record->articulation_host, 0u);
            if (max_dof == 0u) {
                return NUKA_RESULT_NOT_SUPPORTED;
            }
            auto batched_host =
                nuka::runtime::articulation::ReplicateArticulationHostState(
                    record->articulation_host, desc->env_count);
            record->batched =
                std::make_unique<nuka::runtime::gpu::BatchedArticulatedWorld>(
                    record->device->context,
                    batched_host,
                    feet,
                    max_dof,
                    ground_height);

            // Replicate the base hold drives across all envs (link-major).
            record->batched_drive_targets_device =
                nuka::c_abi::UploadReplicatedDrive(record->drive_targets_host,
                                                   desc->env_count);
            record->batched_drive_stiffness_device =
                nuka::c_abi::UploadReplicatedDrive(record->drive_stiffness_host,
                                                   desc->env_count);
            record->batched_drive_damping_device =
                nuka::c_abi::UploadReplicatedDrive(record->drive_damping_host,
                                                   desc->env_count);
            record->batched_drive_force_limits_device =
                nuka::c_abi::UploadReplicatedDrive(record->drive_force_limits_host,
                                                   desc->env_count);
            (void)base_link_count;

            nuka::runtime::gpu::BatchedArticulatedStepParams params;
            params.drive_targets = static_cast<const float*>(
                record->batched_drive_targets_device.Data());
            params.drive_stiffness = static_cast<const float*>(
                record->batched_drive_stiffness_device.Data());
            params.drive_damping = static_cast<const float*>(
                record->batched_drive_damping_device.Data());
            params.drive_force_limits = static_cast<const float*>(
                record->batched_drive_force_limits_device.Data());
            params.gravity_z = record->step_options.gravity.z;
            params.dt = record->step_options.dt;
            params.friction_coefficient =
                nuka::runtime::articulation::kContactFriction;
            params.baumgarte_max_velocity = nuka::c_abi::kBatchedBaumgarteMaxVelocity;
            record->batched_step_params = params;

            // Enable contacts in the step-options bookkeeping for the batched
            // regime (the single-env oracle keeps contacts off).
            record->step_options.enable_contacts = true;
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
        // Multi-env batched path (env_count > 1): advance the batched world and
        // skip the single-env host round-trip / RefreshWorldBuffers (readback is
        // served zero-copy from the batched device buffers). The single-env arm
        // is byte-for-byte unchanged.
        if (record->batched != nullptr) {
            return nuka::c_abi::StepBatchedWorld(*record, step_count);
        }
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

nuka_result_t nuka_world_reset(nuka_world_handle world) {
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    // Reset is a batched-path primitive (the RL autoreset path is always
    // batched). The single-env oracle path has no per-env state to reset.
    if (record->batched == nullptr) {
        return NUKA_RESULT_NOT_SUPPORTED;
    }
    try {
        record->batched->Reset();
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_world_reset_envs(nuka_world_handle world,
                                    const uint32_t* env_ids,
                                    uint32_t count) {
    if (count > 0u && env_ids == nullptr) {
        return NUKA_RESULT_INVALID_ARG;
    }
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    if (record->batched == nullptr) {
        return NUKA_RESULT_NOT_SUPPORTED;
    }
    if (count == 0u) {
        return NUKA_RESULT_OK;
    }
    // Bounds-check the ids here so an out-of-range id rejects cleanly (the engine
    // also guards, but reporting INVALID_ARG without touching any env state is
    // the friendlier control-plane contract).
    const uint32_t env_count = record->batched->EnvCount();
    for (uint32_t i = 0u; i < count; ++i) {
        if (env_ids[i] >= env_count) {
            return NUKA_RESULT_INVALID_ARG;
        }
    }
    try {
        record->batched->ResetEnvs(env_ids, count);
        return NUKA_RESULT_OK;
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
