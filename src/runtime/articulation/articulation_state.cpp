// ---------------------------------------------------------------------------
// nuka::runtime::articulation -- CUDA-resident Featherstone state layout
// ---------------------------------------------------------------------------

#include "runtime/articulation/articulation_state.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nuka::runtime::articulation {

namespace {

template <typename T>
phi::Buffer UploadVector(const std::vector<T>& values) {
    phi::Buffer buffer(values.size() * sizeof(T), phi::MemoryKind::Device);
    if (!values.empty()) {
        buffer.CopyFromHost(values.data(), values.size() * sizeof(T));
    }
    return buffer;
}

template <typename T>
void DownloadVector(const phi::Buffer& buffer, std::vector<T>* out) {
    if (out == nullptr) {
        return;
    }
    out->resize(buffer.Size() / sizeof(T));
    if (!out->empty()) {
        buffer.CopyToHost(out->data(), out->size() * sizeof(T));
    }
}

float MassForBody(const scene::CookedBodyTable& bodies,
                  const ArticulationCookedTopology& topology,
                  uint32_t local_link) {
    if (local_link < topology.masses.size() && topology.masses[local_link] > 0.0f) {
        return topology.masses[local_link];
    }
    if (local_link < topology.link_bodies.size()) {
        const scene::BodyId body = topology.link_bodies[local_link];
        if (body < bodies.masses.size()) {
            return bodies.masses[body];
        }
        if (body < bodies.inv_masses.size() && bodies.inv_masses[body] > 0.0f) {
            return 1.0f / bodies.inv_masses[body];
        }
    }
    return 0.0f;
}

math::Vec3 InertiaForBody(const scene::CookedBodyTable& bodies,
                          const ArticulationCookedTopology& topology,
                          uint32_t local_link) {
    if (local_link < topology.inertias.size()) {
        return topology.inertias[local_link];
    }
    if (local_link < topology.link_bodies.size()) {
        const scene::BodyId body = topology.link_bodies[local_link];
        if (body < bodies.inertias.size()) {
            return bodies.inertias[body];
        }
        if (body < bodies.inv_inertias.size()) {
            const math::Vec3 inv = bodies.inv_inertias[body];
            return {
                inv.x > 0.0f ? 1.0f / inv.x : 0.0f,
                inv.y > 0.0f ? 1.0f / inv.y : 0.0f,
                inv.z > 0.0f ? 1.0f / inv.z : 0.0f
            };
        }
    }
    return math::Vec3::Zero();
}

math::Transform PoseForBody(const scene::CookedBodyTable& bodies, scene::BodyId body) {
    if (body < bodies.poses.size()) {
        return bodies.poses[body];
    }
    return math::Transform::Identity();
}

math::Transform LocalPoseForBody(const scene::CookedBodyTable& bodies, scene::BodyId body) {
    if (body < bodies.local_poses.size()) {
        return bodies.local_poses[body];
    }
    return math::Transform::Identity();
}

math::Transform InertialFrameForBody(const scene::CookedBodyTable& bodies,
                                     const ArticulationCookedTopology& topology,
                                     uint32_t local_link) {
    if (local_link < topology.inertial_frames.size()) {
        return topology.inertial_frames[local_link];
    }
    if (local_link < topology.link_bodies.size()) {
        const scene::BodyId body = topology.link_bodies[local_link];
        if (body < bodies.inertial_frames.size()) {
            return bodies.inertial_frames[body];
        }
    }
    return math::Transform::Identity();
}

void SetBlock33(float* matrix,
                uint32_t block_row,
                uint32_t block_col,
                uint32_t row,
                uint32_t col,
                float value) {
    matrix[(block_row + row) * 6u + block_col + col] = value;
}

void AddBlock33(float* matrix,
                uint32_t block_row,
                uint32_t block_col,
                uint32_t row,
                uint32_t col,
                float value) {
    matrix[(block_row + row) * 6u + block_col + col] += value;
}

void RotationMatrixFromQuat(math::Quat q, float* out) {
    q = q.Normalized();
    const float xx = q.x * q.x;
    const float yy = q.y * q.y;
    const float zz = q.z * q.z;
    const float xy = q.x * q.y;
    const float xz = q.x * q.z;
    const float yz = q.y * q.z;
    const float wx = q.w * q.x;
    const float wy = q.w * q.y;
    const float wz = q.w * q.z;

    out[0] = 1.0f - 2.0f * (yy + zz);
    out[1] = 2.0f * (xy - wz);
    out[2] = 2.0f * (xz + wy);
    out[3] = 2.0f * (xy + wz);
    out[4] = 1.0f - 2.0f * (xx + zz);
    out[5] = 2.0f * (yz - wx);
    out[6] = 2.0f * (xz - wy);
    out[7] = 2.0f * (yz + wx);
    out[8] = 1.0f - 2.0f * (xx + yy);
}

void RotateDiagonalInertia(math::Vec3 diagonal, math::Quat rotation, float* out) {
    float r[9];
    RotationMatrixFromQuat(rotation, r);
    const float d[3] = {diagonal.x, diagonal.y, diagonal.z};
    for (uint32_t row = 0u; row < 3u; ++row) {
        for (uint32_t col = 0u; col < 3u; ++col) {
            float value = 0.0f;
            for (uint32_t k = 0u; k < 3u; ++k) {
                value += r[row * 3u + k] * d[k] * r[col * 3u + k];
            }
            out[row * 3u + col] = value;
        }
    }
}

void CrossMatrix(math::Vec3 c, float* out) {
    out[0] = 0.0f;
    out[1] = -c.z;
    out[2] = c.y;
    out[3] = c.z;
    out[4] = 0.0f;
    out[5] = -c.x;
    out[6] = -c.y;
    out[7] = c.x;
    out[8] = 0.0f;
}

} // namespace

ArticulationDeviceState ArticulationDeviceBuffers::View() {
    ArticulationDeviceState state;
    state.link_inertia = static_cast<LinkSpatialInertia*>(link_inertia.Data());
    state.link_velocity = static_cast<LinkSpatialVel*>(link_velocity.Data());
    state.link_acceleration = static_cast<LinkSpatialAccel*>(link_acceleration.Data());
    state.link_xup = static_cast<LinkSpatialTransform*>(link_xup.Data());
    state.link_velocity_bias = static_cast<LinkSpatialVel*>(link_velocity_bias.Data());
    state.link_articulated_I =
        static_cast<LinkArticulatedInertia*>(link_articulated_inertia.Data());
    state.link_bias_force = static_cast<LinkBiasForce*>(link_bias_force.Data());
    state.link_u_spatial = static_cast<LinkBiasForce*>(link_u_spatial.Data());
    state.joint_motion_subspace =
        static_cast<LinkMotionSubspace*>(joint_motion_subspace.Data());
    state.link_pose = static_cast<math::Transform*>(link_pose.Data());
    state.link_local_pose = static_cast<math::Transform*>(link_local_pose.Data());
    state.link_inertial_frame =
        static_cast<math::Transform*>(link_inertial_frame.Data());
    state.q = static_cast<float*>(q.Data());
    state.qdot = static_cast<float*>(qdot.Data());
    state.qddot = static_cast<float*>(qddot.Data());
    state.tau = static_cast<float*>(tau.Data());
    state.joint_damping = static_cast<float*>(joint_damping.Data());
    state.joint_armature = static_cast<float*>(joint_armature.Data());
    state.joint_diagonal = static_cast<float*>(joint_diagonal.Data());
    state.joint_force = static_cast<float*>(joint_force.Data());
    state.joint_axis = static_cast<math::Vec3*>(joint_axis.Data());
    state.parent_offset = static_cast<math::Vec3*>(parent_offset.Data());
    state.joint_type = static_cast<ArticulationJointType*>(joint_type.Data());
    state.parent_link = static_cast<uint32_t*>(parent_link.Data());
    state.link_body = static_cast<uint32_t*>(link_body.Data());
    state.link_to_articulation = static_cast<uint32_t*>(link_to_articulation.Data());
    state.articulation_link_count =
        static_cast<uint32_t*>(articulation_link_count.Data());
    state.articulation_link_offset =
        static_cast<uint32_t*>(articulation_link_offset.Data());
    state.total_link_count = total_link_count;
    state.articulation_count = articulation_count;
    return state;
}

ArticulationDeviceState ArticulationDeviceBuffers::View() const {
    return const_cast<ArticulationDeviceBuffers*>(this)->View();
}

bool ArticulationDeviceBuffers::Empty() const {
    return total_link_count == 0u || articulation_count == 0u;
}

LinkSpatialInertia MakeDiagonalSpatialInertia(float mass, math::Vec3 diagonal_inertia) {
    return MakeSpatialInertia(mass, diagonal_inertia, math::Transform::Identity());
}

LinkSpatialInertia MakeSpatialInertia(float mass,
                                      math::Vec3 diagonal_inertia,
                                      const math::Transform& inertial_frame) {
    LinkSpatialInertia inertia;
    if (mass <= 0.0f) {
        return inertia;
    }

    float com_inertia[9];
    RotateDiagonalInertia(diagonal_inertia, inertial_frame.rotation, com_inertia);

    const math::Vec3 c = inertial_frame.position;
    const float c_dot = c.Dot(c);
    const float parallel_axis[9] = {
        c_dot - c.x * c.x, -c.x * c.y, -c.x * c.z,
        -c.y * c.x, c_dot - c.y * c.y, -c.y * c.z,
        -c.z * c.x, -c.z * c.y, c_dot - c.z * c.z
    };
    for (uint32_t row = 0u; row < 3u; ++row) {
        for (uint32_t col = 0u; col < 3u; ++col) {
            SetBlock33(inertia.I, 0u, 0u, row, col,
                       com_inertia[row * 3u + col] +
                           mass * parallel_axis[row * 3u + col]);
        }
    }

    float c_cross[9];
    CrossMatrix(c, c_cross);
    for (uint32_t row = 0u; row < 3u; ++row) {
        for (uint32_t col = 0u; col < 3u; ++col) {
            AddBlock33(inertia.I, 0u, 3u, row, col, mass * c_cross[row * 3u + col]);
            AddBlock33(inertia.I, 3u, 0u, row, col, -mass * c_cross[row * 3u + col]);
            SetBlock33(inertia.I, 3u, 3u, row, col, row == col ? mass : 0.0f);
        }
    }
    return inertia;
}

ArticulationHostState BuildArticulationHostState(
    const std::vector<ArticulationCookedTopology>& topologies,
    const scene::CookedBodyTable& bodies) {
    ArticulationHostState result;
    result.articulations = topologies;
    result.articulation_link_count.reserve(topologies.size());
    result.articulation_link_offset.reserve(topologies.size());

    uint32_t global_offset = 0u;
    for (uint32_t articulation_index = 0u;
         articulation_index < static_cast<uint32_t>(topologies.size());
         ++articulation_index) {
        const auto& topology = topologies[articulation_index];
        const uint32_t link_count = static_cast<uint32_t>(topology.link_bodies.size());
        if (link_count == 0u) {
            continue;
        }
        if (topology.parent_links.size() != link_count ||
            topology.joint_types.size() != link_count ||
            topology.joint_axes.size() != link_count ||
            topology.joint_dampings.size() != link_count ||
            topology.joint_armatures.size() != link_count) {
            throw std::runtime_error("articulation topology arrays must match link count");
        }

        result.articulation_link_offset.push_back(global_offset);
        result.articulation_link_count.push_back(link_count);

        for (uint32_t local_link = 0u; local_link < link_count; ++local_link) {
            const float mass = MassForBody(bodies, topology, local_link);
            const math::Vec3 inertia = InertiaForBody(bodies, topology, local_link);
            const math::Transform inertial_frame =
                InertialFrameForBody(bodies, topology, local_link);
            result.link_inertia.push_back(
                MakeSpatialInertia(mass, inertia, inertial_frame));
            result.link_velocity.emplace_back();
            result.link_acceleration.emplace_back();
            result.link_xup.emplace_back();
            result.link_velocity_bias.emplace_back();
            result.link_articulated_inertia.emplace_back();
            result.link_bias_force.emplace_back();
            result.link_u_spatial.emplace_back();
            result.joint_motion_subspace.emplace_back();
            const scene::BodyId body = topology.link_bodies[local_link];
            const math::Transform parent_frame =
                local_link < topology.parent_frames.size()
                    ? topology.parent_frames[local_link]
                    : math::Transform::Identity();
            result.link_pose.push_back(PoseForBody(bodies, body));
            result.link_local_pose.push_back(LocalPoseForBody(bodies, body));
            result.link_inertial_frame.push_back(inertial_frame);
            result.q.push_back(local_link < topology.initial_positions.size()
                ? topology.initial_positions[local_link]
                : 0.0f);
            result.qdot.push_back(0.0f);
            result.qddot.push_back(0.0f);
            result.tau.push_back(0.0f);
            result.joint_damping.push_back(topology.joint_dampings[local_link]);
            result.joint_armature.push_back(topology.joint_armatures[local_link]);
            result.joint_diagonal.push_back(0.0f);
            result.joint_force.push_back(0.0f);
            result.joint_axis.push_back(topology.joint_axes[local_link]);
            result.parent_offset.push_back(parent_frame.position);
            result.joint_type.push_back(topology.joint_types[local_link]);
            result.parent_link.push_back(topology.parent_links[local_link]);
            result.link_body.push_back(topology.link_bodies[local_link]);
            result.link_to_articulation.push_back(articulation_index);
        }
        global_offset += link_count;
    }

    return result;
}

namespace {

// Appends `env_count` copies of `source` to `target`. Pure value tiling: used
// for every array whose entries carry no cross-replica index (inertia, poses,
// q/qdot, joint metadata, parent_link, link_body, ...).
template <typename T>
void TileConcat(const std::vector<T>& source, uint32_t env_count, std::vector<T>* target) {
    target->reserve(source.size() * env_count);
    for (uint32_t env = 0u; env < env_count; ++env) {
        target->insert(target->end(), source.begin(), source.end());
    }
}

// Appends `env_count` copies of `source`, adding `env * stride` to each entry of
// replica `env`. Used for the only two arrays that hold replica-relative indices:
// link_to_articulation (stride = articulation count) and articulation_link_offset
// (stride = total link count).
void TileConcatOffset(const std::vector<uint32_t>& source,
                      uint32_t env_count,
                      uint32_t stride,
                      std::vector<uint32_t>* target) {
    target->reserve(source.size() * env_count);
    for (uint32_t env = 0u; env < env_count; ++env) {
        const uint32_t delta = env * stride;
        for (const uint32_t value : source) {
            target->push_back(value + delta);
        }
    }
}

} // namespace

ArticulationHostState ReplicateArticulationHostState(const ArticulationHostState& base,
                                                     uint32_t env_count) {
    if (env_count == 0u) {
        return ArticulationHostState{};
    }
    if (env_count == 1u) {
        return base;
    }

    const uint32_t base_link_count = base.TotalLinkCount();
    const uint32_t base_articulation_count = base.ArticulationCount();

    ArticulationHostState result;

    // Topology metadata is per-articulation local data (root_body, link_bodies,
    // parent_links are all articulation-relative). The kernels never read it and
    // UploadArticulationState does not upload it, so plain replication is correct.
    TileConcat(base.articulations, env_count, &result.articulations);

    // --- Per-link / per-DOF state arrays: plain value tiling --------------------
    // None of these carry a cross-replica index, so concatenation in replica order
    // is correct. parent_link is articulation-LOCAL: the ABA kernels compute the
    // global parent as `articulation_link_offset[artic] + parent_link[link]`
    // (featherstone_aba.cu:388-390, 452-465, 505-509), so the stored values stay
    // unchanged and the ~0u root sentinel is preserved automatically. link_body
    // indexes the (un-replicated) CookedBodyTable and is read only at build time,
    // never by the kernels, so it is also copied as-is.
    TileConcat(base.link_inertia, env_count, &result.link_inertia);
    TileConcat(base.link_velocity, env_count, &result.link_velocity);
    TileConcat(base.link_acceleration, env_count, &result.link_acceleration);
    TileConcat(base.link_xup, env_count, &result.link_xup);
    TileConcat(base.link_velocity_bias, env_count, &result.link_velocity_bias);
    TileConcat(base.link_articulated_inertia, env_count, &result.link_articulated_inertia);
    TileConcat(base.link_bias_force, env_count, &result.link_bias_force);
    TileConcat(base.link_u_spatial, env_count, &result.link_u_spatial);
    TileConcat(base.joint_motion_subspace, env_count, &result.joint_motion_subspace);
    TileConcat(base.link_pose, env_count, &result.link_pose);
    TileConcat(base.link_local_pose, env_count, &result.link_local_pose);
    TileConcat(base.link_inertial_frame, env_count, &result.link_inertial_frame);
    TileConcat(base.q, env_count, &result.q);
    TileConcat(base.qdot, env_count, &result.qdot);
    TileConcat(base.qddot, env_count, &result.qddot);
    TileConcat(base.tau, env_count, &result.tau);
    TileConcat(base.joint_damping, env_count, &result.joint_damping);
    TileConcat(base.joint_armature, env_count, &result.joint_armature);
    TileConcat(base.joint_diagonal, env_count, &result.joint_diagonal);
    TileConcat(base.joint_force, env_count, &result.joint_force);
    TileConcat(base.joint_axis, env_count, &result.joint_axis);
    TileConcat(base.parent_offset, env_count, &result.parent_offset);
    TileConcat(base.joint_type, env_count, &result.joint_type);
    TileConcat(base.parent_link, env_count, &result.parent_link);
    TileConcat(base.link_body, env_count, &result.link_body);

    // --- Index arrays: tile with a per-replica offset ---------------------------
    // link_to_articulation must point at the replica's own articulations.
    TileConcatOffset(base.link_to_articulation, env_count, base_articulation_count,
                     &result.link_to_articulation);
    // articulation_link_offset is the global start of each articulation's link
    // block; replica e's links begin at e * base_link_count.
    TileConcatOffset(base.articulation_link_offset, env_count, base_link_count,
                     &result.articulation_link_offset);
    // articulation_link_count is a count (no index), so plain tiling.
    TileConcat(base.articulation_link_count, env_count, &result.articulation_link_count);

    return result;
}

ArticulationDeviceBuffers UploadArticulationState(const phi::DeviceContext& context,
                                                  const ArticulationHostState& host_state) {
    phi::ScopedDeviceGuard guard(context.device_id);
    ArticulationDeviceBuffers result;
    result.total_link_count = host_state.TotalLinkCount();
    result.articulation_count = host_state.ArticulationCount();
    result.link_inertia = UploadVector(host_state.link_inertia);
    result.link_velocity = UploadVector(host_state.link_velocity);
    result.link_acceleration = UploadVector(host_state.link_acceleration);
    result.link_xup = UploadVector(host_state.link_xup);
    result.link_velocity_bias = UploadVector(host_state.link_velocity_bias);
    result.link_articulated_inertia = UploadVector(host_state.link_articulated_inertia);
    result.link_bias_force = UploadVector(host_state.link_bias_force);
    result.link_u_spatial = UploadVector(host_state.link_u_spatial);
    result.joint_motion_subspace = UploadVector(host_state.joint_motion_subspace);
    result.link_pose = UploadVector(host_state.link_pose);
    result.link_local_pose = UploadVector(host_state.link_local_pose);
    result.link_inertial_frame = UploadVector(host_state.link_inertial_frame);
    result.q = UploadVector(host_state.q);
    result.qdot = UploadVector(host_state.qdot);
    result.qddot = UploadVector(host_state.qddot);
    result.tau = UploadVector(host_state.tau);
    result.joint_damping = UploadVector(host_state.joint_damping);
    result.joint_armature = UploadVector(host_state.joint_armature);
    result.joint_diagonal = UploadVector(host_state.joint_diagonal);
    result.joint_force = UploadVector(host_state.joint_force);
    result.joint_axis = UploadVector(host_state.joint_axis);
    result.parent_offset = UploadVector(host_state.parent_offset);
    result.joint_type = UploadVector(host_state.joint_type);
    result.parent_link = UploadVector(host_state.parent_link);
    result.link_body = UploadVector(host_state.link_body);
    result.link_to_articulation = UploadVector(host_state.link_to_articulation);
    result.articulation_link_count = UploadVector(host_state.articulation_link_count);
    result.articulation_link_offset = UploadVector(host_state.articulation_link_offset);
    return result;
}

ArticulationDeviceBuffers UploadArticulationState(const ArticulationHostState& host_state) {
    auto context = phi::MakeDefaultDeviceContext();
    return UploadArticulationState(context, host_state);
}

void DownloadArticulationState(const ArticulationDeviceBuffers& device_state,
                               ArticulationHostState* host_state) {
    if (host_state == nullptr) {
        return;
    }
    DownloadVector(device_state.link_inertia, &host_state->link_inertia);
    DownloadVector(device_state.link_velocity, &host_state->link_velocity);
    DownloadVector(device_state.link_acceleration, &host_state->link_acceleration);
    DownloadVector(device_state.link_xup, &host_state->link_xup);
    DownloadVector(device_state.link_velocity_bias, &host_state->link_velocity_bias);
    DownloadVector(device_state.link_articulated_inertia,
                   &host_state->link_articulated_inertia);
    DownloadVector(device_state.link_bias_force, &host_state->link_bias_force);
    DownloadVector(device_state.link_u_spatial, &host_state->link_u_spatial);
    DownloadVector(device_state.joint_motion_subspace,
                   &host_state->joint_motion_subspace);
    DownloadVector(device_state.link_pose, &host_state->link_pose);
    DownloadVector(device_state.link_local_pose, &host_state->link_local_pose);
    DownloadVector(device_state.link_inertial_frame, &host_state->link_inertial_frame);
    DownloadVector(device_state.q, &host_state->q);
    DownloadVector(device_state.qdot, &host_state->qdot);
    DownloadVector(device_state.qddot, &host_state->qddot);
    DownloadVector(device_state.tau, &host_state->tau);
    DownloadVector(device_state.joint_damping, &host_state->joint_damping);
    DownloadVector(device_state.joint_armature, &host_state->joint_armature);
    DownloadVector(device_state.joint_diagonal, &host_state->joint_diagonal);
    DownloadVector(device_state.joint_force, &host_state->joint_force);
    DownloadVector(device_state.joint_axis, &host_state->joint_axis);
    DownloadVector(device_state.parent_offset, &host_state->parent_offset);
    DownloadVector(device_state.joint_type, &host_state->joint_type);
    DownloadVector(device_state.parent_link, &host_state->parent_link);
    DownloadVector(device_state.link_body, &host_state->link_body);
    DownloadVector(device_state.link_to_articulation, &host_state->link_to_articulation);
    DownloadVector(device_state.articulation_link_count,
                   &host_state->articulation_link_count);
    DownloadVector(device_state.articulation_link_offset,
                   &host_state->articulation_link_offset);
}

} // namespace nuka::runtime::articulation
