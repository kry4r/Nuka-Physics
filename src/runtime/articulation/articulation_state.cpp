// ---------------------------------------------------------------------------
// nuka::runtime::articulation -- CUDA-resident Featherstone state layout
// ---------------------------------------------------------------------------

#include "runtime/articulation/articulation_state.hpp"

#include "phi/backend.hpp"             // DeviceBufferType, InitBestDevice
#include "phi/buffer_transfer_v2.hpp"  // UploadVectorV2 / DownloadVectorV2

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nuka::runtime::articulation {

namespace {

// phi v2 transfer helpers (UploadVectorV2 / out-param DownloadVectorV2). The v2
// forms take an explicit element `count` (the opaque v2 Buffer has no Size());
// the caller derives every count from the device-buffer dimensions
// (total_link_count / articulation_count), which mirror the host_state that was
// uploaded. Byte math is unchanged (size*sizeof(T)).
using ::nuka::phi::DownloadVectorV2;
using ::nuka::phi::UploadVectorV2;

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

namespace {
// Null-safe v2 base-pointer accessor: BufferBase dereferences the handle's
// vtable, so a null member (e.g. an empty/zero-byte upload) must short-circuit to
// nullptr -- exactly what the legacy Buffer::Data() returned for an empty buffer.
void* BaseOrNull(phi::Buffer* b) {
    return b != nullptr ? phi::BufferBase(b) : nullptr;
}
}  // namespace

ArticulationDeviceState ArticulationDeviceBuffers::View() {
    ArticulationDeviceState state;
    state.link_inertia = static_cast<LinkSpatialInertia*>(BaseOrNull(link_inertia));
    state.link_velocity = static_cast<LinkSpatialVel*>(BaseOrNull(link_velocity));
    state.link_acceleration = static_cast<LinkSpatialAccel*>(BaseOrNull(link_acceleration));
    state.link_xup = static_cast<LinkSpatialTransform*>(BaseOrNull(link_xup));
    state.link_velocity_bias = static_cast<LinkSpatialVel*>(BaseOrNull(link_velocity_bias));
    state.link_articulated_I =
        static_cast<LinkArticulatedInertia*>(BaseOrNull(link_articulated_inertia));
    state.link_bias_force = static_cast<LinkBiasForce*>(BaseOrNull(link_bias_force));
    state.link_u_spatial = static_cast<LinkBiasForce*>(BaseOrNull(link_u_spatial));
    state.joint_motion_subspace =
        static_cast<LinkMotionSubspace*>(BaseOrNull(joint_motion_subspace));
    state.link_pose = static_cast<math::Transform*>(BaseOrNull(link_pose));
    state.link_local_pose = static_cast<math::Transform*>(BaseOrNull(link_local_pose));
    state.link_inertial_frame =
        static_cast<math::Transform*>(BaseOrNull(link_inertial_frame));
    state.base_pose = static_cast<math::Transform*>(BaseOrNull(base_pose));
    state.q = static_cast<float*>(BaseOrNull(q));
    state.qdot = static_cast<float*>(BaseOrNull(qdot));
    state.qddot = static_cast<float*>(BaseOrNull(qddot));
    state.tau = static_cast<float*>(BaseOrNull(tau));
    state.joint_damping = static_cast<float*>(BaseOrNull(joint_damping));
    state.joint_armature = static_cast<float*>(BaseOrNull(joint_armature));
    state.joint_diagonal = static_cast<float*>(BaseOrNull(joint_diagonal));
    state.joint_force = static_cast<float*>(BaseOrNull(joint_force));
    state.joint_axis = static_cast<math::Vec3*>(BaseOrNull(joint_axis));
    state.parent_offset = static_cast<math::Vec3*>(BaseOrNull(parent_offset));
    state.joint_type = static_cast<ArticulationJointType*>(BaseOrNull(joint_type));
    state.parent_link = static_cast<uint32_t*>(BaseOrNull(parent_link));
    state.link_body = static_cast<uint32_t*>(BaseOrNull(link_body));
    state.link_to_articulation = static_cast<uint32_t*>(BaseOrNull(link_to_articulation));
    state.articulation_link_count =
        static_cast<uint32_t*>(BaseOrNull(articulation_link_count));
    state.articulation_link_offset =
        static_cast<uint32_t*>(BaseOrNull(articulation_link_offset));
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

namespace {
// Steal+null every member buffer from `src` into `dst` (move semantics) and free
// `dst`'s prior holdings first. Declared here so the move ctor/assign + dtor
// share one member list.
void BufferMemberPtrs(ArticulationDeviceBuffers* b, phi::Buffer** out[]) {
    out[0] = &b->link_inertia;
    out[1] = &b->link_velocity;
    out[2] = &b->link_acceleration;
    out[3] = &b->link_xup;
    out[4] = &b->link_velocity_bias;
    out[5] = &b->link_articulated_inertia;
    out[6] = &b->link_bias_force;
    out[7] = &b->link_u_spatial;
    out[8] = &b->joint_motion_subspace;
    out[9] = &b->link_pose;
    out[10] = &b->link_local_pose;
    out[11] = &b->link_inertial_frame;
    out[12] = &b->base_pose;
    out[13] = &b->q;
    out[14] = &b->qdot;
    out[15] = &b->qddot;
    out[16] = &b->tau;
    out[17] = &b->joint_damping;
    out[18] = &b->joint_armature;
    out[19] = &b->joint_diagonal;
    out[20] = &b->joint_force;
    out[21] = &b->joint_axis;
    out[22] = &b->parent_offset;
    out[23] = &b->joint_type;
    out[24] = &b->parent_link;
    out[25] = &b->link_body;
    out[26] = &b->link_to_articulation;
    out[27] = &b->articulation_link_count;
    out[28] = &b->articulation_link_offset;
}
constexpr int kArticulationBufferCount = 29;
}  // namespace

ArticulationDeviceBuffers::~ArticulationDeviceBuffers() {
    phi::Buffer** members[kArticulationBufferCount];
    BufferMemberPtrs(this, members);
    for (int i = 0; i < kArticulationBufferCount; ++i) {
        if (*members[i] != nullptr) {
            phi::BufferFree(*members[i]);
            *members[i] = nullptr;
        }
    }
}

ArticulationDeviceBuffers::ArticulationDeviceBuffers(
    ArticulationDeviceBuffers&& other) noexcept {
    phi::Buffer** dst[kArticulationBufferCount];
    phi::Buffer** src[kArticulationBufferCount];
    BufferMemberPtrs(this, dst);
    BufferMemberPtrs(&other, src);
    for (int i = 0; i < kArticulationBufferCount; ++i) {
        *dst[i] = *src[i];
        *src[i] = nullptr;
    }
    total_link_count = other.total_link_count;
    articulation_count = other.articulation_count;
    other.total_link_count = 0u;
    other.articulation_count = 0u;
}

ArticulationDeviceBuffers& ArticulationDeviceBuffers::operator=(
    ArticulationDeviceBuffers&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    phi::Buffer** dst[kArticulationBufferCount];
    phi::Buffer** src[kArticulationBufferCount];
    BufferMemberPtrs(this, dst);
    BufferMemberPtrs(&other, src);
    for (int i = 0; i < kArticulationBufferCount; ++i) {
        if (*dst[i] != nullptr) {
            phi::BufferFree(*dst[i]);
        }
        *dst[i] = *src[i];
        *src[i] = nullptr;
    }
    total_link_count = other.total_link_count;
    articulation_count = other.articulation_count;
    other.total_link_count = 0u;
    other.articulation_count = 0u;
    return *this;
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

        // T8a: seed the live base pose from the root link's cook pose. The root
        // is local_link 0 (the cooker emits the root first). For a fixed root
        // this value is never read; for a floating root the first FK before any
        // integration step must match the static cook seed, so initialize it to
        // the same pose the FK would otherwise read from link_pose[root].
        {
            const scene::BodyId root_body = topology.link_bodies.empty()
                ? scene::kInvalidBody
                : topology.link_bodies[0];
            result.base_pose.push_back(root_body == scene::kInvalidBody
                ? math::Transform::Identity()
                : PoseForBody(bodies, root_body));
        }

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
    // base_pose is a per-articulation pose carrying no cross-replica index (the
    // root's world pose is identical for every replica), so plain value tiling.
    TileConcat(base.base_pose, env_count, &result.base_pose);

    return result;
}

ArticulationDeviceBuffers UploadArticulationState(cudaStream_t stream, int device_id,
                                                  const ArticulationHostState& host_state) {
    (void)stream;  // v2 transfers run on the device-level DEFAULT (stream 0).
    phi::ScopedDeviceGuard guard(device_id);
    // The device-level DEFAULT (stream-0) BufferType: v2 upload transfers run on
    // stream 0 exactly like the legacy synchronous CopyFromHost cudaMemcpy. The
    // returned Buffer*'s pointers feed kernels launched on the caller's stream
    // (raw cudaMemcpyAsync / launches, unchanged) -- the test harness coordinates
    // upload vs kernel via NULL-stream implicit sync, the legacy oracle path.
    phi::BufferType* bt = phi::DeviceBufferType(phi::InitBestDevice());
    ArticulationDeviceBuffers result;
    result.total_link_count = host_state.TotalLinkCount();
    result.articulation_count = host_state.ArticulationCount();
    result.link_inertia = UploadVectorV2(bt, host_state.link_inertia);
    result.link_velocity = UploadVectorV2(bt, host_state.link_velocity);
    result.link_acceleration = UploadVectorV2(bt, host_state.link_acceleration);
    result.link_xup = UploadVectorV2(bt, host_state.link_xup);
    result.link_velocity_bias = UploadVectorV2(bt, host_state.link_velocity_bias);
    result.link_articulated_inertia = UploadVectorV2(bt, host_state.link_articulated_inertia);
    result.link_bias_force = UploadVectorV2(bt, host_state.link_bias_force);
    result.link_u_spatial = UploadVectorV2(bt, host_state.link_u_spatial);
    result.joint_motion_subspace = UploadVectorV2(bt, host_state.joint_motion_subspace);
    result.link_pose = UploadVectorV2(bt, host_state.link_pose);
    result.link_local_pose = UploadVectorV2(bt, host_state.link_local_pose);
    result.link_inertial_frame = UploadVectorV2(bt, host_state.link_inertial_frame);
    result.base_pose = UploadVectorV2(bt, host_state.base_pose);
    result.q = UploadVectorV2(bt, host_state.q);
    result.qdot = UploadVectorV2(bt, host_state.qdot);
    result.qddot = UploadVectorV2(bt, host_state.qddot);
    result.tau = UploadVectorV2(bt, host_state.tau);
    result.joint_damping = UploadVectorV2(bt, host_state.joint_damping);
    result.joint_armature = UploadVectorV2(bt, host_state.joint_armature);
    result.joint_diagonal = UploadVectorV2(bt, host_state.joint_diagonal);
    result.joint_force = UploadVectorV2(bt, host_state.joint_force);
    result.joint_axis = UploadVectorV2(bt, host_state.joint_axis);
    result.parent_offset = UploadVectorV2(bt, host_state.parent_offset);
    result.joint_type = UploadVectorV2(bt, host_state.joint_type);
    result.parent_link = UploadVectorV2(bt, host_state.parent_link);
    result.link_body = UploadVectorV2(bt, host_state.link_body);
    result.link_to_articulation = UploadVectorV2(bt, host_state.link_to_articulation);
    result.articulation_link_count = UploadVectorV2(bt, host_state.articulation_link_count);
    result.articulation_link_offset = UploadVectorV2(bt, host_state.articulation_link_offset);
    return result;
}

ArticulationDeviceBuffers UploadArticulationState(const ArticulationHostState& host_state) {
    return UploadArticulationState(/*stream=*/nullptr, /*device_id=*/0, host_state);
}

ArticulationDeviceState MakeArticulationDeviceStateFromViews(
    const phi::ModelView& model_view, const phi::DataView& data_view,
    uint32_t total_link_count, uint32_t articulation_count) {
    ArticulationDeviceState s;
    s.link_inertia = reinterpret_cast<LinkSpatialInertia*>(model_view.link_inertia);
    s.link_velocity = reinterpret_cast<LinkSpatialVel*>(data_view.link_velocity);
    s.link_acceleration =
        reinterpret_cast<LinkSpatialAccel*>(data_view.link_acceleration);
    s.link_xup = reinterpret_cast<LinkSpatialTransform*>(data_view.link_xup);
    s.link_velocity_bias =
        reinterpret_cast<LinkSpatialVel*>(data_view.link_velocity_bias);
    s.link_articulated_I =
        reinterpret_cast<LinkArticulatedInertia*>(data_view.link_articulated_I);
    s.link_bias_force = reinterpret_cast<LinkBiasForce*>(data_view.link_bias_force);
    s.link_u_spatial = reinterpret_cast<LinkBiasForce*>(data_view.link_u_spatial);
    s.joint_motion_subspace =
        reinterpret_cast<LinkMotionSubspace*>(data_view.joint_motion_subspace);
    s.link_pose = data_view.link_pose;
    s.link_local_pose = model_view.link_local_pose;
    s.link_inertial_frame = model_view.link_inertial_frame;
    s.base_pose = data_view.base_pose;
    s.q = data_view.q;
    s.qdot = data_view.qdot;
    s.qddot = data_view.qddot;
    s.tau = data_view.tau;
    s.joint_damping = model_view.joint_damping;
    s.joint_armature = model_view.joint_armature;
    s.joint_diagonal = data_view.joint_diagonal;
    s.joint_force = data_view.joint_force;
    s.joint_axis = model_view.joint_axis;
    s.parent_offset = model_view.parent_offset;
    s.joint_type = reinterpret_cast<ArticulationJointType*>(model_view.joint_type);
    s.parent_link = model_view.parent_link;
    s.link_body = model_view.link_body;
    s.link_to_articulation = model_view.link_to_articulation;
    s.articulation_link_count = model_view.articulation_link_count;
    s.articulation_link_offset = model_view.articulation_link_offset;
    s.total_link_count = total_link_count;
    s.articulation_count = articulation_count;
    return s;
}

void DownloadArticulationState(const ArticulationDeviceBuffers& device_state,
                               ArticulationHostState* host_state) {
    if (host_state == nullptr) {
        return;
    }
    // The opaque v2 Buffer has no Size(), so element counts come from the device
    // buffers' dimensions: per-link arrays are total_link_count, per-articulation
    // arrays (base_pose / articulation_link_*) are articulation_count -- exactly
    // the sizes UploadArticulationState wrote (mirrors the host_state layout
    // BuildArticulationHostState emits). DownloadVectorV2 self-syncs (stream 0).
    const std::size_t links = device_state.total_link_count;
    const std::size_t artics = device_state.articulation_count;
    DownloadVectorV2(device_state.link_inertia, &host_state->link_inertia, links);
    DownloadVectorV2(device_state.link_velocity, &host_state->link_velocity, links);
    DownloadVectorV2(device_state.link_acceleration, &host_state->link_acceleration, links);
    DownloadVectorV2(device_state.link_xup, &host_state->link_xup, links);
    DownloadVectorV2(device_state.link_velocity_bias, &host_state->link_velocity_bias, links);
    DownloadVectorV2(device_state.link_articulated_inertia,
                     &host_state->link_articulated_inertia, links);
    DownloadVectorV2(device_state.link_bias_force, &host_state->link_bias_force, links);
    DownloadVectorV2(device_state.link_u_spatial, &host_state->link_u_spatial, links);
    DownloadVectorV2(device_state.joint_motion_subspace,
                     &host_state->joint_motion_subspace, links);
    DownloadVectorV2(device_state.link_pose, &host_state->link_pose, links);
    DownloadVectorV2(device_state.link_local_pose, &host_state->link_local_pose, links);
    DownloadVectorV2(device_state.link_inertial_frame, &host_state->link_inertial_frame, links);
    DownloadVectorV2(device_state.base_pose, &host_state->base_pose, artics);
    DownloadVectorV2(device_state.q, &host_state->q, links);
    DownloadVectorV2(device_state.qdot, &host_state->qdot, links);
    DownloadVectorV2(device_state.qddot, &host_state->qddot, links);
    DownloadVectorV2(device_state.tau, &host_state->tau, links);
    DownloadVectorV2(device_state.joint_damping, &host_state->joint_damping, links);
    DownloadVectorV2(device_state.joint_armature, &host_state->joint_armature, links);
    DownloadVectorV2(device_state.joint_diagonal, &host_state->joint_diagonal, links);
    DownloadVectorV2(device_state.joint_force, &host_state->joint_force, links);
    DownloadVectorV2(device_state.joint_axis, &host_state->joint_axis, links);
    DownloadVectorV2(device_state.parent_offset, &host_state->parent_offset, links);
    DownloadVectorV2(device_state.joint_type, &host_state->joint_type, links);
    DownloadVectorV2(device_state.parent_link, &host_state->parent_link, links);
    DownloadVectorV2(device_state.link_body, &host_state->link_body, links);
    DownloadVectorV2(device_state.link_to_articulation, &host_state->link_to_articulation, links);
    DownloadVectorV2(device_state.articulation_link_count,
                     &host_state->articulation_link_count, artics);
    DownloadVectorV2(device_state.articulation_link_offset,
                     &host_state->articulation_link_offset, artics);
}

} // namespace nuka::runtime::articulation
