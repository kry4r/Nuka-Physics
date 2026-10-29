#pragma once

#include <cstdint>

#include "math/cuda_vec_ops.cuh"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/views.hpp"
#include "phi/op_schema.hpp"
#include "phi/articulation_contract.hpp"

namespace nuka::phi::nkops {

using phi::ArticulationJointType;

struct LinkSpatialInertia {
    float I[36] = {};
};

struct LinkSpatialVel {
    float v[6] = {};
};

struct LinkSpatialAccel {
    float a[6] = {};
};

struct LinkSpatialTransform {
    float X[36] = {};
};

struct LinkBiasForce {
    float p[6] = {};
};

struct LinkMotionSubspace {
    float s[6] = {};
};

struct ArticulationTreeView {
    const uint32_t* depth;
    const uint32_t* max_depth;
    const uint32_t* child_range;
    const uint32_t* children;
};

inline ArticulationTreeView MakeArticulationTreeView(const ModelView& model) {
    return {model.link_tree_depth, model.articulation_tree_depth,
            model.link_child_range, model.link_children};
}

struct ArticulationDeviceState {
    LinkSpatialInertia* link_inertia = nullptr;
    LinkSpatialVel* link_velocity = nullptr;
    LinkSpatialAccel* link_acceleration = nullptr;
    LinkSpatialTransform* link_xup = nullptr;
    LinkSpatialVel* link_velocity_bias = nullptr;
    LinkBiasForce* link_bias_force = nullptr;
    LinkMotionSubspace* joint_motion_subspace = nullptr;
    math::Transform* link_pose = nullptr;
    math::Transform* link_local_pose = nullptr;
    math::Transform* link_inertial_frame = nullptr;
    math::Transform* base_pose = nullptr;
    float* q = nullptr;
    float* qdot = nullptr;
    float* qddot = nullptr;
    float* tau = nullptr;
    math::Quat* joint_rotation = nullptr;
    math::Vec3* joint_angular_velocity = nullptr;
    float* joint_damping = nullptr;
    float* joint_armature = nullptr;
    float* joint_frictionloss = nullptr;
    float* joint_stiffness = nullptr;
    float* joint_spring_reference = nullptr;
    math::Vec3* joint_axis = nullptr;
    math::Vec3* parent_offset = nullptr;
    ArticulationJointType* joint_type = nullptr;
    uint32_t* parent_link = nullptr;
    uint32_t* link_body = nullptr;
    uint32_t* link_to_articulation = nullptr;
    uint32_t* articulation_link_count = nullptr;
    uint32_t* articulation_link_offset = nullptr;
    uint32_t total_link_count = 0;
    uint32_t articulation_count = 0;
};

constexpr uint32_t kMaxFootContactsPerEnv = 4u;

constexpr uint32_t kMaxContactsPerArtic = 12u;

struct FootShape {
    uint32_t calf_local_link = 0u;
    math::Vec3 local_offset{};
    float radius = 0.0f;
};

constexpr float kInertiaDiagonalEpsilon = 1.0e-6f;
constexpr float kBaumgarteBeta = 0.2f;
constexpr float kPenetrationSlop = 1.0e-4f;

__forceinline__ __device__ uint32_t JointDofCountDevice(ArticulationJointType type) {
    switch (type) {
        case ArticulationJointType::Revolute:
        case ArticulationJointType::Prismatic:
            return 1u;
        case ArticulationJointType::Fixed:
            return 0u;
        case ArticulationJointType::FloatingBase:
            return 6u;  // T8a: free-floating root contributes 6 DOF.
        case ArticulationJointType::Spherical:
            return 3u;
    }
    return 0u;
}

__forceinline__ __device__ uint32_t LocalDofIndexDevice(const ArticulationDeviceState& state,
                                                        uint32_t offset,
                                                        uint32_t link) {
    uint32_t index = 0u;
    for (uint32_t k = offset; k < link; ++k) {
        index += JointDofCountDevice(state.joint_type[k]);
    }
    return index;
}

__device__ inline bool ArticulationDofLocation(
    const ArticulationDeviceState& state, uint32_t articulation, uint32_t dof,
    uint32_t* link, uint32_t* component) {
    if (link == nullptr || component == nullptr || articulation >= state.articulation_count ||
        state.articulation_link_offset == nullptr || state.articulation_link_count == nullptr ||
        state.joint_type == nullptr) return false;
    const uint32_t offset = state.articulation_link_offset[articulation];
    const uint32_t count = state.articulation_link_count[articulation];
    if (offset > state.total_link_count || count > state.total_link_count - offset) return false;
    for (uint32_t local = 0u; local < count; ++local) {
        const uint32_t candidate = offset + local;
        const ArticulationJointType type = state.joint_type[candidate];
        const uint32_t width = JointDofCountDevice(type);
        if (dof < width) {
            if (type == ArticulationJointType::FloatingBase &&
                (local != 0u || state.parent_link == nullptr ||
                 state.parent_link[candidate] != ~0u)) return false;
            *link = candidate;
            *component = width == 1u ? ~0u : dof;
            return true;
        }
        dof -= width;
    }
    return false;
}

// A DOF's velocity: a scalar joint's qdot, a spherical joint's relative spin or a floating root's twist.
__forceinline__ __device__ float& DofVelocity(const ArticulationDeviceState& state, uint32_t link,
                                             uint32_t component) {
    if (component == ~0u) return state.qdot[link];
    if (state.joint_type[link] == ArticulationJointType::Spherical)
        return (&state.joint_angular_velocity[link].x)[component];
    return state.link_velocity[link].v[component];
}

// The free joint a mimic chain ends at, with q_link = scale * q_root + shift along it.
__forceinline__ __device__ uint32_t MimicRootDevice(const ArticulationDeviceState& state,
                                                    const uint32_t* source_link,
                                                    const float* multiplier,
                                                    const float* offset, uint32_t link,
                                                    float* scale, float* shift) {
    const uint32_t articulation = state.link_to_articulation[link];
    float s = 1.0f;
    float c = 0.0f;
    uint32_t root = link;
    for (uint32_t hop = 0u; hop < state.articulation_link_count[articulation]; ++hop) {
        const uint32_t from = source_link[root];
        if (from >= state.total_link_count || state.link_to_articulation[from] != articulation ||
            JointDofCountDevice(state.joint_type[root]) != 1u ||
            JointDofCountDevice(state.joint_type[from]) != 1u)
            break;
        c += s * offset[root];
        s *= multiplier[root];
        root = from;
    }
    *scale = s;
    *shift = c;
    return root;
}

// Generalized force g[dof] += J_dof^T . wrench for ONE joint, the chain-Jacobian
// column builder generalized from a unit scalar direction to a full spatial
// wrench (force f, torque tau at `point`). The scalar-direction case (the
// AssembleRows chain-J) is f=dir, tau=0: tau then contributes an exact +0.0f, so
// that path stays bit-identical. Accumulates into `out` (caller zero-inits), so
// many chains summing onto shared ancestor DOFs (a floating base under several
// feet) compose. Indices guarded by `dof_stride`; FloatingBase emits 6 columns.
__forceinline__ __device__ void AccumulateChainJointForce(
    const ArticulationDeviceState& state, uint32_t articulation, uint32_t link,
    uint32_t dof_index, uint32_t dof_stride, ::nuka::math::Vec3 point,
    ::nuka::math::Vec3 f, ::nuka::math::Vec3 tau, float* out) {
    namespace mg = ::nuka::math::gpu;
    const ArticulationJointType type = state.joint_type[link];
    if (type == ArticulationJointType::FloatingBase) {
        const math::Quat base_rot = state.base_pose[articulation].rotation;
        const math::Vec3 base_origin = state.base_pose[articulation].position;
        const math::Vec3 lever = mg::Sub(point, base_origin);
        const math::Vec3 ex = mg::RotateByQuatNormalized(base_rot, {1.0f, 0.0f, 0.0f});
        const math::Vec3 ey = mg::RotateByQuatNormalized(base_rot, {0.0f, 1.0f, 0.0f});
        const math::Vec3 ez = mg::RotateByQuatNormalized(base_rot, {0.0f, 0.0f, 1.0f});
        const float ang[3] = {mg::Dot(mg::Cross(ex, lever), f) + mg::Dot(ex, tau),
                              mg::Dot(mg::Cross(ey, lever), f) + mg::Dot(ey, tau),
                              mg::Dot(mg::Cross(ez, lever), f) + mg::Dot(ez, tau)};
        const float lin[3] = {mg::Dot(ex, f), mg::Dot(ey, f), mg::Dot(ez, f)};
        for (uint32_t b = 0u; b < 3u; ++b) {
            if (dof_index + b < dof_stride) out[dof_index + b] += ang[b];
            if (dof_index + 3u + b < dof_stride) out[dof_index + 3u + b] += lin[b];
        }
        return;
    }
    if (type == ArticulationJointType::Spherical) {
        // Columns are the child frame's axes through the joint origin.
        const math::Quat rotation = state.link_pose[link].rotation;
        const math::Vec3 lever = mg::Sub(point, state.link_pose[link].position);
        const math::Vec3 moment = mg::Add(mg::Cross(lever, f), tau);
        const math::Vec3 local = mg::RotateByQuatNormalized(
            mg::MakeQuat(rotation.w, -rotation.x, -rotation.y, -rotation.z), moment);
        const float entries[3] = {local.x, local.y, local.z};
        for (uint32_t b = 0u; b < 3u; ++b)
            if (dof_index + b < dof_stride) out[dof_index + b] += entries[b];
        return;
    }
    const math::Vec3 axis_world =
        mg::RotateByQuatNormalized(state.link_pose[link].rotation, state.joint_axis[link]);
    float entry = 0.0f;
    if (type == ArticulationJointType::Prismatic) {
        entry = mg::Dot(axis_world, f);
    } else {  // Revolute
        const math::Vec3 lever = mg::Sub(point, state.link_pose[link].position);
        entry = mg::Dot(mg::Cross(axis_world, lever), f) + mg::Dot(axis_world, tau);
    }
    if (dof_index < dof_stride) out[dof_index] += entry;
}

// --- ModelView/DataView -> legacy kernel-parameter struct ------------------
// The pointer-source rebinding the plan mandates: every member name matches the
// legacy field 1:1 BY DESIGN (M3a codegen), so this is a pure re-aim.
inline ArticulationDeviceState MakeArticulationDeviceState(
    const ModelView& model, const DataView& data,
    uint32_t total_link_count, uint32_t articulation_count) {
    ArticulationDeviceState s;
    s.link_inertia = reinterpret_cast<LinkSpatialInertia*>(model.link_inertia);
    s.link_velocity = reinterpret_cast<LinkSpatialVel*>(data.link_velocity);
    s.link_acceleration = reinterpret_cast<LinkSpatialAccel*>(data.link_acceleration);
    s.link_xup = reinterpret_cast<LinkSpatialTransform*>(data.link_xup);
    s.link_velocity_bias = reinterpret_cast<LinkSpatialVel*>(data.link_velocity_bias);
    s.link_bias_force = reinterpret_cast<LinkBiasForce*>(data.link_bias_force);
    s.joint_motion_subspace =
        reinterpret_cast<LinkMotionSubspace*>(data.joint_motion_subspace);
    s.link_pose = data.link_pose;
    s.link_local_pose = model.link_local_pose;
    s.link_inertial_frame = model.link_inertial_frame;
    s.base_pose = data.base_pose;
    s.q = data.q;
    s.qdot = data.qdot;
    s.qddot = data.qddot;
    s.tau = data.tau;
    s.joint_rotation = data.joint_rotation;
    s.joint_angular_velocity = data.joint_angular_velocity;
    s.joint_damping = model.joint_damping;
    s.joint_armature = model.joint_armature;
    s.joint_frictionloss = model.joint_frictionloss;
    s.joint_stiffness = model.joint_stiffness;
    s.joint_spring_reference = model.joint_spring_reference;
    s.joint_axis = model.joint_axis;
    s.parent_offset = model.parent_offset;
    s.joint_type = reinterpret_cast<ArticulationJointType*>(model.joint_type);
    s.parent_link = model.parent_link;
    s.link_body = model.link_body;
    s.link_to_articulation = model.link_to_articulation;
    s.articulation_link_count = model.articulation_link_count;
    s.articulation_link_offset = model.articulation_link_offset;
    s.total_link_count = total_link_count;
    s.articulation_count = articulation_count;
    return s;
}

} // namespace nuka::phi::nkops
