#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::articulation -- CUDA-resident Featherstone state layout
// ---------------------------------------------------------------------------

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"
#include "scene/canonical_types.hpp"
#include "scene/cooked_blob.hpp"

#include <cstdint>
#include <vector>

namespace nuka::runtime::articulation {

enum class ArticulationJointType : uint8_t {
    Revolute = 0,
    Prismatic = 1,
    Fixed = 2,
};

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

struct LinkArticulatedInertia {
    float Ia[36] = {};
};

struct LinkBiasForce {
    float p[6] = {};
};

struct LinkMotionSubspace {
    float s[6] = {};
};

struct ArticulationDeviceState {
    LinkSpatialInertia* link_inertia = nullptr;
    LinkSpatialVel* link_velocity = nullptr;
    LinkSpatialAccel* link_acceleration = nullptr;
    LinkSpatialTransform* link_xup = nullptr;
    LinkSpatialVel* link_velocity_bias = nullptr;
    LinkArticulatedInertia* link_articulated_I = nullptr;
    LinkBiasForce* link_bias_force = nullptr;
    LinkBiasForce* link_u_spatial = nullptr;
    LinkMotionSubspace* joint_motion_subspace = nullptr;
    math::Transform* link_pose = nullptr;
    math::Transform* link_local_pose = nullptr;
    math::Transform* link_inertial_frame = nullptr;
    float* q = nullptr;
    float* qdot = nullptr;
    float* qddot = nullptr;
    float* tau = nullptr;
    float* joint_damping = nullptr;
    float* joint_armature = nullptr;
    float* joint_diagonal = nullptr;
    float* joint_force = nullptr;
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

struct ArticulationCookedTopology {
    uint32_t root_body = scene::kInvalidBody;
    std::vector<scene::BodyId> link_bodies;
    std::vector<uint32_t> parent_links;
    std::vector<ArticulationJointType> joint_types;
    std::vector<math::Vec3> joint_axes;
    std::vector<math::Transform> parent_frames;
    std::vector<math::Transform> child_frames;
    std::vector<math::Transform> local_poses;
    std::vector<math::Transform> inertial_frames;
    std::vector<float> joint_dampings;
    std::vector<float> joint_armatures;
    std::vector<float> masses;
    std::vector<math::Vec3> inertias;
};

struct ArticulationHostState {
    std::vector<ArticulationCookedTopology> articulations;
    std::vector<LinkSpatialInertia> link_inertia;
    std::vector<LinkSpatialVel> link_velocity;
    std::vector<LinkSpatialAccel> link_acceleration;
    std::vector<LinkSpatialTransform> link_xup;
    std::vector<LinkSpatialVel> link_velocity_bias;
    std::vector<LinkArticulatedInertia> link_articulated_inertia;
    std::vector<LinkBiasForce> link_bias_force;
    std::vector<LinkBiasForce> link_u_spatial;
    std::vector<LinkMotionSubspace> joint_motion_subspace;
    std::vector<math::Transform> link_pose;
    std::vector<math::Transform> link_local_pose;
    std::vector<math::Transform> link_inertial_frame;
    std::vector<float> q;
    std::vector<float> qdot;
    std::vector<float> qddot;
    std::vector<float> tau;
    std::vector<float> joint_damping;
    std::vector<float> joint_armature;
    std::vector<float> joint_diagonal;
    std::vector<float> joint_force;
    std::vector<math::Vec3> joint_axis;
    std::vector<math::Vec3> parent_offset;
    std::vector<ArticulationJointType> joint_type;
    std::vector<uint32_t> parent_link;
    std::vector<uint32_t> link_body;
    std::vector<uint32_t> link_to_articulation;
    std::vector<uint32_t> articulation_link_count;
    std::vector<uint32_t> articulation_link_offset;

    uint32_t TotalLinkCount() const {
        return static_cast<uint32_t>(link_body.size());
    }

    uint32_t ArticulationCount() const {
        return static_cast<uint32_t>(articulation_link_count.size());
    }
};

struct ArticulationDeviceBuffers {
    phi::Buffer link_inertia;
    phi::Buffer link_velocity;
    phi::Buffer link_acceleration;
    phi::Buffer link_xup;
    phi::Buffer link_velocity_bias;
    phi::Buffer link_articulated_inertia;
    phi::Buffer link_bias_force;
    phi::Buffer link_u_spatial;
    phi::Buffer joint_motion_subspace;
    phi::Buffer link_pose;
    phi::Buffer link_local_pose;
    phi::Buffer link_inertial_frame;
    phi::Buffer q;
    phi::Buffer qdot;
    phi::Buffer qddot;
    phi::Buffer tau;
    phi::Buffer joint_damping;
    phi::Buffer joint_armature;
    phi::Buffer joint_diagonal;
    phi::Buffer joint_force;
    phi::Buffer joint_axis;
    phi::Buffer parent_offset;
    phi::Buffer joint_type;
    phi::Buffer parent_link;
    phi::Buffer link_body;
    phi::Buffer link_to_articulation;
    phi::Buffer articulation_link_count;
    phi::Buffer articulation_link_offset;
    uint32_t total_link_count = 0;
    uint32_t articulation_count = 0;

    ArticulationDeviceState View();
    ArticulationDeviceState View() const;
    bool Empty() const;
};

LinkSpatialInertia MakeDiagonalSpatialInertia(float mass, math::Vec3 diagonal_inertia);
LinkSpatialInertia MakeSpatialInertia(float mass,
                                      math::Vec3 diagonal_inertia,
                                      const math::Transform& inertial_frame);
ArticulationHostState BuildArticulationHostState(
    const std::vector<ArticulationCookedTopology>& topologies,
    const scene::CookedBodyTable& bodies);
ArticulationDeviceBuffers UploadArticulationState(const phi::DeviceContext& context,
                                                  const ArticulationHostState& host_state);
ArticulationDeviceBuffers UploadArticulationState(const ArticulationHostState& host_state);
void DownloadArticulationState(const ArticulationDeviceBuffers& device_state,
                               ArticulationHostState* host_state);

} // namespace nuka::runtime::articulation
