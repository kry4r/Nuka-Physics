#pragma once
// Host forward and inverse kinematics of one environment's articulations, in the conventions of
// the device pose update: one coordinate per link, base poses at floating roots, mimic followers.

#include "math/transform.hpp"
#include "nk/model/model.hpp"

#include <cstdint>
#include <vector>

namespace nuka::nk {

// One environment's world link poses from its per-link coordinates and per-articulation base poses.
std::vector<math::Transform> ArticulationLinkPoses(const ModelArticulation& art, const float* q,
                                                   const math::Transform* base_poses);

// A link point that should reach a world position and, when `orient`, the link frame's world rotation.
struct IkTarget {
    uint32_t link = 0u;
    math::Vec3 point = math::Vec3::Zero();  // link frame
    math::Vec3 position = math::Vec3::Zero();
    math::Quat rotation = math::Quat::Identity();
    bool orient = false;
    float weight = 1.0f;
};

struct IkOptions {
    uint32_t root = ~0u;           // the link the chains start below; ~0u is each target's articulation root
    float rotation_weight = 0.1f;  // metres per radian
    float damping = 1.0e-3f;
    float max_step = 0.2f;  // rad or m per iteration and joint
    uint32_t iterations = 200u;
    float position_tolerance = 1.0e-5f;  // m
    float rotation_tolerance = 1.0e-4f;  // rad
};

struct IkSolution {
    std::vector<float> q;               // per link
    std::vector<uint32_t> moved;        // the links the solve sets: its free joints and their followers
    std::vector<float> position_error;  // per target, m
    std::vector<float> rotation_error;  // per target, rad; 0 without `orient`
    uint32_t iterations = 0u;
    uint32_t joints = 0u;    // free joints
    uint32_t at_limit = 0u;  // free joints ending at a limit
};

// Damped least squares from `q` over the revolute and prismatic joints between the root and the
// targets, a mimic follower moving with its source; false when a target's link is not below the root.
bool SolveArticulationIk(const ModelArticulation& art, const float* q, const math::Transform* base_poses,
                         const std::vector<IkTarget>& targets, const IkOptions& options, IkSolution* out);

}  // namespace nuka::nk
