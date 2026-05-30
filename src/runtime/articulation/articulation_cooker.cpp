// ---------------------------------------------------------------------------
// nuka::runtime::articulation -- articulation cooker
// ---------------------------------------------------------------------------

#include "runtime/articulation/articulation_cooker.hpp"

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nuka::runtime::articulation {

namespace {

ArticulationJointType ToArticulationJointType(scene::JointType type) {
    if (type == scene::JointType::Prismatic) {
        return ArticulationJointType::Prismatic;
    }
    if (type == scene::JointType::Fixed) {
        return ArticulationJointType::Fixed;
    }
    return ArticulationJointType::Revolute;
}

bool IsSupportedJoint(scene::JointType type) {
    return type == scene::JointType::Revolute ||
           type == scene::JointType::Prismatic ||
           type == scene::JointType::Fixed;
}

math::Vec3 BodyInertia(const scene::CookedBlob& blob, scene::BodyId body) {
    if (body < blob.bodies.inertias.size()) {
        return blob.bodies.inertias[body];
    }
    if (body < blob.bodies.inv_inertias.size()) {
        const math::Vec3 inv = blob.bodies.inv_inertias[body];
        return {
            inv.x > 0.0f ? 1.0f / inv.x : 0.0f,
            inv.y > 0.0f ? 1.0f / inv.y : 0.0f,
            inv.z > 0.0f ? 1.0f / inv.z : 0.0f
        };
    }
    return math::Vec3::Zero();
}

float BodyMass(const scene::CookedBlob& blob, scene::BodyId body) {
    if (body < blob.bodies.masses.size()) {
        return blob.bodies.masses[body];
    }
    if (body < blob.bodies.inv_masses.size() && blob.bodies.inv_masses[body] > 0.0f) {
        return 1.0f / blob.bodies.inv_masses[body];
    }
    return 0.0f;
}

math::Transform FrameOrIdentity(const std::vector<math::Transform>& frames, uint32_t index) {
    if (index < frames.size()) {
        return frames[index];
    }
    return math::Transform::Identity();
}

math::Transform BodyLocalPose(const scene::CookedBlob& blob, scene::BodyId body) {
    if (body < blob.bodies.local_poses.size()) {
        return blob.bodies.local_poses[body];
    }
    return math::Transform::Identity();
}

math::Transform BodyInertialFrame(const scene::CookedBlob& blob, scene::BodyId body) {
    if (body < blob.bodies.inertial_frames.size()) {
        return blob.bodies.inertial_frames[body];
    }
    return math::Transform::Identity();
}

math::Vec3 AxisOrDefault(const std::vector<math::Vec3>& axes, uint32_t index) {
    if (index < axes.size()) {
        return axes[index];
    }
    return math::Vec3::UnitZ();
}

float FloatOrDefault(const std::vector<float>& values, uint32_t index, float fallback) {
    if (index < values.size()) {
        return values[index];
    }
    return fallback;
}

} // namespace

std::vector<ArticulationCookedTopology> CookArticulations(const scene::CookedBlob& blob) {
    std::vector<uint32_t> child_joint(blob.body_count, scene::kInvalidJoint);
    std::vector<std::vector<uint32_t>> children(blob.body_count);

    for (uint32_t joint_index = 0u; joint_index < blob.joint_count; ++joint_index) {
        if (joint_index >= blob.joints.parent_bodies.size() ||
            joint_index >= blob.joints.child_bodies.size()) {
            continue;
        }
        const scene::JointType type = joint_index < blob.joints.types.size()
            ? blob.joints.types[joint_index]
            : scene::JointType::Revolute;
        if (!IsSupportedJoint(type)) {
            continue;
        }
        const scene::BodyId parent = blob.joints.parent_bodies[joint_index];
        const scene::BodyId child = blob.joints.child_bodies[joint_index];
        if (parent >= blob.body_count || child >= blob.body_count) {
            continue;
        }
        child_joint[child] = joint_index;
        children[parent].push_back(joint_index);
    }

    std::vector<uint8_t> has_parent(blob.body_count, 0u);
    for (scene::BodyId body = 0u; body < blob.body_count; ++body) {
        if (child_joint[body] != scene::kInvalidJoint) {
            has_parent[body] = 1u;
        }
    }

    std::vector<ArticulationCookedTopology> result;
    std::vector<uint8_t> visited(blob.body_count, 0u);

    for (scene::BodyId root = 0u; root < blob.body_count; ++root) {
        if (visited[root] != 0u || children[root].empty() || has_parent[root] != 0u) {
            continue;
        }

        ArticulationCookedTopology topology;
        topology.root_body = root;
        std::unordered_map<scene::BodyId, uint32_t> local_link_for_body;
        std::vector<scene::BodyId> stack{root};

        while (!stack.empty()) {
            const scene::BodyId body = stack.back();
            stack.pop_back();
            if (body >= blob.body_count || visited[body] != 0u) {
                continue;
            }

            const uint32_t local_link = static_cast<uint32_t>(topology.link_bodies.size());
            local_link_for_body[body] = local_link;
            topology.link_bodies.push_back(body);
            topology.masses.push_back(BodyMass(blob, body));
            topology.inertias.push_back(BodyInertia(blob, body));
            topology.local_poses.push_back(BodyLocalPose(blob, body));
            topology.inertial_frames.push_back(BodyInertialFrame(blob, body));

            const uint32_t incoming_joint = child_joint[body];
            if (incoming_joint == scene::kInvalidJoint) {
                // T8a: this body is the articulation ROOT (no incoming joint).
                // A movable root (non-static AND positive mass) cooks to a
                // free-floating 6-DOF base; a kinematic / mass-0 root keeps
                // cooking Fixed exactly as before (so the go2_stand oracle and
                // every existing fixed-base scene are byte-for-byte unchanged).
                const bool root_is_static =
                    body < blob.bodies.is_static.size() && blob.bodies.is_static[body] != 0u;
                const bool root_is_floating =
                    !root_is_static && BodyMass(blob, body) > 0.0f;
                topology.parent_links.push_back(scene::kInvalidBody);
                topology.joint_types.push_back(root_is_floating
                    ? ArticulationJointType::FloatingBase
                    : ArticulationJointType::Fixed);
                topology.joint_axes.push_back(math::Vec3::UnitZ());
                topology.parent_frames.push_back(math::Transform::Identity());
                topology.child_frames.push_back(math::Transform::Identity());
                topology.initial_positions.push_back(0.0f);
                topology.joint_dampings.push_back(0.0f);
                topology.joint_armatures.push_back(0.0f);
            } else {
                const scene::BodyId parent = blob.joints.parent_bodies[incoming_joint];
                const auto parent_it = local_link_for_body.find(parent);
                topology.parent_links.push_back(parent_it == local_link_for_body.end()
                    ? scene::kInvalidBody
                    : parent_it->second);
                const scene::JointType joint_type = incoming_joint < blob.joints.types.size()
                    ? blob.joints.types[incoming_joint]
                    : scene::JointType::Revolute;
                topology.joint_types.push_back(ToArticulationJointType(joint_type));
                topology.joint_axes.push_back(AxisOrDefault(blob.joints.axes, incoming_joint));
                topology.parent_frames.push_back(
                    FrameOrIdentity(blob.joints.parent_frames, incoming_joint));
                topology.child_frames.push_back(
                    FrameOrIdentity(blob.joints.child_frames, incoming_joint));
                topology.initial_positions.push_back(
                    FloatOrDefault(blob.joints.initial_positions, incoming_joint, 0.0f));
                topology.joint_dampings.push_back(
                    FloatOrDefault(blob.joints.dampings, incoming_joint, 0.0f));
                topology.joint_armatures.push_back(
                    FloatOrDefault(blob.joints.armatures, incoming_joint, 0.0f));
            }

            visited[body] = 1u;

            auto ordered_children = children[body];
            std::sort(ordered_children.rbegin(), ordered_children.rend());
            for (const uint32_t joint_index : ordered_children) {
                const scene::BodyId child = blob.joints.child_bodies[joint_index];
                if (child < blob.body_count && visited[child] == 0u) {
                    stack.push_back(child);
                }
            }
        }

        if (topology.link_bodies.size() > 1u) {
            result.push_back(std::move(topology));
        }
    }

    return result;
}

} // namespace nuka::runtime::articulation
