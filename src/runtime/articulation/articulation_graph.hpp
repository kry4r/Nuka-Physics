#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::articulation – Articulation graph for multi-body systems
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "math/transform.hpp"
#include "scene/canonical_types.hpp"

#include <cstdint>
#include <vector>

namespace nuka::runtime::articulation {

struct ArticulationLink {
    uint32_t body_id;
    uint32_t parent_link = ~0u;
    scene::JointType joint_type = scene::JointType::Revolute;
    math::Vec3 joint_axis = {0, 0, 1};
    math::Transform parent_frame = math::Transform::Identity();
    math::Transform child_frame = math::Transform::Identity();
    float mass = 0.0f;
    math::Vec3 diagonal_inertia = math::Vec3::Zero();
};

struct ArticulationGraph {
    std::vector<ArticulationLink> links;
    uint32_t root_body = 0;

    void AddLink(ArticulationLink link) {
        links.push_back(link);
    }

    size_t LinkCount() const {
        return links.size();
    }
};

} // namespace nuka::runtime::articulation
