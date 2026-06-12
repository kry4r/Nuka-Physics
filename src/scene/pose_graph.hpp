#pragma once
// ---------------------------------------------------------------------------
// nuka::scene::PoseGraph - flat body-pose mirror (legacy local/world transform
// table). NOT a scene tree: the owner-mandated node tree lives in
// scene/graph/scene_graph.hpp. This legacy mirror retires in M9 with
// scene_pipeline; it was renamed from SceneGraph to free that name.
// ---------------------------------------------------------------------------

#include "math/transform.hpp"
#include "scene/canonical_types.hpp"

#include <string>
#include <vector>

namespace nuka::scene {

struct PoseGraphNode {
    BodyId body_id = kInvalidBody;
    BodyId parent = kInvalidBody;
    std::string name;
    math::Transform local_transform = math::Transform::Identity();
    math::Transform world_transform = math::Transform::Identity();
};

class PoseGraph {
public:
    BodyId AddNode(PoseGraphNode node);
    size_t NodeCount() const;
    const PoseGraphNode& GetNode(BodyId id) const;
    PoseGraphNode& GetNodeMut(BodyId id);
    const std::vector<PoseGraphNode>& Nodes() const;

private:
    std::vector<PoseGraphNode> nodes_;
};

} // namespace nuka::scene
