#pragma once
// ---------------------------------------------------------------------------
// nuka::scene::SceneGraph - hierarchical scene transform graph
// ---------------------------------------------------------------------------

#include "math/transform.hpp"
#include "scene/canonical_types.hpp"

#include <string>
#include <vector>

namespace nuka::scene {

struct SceneGraphNode {
    BodyId body_id = kInvalidBody;
    BodyId parent = kInvalidBody;
    std::string name;
    math::Transform local_transform = math::Transform::Identity();
    math::Transform world_transform = math::Transform::Identity();
};

class SceneGraph {
public:
    BodyId AddNode(SceneGraphNode node);
    size_t NodeCount() const;
    const SceneGraphNode& GetNode(BodyId id) const;
    SceneGraphNode& GetNodeMut(BodyId id);
    const std::vector<SceneGraphNode>& Nodes() const;

private:
    std::vector<SceneGraphNode> nodes_;
};

} // namespace nuka::scene
