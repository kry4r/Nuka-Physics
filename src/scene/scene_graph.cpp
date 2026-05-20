// ---------------------------------------------------------------------------
// nuka::scene::SceneGraph implementation
// ---------------------------------------------------------------------------

#include "scene/scene_graph.hpp"

#include <stdexcept>
#include <utility>

namespace nuka::scene {

BodyId SceneGraph::AddNode(SceneGraphNode node) {
    const auto id = static_cast<BodyId>(nodes_.size());
    node.body_id = id;
    nodes_.push_back(std::move(node));
    return id;
}

size_t SceneGraph::NodeCount() const {
    return nodes_.size();
}

const SceneGraphNode& SceneGraph::GetNode(BodyId id) const {
    if (id >= nodes_.size()) {
        throw std::out_of_range("SceneGraph::GetNode - invalid BodyId");
    }
    return nodes_[id];
}

SceneGraphNode& SceneGraph::GetNodeMut(BodyId id) {
    if (id >= nodes_.size()) {
        throw std::out_of_range("SceneGraph::GetNodeMut - invalid BodyId");
    }
    return nodes_[id];
}

const std::vector<SceneGraphNode>& SceneGraph::Nodes() const {
    return nodes_;
}

} // namespace nuka::scene
