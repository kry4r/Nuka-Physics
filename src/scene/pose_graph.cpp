// ---------------------------------------------------------------------------
// nuka::scene::PoseGraph implementation
// ---------------------------------------------------------------------------

#include "scene/pose_graph.hpp"

#include <stdexcept>
#include <utility>

namespace nuka::scene {

BodyId PoseGraph::AddNode(PoseGraphNode node) {
    const auto id = static_cast<BodyId>(nodes_.size());
    node.body_id = id;
    nodes_.push_back(std::move(node));
    return id;
}

size_t PoseGraph::NodeCount() const {
    return nodes_.size();
}

const PoseGraphNode& PoseGraph::GetNode(BodyId id) const {
    if (id >= nodes_.size()) {
        throw std::out_of_range("PoseGraph::GetNode - invalid BodyId");
    }
    return nodes_[id];
}

PoseGraphNode& PoseGraph::GetNodeMut(BodyId id) {
    if (id >= nodes_.size()) {
        throw std::out_of_range("PoseGraph::GetNodeMut - invalid BodyId");
    }
    return nodes_[id];
}

const std::vector<PoseGraphNode>& PoseGraph::Nodes() const {
    return nodes_;
}

} // namespace nuka::scene
