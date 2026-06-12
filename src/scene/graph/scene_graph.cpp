// ---------------------------------------------------------------------------
// nuka::scene::SceneGraph implementation (the owner-mandated scene TREE).
// ---------------------------------------------------------------------------

#include "scene/graph/scene_graph.hpp"

#include "scene/ecs/registry.hpp"

#include <utility>
#include <vector>

namespace nuka::scene {

namespace {

// Split a path on '/', dropping empty segments (so "a//b", leading/trailing
// slashes are tolerated). A purely empty/slash path yields no segments.
std::vector<std::string> SplitPath(const std::string& path) {
    std::vector<std::string> out;
    std::string segment;
    for (char ch : path) {
        if (ch == '/') {
            if (!segment.empty()) {
                out.push_back(std::move(segment));
                segment.clear();
            }
        } else {
            segment.push_back(ch);
        }
    }
    if (!segment.empty()) {
        out.push_back(std::move(segment));
    }
    return out;
}

} // namespace

SceneGraph::SceneGraph() {
    root_ = std::make_shared<SceneNode>();
    root_->id = next_id_++;          // root gets id 0
    root_->name = "Scene";
    root_->entity = kInvalidEntity;  // sentinel: the root owns no real entity
}

std::string SceneGraph::PathOf(const std::shared_ptr<SceneNode>& node) const {
    if (!node || node == root_) {
        return "";
    }
    // Walk up collecting names (excluding the root), then reverse-join.
    std::vector<std::string> names;
    for (auto cur = node; cur && cur != root_; cur = cur->parent.lock()) {
        names.push_back(cur->name);
    }
    std::string path;
    for (auto it = names.rbegin(); it != names.rend(); ++it) {
        if (!path.empty()) {
            path.push_back('/');
        }
        path += *it;
    }
    return path;
}

std::shared_ptr<SceneNode> SceneGraph::NodeOf(const std::string& path) const {
    auto segments = SplitPath(path);
    if (segments.empty()) {
        return root_;  // "" or "/" -> root
    }
    // Tolerate a leading "Scene" segment (the root's own name).
    size_t start = 0;
    if (segments.front() == root_->name) {
        start = 1;
        if (start >= segments.size()) {
            return root_;
        }
    }
    auto cur = root_;
    for (size_t i = start; i < segments.size(); ++i) {
        std::shared_ptr<SceneNode> match;
        for (auto child = cur->first_child; child; child = child->next_sibling) {
            if (child->name == segments[i]) {
                match = child;
                break;
            }
        }
        if (!match) {
            return nullptr;
        }
        cur = match;
    }
    return cur;
}

std::shared_ptr<SceneNode> SceneGraph::ParentOf(const std::shared_ptr<SceneNode>& node) const {
    return node ? node->parent.lock() : nullptr;
}

std::shared_ptr<SceneNode> SceneGraph::FirstChildOf(const std::shared_ptr<SceneNode>& node) const {
    return node ? node->first_child : nullptr;
}

std::shared_ptr<SceneNode> SceneGraph::NextSiblingOf(const std::shared_ptr<SceneNode>& node) const {
    return node ? node->next_sibling : nullptr;
}

std::string SceneGraph::UniqueSiblingName(const std::shared_ptr<SceneNode>& parent,
                                          const std::string& desired,
                                          uint64_t node_id) const {
    std::string base = desired.empty() ? ("Entity " + std::to_string(node_id)) : desired;

    auto taken = [&](const std::string& candidate) {
        for (auto child = parent->first_child; child; child = child->next_sibling) {
            if (child->name == candidate) {
                return true;
            }
        }
        return false;
    };

    if (!taken(base)) {
        return base;
    }
    // Suffix _1/_2/... and re-check (the suffixed name must itself be unique).
    for (uint64_t n = 1;; ++n) {
        std::string candidate = base + "_" + std::to_string(n);
        if (!taken(candidate)) {
            return candidate;
        }
    }
}

void SceneGraph::AppendChild(const std::shared_ptr<SceneNode>& parent,
                             const std::shared_ptr<SceneNode>& node) {
    node->name = UniqueSiblingName(parent, node->name, node->id);
    node->parent = parent;
    node->next_sibling.reset();

    if (!parent->first_child) {
        parent->first_child = node;
        return;
    }
    auto tail = parent->first_child;
    while (tail->next_sibling) {
        tail = tail->next_sibling;
    }
    tail->next_sibling = node;
}

std::shared_ptr<SceneNode> SceneGraph::AddEntity(EntityId entity,
                                                 std::shared_ptr<SceneNode> parent,
                                                 const std::string& name) {
    if (!parent) {
        parent = root_;
    }
    auto node = std::make_shared<SceneNode>();
    node->id = next_id_++;
    node->entity = entity;
    node->name = name;  // deduped against siblings inside AppendChild
    AppendChild(parent, node);
    return node;
}

void SceneGraph::AttachSubtree(std::shared_ptr<SceneNode> subtree,
                               std::shared_ptr<SceneNode> parent) {
    if (!subtree || subtree == root_) {
        return;  // never graft a graph's own root under something
    }
    if (!parent) {
        parent = root_;
    }
    // Cycle guard: refuse to graft a node under itself or under any of its own
    // descendants (walking up from `parent` must not reach `subtree`).
    for (auto cur = parent; cur; cur = cur->parent.lock()) {
        if (cur == subtree) {
            return;
        }
    }
    // If the subtree is still attached somewhere, unlink it first. Appending a
    // still-linked node would otherwise truncate its old sibling chain
    // (AppendChild resets next_sibling), corrupting the source tree.
    if (subtree->parent.lock() || subtree->next_sibling) {
        Detach(subtree);
    }
    AppendChild(parent, subtree);
}

std::shared_ptr<SceneNode> SceneGraph::Detach(std::shared_ptr<SceneNode> node) {
    if (!node || node == root_) {
        return node;
    }
    auto parent = node->parent.lock();
    if (parent) {
        if (parent->first_child == node) {
            parent->first_child = node->next_sibling;
        } else {
            for (auto prev = parent->first_child; prev; prev = prev->next_sibling) {
                if (prev->next_sibling == node) {
                    prev->next_sibling = node->next_sibling;
                    break;
                }
            }
        }
    }
    node->parent.reset();
    node->next_sibling.reset();
    return node;
}

void SceneGraph::DestroyRecursive(std::shared_ptr<SceneNode> node, Registry& registry) {
    if (!node) {
        return;
    }
    // Unlink the subtree first so the structure stays consistent while we tear
    // it down (and so a child's next_sibling doesn't drag in siblings).
    if (node != root_) {
        Detach(node);
    }

    // Post-order: destroy children (and their entities) before the node itself.
    // Collect children first because DestroyRecursive mutates the sibling list.
    std::vector<std::shared_ptr<SceneNode>> children;
    for (auto child = node->first_child; child; child = child->next_sibling) {
        children.push_back(child);
    }
    for (auto& child : children) {
        DestroyRecursive(child, registry);
    }
    node->first_child.reset();

    if (node == root_) {
        // The root holds a sentinel entity and stays alive; only its subtree is
        // gone. (DestroyRecursive on the root clears the whole scene.)
        return;
    }
    if (node->entity != kInvalidEntity) {
        registry.Destroy(node->entity);
    }
    if (selected_.lock() == node) {
        selected_.reset();
    }
}

} // namespace nuka::scene
