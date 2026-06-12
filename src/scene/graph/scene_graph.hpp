#pragma once
// ---------------------------------------------------------------------------
// nuka::scene::SceneGraph - the owner-mandated scene TREE.
//
// Modeled on the owner's Mangifera engine scene-graph: a first-child /
// next-sibling node tree. `name` is a SINGLE segment; `path`
// ("h1/pelvis/left_hip") is always DERIVED by walking the tree (PathOf), never
// stored. Each node maps 1:1 to an ECS EntityId (the root holds a sentinel).
// One SceneGraph per Scene (NOT a singleton): the compose layer needs many
// coexisting graphs.
//
// ★ marks extensions over the Mangifera base: per-node EntityId, weak parent
// (breaks the ref cycle), sibling-name uniqueness with auto-suffix,
// AttachSubtree / Detach / DestroyRecursive, and a pre-order Traverse.
// ---------------------------------------------------------------------------

#include "scene/ecs/entity.hpp"

#include <memory>
#include <string>

namespace nuka::scene {

class Registry;  // scene/ecs/registry.hpp (DestroyRecursive only)

struct SceneNode {                              // first-child / next-sibling
    uint64_t    id = 0;                         // unique within the graph
    std::string name;                           // single segment, unique among siblings
    EntityId    entity = kInvalidEntity;        // node <-> entity 1:1
    std::weak_ptr<SceneNode>   parent;          // weak -> no ref cycle
    std::shared_ptr<SceneNode> first_child;
    std::shared_ptr<SceneNode> next_sibling;
};

class SceneGraph {
public:
    SceneGraph();

    std::shared_ptr<SceneNode> Root() const { return root_; }

    // editor selection state (M11 viewer)
    std::shared_ptr<SceneNode> Selected() const { return selected_.lock(); }
    void SetSelected(std::shared_ptr<SceneNode> node) { selected_ = node; }

    // -- Mangifera-faithful API (semantics unchanged) -----------------------

    // Walk up to the root collecting names, join with '/', EXCLUDING the root
    // segment: "h1/right_hand_link". Root (or null) -> "".
    std::string PathOf(const std::shared_ptr<SceneNode>& node) const;

    // Split on '/', walk down by child name. Tolerates a leading "Scene"
    // segment. "" or "/" -> root. nullptr if any segment is not found.
    std::shared_ptr<SceneNode> NodeOf(const std::string& path) const;

    std::shared_ptr<SceneNode> ParentOf(const std::shared_ptr<SceneNode>& node) const;
    std::shared_ptr<SceneNode> FirstChildOf(const std::shared_ptr<SceneNode>& node) const;
    std::shared_ptr<SceneNode> NextSiblingOf(const std::shared_ptr<SceneNode>& node) const;

    // Create a node for `entity` under `parent` (null -> root). `name` empty ->
    // "Entity <id>". Tail-appended among siblings for a deterministic order
    // (serialization / D1 anchor). Sibling-name collisions auto-suffix _1/_2/...
    std::shared_ptr<SceneNode> AddEntity(EntityId entity,
                                         std::shared_ptr<SceneNode> parent,
                                         const std::string& name);

    // -- ★ extensions -------------------------------------------------------

    // Graft an already-detached subtree under `parent` (null -> root).
    // Tail-appended; the subtree root's name is deduped among the new siblings.
    // Cross-graph entity remap is the compose layer's job, not here.
    void AttachSubtree(std::shared_ptr<SceneNode> subtree,
                       std::shared_ptr<SceneNode> parent);

    // Unlink a subtree from its parent (entities untouched); returns it.
    std::shared_ptr<SceneNode> Detach(std::shared_ptr<SceneNode> node);

    // Post-order destroy of the subtree's nodes AND their entities.
    void DestroyRecursive(std::shared_ptr<SceneNode> node, Registry& registry);

    // Pre-order traversal, children in sibling order (deterministic). The
    // callback is invoked as fn(const std::shared_ptr<SceneNode>&).
    template <class F>
    void Traverse(const std::shared_ptr<SceneNode>& start, F&& fn) const {
        if (!start) {
            return;
        }
        fn(start);
        for (auto child = start->first_child; child; child = child->next_sibling) {
            Traverse(child, fn);
        }
    }

private:
    // Append `node` as the last child of `parent`, deduplicating its name
    // against the existing siblings. Sets node->parent.
    void AppendChild(const std::shared_ptr<SceneNode>& parent,
                     const std::shared_ptr<SceneNode>& node);

    // Return a name unique among `parent`'s current children, suffixing
    // _1/_2/... and re-checking until free. `desired` empty -> "Entity <id>".
    std::string UniqueSiblingName(const std::shared_ptr<SceneNode>& parent,
                                  const std::string& desired,
                                  uint64_t node_id) const;

    std::shared_ptr<SceneNode> root_;
    std::weak_ptr<SceneNode>   selected_;
    uint64_t                   next_id_ = 0;  // 0 consumed by the root
};

} // namespace nuka::scene
