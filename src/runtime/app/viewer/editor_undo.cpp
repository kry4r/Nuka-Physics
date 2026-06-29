// ---------------------------------------------------------------------------
// The editor's reversible-edit history (see editor_undo.hpp).
//
// HOST-ONLY / zero-CUDA-token.
// ---------------------------------------------------------------------------

#include "runtime/app/viewer/editor_undo.hpp"

#include "scene/graph/scene_graph.hpp"

#include <cstdio>
#include <utility>

namespace nuka::runtime::app::viewer {

namespace {

// Resolve a node PATH to its entity against the CURRENT (re-projected) facade.
// Paths are the one identity stable across a re-cook / facade rebuild; entity ids
// are not, so every op resolves by path at apply/revert time.
scene::EntityId EntityOfPath(const EditorScene& es, const std::string& path) {
    if (const auto node = es.scene.Tree().NodeOf(path)) return node->entity;
    return scene::kInvalidEntity;
}

// Re-apply a world transform to the node at `path` through the general mutator.
void ApplyTransformValue(EditorScene* es, const std::string& path,
                         const math::Transform& world) {
    const scene::EntityId e = EntityOfPath(*es, path);
    if (e == scene::kInvalidEntity) {
        std::fprintf(stderr, "[nuka_editor] undo: no node at '%s'\n", path.c_str());
        return;
    }
    EntityRecordIndex idx;
    idx.Build(es->scene);
    ApplyTransformEdit(*es, idx, e, world, /*commit=*/true);
}

// Re-apply a material to the node at `path` through the general mutator.
void ApplyMaterialValue(EditorScene* es, const std::string& path,
                        const scene::RenderMaterial& mat) {
    const scene::EntityId e = EntityOfPath(*es, path);
    if (e == scene::kInvalidEntity) {
        std::fprintf(stderr, "[nuka_editor] undo: no node at '%s'\n", path.c_str());
        return;
    }
    EntityRecordIndex idx;
    idx.Build(es->scene);
    const MaterialBinding mb = ResolveMaterial(*es, idx, e);
    ApplyMaterialEdit(*es, mb, mat, /*commit=*/true);
}

}  // namespace

void EditStack::Record(EditOp op) {
    redo_.clear();  // a new edit truncates the redo branch (linear history)
    undo_.push_back(std::move(op));
    if (undo_.size() > kCap) {
        // Drop the oldest history past the uniform cap -- loud, never silent.
        std::fprintf(stderr,
                     "[nuka_editor] edit history exceeded cap %zu; dropping oldest undo\n",
                     kCap);
        undo_.erase(undo_.begin());
    }
}

bool EditStack::Undo() {
    if (undo_.empty()) return false;
    EditOp op = std::move(undo_.back());
    undo_.pop_back();
    if (op.revert) op.revert();
    redo_.push_back(std::move(op));
    return true;
}

bool EditStack::Redo() {
    if (redo_.empty()) return false;
    EditOp op = std::move(redo_.back());
    redo_.pop_back();
    if (op.apply) op.apply();
    undo_.push_back(std::move(op));
    return true;
}

void EditStack::Clear() {
    undo_.clear();
    redo_.clear();
}

EditOp MakeTransformOp(EditorScene* es, std::string path,
                       const math::Transform& before, const math::Transform& after) {
    EditOp op;
    op.apply  = [es, path, after]()  { ApplyTransformValue(es, path, after); };
    op.revert = [es, path, before]() { ApplyTransformValue(es, path, before); };
    return op;
}

EditOp MakeMaterialOp(EditorScene* es, std::string path,
                      const scene::RenderMaterial& before,
                      const scene::RenderMaterial& after) {
    EditOp op;
    op.apply  = [es, path, after]()  { ApplyMaterialValue(es, path, after); };
    op.revert = [es, path, before]() { ApplyMaterialValue(es, path, before); };
    return op;
}

EditOp MakeStructuralOp(EditorScene* es, EntityRecordIndex* idx,
                        nuka::phi::Device* device, nuka::phi::Backend* backend,
                        float dt, scene::SceneIR before, scene::SceneIR after) {
    // Restore a snapshot into the authority, re-cook the world from it, and rebuild
    // the entity->record index (entity ids re-project, so the index is stale).
    auto restore = [es, idx, device, backend, dt](const scene::SceneIR& snap) {
        es->scene = snap;
        if (!RecookEditorScene(*es, device, backend, dt))
            std::fprintf(stderr, "[nuka_editor] undo re-cook failed\n");
        idx->Build(es->scene);
    };
    EditOp op;
    op.apply  = [restore, snap = std::move(after)]()  { restore(snap); };
    op.revert = [restore, snap = std::move(before)]() { restore(snap); };
    return op;
}

}  // namespace nuka::runtime::app::viewer
