#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::app::viewer -- the editor's reversible-edit history (EditStack).
//
// A type-agnostic undo/redo stack of EditOps: each EditOp is a pair of closures
// (apply / revert) the stack invokes without inspecting what they do, so a new
// edit kind needs no stack branch. The op FACTORIES wrap the editor_edits
// mutators so every committed edit records ONE reversible op -- property edits
// re-apply the prior value through the same mutator (cheap, exact); structural
// edits restore a SceneIR snapshot + re-cook. ONE general path, no per-type stack.
//
// HOST-ONLY / zero-CUDA-token: the only device touch is the re-cook a structural
// revert delegates to RecookEditorScene (behind the phi vtable).
// ---------------------------------------------------------------------------

#include "math/transform.hpp"
#include "runtime/app/viewer/editor_edits.hpp"  // EditorScene, EntityRecordIndex, mutators
#include "scene/ecs/components.hpp"              // RenderMaterial
#include "scene/scene_ir.hpp"                    // SceneIR (snapshot by value)

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace nuka::phi {
class Device;
class Backend;
}  // namespace nuka::phi

namespace nuka::runtime::app::viewer {

// One reversible edit: apply() lands (or re-lands) it, revert() undoes it. The
// stack only invokes these -- a new edit kind supplies its own two closures.
struct EditOp {
    std::function<void()> apply;
    std::function<void()> revert;
};

// A linear undo/redo history. Record() pushes a freshly-applied op and clears the
// redo branch; Undo() reverts the newest op onto the redo branch; Redo() re-applies
// it. Depth is bounded by a uniform cap with a loud overflow drop (the cap
// discipline -- no per-scene number).
class EditStack {
public:
    // The uniform history-depth cap. Oldest entries drop (loud) past it.
    static constexpr std::size_t kCap = 256u;

    void Record(EditOp op);
    bool Undo();
    bool Redo();
    void Clear();

    bool        CanUndo() const { return !undo_.empty(); }
    bool        CanRedo() const { return !redo_.empty(); }
    std::size_t UndoDepth() const { return undo_.size(); }
    std::size_t RedoDepth() const { return redo_.size(); }

private:
    std::vector<EditOp> undo_;
    std::vector<EditOp> redo_;
};

// -- op factories: wrap the editor_edits mutators as reversible ops -----------
// All resolve identity by tree PATH (the one identity stable across a re-cook /
// facade rebuild) and `es` outlives the op (the stack is cleared on a scene swap).

// A property transform edit on the node at `path`: revert re-applies `before`,
// redo re-applies `after`, both through ApplyTransformEdit (commit) -- exact and
// cheap, no snapshot.
EditOp MakeTransformOp(EditorScene* es, std::string path,
                       const math::Transform& before, const math::Transform& after);

// A property material edit on the node at `path`: revert/redo re-apply before /
// after through ApplyMaterialEdit (commit) against the freshly-resolved binding.
EditOp MakeMaterialOp(EditorScene* es, std::string path,
                      const nuka::scene::RenderMaterial& before,
                      const nuka::scene::RenderMaterial& after);

// A structural edit (add / delete / rename / reparent): the simplest correct
// general revert is a SceneIR snapshot. revert restores `before`, redo restores
// `after`, each re-cooking the world (RecookEditorScene) + rebuilding `idx`. The
// caller WaitIdle's before invoking undo/redo (the re-cook contract).
EditOp MakeStructuralOp(EditorScene* es, EntityRecordIndex* idx,
                        nuka::phi::Device* device, nuka::phi::Backend* backend,
                        float dt, nuka::scene::SceneIR before,
                        nuka::scene::SceneIR after);

}  // namespace nuka::runtime::app::viewer
