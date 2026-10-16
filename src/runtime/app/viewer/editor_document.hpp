#pragma once
// The editor's document: the edited scene and a history of serializable edit transactions.
// Every edit entry point submits an EditTxn; undo and redo dispatch through the same path.

#include "math/transform.hpp"
#include "nuka/nuka.h"
#include "runtime/app/viewer/editor_edits.hpp"
#include "runtime/app/viewer/primitive_kind.hpp"
#include "scene/ecs/components.hpp"
#include "scene/format/json.hpp"
#include "scene/scene_ir.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nuka::runtime::app::viewer {

// One field change on the node at `target` ("" is the world). A null `before` is read from
// the document when the op applies.
struct EditOp {
    std::string target;
    std::string field;
    scene::json::Value before;
    scene::json::Value after;
};

struct EditTxn {
    std::string label;
    std::vector<EditOp> ops;
};

scene::json::Value TxnToJson(const EditTxn& txn);
// Throws on a malformed transaction.
EditTxn TxnFromJson(const scene::json::Value& v);
scene::json::Value TransformToJson(const math::Transform& t);
scene::json::Value MaterialToJson(const scene::RenderMaterial& m);
bool ParsePrimitiveKind(const std::string& s, PrimitiveKind* out);
const char* PrimitiveKindName(PrimitiveKind kind);

// A rebuild cut the timeline at this tick; the rebuilt world continues from it.
struct TimelineBranch {
    uint64_t tick = 0u;
    uint64_t generation = 0u;
    std::string label;
};

// A live registry write for the physics thread.
struct ParamWrite {
    uint32_t param = 0u;
    uint32_t element = 0u;
    double value = 0.0;
};

// What the document needs from the editor around it.
class DocumentHost {
public:
    virtual ~DocumentHost() = default;
    virtual void PushParam(const ParamWrite& write) = 0;
    // Rebuilds the world from the document with the worker parked; false when refused.
    virtual bool Rebuild(const std::string& label) = 0;
    virtual void ScriptsChanged() = 0;
};

struct ApplyResult {
    bool ok = true;
    std::string focus;  // the node an edit created, renamed or moved
    std::string error;
};

class EditorDocument {
public:
    static constexpr size_t kCap = 256u;

    // A new scene: the history, branches and saved state start over.
    void Attach(EditorScene* es, DocumentHost* host);
    void Detach();
    void SetParams(std::vector<nuka_param_info_t> infos, std::vector<double> world_values);

    // Callers hold the worker parked, since structural edits rebuild the world.
    ApplyResult Apply(const EditTxn& txn);
    ApplyResult Undo();
    ApplyResult Redo();

    bool CanUndo() const { return !undo_.empty(); }
    bool CanRedo() const { return !redo_.empty(); }
    size_t UndoDepth() const { return undo_.size(); }
    size_t RedoDepth() const { return redo_.size(); }
    const EntityRecordIndex& Records() const { return records_; }
    void MarkSaved() { saved_ = StateId(); }
    bool Dirty() const { return StateId() != saved_; }
    uint64_t Revision() const { return revision_; }
    void AddBranch(TimelineBranch branch) { branches_.push_back(std::move(branch)); }
    const std::vector<TimelineBranch>& Branches() const { return branches_; }

private:
    struct Entry {
        uint64_t id = 0u;
        EditTxn forward;
        EditTxn inverse;  // property ops undone in reverse order
        bool snapshot = false;
        bool rebuild = false;
        std::string focus;  // node the forward edit selected
        scene::SceneIR before;
        scene::SceneIR after;
    };
    struct OpOutcome {
        std::vector<ParamWrite> writes;
        bool rebuild = false;
        std::string focus;
    };

    uint64_t StateId() const { return undo_.empty() ? 0u : undo_.back().id; }
    scene::EntityId EntityAt(const std::string& path) const;
    uint32_t ParamOf(const std::string& field) const;
    bool LiveElement(const std::string& target, uint32_t param, uint32_t* element) const;
    bool NeedsSnapshot(const EditOp& op) const;
    bool ApplyOp(const EditOp& op, EditOp* done, EditOp* inverse, OpOutcome* out, std::string* error);
    bool ApplyParam(const EditOp& op, EditOp* done, OpOutcome* out, std::string* error);
    // Replays the property ops of a txn after its snapshot was restored.
    void ReplayProperties(const EditTxn& txn, bool rebuilt);
    ApplyResult Restore(const Entry& entry, bool forward);

    EditorScene* es_ = nullptr;
    DocumentHost* host_ = nullptr;
    EntityRecordIndex records_;
    std::vector<Entry> undo_;
    std::vector<Entry> redo_;
    std::vector<TimelineBranch> branches_;
    std::vector<nuka_param_info_t> infos_;
    std::vector<double> world_values_;
    uint64_t next_id_ = 0u;
    uint64_t saved_ = 0u;
    uint64_t revision_ = 0u;
};

}  // namespace nuka::runtime::app::viewer
