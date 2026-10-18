#pragma once
// What the Clip Editor shows: the document's clips, the edited one with its playhead and
// selection, and the clip playing.

#include "runtime/app/viewer/clips.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace nuka::runtime::app::viewer {

struct ClipView {
    std::vector<Clip> clips;
    bool open = false;
    int edited = -1;         // the clip the editor shows; -1 when there is none
    float playhead = 0.0f;   // s into it
    bool auto_key = false;   // drive target edits key the edited clip at the playhead
    ClipInterp interp = ClipInterp::Linear;  // what new keys take
    std::vector<ClipKeyRef> selected;
    std::vector<std::pair<std::string, ClipKey>> clipboard;  // times from the earliest copied key
    std::string curve;       // the joint the curve editor edits
    int playing = -1;        // the clip driving the shown environment; -1 when none
    bool looping = false;
    uint32_t rows = 0u;      // its steps
    uint32_t left = 0u;
    float dt = 0.0f;         // the simulation step, which keys snap to [s]
    bool focused = false;    // the editor has the keyboard: undo, redo and delete act on its keys
    bool commit = false;     // the editor changed `clips` in place; the host makes an undo point
    bool can_undo = false;
    bool can_redo = false;
    char name_edit[128] = {};  // the rename field
};

}  // namespace nuka::runtime::app::viewer
