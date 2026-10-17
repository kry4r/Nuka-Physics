#pragma once
// The Script panel's documents: files, untitled buffers and /script node sources, each in a
// code editor of its own, with the error markers of the last run and the file stamps hot
// reload compares.

#include "scene/format/json.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

class TextEditor;

namespace nuka::runtime::app::viewer {

struct ScriptMarker {
    int line = 0;  // 1-based
    std::string message;
};

struct ScriptTab {
    ScriptTab();
    ~ScriptTab();
    // What its runs compile as, so tracebacks name it: the file, else the title.
    const std::string& Name() const { return path.empty() ? title : path; }
    std::string Text() const;
    bool Dirty() const;

    uint64_t id = 0u;
    std::string path;   // absolute; empty for a buffer
    std::string title;  // the file name, "<untitled-N>" or the /script node's path
    std::string node;   // the /script node the source came from
    std::unique_ptr<TextEditor> editor;
    size_t saved_undo = 0u;  // the editor's undo index at the last load or save
    std::filesystem::file_time_type stamp{};
    bool disk_changed = false;  // the file changed on disk while the tab had edits
    std::vector<ScriptMarker> markers;
    uint64_t completion = 0u;  // the completion request in flight
    bool completion_set = false;
};

class ScriptTabs {
public:
    // A new untitled buffer holding `text`, made active.
    ScriptTab& Add(const std::string& text = std::string());
    // A buffer over a /script node's source; the node's open buffer when there is one.
    ScriptTab& AddNode(const std::string& node, const std::string& source);
    // Shows `path` in a tab of its own, or activates the one showing it; throws when unreadable.
    ScriptTab& Open(const std::string& path);
    ScriptTab* Find(uint64_t id);
    // The tab a traceback calls `name`.
    ScriptTab* Named(const std::string& name);
    ScriptTab& Active();
    void Activate(uint64_t id);
    // The last tab closed leaves an empty buffer in its place.
    void Close(uint64_t id);
    // Writes the tab to `path` (empty: its own file), which becomes its file; throws on failure.
    void Save(ScriptTab& tab, const std::string& path = std::string());
    // Reads the tab's file again, dropping its edits but keeping the caret line.
    void Reload(ScriptTab& tab);
    // Clean tabs whose file changed reload, ones with edits get disk_changed; the reloaded ids.
    std::vector<uint64_t> Poll();
    // Puts the caret on 1-based `line` and scrolls it into view.
    void Goto(ScriptTab& tab, int line);
    // The markers follow the last run: cleared, then set where a failure's traceback passes
    // through an open tab, with the error it ended in.
    void Mark(const std::string& output, bool failed);
    scene::json::Value List() const;

    std::vector<std::unique_ptr<ScriptTab>> tabs;
    uint64_t active = 0u;
    uint64_t version = 0u;  // bumped when tabs open, close or change files

private:
    ScriptTab& Make();
    void Load(ScriptTab& tab, std::string text);

    uint64_t next_id_ = 1u;
    int next_untitled_ = 1;
};

}  // namespace nuka::runtime::app::viewer
