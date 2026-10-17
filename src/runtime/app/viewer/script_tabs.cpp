#include "runtime/app/viewer/script_tabs.hpp"

#include "render/imgui/nuka_theme.hpp"
#include "runtime/app/viewer/nui/nui.hpp"

#include "TextEditor.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>

namespace nuka::runtime::app::viewer {
namespace {

namespace fs = std::filesystem;
namespace json = scene::json;
namespace theme = nuka::render::imgui::theme;

fs::path U8Path(const std::string& text) { return fs::path(std::u8string(text.begin(), text.end())); }

std::string U8Text(const fs::path& path) {
    const std::u8string text = path.u8string();
    return std::string(text.begin(), text.end());
}

std::string Normal(const std::string& path) { return U8Text(fs::absolute(U8Path(path)).lexically_normal()); }

std::string ReadFile(const std::string& path) {
    std::ifstream in(U8Path(path), std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::ostringstream text;
    text << in.rdbuf();
    return text.str();
}

fs::file_time_type Stamp(const std::string& path) {
    std::error_code error;
    return fs::last_write_time(U8Path(path), error);
}

}  // namespace

ScriptTab::ScriptTab() : editor(std::make_unique<TextEditor>()) { nui::SetupCodeEditor(*editor); }

ScriptTab::~ScriptTab() = default;

std::string ScriptTab::Text() const { return editor->GetText(); }

bool ScriptTab::Dirty() const { return editor->GetUndoIndex() != saved_undo; }

ScriptTab& ScriptTabs::Make() {
    tabs.push_back(std::make_unique<ScriptTab>());
    ScriptTab& tab = *tabs.back();
    tab.id = next_id_++;
    active = tab.id;
    ++version;
    return tab;
}

// Line ends arrive as written and leave as LF.
void ScriptTabs::Load(ScriptTab& tab, std::string text) {
    text.erase(std::remove(text.begin(), text.end(), '\r'), text.end());
    tab.editor->SetText(text);
    tab.saved_undo = tab.editor->GetUndoIndex();
    tab.markers.clear();
}

ScriptTab& ScriptTabs::Add(const std::string& text) {
    ScriptTab& tab = Make();
    tab.title = "<untitled-" + std::to_string(next_untitled_++) + ">";
    Load(tab, text);
    return tab;
}

ScriptTab& ScriptTabs::AddNode(const std::string& node, const std::string& source) {
    for (const auto& t : tabs)
        if (t->path.empty() && t->node == node) {
            Activate(t->id);
            return *t;
        }
    ScriptTab& tab = Make();
    tab.node = node;
    tab.title = node;
    Load(tab, source);
    return tab;
}

ScriptTab& ScriptTabs::Open(const std::string& path) {
    const std::string file = Normal(path);
    for (const auto& t : tabs)
        if (t->path == file) {
            Activate(t->id);
            return *t;
        }
    std::string text = ReadFile(file);
    ScriptTab& tab = Make();
    tab.path = file;
    tab.title = U8Text(U8Path(file).filename());
    tab.stamp = Stamp(file);
    Load(tab, std::move(text));
    return tab;
}

ScriptTab* ScriptTabs::Find(uint64_t id) {
    for (const auto& t : tabs)
        if (t->id == id) return t.get();
    return nullptr;
}

ScriptTab* ScriptTabs::Named(const std::string& name) {
    for (const auto& t : tabs)
        if (t->path.empty() && t->title == name) return t.get();
    if (name.empty() || name.front() == '<') return nullptr;
    const std::string file = Normal(name);
    for (const auto& t : tabs)
        if (t->path == file) return t.get();
    return nullptr;
}

ScriptTab& ScriptTabs::Active() {
    if (ScriptTab* tab = Find(active)) return *tab;
    return tabs.empty() ? Add() : *tabs.back();
}

void ScriptTabs::Activate(uint64_t id) {
    if (active == id) return;
    active = id;
    ++version;
}

void ScriptTabs::Close(uint64_t id) {
    const auto at = std::find_if(tabs.begin(), tabs.end(), [&](const auto& t) { return t->id == id; });
    const size_t index = static_cast<size_t>(at - tabs.begin());
    tabs.erase(at);
    ++version;
    if (tabs.empty()) {
        Add();
    } else if (active == id) {
        active = tabs[std::min(index, tabs.size() - 1u)]->id;
    }
}

void ScriptTabs::Save(ScriptTab& tab, const std::string& path) {
    const std::string file = Normal(path.empty() ? tab.path : path);
    const std::string text = tab.Text();
    {
        std::ofstream out(U8Path(file), std::ios::binary | std::ios::trunc);
        out.write(text.data(), static_cast<std::streamsize>(text.size()));
        if (!out) throw std::runtime_error("cannot write " + file);
    }
    if (tab.path != file) ++version;
    tab.path = file;
    tab.title = U8Text(U8Path(file).filename());
    tab.node.clear();
    tab.saved_undo = tab.editor->GetUndoIndex();
    tab.stamp = Stamp(file);
    tab.disk_changed = false;
}

void ScriptTabs::Reload(ScriptTab& tab) {
    const size_t line = tab.editor->GetMainCursorPosition().line;
    Load(tab, ReadFile(tab.path));
    tab.stamp = Stamp(tab.path);
    tab.disk_changed = false;
    tab.editor->SetCursor(TextEditor::DocPos(line, 0));
}

std::vector<uint64_t> ScriptTabs::Poll() {
    std::vector<uint64_t> reloaded;
    for (const auto& t : tabs) {
        if (t->path.empty()) continue;
        const fs::file_time_type now = Stamp(t->path);
        // A file gone or mid-write reads as no stamp; the next poll sees it again.
        if (now == fs::file_time_type{} || now == t->stamp) continue;
        t->stamp = now;
        if (t->Dirty()) {
            t->disk_changed = true;
            continue;
        }
        Reload(*t);
        reloaded.push_back(t->id);
    }
    return reloaded;
}

void ScriptTabs::Goto(ScriptTab& tab, int line) {
    const size_t row = static_cast<size_t>(std::max(line, 1) - 1);
    tab.editor->SetCursor(TextEditor::DocPos(row, 0));
    tab.editor->ScrollToLine(row, TextEditor::Scroll::alignMiddle);
    tab.editor->SetFocus();
}

void ScriptTabs::Mark(const std::string& output, bool failed) {
    for (const auto& t : tabs) {
        t->markers.clear();
        t->editor->ClearMarkers();
    }
    if (!failed) return;
    // Entries run outermost first, so the last one in a file is where it raised there.
    std::vector<std::pair<ScriptTab*, int>> hits;
    std::string message;
    std::istringstream lines(output);
    for (std::string line; std::getline(lines, line);) {
        std::string file;
        int number = 0;
        if (nui::ParseTraceEntry(line, &file, &number)) {
            if (ScriptTab* tab = Named(file)) {
                const auto at = std::find_if(hits.begin(), hits.end(), [&](const auto& h) { return h.first == tab; });
                if (at == hits.end()) hits.emplace_back(tab, number);
                else at->second = number;
            }
        } else if (!line.empty() && line.front() != ' ' && line.rfind("[script]", 0) != 0 &&
                   line.rfind("Traceback", 0) != 0) {
            message = line;
        }
    }
    const ImU32 number = ImGui::ColorConvertFloat4ToU32(theme::kErr);
    const ImU32 text = ImGui::ColorConvertFloat4ToU32(theme::Alpha(theme::kErr, 0.18f));
    for (const auto& [tab, line] : hits) {
        tab->markers.push_back(ScriptMarker{line, message});
        tab->editor->AddMarker(static_cast<size_t>(line - 1), number, text, message, message);
    }
}

json::Value ScriptTabs::List() const {
    json::Value list = json::Value::Array();
    for (const auto& t : tabs) {
        const TextEditor::DocPos at = t->editor->GetMainCursorPosition();
        json::Value markers = json::Value::Array();
        for (const ScriptMarker& m : t->markers) {
            json::Value v = json::Value::Object();
            v.Set("line", json::Value::Int(m.line));
            v.Set("message", json::Value::Str(m.message));
            markers.PushBack(std::move(v));
        }
        json::Value v = json::Value::Object();
        v.Set("tab", json::Value::Int(static_cast<int64_t>(t->id)));
        v.Set("title", json::Value::Str(t->title));
        v.Set("name", json::Value::Str(t->Name()));
        v.Set("path", json::Value::Str(t->path));
        v.Set("node", json::Value::Str(t->node));
        v.Set("active", json::Value::Bool(t->id == active));
        v.Set("dirty", json::Value::Bool(t->Dirty()));
        v.Set("disk_changed", json::Value::Bool(t->disk_changed));
        v.Set("line", json::Value::Int(static_cast<int64_t>(at.line) + 1));
        v.Set("column", json::Value::Int(static_cast<int64_t>(at.index) + 1));
        v.Set("lines", json::Value::Int(static_cast<int64_t>(t->editor->GetLineCount())));
        v.Set("markers", std::move(markers));
        list.PushBack(std::move(v));
    }
    return list;
}

}  // namespace nuka::runtime::app::viewer
