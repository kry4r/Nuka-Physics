// ImGui calls these hooks for every item while IMGUI_ENABLE_TEST_ENGINE is defined and
// TestEngineHookItems is set; the editor sets it only while it looks an item up.
#include "imgui_item_hooks.hpp"

#include "imgui.h"
#include "imgui_internal.h"

#include <unordered_map>
#include <utility>

namespace nuka::render::imgui {
namespace {

std::vector<HookedItem> g_items;
std::unordered_map<ImGuiID, size_t> g_latest;

}  // namespace

void BeginItemCapture() {
    g_items.clear();
    g_latest.clear();
    ImGui::GetCurrentContext()->TestEngineHookItems = true;
}

std::vector<HookedItem> EndItemCapture() {
    ImGui::GetCurrentContext()->TestEngineHookItems = false;
    g_latest.clear();
    return std::exchange(g_items, {});
}

bool CapturingItems() {
    const ImGuiContext* g = ImGui::GetCurrentContext();
    return g != nullptr && g->TestEngineHookItems;
}

void NoteItemLabel(uint32_t id, const char* label) {
    if (const auto it = g_latest.find(id); it != g_latest.end()) g_items[it->second].label = label;
}

}  // namespace nuka::render::imgui

void ImGuiTestEngineHook_ItemAdd(ImGuiContext* ctx, ImGuiID id, const ImRect& bb, const ImGuiLastItemData*) {
    nuka::render::imgui::HookedItem item;
    item.id = id;
    item.min[0] = bb.Min.x;
    item.min[1] = bb.Min.y;
    item.max[0] = bb.Max.x;
    item.max[1] = bb.Max.y;
    if (ImGuiWindow* window = ctx->CurrentWindow) {
        ImGuiWindow* root = window->RootWindow != nullptr ? window->RootWindow : window;
        // Popups and tooltips draw over every plain window.
        const bool over = (root->Flags & (ImGuiWindowFlags_Popup | ImGuiWindowFlags_Tooltip)) != 0;
        item.window = window->Name;
        item.window_order = ctx->Windows.index_from_ptr(ctx->Windows.find(root)) + (over ? ctx->Windows.Size : 0);
        item.visible = window->ClipRect.Overlaps(bb);
    }
    nuka::render::imgui::g_latest[id] = nuka::render::imgui::g_items.size();
    nuka::render::imgui::g_items.push_back(std::move(item));
}

void ImGuiTestEngineHook_ItemInfo(ImGuiContext*, ImGuiID id, const char* label, ImGuiItemStatusFlags) {
    if (label != nullptr) nuka::render::imgui::NoteItemLabel(id, label);
}

void ImGuiTestEngineHook_Log(ImGuiContext*, const char*, ...) {}

const char* ImGuiTestEngine_FindItemDebugLabel(ImGuiContext*, ImGuiID) { return nullptr; }
