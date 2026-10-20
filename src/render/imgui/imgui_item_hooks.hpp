#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace nuka::render::imgui {

// One item ImGui submitted while the hooks recorded: its rectangle in window points, the label its
// widget gave (empty for custom widgets that name none), and the window it sits in.
struct HookedItem {
    uint32_t id = 0u;
    float min[2] = {0.0f, 0.0f};
    float max[2] = {0.0f, 0.0f};
    std::string label;
    std::string window;
    int window_order = 0;  // where its window draws; higher is in front
    bool visible = false;  // inside its window's clip rectangle
};

// Starts recording every item the current ImGui context submits.
void BeginItemCapture();
// Stops recording and hands over the items submitted since BeginItemCapture.
std::vector<HookedItem> EndItemCapture();
// True while recording; custom widgets name their last item through NoteItemLabel then.
bool CapturingItems();
void NoteItemLabel(uint32_t id, const char* label);

}  // namespace nuka::render::imgui
