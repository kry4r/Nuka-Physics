#pragma once
// A drawn mouse pointer that scripts move and click with, for demos and recorded walkthroughs.

#include "render/imgui/imgui_item_hooks.hpp"
#include "render/window/window_surface.hpp"

#include <chrono>
#include <deque>
#include <string>
#include <vector>

struct ImDrawList;
struct ImGuiIO;

namespace nuka::runtime::app::viewer {

// Plays queued glides, presses and wheel turns, one frame at a time, in window points. While it is
// on, the real mouse is ignored: the editor takes its mouse events from here instead.
class DemoPointer {
public:
    struct Move {
        std::string label;   // the item to go to, found by its label once the move starts
        std::string window;  // only items in windows whose name holds this
        float x = 0.0f, y = 0.0f;  // the point to go to without a label, else the offset from the item's centre
        float seconds = -1.0f;     // the glide; below 0 it follows the distance
        int button = 0;            // 0 left, 1 middle, 2 right
        int clicks = 0;
        float hold = 0.0f;         // seconds the button stays down before a drag or as a long press
        bool drag = false;
        std::string drag_label;       // drag onto this item, else by drag_x, drag_y
        float drag_x = 0.0f, drag_y = 0.0f;
        float drag_seconds = -1.0f;
        int wheel = 0;     // notches turned at the target, + away from the user
        float wait = 0.0f;  // a pause after everything
    };

    bool On() const { return on_; }
    // Shows the pointer at x, y, or hides it and drops what is queued.
    void Show(bool on, float x, float y);
    void Queue(const Move& move);
    bool Busy() const { return !steps_.empty(); }
    size_t Queued() const { return steps_.size(); }
    const std::string& Error() const { return error_; }
    float X() const { return x_; }
    float Y() const { return y_; }
    bool Pressed() const { return down_ >= 0; }

    // Once a frame after the window's events: plays the steps and swaps the real mouse events for
    // its own.
    void Advance(std::vector<render::window::WindowEvent>* events);
    // The same frame's mouse state for ImGui, after the platform backend's NewFrame.
    void FeedImGui(ImGuiIO& io) const;
    // True when this frame must record ImGui's items for a label to be found.
    bool Finding() const;
    void Found(const std::vector<render::imgui::HookedItem>& items);
    void Draw(ImDrawList* list, float scale) const;

private:
    using Clock = std::chrono::steady_clock;
    struct Step {
        enum class Kind { Find, Glide, Button, Wheel, Wait } kind = Kind::Wait;
        std::string label, window;
        float x = 0.0f, y = 0.0f;
        bool relative = false;  // a glide by x, y from where the pointer is
        float seconds = 0.0f;
        int button = 0;
        bool down = false;
        int notches = 0;
    };

    void Push(Step step) { steps_.push_back(std::move(step)); }
    void Fail(std::string why);

    bool on_ = false;
    bool closing_ = false;  // hiding once the held button is released
    bool placed_ = false;   // moved by Show since the last frame
    float x_ = 0.0f, y_ = 0.0f;
    int down_ = -1;  // the button held
    std::deque<Step> steps_;
    bool started_ = false;  // the front step has begun
    Clock::time_point began_;
    float from_x_ = 0.0f, from_y_ = 0.0f, to_x_ = 0.0f, to_y_ = 0.0f, glide_ = 0.0f;
    std::string error_;
    // What this frame did, for ImGui.
    bool active_ = false;
    bool moved_ = false;
    int changed_button_ = -1;
    int wheel_ = 0;
    Clock::time_point pressed_at_;
};

}  // namespace nuka::runtime::app::viewer
