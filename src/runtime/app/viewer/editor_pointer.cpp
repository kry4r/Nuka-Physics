#include "runtime/app/viewer/editor_pointer.hpp"

#include "imgui.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <utility>

namespace nuka::runtime::app::viewer {
namespace {

using render::window::WindowEvent;

constexpr float kClickSeconds = 0.09f;   // a click's press
constexpr float kAfterClick = 0.15f;
constexpr float kRippleSeconds = 0.45f;

// The label as shown: before "##", without icon glyphs (the private use areas) or outer spaces.
std::string Shown(const std::string& label) {
    const std::string text = label.substr(0, label.find("##"));
    std::string out;
    for (size_t i = 0; i < text.size();) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        const size_t n = c < 0x80u ? 1u : c < 0xE0u ? 2u : c < 0xF0u ? 3u : 4u;
        uint32_t cp = n == 1u ? c : n == 2u ? (c & 0x1Fu) : n == 3u ? (c & 0x0Fu) : (c & 0x07u);
        for (size_t k = 1u; k < n && i + k < text.size(); ++k)
            cp = (cp << 6u) | (static_cast<unsigned char>(text[i + k]) & 0x3Fu);
        if (!(cp >= 0xE000u && cp <= 0xF8FFu) && cp < 0xF0000u) out.append(text, i, n);
        i += n;
    }
    const size_t first = out.find_first_not_of(' ');
    return first == std::string::npos ? std::string() : out.substr(first, out.find_last_not_of(' ') - first + 1u);
}

// The hidden id after the last "##", without the extra '#' of "###".
std::string IdPart(const std::string& label) {
    const size_t at = label.rfind("##");
    if (at == std::string::npos) return std::string();
    std::string id = label.substr(at + 2u);
    return id.substr(std::min(id.find_first_not_of('#'), id.size()));
}

std::string Lower(std::string text) {
    for (char& c : text) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return text;
}

}  // namespace

void DemoPointer::Show(bool on, float x, float y) {
    if (on) {
        on_ = true;
        closing_ = false;
        x_ = x;
        y_ = y;
        placed_ = true;
        return;
    }
    steps_.clear();
    started_ = false;
    if (down_ >= 0) {
        Push({Step::Kind::Button, {}, {}, 0.0f, 0.0f, false, 0.0f, down_, false});
        closing_ = true;
    } else {
        on_ = false;
    }
}

void DemoPointer::Queue(const Move& move) {
    error_.clear();
    if (!move.label.empty()) Push({Step::Kind::Find, move.label, move.window, move.x, move.y});
    Push({Step::Kind::Glide, {}, {}, move.x, move.y, false, move.seconds});
    const auto button = [&](bool down) {
        Push({Step::Kind::Button, {}, {}, 0.0f, 0.0f, false, 0.0f, move.button, down});
    };
    const auto wait = [&](float seconds) { Push({Step::Kind::Wait, {}, {}, 0.0f, 0.0f, false, seconds}); };
    if (move.drag) {
        button(true);
        wait(std::max(move.hold, 0.15f));
        if (!move.drag_label.empty()) Push({Step::Kind::Find, move.drag_label, move.window});
        Push({Step::Kind::Glide, {}, {}, move.drag_x, move.drag_y, move.drag_label.empty(), move.drag_seconds});
        wait(0.1f);
        button(false);
        wait(kAfterClick);
    } else if (move.hold > 0.0f) {
        button(true);
        wait(move.hold);
        button(false);
        wait(kAfterClick);
    }
    for (int i = 0; i < move.clicks; ++i) {
        button(true);
        wait(i + 1 < move.clicks ? 0.05f : kClickSeconds);
        button(false);
        wait(i + 1 < move.clicks ? 0.05f : kAfterClick);
    }
    for (int i = 0; i < std::abs(move.wheel); ++i) {
        Push({Step::Kind::Wheel, {}, {}, 0.0f, 0.0f, false, 0.0f, 0, false, move.wheel > 0 ? 1 : -1});
        wait(0.06f);
    }
    if (move.wait > 0.0f) wait(move.wait);
}

void DemoPointer::Fail(std::string why) {
    error_ = std::move(why);
    steps_.clear();
    started_ = false;
    if (down_ >= 0) Push({Step::Kind::Button, {}, {}, 0.0f, 0.0f, false, 0.0f, down_, false});
}

void DemoPointer::Advance(std::vector<WindowEvent>* events) {
    active_ = on_;
    moved_ = placed_;
    placed_ = false;
    changed_button_ = -1;
    wheel_ = 0;
    if (!on_) return;
    events->erase(std::remove_if(events->begin(), events->end(),
                                 [](const WindowEvent& e) {
                                     return e.type == WindowEvent::Type::MouseMove ||
                                            e.type == WindowEvent::Type::MouseButton ||
                                            e.type == WindowEvent::Type::Scroll;
                                 }),
                  events->end());
    const Clock::time_point now = Clock::now();
    // At most one press or release a frame, so ImGui sees each one.
    while (!steps_.empty()) {
        Step& s = steps_.front();
        if (!started_) {
            started_ = true;
            began_ = now;
            if (s.kind == Step::Kind::Glide) {
                from_x_ = x_;
                from_y_ = y_;
                to_x_ = s.relative ? x_ + s.x : s.x;
                to_y_ = s.relative ? y_ + s.y : s.y;
                const float distance = std::hypot(to_x_ - from_x_, to_y_ - from_y_);
                glide_ = s.seconds >= 0.0f ? s.seconds : std::clamp(0.25f + distance / 2400.0f, 0.3f, 0.9f);
            }
        }
        if (s.kind == Step::Kind::Find) break;
        const float t = std::chrono::duration<float>(now - began_).count();
        bool done = true;
        bool stop = true;
        switch (s.kind) {
            case Step::Kind::Glide: {
                const float u = glide_ > 0.0f ? std::min(1.0f, t / glide_) : 1.0f;
                const float e = u * u * u * (u * (u * 6.0f - 15.0f) + 10.0f);
                x_ = from_x_ + (to_x_ - from_x_) * e;
                y_ = from_y_ + (to_y_ - from_y_) * e;
                moved_ = true;
                done = u >= 1.0f;
                break;
            }
            case Step::Kind::Button:
                down_ = s.down ? s.button : -1;
                changed_button_ = s.button;
                if (s.down) pressed_at_ = now;
                break;
            case Step::Kind::Wheel:
                wheel_ = s.notches;
                break;
            default:
                done = t >= s.seconds;
                stop = !done;
                break;
        }
        if (done) {
            steps_.pop_front();
            started_ = false;
        }
        if (stop) break;
    }
    if (closing_ && steps_.empty()) {
        on_ = false;
        closing_ = false;
    }
    WindowEvent e;
    e.mouse_x = static_cast<int32_t>(std::lround(x_));
    e.mouse_y = static_cast<int32_t>(std::lround(y_));
    if (moved_) {
        e.type = WindowEvent::Type::MouseMove;
        events->push_back(e);
    }
    if (changed_button_ >= 0) {
        e.type = WindowEvent::Type::MouseButton;
        e.button = static_cast<uint32_t>(changed_button_);
        e.pressed = down_ == changed_button_;
        events->push_back(e);
    }
    if (wheel_ != 0) {
        e.type = WindowEvent::Type::Scroll;
        e.scroll_delta = wheel_;
        events->push_back(e);
    }
}

void DemoPointer::FeedImGui(ImGuiIO& io) const {
    if (!active_) return;
    io.AddMouseSourceEvent(ImGuiMouseSource_Mouse);
    io.AddMousePosEvent(x_, y_);
    if (changed_button_ >= 0) io.AddMouseButtonEvent(changed_button_ == 2 ? 1 : changed_button_ == 1 ? 2 : 0,
                                                     down_ == changed_button_);
    if (wheel_ != 0) io.AddMouseWheelEvent(0.0f, static_cast<float>(wheel_));
}

bool DemoPointer::Finding() const { return on_ && !steps_.empty() && steps_.front().kind == Step::Kind::Find; }

void DemoPointer::Found(const std::vector<render::imgui::HookedItem>& items) {
    const Step find = steps_.front();
    const std::string lower = Lower(find.label);
    const std::string window = Lower(find.window);
    const render::imgui::HookedItem* best = nullptr;
    int best_score = 0;
    for (const render::imgui::HookedItem& item : items) {
        if (!item.visible || (!window.empty() && Lower(item.window).find(window) == std::string::npos)) continue;
        const std::string shown = Shown(item.label);
        const int score = shown == find.label ? 3
                          : IdPart(item.label) == find.label ? 2
                          : !shown.empty() && Lower(shown).find(lower) != std::string::npos ? 1
                                                                                            : 0;
        // Later items in a window draw over earlier ones.
        if (score > best_score || (score == best_score && score > 0 && item.window_order >= best->window_order)) {
            best = &item;
            best_score = score;
        }
    }
    if (best == nullptr) {
        Fail("no visible item is labelled \"" + find.label + "\"" +
             (find.window.empty() ? std::string() : " in a window named like \"" + find.window + "\""));
        return;
    }
    steps_.pop_front();
    started_ = false;
    Step& glide = steps_.front();
    glide.x = 0.5f * (best->min[0] + best->max[0]) + find.x;
    glide.y = 0.5f * (best->min[1] + best->max[1]) + find.y;
    glide.relative = false;
}

void DemoPointer::Draw(ImDrawList* list, float scale) const {
    if (!on_) return;
    const ImVec2 tip(x_, y_);
    const float since = std::chrono::duration<float>(Clock::now() - pressed_at_).count();
    if (since < kRippleSeconds) {
        const float u = since / kRippleSeconds;
        const int alpha = static_cast<int>(230.0f * (1.0f - u));
        list->AddCircle(tip, (8.0f + 26.0f * u) * scale, IM_COL32(166, 196, 82, alpha), 0, 3.0f * scale);
    }
    if (down_ >= 0) list->AddCircleFilled(tip, 10.0f * scale, IM_COL32(166, 196, 82, 120));
    // The usual arrow, its tip at the point.
    static constexpr float kArrow[7][2] = {{0.0f, 0.0f},  {0.0f, 17.0f}, {4.2f, 13.2f}, {7.0f, 19.5f},
                                           {9.8f, 18.3f}, {7.1f, 12.2f}, {12.5f, 12.2f}};
    ImVec2 body[7], shadow[7];
    for (int i = 0; i < 7; ++i) {
        body[i] = ImVec2(x_ + kArrow[i][0] * 1.3f * scale, y_ + kArrow[i][1] * 1.3f * scale);
        shadow[i] = ImVec2(body[i].x + 1.5f * scale, body[i].y + 2.0f * scale);
    }
    list->AddConcavePolyFilled(shadow, 7, IM_COL32(0, 0, 0, 80));
    list->AddConcavePolyFilled(body, 7, IM_COL32(255, 255, 255, 255));
    list->AddPolyline(body, 7, IM_COL32(18, 20, 24, 255), ImDrawFlags_Closed, 1.4f * scale);
}

}  // namespace nuka::runtime::app::viewer
