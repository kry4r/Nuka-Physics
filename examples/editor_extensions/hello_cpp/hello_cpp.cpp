// A sample native editor extension over nuka/editor/plugin.hpp: a panel, an action, a ring
// drawn over the viewport and, when built with Dear ImGui, a panel the plugin paints itself.

#include "nuka/editor/plugin.hpp"

#include <cmath>
#include <cstdint>
#include <string>

#ifndef HELLO_CPP_BUILD
#define HELLO_CPP_BUILD "cmake"
#endif

namespace {

using nuka::editor::Host;
using nuka::editor::Overlay;
using nuka::editor::Quote;
using nuka::editor::Reply;

constexpr float kPi = 3.14159265f;
constexpr int kRingSegments = 48;

// The manifest's [settings] arrive as JSON text; the sample reads one string from it.
std::string SettingString(const std::string& json, const std::string& key, const std::string& fallback) {
    const std::string tag = "\"" + key + "\":\"";
    const size_t at = json.find(tag);
    if (at == std::string::npos) return fallback;
    const size_t start = at + tag.size();
    return json.substr(start, json.find('"', start) - start);
}

class HelloCpp : public nuka::editor::Plugin {
public:
    bool OnStartup(Host& host) override {
        host_ = &host;
        greeting_ = SettingString(host.SettingsJson(), "greeting", "Hello from C++");
        const uint32_t root = host.Panel("hello_cpp", "Hello C++", "right");
        host.Add(root, "label",
                 "{\"text\":" + Quote("A native extension: its panel, action and overlay go away when it is disabled.") +
                     ",\"tone\":\"muted\",\"wrap\":true}");
        counter_ = host.Add(root, "stat", "{\"label\":\"Greetings\",\"value\":\"0\"}");
        const uint32_t row = host.Add(root, "hstack");
        const uint32_t greet = host.Add(row, "button", "{\"label\":\"Greet\",\"icon\":\"file_cpp\",\"primary\":true}");
        host.OnEvent(greet, [this](uint32_t, const std::string&) { Greet(); });
        const uint32_t ring = host.Add(row, "toggle", "{\"label\":\"Ring\",\"value\":true}");
        host.OnEvent(ring, [this](uint32_t, const std::string& event) {
            ring_ = event.find("\"value\":true") != std::string::npos;
        });
        host.AddAction({"hello_cpp.greet", "Greet from C++", "tools", nullptr, "file_cpp"},
                       [this](const std::string& args) {
                           if (args.find("\"fail\":true") != std::string::npos) return Reply::Fail("asked to fail");
                           return Reply::Value(Greet());
                       });
        host.AddOverlay("hello_cpp.ring", [this](const Overlay& overlay) {
            DrawRing(overlay);
            return true;
        });
#ifdef NUKA_EDITOR_PLUGIN_IMGUI
        if (host.Api()->imgui_context != nullptr)
            host.AddImGuiPanel("hello_cpp_imgui", "Hello ImGui", "right", [this] { DrawImGui(); });
#endif
        host.Log(NUKA_NUI_INFO, std::string("hello_cpp, ") + HELLO_CPP_BUILD + " build, settings " + host.SettingsJson());
        return true;
    }

private:
    std::string Greet() {
        ++count_;
        host_->Set(counter_, "{\"value\":\"" + std::to_string(count_) + "\"}");
        host_->Toast(NUKA_NUI_OK, greeting_);
        return "{\"count\":" + std::to_string(count_) + ",\"text\":" + Quote(greeting_) +
               ",\"build\":" + Quote(HELLO_CPP_BUILD) + "}";
    }

    // A ring on the ground around the origin, turning with the simulation tick.
    void DrawRing(const Overlay& overlay) const {
        if (!ring_) return;
        const float phase = static_cast<float>(overlay.Tick() % 2400u) / 2400.0f * 2.0f * kPi;
        const float step = 2.0f * kPi / static_cast<float>(kRingSegments);
        for (int i = 0; i < kRingSegments; ++i) {
            const float a0 = phase + step * static_cast<float>(i);
            const float a1 = a0 + step;
            const float p[3] = {radius_ * std::cos(a0), radius_ * std::sin(a0), 0.0f};
            const float q[3] = {radius_ * std::cos(a1), radius_ * std::sin(a1), 0.0f};
            overlay.Line(p, q, i % 4 == 0 ? 0xC6F432FFu : 0x6E9A2CFFu, 2.0f);
        }
        const float at[3] = {radius_, 0.0f, 0.05f};
        overlay.Text(at, 0xE8ECF0FFu, "hello_cpp");
    }

#ifdef NUKA_EDITOR_PLUGIN_IMGUI
    void DrawImGui() {
        ImGui::TextUnformatted("Painted by the plugin with Dear ImGui " IMGUI_VERSION);
        ImGui::SliderFloat("Ring radius", &radius_, 0.1f, 2.0f);
        ImGui::Text("Greetings: %d", count_);
        if (ImGui::Button("Greet")) Greet();
    }
#endif

    Host* host_ = nullptr;
    std::string greeting_;
    uint32_t counter_ = 0u;
    int count_ = 0;
    bool ring_ = true;
    float radius_ = 0.6f;
};

}  // namespace

NUKA_EDITOR_PLUGIN(HelloCpp)
