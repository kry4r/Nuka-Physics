// nui controls, fields and display widgets.

#include "runtime/app/viewer/nui/nui_internal.hpp"

#include "implot.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <iterator>

namespace nuka::nui {

using namespace detail;

namespace {

json::Value ToJson(float v) { return json::Value::Float(v); }
json::Value ToJson(int v) { return json::Value::Int(v); }
json::Value ToJson(bool v) { return json::Value::Bool(v); }
json::Value ToJson(const std::string& v) { return json::Value::Str(v); }
template <size_t N>
json::Value ToJson(const std::array<float, N>& v) {
    json::Value out = json::Value::Array();
    for (const float x : v) out.PushBack(json::Value::Float(x));
    return out;
}

void FromJson(const json::Value& j, float& out) { out = j.AsFloat(); }
void FromJson(const json::Value& j, int& out) { out = static_cast<int>(j.AsInt()); }
void FromJson(const json::Value& j, bool& out) { out = j.AsBool(); }
void FromJson(const json::Value& j, std::string& out) { out = j.AsString(); }
template <size_t N>
void FromJson(const json::Value& j, std::array<float, N>& out) {
    for (size_t i = 0; i < N && i < j.Size(); ++i) out[i] = j.Elements()[i].AsFloat();
}

// A printf format with the unit appended; a '%' in the unit is escaped.
std::string WithUnit(const std::string& format, const std::string& unit) {
    if (unit.empty()) return format;
    std::string out = format + " ";
    for (const char c : unit) {
        out += c;
        if (c == '%') out += '%';
    }
    return out;
}

int ResizeString(ImGuiInputTextCallbackData* data) {
    if (data->EventFlag == ImGuiInputTextFlags_CallbackResize) {
        auto* text = static_cast<std::string*>(data->UserData);
        text->resize(static_cast<size_t>(data->BufTextLen));
        data->Buf = text->data();
    }
    return 0;
}

bool InputString(const char* id, const char* hint, std::string& text, ImGuiInputTextFlags flags) {
    return ImGui::InputTextWithHint(id, hint, text.data(), text.capacity() + 1u, flags | ImGuiInputTextFlags_CallbackResize,
                                    ResizeString, &text);
}

// Three drags in a row, each marked with its axis color; `unit` follows them once.
bool AxisDrags(float* v, float speed, const char* format, const std::string& unit) {
    const float s = Scale();
    const float gap = theme::kSpace * s;
    const float unit_w = unit.empty() ? 0.0f : ImGui::CalcTextSize(unit.c_str()).x + gap;
    const float w = (ImGui::GetContentRegionAvail().x - 2.0f * gap - unit_w) / 3.0f;
    const ImVec4 axes[3] = {theme::kAxisX, theme::kAxisY, theme::kAxisZ};
    ImDrawList* list = ImGui::GetWindowDrawList();
    bool changed = false;
    for (int i = 0; i < 3; ++i) {
        if (i > 0) ImGui::SameLine(0.0f, gap);
        ImGui::PushID(i);
        ImGui::SetNextItemWidth(w);
        if (ImGui::DragFloat("##axis", &v[i], speed, 0.0f, 0.0f, format)) changed = true;
        const ImVec2 a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
        list->AddRectFilled(a, ImVec2(a.x + 2.0f * s, b.y), U32(axes[i]), theme::kRadius * s, ImDrawFlags_RoundCornersLeft);
        ImGui::PopID();
    }
    if (!unit.empty()) {
        ImGui::SameLine(0.0f, gap);
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(theme::kFgDim, "%s", unit.c_str());
    }
    return changed;
}

constexpr float kDegrees = 57.29577951308232f;

// Intrinsic XYZ: R = Rx(a) Ry(b) Rz(c).
void QuatToEuler(const Quat& q, float* deg) {
    const float x = q[0], y = q[1], z = q[2], w = q[3];
    const float r02 = 2.0f * (x * z + w * y);
    const float b = std::asin(std::clamp(r02, -1.0f, 1.0f));
    float a = 0.0f, c = 0.0f;
    if (std::fabs(r02) < 0.9999f) {
        a = std::atan2(-2.0f * (y * z - w * x), 1.0f - 2.0f * (x * x + y * y));
        c = std::atan2(-2.0f * (x * y - w * z), 1.0f - 2.0f * (y * y + z * z));
    } else {
        a = std::atan2(2.0f * (y * z + w * x), 1.0f - 2.0f * (x * x + z * z));
    }
    deg[0] = a * kDegrees;
    deg[1] = b * kDegrees;
    deg[2] = c * kDegrees;
}

Quat Multiply(const Quat& p, const Quat& q) {
    return {p[3] * q[0] + p[0] * q[3] + p[1] * q[2] - p[2] * q[1], p[3] * q[1] - p[0] * q[2] + p[1] * q[3] + p[2] * q[0],
            p[3] * q[2] + p[0] * q[1] - p[1] * q[0] + p[2] * q[3], p[3] * q[3] - p[0] * q[0] - p[1] * q[1] - p[2] * q[2]};
}

Quat EulerToQuat(const float* deg) {
    const float a = deg[0] / kDegrees * 0.5f, b = deg[1] / kDegrees * 0.5f, c = deg[2] / kDegrees * 0.5f;
    const Quat qx = {std::sin(a), 0.0f, 0.0f, std::cos(a)};
    const Quat qy = {0.0f, std::sin(b), 0.0f, std::cos(b)};
    const Quat qz = {0.0f, 0.0f, std::sin(c), std::cos(c)};
    return Multiply(Multiply(qx, qy), qz);
}

ImVec4 LineTone(std::string_view line) {
    const std::string lower = Lower(line);
    if (lower.find("error") != std::string::npos || lower.find("traceback") != std::string::npos ||
        lower.find("failed") != std::string::npos)
        return theme::kErr;
    if (lower.find("warn") != std::string::npos) return theme::kWarn;
    if (line.rfind(">>>", 0) == 0) return theme::kAccent;
    return theme::kFg;
}

}  // namespace

// ---- basic controls -----------------------------------------------------------------------

void Label::SetProps(const json::Value& props) {
    Node::SetProps(props);
    Read(props, "text", text);
    Read(props, "tone", tone);
    Read(props, "style", style);
    Read(props, "wrap", wrap);
}

void Label::DrawSelf() {
    std::string dynamic;
    const std::string* shown = &Tr(text);
    if (text_fn) {
        dynamic = text_fn();
        shown = &dynamic;
    }
    PushTextStyle(style);
    ImGui::PushStyleColor(ImGuiCol_Text, ToneColor(tone_fn ? tone_fn() : tone));
    if (style == TextStyle::Label) {
        const std::string upper = Upper(*shown);
        ImGui::TextUnformatted(upper.c_str());
    } else if (wrap) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextUnformatted(shown->data(), shown->data() + shown->size());
        ImGui::PopTextWrapPos();
    } else {
        ImGui::TextUnformatted(shown->data(), shown->data() + shown->size());
    }
    ImGui::PopStyleColor();
    PopTextStyle();
}

void Stat::SetProps(const json::Value& props) {
    Node::SetProps(props);
    Read(props, "label", label);
    Read(props, "value", value);
    Read(props, "tone", tone);
}

void Stat::DrawSelf() {
    const std::string shown = value_fn ? value_fn() : Tr(value);
    const float right = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
    ImGui::PushStyleColor(ImGuiCol_Text, theme::kFgMuted);
    ImGui::TextUnformatted(Tr(label).c_str());
    ImGui::PopStyleColor();
    PushTextStyle(TextStyle::Mono);
    const float w = ImGui::CalcTextSize(shown.c_str()).x;
    // Screen space: SameLine offsets would count a surrounding group's indent twice.
    ImGui::SameLine(0.0f, theme::kSpace * Scale());
    const ImVec2 at = ImGui::GetCursorScreenPos();
    ImGui::SetCursorScreenPos(ImVec2(std::max(right - w, at.x), at.y));
    ImGui::PushStyleColor(ImGuiCol_Text, ToneColor(tone_fn ? tone_fn() : tone));
    ImGui::TextUnformatted(shown.c_str());
    ImGui::PopStyleColor();
    PopTextStyle();
}

Button& Button::Action(std::string action_id, json::Value action_args) {
    action = std::move(action_id);
    args = std::move(action_args);
    return *this;
}

void Button::SetProps(const json::Value& props) {
    Node::SetProps(props);
    Read(props, "label", label);
    ReadIcon(props, "icon", icon);
    Read(props, "action", action);
    if (const json::Value* v = props.Find("args")) args = *v;
    Read(props, "primary", primary);
}

void Button::DrawSelf() {
    std::string caption = label;
    bool can_run = true;
    if (!action.empty()) {
        const ActionHooks& hooks = Host::Current()->actions;
        if (caption.empty()) caption = hooks.label(action);
        can_run = hooks.enabled(action);
    }
    const std::string text = WithIcon(icon, Tr(caption));
    if (!can_run) ImGui::BeginDisabled();
    if (primary) {
        ImGui::PushStyleColor(ImGuiCol_Button, theme::kAccent);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, theme::Hex(0xB8E566));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, theme::kAccentDim);
        ImGui::PushStyleColor(ImGuiCol_Text, theme::kAccentInk);
    }
    const bool clicked = ImGui::Button(text.c_str(), ImVec2(ItemWidth(0.0f), 0.0f));
    if (primary) ImGui::PopStyleColor(4);
    if (!action.empty() && tooltip.empty() && ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip | ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("%s", ActionTip(action).c_str());
    if (!can_run) ImGui::EndDisabled();
    if (!clicked) return;
    if (on_click) on_click();
    if (!action.empty()) RunAction(action, args);
    Emit("click");
}

IconButton& IconButton::Action(std::string action_id, json::Value action_args) {
    action = std::move(action_id);
    args = std::move(action_args);
    return *this;
}

void IconButton::SetProps(const json::Value& props) {
    Node::SetProps(props);
    ReadIcon(props, "icon", glyph);
    ReadIcon(props, "checked_icon", glyph_checked);
    Read(props, "action", action);
    if (const json::Value* v = props.Find("args")) args = *v;
    Read(props, "tone", tone);
}

void IconButton::DrawSelf() {
    bool checked = checked_fn ? checked_fn() : false;
    bool can_run = true;
    if (!action.empty()) {
        const ActionHooks& hooks = Host::Current()->actions;
        const int state = hooks.checked(action);
        if (state >= 0) checked = state > 0;
        can_run = hooks.enabled(action);
    }
    const std::string& shown = checked && !glyph_checked.empty() ? glyph_checked : glyph;
    if (!can_run) ImGui::BeginDisabled();
    const bool clicked = draw::IconButton("##icon", shown.c_str(), checked, nullptr, tone);
    if (!action.empty() && tooltip.empty() && ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip | ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("%s", ActionTip(action).c_str());
    if (!can_run) ImGui::EndDisabled();
    if (!clicked) return;
    if (on_click) on_click();
    if (!action.empty()) RunAction(action, args);
    Emit("click");
}

ToggleButton& ToggleButton::Action(std::string action_id) {
    action = std::move(action_id);
    return *this;
}

void ToggleButton::SetProps(const json::Value& props) {
    Node::SetProps(props);
    Read(props, "label", label);
    Read(props, "action", action);
    if (const json::Value* v = props.Find("value")) model->Set(v->AsBool());
}

json::Value ToggleButton::GetValue() const { return json::Value::Bool(model->Get()); }

void ToggleButton::DrawSelf() {
    const float s = Scale();
    const float h = ImGui::GetFrameHeight();
    bool on = action.empty() ? model->Get() : Host::Current()->actions.checked(action) > 0;
    const ImVec2 track(28.0f * s, 14.0f * s);
    const std::string& caption = Tr(label);
    const float text_w = caption.empty() ? 0.0f : ImGui::CalcTextSize(caption.c_str()).x + 8.0f * s;
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const bool clicked = ImGui::InvisibleButton("##toggle", ImVec2(track.x + text_w, h));
    const bool hovered = ImGui::IsItemHovered();
    if (clicked) {
        on = !on;
        if (!action.empty()) {
            json::Value args = json::Value::Object();
            args.Set("on", json::Value::Bool(on));
            RunAction(action, args);
        } else {
            model->Set(on);
        }
        if (on_change) on_change(on);
        Emit("change", json::Value::Bool(on));
    }
    ImDrawList* list = ImGui::GetWindowDrawList();
    const ImVec2 a(pos.x, pos.y + (h - track.y) * 0.5f), b(a.x + track.x, a.y + track.y);
    list->AddRectFilled(a, b, U32(on ? theme::kAccent : hovered ? theme::kLineHi : theme::kBg3), track.y * 0.5f);
    const ImVec2 knob(on ? b.x - track.y * 0.5f : a.x + track.y * 0.5f, (a.y + b.y) * 0.5f);
    list->AddCircleFilled(knob, track.y * 0.5f - 2.0f * s, U32(on ? theme::kAccentInk : theme::kFgMuted));
    if (!caption.empty())
        list->AddText(ImVec2(b.x + 8.0f * s, pos.y + (h - ImGui::GetFontSize()) * 0.5f), U32(theme::kFg), caption.c_str());
}

void Segmented::SetProps(const json::Value& props) {
    Node::SetProps(props);
    Read(props, "options", options);
    if (const json::Value* v = props.Find("icons")) {
        icons.clear();
        for (const json::Value& e : v->Elements()) icons.push_back(IconOf(e.AsString()));
    }
    Read(props, "tips", tips);
    if (const json::Value* v = props.Find("value")) model->Set(static_cast<int>(v->AsInt()));
}

json::Value Segmented::GetValue() const { return json::Value::Int(model->Get()); }

void Segmented::DrawSelf() {
    const float s = Scale();
    const float h = ImGui::GetFrameHeight();
    const int value = model->Get();
    ImDrawList* list = ImGui::GetWindowDrawList();
    const ImVec2 start = ImGui::GetCursorScreenPos();
    ImGui::BeginGroup();
    for (size_t i = 0; i < options.size(); ++i) {
        const std::string caption = WithIcon(i < icons.size() ? icons[i] : std::string(), Tr(options[i]));
        const ImVec2 text = ImGui::CalcTextSize(caption.c_str());
        const ImVec2 size(text.x + 16.0f * s, h);
        if (i > 0) ImGui::SameLine(0.0f, 0.0f);
        ImGui::PushID(static_cast<int>(i));
        const bool clicked = ImGui::InvisibleButton("##segment", size);
        ImGui::PopID();
        const bool hovered = ImGui::IsItemHovered();
        const bool selected = static_cast<int>(i) == value;
        const ImVec2 a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
        if (selected) {
            list->AddRectFilled(a, b, U32(theme::Alpha(theme::kAccent, 0.14f)));
            list->AddLine(ImVec2(a.x + 4.0f * s, b.y - 1.0f * s), ImVec2(b.x - 4.0f * s, b.y - 1.0f * s), U32(theme::kAccent), 2.0f * s);
        } else if (hovered) {
            list->AddRectFilled(a, b, U32(theme::kBg2));
        }
        list->AddText(ImVec2(a.x + (size.x - text.x) * 0.5f, a.y + (h - text.y) * 0.5f),
                      U32(selected ? theme::kAccent : hovered ? theme::kFg : theme::kFgMuted), caption.c_str());
        if (i < tips.size() && !tips[i].empty() && ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip))
            ImGui::SetTooltip("%s", Tr(tips[i]).c_str());
        if (clicked && !selected) {
            model->Set(static_cast<int>(i));
            if (on_change) on_change(static_cast<int>(i));
            Emit("change", json::Value::Int(static_cast<int64_t>(i)));
        }
    }
    ImGui::EndGroup();
    list->AddRect(start, ImGui::GetItemRectMax(), U32(theme::kLine), theme::kRadius * s);
}

void ComboBox::SetProps(const json::Value& props) {
    Node::SetProps(props);
    Read(props, "label", label);
    Read(props, "options", options);
    if (const json::Value* v = props.Find("value")) model->Set(static_cast<int>(v->AsInt()));
}

json::Value ComboBox::GetValue() const { return json::Value::Int(model->Get()); }

void ComboBox::DrawSelf() {
    if (!label.empty()) {
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(theme::kFgMuted, "%s", Tr(label).c_str());
        ImGui::SameLine();
    }
    const int value = model->Get();
    const char* preview = value >= 0 && value < static_cast<int>(options.size()) ? Tr(options[static_cast<size_t>(value)]).c_str() : "";
    ImGui::SetNextItemWidth(ItemWidth(160.0f * Scale()));
    if (!ImGui::BeginCombo("##combo", preview)) return;
    for (size_t i = 0; i < options.size(); ++i) {
        const bool selected = static_cast<int>(i) == value;
        ImGui::PushID(static_cast<int>(i));
        const bool clicked = ImGui::Selectable(Tr(options[i]).c_str(), selected);
        ImGui::PopID();
        if (clicked && !selected) {
            model->Set(static_cast<int>(i));
            if (on_change) on_change(static_cast<int>(i));
            Emit("change", json::Value::Int(static_cast<int64_t>(i)));
        }
        if (selected) ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
}

// ---- fields -------------------------------------------------------------------------------

template <class T>
void Field<T>::SetProps(const json::Value& props) {
    Node::SetProps(props);
    Read(props, "label", label);
    Read(props, "unit", unit);
    if (const json::Value* v = props.Find("tier"))
        tier = v->AsString() == "live" ? Tier::Live : v->AsString() == "rebuild" ? Tier::Rebuild : Tier::None;
    if (const json::Value* v = props.Find("value")) {
        T value{};
        FromJson(*v, value);
        model->Set(value);
    }
}

template <class T>
json::Value Field<T>::GetValue() const {
    return ToJson(model->Get());
}

// The editor group reports activation and the end of the gesture as one item.
template <class T>
void Field<T>::DrawSelf() {
    FieldLabel(label, tier, InGrid(parent_));
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::BeginGroup();
    T value = model->Get();
    const T prior = value;
    const bool changed = Edit(value);
    ImGui::EndGroup();
    if (ImGui::IsItemActivated()) before_ = prior;
    if (changed) {
        model->Set(value);
        if (on_change) on_change(value);
        Emit("change", ToJson(value));
    }
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        const T after = model->Get();
        if (on_commit) on_commit(before_, after);
        Emit("commit", ToJson(after));
    }
}

template class Field<float>;
template class Field<int>;
template class Field<bool>;
template class Field<Vec3>;
template class Field<Quat>;
template class Field<std::string>;

void FloatField::SetProps(const json::Value& props) {
    Field<float>::SetProps(props);
    Read(props, "speed", speed);
    Read(props, "min", min);
    Read(props, "max", max);
    Read(props, "slider", slider);
    Read(props, "format", format);
}

bool FloatField::Edit(float& value) {
    const std::string fmt = WithUnit(format, unit);
    if (slider) return ImGui::SliderFloat("##value", &value, min, max, fmt.c_str());
    return ImGui::DragFloat("##value", &value, speed, min, max, fmt.c_str());
}

void IntField::SetProps(const json::Value& props) {
    Field<int>::SetProps(props);
    Read(props, "min", min);
    Read(props, "max", max);
    Read(props, "speed", speed);
}

bool IntField::Edit(int& value) {
    const std::string fmt = WithUnit("%d", unit);
    return ImGui::DragInt("##value", &value, speed, min, max, fmt.c_str());
}

bool BoolField::Edit(bool& value) { return ImGui::Checkbox("##value", &value); }

void Vec3Field::SetProps(const json::Value& props) {
    Field<Vec3>::SetProps(props);
    Read(props, "speed", speed);
    Read(props, "format", format);
}

bool Vec3Field::Edit(Vec3& value) { return AxisDrags(value.data(), speed, format.c_str(), unit); }

bool QuatField::Edit(Quat& value) {
    float euler[3];
    QuatToEuler(value, euler);
    if (!AxisDrags(euler, 0.5f, "%.1f", "\xC2\xB0")) return false;
    value = EulerToQuat(euler);
    return true;
}

void ColorField::SetProps(const json::Value& props) {
    Field<Color>::SetProps(props);
    Read(props, "alpha", alpha);
    Read(props, "hdr", hdr);
}

bool ColorField::Edit(Color& value) {
    ImGuiColorEditFlags flags = alpha ? ImGuiColorEditFlags_AlphaBar : ImGuiColorEditFlags_NoAlpha;
    if (hdr) flags |= ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float;
    return alpha ? ImGui::ColorEdit4("##value", value.data(), flags) : ImGui::ColorEdit3("##value", value.data(), flags);
}

void ColorField::DrawSelf() {
    const Color prior = model->Get();
    Field<Color>::DrawSelf();
    ImGui::PushID("##value");
    const bool open = ImGui::IsPopupOpen("picker");
    ImGui::PopID();
    if (open && !picking_) pick_before_ = prior;
    if (open && model->Get() != prior) picked_ = true;
    if (!open && picking_ && picked_) {
        const Color after = model->Get();
        if (on_commit) on_commit(pick_before_, after);
        Emit("commit", ToJson(after));
    }
    if (!open) picked_ = false;
    picking_ = open;
}

void StringField::SetProps(const json::Value& props) {
    Field<std::string>::SetProps(props);
    Read(props, "hint", hint);
}

bool StringField::Edit(std::string& value) { return InputString("##value", Tr(hint).c_str(), value, 0); }

bool PathField::Edit(std::string& value) {
    const float s = Scale();
    const float button = ImGui::GetFrameHeight() + 2.0f * s;
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - button - theme::kSpace * s);
    const bool changed = InputString("##value", "", value, 0);
    ImGui::SameLine(0.0f, theme::kSpace * s);
    if (draw::IconButton("##browse", icon::kOpen, false, Tr("Browse"))) {
        if (on_browse) on_browse();
        Emit("browse");
    }
    return changed;
}

void TextBufferField::DrawSelf() {
    FieldLabel(label, Tier::None, InGrid(parent_));
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::InputTextWithHint("##text", Tr(hint).c_str(), buffer_, capacity_, ImGuiInputTextFlags_EnterReturnsTrue)) {
        if (on_enter) on_enter();
        Emit("enter", json::Value::Str(buffer_));
    }
}

void Slider::SetProps(const json::Value& props) {
    Node::SetProps(props);
    Read(props, "min", min);
    Read(props, "max", max);
    Read(props, "format", format);
    if (const json::Value* v = props.Find("value")) model->Set(v->AsFloat());
}

json::Value Slider::GetValue() const { return json::Value::Float(model->Get()); }

void Slider::DrawSelf() {
    const float prior = model->Get();
    float value = prior;
    ImGui::SetNextItemWidth(ItemWidth(-FLT_MIN));
    const bool changed = ImGui::SliderFloat("##slider", &value, min, max, format.c_str());
    if (ImGui::IsItemActivated()) before_ = prior;
    if (changed) {
        model->Set(value);
        if (on_change) on_change(value);
        Emit("change", json::Value::Float(value));
    }
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        if (on_commit) on_commit(before_, model->Get());
        Emit("commit", json::Value::Float(model->Get()));
    }
}

void Drag::SetProps(const json::Value& props) {
    Node::SetProps(props);
    Read(props, "speed", speed);
    Read(props, "min", min);
    Read(props, "max", max);
    Read(props, "format", format);
    if (const json::Value* v = props.Find("value")) model->Set(v->AsFloat());
}

json::Value Drag::GetValue() const { return json::Value::Float(model->Get()); }

void Drag::DrawSelf() {
    float value = model->Get();
    ImGui::SetNextItemWidth(ItemWidth(-FLT_MIN));
    if (ImGui::DragFloat("##drag", &value, speed, min, max, format.c_str())) {
        model->Set(value);
        if (on_change) on_change(value);
        Emit("change", json::Value::Float(value));
    }
    if (ImGui::IsItemDeactivatedAfterEdit()) Emit("commit", json::Value::Float(model->Get()));
}

// ---- display ------------------------------------------------------------------------------

void Plot::SetProps(const json::Value& props) {
    Node::SetProps(props);
    Read(props, "height", height);
    Read(props, "sparkline", sparkline);
    Read(props, "y_min", y_min);
    Read(props, "y_max", y_max);
    if (const json::Value* v = props.Find("series")) {
        series.clear();
        for (const json::Value& e : v->Elements()) {
            Series out;
            Read(e, "name", out.name);
            if (const json::Value* xs = e.Find("x"))
                for (const json::Value& x : xs->Elements()) out.x.push_back(x.AsFloat());
            if (const json::Value* ys = e.Find("y"))
                for (const json::Value& y : ys->Elements()) out.y.push_back(y.AsFloat());
            series.push_back(std::move(out));
        }
    }
}

void Plot::Push(size_t index, float x, float y, size_t keep) {
    if (series.size() <= index) series.resize(index + 1u);
    Series& target = series[index];
    target.x.push_back(x);
    target.y.push_back(y);
    if (target.x.size() > keep) {
        const auto drop = static_cast<std::ptrdiff_t>(target.x.size() - keep);
        target.x.erase(target.x.begin(), target.x.begin() + drop);
        target.y.erase(target.y.begin(), target.y.begin() + drop);
    }
}

void Plot::DrawSelf() {
    const float s = Scale();
    const ImVec4 colors[] = {theme::kAccent, theme::kInfo, theme::kWarn, theme::kErr, theme::kOk, theme::kAxisZ};
    const bool fixed = y_max > y_min;
    ImPlotFlags flags = ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect | ImPlotFlags_NoFrame;
    ImPlotAxisFlags x_flags = ImPlotAxisFlags_AutoFit;
    ImPlotAxisFlags y_flags = fixed ? ImPlotAxisFlags_None : ImPlotAxisFlags_AutoFit;
    if (sparkline) {
        flags = ImPlotFlags_CanvasOnly | ImPlotFlags_NoInputs | ImPlotFlags_NoFrame;
        x_flags |= ImPlotAxisFlags_NoDecorations;
        y_flags |= ImPlotAxisFlags_NoDecorations;
    }
    ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, sparkline ? ImVec2(0.0f, 2.0f * s) : ImVec2(theme::kSpace * s, theme::kSpace * s));
    if (ImPlot::BeginPlot("##plot", ImVec2(ItemWidth(-FLT_MIN), height * s), flags)) {
        ImPlot::SetupAxes(nullptr, nullptr, x_flags, y_flags);
        if (fixed) ImPlot::SetupAxisLimits(ImAxis_Y1, y_min, y_max, ImPlotCond_Always);
        for (size_t i = 0; i < series.size(); ++i) {
            const Series& line = series[i];
            const int count = static_cast<int>(std::min(line.x.size(), line.y.size()));
            const ImVec4 color = colors[i % std::size(colors)];
            ImPlotSpec spec;
            spec.LineColor = color;
            spec.LineWeight = 1.5f * s;
            if (sparkline && count > 0) {
                const float base = fixed ? y_min : *std::min_element(line.y.begin(), line.y.end());
                spec.FillColor = theme::Alpha(color, 0.16f);
                ImPlot::PlotShaded(line.name.c_str(), line.x.data(), line.y.data(), count, static_cast<double>(base), spec);
            }
            ImPlot::PlotLine(line.name.c_str(), line.x.data(), line.y.data(), count, spec);
        }
        ImPlot::EndPlot();
    }
    ImPlot::PopStyleVar();
}

void Image::DrawSelf() {
    const float s = Scale();
    ImGui::Image(texture, ImVec2(size.x * s, size.y * s), uv0, uv1);
}

void CodeEditor::SetProps(const json::Value& props) {
    Node::SetProps(props);
    Read(props, "height", height);
    if (const json::Value* v = props.Find("text")) {
        const std::string& text = v->AsString();
        own_.assign(std::max<size_t>(64u * 1024u, text.size() + 1u), '\0');
        std::memcpy(own_.data(), text.data(), text.size());
    }
}

json::Value CodeEditor::GetValue() const {
    if (buffer_ != nullptr) return json::Value::Str(buffer_);
    return json::Value::Str(own_.empty() ? std::string() : std::string(own_.data()));
}

void CodeEditor::DrawSelf() {
    if (buffer_ == nullptr && own_.empty()) own_.assign(64u * 1024u, '\0');
    char* buf = buffer_ != nullptr ? buffer_ : own_.data();
    const size_t cap = buffer_ != nullptr ? capacity_ : own_.size();
    const float s = Scale();
    const float avail = ImGui::GetContentRegionAvail().y;
    const float h = height > 0.0f ? height * s : std::max(height < 0.0f ? avail : avail * 0.5f, 80.0f * s);
    PushTextStyle(TextStyle::Mono);
    ImGui::PushStyleColor(ImGuiCol_FrameBg, theme::kBg0);
    ImGui::InputTextMultiline("##code", buf, cap, ImVec2(ItemWidth(-FLT_MIN), h), ImGuiInputTextFlags_AllowTabInput);
    ImGui::PopStyleColor();
    PopTextStyle();
    // Ctrl+Enter ends the edit without a newline; it runs while this panel has focus.
    const ImGuiIO& io = ImGui::GetIO();
    if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) && io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Enter, false)) {
        if (on_run) on_run();
        Emit("run", json::Value::Str(buf));
    }
}

void LogView::SetProps(const json::Value& props) {
    Node::SetProps(props);
    Read(props, "height", height);
    Read(props, "text", own_);
    if (const json::Value* v = props.Find("append")) own_ += v->AsString();
}

void LogView::DrawSelf() {
    const std::string& text = text_ != nullptr ? *text_ : own_;
    if (indexed_size_ == ~size_t(0) || text.size() < indexed_size_) {
        lines_.clear();
        indexed_size_ = 0u;
    }
    if (text.size() != indexed_size_) {
        size_t start = 0u;
        if (!lines_.empty()) {
            start = lines_.back().first;
            lines_.pop_back();
        }
        while (start < text.size()) {
            const size_t nl = text.find('\n', start);
            const size_t end = nl == std::string::npos ? text.size() : nl;
            lines_.emplace_back(start, end);
            start = end + 1u;
        }
        indexed_size_ = text.size();
    }
    std::vector<size_t> shown;
    const std::string needle = filter ? Lower(filter->Get()) : std::string();
    if (!needle.empty())
        for (size_t i = 0; i < lines_.size(); ++i)
            if (Lower(std::string_view(text).substr(lines_[i].first, lines_[i].second - lines_[i].first)).find(needle) != std::string::npos)
                shown.push_back(i);
    const size_t count = needle.empty() ? lines_.size() : shown.size();
    ImGui::PushStyleColor(ImGuiCol_ChildBg, theme::kBg0);
    if (ImGui::BeginChild("##log", ImVec2(0.0f, height > 0.0f ? height * Scale() : 0.0f), ImGuiChildFlags_Borders,
                          ImGuiWindowFlags_HorizontalScrollbar)) {
        const bool pinned = ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 1.0f;
        PushTextStyle(TextStyle::Mono);
        if (count == 0u) ImGui::TextColored(theme::kFgDim, "%s", Tr(needle.empty() ? "(no output yet)" : "(no match)"));
        ImGuiListClipper clipper;
        clipper.Begin(static_cast<int>(count));
        while (clipper.Step()) {
            for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
                const auto& span = lines_[needle.empty() ? static_cast<size_t>(row) : shown[static_cast<size_t>(row)]];
                const std::string_view line = std::string_view(text).substr(span.first, span.second - span.first);
                ImGui::PushStyleColor(ImGuiCol_Text, LineTone(line));
                ImGui::TextUnformatted(line.data(), line.data() + line.size());
                ImGui::PopStyleColor();
            }
        }
        PopTextStyle();
        if (pinned) ImGui::SetScrollHereY(1.0f);
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();
}

void Badge::SetProps(const json::Value& props) {
    Node::SetProps(props);
    Read(props, "text", text);
    ReadIcon(props, "icon", icon);
    Read(props, "tone", tone);
}

void Badge::DrawSelf() {
    const std::string shown = text_fn ? text_fn() : Tr(text);
    draw::Badge(shown.c_str(), tone_fn ? tone_fn() : tone, icon.c_str());
}

void StatusDot::SetProps(const json::Value& props) {
    Node::SetProps(props);
    Read(props, "label", label);
    Read(props, "tone", tone);
    Read(props, "pulse", pulse);
}

void StatusDot::DrawSelf() {
    draw::StatusDot(tone_fn ? tone_fn() : tone, pulse);
    const std::string shown = label_fn ? label_fn() : Tr(label);
    if (shown.empty()) return;
    ImGui::SameLine(0.0f, theme::kSpace * Scale());
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(theme::kFgMuted, "%s", shown.c_str());
}

void ProgressBar::SetProps(const json::Value& props) {
    Node::SetProps(props);
    Read(props, "label", label);
    if (const json::Value* v = props.Find("value")) fraction->Set(v->AsFloat());
}

json::Value ProgressBar::GetValue() const { return json::Value::Float(fraction->Get()); }

void ProgressBar::DrawSelf() {
    const float s = Scale();
    const float w = assigned_width > 0.0f ? assigned_width : width > 0.0f ? width * s : ImGui::GetContentRegionAvail().x;
    ImGui::BeginGroup();
    if (!label.empty()) ImGui::TextColored(theme::kFgMuted, "%s", Tr(label).c_str());
    draw::ThinProgress(fraction->Get(), w);
    ImGui::EndGroup();
}

void ContextMenu::DrawSelf() {
    if (!ImGui::BeginPopupContextItem("##context")) return;
    DrawChildren();
    ImGui::EndPopup();
}

void Menu::DrawSelf() {
    const std::string name = Tr(label) + "###" + label;
    if (!ImGui::BeginMenu(name.c_str())) return;
    DrawChildren();
    ImGui::EndMenu();
}

MenuItem& MenuItem::Action(std::string action_id, json::Value action_args) {
    action = std::move(action_id);
    args = std::move(action_args);
    return *this;
}

void MenuItem::SetProps(const json::Value& props) {
    Node::SetProps(props);
    Read(props, "label", label);
    Read(props, "action", action);
    if (const json::Value* v = props.Find("args")) args = *v;
}

void MenuItem::DrawSelf() {
    std::string caption = label;
    std::string keys;
    bool can_run = true;
    bool checked = false;
    if (!action.empty()) {
        const ActionHooks& hooks = Host::Current()->actions;
        if (caption.empty()) caption = hooks.label(action);
        keys = hooks.shortcut(action);
        can_run = hooks.enabled(action);
        checked = hooks.checked(action) > 0;
    }
    if (!ImGui::MenuItem(Tr(caption).c_str(), keys.empty() ? nullptr : keys.c_str(), checked, can_run)) return;
    if (on_click) on_click();
    if (!action.empty()) RunAction(action, args);
    Emit("click");
}

void Tooltip::DrawSelf() {
    if (!ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip) || !ImGui::BeginTooltip()) return;
    DrawChildren();
    ImGui::EndTooltip();
}

}  // namespace nuka::nui
