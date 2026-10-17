#pragma once
// nui: a retained component tree drawn through Dear ImGui every frame. A panel rebuilds its
// tree only when its key changes; between rebuilds the widgets read their models and draw.

#include "imgui.h"
#include "nuka/nuka_nui.h"
#include "scene/format/json.hpp"

#include <array>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

class TextEditor;

namespace nuka::nui {

namespace json = nuka::scene::json;

using Vec3 = std::array<float, 3>;
using Color = std::array<float, 4>;
using Quat = std::array<float, 4>;  // x, y, z, w

// ---- models -------------------------------------------------------------------------------

template <class T>
class Model {
public:
    virtual ~Model() = default;
    virtual T Get() const = 0;
    virtual void Set(const T& value) = 0;
};
template <class T>
using ModelPtr = std::shared_ptr<Model<T>>;

template <class T>
class ValueModel final : public Model<T> {
public:
    explicit ValueModel(T value = T()) : value_(std::move(value)) {}
    T Get() const override { return value_; }
    void Set(const T& value) override { value_ = value; }

private:
    T value_;
};

template <class T>
class FnModel final : public Model<T> {
public:
    FnModel(std::function<T()> get, std::function<void(const T&)> set) : get_(std::move(get)), set_(std::move(set)) {}
    T Get() const override { return get_(); }
    void Set(const T& value) override {
        if (set_) set_(value);
    }

private:
    std::function<T()> get_;
    std::function<void(const T&)> set_;
};

template <class T>
ModelPtr<T> Value(T value = T()) {
    return std::make_shared<ValueModel<T>>(std::move(value));
}
template <class T>
ModelPtr<T> Bind(std::function<T()> get, std::function<void(const T&)> set = {}) {
    return std::make_shared<FnModel<T>>(std::move(get), std::move(set));
}
template <class T>
ModelPtr<T> Ref(T* target) {
    return Bind<T>([target] { return *target; }, [target](const T& v) { *target = v; });
}

// A tree of item ids; item 0 is the hidden root. Version changes whenever the tree does.
class ItemModel {
public:
    virtual ~ItemModel() = default;
    virtual uint64_t Version() = 0;
    virtual void Children(uint64_t item, std::vector<uint64_t>& out) const = 0;
    virtual std::string_view Label(uint64_t item) const = 0;
    virtual const char* Icon(uint64_t item) const;
    virtual ImVec4 IconColor(uint64_t item) const;
    virtual std::string_view Badge(uint64_t item) const;
};

// ---- look ---------------------------------------------------------------------------------

enum class Tone : uint8_t { Normal, Muted, Dim, Accent, Ok, Warn, Err, Info };
enum class TextStyle : uint8_t { Body, Strong, Label, Mono, Hero };
enum class Tier : uint8_t { None, Live, Rebuild };
enum class Level : uint8_t { Info, Ok, Warn, Err };

ImVec4 ToneColor(Tone tone);
Tone ToneOf(std::string_view name);
TextStyle TextStyleOf(std::string_view name);
// A Phosphor glyph by name ("play", "robot", ...); unknown names pass through as text.
const char* IconOf(const std::string& name);
float Scale();
void PushTextStyle(TextStyle style);
void PopTextStyle();

// ---- strings ------------------------------------------------------------------------------

// UI text is looked up by its English source, and text the active table lacks shows as
// written. Tables change only in ApplyStrings, so a result stays valid for the frame.
const std::string& Tr(const std::string& text);
const char* Tr(const char* text);
// A copy, for threads other than the UI thread.
std::string TrCopy(const std::string& text);

struct LocaleInfo {
    std::string id;
    std::string name;  // in its own language, e.g. "简体中文"
    std::string tag;   // the toolbar's short form, e.g. "中"
};
// "en" first, then every locale a table names.
std::vector<LocaleInfo> Locales();
std::string Locale();
void SetLocale(const std::string& locale);
void AddTranslations(const std::string& locale, const std::string& owner,
                     const std::vector<std::pair<std::string, std::string>>& strings);
void RemoveTranslations(const std::string& owner);
// A TOML table: [meta] locale, name, tag and [strings] "English" = "translated".
void LoadTranslations(const std::filesystem::path& file, const std::string& owner);
void LoadTranslationDir(const std::filesystem::path& dir, const std::string& owner);
// The editor's own tables: next to the module once installed, else in the source tree.
std::filesystem::path BuiltinTranslationDir();
// Applies locale and table changes; the host calls it at the top of each frame.
void ApplyStrings();
// UI text with letters the active locale has no entry for, since the locale last changed.
std::vector<std::string> MissingStrings();

// Immediate helpers the widgets and the editor shell share.
namespace draw {
// A rectangle with its top-left and bottom-right corners cut, the shell's signature shape.
void Chamfer(ImDrawList* list, const ImVec2& min, const ImVec2& max, float cut, ImU32 fill, ImU32 border);
// Draws one icon with its ink centered on `center`; `filled` takes the fill face.
void Glyph(ImDrawList* list, const ImVec2& center, const char* glyph, ImU32 color, bool filled = false,
           float size = 0.0f);
void SectionHeader(const char* label, const char* trailing = nullptr);
void Badge(const char* text, Tone tone, const char* icon = nullptr);
void StatusDot(Tone tone, bool pulse = false);
bool IconButton(const char* id, const char* glyph, bool checked = false, const char* tooltip = nullptr,
                Tone tone = Tone::Normal);
void ThinProgress(float fraction, float width, Tone tone = Tone::Accent);
// Corner brackets and a HUD chip over the scene viewport.
void ViewportFrame(const ImVec2& min, const ImVec2& max, const std::vector<std::string>& hud);
}  // namespace draw

class Host;

// ---- node ---------------------------------------------------------------------------------

class Node {
public:
    virtual ~Node() = default;
    virtual const char* Kind() const = 0;

    void Draw();
    bool Shown() const { return visible && (!visible_if || visible_if()); }
    virtual bool IsField() const { return false; }
    // The width a field's label needs, for its PropertyGrid to fit the label column.
    virtual float LabelWidth() const { return 0.0f; }
    template <class T, class... Args>
    T& Add(Args&&... args) {
        auto node = std::make_unique<T>(std::forward<Args>(args)...);
        T& ref = *node;
        node->parent_ = this;
        children_.push_back(std::move(node));
        return ref;
    }
    Node& Adopt(std::unique_ptr<Node> child);
    void Clear() { children_.clear(); }
    std::unique_ptr<Node> Detach(Node* child);
    Node* Parent() const { return parent_; }
    const std::vector<std::unique_ptr<Node>>& Children() const { return children_; }

    // Props and values as JSON, for the C API and Python.
    virtual void SetProps(const json::Value& props);
    virtual json::Value GetValue() const { return json::Value(); }
    void Emit(const char* type, json::Value value = json::Value());

    std::string id;
    float width = 0.0f;  // 0 natural, < 0 fills the row, > 0 unscaled pixels
    std::string tooltip;
    bool visible = true;
    bool enabled = true;
    std::function<bool()> visible_if;
    std::function<bool()> enabled_if;
    std::function<void(const char* type, const json::Value& value)> on_event;
    uint32_t handle = 0u;
    ImVec2 rect_min{0.0f, 0.0f};
    ImVec2 rect_max{0.0f, 0.0f};
    uint64_t draws = 0u;
    float assigned_width = 0.0f;  // set by an HStack for fill children

protected:
    virtual void DrawSelf() = 0;
    void DrawChildren();
    // The width the next ImGui item takes; `fallback` when the node sets none.
    float ItemWidth(float fallback) const;

    std::vector<std::unique_ptr<Node>> children_;
    Node* parent_ = nullptr;
};

// ---- layout -------------------------------------------------------------------------------

class VStack : public Node {
public:
    const char* Kind() const override { return "vstack"; }
    void SetProps(const json::Value& props) override;
    float spacing = -1.0f;  // < 0: style spacing

protected:
    void DrawSelf() override;
};

class HStack : public Node {
public:
    const char* Kind() const override { return "hstack"; }
    void SetProps(const json::Value& props) override;
    float spacing = -1.0f;

protected:
    void DrawSelf() override;
};

class ZStack : public Node {
public:
    const char* Kind() const override { return "zstack"; }

protected:
    void DrawSelf() override;
};

class Grid : public Node {
public:
    explicit Grid(int column_count = 2) : columns(column_count) {}
    const char* Kind() const override { return "grid"; }
    void SetProps(const json::Value& props) override;
    int columns;

protected:
    void DrawSelf() override;
};

class Spacer : public Node {
public:
    explicit Spacer(float gap = 12.0f) : size(gap) {}
    const char* Kind() const override { return "spacer"; }
    void SetProps(const json::Value& props) override;
    float size;

protected:
    void DrawSelf() override;
};

class Separator : public Node {
public:
    explicit Separator(std::string text = std::string()) : label(std::move(text)) {}
    const char* Kind() const override { return "separator"; }
    void SetProps(const json::Value& props) override;
    std::string label;

protected:
    void DrawSelf() override;
};

class Collapsible : public Node {
public:
    explicit Collapsible(std::string heading, bool start_open = true, const char* glyph = nullptr)
        : title(std::move(heading)), icon(glyph != nullptr ? glyph : ""), open(start_open) {}
    const char* Kind() const override { return "collapsible"; }
    void SetProps(const json::Value& props) override;
    std::string title;
    std::string icon;
    std::string trailing;  // right-aligned glyph, e.g. a tier mark
    bool open;

protected:
    void DrawSelf() override;
};

class Tabs : public Node {
public:
    const char* Kind() const override { return "tabs"; }

protected:
    void DrawSelf() override;
};

class Tab : public Node {
public:
    explicit Tab(std::string heading) : title(std::move(heading)) {}
    const char* Kind() const override { return "tab"; }
    void SetProps(const json::Value& props) override;
    std::string title;

protected:
    void DrawSelf() override;
};

class ScrollFrame : public Node {
public:
    explicit ScrollFrame(float frame_height = 0.0f) : height(frame_height) {}
    const char* Kind() const override { return "scroll"; }
    void SetProps(const json::Value& props) override;
    float height;  // 0 fills the rest of the panel

protected:
    void DrawSelf() override;
};

// ---- basic controls -----------------------------------------------------------------------

class Label : public Node {
public:
    explicit Label(std::string content = std::string(), Tone color = Tone::Normal, TextStyle face = TextStyle::Body)
        : text(std::move(content)), tone(color), style(face) {}
    const char* Kind() const override { return "label"; }
    void SetProps(const json::Value& props) override;
    std::string text;
    std::function<std::string()> text_fn;
    std::function<Tone()> tone_fn;
    Tone tone;
    TextStyle style;
    bool wrap = false;

protected:
    void DrawSelf() override;
};

// A label/value row; the value is drawn in the mono face.
class Stat : public Node {
public:
    Stat(std::string name, std::function<std::string()> fn, Tone color = Tone::Normal)
        : label(std::move(name)), value_fn(std::move(fn)), tone(color) {}
    const char* Kind() const override { return "stat"; }
    void SetProps(const json::Value& props) override;
    std::string label;
    std::string value;
    std::function<std::string()> value_fn;
    std::function<Tone()> tone_fn;
    Tone tone;

protected:
    void DrawSelf() override;
};

// An action-bound button takes its label, enabled and checked state from the action.
class Button : public Node {
public:
    explicit Button(std::string caption = std::string(), const char* glyph = nullptr)
        : label(std::move(caption)), icon(glyph != nullptr ? glyph : "") {}
    const char* Kind() const override { return "button"; }
    void SetProps(const json::Value& props) override;
    Button& Action(std::string action_id, json::Value action_args = json::Value::Object());
    std::string label;
    std::string icon;
    std::string action;
    json::Value args = json::Value::Object();
    std::function<void()> on_click;
    bool primary = false;

protected:
    void DrawSelf() override;
};

class IconButton : public Node {
public:
    explicit IconButton(const char* icon, std::string tip = std::string(), const char* checked_icon = nullptr)
        : glyph(icon), glyph_checked(checked_icon != nullptr ? checked_icon : "") {
        tooltip = std::move(tip);
    }
    const char* Kind() const override { return "icon_button"; }
    void SetProps(const json::Value& props) override;
    IconButton& Action(std::string action_id, json::Value action_args = json::Value::Object());
    std::string glyph;
    std::string glyph_checked;
    std::string action;
    json::Value args = json::Value::Object();
    std::function<void()> on_click;
    std::function<bool()> checked_fn;
    Tone tone = Tone::Normal;

protected:
    void DrawSelf() override;
};

// A pill switch over a bool model, or over a toggle action's checked state.
class ToggleButton : public Node {
public:
    explicit ToggleButton(std::string caption = std::string(), ModelPtr<bool> value = Value<bool>(false))
        : label(std::move(caption)), model(std::move(value)) {}
    const char* Kind() const override { return "toggle"; }
    void SetProps(const json::Value& props) override;
    json::Value GetValue() const override;
    ToggleButton& Action(std::string action_id);
    std::string label;
    ModelPtr<bool> model;
    std::string action;
    std::function<void(bool)> on_change;

protected:
    void DrawSelf() override;
};

class Segmented : public Node {
public:
    Segmented(std::vector<std::string> choices = {}, ModelPtr<int> value = Value<int>(0))
        : options(std::move(choices)), model(std::move(value)) {}
    const char* Kind() const override { return "segmented"; }
    void SetProps(const json::Value& props) override;
    json::Value GetValue() const override;
    std::vector<std::string> options;
    std::vector<std::string> icons;
    std::vector<std::string> tips;
    ModelPtr<int> model;
    std::function<void(int)> on_change;

protected:
    void DrawSelf() override;
};

class ComboBox : public Node {
public:
    ComboBox(std::string caption = std::string(), std::vector<std::string> choices = {},
             ModelPtr<int> value = Value<int>(0))
        : label(std::move(caption)), options(std::move(choices)), model(std::move(value)) {}
    const char* Kind() const override { return "combo"; }
    void SetProps(const json::Value& props) override;
    json::Value GetValue() const override;
    std::string label;
    std::vector<std::string> options;
    ModelPtr<int> model;
    std::function<void(int)> on_change;

protected:
    void DrawSelf() override;
};

// ---- fields -------------------------------------------------------------------------------

// A labelled editor row: the label takes the left 40%, or a PropertyGrid column fitted to its labels.
// on_change fires every edited frame; on_commit(before, after) once the gesture ends.
template <class T>
class Field : public Node {
public:
    Field(std::string caption, ModelPtr<T> value) : label(std::move(caption)), model(std::move(value)) {}
    bool IsField() const override { return true; }
    float LabelWidth() const override;
    void SetProps(const json::Value& props) override;
    json::Value GetValue() const override;
    std::string label;
    std::string unit;
    Tier tier = Tier::None;
    ModelPtr<T> model;
    std::function<void(const T&)> on_change;
    std::function<void(const T& before, const T& after)> on_commit;

protected:
    void DrawSelf() override;
    // Draws the editor for `value`; true when it changed this frame.
    virtual bool Edit(T& value) = 0;
    T before_{};
};

class FloatField : public Field<float> {
public:
    FloatField(std::string caption, ModelPtr<float> value, float drag_speed = 0.01f, float lo = 0.0f, float hi = 0.0f)
        : Field(std::move(caption), std::move(value)), speed(drag_speed), min(lo), max(hi) {}
    const char* Kind() const override { return "float"; }
    void SetProps(const json::Value& props) override;
    float speed, min, max;
    bool slider = false;
    std::string format = "%.3f";

protected:
    bool Edit(float& value) override;
};

class IntField : public Field<int> {
public:
    IntField(std::string caption, ModelPtr<int> value, int lo = 0, int hi = 0)
        : Field(std::move(caption), std::move(value)), min(lo), max(hi) {}
    const char* Kind() const override { return "int"; }
    void SetProps(const json::Value& props) override;
    int min, max;
    float speed = 0.1f;

protected:
    bool Edit(int& value) override;
};

class BoolField : public Field<bool> {
public:
    BoolField(std::string caption, ModelPtr<bool> value) : Field(std::move(caption), std::move(value)) {}
    const char* Kind() const override { return "bool"; }

protected:
    bool Edit(bool& value) override;
};

class Vec3Field : public Field<Vec3> {
public:
    Vec3Field(std::string caption, ModelPtr<Vec3> value, float drag_speed = 0.01f)
        : Field(std::move(caption), std::move(value)), speed(drag_speed) {}
    const char* Kind() const override { return "vec3"; }
    void SetProps(const json::Value& props) override;
    float speed;
    std::string format = "%.3f";

protected:
    bool Edit(Vec3& value) override;
};

// Shows intrinsic-XYZ Euler degrees; the model stores a unit quaternion.
class QuatField : public Field<Quat> {
public:
    QuatField(std::string caption, ModelPtr<Quat> value) : Field(std::move(caption), std::move(value)) {}
    const char* Kind() const override { return "quat"; }

protected:
    bool Edit(Quat& value) override;
};

class ColorField : public Field<Color> {
public:
    ColorField(std::string caption, ModelPtr<Color> value, bool with_alpha = true, bool high_range = false)
        : Field(std::move(caption), std::move(value)), alpha(with_alpha), hdr(high_range) {}
    const char* Kind() const override { return "color"; }
    void SetProps(const json::Value& props) override;
    bool alpha, hdr;

protected:
    void DrawSelf() override;
    bool Edit(Color& value) override;

private:
    // The picker popup edits outside the row's item, so its gesture ends when it closes.
    bool picking_ = false;
    bool picked_ = false;
    Color pick_before_{};
};

class StringField : public Field<std::string> {
public:
    StringField(std::string caption, ModelPtr<std::string> value, std::string placeholder = std::string())
        : Field(std::move(caption), std::move(value)), hint(std::move(placeholder)) {}
    const char* Kind() const override { return "string"; }
    void SetProps(const json::Value& props) override;
    std::string hint;

protected:
    bool Edit(std::string& value) override;
};

// A path box with a browse button; on_browse may set the model.
class PathField : public Field<std::string> {
public:
    PathField(std::string caption, ModelPtr<std::string> value) : Field(std::move(caption), std::move(value)) {}
    const char* Kind() const override { return "path"; }
    std::function<void()> on_browse;

protected:
    bool Edit(std::string& value) override;
};

// A field row over a fixed char buffer, for host state that stores C strings.
class TextBufferField : public Node {
public:
    TextBufferField(std::string caption, char* buffer, size_t capacity, std::string placeholder = std::string())
        : label(std::move(caption)), hint(std::move(placeholder)), buffer_(buffer), capacity_(capacity) {}
    const char* Kind() const override { return "text_buffer"; }
    bool IsField() const override { return true; }
    float LabelWidth() const override;
    std::string label;
    std::string hint;
    std::function<void()> on_enter;

protected:
    void DrawSelf() override;

private:
    char* buffer_;
    size_t capacity_;
};

class Slider : public Node {
public:
    Slider(ModelPtr<float> value = Value<float>(0.0f), float lo = 0.0f, float hi = 1.0f)
        : model(std::move(value)), min(lo), max(hi) {}
    const char* Kind() const override { return "slider"; }
    void SetProps(const json::Value& props) override;
    json::Value GetValue() const override;
    ModelPtr<float> model;
    float min, max;
    std::string format = "%.3f";
    std::function<void(float)> on_change;
    std::function<void(float before, float after)> on_commit;

protected:
    void DrawSelf() override;

private:
    float before_ = 0.0f;
};

class Drag : public Node {
public:
    Drag(ModelPtr<float> value = Value<float>(0.0f), float drag_speed = 0.01f)
        : model(std::move(value)), speed(drag_speed) {}
    const char* Kind() const override { return "drag"; }
    void SetProps(const json::Value& props) override;
    json::Value GetValue() const override;
    ModelPtr<float> model;
    float speed;
    float min = 0.0f, max = 0.0f;
    std::string format = "%.3f";
    std::function<void(float)> on_change;

protected:
    void DrawSelf() override;
};

// ---- data views ---------------------------------------------------------------------------

// A virtualized tree: rows are flattened only when the model version, the expansion or the
// filter changes, and only the visible rows draw.
class TreeView : public Node {
public:
    // A toggle at a row's right end, shown while it is on or the row is hovered.
    struct RowTool {
        const char* icon_on = "";
        const char* icon_off = "";
        std::string tooltip;
        std::function<bool(uint64_t)> on;
        std::function<void(uint64_t)> on_click;
    };

    explicit TreeView(std::shared_ptr<ItemModel> items = nullptr) : model(std::move(items)) {}
    const char* Kind() const override { return "tree"; }
    void SetProps(const json::Value& props) override;
    json::Value GetValue() const override;
    void ExpandAll();
    void CollapseAll();
    void Reveal(uint64_t item);
    size_t RowCount() const { return rows_.size(); }

    std::shared_ptr<ItemModel> model;
    std::function<bool(uint64_t)> is_selected;
    std::function<bool(uint64_t)> is_dimmed;  // drawn muted, e.g. a hidden node
    std::function<void(uint64_t)> on_select;
    // Ctrl+click; without it, a Ctrl+click selects.
    std::function<void(uint64_t)> on_toggle;
    // Shift+click: the rows from the last clicked one through this one, this one last.
    std::function<void(const std::vector<uint64_t>&)> on_range;
    // F2 on the selected row edits its label in place; Enter commits.
    std::function<void(uint64_t, const std::string&)> on_rename;
    std::vector<RowTool> row_tools;
    std::function<void(uint64_t)> on_activate;
    std::function<void(uint64_t)> context_menu;  // draws the popup body
    std::function<void(uint64_t dragged, uint64_t onto)> on_drop;
    ModelPtr<std::string> filter;
    int open_depth = 1;  // levels expanded the first time the tree shows
    float height = 0.0f;

protected:
    void DrawSelf() override;

private:
    struct Row {
        uint64_t item;
        int depth;
        bool has_children;
    };
    void Flatten();
    bool Matches(uint64_t item, const std::string& needle) const;
    void Walk(uint64_t item, int depth, const std::string& needle);
    void Select(uint64_t item);

    std::vector<Row> rows_;
    std::unordered_set<uint64_t> open_;
    uint64_t version_ = ~0ull;
    uint64_t layout_ = 0u;
    uint64_t flattened_layout_ = ~0ull;
    std::string flattened_filter_;
    bool seeded_ = false;
    uint64_t selected_ = 0u;
    uint64_t reveal_ = 0u;
    uint64_t anchor_ = 0u;  // the last row clicked without Shift
    uint64_t renaming_ = 0u;
    bool rename_focus_ = false;
    char rename_buf_[256] = {0};
};

class Table : public Node {
public:
    const char* Kind() const override { return "table"; }
    void SetProps(const json::Value& props) override;
    std::vector<std::string> columns;
    std::function<size_t()> row_count;
    std::function<std::string(size_t row, size_t column)> cell;
    std::vector<std::vector<std::string>> rows;  // used when row_count is unset
    float height = 0.0f;

protected:
    void DrawSelf() override;
};

// Lays its Field children out as a two-column table. Built from a JSON schema it owns one
// field per entry and emits change and commit events carrying {key, value}.
class PropertyGrid : public Node {
public:
    const char* Kind() const override { return "property_grid"; }
    void SetProps(const json::Value& props) override;
    json::Value GetValue() const override;

protected:
    void DrawSelf() override;

private:
    void BuildFromSchema(const json::Value& schema);
};

// ---- display ------------------------------------------------------------------------------

struct Series {
    std::string name;
    std::vector<float> x;
    std::vector<float> y;
};

class Plot : public Node {
public:
    const char* Kind() const override { return "plot"; }
    void SetProps(const json::Value& props) override;
    // Appends one sample and keeps the newest `keep`.
    void Push(size_t index, float x, float y, size_t keep);
    std::vector<Series> series;
    float height = 120.0f;
    bool sparkline = false;  // no axes, fill under the line
    float y_min = 0.0f, y_max = 0.0f;  // equal: auto-fit

protected:
    void DrawSelf() override;
};

class Image : public Node {
public:
    Image(ImTextureRef tex, ImVec2 extent) : texture(tex), size(extent) {}
    const char* Kind() const override { return "image"; }
    ImTextureRef texture;
    ImVec2 size;
    ImVec2 uv0{0.0f, 0.0f}, uv1{1.0f, 1.0f};

protected:
    void DrawSelf() override;
};

// Python language, theme colors, line numbers and four-space indents for a code editor.
void SetupCodeEditor(TextEditor& editor);
// True for a traceback entry, `  File "...", line N`, with the file and 1-based line it names.
bool ParseTraceEntry(std::string_view text, std::string* file, int* line);

// A Python code editor over one the caller keeps (`source`), else over its own text.
// Ctrl+Enter emits "run" instead of breaking the line.
class CodeEditor : public Node {
public:
    explicit CodeEditor(std::function<TextEditor*()> source = {});
    ~CodeEditor() override;
    const char* Kind() const override { return "code"; }
    void SetProps(const json::Value& props) override;
    json::Value GetValue() const override;
    float height = 0.0f;  // 0: half the remaining height, < 0: all of it but -height
    std::function<void()> on_run;

protected:
    void DrawSelf() override;

private:
    TextEditor& Editor() const;

    std::function<TextEditor*()> source_;
    std::unique_ptr<TextEditor> own_;
};

// A scrolling log view. Lines are colored by level: a traceback counts as an error, otherwise
// the level words a line carries decide. `File "...", line N` entries are links.
class LogView : public Node {
public:
    enum class LineLevel : uint8_t { Info, Warn, Err, Echo };
    explicit LogView(const std::string* text = nullptr) : text_(text) {}
    const char* Kind() const override { return "log"; }
    void SetProps(const json::Value& props) override;
    ModelPtr<std::string> filter;
    ModelPtr<int> level;  // 0 every line, 1 info, 2 warnings, 3 errors
    float height = 0.0f;  // 0 fills the rest of the panel, < 0 leaves -height below
    // A link click: the file and 1-based line it names.
    std::function<void(const std::string& file, int line)> on_link;

protected:
    void DrawSelf() override;

private:
    struct Line {
        size_t begin = 0u, end = 0u;
        LineLevel level = LineLevel::Info;
        bool in_trace = false;  // a traceback goes on after this line
    };

    const std::string* text_;
    std::string own_;
    size_t indexed_size_ = ~size_t(0);
    std::string head_;  // the text's first bytes when indexed: a trimmed front reindexes
    std::vector<Line> lines_;
};

class Badge : public Node {
public:
    explicit Badge(std::string content = std::string(), Tone color = Tone::Accent, const char* glyph = nullptr)
        : text(std::move(content)), icon(glyph != nullptr ? glyph : ""), tone(color) {}
    const char* Kind() const override { return "badge"; }
    void SetProps(const json::Value& props) override;
    std::string text;
    std::function<std::string()> text_fn;
    std::function<Tone()> tone_fn;
    std::string icon;
    Tone tone;

protected:
    void DrawSelf() override;
};

class StatusDot : public Node {
public:
    explicit StatusDot(Tone color = Tone::Ok, std::string caption = std::string())
        : label(std::move(caption)), tone(color) {}
    const char* Kind() const override { return "status"; }
    void SetProps(const json::Value& props) override;
    std::string label;
    std::function<std::string()> label_fn;
    std::function<Tone()> tone_fn;
    Tone tone;
    bool pulse = false;

protected:
    void DrawSelf() override;
};

class ProgressBar : public Node {
public:
    explicit ProgressBar(ModelPtr<float> value = Value<float>(0.0f)) : fraction(std::move(value)) {}
    const char* Kind() const override { return "progress"; }
    void SetProps(const json::Value& props) override;
    json::Value GetValue() const override;
    ModelPtr<float> fraction;  // < 0: indeterminate
    std::string label;

protected:
    void DrawSelf() override;
};

// Opens on a right click over the previous sibling; children draw inside the popup.
class ContextMenu : public Node {
public:
    const char* Kind() const override { return "context_menu"; }

protected:
    void DrawSelf() override;
};

class Menu : public Node {
public:
    explicit Menu(std::string caption) : label(std::move(caption)) {}
    const char* Kind() const override { return "menu"; }
    std::string label;

protected:
    void DrawSelf() override;
};

class MenuItem : public Node {
public:
    explicit MenuItem(std::string caption = std::string()) : label(std::move(caption)) {}
    const char* Kind() const override { return "menu_item"; }
    void SetProps(const json::Value& props) override;
    MenuItem& Action(std::string action_id, json::Value action_args = json::Value::Object());
    std::string label;
    std::string action;
    json::Value args = json::Value::Object();
    std::function<void()> on_click;

protected:
    void DrawSelf() override;
};

// Rich tooltip content shown while the previous sibling is hovered.
class Tooltip : public Node {
public:
    const char* Kind() const override { return "tooltip"; }

protected:
    void DrawSelf() override;
};

// Draws a C++ callback in place, for content no widget covers.
class Custom : public Node {
public:
    explicit Custom(std::function<void()> draw_fn) : fn(std::move(draw_fn)) {}
    const char* Kind() const override { return "custom"; }
    std::function<void()> fn;

protected:
    void DrawSelf() override { fn(); }
};

// ---- panels and host ----------------------------------------------------------------------

class Panel {
public:
    Panel(std::string panel_id, std::string caption, const char* glyph);
    VStack& Root() { return *root_; }
    void Invalidate() { built_ = false; }
    std::string WindowName() const;
    void Draw(Host& host);

    std::string id;
    std::string title;
    std::string icon;
    std::string dock;  // "left", "right", "bottom" or "float" for panels the layout does not place
    bool open = true;
    bool focus_request = false;
    std::function<void(VStack& root)> build;
    std::function<uint64_t()> key;
    ImGuiWindowFlags flags = 0;
    uint64_t rebuilds = 0u;
    uint64_t draws = 0u;

private:
    std::unique_ptr<VStack> root_;
    uint64_t key_ = 0u;
    bool built_ = false;
};

struct ActionHooks {
    std::function<void(const std::string& id, const json::Value& args)> run;
    std::function<bool(const std::string& id)> enabled;
    std::function<int(const std::string& id)> checked;  // -1: not a toggle
    std::function<std::string(const std::string& id)> label;
    std::function<std::string(const std::string& id)> shortcut;
};

// Owns the panels, toasts and progress entries, and serves the C API. API calls lock the
// host, so scripts on the physics thread may build panels while the UI thread draws.
class Host {
public:
    Host();
    ~Host();
    Host(const Host&) = delete;
    Host& operator=(const Host&) = delete;

    Panel& AddPanel(std::string id, std::string title, const char* icon);
    Panel* FindPanel(std::string_view id);
    void RemovePanel(std::string_view id);
    const std::vector<std::unique_ptr<Panel>>& Panels() const { return panels_; }
    // Draws the open panels; panels the layout does not place dock where `dock_node` says.
    void DrawPanels(const std::function<ImGuiID(const Panel&)>& dock_node);

    void Toast(Level level, std::string text);
    void Progress(const std::string& id, const std::string& label, float fraction);
    // Toasts stack up from the bottom-right corner of min..max on `list`.
    void DrawNotifications(ImDrawList* list, const ImVec2& min, const ImVec2& max);
    // Applies string table changes, then runs the C-API event callbacks queued since the
    // last call, outside any draw.
    void DispatchEvents();

    std::recursive_mutex& Mutex() { return mu_; }
    static Host* Current();
    ActionHooks actions;

    const nuka_nui_api_t* Api() { return &api_; }
    uint32_t PanelNode(const std::string& id, const std::string& title, const std::string& dock);
    uint32_t AddNode(uint32_t parent, const std::string& kind, const json::Value& props);
    bool SetNode(uint32_t node, const json::Value& props);
    bool RemoveNode(uint32_t node);
    bool ClearNode(uint32_t node);
    Node* Lookup(uint32_t handle);
    bool Listen(uint32_t node, nuka_nui_event_fn fn, void* user);
    std::string reply;
    std::string error;

private:
    struct ToastEntry {
        Level level;
        std::string text;
        double born;
    };
    struct ProgressEntry {
        std::string id;
        std::string label;
        float fraction;
    };
    struct Event {
        nuka_nui_event_fn fn;
        void* user;
        uint32_t node;
        std::string json;
    };
    uint32_t Register(Node& node);
    void Forget(Node& node);

    std::recursive_mutex mu_;
    std::vector<std::unique_ptr<Panel>> panels_;
    std::deque<ToastEntry> toasts_;
    std::vector<ProgressEntry> progress_;
    std::vector<Event> events_;
    std::unordered_map<uint32_t, Node*> nodes_;
    std::unordered_map<uint32_t, Panel*> panel_roots_;
    uint32_t next_handle_ = 1u;
    nuka_nui_api_t api_{};
};

// The node for a kind name ("label", "button", "vstack", ...), or null.
std::unique_ptr<Node> MakeNode(const std::string& kind);

}  // namespace nuka::nui
