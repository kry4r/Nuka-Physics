// The nui node factory and the C function table Python and extensions build panels through.

#include "runtime/app/viewer/nui/nui_internal.hpp"

#include <exception>

namespace nuka::nui {

using namespace detail;

std::unique_ptr<Node> MakeNode(const std::string& kind) {
    if (kind == "vstack") return std::make_unique<VStack>();
    if (kind == "hstack") return std::make_unique<HStack>();
    if (kind == "zstack") return std::make_unique<ZStack>();
    if (kind == "grid") return std::make_unique<Grid>();
    if (kind == "spacer") return std::make_unique<Spacer>();
    if (kind == "separator") return std::make_unique<Separator>();
    if (kind == "collapsible") return std::make_unique<Collapsible>(std::string());
    if (kind == "tabs") return std::make_unique<Tabs>();
    if (kind == "tab") return std::make_unique<Tab>(std::string());
    if (kind == "scroll") return std::make_unique<ScrollFrame>();
    if (kind == "label") return std::make_unique<Label>();
    if (kind == "stat") return std::make_unique<Stat>(std::string(), std::function<std::string()>());
    if (kind == "button") return std::make_unique<Button>();
    if (kind == "icon_button") return std::make_unique<IconButton>("");
    if (kind == "toggle") return std::make_unique<ToggleButton>();
    if (kind == "segmented") return std::make_unique<Segmented>();
    if (kind == "combo") return std::make_unique<ComboBox>();
    if (kind == "float") return std::make_unique<FloatField>(std::string(), Value<float>(0.0f));
    if (kind == "int") return std::make_unique<IntField>(std::string(), Value<int>(0));
    if (kind == "bool") return std::make_unique<BoolField>(std::string(), Value<bool>(false));
    if (kind == "vec3") return std::make_unique<Vec3Field>(std::string(), Value<Vec3>(Vec3{}));
    if (kind == "quat") return std::make_unique<QuatField>(std::string(), Value<Quat>(Quat{0.0f, 0.0f, 0.0f, 1.0f}));
    if (kind == "color") return std::make_unique<ColorField>(std::string(), Value<Color>(Color{1.0f, 1.0f, 1.0f, 1.0f}));
    if (kind == "string") return std::make_unique<StringField>(std::string(), Value<std::string>());
    if (kind == "path") return std::make_unique<PathField>(std::string(), Value<std::string>());
    if (kind == "slider") return std::make_unique<Slider>();
    if (kind == "drag") return std::make_unique<Drag>();
    if (kind == "tree") return std::make_unique<TreeView>();
    if (kind == "table") return std::make_unique<Table>();
    if (kind == "property_grid") return std::make_unique<PropertyGrid>();
    if (kind == "plot") return std::make_unique<Plot>();
    if (kind == "code") return std::make_unique<CodeEditor>();
    if (kind == "log") return std::make_unique<LogView>();
    if (kind == "badge") return std::make_unique<Badge>();
    if (kind == "status") return std::make_unique<StatusDot>();
    if (kind == "progress") return std::make_unique<ProgressBar>();
    if (kind == "context_menu") return std::make_unique<ContextMenu>();
    if (kind == "menu") return std::make_unique<Menu>(std::string());
    if (kind == "menu_item") return std::make_unique<MenuItem>();
    if (kind == "tooltip") return std::make_unique<Tooltip>();
    return nullptr;
}

// ---- host API -----------------------------------------------------------------------------

uint32_t Host::PanelNode(const std::string& panel_id, const std::string& panel_title, const std::string& panel_dock) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    Panel& panel = AddPanel(panel_id, panel_title, nullptr);
    panel.dock = panel_dock;
    panel.script = true;
    if (panel.Root().handle == 0u) panel_roots_[Register(panel.Root())] = &panel;
    return panel.Root().handle;
}

uint32_t Host::AddNode(uint32_t parent, const std::string& kind, const json::Value& props) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    Node* target = Lookup(parent);
    if (target == nullptr) {
        error = "unknown node " + std::to_string(parent);
        return 0u;
    }
    std::unique_ptr<Node> node = MakeNode(kind);
    if (!node) {
        error = "unknown kind '" + kind + "'";
        return 0u;
    }
    node->SetProps(props);
    return Register(target->Adopt(std::move(node)));
}

// A panel root also takes the panel's own props: title, icon, open and focus.
bool Host::SetNode(uint32_t node_handle, const json::Value& props) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    Node* node = Lookup(node_handle);
    if (node == nullptr) {
        error = "unknown node " + std::to_string(node_handle);
        return false;
    }
    node->SetProps(props);
    const auto root = panel_roots_.find(node_handle);
    if (root != panel_roots_.end()) {
        Panel& panel = *root->second;
        Read(props, "title", panel.title);
        ReadIcon(props, "icon", panel.icon);
        Read(props, "open", panel.open);
        Read(props, "focus", panel.focus_request);
    }
    return true;
}

bool Host::RemoveNode(uint32_t node_handle) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    Node* node = Lookup(node_handle);
    if (node == nullptr) {
        error = "unknown node " + std::to_string(node_handle);
        return false;
    }
    const auto root = panel_roots_.find(node_handle);
    if (root != panel_roots_.end()) {
        const std::string panel_id = root->second->id;
        RemovePanel(panel_id);
        return true;
    }
    Forget(*node);
    node->Parent()->Detach(node);
    return true;
}

bool Host::ClearNode(uint32_t node_handle) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    Node* node = Lookup(node_handle);
    if (node == nullptr) {
        error = "unknown node " + std::to_string(node_handle);
        return false;
    }
    for (const auto& child : node->Children()) Forget(*child);
    node->Clear();
    return true;
}

// Events queue as JSON and reach `fn` from DispatchEvents, never from inside a draw.
bool Host::Listen(uint32_t node_handle, nuka_nui_event_fn fn, void* user) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    Node* node = Lookup(node_handle);
    if (node == nullptr) {
        error = "unknown node " + std::to_string(node_handle);
        return false;
    }
    node->on_event = [this, fn, user, node_handle](const char* type, const json::Value& value) {
        json::Value event = json::Value::Object();
        event.Set("type", json::Value::Str(type));
        event.Set("value", value);
        std::lock_guard<std::recursive_mutex> guard(mu_);
        events_.push_back(Event{fn, user, node_handle, event.Dump(0)});
    };
    return true;
}

// ---- C table ------------------------------------------------------------------------------

namespace {

Host& H(void* host) { return *static_cast<Host*>(host); }

json::Value Props(const char* text) {
    return text == nullptr || *text == '\0' ? json::Value::Object() : json::Value::Parse(text);
}

// Bad JSON or a prop of the wrong type fails the call and leaves the reason in last_error.
void Fail(Host& host, const std::exception& e) {
    std::lock_guard<std::recursive_mutex> lock(host.Mutex());
    host.error = e.what();
}

uint32_t ApiPanel(void* host, const char* id, const char* title, const char* dock) {
    return H(host).PanelNode(id, title, dock);
}

uint32_t ApiAdd(void* host, uint32_t parent, const char* kind, const char* props) {
    try {
        return H(host).AddNode(parent, kind, Props(props));
    } catch (const std::exception& e) {
        Fail(H(host), e);
        return 0u;
    }
}

int32_t ApiSet(void* host, uint32_t node, const char* props) {
    try {
        return H(host).SetNode(node, Props(props)) ? 1 : 0;
    } catch (const std::exception& e) {
        Fail(H(host), e);
        return 0;
    }
}

int32_t ApiRemove(void* host, uint32_t node) { return H(host).RemoveNode(node) ? 1 : 0; }

int32_t ApiClear(void* host, uint32_t node) { return H(host).ClearNode(node) ? 1 : 0; }

const char* ApiGet(void* host, uint32_t node) {
    Host& h = H(host);
    std::lock_guard<std::recursive_mutex> lock(h.Mutex());
    Node* target = h.Lookup(node);
    if (target == nullptr) {
        h.error = "unknown node " + std::to_string(node);
        return nullptr;
    }
    h.reply = target->GetValue().Dump(0);
    return h.reply.c_str();
}

int32_t ApiOnEvent(void* host, uint32_t node, nuka_nui_event_fn fn, void* user) {
    return H(host).Listen(node, fn, user) ? 1 : 0;
}

void ApiToast(void* host, nuka_nui_level_t level, const char* text) { H(host).Toast(static_cast<Level>(level), text); }

void ApiProgress(void* host, const char* id, const char* label, float fraction) { H(host).Progress(id, label, fraction); }

const char* ApiLastError(void* host) { return H(host).error.c_str(); }

}  // namespace

namespace detail {

void FillApi(nuka_nui_api_t& api, Host* host) {
    api.struct_size = static_cast<uint32_t>(sizeof(nuka_nui_api_t));
    api.version = NUKA_NUI_API_VERSION;
    api.host = host;
    api.panel = &ApiPanel;
    api.add = &ApiAdd;
    api.set = &ApiSet;
    api.remove = &ApiRemove;
    api.clear = &ApiClear;
    api.get = &ApiGet;
    api.on_event = &ApiOnEvent;
    api.toast = &ApiToast;
    api.progress = &ApiProgress;
    api.last_error = &ApiLastError;
}

}  // namespace detail

}  // namespace nuka::nui
