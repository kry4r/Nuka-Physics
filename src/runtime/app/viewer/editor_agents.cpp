#include "runtime/app/viewer/editor_agents.hpp"

#include "runtime/app/viewer/editor_profile.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace nuka::runtime::app::viewer {

namespace json = scene::json;

namespace {

constexpr std::string_view kModes[] = {"off", "read", "propose", "full"};
// What settles access itself stays with the user, whatever the mode or lists say.
constexpr std::string_view kUserOnly[] = {"agent.mode", "agent.accept", "agent.reject", "agent.approve",
                                          "agent.disconnect"};
constexpr size_t kCallsKept = 50u;
constexpr size_t kDecidedKept = 100u;

bool IsMode(const std::string& mode) {
    return std::find(std::begin(kModes), std::end(kModes), mode) != std::end(kModes);
}

// A pattern matches an action id whole, each '*' standing for any run of characters.
bool Matches(std::string_view pattern, std::string_view id) {
    const size_t star = pattern.find('*');
    if (star == std::string_view::npos) return pattern == id;
    if (id.substr(0, star) != pattern.substr(0, star)) return false;
    const std::string_view rest = pattern.substr(star + 1u);
    for (size_t at = star; at <= id.size(); ++at)
        if (Matches(rest, id.substr(at))) return true;
    return false;
}

bool Listed(const std::vector<std::string>& patterns, const std::string& id) {
    return std::any_of(patterns.begin(), patterns.end(), [&](const std::string& p) { return Matches(p, id); });
}

std::vector<std::string> Strings(const json::Value& list) {
    std::vector<std::string> out;
    for (const json::Value& v : list.Elements()) out.push_back(v.AsString());
    return out;
}

json::Value StringList(const std::vector<std::string>& values) {
    json::Value list = json::Value::Array();
    for (const std::string& v : values) list.PushBack(json::Value::Str(v));
    return list;
}

std::string Joined(const std::vector<std::string>& values) {
    std::string text;
    for (const std::string& v : values) text += (text.empty() ? "" : ", ") + v;
    return text;
}

}  // namespace

void AgentAccess::Load(const EditorProfile& profile) {
    const std::string mode = profile.Setting("agent", "mode", "full");
    mode_ = IsMode(mode) ? mode : "full";
    allow_ = profile.SettingList("agent", "allow");
    deny_ = profile.SettingList("agent", "deny");
    log_dir_ = profile.Dir() / "logs";
    ++version_;
}

void AgentAccess::Save(EditorProfile* profile) const {
    profile->SetSetting("agent", "mode", mode_);
    profile->SetSettingList("agent", "allow", allow_);
    profile->SetSettingList("agent", "deny", deny_);
}

void AgentAccess::Configure(const json::Value& args, EditorProfile* profile) {
    if (const json::Value* mode = args.Find("mode")) {
        if (!IsMode(mode->AsString()))
            throw std::runtime_error("mode is off, read, propose or full, not " + mode->AsString());
        mode_ = mode->AsString();
    }
    if (const json::Value* allow = args.Find("allow")) allow_ = Strings(*allow);
    if (const json::Value* deny = args.Find("deny")) deny_ = Strings(*deny);
    Save(profile);
    ++version_;
}

void AgentAccess::AllowAlways(const std::string& id, EditorProfile* profile) {
    if (std::find(allow_.begin(), allow_.end(), id) == allow_.end()) allow_.push_back(id);
    Save(profile);
    ++version_;
}

json::Value AgentAccess::Settings() const {
    json::Value v = json::Value::Object();
    v.Set("mode", json::Value::Str(mode_));
    v.Set("allow", StringList(allow_));
    v.Set("deny", StringList(deny_));
    return v;
}

AgentAccess::Verdict AgentAccess::Check(const std::string& id, const std::string& kind, std::string* reason) const {
    const auto refuse = [reason](std::string why) {
        *reason = std::move(why);
        return Verdict::Refuse;
    };
    if (mode_ == "off") return refuse("agent access is off");
    if (std::find(std::begin(kUserOnly), std::end(kUserOnly), id) != std::end(kUserOnly))
        return refuse("only the user runs " + id);
    if (Listed(deny_, id)) return refuse(id + " is on the agent deny list");
    if (Listed(allow_, id)) return Verdict::Run;
    if (id == "agent.propose")
        return mode_ == "read" ? refuse("agent access is read: proposals are off") : Verdict::Run;
    if (mode_ == "full" || kind == "read" || kind == "view") return Verdict::Run;
    // An action that declares no kind, such as an extension's, counts as one on the editor itself.
    const std::string touches = kind.empty() ? "app" : kind;
    if (mode_ == "read")
        return refuse("agent access is read: " + id + " (" + touches + ") is refused; the user can widen it in the "
                      "Agents panel");
    if (touches == "sim") return Verdict::Run;
    return touches == "edit" ? Verdict::Propose : Verdict::Ask;
}

uint64_t AgentAccess::Ask(AgentRequest request) {
    request.view.id = next_id_++;
    request.view.state = "pending";
    request.view.time = UnixNow();
    request.view.args = request.args.Dump(0);
    requests_.push_back(std::move(request));
    ++version_;
    return requests_.back().view.id;
}

AgentRequest& AgentAccess::Pending(uint64_t id, const std::string& kind) {
    const auto it =
        std::find_if(requests_.begin(), requests_.end(), [id](const AgentRequest& r) { return r.view.id == id; });
    if (it == requests_.end()) throw std::runtime_error("no agent request " + std::to_string(id));
    if (it->view.state != "pending")
        throw std::runtime_error("agent request " + std::to_string(id) + " is " + it->view.state);
    if (!kind.empty() && it->view.kind != kind)
        throw std::runtime_error("agent request " + std::to_string(id) + " is a " + it->view.kind + ", not a " + kind);
    return *it;
}

void AgentAccess::Decide(AgentRequest& request, const std::string& state, const ActionResult* result) {
    request.view.state = result != nullptr && !result->ok ? "failed" : state;
    if (result != nullptr) {
        request.value = result->value;
        request.view.error = result->error;
    }
    // Pending requests stay; of the settled ones the newest are kept.
    size_t decided = requests_.size() - PendingCount();
    for (auto it = requests_.begin(); decided > kDecidedKept && it != requests_.end();) {
        if (it->view.state != "pending" && it->view.id != request.view.id) {
            it = requests_.erase(it);
            --decided;
        } else {
            ++it;
        }
    }
    ++version_;
}

json::Value AgentAccess::Describe(const AgentRequest& request) const {
    const AgentRequestView& r = request.view;
    json::Value changes = json::Value::Array();
    for (const AgentChange& c : r.changes) {
        json::Value change = json::Value::Object();
        change.Set("target", json::Value::Str(c.target));
        change.Set("field", json::Value::Str(c.field));
        change.Set("before", json::Value::Str(c.before));
        change.Set("after", json::Value::Str(c.after));
        changes.PushBack(std::move(change));
    }
    json::Value v = json::Value::Object();
    v.Set("id", json::Value::Int(static_cast<int64_t>(r.id)));
    v.Set("kind", json::Value::Str(r.kind));
    v.Set("state", json::Value::Str(r.state));
    v.Set("client", json::Value::Str(r.client));
    v.Set("action", json::Value::Str(r.action));
    v.Set("args", request.args);
    v.Set("label", json::Value::Str(r.label));
    v.Set("note", json::Value::Str(r.note));
    v.Set("changes", std::move(changes));
    v.Set("value", request.value);
    v.Set("error", json::Value::Str(r.error));
    v.Set("time", json::Value::Int(r.time));
    return v;
}

json::Value AgentAccess::Requests() const {
    json::Value list = json::Value::Array();
    for (const bool pending : {true, false})
        for (auto it = requests_.rbegin(); it != requests_.rend(); ++it)
            if ((it->view.state == "pending") == pending) list.PushBack(Describe(*it));
    return list;
}

size_t AgentAccess::PendingCount() const {
    return static_cast<size_t>(std::count_if(requests_.begin(), requests_.end(),
                                             [](const AgentRequest& r) { return r.view.state == "pending"; }));
}

void AgentAccess::SetClients(const json::Value& clients) {
    clients_.clear();
    for (const json::Value& c : clients.Elements())
        clients_.push_back(AgentClientView{c.At("client").AsString(), c.At("pid").AsInt(),
                                           static_cast<int64_t>(c.At("since").AsDouble()), 0u});
    ++version_;
}

void AgentAccess::Record(const std::string& client, const std::string& id, const json::Value& args,
                         const ActionResult& result, const std::string& outcome, double ms) {
    const int64_t now = UnixNow();
    calls_.insert(calls_.begin(), AgentCallView{now, client, id, outcome, result.ok, ms});
    if (calls_.size() > kCallsKept) calls_.pop_back();
    ++calls_by_client_[client];
    json::Value line = json::Value::Object();
    line.Set("time", json::Value::Str(LocalTimeText(now, "%Y-%m-%dT%H:%M:%S")));
    line.Set("client", json::Value::Str(client));
    line.Set("action", json::Value::Str(id));
    line.Set("args", args);
    line.Set("outcome", json::Value::Str(outcome));
    line.Set("ok", json::Value::Bool(result.ok));
    line.Set("error", json::Value::Str(result.error));
    line.Set("ms", json::Value::Float(std::round(ms * 10.0) / 10.0));
    std::filesystem::create_directories(log_dir_);
    std::ofstream(log_dir_ / ("agent-" + LocalTimeText(now, "%Y%m%d") + ".jsonl"), std::ios::app)
        << line.Dump(0) << '\n';
    ++version_;
}

void AgentAccess::Fill(AgentView* view) const {
    if (view->version == version_) return;
    view->version = version_;
    view->mode = mode_;
    view->allow = allow_;
    view->deny = deny_;
    std::snprintf(view->allow_edit, sizeof(view->allow_edit), "%s", Joined(allow_).c_str());
    std::snprintf(view->deny_edit, sizeof(view->deny_edit), "%s", Joined(deny_).c_str());
    view->clients = clients_;
    for (AgentClientView& c : view->clients) {
        const auto calls = calls_by_client_.find(c.name);
        c.calls = calls != calls_by_client_.end() ? calls->second : 0u;
    }
    view->requests.clear();
    for (const bool pending : {true, false})
        for (auto it = requests_.rbegin(); it != requests_.rend(); ++it)
            if ((it->view.state == "pending") == pending) view->requests.push_back(it->view);
    view->calls = calls_;
    view->pending = PendingCount();
}

}  // namespace nuka::runtime::app::viewer
