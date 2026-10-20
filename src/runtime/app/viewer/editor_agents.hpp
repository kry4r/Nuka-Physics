#pragma once
// What agents may do. Programs reaching the editor through its endpoint (nuka.editor.rpc, the
// MCP server) run actions as "agent:<client>", and each run passes Check first: the access mode
// and the allow and deny lists say whether it runs, waits on the user or is refused.

#include "runtime/app/viewer/agent_view.hpp"
#include "runtime/app/viewer/editor_actions.hpp"
#include "scene/format/json.hpp"

#include <cstdint>
#include <filesystem>
#include <list>
#include <map>
#include <string>
#include <vector>

namespace nuka::runtime::app::viewer {

class EditorProfile;

// A proposal or approval with the arguments its action runs with and, once it ran, its value.
struct AgentRequest {
    AgentRequestView view;
    scene::json::Value args;
    scene::json::Value value;
};

class AgentAccess {
public:
    enum class Verdict : uint8_t { Run, Refuse, Propose, Ask };

    // The mode and lists come from settings.toml's [agent] table; the audit goes to its logs/.
    void Load(const EditorProfile& profile);
    const std::string& Mode() const { return mode_; }
    // Sets what `args` names of mode, allow and deny and saves it; throws on an unknown mode.
    void Configure(const scene::json::Value& args, EditorProfile* profile);
    // Adds `id` to the allow list, as approving an action for good does.
    void AllowAlways(const std::string& id, EditorProfile* profile);
    // {mode, allow, deny}.
    scene::json::Value Settings() const;
    // Whether the agent action `id` of `kind` runs, waits on the user or is refused, and why not.
    // An empty kind is an action that does not exist, which the registry refuses itself.
    Verdict Check(const std::string& id, const std::string& kind, std::string* reason) const;

    // Files a pending request and returns its id.
    uint64_t Ask(AgentRequest request);
    // The pending request `id`, of `kind` unless that is empty; throws when there is none. The
    // reference stays valid while other requests come and go.
    AgentRequest& Pending(uint64_t id, const std::string& kind);
    // Settles a request as `state` with the run's outcome, when it ran.
    void Decide(AgentRequest& request, const std::string& state, const ActionResult* result);
    scene::json::Value Describe(const AgentRequest& request) const;
    // Every request kept, pending ones first.
    scene::json::Value Requests() const;
    size_t PendingCount() const;

    // The endpoint's clients as nuka.editor.rpc reports them: [{client, pid, since}].
    void SetClients(const scene::json::Value& clients);
    size_t ClientCount() const { return clients_.size(); }
    // Notes one agent call in the recent list and appends it to the day's audit file.
    void Record(const std::string& client, const std::string& id, const scene::json::Value& args,
                const ActionResult& result, const std::string& outcome, double ms);

    // Copies the state into the panel's view when it changed since the last copy.
    void Fill(AgentView* view) const;

private:
    void Save(EditorProfile* profile) const;

    std::string mode_ = "full";
    std::vector<std::string> allow_;
    std::vector<std::string> deny_;
    std::filesystem::path log_dir_;
    std::list<AgentRequest> requests_;
    std::vector<AgentClientView> clients_;
    std::vector<AgentCallView> calls_;
    std::map<std::string, uint64_t> calls_by_client_;
    uint64_t next_id_ = 1u;
    uint64_t version_ = 1u;
};

}  // namespace nuka::runtime::app::viewer
