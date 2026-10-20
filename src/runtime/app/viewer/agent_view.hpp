#pragma once
// What the Agents panel shows: the access mode, the programs connected through the editor's
// endpoint, what they asked the user for and the actions they ran.

#include <cstdint>
#include <string>
#include <vector>

namespace nuka::runtime::app::viewer {

// One field a proposal changes, as text.
struct AgentChange {
    std::string target;
    std::string field;
    std::string before;  // empty when the document has no such value to show
    std::string after;
};

// An agent's proposal (an edit to accept or reject) or approval (an action to allow or deny).
struct AgentRequestView {
    uint64_t id = 0u;
    std::string kind;  // "proposal" or "approval"
    std::string client;
    std::string action;
    std::string args;  // the action's arguments as JSON text
    std::string label;
    std::string note;
    std::vector<AgentChange> changes;
    std::string state;  // pending, accepted, approved, rejected or failed
    std::string error;
    int64_t time = 0;  // seconds since the epoch
};

struct AgentClientView {
    std::string name;
    int64_t pid = 0;
    int64_t since = 0;  // seconds since the epoch
    uint64_t calls = 0u;
};

struct AgentCallView {
    int64_t time = 0;
    std::string client;
    std::string action;
    std::string outcome;  // ran, refused, accepted, approved, or the request it became
    bool ok = true;
    double ms = 0.0;
};

struct AgentView {
    std::string mode = "full";  // off, read, propose or full
    std::vector<std::string> allow;
    std::vector<std::string> deny;
    char allow_edit[512] = {};
    char deny_edit[512] = {};
    std::vector<AgentClientView> clients;
    std::vector<AgentRequestView> requests;  // pending first, then the latest decided
    std::vector<AgentCallView> calls;        // newest first
    size_t pending = 0u;
    uint64_t version = 0u;  // bumped by the host with every change
};

}  // namespace nuka::runtime::app::viewer
