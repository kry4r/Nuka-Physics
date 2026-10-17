#pragma once
// The primary editor of a profile and the launches that hand it files: a local pipe that
// takes only {"open": path} requests. Also starts detached editor processes.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nuka::runtime::app::viewer {

class InstanceServer {
public:
    // Listens on `name` unless another process already does; then Listening() is false.
    explicit InstanceServer(const std::string& name);
    ~InstanceServer();
    InstanceServer(const InstanceServer&) = delete;
    InstanceServer& operator=(const InstanceServer&) = delete;

    bool Listening() const;
    // The requests received since the last call, oldest first.
    std::vector<std::string> Take();

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

// Sends one request to the primary editor listening on `name`; false when none does. The
// primary may then take the foreground.
bool SendToPrimary(const std::string& name, const std::string& request);

// Starts `argv` detached from this process and returns its id, or 0 with `error` set.
uint64_t SpawnDetached(const std::vector<std::string>& argv, std::string* error);

uint32_t ProcessId();
bool ProcessAlive(uint32_t pid);

}  // namespace nuka::runtime::app::viewer
