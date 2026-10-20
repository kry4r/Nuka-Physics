#pragma once
// The primary editor of a profile and the launches that hand it files: a local pipe that
// takes only {"open": path} requests. Also starts detached and piped child processes.

#include <cstddef>
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

// A child process fed through its standard input; its output goes to a log file.
class PipedProcess {
public:
    // Starts `argv` appending its output to `log`; on failure Started() is false and `error` says why.
    PipedProcess(const std::vector<std::string>& argv, const std::string& log, std::string* error);
    ~PipedProcess();
    PipedProcess(const PipedProcess&) = delete;
    PipedProcess& operator=(const PipedProcess&) = delete;

    bool Started() const;
    // Writes all `bytes`; false once the child stopped reading.
    bool Write(const void* data, size_t bytes);
    // Closes the child's input, waits for it to exit and returns its exit code.
    int Finish();

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

uint32_t ProcessId();
bool ProcessAlive(uint32_t pid);

}  // namespace nuka::runtime::app::viewer
