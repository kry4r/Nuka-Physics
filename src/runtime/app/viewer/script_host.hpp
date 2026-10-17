#pragma once
// Console and lifecycle scripts run in the host process through the hooks it installs;
// a host that installs none leaves the editor without scripting. A watchdog interrupts a
// hook call that outlasts the budget.

#include "nuka/nuka_editor.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

namespace nuka::runtime::app::viewer {

class ScriptHost {
public:
    ScriptHost() = default;
    ~ScriptHost();
    ScriptHost(const ScriptHost&) = delete;
    ScriptHost& operator=(const ScriptHost&) = delete;

    void SetHooks(const nuka_editor_script_hooks_t& hooks);
    bool Valid() const { return hooks_.run != nullptr; }

    // An empty `filename` runs as the console; `echo` prints the value of a final expression.
    std::string RunString(const std::string& code, const std::string& filename = std::string(), bool echo = false,
                          bool* failed = nullptr);
    std::string RegisterScript(uint64_t script_id, const std::string& source);
    void UnregisterScript(uint64_t script_id);
    void ClearScripts();
    std::string DispatchReady() { return Dispatch(NUKA_EDITOR_CALLBACK_READY); }
    std::string DispatchStep() { return Dispatch(NUKA_EDITOR_CALLBACK_STEP); }
    std::string DispatchReset() { return Dispatch(NUKA_EDITOR_CALLBACK_RESET); }
    std::string DispatchTick() { return Dispatch(NUKA_EDITOR_CALLBACK_TICK); }
    // The completions as JSON text; any thread may ask.
    std::string Complete(const std::string& code, int line, int column);

    // Stops the hook call in progress, whose output then ends with `why`; false when none runs.
    bool Interrupt(const std::string& why);
    // Seconds a hook call may run before the watchdog stops it; 0 lets it run.
    void SetBudget(double seconds);
    double Budget() const;
    // How long the hook call in progress has run; 0 when none runs.
    double BusySeconds() const;

private:
    using Clock = std::chrono::steady_clock;
    template <class Fn>
    std::string Timed(Fn&& fn);
    std::string Dispatch(nuka_editor_callback_t callback);
    void Watch();

    nuka_editor_script_hooks_t hooks_{};
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::thread watchdog_;
    bool stop_ = false;
    int depth_ = 0;  // hook calls in progress, across threads; the outermost one is timed
    Clock::time_point since_{};
    std::string why_;
    double budget_s_ = 10.0;
};

// Completion requests answered on a thread of their own, so typing never waits on Python.
// Only the newest request matters: an older one still queued is dropped.
class ScriptCompleter {
public:
    explicit ScriptCompleter(ScriptHost* scripts);
    ~ScriptCompleter();
    ScriptCompleter(const ScriptCompleter&) = delete;
    ScriptCompleter& operator=(const ScriptCompleter&) = delete;

    // The ticket whose answer Take hands back.
    uint64_t Request(std::string code, int line, int column);
    // The JSON answer for `ticket` once it is ready.
    bool Take(uint64_t ticket, std::string* json);

private:
    void Loop();

    ScriptHost* scripts_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::thread thread_;
    bool stop_ = false;
    uint64_t asked_ = 0u;
    uint64_t answered_ = 0u;
    std::string code_;
    int line_ = 0;
    int column_ = 0;
    std::string answer_;
};

}  // namespace nuka::runtime::app::viewer
