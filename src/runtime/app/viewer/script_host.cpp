#include "runtime/app/viewer/script_host.hpp"

#include <cstdio>
#include <utility>

namespace nuka::runtime::app::viewer {
namespace {

std::string Text(const char* text) { return text != nullptr ? text : ""; }

}  // namespace

ScriptHost::~ScriptHost() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    if (watchdog_.joinable()) watchdog_.join();
}

void ScriptHost::SetHooks(const nuka_editor_script_hooks_t& hooks) {
    {
        std::lock_guard<std::mutex> lock(mu_);
        hooks_ = hooks;
    }
    if (!watchdog_.joinable()) watchdog_ = std::thread([this] { Watch(); });
}

// The outermost call of those in progress is the one the budget times.
template <class Fn>
std::string ScriptHost::Timed(Fn&& fn) {
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (depth_++ == 0) {
            since_ = Clock::now();
            why_.clear();
        }
    }
    std::string text = fn();
    std::lock_guard<std::mutex> lock(mu_);
    if (--depth_ == 0 && !why_.empty()) text += "[script] stopped: " + why_ + "\n";
    return text;
}

std::string ScriptHost::RunString(const std::string& code, const std::string& filename, bool echo, bool* failed) {
    int32_t raised = 0;
    std::string text;
    if (Valid())
        text = Timed([&] {
            return Text(hooks_.run(hooks_.user_data, code.c_str(), filename.c_str(), echo ? 1u : 0u, &raised));
        });
    if (failed != nullptr) *failed = raised != 0;
    return text;
}

std::string ScriptHost::RegisterScript(uint64_t script_id, const std::string& source) {
    if (!Valid()) return {};
    return Timed([&] { return Text(hooks_.register_script(hooks_.user_data, script_id, source.c_str())); });
}

void ScriptHost::UnregisterScript(uint64_t script_id) {
    if (Valid()) hooks_.unregister_script(hooks_.user_data, script_id);
}

void ScriptHost::ClearScripts() {
    if (Valid()) hooks_.clear_scripts(hooks_.user_data);
}

std::string ScriptHost::Dispatch(nuka_editor_callback_t callback) {
    if (!Valid()) return {};
    return Timed([&] { return Text(hooks_.dispatch(hooks_.user_data, callback)); });
}

std::string ScriptHost::Complete(const std::string& code, int line, int column) {
    if (!Valid()) return "[]";
    const char* text = hooks_.complete(hooks_.user_data, code.c_str(), line, column);
    return text != nullptr ? text : "[]";
}

bool ScriptHost::Interrupt(const std::string& why) {
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (depth_ == 0) return false;
        why_ = why;
    }
    hooks_.interrupt(hooks_.user_data);
    return true;
}

void ScriptHost::SetBudget(double seconds) {
    std::lock_guard<std::mutex> lock(mu_);
    budget_s_ = seconds;
}

double ScriptHost::Budget() const {
    std::lock_guard<std::mutex> lock(mu_);
    return budget_s_;
}

double ScriptHost::BusySeconds() const {
    std::lock_guard<std::mutex> lock(mu_);
    return depth_ > 0 ? std::chrono::duration<double>(Clock::now() - since_).count() : 0.0;
}

// The interrupt hook takes the GIL, so it is called with the lock released.
void ScriptHost::Watch() {
    std::unique_lock<std::mutex> lock(mu_);
    while (!stop_) {
        cv_.wait_for(lock, std::chrono::milliseconds(50));
        if (depth_ == 0 || !why_.empty() || budget_s_ <= 0.0 ||
            std::chrono::duration<double>(Clock::now() - since_).count() <= budget_s_)
            continue;
        char why[96];
        std::snprintf(why, sizeof(why), "ran past the %g s budget (script.set_budget)", budget_s_);
        why_ = why;
        lock.unlock();
        hooks_.interrupt(hooks_.user_data);
        lock.lock();
    }
}

ScriptCompleter::ScriptCompleter(ScriptHost* scripts) : scripts_(scripts), thread_([this] { Loop(); }) {}

ScriptCompleter::~ScriptCompleter() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    thread_.join();
}

uint64_t ScriptCompleter::Request(std::string code, int line, int column) {
    uint64_t ticket = 0u;
    {
        std::lock_guard<std::mutex> lock(mu_);
        ticket = ++asked_;
        code_ = std::move(code);
        line_ = line;
        column_ = column;
    }
    cv_.notify_all();
    return ticket;
}

bool ScriptCompleter::Take(uint64_t ticket, std::string* json) {
    std::lock_guard<std::mutex> lock(mu_);
    if (answered_ != ticket) return false;
    *json = std::move(answer_);
    answered_ = 0u;
    return true;
}

void ScriptCompleter::Loop() {
    std::unique_lock<std::mutex> lock(mu_);
    uint64_t done = 0u;
    while (true) {
        cv_.wait(lock, [&] { return stop_ || asked_ != done; });
        if (stop_) return;
        const uint64_t ticket = asked_;
        const std::string code = code_;
        const int line = line_, column = column_;
        lock.unlock();
        std::string answer = scripts_->Complete(code, line, column);
        lock.lock();
        done = ticket;
        answered_ = ticket;
        answer_ = std::move(answer);
    }
}

}  // namespace nuka::runtime::app::viewer
