#include "runtime/app/viewer/editor_instance.hpp"

#include "runtime/app/viewer/editor_profile.hpp"

#include <algorithm>
#include <mutex>
#include <thread>
#include <utility>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace nuka::runtime::app::viewer {

#if defined(_WIN32)

namespace {

constexpr DWORD kMaxRequest = 64u * 1024u;
constexpr DWORD kBusyWaitMs = 2000u;

std::wstring PipePath(const std::string& name) { return L"\\\\.\\pipe\\" + Utf8Path(name).wstring(); }

// One argument quoted the way the C runtime splits a command line.
std::wstring Quoted(const std::wstring& arg) {
    if (!arg.empty() && arg.find_first_of(L" \t\n\v\"") == std::wstring::npos) return arg;
    std::wstring out = L"\"";
    for (size_t i = 0u;; ++i) {
        size_t slashes = 0u;
        while (i < arg.size() && arg[i] == L'\\') {
            ++i;
            ++slashes;
        }
        if (i == arg.size()) {
            out.append(slashes * 2u, L'\\');
            break;
        }
        out.append(arg[i] == L'"' ? slashes * 2u + 1u : slashes, L'\\');
        out.push_back(arg[i]);
    }
    out.push_back(L'"');
    return out;
}

}  // namespace

struct InstanceServer::Impl {
    HANDLE pipe = INVALID_HANDLE_VALUE;
    HANDLE stop = nullptr;
    std::thread thread;
    std::mutex mu;
    std::vector<std::string> inbox;

    // Waits for `io` to finish; false when the server stops first.
    bool Wait(OVERLAPPED& io) {
        HANDLE events[2] = {io.hEvent, stop};
        if (WaitForMultipleObjects(2u, events, FALSE, INFINITE) == WAIT_OBJECT_0) return true;
        CancelIoEx(pipe, &io);
        DWORD ignored = 0u;
        GetOverlappedResult(pipe, &io, &ignored, TRUE);
        return false;
    }

    // One client at a time: connect, read its one message, disconnect.
    void Serve() {
        OVERLAPPED io{};
        io.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        std::string buffer(kMaxRequest, '\0');
        for (;;) {
            ResetEvent(io.hEvent);
            if (!ConnectNamedPipe(pipe, &io) && GetLastError() == ERROR_IO_PENDING && !Wait(io)) break;
            ResetEvent(io.hEvent);
            if (!ReadFile(pipe, buffer.data(), kMaxRequest, nullptr, &io) && GetLastError() == ERROR_IO_PENDING &&
                !Wait(io))
                break;
            DWORD got = 0u;
            if (GetOverlappedResult(pipe, &io, &got, FALSE) && got > 0u) {
                std::lock_guard<std::mutex> lock(mu);
                inbox.emplace_back(buffer.data(), got);
            }
            DisconnectNamedPipe(pipe);
        }
        CloseHandle(io.hEvent);
    }
};

InstanceServer::InstanceServer(const std::string& name) : impl_(std::make_unique<Impl>()) {
    Impl& m = *impl_;
    m.pipe = CreateNamedPipeW(PipePath(name).c_str(), PIPE_ACCESS_INBOUND | FILE_FLAG_FIRST_PIPE_INSTANCE | FILE_FLAG_OVERLAPPED,
                              PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS, 1u, 0u,
                              kMaxRequest, 0u, nullptr);
    if (m.pipe == INVALID_HANDLE_VALUE) return;
    m.stop = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    m.thread = std::thread([&m] { m.Serve(); });
}

InstanceServer::~InstanceServer() {
    Impl& m = *impl_;
    if (m.pipe == INVALID_HANDLE_VALUE) return;
    SetEvent(m.stop);
    m.thread.join();
    CloseHandle(m.pipe);
    CloseHandle(m.stop);
}

bool InstanceServer::Listening() const { return impl_->pipe != INVALID_HANDLE_VALUE; }

std::vector<std::string> InstanceServer::Take() {
    std::lock_guard<std::mutex> lock(impl_->mu);
    std::vector<std::string> out;
    out.swap(impl_->inbox);
    return out;
}

bool SendToPrimary(const std::string& name, const std::string& request) {
    const std::wstring path = PipePath(name);
    HANDLE pipe = CreateFileW(path.c_str(), GENERIC_WRITE, 0u, nullptr, OPEN_EXISTING, 0u, nullptr);
    // The primary serves one client at a time, so a busy pipe frees up shortly.
    if (pipe == INVALID_HANDLE_VALUE && GetLastError() == ERROR_PIPE_BUSY && WaitNamedPipeW(path.c_str(), kBusyWaitMs))
        pipe = CreateFileW(path.c_str(), GENERIC_WRITE, 0u, nullptr, OPEN_EXISTING, 0u, nullptr);
    if (pipe == INVALID_HANDLE_VALUE) return false;
    ULONG server = 0u;
    if (GetNamedPipeServerProcessId(pipe, &server)) AllowSetForegroundWindow(server);
    DWORD wrote = 0u;
    const BOOL ok = WriteFile(pipe, request.data(), static_cast<DWORD>(request.size()), &wrote, nullptr);
    CloseHandle(pipe);
    return ok && wrote == request.size();
}

uint64_t SpawnDetached(const std::vector<std::string>& argv, std::string* error) {
    std::wstring line;
    for (const std::string& arg : argv) line += (line.empty() ? L"" : L" ") + Quoted(Utf8Path(arg).wstring());
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    if (!CreateProcessW(nullptr, line.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &startup,
                        &process)) {
        *error = "cannot start " + argv.front() + " (Windows error " + std::to_string(GetLastError()) + ")";
        return 0u;
    }
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    return process.dwProcessId;
}

uint32_t ProcessId() { return GetCurrentProcessId(); }

bool ProcessAlive(uint32_t pid) {
    HANDLE process = OpenProcess(SYNCHRONIZE, FALSE, pid);
    if (process == nullptr) return false;
    const bool alive = WaitForSingleObject(process, 0u) == WAIT_TIMEOUT;
    CloseHandle(process);
    return alive;
}

struct PipedProcess::Impl {
    HANDLE process = nullptr;
    HANDLE input = nullptr;
};

PipedProcess::PipedProcess(const std::vector<std::string>& argv, const std::string& log, std::string* error)
    : impl_(std::make_unique<Impl>()) {
    SECURITY_ATTRIBUTES inherit{sizeof(inherit), nullptr, TRUE};
    HANDLE read = nullptr;
    if (!CreatePipe(&read, &impl_->input, &inherit, 0u)) {
        *error = "cannot create a pipe (Windows error " + std::to_string(GetLastError()) + ")";
        return;
    }
    SetHandleInformation(impl_->input, HANDLE_FLAG_INHERIT, 0u);
    HANDLE out = CreateFileW(Utf8Path(log).wstring().c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE,
                             &inherit, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (out == INVALID_HANDLE_VALUE) {
        *error = "cannot open " + log + " (Windows error " + std::to_string(GetLastError()) + ")";
        CloseHandle(read);
        CloseHandle(std::exchange(impl_->input, nullptr));
        return;
    }
    // The child inherits these two handles and no others.
    HANDLE handles[2] = {read, out};
    SIZE_T size = 0u;
    InitializeProcThreadAttributeList(nullptr, 1u, 0u, &size);
    std::vector<char> storage(size);
    auto* attributes = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(storage.data());
    InitializeProcThreadAttributeList(attributes, 1u, 0u, &size);
    UpdateProcThreadAttribute(attributes, 0u, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, handles, sizeof(handles), nullptr,
                              nullptr);
    STARTUPINFOEXW startup{};
    startup.StartupInfo.cb = sizeof(startup);
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdInput = read;
    startup.StartupInfo.hStdOutput = out;
    startup.StartupInfo.hStdError = out;
    startup.lpAttributeList = attributes;
    std::wstring line;
    for (const std::string& arg : argv) line += (line.empty() ? L"" : L" ") + Quoted(Utf8Path(arg).wstring());
    PROCESS_INFORMATION process{};
    const BOOL started = CreateProcessW(nullptr, line.data(), nullptr, nullptr, TRUE,
                                        CREATE_NO_WINDOW | EXTENDED_STARTUPINFO_PRESENT, nullptr, nullptr,
                                        &startup.StartupInfo, &process);
    const DWORD why = GetLastError();
    DeleteProcThreadAttributeList(attributes);
    CloseHandle(read);
    CloseHandle(out);
    if (!started) {
        *error = "cannot start " + argv.front() + " (Windows error " + std::to_string(why) + ")";
        CloseHandle(std::exchange(impl_->input, nullptr));
        return;
    }
    CloseHandle(process.hThread);
    impl_->process = process.hProcess;
}

PipedProcess::~PipedProcess() { Finish(); }

bool PipedProcess::Started() const { return impl_->process != nullptr; }

bool PipedProcess::Write(const void* data, size_t bytes) {
    const char* at = static_cast<const char*>(data);
    while (bytes > 0u) {
        DWORD wrote = 0u;
        const DWORD chunk = static_cast<DWORD>(std::min<size_t>(bytes, size_t{1} << 30));
        if (impl_->input == nullptr || !WriteFile(impl_->input, at, chunk, &wrote, nullptr)) return false;
        at += wrote;
        bytes -= wrote;
    }
    return true;
}

int PipedProcess::Finish() {
    if (impl_->input != nullptr) CloseHandle(std::exchange(impl_->input, nullptr));
    if (impl_->process == nullptr) return -1;
    WaitForSingleObject(impl_->process, INFINITE);
    DWORD code = 0u;
    GetExitCodeProcess(impl_->process, &code);
    CloseHandle(std::exchange(impl_->process, nullptr));
    return static_cast<int>(code);
}

#else

// No Unix-socket primary yet: every launch on this platform runs on its own.
struct InstanceServer::Impl {};

InstanceServer::InstanceServer(const std::string&) : impl_(std::make_unique<Impl>()) {}
InstanceServer::~InstanceServer() = default;
bool InstanceServer::Listening() const { return false; }
std::vector<std::string> InstanceServer::Take() { return {}; }

bool SendToPrimary(const std::string&, const std::string&) { return false; }

uint64_t SpawnDetached(const std::vector<std::string>& argv, std::string* error) {
    std::vector<char*> args;
    for (const std::string& arg : argv) args.push_back(const_cast<char*>(arg.c_str()));
    args.push_back(nullptr);
    pid_t pid = 0;
    const int failed = posix_spawnp(&pid, args.front(), nullptr, nullptr, args.data(), environ);
    if (failed != 0) {
        *error = "cannot start " + argv.front() + " (errno " + std::to_string(failed) + ")";
        return 0u;
    }
    return static_cast<uint64_t>(pid);
}

uint32_t ProcessId() { return static_cast<uint32_t>(getpid()); }

bool ProcessAlive(uint32_t pid) { return kill(static_cast<pid_t>(pid), 0) == 0; }

struct PipedProcess::Impl {
    pid_t pid = 0;
    int input = -1;
};

PipedProcess::PipedProcess(const std::vector<std::string>& argv, const std::string& log, std::string* error)
    : impl_(std::make_unique<Impl>()) {
    int ends[2] = {-1, -1};
    if (pipe(ends) != 0) {
        *error = "cannot create a pipe (errno " + std::to_string(errno) + ")";
        return;
    }
    fcntl(ends[1], F_SETFD, FD_CLOEXEC);
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, ends[0], 0);
    posix_spawn_file_actions_addclose(&actions, ends[0]);
    posix_spawn_file_actions_addopen(&actions, 1, log.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    posix_spawn_file_actions_adddup2(&actions, 1, 2);
    std::vector<char*> args;
    for (const std::string& arg : argv) args.push_back(const_cast<char*>(arg.c_str()));
    args.push_back(nullptr);
    const int failed = posix_spawnp(&impl_->pid, args.front(), &actions, nullptr, args.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    close(ends[0]);
    if (failed != 0) {
        *error = "cannot start " + argv.front() + " (errno " + std::to_string(failed) + ")";
        close(ends[1]);
        impl_->pid = 0;
        return;
    }
    impl_->input = ends[1];
}

PipedProcess::~PipedProcess() { Finish(); }

bool PipedProcess::Started() const { return impl_->pid != 0; }

bool PipedProcess::Write(const void* data, size_t bytes) {
    // A closed pipe then answers EPIPE instead of a SIGPIPE that would end the editor.
    sigset_t pipe_signal;
    sigemptyset(&pipe_signal);
    sigaddset(&pipe_signal, SIGPIPE);
    pthread_sigmask(SIG_BLOCK, &pipe_signal, nullptr);
    const char* at = static_cast<const char*>(data);
    while (bytes > 0u) {
        const ssize_t wrote = impl_->input < 0 ? -1 : write(impl_->input, at, bytes);
        if (wrote < 0 && errno == EINTR) continue;
        if (wrote <= 0) return false;
        at += wrote;
        bytes -= static_cast<size_t>(wrote);
    }
    return true;
}

int PipedProcess::Finish() {
    if (impl_->input >= 0) close(std::exchange(impl_->input, -1));
    if (impl_->pid == 0) return -1;
    int status = 0;
    while (waitpid(impl_->pid, &status, 0) < 0 && errno == EINTR) {
    }
    impl_->pid = 0;
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

#endif

}  // namespace nuka::runtime::app::viewer
