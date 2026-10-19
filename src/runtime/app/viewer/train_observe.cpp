#include "runtime/app/viewer/train_observe.hpp"

#include <atomic>
#include <cstring>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <windows.h>
#endif

namespace nuka::runtime::app::viewer {

namespace json = scene::json;

#if defined(_WIN32)

namespace {

constexpr uint32_t kMagic = 0x424F4B4Eu;
constexpr uint32_t kVersion = 1u;
constexpr size_t kHeader = 64u;
constexpr size_t kSlotHead = 32u;
constexpr DWORD kHandshakeMs = 5000u;
// A frame the writer keeps overwriting is skipped after this many tries; the next call tries again.
constexpr int kTries = 3;

uint32_t U32(const uint8_t* p) {
    uint32_t v = 0u;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

uint64_t Load(const uint8_t* p) {
    const uint64_t v = *reinterpret_cast<const volatile uint64_t*>(p);
    std::atomic_thread_fence(std::memory_order_acquire);
    return v;
}

}  // namespace

struct TrainObserve::Impl {
    bool started = false;
    SOCKET socket = INVALID_SOCKET;
    HANDLE mapping = nullptr;
    const uint8_t* view = nullptr;
    json::Value answer;
    terrain::HeightField terrain;
    uint32_t slots = 0u, envs = 0u, links = 0u, bodies = 0u, articulations = 0u, viewer_bodies = 0u;
    size_t slot_bytes = 0u, floats = 0u;
    std::vector<float> scratch;

    ~Impl() {
        if (view != nullptr) UnmapViewOfFile(view);
        if (mapping != nullptr) CloseHandle(mapping);
        if (socket != INVALID_SOCKET) closesocket(socket);
        if (started) WSACleanup();
    }
};

TrainObserve::TrainObserve(uint16_t port, const json::Value& hello) : impl_(std::make_unique<Impl>()) {
    Impl& m = *impl_;
    WSADATA wsa{};
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) throw std::runtime_error("Winsock did not start");
    m.started = true;
    m.socket = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (m.socket == INVALID_SOCKET) throw std::runtime_error("no socket for the run's port");
    const DWORD timeout = kHandshakeMs;
    setsockopt(m.socket, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeout), sizeof(timeout));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(m.socket, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) != 0)
        throw std::runtime_error("the run does not answer on port " + std::to_string(port));
    const std::string line = hello.Dump(0) + "\n";
    send(m.socket, line.data(), static_cast<int>(line.size()), 0);
    std::string reply;
    char c = 0;
    while (recv(m.socket, &c, 1, 0) == 1 && c != '\n') reply.push_back(c);
    if (c != '\n') throw std::runtime_error("the run closed the connection without answering");
    m.answer = json::Value::Parse(reply);
    if (!m.answer.At("ok").AsBool()) throw std::runtime_error(m.answer.At("error").AsString());
    const std::string shm = m.answer.At("shm").AsString();
    m.mapping = OpenFileMappingA(FILE_MAP_READ, FALSE, shm.c_str());
    if (m.mapping == nullptr) throw std::runtime_error("the run's shared memory " + shm + " is gone");
    const auto bytes = static_cast<size_t>(m.answer.At("bytes").AsInt());
    m.view = static_cast<const uint8_t*>(MapViewOfFile(m.mapping, FILE_MAP_READ, 0, 0, bytes));
    if (m.view == nullptr) throw std::runtime_error("the run's shared memory " + shm + " cannot be mapped");
    if (U32(m.view) != kMagic || U32(m.view + 4u) != kVersion)
        throw std::runtime_error("the run's shared memory is not nuka.observe.v1");
    m.slots = U32(m.view + 8u);
    m.envs = U32(m.view + 12u);
    m.links = U32(m.view + 16u);
    m.bodies = U32(m.view + 20u);
    m.articulations = U32(m.view + 24u);
    m.slot_bytes = U32(m.view + 28u);
    m.viewer_bodies = static_cast<uint32_t>(hello.At("layout").At("bodies").AsInt());
    m.floats = size_t{m.envs} * ((size_t{m.links} + m.bodies + m.articulations) * 7u + m.links + 6u);
    const json::Value& grid = m.answer.At("terrain");
    if (!grid.IsNull()) {
        terrain::HeightField& t = m.terrain;
        t.nrow = static_cast<uint32_t>(grid.At("nrow").AsInt());
        t.ncol = static_cast<uint32_t>(grid.At("ncol").AsInt());
        t.cell_x = grid.At("cell_x").AsFloat();
        t.cell_y = grid.At("cell_y").AsFloat();
        const std::vector<json::Value>& origin = grid.At("origin").Elements();
        t.origin = {origin[0].AsFloat(), origin[1].AsFloat(), origin[2].AsFloat()};
        t.base_z = grid.At("base_z").AsFloat();
        t.scale_z = grid.At("scale_z").AsFloat();
        const auto* heights = reinterpret_cast<const float*>(m.view + grid.At("offset").AsInt());
        t.values.assign(heights, heights + size_t{t.nrow} * t.ncol);
    }
    // From here the connection only tells whether the run is still there.
    u_long nonblocking = 1u;
    ioctlsocket(m.socket, FIONBIO, &nonblocking);
}

TrainObserve::~TrainObserve() = default;

const json::Value& TrainObserve::Answer() const { return impl_->answer; }

const terrain::HeightField& TrainObserve::Terrain() const { return impl_->terrain; }

bool TrainObserve::Read(ObserveFrame* out) {
    Impl& m = *impl_;
    for (int attempt = 0; attempt < kTries; ++attempt) {
        const uint64_t n = Load(m.view + 32u);
        if (n == 0u || n == out->sequence) return false;
        const uint8_t* slot = m.view + kHeader + (n % m.slots) * m.slot_bytes;
        if (Load(slot + 16u) != n) continue;
        m.scratch.resize(m.floats);
        std::memcpy(m.scratch.data(), slot + kSlotHead, m.floats * sizeof(float));
        const uint64_t steps = Load(slot + 8u);
        if (Load(slot) != n) continue;
        const size_t envs = m.envs;
        const float* at = m.scratch.data();
        const auto take = [&](std::vector<float>& v, size_t count) {
            v.assign(at, at + count);
            at += count;
        };
        take(out->links, envs * m.links * 7u);
        // Each environment's first bodies are the scene's; the run's own come after them.
        const size_t kept = size_t{m.viewer_bodies} * 7u;
        out->bodies.resize(envs * kept);
        for (size_t e = 0; e < envs; ++e)
            std::memcpy(out->bodies.data() + e * kept, at + e * m.bodies * 7u, kept * sizeof(float));
        at += envs * m.bodies * 7u;
        take(out->bases, envs * m.articulations * 7u);
        take(out->joints, envs * m.links);
        take(out->command, envs * 3u);
        take(out->velocity, envs * 3u);
        out->sequence = n;
        out->steps = steps;
        return true;
    }
    return false;
}

bool TrainObserve::Connected() {
    char c = 0;
    const int r = recv(impl_->socket, &c, 1, MSG_PEEK);
    return r > 0 || (r == SOCKET_ERROR && WSAGetLastError() == WSAEWOULDBLOCK);
}

#else

struct TrainObserve::Impl {
    json::Value answer;
    terrain::HeightField terrain;
};

TrainObserve::TrainObserve(uint16_t, const json::Value&) : impl_(std::make_unique<Impl>()) {
    throw std::runtime_error("watching a training run needs Windows named shared memory");
}

TrainObserve::~TrainObserve() = default;

const json::Value& TrainObserve::Answer() const { return impl_->answer; }

const terrain::HeightField& TrainObserve::Terrain() const { return impl_->terrain; }

bool TrainObserve::Read(ObserveFrame*) { return false; }

bool TrainObserve::Connected() { return false; }

#endif

}  // namespace nuka::runtime::app::viewer
