#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::app::CommandQueue - thread-safe MPSC command queue (M8 #5).
//
// The frame loop's InputSystem drains this queue once per frame to apply
// out-of-band edits to the world/scene (e.g. the M11 viewer dragging an entity
// emits a MoveEntity). M8 ships only the skeleton: the Command type + a minimal
// many-producer / single-consumer queue. No M8 system drives it yet --
// InputSystem drains it as a stub in M8 Phase 2; the real producers (the M11
// viewer / input devices) arrive later. Anticipating that, the Command variant
// already carries MoveEntity (recon §2 #5 / plan L509) so the queue's public
// shape does not churn when the viewer lands.
//
// Header-only, zero external deps, ZERO CUDA (this dir is under the
// src/runtime/app/** zero-CUDA-token lint red-line): only the C++ standard
// library (<mutex>/<deque>) is used.
// ---------------------------------------------------------------------------

#include "math/transform.hpp"
#include "scene/ecs/entity.hpp"

#include <cstdint>
#include <deque>
#include <mutex>
#include <utility>
#include <vector>

namespace nuka::runtime::app {

// -- command payloads --------------------------------------------------------

// Re-pose an entity (the M11 viewer's drag-to-move; an editor gizmo write).
// Carries the target world transform; the consumer maps entity -> cooked row and
// writes the pose through the appropriate seam.
struct MoveEntity {
    scene::EntityId entity = scene::kInvalidEntity;
    math::Transform world_xform = math::Transform::Identity();
};

// A no-op marker (default-constructed Command). Lets the queue hold a valid
// "nothing" command and keeps Command default-constructible.
struct NoOp {};

// ---------------------------------------------------------------------------
// Command - a tagged union over the supported command payloads. Kept as an
// explicit tag + members (rather than std::variant) for a trivially-copyable,
// dependency-light, ABI-stable shape that mirrors the engine's closed-set style
// (cf. the ECS component set). New command kinds append a tag + a member.
// ---------------------------------------------------------------------------
struct Command {
    enum class Kind : uint8_t { NoOp, MoveEntity };

    Kind kind = Kind::NoOp;

    // Payload union-by-membership (only the member matching `kind` is valid).
    MoveEntity move_entity{};

    Command() = default;

    static Command MakeNoOp() { return Command{}; }

    static Command MakeMoveEntity(scene::EntityId e, const math::Transform& xform) {
        Command c;
        c.kind = Kind::MoveEntity;
        c.move_entity = MoveEntity{e, xform};
        return c;
    }
};

// ---------------------------------------------------------------------------
// CommandQueue - thread-safe multi-producer / single-consumer queue.
//
// Any number of producer threads call Push concurrently; ONE consumer thread
// (the frame loop's InputSystem) calls Drain once per frame to take the whole
// pending batch in FIFO order. A single mutex guards a deque -- ample for the
// low command rate of an interactive viewer, and simple enough to stay
// obviously correct (D1 / determinism: Drain returns commands in exact push
// order on the single consumer).
// ---------------------------------------------------------------------------
class CommandQueue {
public:
    // Enqueue a command (any producer thread). Thread-safe.
    void Push(const Command& cmd) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push_back(cmd);
    }

    void Push(Command&& cmd) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push_back(std::move(cmd));
    }

    // Atomically take all pending commands in FIFO order, leaving the queue
    // empty (single consumer thread). Returns an empty vector when idle.
    std::vector<Command> Drain() {
        std::vector<Command> out;
        std::lock_guard<std::mutex> lock(mutex_);
        out.reserve(queue_.size());
        while (!queue_.empty()) {
            out.push_back(std::move(queue_.front()));
            queue_.pop_front();
        }
        return out;
    }

    // Snapshot count (advisory; a producer may change it immediately after).
    bool Empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    size_t Size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    mutable std::mutex   mutex_;
    std::deque<Command>  queue_;
};

}  // namespace nuka::runtime::app
