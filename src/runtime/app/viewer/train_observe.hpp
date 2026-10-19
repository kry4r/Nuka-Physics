#pragma once
// A viewer of a training run over nuka.observe.v1, which python/nuka/train/observe.py documents:
// the loopback handshake, then whole frames copied out of the run's shared-memory ring.

#include "scene/format/json.hpp"
#include "scene/terrain/heightfield.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace nuka::runtime::app::viewer {

// The run's first environments at one moment, each array environment after environment.
struct ObserveFrame {
    uint64_t sequence = 0u;
    uint64_t steps = 0u;          // the run's environment steps when it was taken
    std::vector<float> links;     // envs x links x 7: position, quaternion w x y z
    std::vector<float> bodies;    // envs x the viewer's bodies x 7
    std::vector<float> bases;     // envs x articulations x 7
    std::vector<float> joints;    // envs x links
    std::vector<float> command;   // envs x 3: world vx, vy and yaw rate
    std::vector<float> velocity;  // envs x 3
};

class TrainObserve {
public:
    // Hands `hello` to the run listening on `port`; throws std::runtime_error with the run's reason
    // when it refuses, or with why it cannot be reached.
    TrainObserve(uint16_t port, const scene::json::Value& hello);
    ~TrainObserve();
    TrainObserve(const TrainObserve&) = delete;
    TrainObserve& operator=(const TrainObserve&) = delete;

    // The run's answer: envs, links, bodies, articulations, run_envs, step_dt and spacing.
    const scene::json::Value& Answer() const;
    // The run's height grid; invalid when the run has none.
    const terrain::HeightField& Terrain() const;
    // Copies the newest whole frame into `out` when it is newer than `out->sequence`.
    bool Read(ObserveFrame* out);
    // False once the run closed the connection.
    bool Connected();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nuka::runtime::app::viewer
