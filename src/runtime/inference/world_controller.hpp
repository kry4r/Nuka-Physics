#pragma once
// A per-step control hook the editor calls before each physics step. It reads and
// writes world fields through the C ABI and owns its own control decimation.

#include "nuka/nuka.h"

#include <cstdint>

namespace nuka::runtime::inference {

class WorldController {
public:
    virtual ~WorldController() = default;

    virtual void OnStep(nuka_world_handle world, uint32_t env_index) = 0;

    // A command consumer (e.g. a locomotion policy taking [vx, vy, wyaw]) overrides these.
    virtual bool AcceptsCommand() const { return false; }
    virtual void SetCommandVector(const float* cmd, uint32_t count) {
        (void)cmd;
        (void)count;
    }
};

}  // namespace nuka::runtime::inference
