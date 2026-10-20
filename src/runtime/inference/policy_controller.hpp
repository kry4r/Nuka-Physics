#pragma once
// A trained locomotion policy the editor runs: robots in one environment, each reading a
// [vx, vy, wyaw] command and writing drive targets as default + scale x action.

#include "runtime/inference/world_controller.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace nuka::runtime::inference {

class PolicyController : public WorldController {
public:
    // Each robot's root link slot, robot after robot.
    virtual const std::vector<uint32_t>& RootLinks() const = 0;
    // Each action's link slot and default joint position, robot after robot.
    virtual const std::vector<uint32_t>& ActionLinks() const = 0;
    virtual const std::vector<float>& ActionDefaults() const = 0;
    virtual uint32_t ObsDim() const = 0;
    virtual uint32_t ActDim() const = 0;
    virtual uint32_t Decimation() const = 0;
    virtual float ActionScale() const = 0;
    virtual bool UsingGpu() const = 0;
    // The command it reads; only on the thread that steps the world.
    virtual std::array<float, 3> Command() const = 0;

    uint32_t Robots() const { return static_cast<uint32_t>(RootLinks().size()); }
    bool AcceptsCommand() const override { return true; }
};

}  // namespace nuka::runtime::inference
