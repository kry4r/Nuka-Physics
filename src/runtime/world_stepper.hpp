#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::world_stepper -- fixed-step CPU stepping for WorldInstance
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "runtime/world_instance.hpp"
#include "runtime/world_template.hpp"

#include <cstdint>

namespace nuka::runtime {

struct WorldStepOptions {
    math::Vec3 gravity = {0.0f, -9.81f, 0.0f};
    float dt = 1.0f / 60.0f;
    uint32_t step_count = 1;
    bool clear_forces_after_step = true;
};

void StepWorldInstance(const WorldTemplate& world_template,
                       WorldInstance& instance,
                       const WorldStepOptions& options = {});

} // namespace nuka::runtime
