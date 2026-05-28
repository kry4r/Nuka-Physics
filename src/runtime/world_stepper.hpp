#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::world_stepper -- fixed-step CPU stepping for WorldInstance
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "runtime/world_instance.hpp"
#include "runtime/world_template.hpp"

#include <cstdint>

namespace nuka::core::diagnostics {
class InvariantSampler;
class TraceSink;
} // namespace nuka::core::diagnostics

namespace nuka::runtime {

struct WorldStepOptions {
    math::Vec3 gravity = {0.0f, -9.81f, 0.0f};
    float dt = 1.0f / 60.0f;
    uint32_t step_count = 1;
    bool clear_forces_after_step = true;
    bool enable_contacts = true;
    uint32_t solver_velocity_iterations = 10;
    uint32_t solver_position_iterations = 4;
    float solver_slop = 0.005f;
    float solver_baumgarte = 0.2f;
};

struct WorldStepReport {
    uint32_t simulated_step_count = 0;
    uint32_t broadphase_pair_count = 0;
    uint32_t contact_manifold_count = 0;
    uint32_t contact_point_count = 0;
    uint32_t joint_constraint_count = 0;
    uint32_t drive_constraint_count = 0;
    uint32_t constraint_block_count = 0;
    uint32_t constraint_row_count = 0;
    uint32_t solver_iterations_used = 0;
    float max_constraint_error = 0.0f;
};

WorldStepReport StepWorldInstance(const WorldTemplate& world_template,
                                  WorldInstance& instance,
                                  const WorldStepOptions& options = {});
WorldStepReport StepWorldInstance(const WorldTemplate& world_template,
                                  WorldInstance& instance,
                                  const WorldStepOptions& options,
                                  core::diagnostics::InvariantSampler* invariant_sampler,
                                  core::diagnostics::TraceSink* trace_sink);

} // namespace nuka::runtime
