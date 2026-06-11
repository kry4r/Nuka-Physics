#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::WorldStepOptions / WorldStepReport – stepping PODs
// ---------------------------------------------------------------------------
// Hoisted out of the (removed) per-instance CPU stepping header. That driver is
// gone, but these two plain-data records are still consumed by kept
// code: the C-ABI WorldRecord caches WorldStepOptions (dt/gravity/contacts read
// by the diff-sim + noise paths) and the diagnostics InvariantWorldView points
// at a WorldStepReport (max_constraint_error). No behaviour, no dependencies
// beyond math::Vec3.
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"

#include <cstdint>

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

} // namespace nuka::runtime
