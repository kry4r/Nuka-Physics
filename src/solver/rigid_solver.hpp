#pragma once
// ---------------------------------------------------------------------------
// nuka::solver – Iterative rigid body constraint solver (PGS)
// ---------------------------------------------------------------------------

#include "constraint/row_buffers.hpp"
#include "runtime/rigid/body_state.hpp"

#include <cstdint>
#include <vector>

namespace nuka::solver {

struct SolveResult {
    float max_penetration;
    uint32_t iterations_used;
};

struct SolverConfig {
    uint32_t velocity_iterations = 10;
    uint32_t position_iterations = 4;
    float slop = 0.005f;       // penetration slop (allowed overlap)
    float baumgarte = 0.2f;    // position correction factor
};

/// Solve a set of constraints for the given body states using PGS.
SolveResult SolveConstraints(
    constraint::RowBuffers& rows,
    std::vector<runtime::rigid::BodyState>& bodies,
    const SolverConfig& config = {}
);

/// Convenience: solve a unit test scenario of a box resting on ground.
SolveResult SolveUnitGroundContact();

} // namespace nuka::solver
