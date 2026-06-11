#pragma once
// ---------------------------------------------------------------------------
// nuka::solver::SolverConfig – iterative constraint-solver tuning POD
// ---------------------------------------------------------------------------
// Hoisted out of the (removed) rigid_solver.hpp so the kept unified-solve /
// coresident paths can reference the config without pulling the legacy CPU PGS
// solver. Plain aggregate; no dependencies.
// ---------------------------------------------------------------------------

#include <cstdint>

namespace nuka::solver {

struct SolverConfig {
    uint32_t velocity_iterations = 10;
    uint32_t position_iterations = 4;
    float slop = 0.005f;       // penetration slop (allowed overlap)
    float baumgarte = 0.2f;    // position correction factor
};

} // namespace nuka::solver
