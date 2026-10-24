#pragma once

#include <cstdint>
#include <vector>

#include "nk/solve/residual_contract.hpp"

namespace nuka::nk {

enum class StepOutcome : uint32_t {
    Unmeasured, Accepted, BackendFailure, LocalMaterialFailure, MaterialOutOfRange,
    CapacityExceeded, InvalidGeometry, SolverFailure, BatchRejected
};

struct DiagnosticStamp {
    uint64_t public_step = 0u;
    uint64_t state_version = 0u;
    uint32_t attempt = 0u;
    bool valid = false;
    bool accepted = false;
};

struct EnvironmentStepReport {
    StepOutcome outcome = StepOutcome::Unmeasured;
    uint32_t flags = 0u;
    uint32_t topology_epoch = 0u;
    uint32_t contact_demand = 0u;  // largest OGC slot request of the step; above contact_capacity on overflow
    uint32_t topology_pending = 0u;  // seams the last attempt opened without a re-solve, once the attempts ran out
    uint32_t range_element = 0u;  // one plus the largest field element the last attempt took out of range, 0 if none
    double time = 0.0;
    double advanced_dt = 0.0;
    std::vector<uint32_t> executed_iterations;
    ResidualReport residual;
};

// Backend completion and material validity are measured separately from full physical convergence.
struct PublicStepReport {
    DiagnosticStamp stamp;
    uint32_t requested_iterations = 0u;
    uint32_t configured_iterations = 0u;
    uint32_t iteration_budget = 0u;
    uint32_t substeps = 0u;
    uint32_t contact_capacity = 0u;  // OGC contact slots per environment
    bool physical_convergence_measured = false;
    bool rolled_back = false;
    std::vector<EnvironmentStepReport> environments;
};

}  // namespace nuka::nk
