#pragma once

#include <cstdint>

namespace nuka::nk {

enum class DerivativeKind : uint32_t { Unmeasured, Consistent, PositiveDefiniteApproximation };
enum class ResidualEquation : uint32_t {
    DiscreteMomentum, ContactNormal, ContactFriction, Kinematics, LocalMaterial, Count
};

struct ResidualNorm {
    double maximum = 0.0;
    double rms = 0.0;
    double reference_scale = 0.0;
    uint64_t equation_count = 0u;
    bool measured = false;
};

// Each norm divides its physical equation by a declared positive reference scale.
struct ResidualReport {
    ResidualNorm equations[static_cast<uint32_t>(ResidualEquation::Count)];
    bool complete = false;
};

struct LinearizationStamp {
    uint64_t state_version = 0u;
    uint64_t public_step = 0u;
    uint32_t attempt = 0u;
    uint32_t substep = 0u;
    uint32_t topology_epoch = 0u;
};

// Residual and Jacobian actions share frozen trial geometry, contact branches and committed history.
struct LinearizationContract {
    LinearizationStamp stamp;
    DerivativeKind derivative = DerivativeKind::Unmeasured;
    bool complete_equations = false;
    bool symmetric = false;
};

}  // namespace nuka::nk
