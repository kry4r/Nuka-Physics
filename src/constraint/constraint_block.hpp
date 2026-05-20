#pragma once
// ---------------------------------------------------------------------------
// nuka::constraint::ConstraintBlock -- per-constraint Jacobian storage
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"

#include <cstdint>

namespace nuka::constraint {

enum class ConstraintType : uint8_t {
    Contact,
    Joint,
    Drive
};

struct ConstraintBlock {
    ConstraintType type = ConstraintType::Contact;
    uint32_t body_a     = ~0u;
    uint32_t body_b     = ~0u;
    uint32_t row_count  = 0;

    // Per-row data (up to 6 rows for a full 6-DOF constraint)
    static constexpr uint32_t kMaxRows = 6;

    math::Vec3 jacobian_linear_a[kMaxRows];
    math::Vec3 jacobian_angular_a[kMaxRows];
    math::Vec3 jacobian_linear_b[kMaxRows];
    math::Vec3 jacobian_angular_b[kMaxRows];

    float rhs[kMaxRows]            = {};
    float lower_limit[kMaxRows]    = {};
    float upper_limit[kMaxRows]    = {};
    float impulse[kMaxRows]        = {};   // accumulated impulse (warm start)
    float effective_mass[kMaxRows] = {};

    // Local joint anchor offsets for position projection. Contact and drive
    // blocks leave these at zero.
    math::Vec3 anchor_local_a;
    math::Vec3 anchor_local_b;
};

} // namespace nuka::constraint
