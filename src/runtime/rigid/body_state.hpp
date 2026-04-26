#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::rigid::BodyState – per-body simulation state
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "math/quat.hpp"

namespace nuka::runtime::rigid {

struct BodyState {
    float     inv_mass          = 0.0f;
    math::Vec3 position         = {0.0f, 0.0f, 0.0f};
    math::Quat orientation      = math::Quat::Identity();
    math::Vec3 linear_velocity  = {0.0f, 0.0f, 0.0f};
    math::Vec3 angular_velocity = {0.0f, 0.0f, 0.0f};
    math::Vec3 force            = {0.0f, 0.0f, 0.0f};
    math::Vec3 torque           = {0.0f, 0.0f, 0.0f};
    math::Vec3 inv_inertia      = {0.0f, 0.0f, 0.0f};
};

} // namespace nuka::runtime::rigid
