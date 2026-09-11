#pragma once

#include <cstdint>
#include <type_traits>

namespace nuka::nk {

// Material rows use SI units and contain no execution-backend state.
struct MpmMaterial {
    static constexpr uint32_t kValueCount = 11u;
    static constexpr float kHenckyJ2 = 5.0f;
    float youngs = 0.0f, poisson = 0.0f, density = 0.0f;
    float dp_friction = 0.0f, dp_cohesion = 0.0f;
    // 0 corotated, 2 Neo-Hookean, 3 Tait fluid, 4 Drucker-Prager, 5 Hencky J2.
    float model_kind = 0.0f;
    float bulk_modulus = 0.0f, tait_gamma = 0.0f, viscosity = 0.0f;
    float yield_stress = 0.0f, hardening_modulus = 0.0f;
};

static_assert(std::is_trivially_copyable<MpmMaterial>::value);
static_assert(sizeof(MpmMaterial) == MpmMaterial::kValueCount * sizeof(float));

}  // namespace nuka::nk
