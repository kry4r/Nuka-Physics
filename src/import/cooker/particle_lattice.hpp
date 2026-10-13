#pragma once
// nuka::runtime::fluid -- CUDA-free lattice particle sets and the host Poly6 kernel
// the fluid cooker and the render surface mesher share.

#include "math/vec3.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nuka::runtime::fluid {

// Poly6 SCALAR kernel value at squared distance r2: const 315/(64*pi*h^9) times
// the (h^2 - r^2)^3 clamp.
inline float Poly6FromR2Host(float r2, float h) {
    constexpr float kPi = 3.14159265358979323846f;
    const float h2 = h * h;
    if (r2 >= h2) {
        return 0.0f;
    }
    const float h3 = h * h * h;
    const float h6 = h3 * h3;
    const float h9 = h6 * h3;
    const float poly6 = 315.0f / (64.0f * kPi * h9);
    const float diff = h2 - r2;  // (h^2 - r^2) >= 0 here.
    return poly6 * diff * diff * diff;
}

// GRADIENT of the Poly6 kernel wrt the sample position, the analytic gradient of
// the SAME field Poly6FromR2Host sums: d/dx [(h^2-r^2)^3] gives
// -6*(h^2-r^2)^2 * r_vec, scaled by the Poly6 constant. Used for density-isosurface
// normals.
inline math::Vec3 Poly6GradientHost(math::Vec3 r_vec, float r2, float h) {
    constexpr float kPi = 3.14159265358979323846f;
    const float h2 = h * h;
    if (r2 >= h2) {
        return math::Vec3{0.0f, 0.0f, 0.0f};
    }
    const float h3 = h * h * h;
    const float h6 = h3 * h3;
    const float h9 = h6 * h3;
    const float poly6 = 315.0f / (64.0f * kPi * h9);
    const float diff = h2 - r2;                      // (h^2 - r^2) >= 0 here.
    const float coeff = poly6 * -6.0f * diff * diff;  // d/dr2 chain on (h^2-r^2)^3.
    return math::Vec3{r_vec.x * coeff, r_vec.y * coeff, r_vec.z * coeff};
}

// A lattice-sampled particle set with uniform mass.
struct ParticleLattice {
    std::vector<math::Vec3> positions;
    std::vector<math::Vec3> velocities;
    float particle_mass = 1.0f;  // m (same for every particle).
};

} // namespace nuka::runtime::fluid
