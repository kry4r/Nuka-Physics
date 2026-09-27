#pragma once

#include <cstdint>
#include <type_traits>

namespace nuka::nk {

// The compact kernel spans one cell per axis on each of two staggered lattices.
inline constexpr uint32_t kMpmLattices = 2u;
inline constexpr uint32_t kMpmStencilWidth = 2u;
inline constexpr uint32_t kMpmLatticeStencilNodes =
    kMpmStencilWidth * kMpmStencilWidth * kMpmStencilWidth;
inline constexpr uint32_t kMpmStencilNodes = kMpmLattices * kMpmLatticeStencilNodes;
// Particles sort by half-cell, which fixes their cell on both lattices.
inline constexpr uint32_t kMpmHalfCellsPerLatticeNode = 8u;
// A stress cell touches lattice-0 nodes i..i+1 and lattice-1 nodes i-1..i+1 per axis.
inline constexpr uint32_t kMpmCellStencilNodes = kMpmLatticeStencilNodes +
    (kMpmStencilWidth + 1u) * (kMpmStencilWidth + 1u) * (kMpmStencilWidth + 1u);
// A stress cell solves its mean stress and five deviator components as one block of rows.
inline constexpr uint32_t kMpmDeviatorComponents = 5u;
inline constexpr uint32_t kMpmStressRowsPerCell = 1u + kMpmDeviatorComponents;

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
