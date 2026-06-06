// GENERATED — DO NOT EDIT
// ============================================================
// Source: tools/codegen/classes/*.yaml
// Regenerate: python tools/codegen/regen.py
// ============================================================

#pragma once

#include <cstdint>

namespace nuka::solver::generated {

enum class RowFlag : uint16_t {
    Equality = uint16_t{1u << 0},
    Unilateral = uint16_t{1u << 1},
    Friction = uint16_t{1u << 2},
    Coupled = uint16_t{1u << 3},
    GradActive = uint16_t{1u << 4},
};

enum class RowClassId : uint32_t {
    MaximalContactRow = 0u,
    MaximalJointRow = 1u,
    MaximalDriveRow = 2u,
    FeatherstoneContactRow = 3u,
    RigidSDFContactRow = 4u,
    FeatherstoneSDFContactRow = 5u,
    XPBDDistanceRow = 6u,
    XPBDBendRow = 7u,
    XPBDVolumeRow = 8u,
    XPBDShapeMatchRow = 9u,
    ParticleParticleContactRow = 10u,
    XpbdCosseratStretchShearRow = 11u,
    XpbdCosseratBendTwistRow = 12u,
    CouplingContactRow = 13u,
};

// Reverse-mode gradient mode; index matches the uint8 stored in Row.gradient_mode.
enum class GradientMode : uint8_t {
    dense_adjoint = 0u,
    stop_grad_on_event = 1u,
    ift_at_convergence = 2u,
    none = 3u,
};

struct Row {
    uint32_t row_class_id = 0u;
    uint32_t body_count = 0u;
    uint32_t body_list_offset = 0u;
    uint32_t jacobian_offset = 0u;
    float rhs = 0.0f;
    float lambda = 0.0f;
    float lower = 0.0f;
    float upper = 0.0f;
    float compliance_alpha = 0.0f;
    float damping_beta = 0.0f;
    uint16_t flags = 0u;
    uint16_t adjoint_kernel_id = 0u;
    uint8_t gradient_mode = 0u;
    uint8_t recompute_mode = 0u;
    uint8_t event_flag_field = 0u;
    uint8_t contact_softness = 0u;
};

inline constexpr uint32_t kRowClassCount = 14u;
inline constexpr uint32_t kMaximalContactRowId = 0u;
inline constexpr uint32_t kMaximalContactRowMaxRowsPerBlock = 6u;
inline constexpr uint8_t kMaximalContactRowGradientMode = 1u;
inline constexpr uint16_t kMaximalContactRowAdjointKernelId = 1u;
inline constexpr uint32_t kMaximalJointRowId = 1u;
inline constexpr uint32_t kMaximalJointRowMaxRowsPerBlock = 6u;
inline constexpr uint8_t kMaximalJointRowGradientMode = 0u;
inline constexpr uint16_t kMaximalJointRowAdjointKernelId = 2u;
inline constexpr uint32_t kMaximalDriveRowId = 2u;
inline constexpr uint32_t kMaximalDriveRowMaxRowsPerBlock = 1u;
inline constexpr uint8_t kMaximalDriveRowGradientMode = 0u;
inline constexpr uint16_t kMaximalDriveRowAdjointKernelId = 3u;
inline constexpr uint32_t kFeatherstoneContactRowId = 3u;
inline constexpr uint32_t kFeatherstoneContactRowMaxRowsPerBlock = 6u;
inline constexpr uint8_t kFeatherstoneContactRowGradientMode = 1u;
inline constexpr uint16_t kFeatherstoneContactRowAdjointKernelId = 4u;
inline constexpr uint32_t kRigidSDFContactRowId = 4u;
inline constexpr uint32_t kRigidSDFContactRowMaxRowsPerBlock = 6u;
inline constexpr uint8_t kRigidSDFContactRowGradientMode = 2u;
inline constexpr uint16_t kRigidSDFContactRowAdjointKernelId = 0u;
inline constexpr uint32_t kFeatherstoneSDFContactRowId = 5u;
inline constexpr uint32_t kFeatherstoneSDFContactRowMaxRowsPerBlock = 6u;
inline constexpr uint8_t kFeatherstoneSDFContactRowGradientMode = 2u;
inline constexpr uint16_t kFeatherstoneSDFContactRowAdjointKernelId = 0u;
inline constexpr uint32_t kXPBDDistanceRowId = 6u;
inline constexpr uint32_t kXPBDDistanceRowMaxRowsPerBlock = 6u;
inline constexpr uint8_t kXPBDDistanceRowGradientMode = 0u;
inline constexpr uint16_t kXPBDDistanceRowAdjointKernelId = 5u;
inline constexpr uint32_t kXPBDBendRowId = 7u;
inline constexpr uint32_t kXPBDBendRowMaxRowsPerBlock = 6u;
inline constexpr uint8_t kXPBDBendRowGradientMode = 0u;
inline constexpr uint16_t kXPBDBendRowAdjointKernelId = 6u;
inline constexpr uint32_t kXPBDVolumeRowId = 8u;
inline constexpr uint32_t kXPBDVolumeRowMaxRowsPerBlock = 6u;
inline constexpr uint8_t kXPBDVolumeRowGradientMode = 0u;
inline constexpr uint16_t kXPBDVolumeRowAdjointKernelId = 7u;
inline constexpr uint32_t kXPBDShapeMatchRowId = 9u;
inline constexpr uint32_t kXPBDShapeMatchRowMaxRowsPerBlock = 6u;
inline constexpr uint8_t kXPBDShapeMatchRowGradientMode = 0u;
inline constexpr uint16_t kXPBDShapeMatchRowAdjointKernelId = 8u;
inline constexpr uint32_t kParticleParticleContactRowId = 10u;
inline constexpr uint32_t kParticleParticleContactRowMaxRowsPerBlock = 6u;
inline constexpr uint8_t kParticleParticleContactRowGradientMode = 0u;
inline constexpr uint16_t kParticleParticleContactRowAdjointKernelId = 9u;
inline constexpr uint32_t kXpbdCosseratStretchShearRowId = 11u;
inline constexpr uint32_t kXpbdCosseratStretchShearRowMaxRowsPerBlock = 6u;
inline constexpr uint8_t kXpbdCosseratStretchShearRowGradientMode = 3u;
inline constexpr uint16_t kXpbdCosseratStretchShearRowAdjointKernelId = 0u;
inline constexpr uint32_t kXpbdCosseratBendTwistRowId = 12u;
inline constexpr uint32_t kXpbdCosseratBendTwistRowMaxRowsPerBlock = 6u;
inline constexpr uint8_t kXpbdCosseratBendTwistRowGradientMode = 3u;
inline constexpr uint16_t kXpbdCosseratBendTwistRowAdjointKernelId = 0u;
inline constexpr uint32_t kCouplingContactRowId = 13u;
inline constexpr uint32_t kCouplingContactRowMaxRowsPerBlock = 6u;
inline constexpr uint8_t kCouplingContactRowGradientMode = 3u;
inline constexpr uint16_t kCouplingContactRowAdjointKernelId = 0u;

const char* RowClassName(uint32_t row_class_id) noexcept;
const char* RowClassForwardKernelSymbol(uint32_t row_class_id) noexcept;
uint32_t RowClassMaxRowsPerBlock(uint32_t row_class_id) noexcept;
bool IsKnownRowClass(uint32_t row_class_id) noexcept;

// Reverse-mode lookups (populate Row.gradient_mode / Row.adjoint_kernel_id).
uint8_t RowClassGradientMode(uint32_t row_class_id) noexcept;
uint16_t RowClassAdjointKernelId(uint32_t row_class_id) noexcept;
const char* RowClassAdjointKernelSymbol(uint32_t row_class_id) noexcept;
bool RowClassHasAdjoint(uint32_t row_class_id) noexcept;

} // namespace nuka::solver::generated
