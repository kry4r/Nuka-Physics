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

inline constexpr uint32_t kRowClassCount = 4u;
inline constexpr uint32_t kMaximalContactRowId = 0u;
inline constexpr uint32_t kMaximalContactRowMaxRowsPerBlock = 6u;
inline constexpr uint32_t kMaximalJointRowId = 1u;
inline constexpr uint32_t kMaximalJointRowMaxRowsPerBlock = 6u;
inline constexpr uint32_t kMaximalDriveRowId = 2u;
inline constexpr uint32_t kMaximalDriveRowMaxRowsPerBlock = 1u;
inline constexpr uint32_t kFeatherstoneContactRowId = 3u;
inline constexpr uint32_t kFeatherstoneContactRowMaxRowsPerBlock = 6u;

const char* RowClassName(uint32_t row_class_id) noexcept;
const char* RowClassForwardKernelSymbol(uint32_t row_class_id) noexcept;
uint32_t RowClassMaxRowsPerBlock(uint32_t row_class_id) noexcept;
bool IsKnownRowClass(uint32_t row_class_id) noexcept;

} // namespace nuka::solver::generated
