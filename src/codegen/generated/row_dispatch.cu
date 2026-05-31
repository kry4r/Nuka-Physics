// GENERATED — DO NOT EDIT
// ============================================================
// Source: tools/codegen/classes/*.yaml
// Regenerate: python tools/codegen/regen.py
// ============================================================

#include "codegen/generated/row_class_registry.hpp"

#include <cstdint>

namespace nuka::solver::generated {

const char* RowClassName(uint32_t row_class_id) noexcept {
    switch (row_class_id) {
    case 0u:
        return "MaximalContactRow";
    case 1u:
        return "MaximalJointRow";
    case 2u:
        return "MaximalDriveRow";
    case 3u:
        return "FeatherstoneContactRow";
    default:
        return "Unknown";
    }
}

const char* RowClassForwardKernelSymbol(uint32_t row_class_id) noexcept {
    switch (row_class_id) {
    case 0u:
        return "maximal_contact_forward_kernel";
    case 1u:
        return "maximal_joint_forward_kernel";
    case 2u:
        return "maximal_drive_forward_kernel";
    case 3u:
        return "featherstone_contact_forward_kernel";
    default:
        return nullptr;
    }
}

uint32_t RowClassMaxRowsPerBlock(uint32_t row_class_id) noexcept {
    switch (row_class_id) {
    case 0u:
        return 6u;
    case 1u:
        return 6u;
    case 2u:
        return 1u;
    case 3u:
        return 6u;
    default:
        return 0u;
    }
}

bool IsKnownRowClass(uint32_t row_class_id) noexcept {
    return RowClassForwardKernelSymbol(row_class_id) != nullptr;
}

uint8_t RowClassGradientMode(uint32_t row_class_id) noexcept {
    switch (row_class_id) {
    case 0u:
        return 1u;
    case 1u:
        return 0u;
    case 2u:
        return 0u;
    case 3u:
        return 1u;
    default:
        return 0u;
    }
}

uint16_t RowClassAdjointKernelId(uint32_t row_class_id) noexcept {
    switch (row_class_id) {
    case 0u:
        return 0u;
    case 1u:
        return 0u;
    case 2u:
        return 1u;
    case 3u:
        return 0u;
    default:
        return 0u;
    }
}

const char* RowClassAdjointKernelSymbol(uint32_t row_class_id) noexcept {
    switch (row_class_id) {
    case 2u:
        return "maximal_drive_adjoint_kernel";
    default:
        return nullptr;
    }
}

bool RowClassHasAdjoint(uint32_t row_class_id) noexcept {
    return RowClassAdjointKernelSymbol(row_class_id) != nullptr;
}

} // namespace nuka::solver::generated
