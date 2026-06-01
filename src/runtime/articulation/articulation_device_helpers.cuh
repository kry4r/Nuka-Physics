#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::articulation -- shared __device__ articulation-DOF indexing
//
// Consolidates the per-TU integer DOF helpers that were re-declared inside the
// anonymous namespace of several articulation CUDA translation units under
// slightly different names but IDENTICAL bodies:
//   articulation_jacobian.cu   JointDofCount         / (inline dof_index walk)
//   articulation_contacts.cu   JointDofCountDevice   / LocalDofIndex
//   articulation_drives.cu     JointDofCountDriveDevice / LocalDofIndexDrive
//
// These depend on the articulation domain types (ArticulationJointType,
// ArticulationDeviceState) and are therefore kept OUT of the pure-math
// src/math/ headers: putting them there would force cuda_vec_ops.cuh to pull in
// articulation_state.hpp into every unrelated math consumer (broadphase,
// particle, sensors, ...). They live next to the state layout instead.
//
// DETERMINISM CONTRACT (D1): integer-only domain dispatch and a fixed-order
// prefix-sum loop -- no floating-point arithmetic, so there is no FMA / reorder
// risk. __forceinline__ is applied (instruction-level no-op); the bodies are
// byte-identical to the former local copies. The function names differ across
// TUs only; the arithmetic is the same single source of truth so M's rows/cols,
// the chain Jacobian's columns and the OSC task-Jacobian all index identically.
//
// NOTE (scope): src/diffsim/osc_adjoint.cu carries an equivalent
// JointDofCountDriveDevice copy on a C++-only adjoint path that the p04.9
// bit-identity oracle does NOT cover, so it is intentionally left
// un-consolidated here (surfaced as future work).
// ---------------------------------------------------------------------------

#include "runtime/articulation/articulation_state.hpp"

#include <cstdint>

#if !defined(__CUDACC__)
#error "articulation_device_helpers.cuh is a CUDA device header; compile with nvcc."
#endif

namespace nuka::runtime::articulation {

// Generalized-coordinate count contributed by a joint of the given type. The
// single source of truth for both per-contact dof_stride and the dof_index
// prefix sum, so the chain Jacobian transparently picks up a 6-DOF floating-base
// root without touching the walk logic.
__forceinline__ __device__ uint32_t JointDofCountDevice(ArticulationJointType type) {
    switch (type) {
        case ArticulationJointType::Revolute:
        case ArticulationJointType::Prismatic:
            return 1u;
        case ArticulationJointType::Fixed:
            return 0u;
        case ArticulationJointType::FloatingBase:
            return 6u;  // T8a: free-floating root contributes 6 DOF.
    }
    return 0u;
}

// Local dof_index of a link: base-inclusive prefix sum of per-joint DOF counts
// over [offset, link). Mirrors the chain Jacobian's dof_index exactly, so M's
// rows/cols line up with J's columns.
__forceinline__ __device__ uint32_t LocalDofIndexDevice(const ArticulationDeviceState& state,
                                                        uint32_t offset,
                                                        uint32_t link) {
    uint32_t index = 0u;
    for (uint32_t k = offset; k < link; ++k) {
        index += JointDofCountDevice(state.joint_type[k]);
    }
    return index;
}

} // namespace nuka::runtime::articulation
