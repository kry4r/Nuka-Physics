// ---------------------------------------------------------------------------
// PHI v2 CUDA backend — M3b CRBA ops: CrbaComputeM / CrbaFactorM.
//
// KERNEL BODIES ARE LINE-BY-LINE PORTS (D1 byte-exact contract) of the CRBA
// section of src/runtime/articulation/articulation_contacts.cu
// (ComputeArticulationInertiaMKernel / FactorArticulationInertiaMKernel).
// Input wiring is the only change: composite scratch = the
// link_composite_inertia field, M tile = the m field, M^-1 = the m_inv field,
// the implicit-damping fold reads the drive_damping field. The known capacity
// constant kMaxFactorDof == kMaxArticulationDof (64) is ported AS-IS (G0
// DOF-honesty is out of scope here; the host-side loud-throw guard becomes a
// Status::Failed return).
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

#include "math/cuda_spatial_ops.cuh"
#include "math/cuda_vec_ops.cuh"
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/articulation_types.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/registry.cuh"

namespace nuka::phi {

namespace {

using namespace ::nuka::phi::nkops;

constexpr uint32_t kInvalidLink = ~0u;
constexpr float kMinDiagonal = 1.0e-6f;

namespace mg = ::nuka::math::gpu;

// The *Local spatial helpers forward to the SAME shared library bodies the
// legacy articulation_contacts.cu consumes (call sites verbatim).
__forceinline__ __device__ float Dot6Local(const float* a, const float* b) {
    return mg::Dot6(a, b);
}

__forceinline__ __device__ void Copy36Local(const float* src, float* dst) {
    mg::Copy36(src, dst);
}

__forceinline__ __device__ void Mat66MulVec6Local(const float* matrix,
                                                  const float* vector,
                                                  float* out) {
    mg::Mat66MulVec6(matrix, vector, out);
}

__forceinline__ __device__ void TransformInertiaToParentLocal(
    const LinkSpatialTransform& transform,
    const float* child_inertia,
    float* parent_delta) {
    mg::TransformInertiaToParent(transform.X, child_inertia, parent_delta);
}

__forceinline__ __device__ void TransformForceTransposeLocal(
    const LinkSpatialTransform& transform,
    const float* in,
    float* out) {
    mg::TransformForceTranspose(transform.X, in, out);
}

__forceinline__ __device__ uint32_t LocalDofIndex(const ArticulationDeviceState& state,
                                                  uint32_t offset,
                                                  uint32_t link) {
    return LocalDofIndexDevice(state, offset, link);
}

// (1) Dense symmetric M per articulation via CRBA. One block, single lane.
__global__ void ComputeArticulationInertiaMKernel(ArticulationDeviceState state,
                                                  uint32_t max_dof,
                                                  LinkSpatialInertia* composite,
                                                  float* out_inertia_M,
                                                  const float* joint_damping,
                                                  float dt) {
    const uint32_t articulation = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (articulation >= state.articulation_count || lane != 0u) {
        return;
    }

    const uint32_t offset = state.articulation_link_offset[articulation];
    const uint32_t count = state.articulation_link_count[articulation];
    const size_t tile_stride = static_cast<size_t>(max_dof) * max_dof;
    float* const M = out_inertia_M + static_cast<size_t>(articulation) * tile_stride;

    // Zero the whole tile (leading dof_count block filled below; padding stays 0).
    for (size_t i = 0u; i < tile_stride; ++i) {
        M[i] = 0.0f;
    }

    // Seed composite inertia from the rigid-body spatial inertia. We must NOT
    // read link_articulated_I (ABA Pass-2 clobbers it); link_inertia is the
    // pristine rigid-body inertia in each link's own frame.
    for (uint32_t local = 0u; local < count; ++local) {
        const uint32_t link = offset + local;
        Copy36Local(state.link_inertia[link].I, composite[link].I);
    }

    // Composite leaf->root: Ic[parent] += X^T Ic[link] X (ABA Pass-2 without the
    // articulated-inertia reduction). Fixed-DOF links still propagate their mass.
    for (uint32_t reverse = count; reverse > 0u; --reverse) {
        const uint32_t link = offset + reverse - 1u;
        const uint32_t parent_local = state.parent_link[link];
        if (parent_local == kInvalidLink) {
            continue;
        }
        float parent_delta[36];
        TransformInertiaToParentLocal(state.link_xup[link], composite[link].I, parent_delta);
        const uint32_t parent_link = offset + parent_local;
        for (uint32_t i = 0u; i < 36u; ++i) {
            composite[parent_link].I[i] += parent_delta[i];
        }
    }

    // T8b: floating-base leading 6x6 block (see articulation_contacts.cu).
    const bool floating_root =
        (count > 0u) &&
        (state.parent_link[offset] == kInvalidLink) &&
        (state.joint_type[offset] == ArticulationJointType::FloatingBase);
    if (floating_root && max_dof >= 6u) {
        const float* const root_I = composite[offset].I;
        for (uint32_t r = 0u; r < 6u; ++r) {
            for (uint32_t c = 0u; c < 6u; ++c) {
                M[static_cast<size_t>(r) * max_dof + c] = root_I[r * 6u + c];
            }
        }
    }

    // For each non-fixed joint i: F = Ic_i S_i (force in i's frame). M[i][i] =
    // S_i^T F. Then walk to the root pushing F up by X^T at each step; at every
    // non-fixed ancestor j, M[i][j] = M[j][i] = S_j^T F (F now in j's frame).
    for (uint32_t local = 0u; local < count; ++local) {
        const uint32_t link = offset + local;
        if (JointDofCountDevice(state.joint_type[link]) == 0u) {
            continue;
        }
        // T8b: the floating root's leading block is filled above (S_base = I6); skip
        // it here -- the scalar S_i path is meaningless for the 6-DOF base.
        if (local == 0u && floating_root) {
            continue;
        }
        const uint32_t dof_i = LocalDofIndex(state, offset, link);
        if (dof_i >= max_dof) {
            continue;
        }

        float force[6];
        Mat66MulVec6Local(composite[link].I, state.joint_motion_subspace[link].s, force);

        float diagonal = Dot6Local(state.joint_motion_subspace[link].s, force);
        // Reflected rotor inertia + floor (matches ABA Pass-2 diagonal guard).
        diagonal += state.joint_armature[link];
        diagonal += kInertiaDiagonalEpsilon;
        // General implicit joint viscous damping (opt-in): fold dt*c into the
        // joint diagonal (see articulation_contacts.cu for the full rationale).
        if (joint_damping != nullptr) {
            diagonal += dt * joint_damping[link];
        }
        M[static_cast<size_t>(dof_i) * max_dof + dof_i] = diagonal;

        uint32_t walk = link;
        while (true) {
            const uint32_t parent_local = state.parent_link[walk];
            if (parent_local == kInvalidLink) {
                break;
            }
            // Push F from `walk`'s frame to its parent's frame.
            float pushed[6];
            TransformForceTransposeLocal(state.link_xup[walk], force, pushed);
            for (uint32_t i = 0u; i < 6u; ++i) {
                force[i] = pushed[i];
            }
            walk = offset + parent_local;
            if (JointDofCountDevice(state.joint_type[walk]) == 0u) {
                continue;
            }
            const uint32_t dof_j = LocalDofIndex(state, offset, walk);
            if (dof_j >= max_dof) {
                continue;
            }
            // T8b: base<->joint coupling (see articulation_contacts.cu).
            if (walk == offset && floating_root) {
                for (uint32_t b = 0u; b < 6u; ++b) {
                    if (b >= max_dof) {
                        break;
                    }
                    M[static_cast<size_t>(dof_i) * max_dof + b] = force[b];
                    M[static_cast<size_t>(b) * max_dof + dof_i] = force[b];
                }
                continue;
            }
            const float entry = Dot6Local(state.joint_motion_subspace[walk].s, force);
            M[static_cast<size_t>(dof_i) * max_dof + dof_j] = entry;
            M[static_cast<size_t>(dof_j) * max_dof + dof_i] = entry;
        }
    }
}

// (2) Per-articulation unpivoted LDL^T of the leading dof_count block, then the
// explicit symmetric inverse. One block, single lane. The dense scratch lives
// in STATIC SHARED memory (see articulation_contacts.cu for the rationale).
constexpr uint32_t kMaxFactorDof = kMaxArticulationDof;

__global__ void FactorArticulationInertiaMKernel(ArticulationDeviceState state,
                                                 uint32_t max_dof,
                                                 const float* inertia_M,
                                                 float* out_inertia_M_inv) {
    __shared__ float a[kMaxFactorDof * kMaxFactorDof];

    const uint32_t articulation = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (articulation >= state.articulation_count || lane != 0u) {
        return;
    }

    const uint32_t offset = state.articulation_link_offset[articulation];
    const uint32_t count = state.articulation_link_count[articulation];
    uint32_t dof = 0u;
    for (uint32_t local = 0u; local < count; ++local) {
        dof += JointDofCountDevice(state.joint_type[offset + local]);
    }
    if (dof > max_dof) {
        // Defensive MEMORY-SAFETY bound only (see articulation_contacts.cu).
        dof = max_dof;
    }

    const size_t tile_stride = static_cast<size_t>(max_dof) * max_dof;
    const float* const M = inertia_M + static_cast<size_t>(articulation) * tile_stride;
    float* const Minv = out_inertia_M_inv + static_cast<size_t>(articulation) * tile_stride;
    for (size_t i = 0u; i < tile_stride; ++i) {
        Minv[i] = 0.0f;
    }
    if (dof == 0u) {
        return;
    }

    // Copy the leading dof x dof block into the dense (shared) scratch.
    for (uint32_t r = 0u; r < dof; ++r) {
        for (uint32_t c = 0u; c < dof; ++c) {
            a[r * kMaxFactorDof + c] = M[static_cast<size_t>(r) * max_dof + c];
        }
    }

    // Unpivoted LDL^T: A = L D L^T, L unit-lower-triangular, D diagonal.
    // L stored in the strict lower triangle of `a`, D on its diagonal.
    float d[kMaxFactorDof];
    for (uint32_t j = 0u; j < dof; ++j) {
        float djj = a[j * kMaxFactorDof + j];
        for (uint32_t k = 0u; k < j; ++k) {
            djj -= a[j * kMaxFactorDof + k] * a[j * kMaxFactorDof + k] * d[k];
        }
        if (djj < kMinDiagonal) {
            djj = kMinDiagonal;  // SPD floor; guards a degenerate config.
        }
        d[j] = djj;
        for (uint32_t i = j + 1u; i < dof; ++i) {
            float lij = a[i * kMaxFactorDof + j];
            for (uint32_t k = 0u; k < j; ++k) {
                lij -= a[i * kMaxFactorDof + k] * a[j * kMaxFactorDof + k] * d[k];
            }
            a[i * kMaxFactorDof + j] = lij / djj;
        }
    }

    // Solve A x = e_col for each identity column to form M^-1 (symmetric).
    for (uint32_t col = 0u; col < dof; ++col) {
        float y[kMaxFactorDof];
        // Forward solve L y = e_col.
        for (uint32_t i = 0u; i < dof; ++i) {
            float value = (i == col) ? 1.0f : 0.0f;
            for (uint32_t k = 0u; k < i; ++k) {
                value -= a[i * kMaxFactorDof + k] * y[k];
            }
            y[i] = value;
        }
        // Diagonal solve D z = y (in place).
        for (uint32_t i = 0u; i < dof; ++i) {
            y[i] /= d[i];
        }
        // Backward solve L^T x = z.
        float x[kMaxFactorDof];
        for (uint32_t ii = dof; ii > 0u; --ii) {
            const uint32_t i = ii - 1u;
            float value = y[i];
            for (uint32_t k = i + 1u; k < dof; ++k) {
                value -= a[k * kMaxFactorDof + i] * x[k];
            }
            x[i] = value;
        }
        for (uint32_t r = 0u; r < dof; ++r) {
            Minv[static_cast<size_t>(r) * max_dof + col] = x[r];
        }
    }
}

// --- op entry points ---------------------------------------------------------

Status OpCrbaComputeM(const ModelView& model, const DataView& data,
                      const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const CrbaComputeMParams*>(params);
    if (p == nullptr) {
        return Status::Failed;
    }
    if (p->articulation_count == 0u || p->total_link_count == 0u || p->max_dof == 0u) {
        return Status::Ok;
    }
    const ArticulationDeviceState state = MakeArticulationDeviceState(
        model, data, p->total_link_count, p->articulation_count);
    const float* joint_damping =
        (p->fold_drive_damping != 0u) ? data.drive_damping : nullptr;
    LaunchCuda(ComputeArticulationInertiaMKernel, dim3(p->articulation_count),
               dim3(32u), 0u, stream, state, p->max_dof,
               reinterpret_cast<LinkSpatialInertia*>(data.link_composite_inertia),
               data.m, joint_damping, p->dt);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

Status OpCrbaFactorM(const ModelView& model, const DataView& data,
                     const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const CrbaFactorMParams*>(params);
    if (p == nullptr) {
        return Status::Failed;
    }
    if (p->articulation_count == 0u || p->max_dof == 0u) {
        return Status::Ok;
    }
    if (p->max_dof > kMaxArticulationDof) {
        // Legacy loud-throw guard (G0 honesty: never a silent clamp).
        return Status::Failed;
    }
    const ArticulationDeviceState state = MakeArticulationDeviceState(
        model, data, /*total_link_count=*/0u, p->articulation_count);
    LaunchCuda(FactorArticulationInertiaMKernel, dim3(p->articulation_count),
               dim3(32u), 0u, stream, state, p->max_dof, data.m, data.m_inv);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

} // namespace

void RegisterNkCrbaOps() {
    SetCudaOp(NkOp::CrbaComputeM, &OpCrbaComputeM);
    SetCudaOp(NkOp::CrbaFactorM, &OpCrbaFactorM);
}

} // namespace nuka::phi
