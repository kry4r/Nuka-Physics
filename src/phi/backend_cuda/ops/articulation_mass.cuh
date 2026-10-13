#pragma once
#include <cuda_runtime.h>
#include <stdlib.h>
#include <limits>
#include "math/cuda_spatial_ops.cuh"
#include "phi/backend_cuda/ops/articulation_types.cuh"

namespace nuka::phi::nkops {
enum class CrbaVector : uint32_t {
    Diagonal, Root, Scale, Compact, FreeDof, DofLink, DofComponent,
    Velocity, Projected, Moment, QdotWork, DampingProduct, Count
};
constexpr uint32_t kCrbaVectorCount = static_cast<uint32_t>(CrbaVector::Count);
static_assert(kCrbaVectorCount == 12u && sizeof(float) == sizeof(uint32_t));

struct CrbaWorkspace {
    uint32_t* words;
    uint32_t max_dof;

    __device__ inline float* Matrix() const {
        return reinterpret_cast<float*>(words);
    }
    __device__ inline uint32_t* Indices(CrbaVector vector) const {
        return words + size_t{max_dof} * max_dof +
            size_t{static_cast<uint32_t>(vector)} * max_dof;
    }
    __device__ inline float* Values(CrbaVector vector) const {
        return reinterpret_cast<float*>(Indices(vector));
    }
};

__device__ inline CrbaWorkspace CrbaWorkspaceFor(
    uint32_t* words, uint32_t articulation, uint32_t max_dof) {
    const size_t stride = size_t{max_dof} * (size_t{max_dof} + kCrbaVectorCount);
    return {words + size_t{articulation} * stride, max_dof};
}

inline bool CrbaWorkspaceFits(uint32_t articulations, uint32_t max_dof) {
    const uint64_t limit = std::numeric_limits<size_t>::max() / sizeof(uint32_t);
    const uint64_t width = uint64_t{max_dof} + kCrbaVectorCount;
    if (max_dof == 0u || articulations == 0u) return true;
    if (uint64_t{max_dof} > limit / width) return false;
    const uint64_t words = uint64_t{max_dof} * width;
    return uint64_t{articulations} <= limit / words;
}

__device__ inline uint32_t ActualCrbaDof(const ArticulationDeviceState& state,
    uint32_t articulation, uint32_t max_dof, uint32_t* err_status) {
    const uint32_t offset = state.articulation_link_offset[articulation];
    const uint32_t count = state.articulation_link_count[articulation];
    uint64_t dof = 0u;
    for (uint32_t local = 0u; local < count; ++local) {
        dof += JointDofCountDevice(state.joint_type[offset + local]);
        if (dof > max_dof) {
            atomicOr(err_status, kEnvStatusDofOverflow);
            return ~0u;
        }
    }
    return static_cast<uint32_t>(dof);
}

namespace mg = ::nuka::math::gpu;

// Spatial operations share the backend math primitives.
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

// Composite inertia preserves child order; independent columns fill the symmetric mass tile.
__device__ inline void ComputeArticulationMass(ArticulationDeviceState state,
                                                  uint32_t max_dof,
                                                  LinkSpatialInertia* composite,
                                                  float* out_inertia_M,
                                                  const float* joint_damping,
                                                  float dt, ArticulationTreeView tree) {
    const uint32_t articulation = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (articulation >= state.articulation_count) {
        return;
    }

    const uint32_t offset = state.articulation_link_offset[articulation];
    const uint32_t count = state.articulation_link_count[articulation];
    const size_t tile_stride = static_cast<size_t>(max_dof) * max_dof;
    float* const M = out_inertia_M + static_cast<size_t>(articulation) * tile_stride;

    for (size_t i = lane; i < tile_stride; i += blockDim.x) {
        M[i] = 0.0f;
    }

    for (uint32_t local = lane; local < count; local += blockDim.x) {
        const uint32_t link = offset + local;
        Copy36Local(state.link_inertia[link].I, composite[link].I);
    }

    __syncthreads();
    for (uint32_t depth = tree.max_depth[articulation]; depth > 0u; --depth) {
        for (uint32_t local = lane; local < count; local += blockDim.x) {
            const uint32_t link = offset + local;
            if (tree.depth[link] + 1u != depth) continue;
            for (uint32_t index = tree.child_range[2u * link]; index < tree.child_range[2u * link + 1u]; ++index) {
                const uint32_t child = tree.children[index];
                float parent_delta[36];
                TransformInertiaToParentLocal(state.link_xup[child], composite[child].I, parent_delta);
                for (uint32_t i = 0u; i < 36u; ++i) composite[link].I[i] += parent_delta[i];
            }
        }
        __syncthreads();
    }

    const bool floating_root =
        (count > 0u) &&
        (state.parent_link[offset] == ~0u) &&
        (state.joint_type[offset] == ArticulationJointType::FloatingBase);
    if (floating_root && max_dof >= 6u) {
        const float* const root_I = composite[offset].I;
        for (uint32_t entry = lane; entry < 36u; entry += blockDim.x)
            M[size_t{entry / 6u} * max_dof + entry % 6u] = root_I[entry];
    }

    for (uint32_t local = lane; local < count; local += blockDim.x) {
        const uint32_t link = offset + local;
        if (JointDofCountDevice(state.joint_type[link]) == 0u) {
            continue;
        }
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
        diagonal += state.joint_armature[link];
        if (joint_damping != nullptr) {
            diagonal += dt * joint_damping[link];
        }
        M[static_cast<size_t>(dof_i) * max_dof + dof_i] = diagonal;

        uint32_t walk = link;
        while (true) {
            const uint32_t parent_local = state.parent_link[walk];
            if (parent_local == ~0u) {
                break;
            }
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

// A = L D L^T, with unit diagonal L and D stored on the factor diagonal.
__device__ inline void FactorMass(const float* mass, float* factor, uint32_t n, uint32_t stride) {
    const uint32_t lane = threadIdx.x;
    for (size_t i = lane; i < size_t{stride} * stride; i += blockDim.x)
        factor[i] = mass[i];
    __syncthreads();
    for (uint32_t j = 0u; j < n; ++j) {
        if (lane == 0u) {
            float diagonal = factor[size_t{j} * stride + j];
            for (uint32_t k = 0u; k < j; ++k) {
                const float l = factor[size_t{j} * stride + k];
                diagonal -= l * l * factor[size_t{k} * stride + k];
            }
            factor[size_t{j} * stride + j] = fmaxf(diagonal, 1.0e-6f);
        }
        __syncthreads();
        for (uint32_t i = j + 1u + lane; i < n; i += blockDim.x) {
            float value = factor[size_t{i} * stride + j];
            for (uint32_t k = 0u; k < j; ++k)
                value -= factor[size_t{i} * stride + k] * factor[size_t{j} * stride + k] *
                         factor[size_t{k} * stride + k];
            factor[size_t{i} * stride + j] = value / factor[size_t{j} * stride + j];
        }
        __syncthreads();
    }
    for (uint32_t i = n + lane; i < stride; i += blockDim.x)
        factor[size_t{i} * stride + i] = 1.0e-6f;
    __syncthreads();
}

__device__ inline void SolveMass(const float* factor, float* value, uint32_t n, uint32_t stride) {
    for (uint32_t i = 0u; i < n; ++i) {
        float sum = value[i];
        for (uint32_t j = 0u; j < i; ++j) sum -= factor[size_t{i} * stride + j] * value[j];
        value[i] = sum;
    }
    for (uint32_t i = 0u; i < n; ++i) value[i] /= factor[size_t{i} * stride + i];
    for (uint32_t end = n; end > 0u; --end) {
        const uint32_t i = end - 1u;
        float sum = value[i];
        for (uint32_t j = i + 1u; j < n; ++j) sum -= factor[size_t{j} * stride + i] * value[j];
        value[i] = sum;
    }
}

// y = L^-1 (v + extra) for the unit-diagonal factor; J A^-1 K^T is then the D^-1-weighted dot of two such rows.
template <typename Scalar>
__device__ inline void ForwardMass(const float* factor, uint32_t n, const float* v, const float* extra, Scalar* y) {
    for (uint32_t i = 0u; i < n; ++i) {
        Scalar a = Scalar(v[i]) + (extra == nullptr ? Scalar(0) : Scalar(extra[i]));
        for (uint32_t j = 0u; j < i; ++j) {
            const Scalar l = factor[size_t{i} * n + j];
            a -= l * y[j];
        }
        y[i] = a;
    }
}

template <typename Scalar>
__device__ inline Scalar MassDot(const float* factor, uint32_t n, const Scalar* x, const Scalar* y) {
    Scalar response = 0;
    for (uint32_t i = 0u; i < n; ++i) response += x[i] * y[i] / Scalar(factor[size_t{i} * n + i]);
    return response;
}

// Evaluates J A^-1 K^T without storing inverse matrices or row responses; a symmetric pair needs one solve.
template <typename Scalar>
__device__ __noinline__ Scalar MassCoupling(const float* factor, uint32_t n,
    const float* lhs, const float* rhs, const float* lhs_extra = nullptr, const float* rhs_extra = nullptr) {
    const bool same = lhs == rhs && lhs_extra == rhs_extra;
    auto* left = static_cast<Scalar*>(alloca(size_t{same ? 1u : 2u} * n * sizeof(Scalar)));
    Scalar* const right = same ? left : left + n;
    ForwardMass(factor, n, lhs, lhs_extra, left);
    if (!same) ForwardMass(factor, n, rhs, rhs_extra, right);
    return MassDot(factor, n, left, right);
}

} // namespace nuka::phi::nkops
