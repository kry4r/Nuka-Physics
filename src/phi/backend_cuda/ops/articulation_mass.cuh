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

// Shared DOF maps of one articulation, written by its first warp once per kernel while every warp counts: DOF ->
// (link, component, ~0 for scalar joints), local link -> first DOF. Returns the DOF count, or ~0 on overflow.
__device__ inline uint32_t BuildDofMaps(const ArticulationDeviceState& state, uint32_t articulation,
    uint32_t max_dof, uint32_t* dof_link, uint32_t* dof_component, uint32_t* link_dof, uint32_t* err_status) {
    const uint32_t lane = threadIdx.x % 32u;
    const bool writes = threadIdx.x < 32u;
    const uint32_t offset = state.articulation_link_offset[articulation];
    const uint32_t count = state.articulation_link_count[articulation];
    uint32_t total = 0u;
    for (uint32_t first = 0u; first < count; first += 32u) {
        const uint32_t local = first + lane;
        const uint32_t width = local < count ? JointDofCountDevice(state.joint_type[offset + local]) : 0u;
        uint32_t scan = width;
        for (uint32_t delta = 1u; delta < 32u; delta <<= 1u) {
            const uint32_t below = __shfl_up_sync(~0u, scan, delta);
            if (lane >= delta) scan += below;
        }
        const uint32_t dof = total + scan - width;
        if (writes && local < count && link_dof != nullptr) link_dof[local] = dof;
        for (uint32_t c = 0u; writes && c < width && dof + c < max_dof; ++c) {
            dof_link[dof + c] = offset + local;
            dof_component[dof + c] = width == 1u ? ~0u : c;
        }
        total += __shfl_sync(~0u, scan, 31u);
    }
    if (total > max_dof && threadIdx.x == 0u) atomicOr(err_status, kEnvStatusDofOverflow);
    __syncthreads();
    return total > max_dof ? ~0u : total;
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

// Composite inertia preserves child order; independent columns fill the symmetric mass tile.
__device__ inline void ComputeArticulationMass(ArticulationDeviceState state,
                                                  uint32_t max_dof,
                                                  LinkSpatialInertia* composite,
                                                  float* out_inertia_M,
                                                  const float* joint_damping,
                                                  float dt, ArticulationTreeView tree,
                                                  const uint32_t* link_dof) {
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
        const uint32_t dof_i = link_dof[local];
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
            const uint32_t dof_j = link_dof[walk - offset];
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

// Row stride of a shared factor block; odd, so lanes walking different rows of one column hit different banks.
__host__ __device__ constexpr uint32_t MassWorkStride(uint32_t n) { return n | 1u; }

// Shared bytes of a factor block for max_dof DOFs plus `vectors` DOF vectors and `words` extra words.
inline size_t MassWorkBytes(uint32_t max_dof, uint32_t vectors, uint32_t words) {
    return (size_t{max_dof} * MassWorkStride(max_dof) + size_t{vectors} * max_dof + words) * sizeof(float);
}

// Threads of a block that factors one articulation's mass.
constexpr uint32_t kMassWarps = 4u, kMassThreads = kMassWarps * 32u;

// A = L D L^T, with unit diagonal L and D on the diagonal, factored into the shared block `work` by kMassThreads
// threads. Entry (i, j) stays in lane i % 32 of warp j % kMassWarps and takes the earlier pivots' updates in
// order, as a one-lane column sweep; `add` joins the diagonal, `factor` gets the tile.
__device__ inline void FactorMass(const float* __restrict__ mass, const float* add, float* __restrict__ work,
                                  float* factor, uint32_t n, uint32_t stride) {
    constexpr uint32_t kRows = (kMaxArticulationDof + 31u) / 32u;
    constexpr uint32_t kColumns = (kMaxArticulationDof + kMassWarps - 1u) / kMassWarps;
    const uint32_t lane = threadIdx.x % 32u, warp = threadIdx.x / 32u, ld = MassWorkStride(n);
    float a[kRows][kColumns];
#pragma unroll
    for (uint32_t r = 0u; r < kRows; ++r)
#pragma unroll
        for (uint32_t c = 0u; c < kColumns; ++c) {
            const uint32_t i = lane + 32u * r, j = warp + kMassWarps * c;
            float value = 0.0f;
            if (i < n && j <= i) {
                value = mass[size_t{i} * stride + j];
                if (add != nullptr) value = value + (i == j ? add[i] : 0.0f);
            }
            a[r][c] = value;
        }
    for (uint32_t k = 0u; k < n; ++k) {
        // The pivot's warp scales its column into the shared factor.
        if (warp == k % kMassWarps) {
            const uint32_t slot = k / kMassWarps;
            float column[kRows];
#pragma unroll
            for (uint32_t r = 0u; r < kRows; ++r) {
                column[r] = a[r][0];
#pragma unroll
                for (uint32_t c = 1u; c < kColumns; ++c) column[r] = c == slot ? a[r][c] : column[r];
            }
            float pivot = column[0];
#pragma unroll
            for (uint32_t r = 1u; r < kRows; ++r) pivot = k / 32u == r ? column[r] : pivot;
            const float diagonal = fmaxf(__shfl_sync(~0u, pivot, k % 32u), 1.0e-6f);
#pragma unroll
            for (uint32_t r = 0u; r < kRows; ++r) {
                const uint32_t i = lane + 32u * r;
                if (i > k && i < n) work[i * ld + k] = column[r] / diagonal;
            }
            if (lane == 0u) work[k * ld + k] = diagonal;
        }
        __syncthreads();
        const float d = work[k * ld + k];
        float pivot_row[kColumns];
#pragma unroll
        for (uint32_t c = 0u; c < kColumns; ++c) {
            const uint32_t j = warp + kMassWarps * c;
            pivot_row[c] = j > k && j < n ? work[j * ld + k] : 0.0f;
        }
#pragma unroll
        for (uint32_t r = 0u; r < kRows; ++r) {
            const uint32_t i = lane + 32u * r;
            if (i <= k || i >= n) continue;
            const float l = work[i * ld + k];
#pragma unroll
            for (uint32_t c = 0u; c < kColumns; ++c) {
                const uint32_t j = warp + kMassWarps * c;
                if (j > k && j <= i) a[r][c] -= l * pivot_row[c] * d;
            }
        }
    }
    if (factor == nullptr) return;
    // Readers take the strict lower triangle and the diagonal; padded DOFs keep unit rows with a tiny pivot.
    for (uint32_t i = 0u; i < stride; ++i)
        for (uint32_t j = threadIdx.x; j <= i; j += blockDim.x)
            factor[size_t{i} * stride + j] = i < n ? work[i * ld + j] : i == j ? 1.0e-6f : 0.0f;
    __syncthreads();
}

// Loads a stored factor's leading block into `work`.
__device__ inline void LoadMassFactor(const float* __restrict__ factor, float* __restrict__ work, uint32_t n,
                                      uint32_t stride) {
    const uint32_t ld = MassWorkStride(n);
    for (uint32_t j = threadIdx.x; j < n; j += blockDim.x) {
        for (uint32_t first = 0u; first < n; first += 8u) {
            float value[8];
#pragma unroll
            for (uint32_t r = 0u; r < 8u; ++r)
                value[r] = first + r < n && j <= first + r ? factor[size_t{first + r} * stride + j] : 0.0f;
#pragma unroll
            for (uint32_t r = 0u; r < 8u; ++r)
                if (first + r < n && j <= first + r) work[(first + r) * ld + j] = value[r];
        }
    }
    __syncthreads();
}

// L D L^T x = value in place with the factor in `work`, called by the whole block: the forward sweep spreads each
// column over the block and the backward sweep stays on one lane, both in the serial operation order.
__device__ inline void SolveMassBlock(const float* work, float* value, uint32_t n) {
    const uint32_t lane = threadIdx.x, ld = MassWorkStride(n);
    for (uint32_t j = 0u; j < n; ++j) {
        __syncthreads();
        const float solved = value[j];
        for (uint32_t i = j + 1u + lane; i < n; i += blockDim.x) value[i] -= work[i * ld + j] * solved;
    }
    __syncthreads();
    for (uint32_t i = lane; i < n; i += blockDim.x) value[i] /= work[i * ld + i];
    __syncthreads();
    if (lane == 0u) {
        for (uint32_t end = n; end > 0u; --end) {
            const uint32_t i = end - 1u;
            float sum = value[i];
#pragma unroll 4
            for (uint32_t j = i + 1u; j < n; ++j) sum -= work[j * ld + i] * value[j];
            value[i] = sum;
        }
    }
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
