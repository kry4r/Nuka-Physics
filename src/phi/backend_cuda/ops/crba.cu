// Articulation mass factorization, implicit damping and mimic projection use per-articulation workspace.

#include <cuda_runtime.h>
#include <limits>

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

enum class CrbaVector : uint32_t {
    Diagonal, Root, Scale, Compact, FreeDof, DofLink, DofComponent,
    Velocity, Projected, Moment, QdotWork, DampingProduct, Count
};
constexpr uint32_t kCrbaVectorCount = static_cast<uint32_t>(CrbaVector::Count);
static_assert(kCrbaVectorCount == 12u && sizeof(float) == sizeof(uint32_t));

struct CrbaWorkspace {
    uint32_t* words;
    uint32_t max_dof;

    __device__ float* Matrix() const {
        return reinterpret_cast<float*>(words);
    }
    __device__ uint32_t* Indices(CrbaVector vector) const {
        return words + size_t{max_dof} * max_dof +
            size_t{static_cast<uint32_t>(vector)} * max_dof;
    }
    __device__ float* Values(CrbaVector vector) const {
        return reinterpret_cast<float*>(Indices(vector));
    }
};

__device__ CrbaWorkspace CrbaWorkspaceFor(
    uint32_t* words, uint32_t articulation, uint32_t max_dof) {
    const size_t stride = size_t{max_dof} * (size_t{max_dof} + kCrbaVectorCount);
    return {words + size_t{articulation} * stride, max_dof};
}

bool CrbaWorkspaceFits(uint32_t articulations, uint32_t max_dof) {
    const uint64_t limit = std::numeric_limits<size_t>::max() / sizeof(uint32_t);
    const uint64_t width = uint64_t{max_dof} + kCrbaVectorCount;
    if (max_dof == 0u || articulations == 0u) return true;
    if (uint64_t{max_dof} > limit / width) return false;
    const uint64_t words = uint64_t{max_dof} * width;
    return uint64_t{articulations} <= limit / words;
}

__device__ uint32_t ActualCrbaDof(const ArticulationDeviceState& state,
    uint32_t articulation, uint32_t max_dof, uint32_t* err_status) {
    const uint32_t offset = state.articulation_link_offset[articulation];
    const uint32_t count = state.articulation_link_count[articulation];
    uint64_t dof = 0u;
    for (uint32_t local = 0u; local < count; ++local) {
        dof += JointDofCountDevice(state.joint_type[offset + local]);
        if (dof > max_dof) {
            atomicOr(err_status, kEnvStatusDofOverflow);
            return kInvalidLink;
        }
    }
    return static_cast<uint32_t>(dof);
}

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

// Each inverse column uses its output column for forward and backward substitution.
__global__ void FactorArticulationInertiaMKernel(ArticulationDeviceState state,
                                                 uint32_t max_dof,
                                                 const float* inertia_M,
                                                 float* out_inertia_M_inv,
                                                 uint32_t* scratch,
                                                 uint32_t* err_status) {
    __shared__ uint32_t dof_sh;

    const uint32_t articulation = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (articulation >= state.articulation_count) {
        return;
    }

    const size_t tile_stride = static_cast<size_t>(max_dof) * max_dof;
    const float* const M = inertia_M + static_cast<size_t>(articulation) * tile_stride;
    float* const Minv = out_inertia_M_inv + static_cast<size_t>(articulation) * tile_stride;
    const auto workspace = CrbaWorkspaceFor(scratch, articulation, max_dof);
    float* const a = workspace.Matrix();
    float* const d = workspace.Values(CrbaVector::Diagonal);

    if (lane == 0u) {
        dof_sh = ActualCrbaDof(state, articulation, max_dof, err_status);
    }
    __syncthreads();
    const uint32_t dof = dof_sh;

    // Zero-fill the output tile (parallel pure-0 writes; the column solves
    // below overwrite the leading dof x dof block).
    for (size_t i = lane; i < tile_stride; i += blockDim.x) {
        Minv[i] = 0.0f;
    }
    if (dof == 0u || dof == kInvalidLink) {
        return;
    }

    if (lane == 0u) {
        // Copy the leading DOF block into this articulation's factor workspace.
        for (uint32_t r = 0u; r < dof; ++r) {
            for (uint32_t c = 0u; c < dof; ++c) {
                a[size_t{r} * max_dof + c] = M[static_cast<size_t>(r) * max_dof + c];
            }
        }

        // Unpivoted LDL^T: A = L D L^T, L unit-lower-triangular, D diagonal.
        // L stored in the strict lower triangle of `a`, D on its diagonal.
        for (uint32_t j = 0u; j < dof; ++j) {
            float djj = a[size_t{j} * max_dof + j];
            for (uint32_t k = 0u; k < j; ++k) {
                djj -= a[size_t{j} * max_dof + k] * a[size_t{j} * max_dof + k] * d[k];
            }
            if (djj < kMinDiagonal) {
                djj = kMinDiagonal;  // SPD floor; guards a degenerate config.
            }
            d[j] = djj;
            for (uint32_t i = j + 1u; i < dof; ++i) {
                float lij = a[size_t{i} * max_dof + j];
                for (uint32_t k = 0u; k < j; ++k) {
                    lij -= a[size_t{i} * max_dof + k] * a[size_t{j} * max_dof + k] * d[k];
                }
                a[size_t{i} * max_dof + j] = lij / djj;
            }
        }
    }
    __syncthreads();

    // Solve A x = e_col for each identity column to form M^-1 (symmetric).
    // INDEPENDENT columns -> one per lane; per-column math byte-identical.
    for (uint32_t col = lane; col < dof; col += blockDim.x) {
        // Forward solve L y = e_col.
        for (uint32_t i = 0u; i < dof; ++i) {
            float value = (i == col) ? 1.0f : 0.0f;
            for (uint32_t k = 0u; k < i; ++k) {
                value -= a[size_t{i} * max_dof + k] * Minv[size_t{k} * max_dof + col];
            }
            Minv[size_t{i} * max_dof + col] = value;
        }
        // Diagonal solve D z = y (in place).
        for (uint32_t i = 0u; i < dof; ++i) {
            Minv[size_t{i} * max_dof + col] /= d[i];
        }
        // Backward solve L^T x = z.
        for (uint32_t ii = dof; ii > 0u; --ii) {
            const uint32_t i = ii - 1u;
            float value = Minv[size_t{i} * max_dof + col];
            for (uint32_t k = i + 1u; k < dof; ++k) {
                value -= a[size_t{k} * max_dof + i] * Minv[size_t{k} * max_dof + col];
            }
            Minv[size_t{i} * max_dof + col] = value;
        }
    }
}

// Backward-Euler damping uses the factored mass and applies its velocity correction once.
__global__ void ApplyImplicitJointDampingKernel(ArticulationDeviceState state,
                                               const float* inertia_M_inv,
                                               const float* joint_damping,
                                               uint32_t dof_stride,
                                               float dt, uint32_t* scratch,
                                               uint32_t* err_status) {
    const uint32_t articulation = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (articulation >= state.articulation_count || lane != 0u) {
        return;
    }

    const uint32_t offset = state.articulation_link_offset[articulation];
    const uint32_t count = state.articulation_link_count[articulation];

    const uint32_t actual_dof = ActualCrbaDof(state, articulation, dof_stride, err_status);
    if (actual_dof == kInvalidLink) return;
    const auto workspace = CrbaWorkspaceFor(scratch, articulation, dof_stride);
    uint32_t* const dof_to_link = workspace.Indices(CrbaVector::DofLink);
    uint32_t* const dof_to_component = workspace.Indices(CrbaVector::DofComponent);
    uint32_t dof = 0u;
    for (uint32_t local = 0u; local < count; ++local) {
        const uint32_t link = offset + local;
        const ArticulationJointType type = state.joint_type[link];
        if (local == 0u && state.parent_link[link] == kInvalidLink &&
            type == ArticulationJointType::FloatingBase) {
            for (uint32_t b = 0u; b < 6u; ++b) {
                dof_to_link[dof] = link;
                dof_to_component[dof] = b;
                ++dof;
            }
            continue;
        }
        if (JointDofCountDevice(type) != 0u) {
            dof_to_link[dof] = link;
            dof_to_component[dof] = kInvalidLink;
            ++dof;
        }
    }
    if (dof == 0u || dof_stride == 0u) {
        return;
    }

    // Working joint-velocity vector. Base DOFs seed from link_velocity[root].v
    // (the omega-first base spatial velocity); scalar joint DOFs from state.qdot.
    float* const qdot_work = workspace.Values(CrbaVector::QdotWork);
    for (uint32_t k = 0u; k < dof; ++k) {
        if (dof_to_component[k] != kInvalidLink) {
            qdot_work[k] = state.link_velocity[dof_to_link[k]].v[dof_to_component[k]];
        } else {
            qdot_work[k] = state.qdot[dof_to_link[k]];
        }
    }

    const size_t tile_stride = static_cast<size_t>(dof_stride) * dof_stride;
    const float* const Minv =
        inertia_M_inv + static_cast<size_t>(articulation) * tile_stride;

    if (joint_damping != nullptr && dt > 0.0f) {
        float* const c_qdot = workspace.Values(CrbaVector::DampingProduct);
        for (uint32_t k = 0u; k < dof; ++k) {
            const float c = (dof_to_component[k] == kInvalidLink)
                                ? joint_damping[dof_to_link[k]]
                                : 0.0f;
            c_qdot[k] = c * qdot_work[k];  // C * qdot_half (qdot_half = seeded qdot_work)
        }
        for (uint32_t r = 0u; r < dof; ++r) {
            float acc = 0.0f;
            const float* const minv_row = Minv + static_cast<size_t>(r) * dof_stride;
            for (uint32_t c = 0u; c < dof; ++c) {
                acc += minv_row[c] * c_qdot[c];
            }
            qdot_work[r] -= dt * acc;
        }
    }

    // Base components write to spatial velocity; scalar joints write to qdot.
    for (uint32_t k = 0u; k < dof; ++k) {
        if (dof_to_component[k] != kInvalidLink) {
            state.link_velocity[dof_to_link[k]].v[dof_to_component[k]] = qdot_work[k];
        } else {
            state.qdot[dof_to_link[k]] = qdot_work[k];
        }
    }
}

// Mimic projection factors Z^T M Z and lifts its inverse back to the full DOF space.
__global__ void MimicReduceKernel(ArticulationDeviceState state,
                                  const uint32_t* __restrict__ mimic_source_link,
                                  const float* __restrict__ mimic_multiplier,
                                  const float* __restrict__ mimic_offset,
                                  uint32_t max_dof,
                                  const float* __restrict__ inertia_M,
                                  const float* __restrict__ inertia_M_inv,
                                  uint32_t* __restrict__ root_dof,
                                  float* __restrict__ root_scale,
                                  float* __restrict__ reduced_M,
                                  float* __restrict__ coupled_M_inv,
                                  float* __restrict__ projection_loss,
                                  uint32_t* scratch, uint32_t* err_status) {
    __shared__ uint32_t dof_sh;
    __shared__ uint32_t free_sh;

    const uint32_t articulation = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (articulation >= state.articulation_count) {
        return;
    }
    const size_t tile_stride = static_cast<size_t>(max_dof) * max_dof;
    const float* const M = inertia_M + static_cast<size_t>(articulation) * tile_stride;
    const float* const Minv = inertia_M_inv + static_cast<size_t>(articulation) * tile_stride;
    float* const Mr = reduced_M + static_cast<size_t>(articulation) * tile_stride;
    float* const Wc = coupled_M_inv + static_cast<size_t>(articulation) * tile_stride;
    const auto workspace = CrbaWorkspaceFor(scratch, articulation, max_dof);
    float* const a = workspace.Matrix();
    float* const d = workspace.Values(CrbaVector::Diagonal);
    uint32_t* const root = workspace.Indices(CrbaVector::Root);
    float* const scale = workspace.Values(CrbaVector::Scale);
    uint32_t* const compact = workspace.Indices(CrbaVector::Compact);
    uint32_t* const free_dof = workspace.Indices(CrbaVector::FreeDof);
    uint32_t* const dof_link = workspace.Indices(CrbaVector::DofLink);
    uint32_t* const dof_component = workspace.Indices(CrbaVector::DofComponent);
    float* const velocity = workspace.Values(CrbaVector::Velocity);
    float* const projected = workspace.Values(CrbaVector::Projected);
    float* const moment = workspace.Values(CrbaVector::Moment);
    if (lane == 0u) dof_sh = ActualCrbaDof(state, articulation, max_dof, err_status);
    __syncthreads();
    if (dof_sh == kInvalidLink) return;

    // DOFs in the solver's order; a DOF whose root lies past the tile stays free.
    if (lane == 0u) {
        const uint32_t offset = state.articulation_link_offset[articulation];
        const uint32_t count = state.articulation_link_count[articulation];
        const uint32_t limit = max_dof;
        uint32_t dof = 0u;
        for (uint32_t local = 0u; local < count; ++local) {
            const uint32_t link = offset + local;
            const ArticulationJointType type = state.joint_type[link];
            if (local == 0u && state.parent_link[link] == kInvalidLink &&
                type == ArticulationJointType::FloatingBase) {
                for (uint32_t b = 0u; b < 6u; ++b) {
                    dof_link[dof] = link;
                    dof_component[dof] = b;
                    root[dof] = dof;
                    scale[dof] = 1.0f;
                    ++dof;
                }
                continue;
            }
            if (JointDofCountDevice(type) == 0u) {
                continue;
            }
            float s = 1.0f;
            float c = 0.0f;
            const uint32_t source = MimicRootDevice(state, mimic_source_link, mimic_multiplier,
                                                    mimic_offset, link, &s, &c);
            const uint32_t source_dof =
                source == link ? dof : LocalDofIndexDevice(state, offset, source);
            dof_link[dof] = link;
            dof_component[dof] = kInvalidLink;
            root[dof] = source_dof < limit ? source_dof : dof;
            scale[dof] = source_dof < limit ? s : 1.0f;
            ++dof;
        }
        uint32_t n = 0u;
        for (uint32_t k = 0u; k < dof; ++k) {
            compact[k] = root[k] == k ? n : kInvalidLink;
            if (root[k] == k) {
                free_dof[n++] = k;
            }
        }
        dof_sh = dof;
        free_sh = n;
    }
    __syncthreads();
    const uint32_t dof = dof_sh;
    const uint32_t n = free_sh;
    for (uint32_t k = lane; k < max_dof; k += blockDim.x) {
        root_dof[static_cast<size_t>(articulation) * max_dof + k] = k < dof ? root[k] : k;
        root_scale[static_cast<size_t>(articulation) * max_dof + k] = k < dof ? scale[k] : 1.0f;
    }
    if (n == dof) {
        for (size_t i = lane; i < tile_stride; i += blockDim.x) {
            Mr[i] = M[i];
            Wc[i] = Minv[i];
        }
        if (lane == 0u) {
            projection_loss[articulation] = 0.0f;
        }
        return;
    }

    // Z^T M Z: columns fold onto their roots, then rows, each sum in ascending DOF order.
    for (size_t i = lane; i < tile_stride; i += blockDim.x) {
        Mr[i] = 0.0f;
        Wc[i] = 0.0f;
    }
    for (size_t i = lane; i < size_t{dof} * n; i += blockDim.x) {
        const uint32_t r = static_cast<uint32_t>(i / n);
        const uint32_t column = free_dof[i - size_t{r} * n];
        float value = 0.0f;
        for (uint32_t b = 0u; b < dof; ++b) {
            if (root[b] == column) {
                value += scale[b] * M[static_cast<size_t>(r) * max_dof + b];
            }
        }
        a[size_t{r} * max_dof + i - size_t{r} * n] = value;
    }
    __syncthreads();
    for (size_t i = lane; i < size_t{n} * n; i += blockDim.x) {
        const uint32_t ci = static_cast<uint32_t>(i / n);
        const uint32_t cj = static_cast<uint32_t>(i - size_t{ci} * n);
        if (cj > ci) {
            continue;
        }
        float value = 0.0f;
        for (uint32_t r = 0u; r < dof; ++r) {
            if (root[r] == free_dof[ci]) {
                value += scale[r] * a[size_t{r} * max_dof + cj];
            }
        }
        Mr[static_cast<size_t>(free_dof[ci]) * max_dof + free_dof[cj]] = value;
        Mr[static_cast<size_t>(free_dof[cj]) * max_dof + free_dof[ci]] = value;
    }
    __syncthreads();
    for (size_t i = lane; i < size_t{n} * n; i += blockDim.x) {
        const uint32_t ci = static_cast<uint32_t>(i / n);
        const uint32_t cj = static_cast<uint32_t>(i - size_t{ci} * n);
        a[size_t{ci} * max_dof + cj] = Mr[static_cast<size_t>(free_dof[ci]) * max_dof + free_dof[cj]];
    }
    __syncthreads();

    if (lane == 0u) {
        for (uint32_t j = 0u; j < n; ++j) {
            float djj = a[size_t{j} * max_dof + j];
            for (uint32_t k = 0u; k < j; ++k) {
                djj -= a[size_t{j} * max_dof + k] * a[size_t{j} * max_dof + k] * d[k];
            }
            if (djj < kMinDiagonal) {
                djj = kMinDiagonal;  // SPD floor; guards a degenerate config.
            }
            d[j] = djj;
            for (uint32_t i = j + 1u; i < n; ++i) {
                float lij = a[size_t{i} * max_dof + j];
                for (uint32_t k = 0u; k < j; ++k) {
                    lij -= a[size_t{i} * max_dof + k] * a[size_t{j} * max_dof + k] * d[k];
                }
                a[size_t{i} * max_dof + j] = lij / djj;
            }
        }
    }
    __syncthreads();

    // Each lane solves reduced columns; W_c[r][b] = s_r s_b W_r[root r][root b] for the
    // DOFs b that follow the column's DOF, so lanes write disjoint columns.
    for (uint32_t col = lane; col < n; col += blockDim.x) {
        for (uint32_t i = 0u; i < n; ++i) {
            float value = (i == col) ? 1.0f : 0.0f;
            for (uint32_t k = 0u; k < i; ++k) {
                value -= a[size_t{i} * max_dof + k] * Wc[size_t{free_dof[k]} * max_dof + free_dof[col]];
            }
            Wc[size_t{free_dof[i]} * max_dof + free_dof[col]] = value;
        }
        for (uint32_t i = 0u; i < n; ++i) {
            Wc[size_t{free_dof[i]} * max_dof + free_dof[col]] /= d[i];
        }
        for (uint32_t ii = n; ii > 0u; --ii) {
            const uint32_t i = ii - 1u;
            float value = Wc[size_t{free_dof[i]} * max_dof + free_dof[col]];
            for (uint32_t k = i + 1u; k < n; ++k) {
                value -= a[size_t{k} * max_dof + i] * Wc[size_t{free_dof[k]} * max_dof + free_dof[col]];
            }
            Wc[size_t{free_dof[i]} * max_dof + free_dof[col]] = value;
        }
        for (uint32_t b = 0u; b < dof; ++b) {
            if (root[b] != free_dof[col]) {
                continue;
            }
            for (uint32_t r = 0u; r < dof; ++r) {
                Wc[static_cast<size_t>(r) * max_dof + b] = scale[r] * scale[b] *
                    Wc[size_t{free_dof[compact[root[r]]]} * max_dof + free_dof[col]];
            }
        }
    }
    __syncthreads();

    // The coupling impulse takes the step velocity to its M-orthogonal projection W_c M v;
    // its work 1/2 (v + v_c)^T M (v - v_c) is booked like a row's.
    for (uint32_t k = lane; k < dof; k += blockDim.x) {
        velocity[k] = dof_component[k] != kInvalidLink
                          ? state.link_velocity[dof_link[k]].v[dof_component[k]]
                          : state.qdot[dof_link[k]];
    }
    __syncthreads();
    for (uint32_t r = lane; r < dof; r += blockDim.x) {
        float sum = 0.0f;
        for (uint32_t b = 0u; b < dof; ++b) {
            sum += M[static_cast<size_t>(r) * max_dof + b] * velocity[b];
        }
        moment[r] = sum;
    }
    __syncthreads();
    for (uint32_t r = lane; r < dof; r += blockDim.x) {
        if (root[r] != r) {
            continue;
        }
        float sum = 0.0f;
        for (uint32_t b = 0u; b < dof; ++b) {
            sum += Wc[static_cast<size_t>(r) * max_dof + b] * moment[b];
        }
        projected[r] = sum;
    }
    __syncthreads();
    for (uint32_t r = lane; r < dof; r += blockDim.x) {
        if (root[r] != r) {
            projected[r] = scale[r] * projected[root[r]];
        }
    }
    __syncthreads();
    for (uint32_t r = lane; r < dof; r += blockDim.x) {
        float sum = 0.0f;
        for (uint32_t b = 0u; b < dof; ++b) {
            sum += M[static_cast<size_t>(r) * max_dof + b] * (velocity[b] - projected[b]);
        }
        moment[r] = sum;
    }
    __syncthreads();
    if (lane == 0u) {
        double loss = 0.0;
        for (uint32_t r = 0u; r < dof; ++r) {
            loss += 0.5 * (double(velocity[r]) + projected[r]) * moment[r];
        }
        projection_loss[articulation] = static_cast<float>(loss);
    }
    for (uint32_t k = lane; k < dof; k += blockDim.x) {
        if (dof_component[k] != kInvalidLink) {
            state.link_velocity[dof_link[k]].v[dof_component[k]] = projected[k];
        } else {
            state.qdot[dof_link[k]] = projected[k];
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
        (p->fold_drive_damping != 0u) ? data.drive_dissipation : nullptr;
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
    if (!CrbaWorkspaceFits(p->articulation_count, p->max_dof) ||
        data.crba_scratch == nullptr || data.env_status == nullptr ||
        data.m == nullptr || data.m_inv == nullptr || model.joint_type == nullptr ||
        model.articulation_link_offset == nullptr || model.articulation_link_count == nullptr) {
        return Status::Failed;
    }
    const ArticulationDeviceState state = MakeArticulationDeviceState(
        model, data, /*total_link_count=*/0u, p->articulation_count);
    // Actual DOFs beyond the allocated tile set the overflow diagnostic and skip factorization.
    LaunchCuda(FactorArticulationInertiaMKernel, dim3(p->articulation_count),
               dim3(32u), 0u, stream, state, p->max_dof, data.m, data.m_inv,
               data.crba_scratch, data.env_status);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

Status OpApplyImplicitDamping(const ModelView& model, const DataView& data,
                              const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const ApplyImplicitDampingParams*>(params);
    if (p == nullptr) {
        return Status::Failed;
    }
    if (p->articulation_count == 0u || p->max_dof == 0u) {
        return Status::Ok;
    }
    if (!CrbaWorkspaceFits(p->articulation_count, p->max_dof) ||
        data.crba_scratch == nullptr || data.env_status == nullptr ||
        data.m_inv == nullptr || data.qdot == nullptr || data.link_velocity == nullptr ||
        model.joint_type == nullptr || model.parent_link == nullptr ||
        model.articulation_link_offset == nullptr || model.articulation_link_count == nullptr) {
        return Status::Failed;
    }
    const ArticulationDeviceState state = MakeArticulationDeviceState(
        model, data, /*total_link_count=*/0u, p->articulation_count);
    LaunchCuda(ApplyImplicitJointDampingKernel, dim3(p->articulation_count),
               dim3(32u), 0u, stream, state, data.m_inv, data.drive_dissipation,
               p->max_dof, p->dt, data.crba_scratch, data.env_status);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

Status OpMimicReduce(const ModelView& model, const DataView& data,
                     const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const MimicReduceParams*>(params);
    if (p == nullptr) {
        return Status::Failed;
    }
    if (p->articulation_count == 0u || p->total_link_count == 0u || p->max_dof == 0u) {
        return Status::Ok;
    }
    if (!CrbaWorkspaceFits(p->articulation_count, p->max_dof) ||
        data.crba_scratch == nullptr || data.env_status == nullptr ||
        data.m == nullptr || data.m_inv == nullptr ||
        data.qdot == nullptr || data.link_velocity == nullptr ||
        model.joint_type == nullptr || model.parent_link == nullptr ||
        model.link_to_articulation == nullptr || model.articulation_link_offset == nullptr ||
        model.articulation_link_count == nullptr || model.mimic_source_link == nullptr ||
        model.mimic_multiplier == nullptr || model.mimic_offset == nullptr ||
        data.m_reduced == nullptr || data.m_inv_coupled == nullptr ||
        data.mimic_root_dof == nullptr || data.mimic_root_scale == nullptr ||
        data.mimic_projection_loss == nullptr) {
        return Status::Failed;
    }
    const ArticulationDeviceState state = MakeArticulationDeviceState(
        model, data, p->total_link_count, p->articulation_count);
    LaunchCuda(MimicReduceKernel, dim3(p->articulation_count), dim3(32u), 0u, stream, state,
               static_cast<const uint32_t*>(model.mimic_source_link),
               static_cast<const float*>(model.mimic_multiplier),
               static_cast<const float*>(model.mimic_offset), p->max_dof,
               static_cast<const float*>(data.m), static_cast<const float*>(data.m_inv),
               data.mimic_root_dof, data.mimic_root_scale, data.m_reduced, data.m_inv_coupled,
               data.mimic_projection_loss, data.crba_scratch, data.env_status);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

} // namespace

void RegisterNkCrbaOps() {
    SetCudaOp(NkOp::CrbaComputeM, &OpCrbaComputeM);
    SetCudaOp(NkOp::CrbaFactorM, &OpCrbaFactorM);
    SetCudaOp(NkOp::ApplyImplicitDamping, &OpApplyImplicitDamping);
    SetCudaOp(NkOp::MimicReduce, &OpMimicReduce);
}

} // namespace nuka::phi
