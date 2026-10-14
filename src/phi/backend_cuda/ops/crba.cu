// Articulation mass factorization, implicit damping and mimic projection use per-articulation workspace.

#include <cuda_runtime.h>
#include <limits>

#include "math/cuda_spatial_ops.cuh"
#include "math/cuda_vec_ops.cuh"
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/articulation_mass.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/registry.cuh"

namespace nuka::phi {

namespace {

using namespace ::nuka::phi::nkops;

constexpr uint32_t kInvalidLink = ~0u;
constexpr float kMinDiagonal = 1.0e-6f;

// Mimic projection solves in independent coordinates and stores their mass factor.
__global__ void MimicReduceKernel(ArticulationDeviceState state,
                                  const uint32_t* __restrict__ mimic_source_link,
                                  const float* __restrict__ mimic_multiplier,
                                  const float* __restrict__ mimic_offset,
                                  uint32_t max_dof,
                                  const float* __restrict__ inertia_M,
                                  uint32_t* __restrict__ root_dof,
                                  float* __restrict__ root_scale,
                                  float* __restrict__ reduced_M,
                                  float* __restrict__ mass_factor,
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
    float* const Mr = reduced_M + static_cast<size_t>(articulation) * tile_stride;
    float* const factor = mass_factor + static_cast<size_t>(articulation) * tile_stride;
    const auto workspace = CrbaWorkspaceFor(scratch, articulation, max_dof);
    float* const a = workspace.Matrix();
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
        }
        return;
    }

    // Z^T M Z: columns fold onto their roots, then rows, each sum in ascending DOF order.
    for (size_t i = lane; i < tile_stride; i += blockDim.x) {
        Mr[i] = 0.0f;
        factor[i] = 0.0f;
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

    FactorMass(a, a, n, max_dof);
    for (uint32_t i = lane; i < max_dof; i += blockDim.x)
        factor[size_t{i} * max_dof + i] = kMinDiagonal;
    __syncthreads();
    for (uint32_t i = lane; i < n; i += blockDim.x)
        for (uint32_t j = 0u; j <= i; ++j)
            factor[size_t{free_dof[i]} * max_dof + free_dof[j]] = a[size_t{i} * max_dof + j];
    __syncthreads();

    // The M-orthogonal projection of the step velocity onto the coupled coordinates is the reduced
    // dynamics, so an ideal coupling does no work on the motion it admits.
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
    float* const reduced_velocity = workspace.Values(CrbaVector::QdotWork);
    for (uint32_t r = lane; r < n; r += blockDim.x) {
        float sum = 0.0f;
        for (uint32_t b = 0u; b < dof; ++b)
            if (root[b] == free_dof[r]) sum += scale[b] * moment[b];
        reduced_velocity[r] = sum;
    }
    __syncthreads();
    if (lane == 0u) SolveMass(a, reduced_velocity, n, max_dof);
    __syncthreads();
    for (uint32_t r = lane; r < dof; r += blockDim.x)
        projected[r] = scale[r] * reduced_velocity[compact[root[r]]];
    __syncthreads();
    for (uint32_t k = lane; k < dof; k += blockDim.x) {
        if (dof_component[k] != kInvalidLink) {
            state.link_velocity[dof_link[k]].v[dof_component[k]] = projected[k];
        } else {
            state.qdot[dof_link[k]] = projected[k];
        }
    }
}

// --- op entry points ---------------------------------------------------------

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
        data.m == nullptr || data.m_factor == nullptr ||
        data.qdot == nullptr || data.link_velocity == nullptr ||
        model.joint_type == nullptr || model.parent_link == nullptr ||
        model.link_to_articulation == nullptr || model.articulation_link_offset == nullptr ||
        model.articulation_link_count == nullptr || model.mimic_source_link == nullptr ||
        model.mimic_multiplier == nullptr || model.mimic_offset == nullptr ||
        data.m_reduced == nullptr ||
        data.mimic_root_dof == nullptr || data.mimic_root_scale == nullptr) {
        return Status::Failed;
    }
    const ArticulationDeviceState state = MakeArticulationDeviceState(
        model, data, p->total_link_count, p->articulation_count);
    LaunchCuda(MimicReduceKernel, dim3(p->articulation_count), dim3(32u), 0u, stream, state,
               static_cast<const uint32_t*>(model.mimic_source_link),
               static_cast<const float*>(model.mimic_multiplier),
               static_cast<const float*>(model.mimic_offset), p->max_dof,
               static_cast<const float*>(data.m),
               data.mimic_root_dof, data.mimic_root_scale, data.m_reduced, data.m_factor,
               data.crba_scratch, data.env_status);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

} // namespace

void RegisterNkCrbaOps() {
    SetCudaOp(NkOp::MimicReduce, &OpMimicReduce);
}

} // namespace nuka::phi
