// Vertex-block step boundaries: the second-order inertial target and frozen row response before
// the solve, and the velocity and history after the committed displacement.

#include <cmath>
#include <limits>

#include <cuda_runtime.h>

#include "nk/model/generated/views.hpp"
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/registry.cuh"
#include "phi/backend_cuda/ops/vertex_blocks.cuh"

namespace nuka::phi {

using nkops::VertexBlockView;

namespace {

constexpr uint32_t kBlockSize = 128u;

// BDF2 steps x = xbar + ht v with xbar = (4 x_n - x_{n-1}) / 3 and ht = 2h / 3; without
// history it is backward Euler. The row variable is the displacement rate u = (x - x_n) / h.
__device__ inline float EffectiveStep(bool second_order, float dt) {
    return second_order ? (2.0f / 3.0f) * dt : dt;
}

__global__ void ClothPredictKernel(DataView data, ClothStepParams p) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const VertexBlockLayout& l = p.layout;
    if (item >= l.vertices * l.env_count) return;
    const uint32_t env = item / l.vertices;
    const uint32_t particle = env * l.particles_per_env + l.begin + item % l.vertices;
    const math::Vec3 start = data.particle_pos[particle];
    const math::Vec3 before = data.particle_prev_pos[particle];
    const math::Vec3 velocity = data.particle_vel[particle];
    const math::Vec3 earlier = data.vbd_history_vel[item];
    data.particle_prev_pos[particle] = start;
    data.particle_projection_delta[particle] = {};
    data.vbd_history_vel[item] = velocity;
    const float inv_mass = data.particle_inv_mass[particle];
    if (!(inv_mass > 0.0f)) {
        const math::Vec3 target = data.particle_kinematic_target[particle];
        const math::Vec3 driven_velocity = (target - start) / p.dt;
        data.particle_vel[particle] = data.particle_v_pre[particle] = driven_velocity;
        data.pbf_predicted_pos[particle] = target;
        data.vbd_free_rate[item] = data.vbd_free_velocity[item] = driven_velocity;
        data.vbd_offset[item] = {};
        data.vbd_inertia[item] = 0.0f;
        data.vbd_step[item] = p.dt;
        return;
    }
    // A vertex a row pushed or truncation clipped last step restarts at backward Euler.
    const bool second_order = p.integrator == 0u && data.vbd_history_ready[env] != 0u &&
                              data.vbd_restart[item] == 0u;
    const float h = p.dt;
    const float ht = EffectiveStep(second_order, h);
    data.vbd_step[item] = ht;
    const math::Vec3 gravity{p.gravity[0], p.gravity[1], p.gravity[2]};
    const math::Vec3 offset = second_order ? (start - before) * (1.0f / 3.0f) : math::Vec3{};
    const math::Vec3 inertial = second_order
        ? velocity + (velocity - earlier) * (1.0f / 3.0f) + gravity * ht
        : velocity + gravity * h;
    const math::Vec3 u = nk::vbd::DisplacementRate(inertial, offset, h, ht);
    data.vbd_offset[item] = offset;
    data.vbd_free_velocity[item] = inertial;
    data.vbd_free_rate[item] = u;
    data.vbd_inertia[item] = 1.0f / (inv_mass * ht * ht);
    data.particle_vel[particle] = data.particle_v_pre[particle] = u;
    data.pbf_predicted_pos[particle] = start + u * h;
}

__global__ void CacheMembraneStartKernel(ModelView model, DataView data, ClothStepParams p) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const VertexBlockLayout& l = p.layout;
    if (item >= l.elements * l.env_count) return;
    const uint32_t env = item / l.elements;
    const nk::VbdElement element = model.vbd_elements[item % l.elements];
    if (element.kind != nk::kVbdTriangle) return;
    nk::vbd::ElementGeometry geometry{};
    for (uint32_t j = 0u; j < nk::VbdElementVertexCount(element.kind); ++j) {
        const uint32_t particle = env * l.particles_per_env + l.begin + element.vertex[j];
        geometry.start[j] = data.particle_prev_pos[particle];
    }
    data.vbd_membrane_start[item] = nk::vbd::MembraneStart(element, geometry);
}

// Rows respond through W = (h^2 A)^-1 at the initial guess; other particles keep 1/m.
__global__ void ClothResponseKernel(ModelView model, DataView data, ClothStepParams p,
                                    uint32_t particle_count) {
    const uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= particle_count) return;
    data.particle_row_impulse[particle] = {};
    const VertexBlockLayout& l = p.layout;
    const uint32_t env = particle / l.particles_per_env;
    const uint32_t local = particle - env * l.particles_per_env;
    const float inv_mass = data.particle_inv_mass[particle];
    math::SymmetricMat3 response{inv_mass, inv_mass, inv_mass, 0.0f, 0.0f, 0.0f};
    if (local >= l.begin && local - l.begin < l.vertices && inv_mass > 0.0f) {
        const uint32_t vertex = local - l.begin;
        VertexBlockView b;
        b.elements = model.vbd_elements;
        b.membrane_start = data.vbd_membrane_start;
        b.offsets = model.vbd_incidence_offsets;
        b.incidence = model.vbd_incidence;
        b.start = data.particle_prev_pos;
        b.layout = l;
        b.dt = p.dt;
        math::Vec3 gradient;
        math::SymmetricMat3 hessian;
        nkops::GatherVertexBlock(b, data.particle_vel, env, vertex, 0u, 1u, gradient, hessian);
        nk::vbd::AddIdentity(hessian, data.vbd_inertia[env * l.vertices + vertex]);
        math::SymmetricMat3 inverse;
        if (nk::vbd::Invert(nk::vbd::Scaled(hessian, p.dt * p.dt), 0.0f, &inverse)) {
            response = inverse;
        } else {
            response = {};
            atomicOr(data.env_status + env, kEnvStatusSolverFailure);
        }
    }
    data.particle_response[particle] = response;
}

// Position rows added pseudo displacement to x; it stays out of the velocity. The velocity uses
// the step its target was built with; a row impulse or truncation restarts the next target.
__global__ void ClothFinalizeKernel(DataView data, ClothStepParams p) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const VertexBlockLayout& l = p.layout;
    if (item >= l.vertices * l.env_count) return;
    const uint32_t env = item / l.vertices;
    const uint32_t particle = env * l.particles_per_env + l.begin + item % l.vertices;
    if (!(data.particle_inv_mass[particle] > 0.0f)) {
        data.vbd_restart[item] = 1u;
        return;
    }
    const math::Vec3 u = data.particle_vel[particle];
    const math::Vec3 impulse = data.particle_row_impulse[particle];
    const bool truncated = p.truncation != 0u &&
                           data.dat_particle_beta[particle] < 1.0f;
    data.vbd_restart[item] =
        truncated || impulse.x != 0.0f || impulse.y != 0.0f || impulse.z != 0.0f ? 1u : 0u;
    data.particle_vel[particle] = nk::vbd::PhysicalVelocity(
        u, data.vbd_free_rate[item], data.vbd_free_velocity[item], p.dt, data.vbd_step[item]);
}

__global__ void ClothHistoryKernel(uint32_t* ready, uint32_t env_count) {
    const uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < env_count) ready[env] = 1u;
}

bool ValidLayout(const ClothStepParams* p) {
    const VertexBlockLayout& l = p->layout;
    const uint64_t max_items = uint64_t{std::numeric_limits<uint32_t>::max()} - (kBlockSize - 1u);
    return p->dt > 0.0f && std::isfinite(p->dt) && p->integrator <= 1u &&
           l.particles_per_env > 0u && l.dynamic_vertices <= l.vertices &&
           uint64_t{l.begin} + l.vertices <= l.particles_per_env &&
           (l.elements == 0u || l.vertices > 0u) &&
           uint64_t{l.vertices} * l.env_count <= max_items &&
           uint64_t{l.particles_per_env} * l.env_count <= max_items &&
           uint64_t{l.elements} * l.env_count <= max_items;
}

Status OpClothPredict(const ModelView& model, const DataView& data, const void* params,
                      cudaStream_t stream) {
    const auto* p = static_cast<const ClothStepParams*>(params);
    if (p == nullptr || !ValidLayout(p)) return Status::InvalidArgument;
    if (p->layout.vertices == 0u || p->layout.env_count == 0u) return Status::Ok;
    if (data.particle_response == nullptr || data.particle_row_impulse == nullptr ||
        data.vbd_free_rate == nullptr || data.vbd_free_velocity == nullptr ||
        data.particle_kinematic_target == nullptr || data.vbd_step == nullptr ||
        data.vbd_restart == nullptr ||
        data.vbd_history_ready == nullptr || model.vbd_incidence_offsets == nullptr ||
        (p->layout.elements > 0u && (data.vbd_membrane_start == nullptr ||
            model.vbd_elements == nullptr || model.vbd_incidence == nullptr)))
        return Status::InvalidArgument;
    const uint32_t items = p->layout.vertices * p->layout.env_count;
    LaunchCuda(ClothPredictKernel, dim3((items + kBlockSize - 1u) / kBlockSize),
               dim3(kBlockSize), 0u, stream, data, *p);
    const uint32_t elements = p->layout.elements * p->layout.env_count;
    if (elements > 0u)
        LaunchCuda(CacheMembraneStartKernel, dim3((elements + kBlockSize - 1u) / kBlockSize),
                   dim3(kBlockSize), 0u, stream, model, data, *p);
    const uint32_t particles = p->layout.particles_per_env * p->layout.env_count;
    LaunchCuda(ClothResponseKernel, dim3((particles + kBlockSize - 1u) / kBlockSize),
               dim3(kBlockSize), 0u, stream, model, data, *p, particles);
    return cudaGetLastError() == cudaSuccess ? Status::Ok : Status::Failed;
}

Status OpClothFinalize(const ModelView&, const DataView& data, const void* params,
                       cudaStream_t stream) {
    const auto* p = static_cast<const ClothStepParams*>(params);
    if (p == nullptr || !ValidLayout(p)) return Status::InvalidArgument;
    if (p->layout.vertices == 0u || p->layout.env_count == 0u) return Status::Ok;
    if (data.vbd_free_rate == nullptr || data.vbd_free_velocity == nullptr ||
        data.particle_row_impulse == nullptr ||
        data.vbd_history_ready == nullptr ||
        data.vbd_step == nullptr || data.vbd_restart == nullptr ||
        (p->truncation != 0u && data.dat_particle_beta == nullptr))
        return Status::InvalidArgument;
    const uint32_t items = p->layout.vertices * p->layout.env_count;
    LaunchCuda(ClothFinalizeKernel, dim3((items + kBlockSize - 1u) / kBlockSize),
               dim3(kBlockSize), 0u, stream, data, *p);
    LaunchCuda(ClothHistoryKernel, dim3((p->layout.env_count + kBlockSize - 1u) / kBlockSize),
               dim3(kBlockSize), 0u, stream, data.vbd_history_ready, p->layout.env_count);
    return cudaGetLastError() == cudaSuccess ? Status::Ok : Status::Failed;
}

}  // namespace

void RegisterNkVertexBlockOps() {
    SetCudaOp(NkOp::ClothPredict, &OpClothPredict);
    SetCudaOp(NkOp::ClothFinalize, &OpClothFinalize);
}

}  // namespace nuka::phi
