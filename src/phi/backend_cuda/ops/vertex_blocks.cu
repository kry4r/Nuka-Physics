// Vertex-block step boundaries: the second-order inertial target and frozen row response before
// the solve, and the velocity and history after the committed displacement.

#include <cmath>
#include <limits>

#include <cuda_runtime.h>

#include "nk/model/generated/views.hpp"
#include "nk/solve/collidable_owner.hpp"
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/registry.cuh"
#include "phi/backend_cuda/ops/rigid_types.cuh"
#include "phi/backend_cuda/ops/prims_types.cuh"
#include "phi/backend_cuda/ops/vertex_blocks.cuh"

namespace nuka::phi {

using nkops::VertexBlockView;

namespace {

constexpr uint32_t kBlockSize = 128u;

__device__ bool ContactVertex(const ClothStepParams& p, uint32_t env, uint32_t particle,
                             uint32_t* item) {
    const auto& l = p.layout;
    const uint32_t vertex = particle - env * l.particles_per_env - l.begin;
    *item = env * l.vertices + vertex;
    return vertex < l.vertices;
}

__device__ double ContactPointRate(const ClothContactRestartParams& p, const DataView& d,
    uint32_t env, const nk::NkRowSide& side, bool* has_vertex, bool mark) {
    nkops::PointMassView points;
    points.ranges = d.point_endpoint_ranges;
    points.terms = d.point_endpoint_terms;
    double rate = 0.0;
    for (uint32_t i = 0u; i < points.Count(side); ++i) {
        const auto term = points.At(side, i);
        uint32_t item = 0u;
        const bool vertex = term.kind == nk::kNkSideParticle &&
            ContactVertex(p.step, env, term.index, &item) &&
            d.particle_inv_mass[term.index] > 0.0f && term.jacobian.LengthSq() > 0.0f;
        *has_vertex |= vertex;
        if (mark && vertex) atomicOr(d.vbd_history_ready + env, nk::kVbdHistoryRestart);
        const auto velocity = term.kind == nk::kNkSideGrid ? d.grid_velocity[term.index] :
            vertex ? d.vbd_free_rate[item] : d.particle_vel[term.index];
        rate += double(term.jacobian.x) * velocity.x + double(term.jacobian.y) * velocity.y +
                double(term.jacobian.z) * velocity.z;
    }
    return rate;
}

__device__ double ContactSideRate(const ClothContactRestartParams& p, const ModelView& m,
    const DataView& d, uint32_t env, uint32_t kind, uint32_t index, math::Vec3 point,
    math::Vec3 normal, bool* has_vertex, bool mark) {
    if (kind == nk::kUContactSideBody) {
        const auto shape = nkops::LoadPrimShape(m.shape_table, index);
        const auto owner = nk::ResolveCollidableOwner(shape.body_id, env, index,
            p.bodies_per_env, p.links_per_env, p.artics_per_env, m.body_to_link,
            m.body_to_articulation, m.body_collidable_body);
        math::Vec3 angular{}, linear{}, origin{};
        if (owner.kind == nk::kNkSideArtic) {
            const auto pose = d.link_pose[owner.link];
            const auto velocity = d.link_velocity[owner.link];
            angular = nkops::PrimRotate(pose.rotation, {velocity.v[0], velocity.v[1], velocity.v[2]});
            linear = nkops::PrimRotate(pose.rotation, {velocity.v[3], velocity.v[4], velocity.v[5]});
            origin = pose.position;
        } else if (owner.kind == nk::kNkSideRigid) {
            angular = d.body_angular_velocity[owner.body];
            linear = d.body_linear_velocity[owner.body];
            origin = nkops::BodyCenterOfMass(d.body_pose[owner.body], d.body_inertial_frame[owner.body]);
        } else if (owner.kind != nk::kNkSideStatic) return nan("");
        return normal.Dot(linear + angular.Cross(point - origin));
    }
    if (kind == nk::kUContactSideBoundary) return 0.0;
    nk::NkRowSide side{};
    side.kind = kind == nk::kUContactSidePointEndpoint ? nk::kNkSidePointEndpoint :
                kind == nk::kUContactSideGrid ? nk::kNkSideGrid : nk::kNkSideParticle;
    side.index = index;
    side.jlin = normal;
    return ContactPointRate(p, d, env, side, has_vertex, mark);
}

// A unilateral impact invalidates smooth multistep history before its impulse is solved.
__global__ void MarkContactRestartKernel(ModelView m, DataView d, ClothContactRestartParams p) {
    const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= p.contacts_per_env * p.step.layout.env_count ||
        d.ucontact_law[slot] == nk::kContactLawCompliant) return;
    const uint32_t env = slot / p.contacts_per_env;
    const uint32_t count = min(d.ucontact_count[slot], 4u);
    for (uint32_t i = 0u; i < count; ++i) {
        const size_t at = size_t{slot} * 4u + i;
        auto normal = d.ucontact_normal[at];
        const float length = sqrtf(normal.LengthSq());
        if (!(length > 0.0f)) continue;
        normal = normal / length;
        bool vertex = false;
        const auto point = d.ucontact_point[at];
        const double rate = ContactSideRate(p, m, d, env, d.ucontact_a_kind[at],
            d.ucontact_a[at], point, normal, &vertex, false) +
            ContactSideRate(p, m, d, env, d.ucontact_b_kind[at], d.ucontact_b[at],
                            point, normal * -1.0f, &vertex, false);
        if (!isfinite(rate)) {
            atomicOr(d.env_status + env, kEnvStatusInvalidEndpoint);
            continue;
        }
        if (!vertex || double(d.ucontact_depth[at]) < double(p.step.dt) * rate) continue;
        ContactSideRate(p, m, d, env, d.ucontact_a_kind[at], d.ucontact_a[at], point, normal, &vertex, true);
        ContactSideRate(p, m, d, env, d.ucontact_b_kind[at], d.ucontact_b[at], point, normal * -1.0f, &vertex, true);
    }
}

__global__ void RestartContactPredictKernel(DataView d, ClothStepParams p) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const auto& l = p.layout;
    if (item >= l.vertices * l.env_count ||
        (d.vbd_history_ready[item / l.vertices] & nk::kVbdHistoryRestart) == 0u) return;
    const uint32_t particle = (item / l.vertices) * l.particles_per_env + l.begin + item % l.vertices;
    const float inverse_mass = d.particle_inv_mass[particle];
    if (!(inverse_mass > 0.0f)) return;
    const math::Vec3 gravity{p.gravity[0], p.gravity[1], p.gravity[2]};
    const auto velocity = d.vbd_history_vel[item] + gravity * p.dt;
    d.vbd_restart[item] = 1u;
    d.vbd_step[item] = p.dt;
    d.vbd_offset[item] = {};
    d.vbd_free_rate[item] = d.vbd_free_velocity[item] = velocity;
    d.vbd_inertia[item] = 1.0f / (inverse_mass * p.dt * p.dt);
    d.particle_vel[particle] = d.particle_v_pre[particle] = velocity;
    d.pbf_predicted_pos[particle] = d.particle_prev_pos[particle] + velocity * p.dt;
}

// All elastic coordinates share an integration history so internal forces retain equal time weights.
__global__ void MarkHistoryRestartKernel(DataView d, ClothStepParams p) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= p.layout.vertices * p.layout.env_count) return;
    const uint32_t env = item / p.layout.vertices;
    const uint32_t particle = env * p.layout.particles_per_env + p.layout.begin + item % p.layout.vertices;
    if (d.vbd_restart[item] != 0u && d.particle_inv_mass[particle] > 0.0f &&
        (d.vbd_history_ready[env] & nk::kVbdHistoryValid) != 0u)
        atomicOr(d.vbd_history_ready + env, nk::kVbdHistoryRestart);
}

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
    const bool has_history = (data.vbd_history_ready[env] & nk::kVbdHistoryValid) != 0u;
    const math::Vec3 earlier = has_history ? data.vbd_history_vel[item] : velocity;
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
    // A vertex pushed by a row restarts at backward Euler.
    const bool second_order = p.integrator == 0u && has_history &&
                              (data.vbd_history_ready[env] & nk::kVbdHistoryRestart) == 0u;
    const float h = p.dt;
    const float ht = EffectiveStep(second_order, h);
    data.vbd_step[item] = ht;
    const math::Vec3 gravity{p.gravity[0], p.gravity[1], p.gravity[2]};
    const math::Vec3 offset = second_order ? (start - before) * (1.0f / 3.0f) : math::Vec3{};
    const math::Vec3 inertial = second_order
        ? velocity + (velocity - earlier) * (1.0f / 3.0f) + gravity * ht
        : velocity + gravity * h;
    const math::Vec3 u = nk::vbd::DisplacementRate(inertial, offset, h, ht);
    // The iterate starts with gravity scaled by the last step's acceleration along it, clamped to [0, 1]
    // (VBD adaptive initialization), so cloth at rest starts at rest while the target keeps full gravity.
    const float gravity_sq = gravity.Dot(gravity);
    const float along = has_history && gravity_sq > 0.0f
        ? fminf(fmaxf((velocity - earlier).Dot(gravity) / (h * gravity_sq), 0.0f), 1.0f) : 1.0f;
    const math::Vec3 guess = nk::vbd::DisplacementRate(inertial - gravity * ((1.0f - along) * ht), offset, h, ht);
    data.vbd_offset[item] = offset;
    data.vbd_free_velocity[item] = inertial;
    data.vbd_free_rate[item] = u;
    data.vbd_inertia[item] = 1.0f / (inv_mass * ht * ht);
    data.particle_vel[particle] = data.particle_v_pre[particle] = guess;
    data.pbf_predicted_pos[particle] = start + guess * h;
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
        b.free_rate = data.vbd_free_rate;
        b.free_velocity = data.vbd_free_velocity;
        b.inertia = data.vbd_inertia;
        b.effective_step = data.vbd_step;
        b.layout = l;
        b.dt = p.dt;
        double3 gradient;
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
// the step its target was built with; impulses and position recovery restart the next target.
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
    const bool corrected = p.position_pass != 0u && data.particle_pseudo_vel[particle].LengthSq() > 0.0f;
    data.vbd_restart[item] =
        corrected || impulse.x != 0.0f || impulse.y != 0.0f || impulse.z != 0.0f ? 1u : 0u;
    data.particle_vel[particle] = nk::vbd::PhysicalVelocity(
        u, data.vbd_free_rate[item], data.vbd_free_velocity[item], p.dt, data.vbd_step[item]);
}

__global__ void ClothHistoryKernel(uint32_t* ready, uint32_t env_count) {
    const uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < env_count) ready[env] = nk::kVbdHistoryValid;
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
    LaunchCuda(MarkHistoryRestartKernel, dim3((items + kBlockSize - 1u) / kBlockSize),
               dim3(kBlockSize), 0u, stream, data, *p);
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
        data.vbd_step == nullptr || data.vbd_restart == nullptr)
        return Status::InvalidArgument;
    const uint32_t items = p->layout.vertices * p->layout.env_count;
    LaunchCuda(ClothFinalizeKernel, dim3((items + kBlockSize - 1u) / kBlockSize),
               dim3(kBlockSize), 0u, stream, data, *p);
    LaunchCuda(ClothHistoryKernel, dim3((p->layout.env_count + kBlockSize - 1u) / kBlockSize),
               dim3(kBlockSize), 0u, stream, data.vbd_history_ready, p->layout.env_count);
    return cudaGetLastError() == cudaSuccess ? Status::Ok : Status::Failed;
}

Status OpClothContactRestart(const ModelView& model, const DataView& data, const void* params,
                            cudaStream_t stream) {
    const auto* p = static_cast<const ClothContactRestartParams*>(params);
    if (p == nullptr || !ValidLayout(&p->step)) return Status::InvalidArgument;
    const auto& l = p->step.layout;
    if (l.vertices == 0u || l.env_count == 0u || p->contacts_per_env == 0u || p->step.integrator != 0u)
        return Status::Ok;
    if (!data.ucontact_count || !data.ucontact_law || !data.ucontact_normal ||
        !data.ucontact_depth || !data.ucontact_a_kind || !data.ucontact_b_kind ||
        !data.ucontact_a || !data.ucontact_b || !data.ucontact_point ||
        !data.vbd_restart || !data.vbd_history_vel || !data.particle_response)
        return Status::InvalidArgument;
    const uint32_t slots = l.env_count * p->contacts_per_env;
    LaunchCuda(MarkContactRestartKernel, dim3((slots + kBlockSize - 1u) / kBlockSize),
               dim3(kBlockSize), 0u, stream, model, data, *p);
    const uint32_t vertices = l.env_count * l.vertices;
    LaunchCuda(RestartContactPredictKernel, dim3((vertices + kBlockSize - 1u) / kBlockSize),
               dim3(kBlockSize), 0u, stream, data, p->step);
    const uint32_t particles = l.env_count * l.particles_per_env;
    LaunchCuda(ClothResponseKernel, dim3((particles + kBlockSize - 1u) / kBlockSize),
               dim3(kBlockSize), 0u, stream, model, data, p->step, particles);
    return cudaGetLastError() == cudaSuccess ? Status::Ok : Status::Failed;
}

}  // namespace

void RegisterNkVertexBlockOps() {
    SetCudaOp(NkOp::ClothPredict, &OpClothPredict);
    SetCudaOp(NkOp::ClothContactRestart, &OpClothContactRestart);
    SetCudaOp(NkOp::ClothFinalize, &OpClothFinalize);
}

}  // namespace nuka::phi
