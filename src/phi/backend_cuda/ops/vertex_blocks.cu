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
// history it is backward Euler, as material-field vertices always are. The row variable is the
// displacement rate u = (x - x_n) / h.
__device__ inline float EffectiveStep(bool second_order, float dt) {
    return second_order ? (2.0f / 3.0f) * dt : dt;
}

// Field element `item` (env-major) on the corners its env's splits gave it.
__device__ inline nk::MfElement CorneredElement(const ModelView& model, const DataView& data,
                                                const VertexBlockLayout& l, uint32_t item) {
    return nk::WithCorners(model.mf_elements[item % l.field_elements],
                           data.mf_corner != nullptr ? data.mf_corner + size_t{item} * nk::kMfMaxVertices : nullptr);
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
    const bool has_history = data.vbd_history_ready[env] != 0u;
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
        data.vbd_solve_rate[item] = driven_velocity;
        data.vbd_offset[item] = {};
        data.vbd_inertia[item] = 0.0f;
        data.vbd_step[item] = p.dt;
        return;
    }
    // A vertex a row pushed or truncation clipped last step restarts at backward Euler.
    const bool second_order = p.integrator == 0u && has_history && data.vbd_restart[item] == 0u &&
                              item % l.vertices < l.vertices - l.field_vertices;
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
    const float inertia = 1.0f / (inv_mass * ht * ht);
    const float stiffness = data.particle_attachment_stiffness[particle];
    // Complete the inertial and attachment quadratics; retain the free rate for physical velocity conversion.
    data.vbd_inertia[item] = inertia + stiffness;
    data.vbd_solve_rate[item] = nk::vbd::AttachmentRate(u, data.particle_kinematic_target[particle] - start,
                                                     inertia, stiffness, h);
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

__global__ void CacheFieldStartKernel(ModelView model, DataView data, ClothStepParams p) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const VertexBlockLayout& l = p.layout;
    if (item >= l.field_elements * l.env_count) return;
    const uint32_t env = item / l.field_elements;
    const nk::MfElement element = CorneredElement(model, data, l, item);
    const uint32_t base = env * l.particles_per_env + l.begin;
    // A mixed-mode seam's crack normal: from each face's edge toward its opposite corner, first side to second.
    if (element.kind == nk::kMfSeam && model.mf_materials[element.material].fracture_law == nk::kMfMixedModeCohesive) {
        math::Vec3 inward[2];
        for (uint32_t side = 0u; side < 2u; ++side) {
            const nk::MfElement bend =
                CorneredElement(model, data, l, env * l.field_elements + element.vertex[4u + side]);
            const uint32_t k = (element.hinges >> (2u * side)) & 3u;
            const math::Vec3 a = data.particle_prev_pos[base + bend.vertex[(k + 1u) % 3u]];
            const math::Vec3 edge = data.particle_prev_pos[base + bend.vertex[(k + 2u) % 3u]] - a;
            const math::Vec3 arm = data.particle_prev_pos[base + bend.vertex[k]] - a;
            const math::Vec3 in = arm - edge * (arm.Dot(edge) / edge.LengthSq());
            inward[side] = in / sqrtf(in.LengthSq());
        }
        const math::Vec3 normal = inward[1] - inward[0];
        const math::Vec3 unit = normal / sqrtf(normal.LengthSq());
        float* history = data.mf_history + size_t{env} * l.field_state_scalars + element.state_offset;
        history[nk::kMfSeamNormal] = unit.x;
        history[nk::kMfSeamNormal + 1u] = unit.y;
        history[nk::kMfSeamNormal + 2u] = unit.z;
        return;
    }
    if (element.kind != nk::kMfTriangle) return;
    nk::vbd::ElementGeometry geometry{};
    for (uint32_t j = 0u; j < 3u; ++j)
        geometry.start[j] = data.particle_prev_pos[base + element.vertex[j]];
    data.mf_start_metric[item] = nk::vbd::MembraneStart(nk::mf::RestEdges(element), geometry);
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
        nkops::BindFieldTables(b, model, data);
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
    data.particle_attachment_previous_stiffness[particle] = data.particle_attachment_stiffness[particle];
    data.particle_attachment_previous_target[particle] = data.particle_kinematic_target[particle];
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

// Each fracture seam gathers the mid-surface normal traction of both faces in their material frames, the opening its
// cohesive law carries.
__device__ void AddEdgeTractions(const nk::MfElement& e, math::Vec3 membrane, float* traction) {
    const float d = e.rest[0] * e.rest[3] - e.rest[1] * e.rest[2];
    const float x[3] = {0.0f, e.rest[3] / d, -e.rest[1] / d}, y[3] = {0.0f, -e.rest[2] / d, e.rest[0] / d};
    for (uint32_t k = 0u; k < 3u; ++k) {
        if (e.vertex[3u + k] == ~0u) continue;
        const float nx = y[(k + 2u) % 3u] - y[(k + 1u) % 3u], ny = x[(k + 1u) % 3u] - x[(k + 2u) % 3u];
        const float normal = (nx * nx * membrane.x + ny * ny * membrane.y + 2.0f * nx * ny * membrane.z) /
                              (nx * nx + ny * ny);
        atomicAdd(traction + e.vertex[3u + k], 0.5f * normal);
    }
}

__device__ void SurfaceStress(const ModelView& model, const DataView& data, const VertexBlockLayout& l,
                              uint32_t env, uint32_t face, math::Vec3& membrane, math::Vec3& bending) {
    const float thickness = model.mf_materials[model.mf_elements[face].material].thickness;
    const size_t base = size_t{env} * l.field_elements;
    membrane = data.mf_stress[base + face] / thickness;
    const uint32_t bend = model.mf_face_partner[face];
    bending = bend != ~0u ? data.mf_stress[base + bend] * (6.0f / (thickness * thickness)) : math::Vec3{};
}

__global__ void FieldTractionKernel(ModelView model, DataView data, ClothStepParams p) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const VertexBlockLayout& l = p.layout;
    if (item >= l.field_elements * l.env_count) return;
    const uint32_t env = item / l.field_elements, face = item % l.field_elements;
    const nk::MfElement& element = model.mf_elements[face];
    if (element.kind != nk::kMfTriangle) return;
    math::Vec3 membrane, bending;
    SurfaceStress(model, data, l, env, face, membrane, bending);
    AddEdgeTractions(element, membrane, data.mf_traction + size_t{env} * l.field_elements);
    data.mf_traction[item] = nk::mf::PrincipalTension(membrane);
}

// Each field element evaluates its law at the step's final positions; the trial history is followed by
// the step's area-scaled plastic and viscous losses.
__global__ void FieldTrialKernel(ModelView model, DataView data, ClothStepParams p) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const VertexBlockLayout& l = p.layout;
    if (item >= l.field_elements * l.env_count) return;
    const uint32_t env = item / l.field_elements;
    const nk::MfElement element = CorneredElement(model, data, l, item);
    nk::mf::FieldGeometry g{};
    for (uint32_t j = 0u; j < nk::MfElementVertexCount(element); ++j)
        g.start[j] = data.particle_pos[env * l.particles_per_env + l.begin + element.vertex[j]];
    nk::mf::FieldResponse<float> r{};
    const float* state = data.mf_history + size_t{env} * l.field_state_scalars + element.state_offset;
    const nk::MfStatus status = nk::mf::ElementTrial(element, model.mf_materials[element.material],
                                                     state, g, p.dt, r);
    float* trial = data.mf_trial + size_t{env} * l.field_trial_scalars + element.trial_offset;
    for (uint32_t k = 0u; k < element.state_count; ++k) trial[k] = r.history[k];
    trial[element.state_count] = nk::mf::ElementArea(element) * r.plastic;
    trial[element.state_count + 1u] = nk::mf::ElementArea(element) * r.viscous;
    if (data.mf_stress != nullptr) data.mf_stress[item] = {r.stress[0], r.stress[1], r.stress[2]};
    const uint32_t flags = (status != nk::MfStatus::Ok ? kEnvStatusMaterialFieldFailure : 0u) |
                           (r.beyond_range ? kEnvStatusMaterialFieldRange : 0u);
    if (flags == 0u) return;
    atomicOr(data.mf_env_commit + env, flags);
    atomicOr(data.env_status + env, flags);
    if (r.beyond_range) atomicMax(data.mf_range_element + env, item % l.field_elements + 1u);
}

// An environment whose field law failed anywhere keeps its whole committed history.
__global__ void FieldCommitKernel(ModelView model, DataView data, ClothStepParams p) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const VertexBlockLayout& l = p.layout;
    if (item >= l.field_elements * l.env_count) return;
    if ((data.mf_env_commit[item / l.field_elements] & kEnvStatusMaterialFieldFailure) != 0u) return;
    const uint32_t env = item / l.field_elements;
    const nk::MfElement& element = model.mf_elements[item % l.field_elements];
    const float* trial = data.mf_trial + size_t{env} * l.field_trial_scalars + element.trial_offset;
    float* state = data.mf_history + size_t{env} * l.field_state_scalars + element.state_offset;
    for (uint32_t k = 0u; k < element.state_count; ++k) state[k] = trial[k];
}

// The strength a committed seam has lost, averaged over its pairs (a joined pair keeps all of it), weakens the hinge
// over it on both sides, so the moment across the seam falls with its traction capacity; the weakening never heals.
__global__ void FieldSeamKernel(ModelView model, DataView data, ClothStepParams p) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const VertexBlockLayout& l = p.layout;
    if (item >= l.field_elements * l.env_count) return;
    const uint32_t env = item / l.field_elements;
    const nk::MfElement element = CorneredElement(model, data, l, item);
    if (element.kind != nk::kMfSeam || (data.mf_env_commit[env] & kEnvStatusMaterialFieldFailure) != 0u) return;
    const nk::MfMaterial& material = model.mf_materials[element.material];
    const nk::mf::CohesiveLaw law = nk::mf::SeamLaw(element, material);
    const float* history = data.mf_history + size_t{env} * l.field_state_scalars + element.state_offset;
    const uint32_t pairs = element.hinges >> 4u;
    double damage = 0.0;
    for (uint32_t pair = 0u; pair < pairs; ++pair) {
        if (nk::mf::SeamPairJoined(element, pair)) continue;
        damage += material.fracture_law == nk::kMfMixedModeCohesive
            ? nk::mf::MixedSpent(nk::mf::MixedSeamLaw(element, material), history[pair],
                                 history[nk::kMfSeamMixity + pair])
            : nk::mf::SeamSpent(law, history[pair]);
    }
    for (uint32_t side = 0u; side < 2u; ++side) {
        float& hinge = data.mf_history[size_t{env} * l.field_state_scalars +
                                       model.mf_elements[element.vertex[4u + side]].state_offset +
                                       nk::kMfHingeDamage + ((element.hinges >> (2u * side)) & 3u)];
        if (pairs > 0u) hinge = fmaxf(hinge, float(damage / pairs));
    }
}

// Whether a seam's edge is cut or cracked.
__device__ bool SeamCracked(const ModelView& model, const float* history, uint32_t seam) {
    return !nk::mf::SeamPairJoined(model.mf_elements[seam], 0u) ||
           history[model.mf_elements[seam].state_offset + nk::kMfSeamCracked] != 0.0f;
}

// Triangles one vertex copy can hold around its source vertex.
constexpr uint32_t kMaxFan = 16u;

__device__ uint32_t FanPart(const uint32_t (&part)[kMaxFan], uint32_t a) {
    while (part[a] != a) a = part[a];
    return a;
}

// Every incidence of a source vertex takes the corner its side triangle holds.
__device__ void FollowAnchors(const ModelView& model, uint32_t* corner, uint32_t begin, uint32_t end) {
    for (uint32_t i = begin; i < end; ++i) {
        const uint32_t packed = model.mf_incidence[i], anchor = model.mf_incidence_anchor[i];
        corner[nk::MfIncidenceElement(packed) * nk::kMfMaxVertices + nk::MfIncidenceLocal(packed)] =
            corner[nk::MfIncidenceElement(anchor) * nk::kMfMaxVertices + nk::MfIncidenceLocal(anchor)];
    }
}

// Splits copy `copy` of source vertex `root` where cracked or cut edges separate its fan: the part with its first
// triangle keeps it, each other part moves to a free spare of a color its new neighbors lack, with its area's mass.
__device__ bool SplitVertex(const ModelView& model, const DataView& data, const VertexBlockLayout& l, uint32_t env,
                            uint32_t root, uint32_t copy) {
    const uint32_t elements = l.field_elements, first = l.vertices - l.field_vertices;
    uint32_t* corner = data.mf_corner + size_t{env} * elements * nk::kMfMaxVertices;
    const float* history = data.mf_history + size_t{env} * l.field_state_scalars;
    uint32_t* root_of = data.particle_root + size_t{env} * l.particles_per_env + l.begin;
    const uint32_t base = env * l.particles_per_env + l.begin;
    const float inv_mass = data.particle_inv_mass[base + copy];
    if (!(inv_mass > 0.0f)) return false;
    const uint32_t begin = model.mf_incidence_offsets[root - first], end = model.mf_incidence_offsets[root - first + 1u];
    uint32_t face[kMaxFan], slot[kMaxFan], part[kMaxFan], faces = 0u;
    for (uint32_t i = begin; i < end; ++i) {
        const uint32_t e = nk::MfIncidenceElement(model.mf_incidence[i]), j = nk::MfIncidenceLocal(model.mf_incidence[i]);
        if (model.mf_elements[e].kind != nk::kMfTriangle || corner[e * nk::kMfMaxVertices + j] != copy) continue;
        if (faces == kMaxFan) return false;
        face[faces] = e;
        slot[faces] = j;
        part[faces] = faces;
        ++faces;
    }
    // Triangles stay joined across their edges through the vertex that are neither cut nor cracked.
    for (uint32_t a = 0u; a < faces; ++a)
        for (uint32_t side = 1u; side < 3u; ++side) {
            const uint32_t seam = model.mf_elements[face[a]].vertex[3u + (slot[a] + side) % 3u];
            if (seam == ~0u || SeamCracked(model, history, seam)) continue;
            for (uint32_t b = a + 1u; b < faces; ++b) {
                const nk::MfElement& t = model.mf_elements[face[b]];
                if (t.vertex[3] != seam && t.vertex[4] != seam && t.vertex[5] != seam) continue;
                const uint32_t pa = FanPart(part, a), pb = FanPart(part, b);
                part[max(pa, pb)] = min(pa, pb);
            }
        }
    float area[kMaxFan] = {}, total = 0.0f;
    for (uint32_t a = 0u; a < faces; ++a) {
        area[FanPart(part, a)] += model.mf_elements[face[a]].rest[4];
        total += model.mf_elements[face[a]].rest[4];
    }
    float kept = total;
    constexpr uint32_t kMoving = ~0u - 1u;
    for (uint32_t r = 1u; r < faces; ++r) {
        if (FanPart(part, r) != r) continue;
        for (uint32_t a = 0u; a < faces; ++a)
            if (FanPart(part, a) == r) corner[face[a] * nk::kMfMaxVertices + slot[a]] = kMoving;
        FollowAnchors(model, corner, begin, end);
        uint32_t taken[8] = {};
        for (uint32_t i = begin; i < end; ++i) {
            const uint32_t e = nk::MfIncidenceElement(model.mf_incidence[i]);
            if (corner[e * nk::kMfMaxVertices + nk::MfIncidenceLocal(model.mf_incidence[i])] != kMoving) continue;
            for (uint32_t k = 0u; k < nk::MfElementVertexCount(model.mf_elements[e]); ++k) {
                const uint32_t x = corner[e * nk::kMfMaxVertices + k];
                const uint32_t color = x != kMoving ? model.mf_color[x - first] : nk::kVbdNoColor;
                if (color != nk::kVbdNoColor) taken[color >> 5u] |= 1u << (color & 31u);
            }
        }
        uint32_t spare = 0u;
        while (spare < l.spares && (root_of[l.spare_first + spare] != l.begin + l.spare_first + spare ||
                                    ((taken[model.mf_color[l.spare_first + spare - first] >> 5u] >>
                                      (model.mf_color[l.spare_first + spare - first] & 31u)) & 1u) != 0u))
            ++spare;
        const uint32_t to = spare < l.spares ? l.spare_first + spare : copy;
        for (uint32_t a = 0u; a < faces; ++a)
            if (FanPart(part, a) == r) corner[face[a] * nk::kMfMaxVertices + slot[a]] = to;
        FollowAnchors(model, corner, begin, end);
        if (to == copy) continue;
        root_of[to] = l.begin + root;
        data.particle_pos[base + to] = data.particle_pos[base + copy];
        data.particle_prev_pos[base + to] = data.particle_prev_pos[base + copy];
        data.particle_vel[base + to] = data.particle_vel[base + copy];
        data.particle_kinematic_target[base + to] = data.particle_kinematic_target[base + copy];
        data.particle_attachment_previous_target[base + to] = data.particle_attachment_previous_target[base + copy];
        data.particle_attachment_stiffness[base + to] = data.particle_attachment_stiffness[base + copy] * area[r] / total;
        data.particle_attachment_previous_stiffness[base + to] =
            data.particle_attachment_previous_stiffness[base + copy] * area[r] / total;
        data.particle_inv_mass[base + to] = inv_mass * total / area[r];
        kept -= area[r];
    }
    if (kept == total) return false;
    data.particle_inv_mass[base + copy] = inv_mass * total / kept;
    data.particle_attachment_stiffness[base + copy] *= kept / total;
    data.particle_attachment_previous_stiffness[base + copy] *= kept / total;
    return true;
}

constexpr uint32_t kCrackFree = 0u, kCrackTip = 1u, kCrackFlank = 2u;

// What the cracks make of a seam's end `pair`: free of them, a tip (an interior vertex one cracked or cut edge ends
// at), or a flank (along a crack, at its mouth on the sheet's edge, or where cracks meet).
__device__ uint32_t CrackEnd(const ModelView& model, const float* history, const VertexBlockLayout& l,
                             uint32_t seam, uint32_t pair) {
    const uint32_t root = model.mf_elements[seam].vertex[2u * pair] - (l.vertices - l.field_vertices);
    uint32_t seams = 0u, triangles = 0u, cracked = 0u;
    for (uint32_t i = model.mf_incidence_offsets[root]; i < model.mf_incidence_offsets[root + 1u]; ++i) {
        const uint32_t e = nk::MfIncidenceElement(model.mf_incidence[i]);
        const nk::MfElement& element = model.mf_elements[e];
        triangles += element.kind == nk::kMfTriangle;
        if (element.kind != nk::kMfSeam || (nk::MfIncidenceLocal(model.mf_incidence[i]) & 1u) != 0u) continue;
        ++seams;
        cracked += SeamCracked(model, history, e);
    }
    return cracked == 0u ? kCrackFree : cracked == 1u && seams == triangles ? kCrackTip : kCrackFlank;
}

__device__ float CrackLoad(const ModelView& model, const DataView& data, const VertexBlockLayout& l, uint32_t env,
                           uint32_t seam) {
    return data.mf_crack_load[size_t{env} * l.field_elements + seam];
}

constexpr uint32_t kCrackNucleus = 1u << 30u, kCrackSeam = kCrackNucleus - 1u;

// A seam's mixed mid-surface load over its strength scaled by `scatter`, averaged over its faces.
__device__ float EdgeMixedLoad(const ModelView& model, const DataView& data, const VertexBlockLayout& l,
                                uint32_t env, uint32_t seam, float scatter) {
    const nk::MfElement& s = model.mf_elements[seam];
    const nk::MfMaterial& m = model.mf_materials[s.material];
    float load = 0.0f;
    for (uint32_t side = 0u; side < 2u; ++side) {
        const uint32_t face = model.mf_face_partner[s.vertex[4u + side]], k = (s.hinges >> (2u * side)) & 3u;
        const nk::MfElement& t = model.mf_elements[face];
        const float d = t.rest[0] * t.rest[3] - t.rest[1] * t.rest[2];
        const float x[3] = {0.0f, t.rest[3] / d, -t.rest[1] / d}, y[3] = {0.0f, -t.rest[2] / d, t.rest[0] / d};
        math::Vec3 normal{y[(k + 2u) % 3u] - y[(k + 1u) % 3u], x[(k + 1u) % 3u] - x[(k + 2u) % 3u], 0.0f};
        normal = normal / sqrtf(normal.LengthSq());
        math::Vec3 membrane, bending;
        SurfaceStress(model, data, l, env, face, membrane, bending);
        const float tensile = m.tear_strength * scatter;
        const float shear = (m.shear_strength > 0.0f ? m.shear_strength : m.tear_strength) * scatter;
        load += 0.5f * nk::mf::MixedTraction(membrane, normal, tensile, shear);
    }
    return load;
}

// A seam opens where its local mixed load reaches the peak its inserted law starts from, nucleating at its
// weakest-link strength away from cracks or extending a crack at its tip; it then holds until it spends its energy.
__global__ void FieldCrackKernel(ModelView model, DataView data, ClothStepParams p) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const VertexBlockLayout& l = p.layout;
    if (item >= l.field_elements * l.env_count) return;
    const uint32_t env = item / l.field_elements, seam = item % l.field_elements;
    data.mf_crack_load[item] = 0.0f;
    const nk::MfElement element = CorneredElement(model, data, l, item);
    if (element.kind != nk::kMfSeam || !nk::mf::SeamPairJoined(element, 0u) ||
        data.mf_history[size_t{env} * l.field_state_scalars + element.state_offset + nk::kMfSeamCracked] != 0.0f ||
        (data.mf_env_commit[env] & kEnvStatusMaterialFieldFailure) != 0u)
        return;
    const uint32_t base = env * l.particles_per_env + l.begin;
    const float* history = data.mf_history + size_t{env} * l.field_state_scalars;
    if (!(data.particle_inv_mass[base + element.vertex[0]] > 0.0f) ||
        !(data.particle_inv_mass[base + element.vertex[2]] > 0.0f))
        return;
    const uint32_t first = CrackEnd(model, history, l, seam, 0u), second = CrackEnd(model, history, l, seam, 1u);
    const bool nucleus = first == kCrackFree && second == kCrackFree;
    if (!nucleus && first != kCrackTip && second != kCrackTip) return;
    const float load = EdgeMixedLoad(model, data, l, env, seam, element.rest[nucleus ? 2u : 3u]);
    data.mf_crack_load[item] = load;
    if (!(load >= 1.0f)) return;
    data.mf_cracks[size_t{env} * l.field_elements + atomicAdd(data.mf_crack_count + env, 1u)] =
        seam | (nucleus ? kCrackNucleus : 0u);
}

// Marks an env's listed seams cracked, then splits the vertices their pairs separate.
__device__ void OpenCracks(const ModelView& model, const DataView& data, const VertexBlockLayout& l, uint32_t env,
                           const uint32_t* cracks, uint32_t count) {
    for (uint32_t c = 0u; c < count; ++c)
        data.mf_history[size_t{env} * l.field_state_scalars + model.mf_elements[cracks[c]].state_offset +
                        nk::kMfSeamCracked] = 1.0f;
    bool split = false;
    for (uint32_t c = 0u; c < count; ++c) {
        const size_t seam = size_t{env} * l.field_elements + cracks[c];
        const nk::MfElement& element = model.mf_elements[cracks[c]];
        for (uint32_t pair = 0u; pair < (element.hinges >> 4u); ++pair)
            split |= SplitVertex(model, data, l, env, element.vertex[2u * pair], data.mf_corner[seam * nk::kMfMaxVertices + 2u * pair]);
    }
    if (split) ++data.mf_topology[env];
}

// Per step each crack tip grows along its most loaded edge and only the most loaded nucleus starts a crack. The opened
// seams are appended to the step's list for the insertion event loop. One env per thread.
__global__ void FieldSplitKernel(ModelView model, DataView data, ClothStepParams p) {
    const uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    const VertexBlockLayout& l = p.layout;
    if (env >= l.env_count || data.mf_crack_count[env] == 0u ||
        (data.mf_env_commit[env] & kEnvStatusMaterialFieldFailure) != 0u)
        return;
    uint32_t* cracks = data.mf_cracks + size_t{env} * l.field_elements;
    const float* history = data.mf_history + size_t{env} * l.field_state_scalars;
    const uint32_t found = data.mf_crack_count[env];
    for (uint32_t i = 1u; i < found; ++i)
        for (uint32_t j = i; j > 0u && cracks[j - 1u] > cracks[j]; --j) {
            const uint32_t swap = cracks[j];
            cracks[j] = cracks[j - 1u];
            cracks[j - 1u] = swap;
        }
    constexpr uint32_t kOpens = 1u << 31u;
    uint32_t nucleus = ~0u;
    float worst = 0.0f;
    for (uint32_t c = 0u; c < found; ++c) {
        const uint32_t s = cracks[c] & kCrackSeam;
        const float load = CrackLoad(model, data, l, env, s);
        if ((cracks[c] & kCrackNucleus) != 0u) {
            if (load > worst) {
                worst = load;
                nucleus = c;
            }
            continue;
        }
        bool best = true;
        for (uint32_t pair = 0u; pair < 2u; ++pair) {
            if (CrackEnd(model, history, l, s, pair) != kCrackTip) continue;
            const uint32_t end = model.mf_elements[s].vertex[2u * pair];
            for (uint32_t d = 0u; d < found; ++d) {
                const uint32_t t = cracks[d] & kCrackSeam;
                if (t == s || (model.mf_elements[t].vertex[0] != end && model.mf_elements[t].vertex[2] != end))
                    continue;
                const float rival = CrackLoad(model, data, l, env, t);
                best &= rival < load || (rival == load && t > s);
            }
        }
        if (best) cracks[c] |= kOpens;
    }
    if (nucleus != ~0u) cracks[nucleus] |= kOpens;
    uint32_t count = 0u;
    for (uint32_t c = 0u; c < found; ++c)
        if ((cracks[c] & kOpens) != 0u) {
            cracks[count] = cracks[c] & kCrackSeam;
            data.mf_history[size_t{env} * l.field_state_scalars +
                            model.mf_elements[cracks[count++]].state_offset + nk::kMfSeamCracked] = 1.0f;
        }
    // Every seam opens at most once per step, so the step's list fits one slot per field element.
    uint32_t* opened = data.mf_opened + size_t{env} * l.field_elements + data.mf_opened_count[env];
    for (uint32_t c = 0u; c < count; ++c) opened[c] = cracks[c];
    data.mf_opened_count[env] += count;
    OpenCracks(model, data, l, env, cracks, count);
}

// Opens the seam list the host placed for each env, in the order the attempt that listed it opened them.
__global__ void FieldReplayKernel(ModelView model, DataView data, ClothStepParams p) {
    const uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= p.layout.env_count) return;
    OpenCracks(model, data, p.layout, env, data.mf_opened + size_t{env} * p.layout.field_elements,
               data.mf_opened_count[env]);
}

// Each collision corner on the field takes the particle its field corner holds after this step's splits.
__global__ void SurfaceCornerKernel(ModelView model, DataView data, ClothStepParams p) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const VertexBlockLayout& l = p.layout;
    const uint32_t corners = l.surface_triangles * 3u;
    if (item >= corners * l.env_count) return;
    const uint32_t env = item / corners, corner = item % corners;
    const uint32_t source = model.particle_surface_corner_source[corner];
    if (source == ~0u) return;
    uint32_t surface = 0u;
    while (corner / 3u - model.particle_surface_info[surface].triangle_offset >=
           model.particle_surface_info[surface].triangle_count)
        ++surface;
    data.particle_surface_corners[item] = l.begin + data.mf_corner[size_t{env} * l.field_elements * nk::kMfMaxVertices + source] -
                                          model.particle_surface_info[surface].vertex_offset;
}

// An edge whose triangles no longer share both its ends keeps its first triangle's side and its twin takes the
// second's; a twin whose edge holds together has both ends on one vertex, which is no edge.
__global__ void SurfaceEdgeKernel(ModelView model, DataView data, ClothStepParams p) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const VertexBlockLayout& l = p.layout;
    if (item >= l.surface_edges * l.env_count) return;
    const uint32_t env = item / l.surface_edges, local = item % l.surface_edges;
    collision::MeshEdge edge = model.particle_surface_edges[local];
    if (edge.triangle0 != ~0u) {
        uint32_t surface = 0u;
        while (local - model.particle_surface_edge_info[surface].edge_offset >=
               model.particle_surface_edge_info[surface].edge_count)
            ++surface;
        const collision::MeshEdgeInfo& info = model.particle_surface_edge_info[surface];
        const uint32_t half = (model.particle_surface_info[surface].flags & collision::kMeshSurfaceTwinEdges) != 0u
                                  ? info.edge_count / 2u : info.edge_count;
        const bool twin = local - info.edge_offset >= half;
        const uint32_t offset = model.particle_surface_info[surface].triangle_offset;
        const uint32_t* corners = data.particle_surface_corners + size_t{env} * l.surface_triangles * 3u;
        const auto at = [&](uint32_t triangle, uint32_t vertex) {
            const size_t first = (size_t{offset} + triangle) * 3u;
            const uint32_t* cooked = model.particle_surface_triangles + first;
            return corners[first + (cooked[0] == vertex ? 0u : cooked[1] == vertex ? 1u : 2u)];
        };
        const collision::MeshEdge cooked = model.particle_surface_edges[twin ? local - half : local];
        edge.vertex0 = at(cooked.triangle0, cooked.vertex0);
        edge.vertex1 = at(cooked.triangle0, cooked.vertex1);
        edge.opposite0 = at(cooked.triangle0, cooked.opposite0);
        const bool joined = cooked.triangle1 == ~0u || (at(cooked.triangle1, cooked.vertex0) == edge.vertex0 &&
                                                        at(cooked.triangle1, cooked.vertex1) == edge.vertex1);
        if (twin && joined) {
            edge.vertex1 = edge.vertex0;
        } else if (twin) {
            edge.vertex0 = at(cooked.triangle1, cooked.vertex0);
            edge.vertex1 = at(cooked.triangle1, cooked.vertex1);
            edge.opposite0 = at(cooked.triangle1, cooked.opposite1);
        } else if (cooked.triangle1 != ~0u) {
            edge.opposite1 = joined ? at(cooked.triangle1, cooked.opposite1) : ~0u;
            edge.triangle1 = joined ? cooked.triangle1 : ~0u;
        }
    }
    data.particle_surface_live_edges[item] = edge;
}

// Collision corners and live edges follow the field corners after splits.
void LaunchSurfaceTopology(const ModelView& model, const DataView& data, const ClothStepParams& p,
                           cudaStream_t stream) {
    const uint32_t corners = p.layout.surface_triangles * 3u * p.layout.env_count;
    const uint32_t edges = p.layout.surface_edges * p.layout.env_count;
    if (corners > 0u)
        LaunchCuda(SurfaceCornerKernel, dim3((corners + kBlockSize - 1u) / kBlockSize), dim3(kBlockSize), 0u,
                   stream, model, data, p);
    if (corners > 0u && edges > 0u)
        LaunchCuda(SurfaceEdgeKernel, dim3((edges + kBlockSize - 1u) / kBlockSize), dim3(kBlockSize), 0u,
                   stream, model, data, p);
}

bool ValidLayout(const ClothStepParams* p) {
    const VertexBlockLayout& l = p->layout;
    const uint64_t max_items = uint64_t{std::numeric_limits<uint32_t>::max()} - (kBlockSize - 1u);
    return p->dt > 0.0f && std::isfinite(p->dt) && p->integrator <= 1u &&
           l.particles_per_env > 0u && l.dynamic_vertices <= l.vertices &&
           uint64_t{l.begin} + l.vertices <= l.particles_per_env &&
           (l.elements == 0u || l.vertices > 0u) && l.field_vertices <= l.vertices &&
           (l.field_elements == 0u || l.field_vertices > 0u) &&
           uint64_t{l.vertices} * l.env_count <= max_items &&
           uint64_t{l.particles_per_env} * l.env_count <= max_items &&
           uint64_t{l.elements} * l.env_count <= max_items &&
           uint64_t{l.field_elements} * l.env_count <= max_items;
}

Status OpClothPredict(const ModelView& model, const DataView& data, const void* params,
                      cudaStream_t stream) {
    const auto* p = static_cast<const ClothStepParams*>(params);
    if (p == nullptr || !ValidLayout(p)) return Status::InvalidArgument;
    if (p->layout.vertices == 0u || p->layout.env_count == 0u) return Status::Ok;
    if (data.particle_response == nullptr || data.particle_row_impulse == nullptr ||
        data.vbd_free_rate == nullptr || data.vbd_solve_rate == nullptr || data.vbd_free_velocity == nullptr ||
        data.particle_attachment_stiffness == nullptr ||
        data.particle_kinematic_target == nullptr || data.vbd_step == nullptr ||
        data.vbd_restart == nullptr ||
        data.vbd_history_ready == nullptr || model.vbd_incidence_offsets == nullptr ||
        (p->layout.elements > 0u && (data.vbd_membrane_start == nullptr ||
            model.vbd_elements == nullptr || model.vbd_incidence == nullptr)) ||
        (p->layout.field_vertices > 0u && (model.mf_incidence_offsets == nullptr ||
            (p->layout.field_elements > 0u && (data.mf_start_metric == nullptr || data.mf_history == nullptr ||
                                               model.mf_elements == nullptr || model.mf_materials == nullptr ||
                                               model.mf_incidence == nullptr)))))
        return Status::InvalidArgument;
    const uint32_t items = p->layout.vertices * p->layout.env_count;
    LaunchCuda(ClothPredictKernel, dim3((items + kBlockSize - 1u) / kBlockSize),
               dim3(kBlockSize), 0u, stream, data, *p);
    const uint32_t elements = p->layout.elements * p->layout.env_count;
    if (elements > 0u)
        LaunchCuda(CacheMembraneStartKernel, dim3((elements + kBlockSize - 1u) / kBlockSize),
                   dim3(kBlockSize), 0u, stream, model, data, *p);
    const uint32_t field_elements = p->layout.field_elements * p->layout.env_count;
    if (field_elements > 0u)
        LaunchCuda(CacheFieldStartKernel, dim3((field_elements + kBlockSize - 1u) / kBlockSize),
                   dim3(kBlockSize), 0u, stream, model, data, *p);
    const uint32_t particles = p->layout.particles_per_env * p->layout.env_count;
    LaunchCuda(ClothResponseKernel, dim3((particles + kBlockSize - 1u) / kBlockSize),
               dim3(kBlockSize), 0u, stream, model, data, *p, particles);
    return cudaGetLastError() == cudaSuccess ? Status::Ok : Status::Failed;
}

Status OpClothFinalize(const ModelView& model, const DataView& data, const void* params,
                       cudaStream_t stream) {
    const auto* p = static_cast<const ClothStepParams*>(params);
    if (p == nullptr || !ValidLayout(p)) return Status::InvalidArgument;
    if (p->layout.vertices == 0u || p->layout.env_count == 0u) return Status::Ok;
    if (data.vbd_free_rate == nullptr || data.vbd_free_velocity == nullptr ||
        data.particle_row_impulse == nullptr ||
        data.vbd_history_ready == nullptr ||
        data.vbd_step == nullptr || data.vbd_restart == nullptr ||
        (p->truncation != 0u && data.dat_particle_beta == nullptr) ||
        (p->layout.field_elements > 0u && (data.mf_history == nullptr || data.mf_trial == nullptr ||
                                           data.mf_env_commit == nullptr || model.mf_elements == nullptr ||
                                           model.mf_materials == nullptr)) ||
        (data.mf_corner != nullptr && (data.mf_traction == nullptr || data.mf_cracks == nullptr ||
                                       data.mf_crack_count == nullptr || data.mf_opened == nullptr ||
                                       data.mf_opened_count == nullptr || data.particle_root == nullptr ||
                                       data.mf_topology == nullptr || model.mf_incidence_anchor == nullptr ||
                                       model.mf_color == nullptr ||
                                       (p->layout.surface_triangles > 0u &&
                                        (data.particle_surface_corners == nullptr ||
                                         data.particle_surface_live_edges == nullptr ||
                                         model.particle_surface_corner_source == nullptr)))))
        return Status::InvalidArgument;
    const uint32_t items = p->layout.vertices * p->layout.env_count;
    LaunchCuda(ClothFinalizeKernel, dim3((items + kBlockSize - 1u) / kBlockSize),
               dim3(kBlockSize), 0u, stream, data, *p);
    LaunchCuda(ClothHistoryKernel, dim3((p->layout.env_count + kBlockSize - 1u) / kBlockSize),
               dim3(kBlockSize), 0u, stream, data.vbd_history_ready, p->layout.env_count);
    const uint32_t field_elements = p->layout.field_elements * p->layout.env_count;
    if (field_elements > 0u) {
        if (cudaMemsetAsync(data.mf_env_commit, 0, p->layout.env_count * sizeof(uint32_t), stream) != cudaSuccess)
            return Status::Failed;
        if (data.mf_corner != nullptr &&
            (cudaMemsetAsync(data.mf_traction, 0, field_elements * sizeof(float), stream) != cudaSuccess ||
             cudaMemsetAsync(data.mf_crack_count, 0, p->layout.env_count * sizeof(uint32_t), stream) != cudaSuccess))
            return Status::Failed;
        const dim3 grid((field_elements + kBlockSize - 1u) / kBlockSize);
        LaunchCuda(FieldTrialKernel, grid, dim3(kBlockSize), 0u, stream, model, data, *p);
        if (data.mf_traction != nullptr)
            LaunchCuda(FieldTractionKernel, grid, dim3(kBlockSize), 0u, stream, model, data, *p);
        if (data.mf_corner != nullptr)
            LaunchCuda(FieldCrackKernel, grid, dim3(kBlockSize), 0u, stream, model, data, *p);
        LaunchCuda(FieldCommitKernel, grid, dim3(kBlockSize), 0u, stream, model, data, *p);
        LaunchCuda(FieldSeamKernel, grid, dim3(kBlockSize), 0u, stream, model, data, *p);
        if (data.mf_corner != nullptr) {
            LaunchCuda(FieldSplitKernel, dim3((p->layout.env_count + kBlockSize - 1u) / kBlockSize), dim3(kBlockSize),
                       0u, stream, model, data, *p);
            LaunchSurfaceTopology(model, data, *p, stream);
        }
    }
    return cudaGetLastError() == cudaSuccess ? Status::Ok : Status::Failed;
}

// The insertion event loop reopens, on the restored step-start state, the seams an earlier attempt opened.
Status OpClothReplaySplits(const ModelView& model, const DataView& data, const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const ClothStepParams*>(params);
    if (p == nullptr || !ValidLayout(p)) return Status::InvalidArgument;
    if (p->layout.field_elements == 0u || p->layout.env_count == 0u || data.mf_corner == nullptr) return Status::Ok;
    if (data.mf_opened == nullptr || data.mf_opened_count == nullptr || data.mf_history == nullptr ||
        data.particle_root == nullptr || data.mf_topology == nullptr || model.mf_elements == nullptr ||
        model.mf_incidence_anchor == nullptr || model.mf_color == nullptr ||
        (p->layout.surface_triangles > 0u &&
         (data.particle_surface_corners == nullptr || data.particle_surface_live_edges == nullptr ||
          model.particle_surface_corner_source == nullptr)))
        return Status::InvalidArgument;
    LaunchCuda(FieldReplayKernel, dim3((p->layout.env_count + kBlockSize - 1u) / kBlockSize), dim3(kBlockSize), 0u,
               stream, model, data, *p);
    LaunchSurfaceTopology(model, data, *p, stream);
    return cudaGetLastError() == cudaSuccess ? Status::Ok : Status::Failed;
}

}  // namespace

void RegisterNkVertexBlockOps() {
    SetCudaOp(NkOp::ClothPredict, &OpClothPredict);
    SetCudaOp(NkOp::ClothFinalize, &OpClothFinalize);
    SetCudaOp(NkOp::ClothReplaySplits, &OpClothReplaySplits);
}

}  // namespace nuka::phi
