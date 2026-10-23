#pragma once
// Vertex blocks descend inertial and elastic energy through the current local curvature.
// Rows use the step's frozen response W; their accumulated impulses stay fixed during each descent.

#include <cstdint>

#include "math/symmetric_mat3.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/views.hpp"
#include "nk/readout/physics_diagnostics.hpp"
#include "nk/solve/field_block.hpp"
#include "nk/solve/vertex_block.hpp"
#include "phi/backend_cuda/ops/union_types.cuh"
#include "phi/op_schema.hpp"

namespace nuka::phi::nkops {

struct VertexBlockView {
    const nk::VbdElement* elements = nullptr;
    const nk::vbd::MembraneStartState* membrane_start = nullptr;
    const uint32_t* offsets = nullptr;
    const uint32_t* incidence = nullptr;
    const uint32_t* color_vertices = nullptr;
    const uint32_t* color_segments = nullptr;
    const math::Vec3* start = nullptr;  // interval-start positions
    const math::Vec3* free_rate = nullptr;
    const float* inertia = nullptr;
    const float* effective_step = nullptr;
    const uint32_t* row_first = nullptr;
    float* solver_audit = nullptr;
    uint32_t* env_status = nullptr;
    // Material-field elements, materials, incidence of the field vertices, interval-start metrics and the
    // committed history of each element.
    const nk::MfElement* field_elements = nullptr;
    const nk::MfMaterial* field_materials = nullptr;
    const uint32_t* field_offsets = nullptr;
    const uint32_t* field_incidence = nullptr;
    const nk::vbd::MembraneStartState* field_start = nullptr;
    const float* field_history = nullptr;
    VertexBlockLayout layout{};
    float dt = 0.0f;

    __device__ uint32_t Particle(uint32_t env, uint32_t vertex) const {
        return env * layout.particles_per_env + layout.begin + vertex;
    }
    __device__ uint32_t Slot(uint32_t env, uint32_t vertex) const {
        return env * layout.vertices + vertex;
    }
    __device__ uint32_t FieldFirst() const { return layout.vertices - layout.field_vertices; }
};

__host__ __device__ inline void BindFieldTables(VertexBlockView& b, const ModelView& model, const DataView& data) {
    b.field_elements = model.mf_elements;
    b.field_materials = model.mf_materials;
    b.field_offsets = model.mf_incidence_offsets;
    b.field_incidence = model.mf_incidence;
    b.field_start = data.mf_start_metric;
    b.field_history = data.mf_history;
}

// A vertex's element list: its vertex-block elements, or its material-field elements past the first field vertex.
struct VertexElementList {
    const uint32_t* incidence;
    uint32_t begin;
    uint32_t end;
    bool field;
};

__device__ inline VertexElementList ElementsOf(const VertexBlockView& b, uint32_t vertex) {
    if (vertex < b.FieldFirst()) return {b.incidence, b.offsets[vertex], b.offsets[vertex + 1u], false};
    const uint32_t v = vertex - b.FieldFirst();
    return {b.field_incidence, b.field_offsets[v], b.field_offsets[v + 1u], true};
}

__device__ inline void RecordVertexDescentAudit(const VertexBlockView& b, uint32_t env,
                                               uint32_t vertex, math::Vec3 force,
                                               math::Vec3 direction, math::Vec3 from,
                                               math::Vec3 next, float scale, float change,
                                               uint32_t halvings, bool accepted, bool descending) {
    if (b.solver_audit == nullptr) return;
    using Column = nk::VbdSolveAuditColumn;
    float* record = b.solver_audit + size_t{b.Slot(env, vertex)} * nk::kVbdSolveAuditColumnCount;
    const auto set = [&](Column column, float value) { record[static_cast<uint32_t>(column)] = value; };
    const auto add = [&](Column column) { record[static_cast<uint32_t>(column)] += 1.0f; };
    set(Column::LastDirectionX, direction.x);
    set(Column::LastDirectionY, direction.y);
    set(Column::LastDirectionZ, direction.z);
    set(Column::LastScale, accepted ? scale : 0.0f);
    set(Column::LastEnergyChange, accepted ? change : 0.0f);
    set(Column::LastHalvings, static_cast<float>(halvings));
    set(Column::LastPrimalForceX, force.x);
    set(Column::LastPrimalForceY, force.y);
    set(Column::LastPrimalForceZ, force.z);
    add(Column::TotalSteps);
    if (accepted) {
        add(Column::AcceptedSteps);
        if (next.x == from.x && next.y == from.y && next.z == from.z) add(Column::RoundedSteps);
    } else if (descending) {
        add(Column::RejectedSteps);
    } else {
        add(Column::ZeroSlopeSteps);
    }
}

__device__ inline void RecordVertexEquationAudit(const VertexBlockView& b, uint32_t env,
                                                uint32_t vertex, math::Vec3 force,
                                                math::Vec3 correction) {
    if (b.solver_audit == nullptr) return;
    using Column = nk::VbdSolveAuditColumn;
    float* record = b.solver_audit + size_t{b.Slot(env, vertex)} * nk::kVbdSolveAuditColumnCount;
    record[static_cast<uint32_t>(Column::ForceX)] = force.x;
    record[static_cast<uint32_t>(Column::ForceY)] = force.y;
    record[static_cast<uint32_t>(Column::ForceZ)] = force.z;
    record[static_cast<uint32_t>(Column::NewtonCorrectionX)] = correction.x;
    record[static_cast<uint32_t>(Column::NewtonCorrectionY)] = correction.y;
    record[static_cast<uint32_t>(Column::NewtonCorrectionZ)] = correction.z;
}

// Whether a point side moves a vertex-block particle.
__device__ inline bool TouchesVertexBlock(const NkRowSide& side, const PointMassView& points,
                                          const VertexBlockLayout& l) {
    if (l.vertices == 0u || !PointMassView::IsPointSide(side.kind)) return false;
    for (uint32_t i = 0u; i < points.Count(side); ++i) {
        const auto term = points.At(side, i);
        if (term.kind != kNkSideParticle) continue;
        const uint32_t local = term.index % l.particles_per_env;
        if (local >= l.begin && local - l.begin < l.vertices) return true;
    }
    return false;
}

// Interval-start positions and relative in-step displacements reach the elements separately.
template <class Geometry = nk::vbd::ElementGeometry>
__device__ inline Geometry StepGeometry(const VertexBlockView& b, const math::Vec3* velocity,
                                        const uint32_t* vertex, uint32_t count, uint32_t env) {
    Geometry geometry;
    const math::Vec3 rate = velocity[b.Particle(env, vertex[0])];
    // A loop over every slot, not to `count`, keeps the slots in registers.
    for (uint32_t j = 0u; j < sizeof(Geometry::start) / sizeof(math::Vec3); ++j) {
        if (j >= count) continue;
        const uint32_t particle = b.Particle(env, vertex[j]);
        geometry.start[j] = b.start[particle];
        geometry.delta[j] = (velocity[particle] - rate) * b.dt;
    }
    return geometry;
}

// The displacement of slot `local`, set through selects for the same reason.
template <class Geometry>
__device__ inline void SetDelta(Geometry& geometry, uint32_t local, math::Vec3 delta) {
    for (uint32_t j = 0u; j < sizeof(Geometry::delta) / sizeof(math::Vec3); ++j)
        if (j == local) geometry.delta[j] = delta;
}

__device__ inline nk::vbd::ElementGeometry ElementStepGeometry(const VertexBlockView& b,
                                                              const math::Vec3* velocity,
                                                              const nk::VbdElement& element,
                                                              uint32_t env) {
    return StepGeometry(b, velocity, element.vertex, nk::VbdElementVertexCount(element.kind), env);
}

__device__ inline const nk::vbd::MembraneStartState* FieldStart(const VertexBlockView& b, uint32_t env,
                                                                uint32_t element) {
    return b.field_start != nullptr ? b.field_start + env * b.layout.field_elements + element : nullptr;
}

__device__ inline const float* FieldHistory(const VertexBlockView& b, uint32_t env, uint32_t element) {
    return b.field_history + (size_t{env} * b.layout.field_elements + element) * nk::kMfHistory;
}

__device__ inline nk::mf::FieldGeometry FieldStepGeometry(const VertexBlockView& b, const math::Vec3* velocity,
                                                         const nk::MfElement& element, uint32_t env) {
    return StepGeometry<nk::mf::FieldGeometry>(b, velocity, element.vertex, nk::MfElementVertexCount(element), env);
}

// Element damping uses all nodal rates and a material metric frozen at the interval start.
__device__ inline void VertexElementRayleigh(const VertexBlockView& b, const math::Vec3* velocity,
                                            const nk::VbdElement& element, uint32_t env,
                                            uint32_t local, math::Vec3 own,
                                            math::Vec3& applied, math::SymmetricMat3& diagonal,
                                            const nk::vbd::MembraneStartState* membrane = nullptr) {
    nk::vbd::ElementGeometry start;
    math::Vec3 rates[4];
    for (uint32_t j = 0u; j < 4u; ++j) {
        if (j >= nk::VbdElementVertexCount(element.kind)) continue;
        const uint32_t particle = b.Particle(env, element.vertex[j]);
        start.start[j] = b.start[particle];
        rates[j] = j == local ? own : velocity[particle];
    }
    nk::vbd::ElementRayleighBlock(element, local, start, rates, &applied, &diagonal, membrane);
}

// Rayleigh forces and their local curvature come from the same complete element potential.
__device__ inline void VertexElementBlock(const VertexBlockView& b, const math::Vec3* velocity,
                                          uint32_t env, math::Vec3 own, uint32_t packed,
                                          math::Vec3& g, math::SymmetricMat3& h) {
    const uint32_t element_index = nk::VbdIncidenceElement(packed);
    const nk::VbdElement element = b.elements[element_index];
    const auto* start = b.membrane_start != nullptr
        ? b.membrane_start + env * b.layout.elements + element_index : nullptr;
    const nk::vbd::ElementGeometry geometry = ElementStepGeometry(b, velocity, element, env);
    const uint32_t local = nk::VbdIncidenceLocal(packed);
    nk::vbd::ElementBlock(element, local, geometry, &g, &h, start);
    if (element.damping > 0.0f) {
        math::Vec3 damping_force;
        math::SymmetricMat3 damping_diagonal;
        VertexElementRayleigh(b, velocity, element, env, local, own,
                              damping_force, damping_diagonal, start);
        g = g + damping_force * element.damping;
        nk::vbd::AddTo(h, nk::vbd::Scaled(damping_diagonal, element.damping / b.dt));
    }
}

// A material-field element's gradient and Hessian block at its vertex, at the committed history.
__device__ inline void FieldElementBlock(const VertexBlockView& b, const math::Vec3* velocity, uint32_t env,
                                         uint32_t packed, math::Vec3& g, math::SymmetricMat3& h) {
    const uint32_t index = nk::MfIncidenceElement(packed);
    const nk::MfElement element = b.field_elements[index];
    nk::mf::ElementBlock(element, b.field_materials[element.material], FieldHistory(b, env, index),
                         nk::MfIncidenceLocal(packed), FieldStepGeometry(b, velocity, element, env),
                         FieldStart(b, env, index), b.dt, &g, &h);
}

__device__ inline void ListElementBlock(const VertexBlockView& b, const math::Vec3* velocity, uint32_t env,
                                        math::Vec3 own, bool field, uint32_t packed, math::Vec3& g,
                                        math::SymmetricMat3& h) {
    if (field) FieldElementBlock(b, velocity, env, packed, g, h);
    else VertexElementBlock(b, velocity, env, own, packed, g, h);
}

__device__ inline void GatherVertexBlock(const VertexBlockView& b, const math::Vec3* velocity,
                                         uint32_t env, uint32_t vertex, uint32_t lane,
                                         uint32_t width, math::Vec3& gradient,
                                         math::SymmetricMat3& hessian, math::Vec3* magnitude = nullptr) {
    gradient = {};
    hessian = {};
    const math::Vec3 own = velocity[b.Particle(env, vertex)];
    const VertexElementList list = ElementsOf(b, vertex);
    for (uint32_t i = list.begin + lane; i < list.end; i += width) {
        math::Vec3 g;
        math::SymmetricMat3 h;
        ListElementBlock(b, velocity, env, own, list.field, list.incidence[i], g, h);
        gradient = gradient + g;
        nk::vbd::AddTo(hessian, h);
        if (magnitude != nullptr) *magnitude = *magnitude + math::Vec3{fabsf(g.x), fabsf(g.y), fabsf(g.z)};
    }
}

// The vertex force with its terms summed in double, one rounding from their exact sum in any order.
__device__ inline math::Vec3 VertexBlockForceDouble(const VertexBlockView& b, const math::Vec3* velocity,
                                                    uint32_t env, uint32_t vertex, math::Vec3 inertial,
                                                    math::Vec3 impulse) {
    const math::Vec3 own = velocity[b.Particle(env, vertex)];
    double force[3] = {double(impulse.x) - inertial.x, double(impulse.y) - inertial.y,
                       double(impulse.z) - inertial.z};
    const VertexElementList list = ElementsOf(b, vertex);
    for (uint32_t i = list.begin; i < list.end; ++i) {
        math::Vec3 g;
        math::SymmetricMat3 h;
        ListElementBlock(b, velocity, env, own, list.field, list.incidence[i], g, h);
        force[0] -= g.x;
        force[1] -= g.y;
        force[2] -= g.z;
    }
    return {static_cast<float>(force[0]), static_cast<float>(force[1]), static_cast<float>(force[2])};
}

// Backtracking covers the significand resolution of a single-precision step.
constexpr uint32_t kVertexStepHalvings = 24u;
// Fraction of the first-order decrease a vertex step must realize.
constexpr float kVertexStepDecrease = 1.0e-4f;
// Trial steps start over-relaxed to carry smooth error across stiff membranes in fewer sweeps;
// Armijo backtracking still enforces descent. Row-coupled blocks start at their dominating-metric step.
constexpr float kVertexStepRelaxation = 1.9f;

// Change of the elastic and Rayleigh energy at a vertex whose rate moves from `from` by `move`.
// The Rayleigh change is d h (move^T K u + move^T K_ii move / 2) with a frozen element K.
// One element's share: the elastic change, the Rayleigh power factor and its damping coefficient.
struct VertexElementChangeTerms {
    float elastic;
    float rayleigh;
    float damping;
};

__device__ inline VertexElementChangeTerms VertexElementChange(
    const VertexBlockView& b, const math::Vec3* velocity, uint32_t env, uint32_t packed,
    math::Vec3 from, math::Vec3 move) {
    const uint32_t element_index = nk::VbdIncidenceElement(packed);
    const nk::VbdElement element = b.elements[element_index];
    const auto* start = b.membrane_start != nullptr
        ? b.membrane_start + env * b.layout.elements + element_index : nullptr;
    const uint32_t local = nk::VbdIncidenceLocal(packed);
    nk::vbd::ElementGeometry geometry = ElementStepGeometry(b, velocity, element, env);
    SetDelta(geometry, local, (from - velocity[b.Particle(env, element.vertex[0])]) * b.dt);
    VertexElementChangeTerms terms{
        nk::vbd::ElementEnergyChange(element, local, geometry, move * b.dt, start), 0.0f,
        element.damping};
    if (element.damping > 0.0f) {
        math::Vec3 damping_force;
        math::SymmetricMat3 damping_diagonal;
        VertexElementRayleigh(b, velocity, element, env, local, from,
                              damping_force, damping_diagonal, start);
        terms.rayleigh = move.Dot(damping_force + damping_diagonal.Multiply(move) * 0.5f);
    }
    return terms;
}

// A material-field element carries no Rayleigh term; its rate dependence lives in the response.
__device__ inline VertexElementChangeTerms FieldElementChange(
    const VertexBlockView& b, const math::Vec3* velocity, uint32_t env, uint32_t packed,
    math::Vec3 from, math::Vec3 move) {
    const uint32_t index = nk::MfIncidenceElement(packed);
    const nk::MfElement element = b.field_elements[index];
    const uint32_t local = nk::MfIncidenceLocal(packed);
    nk::mf::FieldGeometry geometry = FieldStepGeometry(b, velocity, element, env);
    SetDelta(geometry, local, (from - velocity[b.Particle(env, element.vertex[0])]) * b.dt);
    return {nk::mf::ElementEnergyChange(element, b.field_materials[element.material], FieldHistory(b, env, index),
                                        local, geometry, FieldStart(b, env, index), b.dt, move * b.dt),
            0.0f, 0.0f};
}

__device__ inline VertexElementChangeTerms ListElementChange(
    const VertexBlockView& b, const math::Vec3* velocity, uint32_t env, bool field, uint32_t packed,
    math::Vec3 from, math::Vec3 move) {
    return field ? FieldElementChange(b, velocity, env, packed, from, move)
                 : VertexElementChange(b, velocity, env, packed, from, move);
}

// `size` gathers the summed terms' magnitudes, which bound the sum's rounding.
__device__ inline void AddVertexElementChange(const VertexBlockView& b, const VertexElementChangeTerms& terms,
                                              float& change, float* size = nullptr) {
    change += terms.elastic;
    if (terms.damping > 0.0f) change += terms.damping * b.dt * terms.rayleigh;
    if (size != nullptr) *size += fabsf(terms.elastic) + fabsf(terms.damping * b.dt * terms.rayleigh);
}

__device__ inline float VertexEnergyChange(const VertexBlockView& b, const math::Vec3* velocity,
                                           uint32_t env, uint32_t vertex, math::Vec3 from,
                                           math::Vec3 move, uint32_t lane, uint32_t width,
                                           float* size = nullptr) {
    float change = 0.0f;
    const VertexElementList list = ElementsOf(b, vertex);
    for (uint32_t i = list.begin + lane; i < list.end; i += width)
        AddVertexElementChange(b, ListElementChange(b, velocity, env, list.field, list.incidence[i], from, move),
                               change, size);
    return change;
}

// A material-field element at a valid state keeps every triangle's area and a finite stored energy.
__device__ inline bool FieldElementStateValid(const VertexBlockView& b, const math::Vec3* velocity, uint32_t env,
                                              uint32_t packed, math::Vec3 from) {
    const uint32_t index = nk::MfIncidenceElement(packed);
    const nk::MfElement element = b.field_elements[index];
    nk::mf::FieldGeometry geometry = FieldStepGeometry(b, velocity, element, env);
    SetDelta(geometry, nk::MfIncidenceLocal(packed), (from - velocity[b.Particle(env, element.vertex[0])]) * b.dt);
    return nk::mf::ElementStateValid(element, geometry, FieldStart(b, env, index)) &&
           fabs(nk::mf::ElementEnergy(element, b.field_materials[element.material], FieldHistory(b, env, index),
                                      geometry)) <= FLT_MAX;
}

__device__ inline bool VertexEnergyStateValid(const VertexBlockView& b, const math::Vec3* velocity,
                                              uint32_t env, uint32_t vertex, math::Vec3 from,
                                              uint32_t lane, uint32_t width) {
    bool valid = true;
    const VertexElementList list = ElementsOf(b, vertex);
    for (uint32_t i = list.begin + lane; i < list.end; i += width) {
        const uint32_t packed = list.incidence[i];
        if (list.field) {
            valid &= FieldElementStateValid(b, velocity, env, packed, from);
            continue;
        }
        const uint32_t element_index = nk::VbdIncidenceElement(packed);
        const nk::VbdElement element = b.elements[element_index];
        const auto* start = b.membrane_start != nullptr
            ? b.membrane_start + env * b.layout.elements + element_index : nullptr;
        const uint32_t local = nk::VbdIncidenceLocal(packed);
        nk::vbd::ElementGeometry geometry = ElementStepGeometry(b, velocity, element, env);
        SetDelta(geometry, local, (from - velocity[b.Particle(env, element.vertex[0])]) * b.dt);
        valid &= fabsf(nk::vbd::ElementEnergy(element, geometry)) <= FLT_MAX;
        if (element.kind == nk::kVbdTriangle) {
            const auto strain = start != nullptr ? nk::vbd::MembraneStrain(element, geometry, *start)
                                                  : nk::vbd::MembraneStrain(element, geometry);
            valid &= strain.area_change > -1.0f;
        }
        if (element.damping > 0.0f) {
            nk::vbd::ElementGeometry frozen;
            math::Vec3 rates[4];
            const uint32_t count = nk::VbdElementVertexCount(element.kind);
            for (uint32_t j = 0u; j < count; ++j) {
                const uint32_t particle = b.Particle(env, element.vertex[j]);
                frozen.start[j] = b.start[particle];
                rates[j] = j == local ? from : velocity[particle];
            }
            float power = 0.0f;
            for (uint32_t j = 0u; j < count; ++j) {
                math::Vec3 applied;
                math::SymmetricMat3 diagonal;
                nk::vbd::ElementRayleighBlock(element, j, frozen, rates, &applied, &diagonal);
                power += rates[j].Dot(applied);
            }
            valid &= fabsf(0.5f * element.damping * b.dt * power) <= FLT_MAX;
        }
    }
    return valid;
}

__device__ inline void VertexBlockEquationWarp(const VertexBlockView& b, PointMassView points,
                                               const math::Vec3* particle_error, uint32_t env,
                                               uint32_t vertex, uint32_t lane, math::Vec3& u,
                                               math::Vec3& force, math::SymmetricMat3& hessian) {
    const uint32_t particle = b.Particle(env, vertex);
    math::Vec3 gradient;
    GatherVertexBlock(b, points.particle_velocity, env, vertex, lane, warpSize, gradient, hessian);
    // The sums reach every lane, so every lane forms the same step and search.
    gradient = {WarpSum(gradient.x), WarpSum(gradient.y), WarpSum(gradient.z)};
    hessian = {WarpSum(hessian.xx), WarpSum(hessian.yy), WarpSum(hessian.zz),
               WarpSum(hessian.xy), WarpSum(hessian.xz), WarpSum(hessian.yz)};
    const uint32_t slot = b.Slot(env, vertex);
    u = points.particle_velocity[particle];
    if (particle_error != nullptr) u = u - particle_error[particle];
    const math::Vec3 impulse = points.particle_row_impulse[particle];
    const float inertia = b.inertia[slot];
    const float h = b.dt;
    force = -nk::vbd::InertialGradient(u, b.free_rate[slot], inertia, h) - gradient + impulse / h;
    nk::vbd::AddIdentity(hessian, inertia);
}

// The vertex equation's residual after all neighboring row updates: the larger of the Newton correction and
// the momentum defect in m/s. The defect carries this warp sum's rounding bound, as the ledger sums in double.
__device__ inline float VertexBlockStationaryResidualWarp(const VertexBlockView& b, PointMassView points,
                                                          uint32_t env, uint32_t vertex, uint32_t lane) {
    const uint32_t particle = b.Particle(env, vertex);
    if (!(points.particle_inv_mass[particle] > 0.0f)) return 0.0f;
    const uint32_t slot = b.Slot(env, vertex);
    math::Vec3 gradient, magnitude{};
    math::SymmetricMat3 hessian, inverse;
    GatherVertexBlock(b, points.particle_velocity, env, vertex, lane, warpSize, gradient, hessian, &magnitude);
    gradient = {WarpSum(gradient.x), WarpSum(gradient.y), WarpSum(gradient.z)};
    magnitude = {WarpSum(magnitude.x), WarpSum(magnitude.y), WarpSum(magnitude.z)};
    hessian = {WarpSum(hessian.xx), WarpSum(hessian.yy), WarpSum(hessian.zz),
               WarpSum(hessian.xy), WarpSum(hessian.xz), WarpSum(hessian.yz)};
    const float inertia = b.inertia[slot];
    const math::Vec3 inertial =
        nk::vbd::InertialGradient(points.particle_velocity[particle], b.free_rate[slot], inertia, b.dt);
    const math::Vec3 impulse = points.particle_row_impulse[particle] / b.dt;
    const math::Vec3 force = -inertial - gradient + impulse;
    nk::vbd::AddIdentity(hessian, inertia);
    if (!(fabsf(force.x) <= FLT_MAX && fabsf(force.y) <= FLT_MAX && fabsf(force.z) <= FLT_MAX)) {
        if (lane == 0u) atomicOr(b.env_status + env, kEnvStatusSolverFailure);
        return FLT_MAX;
    }
    if (!nk::vbd::Invert(hessian, 1.0e-6f * inertia * inertia * inertia, &inverse)) return FLT_MAX;
    const math::Vec3 correction = inverse.Multiply(force) / b.dt;
    // Each term passes at most ceil(n/32) + 6 roundings here (lane sum, five shuffle levels, assembly); with the
    // ledger's own roundings its readout exceeds this force by at most (ceil(n/32) + 10) u per term magnitude.
    const VertexElementList list = ElementsOf(b, vertex);
    const uint32_t lane_terms = (list.end - list.begin + warpSize - 1u) / warpSize;
    const float rounding = 0.5f * static_cast<float>(lane_terms + 10u) * FLT_EPSILON;
    const math::Vec3 bound = (magnitude + math::Vec3{fabsf(inertial.x), fabsf(inertial.y), fabsf(inertial.z)} +
                              math::Vec3{fabsf(impulse.x), fabsf(impulse.y), fabsf(impulse.z)}) * rounding;
    const math::Vec3 momentum_error = nk::vbd::MomentumResidualVelocity(
        math::Vec3{fabsf(force.x), fabsf(force.y), fabsf(force.z)} + bound, points.particle_inv_mass[particle],
        b.effective_step[slot]);
    return fmaxf(fmaxf(fabsf(correction.x), fmaxf(fabsf(correction.y), fabsf(correction.z))),
                 fmaxf(fabsf(momentum_error.x), fmaxf(fabsf(momentum_error.y), fabsf(momentum_error.z))));
}

// Descend the incremental potential through local curvature, holding accumulated row impulses fixed.
// Convergence bounds both the Newton correction and the physical momentum-equation defect.
__device__ inline bool SolveVertexBlockWarp(const VertexBlockView& b, PointMassView points,
                                            math::Vec3* particle_error, uint32_t env,
                                            uint32_t vertex, uint32_t lane, float tolerance) {
    const uint32_t particle = b.Particle(env, vertex);
    if (!(points.particle_inv_mass[particle] > 0.0f)) return false;
    math::Vec3 u, force;
    math::SymmetricMat3 hessian;
    VertexBlockEquationWarp(b, points, particle_error, env, vertex, lane, u, force, hessian);
    const uint32_t slot = b.Slot(env, vertex);
    const math::Vec3 impulse = points.particle_row_impulse[particle];
    const float inertia = b.inertia[slot], h = b.dt;
    const math::Vec3 free_rate = b.free_rate[slot];
    if (!(fabsf(force.x) <= FLT_MAX && fabsf(force.y) <= FLT_MAX && fabsf(force.z) <= FLT_MAX)) {
        if (lane == 0u) atomicOr(b.env_status + env, kEnvStatusSolverFailure);
        return true;
    }
    math::SymmetricMat3 inverse;
    const bool invertible = nk::vbd::Invert(hessian, 1.0e-6f * inertia * inertia * inertia, &inverse);
    const math::Vec3 correction = invertible ? inverse.Multiply(force) / h : math::Vec3{};
    const math::Vec3 momentum_error = nk::vbd::MomentumResidualVelocity(
        force, points.particle_inv_mass[particle], b.effective_step[slot]);
    // Row-coupled blocks add the least share of the frozen row metric that makes the search metric
    // dominate this block and the row response; gradient and stationarity correction are unchanged.
    math::SymmetricMat3 search_inverse = inverse;
    bool search_invertible = invertible;
    // Only rows that have pushed this block in the step couple it; an idle speculative row exchanges nothing.
    const bool rows = b.row_first != nullptr && b.row_first[particle] != ~0u &&
                      (impulse.x != 0.0f || impulse.y != 0.0f || impulse.z != 0.0f);
    if (rows) {
        const math::SymmetricMat3 response = points.particle_response[particle];
        math::SymmetricMat3 frozen_metric;
        const bool frozen_invertible = nk::vbd::Invert(response, 0.0f, &frozen_metric);
        if (frozen_invertible) {
            const double ratio =
                nk::vbd::SmallestRelativeEigenvalue(hessian, nk::vbd::Scaled(response, h * h));
            const float share = float(fmin(fmax(1.0 - ratio, 0.0), 1.0));
            nk::vbd::AddTo(hessian, nk::vbd::Scaled(frozen_metric, share / (h * h)));
        }
        search_invertible = frozen_invertible && nk::vbd::Invert(hessian, 0.0f, &search_inverse);
    }
    if (!search_invertible) {
        if (lane == 0u) atomicOr(b.env_status + env, kEnvStatusSolverFailure);
        return true;
    }
    const math::Vec3 step = search_invertible ? search_inverse.Multiply(force) / h : math::Vec3{};
    const float slope = -h * force.Dot(step);
    math::Vec3 next = u;
    float scale = rows ? 1.0f : kVertexStepRelaxation;
    float last_change = 0.0f;
    uint32_t last_halving = 0u;
    bool accepted = false;
    for (uint32_t halving = 0u; halving <= kVertexStepHalvings && slope < 0.0f; ++halving) {
        const math::Vec3 candidate{nk::vbd::TrialValue(u.x, step.x, scale),
                                  nk::vbd::TrialValue(u.y, step.y, scale),
                                  nk::vbd::TrialValue(u.z, step.z, scale)};
        const math::Vec3 move = candidate - u;
        const double trial_slope = -double(h) * (double(force.x) * move.x +
            double(force.y) * move.y + double(force.z) * move.z);
        const float elastic = WarpSum(VertexEnergyChange(b, points.particle_velocity, env, vertex,
                                                         u, move, lane, warpSize));
        const float change = static_cast<float>(double(elastic) - impulse.Dot(move) +
            nk::vbd::InertialEnergyChange(u, free_rate, move, inertia, h));
        last_change = change;
        last_halving = halving;
        if (trial_slope <= 0.0 && double(change) <= double(kVertexStepDecrease) * trial_slope) {
            next = candidate;
            accepted = true;
            break;
        }
        scale *= 0.5f;
    }
    // Every lane has read the vertex state before lane 0 overwrites it.
    __syncwarp();
    if (lane == 0u) {
        RecordVertexDescentAudit(b, env, vertex, force, step, u, next, scale, last_change,
                                 last_halving, accepted, slope < 0.0f);
        points.particle_velocity[particle] = next;
        if (particle_error != nullptr) particle_error[particle] = {};
    }
    if (!invertible || !search_invertible) return force.x != 0.0f || force.y != 0.0f || force.z != 0.0f;
    return tolerance > 0.0f
        ? !(fabsf(correction.x) <= tolerance && fabsf(correction.y) <= tolerance &&
            fabsf(correction.z) <= tolerance && fabsf(momentum_error.x) <= tolerance &&
            fabsf(momentum_error.y) <= tolerance && fabsf(momentum_error.z) <= tolerance)
        : (force.x != 0.0f || force.y != 0.0f || force.z != 0.0f);
}

}  // namespace nuka::phi::nkops
