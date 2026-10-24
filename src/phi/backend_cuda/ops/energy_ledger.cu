#include <cuda_runtime.h>
#include "phi/backend_cuda/ops/contact_row_layout.cuh"

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <limits>

#include "nk/readout/energy_ledger.hpp"
#include "nk/readout/physics_diagnostics.hpp"
#include "nk/solve/vertex_block.hpp"
#include "collision/ogc_geometry.hpp"
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/articulation_types.cuh"
#include "phi/backend_cuda/ops/dat_motion.cuh"
#include "phi/backend_cuda/ops/kinematics.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/registry.cuh"
#include "phi/backend_cuda/ops/surface_query.cuh"
#include "phi/backend_cuda/ops/union_types.cuh"
#include "phi/backend_cuda/ops/vertex_blocks.cuh"

namespace nuka::phi {
namespace {

using namespace nkops;
using nk::EnergyColumn;
using nk::EnergyStage;
using math::Vec3;
namespace mg = math::gpu;
constexpr uint32_t kThreads = 128u;
constexpr uint32_t kColumns = nk::kEnergyColumnCount;
constexpr uint32_t kForceWorkColumn = kColumns;
constexpr uint32_t kReductionColumns = kColumns + 1u;
static_assert(kColumns == 36u, "energy field record must match its named columns");
// A partial row holds one block's column totals, wide enough for the stage metrics.
constexpr uint32_t kPartialWidth = nk::kPhysicsStageColumnCount;
static_assert(kReductionColumns <= kPartialWidth);

__device__ uint32_t Column(EnergyColumn value) { return static_cast<uint32_t>(value); }
__device__ double Dot(Vec3 a, Vec3 b) {
    return double(a.x) * b.x + double(a.y) * b.y + double(a.z) * b.z;
}
__device__ Vec3 InverseRotate(math::Quat q, Vec3 value) {
    return mg::RotateByQuatNormalized(mg::MakeQuat(q.w, -q.x, -q.y, -q.z), value);
}
__device__ bool IsVertex(const ReadoutEnergyLedgerParams& p, uint32_t particle) {
    const uint32_t local = particle % p.particles_per_env;
    return local >= p.vbd_begin && local - p.vbd_begin < p.vbd_vertices;
}
__device__ float ParticleStep(const DataView& d, const ReadoutEnergyLedgerParams& p,
                             uint32_t particle) {
    const uint32_t env = particle / p.particles_per_env;
    return IsVertex(p, particle) ? d.vbd_step[env * p.vbd_vertices +
        particle % p.particles_per_env - p.vbd_begin] : p.dt;
}
__device__ Vec3 ParticleVelocity(const DataView& d, const ReadoutEnergyLedgerParams& p,
                               uint32_t particle) {
    if (!IsVertex(p, particle) || !(d.particle_inv_mass[particle] > 0.0f) ||
        p.stage == EnergyStage::Begin || p.stage == EnergyStage::End)
        return d.particle_vel[particle];
    const uint32_t slot = (particle / p.particles_per_env) * p.vbd_vertices +
        particle % p.particles_per_env - p.vbd_begin;
    return nk::vbd::PhysicalVelocity(d.particle_vel[particle], d.vbd_free_rate[slot],
                                    d.vbd_free_velocity[slot], p.dt, d.vbd_step[slot]);
}

// Skip cleared contact rows while retaining each thread's row order and noncontact assignments.
__device__ uint32_t LedgerRow(uint32_t row, uint32_t bound, uint32_t contacts, uint32_t stride) {
    if (row < contacts && row >= bound)
        row += ((contacts - row - 1u) / stride + 1u) * stride;
    return row;
}

// A readout block's share of its env: the env's blocks stride its item lists together.
struct ReadoutSlice {
    uint32_t env;
    uint32_t first;
    uint32_t stride;
};

__device__ ReadoutSlice Slice(uint32_t env_count) {
    const uint32_t parts = gridDim.x / env_count, part = blockIdx.x % parts;
    return {blockIdx.x / parts, part * blockDim.x + threadIdx.x, parts * blockDim.x};
}

struct SumColumns {
    __device__ double operator()(uint32_t, double a, double b) const { return a + b; }
};

// Stage metrics combine by column kind; a nonfinite operand poisons the column.
struct CombineStageMetric {
    __device__ double operator()(uint32_t c, double a, double b) const {
        using C = nk::PhysicsStageColumn;
        if (!isfinite(a) || !isfinite(b)) return double(nanf(""));
        if (c == static_cast<uint32_t>(C::MinVbdEffectiveDt)) return fmin(a, b);
        if (c == static_cast<uint32_t>(C::MaxVbdMomentumVelocityError) || c == static_cast<uint32_t>(C::MaxVbdForce) ||
            c == static_cast<uint32_t>(C::MaxVbdEffectiveDt) || c == static_cast<uint32_t>(C::MaxPassiveRowEnergyGain) ||
            c == static_cast<uint32_t>(C::MaxAbsoluteVertexResidualWork)) return fmax(a, b);
        return a + b;
    }
};

// Stores the block's column totals in its partial row: warp shuffles, then one thread per column over the warps.
template <uint32_t columns, class Combine>
__device__ void StorePartial(const double (&values)[columns], double* partial, Combine combine) {
    __shared__ double warps[kThreads / 32u][columns];
    const uint32_t lane = threadIdx.x % 32u, warp = threadIdx.x / 32u;
    for (uint32_t c = 0u; c < columns; ++c) {
        double v = values[c];
        for (uint32_t offset = 16u; offset > 0u; offset /= 2u)
            v = combine(c, v, __shfl_down_sync(0xffffffffu, v, offset));
        if (lane == 0u) warps[warp][c] = v;
    }
    __syncthreads();
    for (uint32_t c = threadIdx.x; c < columns; c += blockDim.x) {
        double v = warps[0][c];
        for (uint32_t w = 1u; w < kThreads / 32u; ++w) v = combine(c, v, warps[w][c]);
        partial[c] = v;
    }
}

// One column of an env's partial rows, combined in part order.
template <class Combine>
__device__ double CombineParts(const double* partials, uint32_t parts, uint32_t c, Combine combine) {
    double v = partials[c];
    for (uint32_t part = 1u; part < parts; ++part) v = combine(c, v, partials[size_t{part} * kPartialWidth + c]);
    return v;
}

// Read-only FK uses private poses and world velocities; dynamics scratch stays untouched.
__global__ void EnergyLinkFramesKernel(ModelView m, DataView d, ReadoutEnergyLedgerParams p) {
    const uint32_t art = blockIdx.x;
    if (art >= p.env_count * p.artics_per_env || threadIdx.x != 0u) return;
    const uint32_t offset = m.articulation_link_offset[art];
    const uint32_t count = m.articulation_link_count[art];
    for (uint32_t local = 0u; local < count; ++local) {
        const uint32_t link = offset + local;
        const uint32_t parent = m.parent_link[link];
        const auto type = static_cast<ArticulationJointType>(m.joint_type[link]);
        math::Transform pose{}, unprojected{};
        Vec3 angular{}, linear{};
        if (parent == ~0u && type == ArticulationJointType::FloatingBase) {
            pose = unprojected = d.base_pose[art];
            const auto velocity = d.link_velocity[link];
            angular = mg::RotateByQuatNormalized(pose.rotation,
                {velocity.v[0], velocity.v[1], velocity.v[2]});
            linear = mg::RotateByQuatNormalized(pose.rotation,
                {velocity.v[3], velocity.v[4], velocity.v[5]});
        } else {
            const float coordinate = d.q[link];
            const float no_position = coordinate -
                (p.stage == EnergyStage::Projected && p.pos_pass ? p.dt * d.qdot_pseudo[link] : 0.0f);
            pose = JointRelativeFrame(type, m.joint_axis[link], m.link_local_pose[link],
                                       m.parent_offset[link], coordinate);
            unprojected = JointRelativeFrame(type, m.joint_axis[link], m.link_local_pose[link],
                                              m.parent_offset[link], no_position);
            if (parent != ~0u) {
                const uint32_t parent_link = offset + parent;
                const auto pv = d.energy_link_velocity[parent_link];
                angular = {pv.v[0], pv.v[1], pv.v[2]};
                pose = ComposeFrame(d.energy_link_pose[parent_link], pose);
                unprojected = ComposeFrame(d.energy_link_unprojected_pose[parent_link], unprojected);
                linear = Vec3{pv.v[3], pv.v[4], pv.v[5]} +
                    angular.Cross(pose.position - d.energy_link_pose[parent_link].position);
            }
            const Vec3 axis = mg::RotateByQuatNormalized(pose.rotation, JointUnitAxis(m.joint_axis[link]));
            if (type == ArticulationJointType::Revolute) angular += axis * d.qdot[link];
            if (type == ArticulationJointType::Prismatic) linear += axis * d.qdot[link];
        }
        d.energy_link_pose[link] = pose;
        d.energy_link_unprojected_pose[link] = unprojected;
        d.energy_link_velocity[link] = {{angular.x, angular.y, angular.z, linear.x, linear.y, linear.z}};
    }
}

struct RowPower {
    double physical = 0.0;
    double rate = 0.0;
    double applied = 0.0;
    double kinematic = 0.0;
};

// Row power at the interval mean velocity, so the gravity integrated before the solve does no row work.
__device__ RowPower SidePower(const NkRowSide& side, bool second, uint32_t row,
    const ModelView&, const DataView& d, const ReadoutEnergyLedgerParams& p) {
    RowPower out;
    if (side.kind == nk::kNkSideArtic) {
        const float* jacobian = (second ? d.chain_jacobian_b : d.chain_jacobian) +
                                size_t{row} * p.dofs_per_artic;
        const size_t tile = size_t{side.index} * p.dofs_per_artic;
        for (uint32_t k = 0u; k < p.dofs_per_artic; ++k)
            out.physical += double(jacobian[k]) * 0.5 * (d.step_qdot_flat[tile + k] + d.qdot_flat[tile + k]);
        out.rate = out.applied = out.physical;
    } else if (side.kind == nk::kNkSideRigid) {
        const uint32_t body = side.index;
        out.physical = 0.5 * (Dot(side.jlin, d.step_body_linear_velocity[body] + d.body_linear_velocity[body]) +
            Dot(side.jang, d.step_body_angular_velocity[body] + d.body_angular_velocity[body]));
        out.rate = out.applied = out.physical;
    } else if (PointMassView::IsPointSide(side.kind)) {
        PointMassView points;
        points.ranges = d.point_endpoint_ranges;
        points.terms = d.point_endpoint_terms;
        for (uint32_t k = 0u; k < points.Count(side); ++k) {
            const auto term = points.At(side, k);
            if (term.kind != nk::kNkSideParticle) continue;
            const uint32_t particle = term.index;
            const double physical = 0.5 * Dot(term.jacobian,
                d.step_particle_velocity[particle] + ParticleVelocity(d, p, particle));
            out.physical += physical;
            out.rate += 0.5 * Dot(term.jacobian,
                d.step_particle_velocity[particle] + d.particle_vel[particle]);
            const bool dynamic = d.particle_inv_mass[particle] > 0.0f;
            out.applied += physical * (dynamic ? ParticleStep(d, p, particle) / p.dt : 1.0f);
            if (!dynamic) out.kinematic += physical;
        }
    }
    return out;
}

// Row rate at the solved displacement; contact and friction laws keep it nonpositive along the impulse.
__device__ double SideDisplacementRate(const NkRowSide& side, bool second, uint32_t row,
    const DataView& d, const ReadoutEnergyLedgerParams& p) {
    double rate = 0.0;
    if (side.kind == nk::kNkSideArtic) {
        const float* jacobian = (second ? d.chain_jacobian_b : d.chain_jacobian) +
                                size_t{row} * p.dofs_per_artic;
        const size_t tile = size_t{side.index} * p.dofs_per_artic;
        for (uint32_t k = 0u; k < p.dofs_per_artic; ++k) rate += double(jacobian[k]) * d.qdot_flat[tile + k];
    } else if (side.kind == nk::kNkSideRigid) {
        rate = Dot(side.jlin, d.body_linear_velocity[side.index]) +
               Dot(side.jang, d.body_angular_velocity[side.index]);
    } else if (PointMassView::IsPointSide(side.kind)) {
        PointMassView points;
        points.ranges = d.point_endpoint_ranges;
        points.terms = d.point_endpoint_terms;
        for (uint32_t k = 0u; k < points.Count(side); ++k) {
            const auto term = points.At(side, k);
            if (term.kind == nk::kNkSideParticle) rate += Dot(term.jacobian, d.particle_vel[term.index]);
        }
    }
    return rate;
}

__device__ Vec3 ElementPosition(const DataView& d, const ReadoutEnergyLedgerParams& p,
    uint32_t particle, bool no_position) {
    if (p.stage == EnergyStage::Solved)
        return d.particle_prev_pos[particle] + d.particle_vel[particle] * p.dt;
    return d.particle_pos[particle] -
        (no_position && p.pos_pass ? d.particle_pseudo_vel[particle] * p.dt : Vec3{});
}

// Contact reactions on prescribed particles are accumulated independently of solver impulses.
__global__ void EnergyKinematicImpulseKernel(DataView d, ReadoutEnergyLedgerParams p) {
    const size_t index = size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (index >= size_t{p.env_count} * p.rows_per_env) return;
    const auto row = reinterpret_cast<const NkRow*>(d.urows)[index];
    if (!(row.flags & nk::nk_row_flags::kActive) || d.lambda[index] == 0.0f) return;
    PointMassView points;
    points.ranges = d.point_endpoint_ranges;
    points.terms = d.point_endpoint_terms;
    const NkRowSide sides[2] = {row.a, row.b};
    for (const auto& side : sides) {
        if (!PointMassView::IsPointSide(side.kind)) continue;
        for (uint32_t k = 0u; k < points.Count(side); ++k) {
            const auto term = points.At(side, k);
            if (term.kind != nk::kNkSideParticle || d.particle_inv_mass[term.index] > 0.0f) continue;
            const Vec3 impulse = term.jacobian * d.lambda[index];
            Vec3* output = d.energy_kinematic_contact_impulse + term.index;
            atomicAdd(&output->x, impulse.x);
            atomicAdd(&output->y, impulse.y);
            atomicAdd(&output->z, impulse.z);
        }
    }
}

struct KinematicElasticResponse { Vec3 gradient{}; double work = 0.0; };

__device__ VertexBlockView LedgerVertices(const ModelView& m, const DataView& d, const ReadoutEnergyLedgerParams& p) {
    VertexBlockView block;
    block.elements = m.vbd_elements;
    block.membrane_start = d.vbd_membrane_start;
    block.offsets = m.vbd_incidence_offsets;
    block.incidence = m.vbd_incidence;
    block.start = d.particle_prev_pos;
    block.layout.begin = p.vbd_begin;
    block.layout.vertices = p.vbd_vertices;
    block.layout.elements = p.vbd_elements;
    block.layout.field_vertices = p.field_vertices;
    block.layout.field_elements = p.field_elements;
    block.layout.field_state_scalars = p.field_state_scalars;
    block.layout.field_trial_scalars = p.field_trial_scalars;
    block.layout.spare_first = p.spare_first;
    block.layout.spares = p.spares;
    block.layout.particles_per_env = p.particles_per_env;
    block.layout.env_count = p.env_count;
    block.dt = p.dt;
    BindFieldTables(block, m, d);
    return block;
}

// A field element on its env's split corners.
__device__ nk::MfElement LedgerFieldElement(const ModelView& m, const DataView& d, const ReadoutEnergyLedgerParams& p,
                                            uint32_t env, uint32_t index) {
    return nk::WithCorners(m.mf_elements[index],
                           d.mf_corner != nullptr ? d.mf_corner + (size_t{env} * p.field_elements + index) * nk::kMfMaxVertices : nullptr);
}

// Material-field elements push a kinematic vertex through their response at the committed history.
__device__ KinematicElasticResponse KinematicFieldReaction(const ModelView& m, const DataView& d,
    const ReadoutEnergyLedgerParams& p, uint32_t particle, uint32_t field_vertex) {
    const uint32_t base = (particle / p.particles_per_env) * p.particles_per_env + p.vbd_begin;
    KinematicElasticResponse total;
    for (uint32_t k = m.mf_incidence_offsets[field_vertex]; k < m.mf_incidence_offsets[field_vertex + 1u]; ++k) {
        const uint32_t incidence = m.mf_incidence[k];
        const uint32_t index = nk::MfIncidenceElement(incidence);
        const nk::MfElement element = LedgerFieldElement(m, d, p, particle / p.particles_per_env, index);
        if (!nk::MfVertexSlotHolds(element.vertex, nk::MfIncidenceLocal(incidence), particle - base)) continue;
        const float* history = d.mf_history + size_t{particle / p.particles_per_env} * p.field_state_scalars +
                               element.state_offset;
        const uint32_t anchor = base + element.vertex[0];
        nk::mf::FieldGeometry geometry;
        for (uint32_t j = 0u; j < nk::MfElementVertexCount(element); ++j) {
            const uint32_t point = base + element.vertex[j];
            geometry.start[j] = d.particle_prev_pos[point];
            geometry.delta[j] = (d.particle_vel[point] - d.particle_vel[anchor]) * p.dt;
        }
        Vec3 gradient;
        math::SymmetricMat3 hessian;
        nk::mf::ElementBlock(element, m.mf_materials[element.material], history, nk::MfIncidenceLocal(incidence),
                             geometry, nullptr, p.dt, &gradient, &hessian);
        total.gradient += gradient;
        total.work += Dot(gradient, d.particle_kinematic_target[particle] - d.energy_particle_start[particle]);
    }
    return total;
}

__device__ KinematicElasticResponse KinematicElasticReaction(const ModelView& m, const DataView& d,
    const ReadoutEnergyLedgerParams& p, uint32_t particle, bool rayleigh = true) {
    if (!IsVertex(p, particle)) return {};
    const uint32_t vertex = particle % p.particles_per_env - p.vbd_begin;
    const uint32_t first_field = p.vbd_vertices - p.field_vertices;
    if (vertex >= first_field) return KinematicFieldReaction(m, d, p, particle, vertex - first_field);
    const uint32_t base = (particle / p.particles_per_env) * p.particles_per_env + p.vbd_begin;
    KinematicElasticResponse total;
    for (uint32_t k = m.vbd_incidence_offsets[vertex]; k < m.vbd_incidence_offsets[vertex + 1u]; ++k) {
        const uint32_t incidence = m.vbd_incidence[k];
        const auto element = m.vbd_elements[nk::VbdIncidenceElement(incidence)];
        const uint32_t anchor = base + element.vertex[0];
        nk::vbd::ElementGeometry geometry;
        for (uint32_t j = 0u; j < nk::VbdElementVertexCount(element.kind); ++j) {
            const uint32_t point = base + element.vertex[j];
            geometry.start[j] = d.particle_prev_pos[point];
            geometry.delta[j] = (d.particle_vel[point] - d.particle_vel[anchor]) * p.dt;
        }
        Vec3 gradient;
        math::SymmetricMat3 hessian;
        const uint32_t local = nk::VbdIncidenceLocal(incidence);
        nk::vbd::ElementBlock(element, local, geometry, &gradient, &hessian);
        if (rayleigh && element.damping > 0.0f) {
            nk::vbd::ElementGeometry frozen;
            Vec3 rates[4];
            for (uint32_t j = 0u; j < nk::VbdElementVertexCount(element.kind); ++j) {
                const uint32_t point = base + element.vertex[j];
                frozen.start[j] = d.particle_prev_pos[point];
                rates[j] = d.particle_vel[point];
            }
            Vec3 applied;
            nk::vbd::ElementRayleighBlock(element, local, frozen, rates, &applied, &hessian);
            gradient += applied * element.damping;
        }
        total.gradient += gradient;
        total.work += Dot(gradient, d.particle_kinematic_target[particle] - d.energy_particle_start[particle]);
    }
    return total;
}

__device__ double KinematicElasticWork(const ModelView& m, const DataView& d,
    const ReadoutEnergyLedgerParams& p, uint32_t particle) {
    return KinematicElasticReaction(m, d, p, particle).work;
}

// Opens a substep's ledger before its blocks report coverage.
__global__ void BeginEnergyLedgerKernel(DataView d, ReadoutEnergyLedgerParams p) {
    const size_t record = size_t{blockIdx.x} * p.substeps + p.slot;
    if (threadIdx.x == 0u) {
        d.energy_ledger_status[record] = p.coverage_status;
        d.vbd_force_residual_work[record] = 0.0f;
    }
    for (uint32_t c = threadIdx.x; c < kColumns; c += blockDim.x) {
        d.energy_ledger[record * kColumns + c] = 0.0f;
        d.energy_ledger_sum[record * kColumns + c] = 0.0;
    }
}

// Each block sums its share of one env into its partial row.
__global__ void EnergyLedgerKernel(ModelView m, DataView d, ReadoutEnergyLedgerParams p) {
    const ReadoutSlice s = Slice(p.env_count);
    const uint32_t env = s.env;
    const size_t record = size_t{env} * p.substeps + p.slot;
    uint32_t* coverage = d.energy_ledger_status + record;
    double values[kReductionColumns]{};
    const auto add = [&](EnergyColumn column, double value) {
        const uint32_t c = Column(column);
        values[c] += value;
        if (c >= Column(EnergyColumn::DriveWork) && c <= Column(EnergyColumn::ExternalWork))
            values[Column(EnergyColumn::Throughput)] += fabs(value);
        if ((c >= Column(EnergyColumn::FrictionLoss) && c <= Column(EnergyColumn::RayleighLoss)) ||
            (c >= Column(EnergyColumn::PlasticLoss) && c <= Column(EnergyColumn::NumericalPotential)))
            values[Column(EnergyColumn::Throughput)] += value;
    };
    double kinetic = 0.0, gravity = 0.0, elastic = 0.0, position = 0.0;
    const Vec3 g{p.gravity[0], p.gravity[1], p.gravity[2]};
    for (uint32_t local = s.first; local < p.particles_per_env; local += s.stride) {
        const uint32_t particle = env * p.particles_per_env + local;
        if (p.stage == EnergyStage::Begin) d.energy_particle_start[particle] = d.particle_pos[particle];
        if (p.stage == EnergyStage::Begin) d.energy_particle_start_velocity[particle] = d.particle_vel[particle];
        const Vec3 velocity = ParticleVelocity(d, p, particle);
        if (p.stage == EnergyStage::Free) {
            d.energy_kinematic_contact_impulse[particle] = {};
            Vec3 impulse{};
            if (p.has_aero && d.particle_inv_mass[particle] > 0.0f)
                for (uint32_t k = 0u; k < m.aero_particle_count[particle]; ++k)
                    impulse += d.aero_tri_impulse[m.aero_incident_tri[m.aero_particle_offset[particle] + k]];
            d.energy_aero_impulse[particle] = impulse;
        }
        const float inverse_mass = d.particle_inv_mass[particle];
        if (IsVertex(p, particle)) {
            const double stiffness = d.particle_attachment_stiffness[particle];
            const double previous = d.particle_attachment_previous_stiffness[particle];
            const Vec3 target = d.particle_kinematic_target[particle];
            const Vec3 old_target = d.particle_attachment_previous_target[particle];
            const Vec3 x = ElementPosition(d, p, particle, false);
            elastic += p.stage == EnergyStage::Begin
                ? 0.5 * previous * Dot(x - old_target, x - old_target)
                : 0.5 * stiffness * Dot(x - target, x - target);
            if (p.stage == EnergyStage::Free) {
                const Vec3 start = d.energy_particle_start[particle];
                add(EnergyColumn::ExternalWork, 0.5 * stiffness * Dot(start - target, start - target) -
                    0.5 * previous * Dot(start - old_target, start - old_target));
            }
            if (p.stage == EnergyStage::Projected) {
                const Vec3 unprojected = ElementPosition(d, p, particle, true);
                position += 0.5 * stiffness * (Dot(x - target, x - target) -
                                               Dot(unprojected - target, unprojected - target));
            }
            // Backward Euler loses the end gradient's work over the move less the potential change; the
            // elements subtract their change below, and the attachment's own gap is 0.5 k |dx|^2.
            if (p.stage == EnergyStage::Solved) {
                const Vec3 move = d.particle_vel[particle] * p.dt;
                add(EnergyColumn::NumericalPotential,
                    double(Dot(KinematicElasticReaction(m, d, p, particle, false).gradient, move)) +
                    0.5 * stiffness * double(Dot(move, move)));
            }
        }
        if (!(inverse_mass > 0.0f)) {
            if (p.stage == EnergyStage::Solved) {
                const double elastic_work = KinematicElasticWork(m, d, p, particle);
                const double contact_work = -0.5 * Dot(d.energy_kinematic_contact_impulse[particle],
                    d.step_particle_velocity[particle] + velocity);
                add(EnergyColumn::KinematicElasticWork, elastic_work);
                add(EnergyColumn::KinematicContactWork, contact_work);
                add(EnergyColumn::Throughput, fabs(elastic_work + contact_work));
            }
            continue;
        }
        const double mass = 1.0 / inverse_mass;
        if (p.stage == EnergyStage::Solved && IsVertex(p, particle)) {
            const uint32_t vertex = local - p.vbd_begin;
            const uint32_t slot = env * p.vbd_vertices + vertex;
            const VertexBlockView block = LedgerVertices(m, d, p);
            const Vec3 displacement = d.particle_vel[particle] * p.dt;
            const Vec3 force = VertexBlockForceDouble(block, d.particle_vel, env, vertex,
                nk::vbd::InertialGradient(d.particle_vel[particle], d.vbd_solve_rate[slot], d.vbd_inertia[slot], p.dt),
                d.particle_row_impulse[particle] / p.dt);
            values[kForceWorkColumn] -= Dot(force, displacement);
            const Vec3 jump = d.particle_vel[particle] - d.energy_particle_start_velocity[particle];
            add(EnergyColumn::NumericalKinetic, 0.5 * mass * double(Dot(jump, jump)));
        }
        kinetic += 0.5 * mass * Dot(velocity, velocity);
        gravity -= mass * Dot(g, d.particle_pos[particle]);
        if (p.stage == EnergyStage::Projected && p.pos_pass)
            position -= mass * Dot(g, d.particle_pseudo_vel[particle]) * p.dt;
        if (p.stage == EnergyStage::End && p.has_aero)
            add(EnergyColumn::AeroWork, Dot(d.energy_aero_impulse[particle],
                d.particle_pos[particle] - d.energy_particle_start[particle]) / p.dt);
    }
    for (uint32_t local = s.first; local < p.bodies_per_env; local += s.stride) {
        const uint32_t body = env * p.bodies_per_env + local;
        if (m.body_to_link && m.body_to_link[body] != ~0u) continue;
        if (p.stage == EnergyStage::Begin &&
            (d.body_force[body].LengthSq() != 0.0f || d.body_torque[body].LengthSq() != 0.0f))
            atomicOr(coverage, nk::energy_status::kExternalBodyLoad);
        const float inverse_mass = d.body_inv_mass[body];
        if (!(inverse_mass > 0.0f)) continue;
        const auto principal = ComposeFrame(d.body_pose[body], d.body_inertial_frame[body]);
        const Vec3 angular = InverseRotate(principal.rotation, d.body_angular_velocity[body]);
        const Vec3 inverse_inertia = d.body_inv_inertia[body];
        const double mass = 1.0 / inverse_mass;
        kinetic += 0.5 * mass * Dot(d.body_linear_velocity[body], d.body_linear_velocity[body]);
        if (inverse_inertia.x > 0.0f) kinetic += 0.5 * double(angular.x) * angular.x / inverse_inertia.x;
        if (inverse_inertia.y > 0.0f) kinetic += 0.5 * double(angular.y) * angular.y / inverse_inertia.y;
        if (inverse_inertia.z > 0.0f) kinetic += 0.5 * double(angular.z) * angular.z / inverse_inertia.z;
        gravity -= mass * Dot(g, principal.position);
        if (p.stage == EnergyStage::Projected && p.pos_pass) {
            position -= mass * Dot(g, d.body_pseudo_linear_velocity[body]) * p.dt;
            if (d.body_pseudo_angular_velocity[body].LengthSq() != 0.0f)
                atomicOr(coverage, nk::energy_status::kRigidAngularPositionWork);
        }
    }
    for (uint32_t local = s.first; local < p.links_per_env; local += s.stride) {
        const uint32_t link = env * p.links_per_env + local;
        if (p.stage == EnergyStage::Begin) {
            d.energy_link_start_q[link] = d.q[link];
            d.energy_link_start_qdot[link] = d.qdot[link];
            bool external = false;
            for (uint32_t k = 0u; k < 6u; ++k) external |= d.link_wrench[link].v[k] != 0.0f;
            if (external) atomicOr(coverage, nk::energy_status::kExternalBodyLoad);
        }
        const auto pose = d.energy_link_pose[link];
        const auto world = d.energy_link_velocity[link];
        const Vec3 angular = InverseRotate(pose.rotation, {world.v[0], world.v[1], world.v[2]});
        const Vec3 linear = InverseRotate(pose.rotation, {world.v[3], world.v[4], world.v[5]});
        const float velocity[6] = {angular.x, angular.y, angular.z, linear.x, linear.y, linear.z};
        const auto inertia = m.link_inertia[link];
        for (uint32_t r = 0u; r < 6u; ++r)
            for (uint32_t c = 0u; c < 6u; ++c)
                kinetic += 0.5 * double(velocity[r]) * inertia.m[r * 6u + c] * velocity[c];
        const double mass = inertia.m[21];
        const auto com = ComposeFrame(pose, m.link_inertial_frame[link]);
        gravity -= mass * Dot(g, com.position);
        // Armature is rotor inertia on a one-coordinate joint's mass diagonal; a floating base keeps it unbalanced.
        const auto joint = static_cast<ArticulationJointType>(m.joint_type[link]);
        if (joint == ArticulationJointType::Revolute || joint == ArticulationJointType::Prismatic)
            kinetic += 0.5 * double(m.joint_armature[link]) * d.qdot[link] * d.qdot[link];
        else if (joint == ArticulationJointType::FloatingBase && m.joint_armature[link] != 0.0f)
            atomicOr(coverage, nk::energy_status::kArmatureEnergy);
        if (p.stage == EnergyStage::Projected && p.pos_pass) {
            const auto no_position = ComposeFrame(d.energy_link_unprojected_pose[link], m.link_inertial_frame[link]);
            position -= mass * Dot(g, com.position - no_position.position);
            if (static_cast<ArticulationJointType>(m.joint_type[link]) == ArticulationJointType::FloatingBase)
                for (uint32_t k = 0u; k < 6u; ++k)
                    if (d.link_velocity_pseudo[link].v[k] != 0.0f)
                        atomicOr(coverage, nk::energy_status::kFloatingPositionWork);
        }
        if (p.stage == EnergyStage::End) {
            const double displacement = double(d.q[link]) - d.energy_link_start_q[link];
            const double effort = p.drive_rows ? d.actuator_effort[link] : 0.0;
            add(EnergyColumn::DriveWork, effort * displacement);
            add(EnergyColumn::ExternalWork, (double(d.tau[link]) - effort) * displacement);
            add(EnergyColumn::PassiveLoss, double(m.joint_damping[link]) *
                d.energy_link_start_qdot[link] * displacement);
        }
    }
    for (uint32_t i = s.first; i < p.vbd_elements; i += s.stride) {
        const auto element = m.vbd_elements[i];
        const uint32_t count = nk::VbdElementVertexCount(element.kind);
        nk::vbd::ElementGeometry x, unprojected, frozen;
        Vec3 rates[4];
        const uint32_t anchor = env * p.particles_per_env + p.vbd_begin + element.vertex[0];
        for (uint32_t j = 0u; j < count; ++j) {
            const uint32_t particle = env * p.particles_per_env + p.vbd_begin + element.vertex[j];
            if (p.stage == EnergyStage::Solved) {
                x.start[j] = d.particle_prev_pos[particle];
                x.delta[j] = (d.particle_vel[particle] - d.particle_vel[anchor]) * p.dt;
                frozen.start[j] = d.particle_prev_pos[particle];
                rates[j] = d.particle_vel[particle];
            } else {
                x.start[j] = d.particle_pos[particle];
            }
            if (p.stage == EnergyStage::Projected)
                unprojected.start[j] = ElementPosition(d, p, particle, true);
        }
        elastic += nk::vbd::ElementEnergy(element, x);
        if (p.stage == EnergyStage::Projected)
            position += double(nk::vbd::ElementEnergy(element, x)) - nk::vbd::ElementEnergy(element, unprojected);
        if (p.stage != EnergyStage::Solved) continue;
        Vec3 move[4];
        for (uint32_t j = 0u; j < count; ++j) move[j] = rates[j] * p.dt;
        add(EnergyColumn::NumericalPotential, -double(nk::vbd::ElementMovesEnergyChange(element, frozen, move)));
        for (uint32_t j = 0u; j < count; ++j) {
            const uint32_t particle = env * p.particles_per_env + p.vbd_begin + element.vertex[j];
            if (!(element.damping > 0.0f)) continue;
            Vec3 applied;
            math::SymmetricMat3 hessian;
            nk::vbd::ElementRayleighBlock(element, j, frozen, rates, &applied, &hessian);
            const Vec3 average = (d.step_particle_velocity[particle] + ParticleVelocity(d, p, particle)) * 0.5f;
            add(EnergyColumn::RayleighLoss, double(element.damping) * ParticleStep(d, p, particle) *
                Dot(applied, average));
        }
    }
    for (uint32_t i = s.first; i < p.field_elements; i += s.stride) {
        const nk::MfElement element = LedgerFieldElement(m, d, p, env, i);
        const nk::MfMaterial material = m.mf_materials[element.material];
        const float* history = d.mf_history + size_t{env} * p.field_state_scalars + element.state_offset;
        nk::mf::FieldGeometry x, unprojected;
        const uint32_t anchor = env * p.particles_per_env + p.vbd_begin + element.vertex[0];
        for (uint32_t j = 0u; j < nk::MfElementVertexCount(element); ++j) {
            const uint32_t particle = env * p.particles_per_env + p.vbd_begin + element.vertex[j];
            if (p.stage == EnergyStage::Solved) {
                x.start[j] = d.particle_prev_pos[particle];
                x.delta[j] = (d.particle_vel[particle] - d.particle_vel[anchor]) * p.dt;
            } else {
                x.start[j] = d.particle_pos[particle];
            }
            if (p.stage == EnergyStage::Projected)
                unprojected.start[j] = ElementPosition(d, p, particle, true);
        }
        const double energy = nk::mf::ElementEnergy(element, material, history, x);
        elastic += energy;
        if (p.stage == EnergyStage::Projected)
            position += energy - nk::mf::ElementEnergy(element, material, history, unprojected);
        if (p.stage == EnergyStage::Solved) {
            nk::mf::FieldGeometry start;
            Vec3 move[nk::kMfMaxVertices];
            for (uint32_t j = 0u; j < nk::MfElementVertexCount(element); ++j) {
                const uint32_t particle = env * p.particles_per_env + p.vbd_begin + element.vertex[j];
                start.start[j] = d.particle_prev_pos[particle];
                move[j] = d.particle_vel[particle] * p.dt;
            }
            add(EnergyColumn::NumericalPotential,
                -double(nk::mf::ElementMovesEnergyChange(element, material, history, start, nullptr, p.dt, move)));
        }
        // The step's commit left its plastic, or a seam's cohesive, and viscous losses after the trial history.
        if (p.stage == EnergyStage::End) {
            const float* trial = d.mf_trial + size_t{env} * p.field_trial_scalars + element.trial_offset;
            add(element.kind == nk::kMfSeam ? EnergyColumn::CohesiveLoss : EnergyColumn::PlasticLoss,
                trial[element.state_count]);
            add(EnergyColumn::ViscousLoss, trial[element.state_count + 1u]);
            // The trial still holds the hinge damage before the step's seams weakened it; the bending energy the
            // lost stiffness held is released as fracture dissipation.
            if (element.kind == nk::kMfBend)
                add(EnergyColumn::CohesiveLoss, nk::mf::ElementEnergy(element, material, trial, x) - energy);
        }
    }
    if (p.stage == EnergyStage::Solved) {
        const uint32_t first_friction = p.contact_rows + p.limit_rows;
        const uint32_t first_drive = first_friction + p.friction_rows;
        const auto* rows = reinterpret_cast<const NkRow*>(d.urows);
        const uint32_t bound = ContactRowBound(d.contact_row_extent, env, p.contact_rows);
        for (uint32_t local = LedgerRow(s.first, bound, p.contact_rows, s.stride); local < p.rows_per_env;
             local = LedgerRow(local + s.stride, bound, p.contact_rows, s.stride)) {
            const uint32_t row = env * p.rows_per_env + local;
            const double impulse = d.lambda[row];
            if (impulse == 0.0) continue;
            const auto r = rows[row];
            if (!(r.flags & nk::nk_row_flags::kActive)) continue;
            const auto a = SidePower(r.a, false, row, m, d, p);
            const auto b = SidePower(r.b, true, row, m, d, p);
            const double work = impulse * (a.physical + b.physical);
            add(EnergyColumn::RowRateWork, impulse * (a.rate + b.rate));
            add(EnergyColumn::RowPhysicalImpulseWork, impulse * (a.applied + b.applied));
            if (local >= first_drive && local < first_drive + p.drive_rows) continue;
            if ((r.flags & nk::nk_row_flags::kBlockTangent) ||
                (local >= first_friction && local < first_drive)) add(EnergyColumn::FrictionLoss, -work);
            else if (r.flags & (nk::nk_row_flags::kContactNormal | nk::nk_row_flags::kBlockNormal))
                add(EnergyColumn::NormalLoss, -work);
            else if (r.flags & nk::nk_row_flags::kJointLimit) add(EnergyColumn::LimitLoss, -work);
            else {
                add(EnergyColumn::UnclassifiedRowWork, work);
                atomicOr(coverage, nk::energy_status::kUnclassifiedRows);
            }
        }
        add(EnergyColumn::SolvedKinetic, kinetic);
    } else if (p.stage == EnergyStage::Begin) {
        add(EnergyColumn::BeginKinetic, kinetic);
        add(EnergyColumn::BeginGravity, gravity);
        add(EnergyColumn::BeginElastic, elastic);
    } else if (p.stage == EnergyStage::Free) add(EnergyColumn::FreeKinetic, kinetic);
    else if (p.stage == EnergyStage::Projected) {
        add(EnergyColumn::BeforeDatKinetic, kinetic);
        add(EnergyColumn::BeforeDatGravity, gravity);
        add(EnergyColumn::BeforeDatElastic, elastic);
        add(EnergyColumn::PositionPotential, position);
    } else {
        add(EnergyColumn::EndKinetic, kinetic);
        add(EnergyColumn::EndGravity, gravity);
        add(EnergyColumn::EndElastic, elastic);
    }
    StorePartial(values, d.readout_partials + size_t{blockIdx.x} * kPartialWidth, SumColumns{});
}

// Adds an env's partial rows to its ledger in part order; the end stage then closes the balance.
__global__ void FinishEnergyLedgerKernel(DataView d, ReadoutEnergyLedgerParams p, uint32_t parts) {
    const uint32_t env = blockIdx.x;
    const size_t record = size_t{env} * p.substeps + p.slot;
    float* output = d.energy_ledger + record * kColumns;
    double* sum = d.energy_ledger_sum + record * kColumns;
    uint32_t* coverage = d.energy_ledger_status + record;
    const double* partials = d.readout_partials + size_t{env} * parts * kPartialWidth;
    for (uint32_t c = threadIdx.x; c < kReductionColumns; c += blockDim.x) {
        const double total = CombineParts(partials, parts, c, SumColumns{});
        if (!isfinite(total)) atomicOr(coverage, nk::energy_status::kNonfinite);
        if (c == kForceWorkColumn) d.vbd_force_residual_work[record] += float(total);
        else output[c] = float(sum[c] += total);
    }
    __syncthreads();
    if (p.stage != EnergyStage::End || threadIdx.x != 0u) return;
    const auto get = [&](EnergyColumn column) { return sum[Column(column)]; };
    const auto set = [&](EnergyColumn column, double value) {
        output[Column(column)] = float(sum[Column(column)] = value);
    };
    set(EnergyColumn::DatKineticLoss, get(EnergyColumn::BeforeDatKinetic) - get(EnergyColumn::EndKinetic));
    set(EnergyColumn::DatPotential, get(EnergyColumn::EndGravity) + get(EnergyColumn::EndElastic) -
        get(EnergyColumn::BeforeDatGravity) - get(EnergyColumn::BeforeDatElastic));
    const double change = get(EnergyColumn::EndKinetic) + get(EnergyColumn::EndGravity) + get(EnergyColumn::EndElastic) -
        get(EnergyColumn::BeginKinetic) - get(EnergyColumn::BeginGravity) - get(EnergyColumn::BeginElastic);
    double work = 0.0, losses = 0.0;
    for (uint32_t c = Column(EnergyColumn::DriveWork); c <= Column(EnergyColumn::KinematicContactWork); ++c) {
        work += sum[c];
    }
    for (uint32_t c = Column(EnergyColumn::FrictionLoss); c <= Column(EnergyColumn::RayleighLoss); ++c) {
        losses += sum[c];
    }
    losses += sum[Column(EnergyColumn::PlasticLoss)] + sum[Column(EnergyColumn::ViscousLoss)] +
              sum[Column(EnergyColumn::CohesiveLoss)] + sum[Column(EnergyColumn::NumericalKinetic)] +
              sum[Column(EnergyColumn::NumericalPotential)];
    const double residual = change - work + losses + get(EnergyColumn::DatKineticLoss) -
                             get(EnergyColumn::PositionPotential);
    output[Column(EnergyColumn::Dt)] = p.dt;
    output[Column(EnergyColumn::Valid)] = 1.0f;
    // A field element past its calibrated range only warns; it voids no balance.
    if ((d.env_status[env] & ~kEnvStatusMaterialFieldRange) != 0u)
        atomicOr(coverage, nk::energy_status::kPhysicsFailure);
    if (!isfinite(residual)) atomicOr(coverage, nk::energy_status::kNonfinite);
    output[Column(EnergyColumn::Residual)] = *coverage == 0u ? float(residual) : nanf("");
    if (*coverage != 0u) output[Column(EnergyColumn::Throughput)] = nanf("");
}

__device__ bool Finite(Vec3 v) { return isfinite(v.x) && isfinite(v.y) && isfinite(v.z); }
__device__ bool Finite(math::Transform pose) {
    return Finite(pose.position) && isfinite(pose.rotation.w) && isfinite(pose.rotation.x) &&
        isfinite(pose.rotation.y) && isfinite(pose.rotation.z);
}

// Momentum is measured at the stage's physical velocity and world evaluation pose.
__global__ void PhysicsStageMetricsKernel(ModelView m, DataView d, ReadoutEnergyLedgerParams p) {
    using C = nk::PhysicsStageColumn;
    constexpr uint32_t columns = nk::kPhysicsStageColumnCount;
    const ReadoutSlice s = Slice(p.env_count);
    const uint32_t env = s.env;
    double values[columns]{};
    values[static_cast<uint32_t>(C::MinVbdEffectiveDt)] = DBL_MAX;
    const auto add = [&](C c, double value) { values[static_cast<uint32_t>(c)] += value; };
    const auto maximum = [&](C c, double value) {
        auto& held = values[static_cast<uint32_t>(c)];
        held = isfinite(value) ? fmax(held, value) : double(nanf(""));
    };
    const auto vector = [&](C first, Vec3 v) {
        const uint32_t i = static_cast<uint32_t>(first);
        values[i] += v.x; values[i + 1u] += v.y; values[i + 2u] += v.z;
    };
    const auto momentum = [&](double mass, Vec3 position, Vec3 velocity, Vec3 spin) {
        add(C::DynamicMass, mass);
        const double linear[3] = {mass * velocity.x, mass * velocity.y, mass * velocity.z};
        for (uint32_t k = 0u; k < 3u; ++k)
            values[static_cast<uint32_t>(C::LinearMomentumX) + k] += linear[k];
        add(C::AngularMomentumX, spin.x + double(position.y) * linear[2] - double(position.z) * linear[1]);
        add(C::AngularMomentumY, spin.y + double(position.z) * linear[0] - double(position.x) * linear[2]);
        add(C::AngularMomentumZ, spin.z + double(position.x) * linear[1] - double(position.y) * linear[0]);
    };
    for (uint32_t local = s.first; local < p.particles_per_env; local += s.stride) {
        const uint32_t particle = env * p.particles_per_env + local;
        const Vec3 velocity = ParticleVelocity(d, p, particle);
        const Vec3 position = p.stage == EnergyStage::Free || p.stage == EnergyStage::Solved
            ? d.particle_prev_pos[particle] + d.particle_vel[particle] * p.dt : d.particle_pos[particle];
        const float inverse_mass = d.particle_inv_mass[particle];
        if (!Finite(position) || !Finite(velocity) || !isfinite(inverse_mass)) add(C::NonfiniteParticles, 1.0);
        if (!(inverse_mass > 0.0f)) {
            if (p.stage == EnergyStage::Solved && IsVertex(p, particle))
                vector(C::VbdElasticBoundaryImpulseX, KinematicElasticReaction(m, d, p, particle).gradient * p.dt);
            continue;
        }
        momentum(1.0 / inverse_mass, position, velocity, {});
        if (!IsVertex(p, particle)) continue;
        const uint32_t vertex = local - p.vbd_begin;
        const uint32_t slot = env * p.vbd_vertices + vertex;
        const bool second_order = p.stage == EnergyStage::Begin
            ? p.cloth_integrator == 0u && d.vbd_history_ready[env] != 0u && d.vbd_restart[slot] == 0u &&
                  vertex < p.vbd_vertices - p.field_vertices
            : ParticleStep(d, p, particle) < p.dt;
        const Vec3 discrete = second_order ? velocity * 1.5f - d.vbd_history_vel[slot] * 0.5f : velocity;
        vector(C::VbdDiscreteMomentumX, discrete / inverse_mass);
        if (second_order) add(C::VbdBdfParticles, 1.0);
        if (p.stage == EnergyStage::Solved) {
            vector(C::VbdGravityImpulseX, Vec3{p.gravity[0], p.gravity[1], p.gravity[2]} * (p.dt / inverse_mass));
            vector(C::VbdRowImpulseX, d.particle_row_impulse[particle]);
            vector(C::VbdElasticBoundaryImpulseX, (d.particle_kinematic_target[particle] - position) *
                (d.particle_attachment_stiffness[particle] * p.dt));
        }
        add(C::VbdDynamicParticles, 1.0);
        if (p.stage == EnergyStage::Begin) continue;
        const float step = ParticleStep(d, p, particle);
        auto& minimum = values[static_cast<uint32_t>(C::MinVbdEffectiveDt)];
        minimum = fmin(minimum, double(step));
        maximum(C::MaxVbdEffectiveDt, step);
        if (p.stage != EnergyStage::Solved) continue;
        const VertexBlockView block = LedgerVertices(m, d, p);
        const Vec3 displacement = d.particle_vel[particle] * p.dt;
        const Vec3 force = VertexBlockForceDouble(block, d.particle_vel, env, vertex,
            nk::vbd::InertialGradient(d.particle_vel[particle], d.vbd_solve_rate[slot], d.vbd_inertia[slot], p.dt),
            d.particle_row_impulse[particle] / p.dt);
        const Vec3 defect = force * p.dt;
        vector(C::VbdMomentumDefectX, defect);
        vector(C::VbdAngularDefectX, position.Cross(defect));
        const Vec3 error = nk::vbd::MomentumResidualVelocity(force, inverse_mass, step);
        maximum(C::MaxVbdMomentumVelocityError, Finite(error)
            ? fmaxf(fabsf(error.x), fmaxf(fabsf(error.y), fabsf(error.z))) : nanf(""));
        maximum(C::MaxVbdForce, Finite(force)
            ? fmaxf(fabsf(force.x), fmaxf(fabsf(force.y), fabsf(force.z))) : nanf(""));
        maximum(C::MaxAbsoluteVertexResidualWork, fabs(Dot(force, displacement)));
    }
    for (uint32_t local = s.first; local < p.bodies_per_env; local += s.stride) {
        const uint32_t body = env * p.bodies_per_env + local;
        if (m.body_to_link && m.body_to_link[body] != ~0u) continue;
        const auto pose = ComposeFrame(d.body_pose[body], d.body_inertial_frame[body]);
        const Vec3 velocity = d.body_linear_velocity[body];
        const Vec3 angular = InverseRotate(pose.rotation, d.body_angular_velocity[body]);
        const Vec3 inverse = d.body_inv_inertia[body];
        const float inverse_mass = d.body_inv_mass[body];
        if (!Finite(pose) || !Finite(velocity) || !Finite(angular) || !Finite(inverse) ||
            !isfinite(inverse_mass)) add(C::NonfiniteBodies, 1.0);
        if (!(inverse_mass > 0.0f)) continue;
        const Vec3 spin = mg::RotateByQuatNormalized(pose.rotation,
            {inverse.x > 0.0f ? angular.x / inverse.x : 0.0f,
             inverse.y > 0.0f ? angular.y / inverse.y : 0.0f,
             inverse.z > 0.0f ? angular.z / inverse.z : 0.0f});
        momentum(1.0 / inverse_mass, pose.position, velocity, spin);
    }
    for (uint32_t local = s.first; local < p.links_per_env; local += s.stride) {
        const uint32_t link = env * p.links_per_env + local;
        const auto pose = d.energy_link_pose[link];
        const auto world = d.energy_link_velocity[link];
        const Vec3 angular = InverseRotate(pose.rotation, {world.v[0], world.v[1], world.v[2]});
        const Vec3 linear = InverseRotate(pose.rotation, {world.v[3], world.v[4], world.v[5]});
        if (!Finite(pose) || !Finite(angular) || !Finite(linear) || !isfinite(d.q[link]) ||
            !isfinite(d.qdot[link])) add(C::NonfiniteLinks, 1.0);
        const float velocity[6] = {angular.x, angular.y, angular.z, linear.x, linear.y, linear.z};
        double spatial[6]{};
        const auto inertia = m.link_inertia[link];
        for (uint32_t r = 0u; r < 6u; ++r)
            for (uint32_t c = 0u; c < 6u; ++c) spatial[r] += double(inertia.m[r * 6u + c]) * velocity[c];
        const Vec3 force = mg::RotateByQuatNormalized(pose.rotation,
            {float(spatial[3]), float(spatial[4]), float(spatial[5])});
        const Vec3 spin = mg::RotateByQuatNormalized(pose.rotation,
            {float(spatial[0]), float(spatial[1]), float(spatial[2])});
        add(C::DynamicMass, inertia.m[21]);
        vector(C::LinearMomentumX, force);
        vector(C::AngularMomentumX, spin + pose.position.Cross(force));
    }
    if (p.stage == EnergyStage::Solved) {
        const auto* rows = reinterpret_cast<const NkRow*>(d.urows);
        const uint32_t first_drive = p.contact_rows + p.limit_rows + p.friction_rows;
        const uint32_t bound = ContactRowBound(d.contact_row_extent, env, p.contact_rows);
        for (uint32_t local = LedgerRow(s.first, bound, p.contact_rows, s.stride); local < p.rows_per_env;
             local = LedgerRow(local + s.stride, bound, p.contact_rows, s.stride)) {
            const uint32_t row = env * p.rows_per_env + local;
            const auto r = rows[row];
            if (!(r.flags & nk::nk_row_flags::kActive)) continue;
            add(C::ActiveRows, 1.0);
            if (local >= first_drive && local < first_drive + p.drive_rows) continue;
            const auto a = SidePower(r.a, false, row, m, d, p);
            const auto b = SidePower(r.b, true, row, m, d, p);
            const double work = double(d.lambda[row]) * (a.physical + b.physical);
            if (work > 0.0) {
                add(C::EnergyInjectingPassiveRows, 1.0);
                maximum(C::MaxPassiveRowEnergyGain, work);
            }
        }
    }
    StorePartial(values, d.readout_partials + size_t{blockIdx.x} * kPartialWidth, CombineStageMetric{});
}

// Combines an env's partial rows of stage metrics in part order.
__global__ void FinishStageMetricsKernel(DataView d, ReadoutEnergyLedgerParams p, uint32_t parts) {
    using C = nk::PhysicsStageColumn;
    constexpr uint32_t columns = nk::kPhysicsStageColumnCount;
    const size_t record = size_t{blockIdx.x} * p.substeps + p.slot;
    const double* partials = d.readout_partials + size_t{blockIdx.x} * parts * kPartialWidth;
    float* output = d.physics_stage_metrics +
        (record * nk::kEnergyStageCount + static_cast<uint32_t>(p.stage)) * columns;
    for (uint32_t c = threadIdx.x; c < columns; c += blockDim.x) {
        const double total = CombineParts(partials, parts, c, CombineStageMetric{});
        output[c] = c == static_cast<uint32_t>(C::Dt) ? p.dt :
            c == static_cast<uint32_t>(C::MinVbdEffectiveDt) && total == DBL_MAX ? 0.0f : float(total);
        if (!isfinite(output[c])) atomicOr(d.energy_ledger_status + record, nk::energy_status::kNonfinite);
    }
}

__device__ Vec3 AuditLinearJacobian(const NkRowSide& side, PointMassView points) {
    if (!PointMassView::IsPointSide(side.kind)) return side.jlin;
    Vec3 total{};
    for (uint32_t i = 0u; i < points.Count(side); ++i) total += points.At(side, i).jacobian;
    return total;
}

__device__ double AuditPointDisplacement(const NkRowSide& side, PointMassView points,
    const DataView& d, bool* measured) {
    if (side.kind == nk::kNkSideStatic) return 0.0;
    if (!PointMassView::IsPointSide(side.kind)) { *measured = false; return 0.0; }
    double value = 0.0;
    for (uint32_t k = 0u; k < points.Count(side); ++k) {
        const auto term = points.At(side, k);
        if (term.kind != nk::kNkSideParticle) { *measured = false; continue; }
        value += Dot(term.jacobian, d.particle_pos[term.index] - d.energy_particle_start[term.index]);
    }
    return value;
}

__device__ uint32_t AuditParticleFeature(const NkRowSide& side, PointMassView points,
    const DataView& d, Vec3* positions) {
    if (!PointMassView::IsPointSide(side.kind)) return 0u;
    const uint32_t count = points.Count(side);
    if (count > 3u) return 0u;
    for (uint32_t k = 0u; k < count; ++k) {
        const auto term = points.At(side, k);
        if (term.kind != nk::kNkSideParticle) return 0u;
        positions[k] = d.particle_pos[term.index];
    }
    return count;
}

// Closest-feature gap less the contact radius the detection recorded.
__device__ bool AuditFeatureGap(const Vec3* a, uint32_t na, const Vec3* b, uint32_t nb,
    const DataView& d, size_t at, double* gap) {
    Vec3 point_a{}, point_b{};
    if (na == 1u && nb == 3u) {
        point_a = a[0]; point_b = collision::ClosestTrianglePoint(a[0], b[0], b[1], b[2]).point;
    } else if (na == 3u && nb == 1u) {
        point_a = collision::ClosestTrianglePoint(b[0], a[0], a[1], a[2]).point; point_b = b[0];
    } else if (na == 2u && nb == 2u) {
        const auto pair = collision::OgcClosestSegments(a[0], a[1], b[0], b[1]);
        point_a = pair.a; point_b = pair.b;
    } else return false;
    const Vec3 initial = d.ucontact_witness_a[at] - d.ucontact_witness_b[at];
    const double radius = sqrt(Dot(initial, initial)) + d.ucontact_depth[at];
    const Vec3 difference = point_a - point_b;
    *gap = sqrt(Dot(difference, difference)) - radius;
    return radius >= 0.0 && isfinite(*gap);
}

__device__ bool AuditOgcParticleGap(const NkRow& row, PointMassView points,
    const DataView& d, size_t at, double* gap) {
    Vec3 a[3]{}, b[3]{};
    const uint32_t na = AuditParticleFeature(row.a, points, d, a);
    const uint32_t nb = AuditParticleFeature(row.b, points, d, b);
    return AuditFeatureGap(a, na, b, nb, d, at, gap);
}

// Mixed slots place their features at the committed poses the DAT certified.
__device__ bool AuditMixedGap(const ModelView& m, const DataView& d, const ReadoutEnergyLedgerParams& p,
    uint32_t env, uint32_t slot, size_t at, double* gap) {
    if (d.dat_pair_kind == nullptr || d.dat_pair_kind[slot] == 0u) return false;
    DatTruncateParams q;
    q.particles_per_env = p.particles_per_env;
    q.surfaces_per_env = p.surfaces_per_env;
    q.bodies_per_env = p.bodies_per_env;
    q.links_per_env = p.links_per_env;
    DatPointRef ra[3], rb[3];
    uint32_t na = 0u, nb = 0u;
    if (!DatMixedRefs(q, m, d, env, slot, ra, &na, rb, &nb)) return false;
    Vec3 a[3]{}, b[3]{};
    for (uint32_t i = 0u; i < na; ++i)
        a[i] = ra[i].particle ? d.particle_pos[ra[i].owner] : DatPointAt(q, m, d, env, ra[i], 1.0f);
    for (uint32_t i = 0u; i < nb; ++i)
        b[i] = rb[i].particle ? d.particle_pos[rb[i].owner] : DatPointAt(q, m, d, env, rb[i], 1.0f);
    return AuditFeatureGap(a, na, b, nb, d, at, gap);
}

// Particle-body slots measure the particle sphere against the body surface at its committed pose.
__device__ bool AuditBodyParticleGap(const ModelView& m, const DataView& d, const ReadoutEnergyLedgerParams& p,
    const SurfaceQueryView& surfaces, uint32_t env, size_t at, double* gap) {
    if (d.ucontact_a_kind[at] != nk::kUContactSideParticle || d.ucontact_b_kind[at] != nk::kUContactSideBody)
        return false;
    const uint32_t particle = d.ucontact_a[at], body = d.ucontact_b[at];
    const uint32_t local = particle - env * p.particles_per_env;
    if (local >= p.particles_per_env || body >= p.bodies_per_env) return false;
    const auto pose = d.body_pose[size_t{env} * p.bodies_per_env + body];
    const auto surface = QueryCollidableSurface(surfaces, body, LoadPrimShape(m.shape_table, body),
        PrimInverseTransformPoint(pose, d.particle_pos[particle]), FLT_MAX);
    *gap = double(surface.distance) -
           (local - p.vbd_begin < p.vbd_vertices ? m.particle_half_thickness[local] : p.particle_radius);
    return surface.valid && isfinite(*gap);
}

// The solved stage opens a substep's contact audit before its blocks count into it.
__global__ void ClearContactAuditKernel(DataView d, ReadoutEnergyLedgerParams p) {
    const size_t record = size_t{blockIdx.x} * p.substeps + p.slot;
    for (uint32_t i = threadIdx.x; i < nk::kContactAuditCountSize; i += blockDim.x)
        d.contact_audit_counts[record * nk::kContactAuditCountSize + i] = 0u;
    for (uint32_t i = threadIdx.x; i < nk::kContactAuditMetricCount; i += blockDim.x)
        d.contact_audit_metrics[record * nk::kContactAuditMetricCount + i] = 0u;
}

// Impulse laws use the solved state; closest-feature gaps use the committed state.
__global__ void ContactAuditKernel(ModelView m, DataView d, ReadoutEnergyLedgerParams p, SurfaceQueryView surfaces) {
    using C = nk::ContactAuditCount;
    using M = nk::ContactAuditMetric;
    const ReadoutSlice s = Slice(p.env_count);
    const uint32_t env = s.env;
    const size_t record = size_t{env} * p.substeps + p.slot;
    uint32_t* counts = d.contact_audit_counts + record * nk::kContactAuditCountSize;
    uint64_t* metrics = d.contact_audit_metrics + record * nk::kContactAuditMetricCount;
    PointMassView points;
    points.ranges = d.point_endpoint_ranges;
    points.terms = d.point_endpoint_terms;
    const auto* rows = reinterpret_cast<const NkRow*>(d.urows);
    const auto count = [&](C column) { atomicAdd(counts + static_cast<uint32_t>(column), 1u); };
    const auto maximum = [&](M column, float value, uint32_t row) {
        const float magnitude = isfinite(value) ? fmaxf(value, 0.0f) : FLT_MAX;
        const unsigned long long packed = (static_cast<unsigned long long>(__float_as_uint(magnitude)) << 32u) | ~row;
        atomicMax(reinterpret_cast<unsigned long long*>(metrics + static_cast<uint32_t>(column)), packed);
    };
    const uint32_t bound = ContactRowBound(d.contact_row_extent, env, p.contact_rows);
    for (uint32_t local = s.first; local < bound; local += s.stride) {
        const uint32_t slot = env * p.rows_per_env + local;
        const auto normal = rows[slot];
        if (!(normal.flags & nk::nk_row_flags::kActive) ||
            !(normal.flags & nk::nk_row_flags::kContactNormal)) continue;
        if (p.stage == EnergyStage::End) {
            const uint32_t global_contact = nkops::ContactSlotForRow(d.ucontact_row_range,
                env * p.contact_slots_per_env, p.contact_slots_per_env, slot);
            const uint32_t local_contact = global_contact - env * p.contact_slots_per_env;
            const uint32_t point = slot - nkops::ContactRowRangeFor(d.ucontact_row_range, global_contact).first;
            const size_t at = size_t{global_contact} * nk::kPairDrivenPtsPerSlot + point;
            double gap = 0.0;
            const bool ogc = local_contact >= p.rigid_contact_slots &&
                local_contact - p.rigid_contact_slots < p.ogc_contact_slots;
            if (ogc ? AuditOgcParticleGap(normal, points, d, at, &gap) ||
                      AuditMixedGap(m, d, p, env, global_contact, at, &gap)
                    : AuditBodyParticleGap(m, d, p, surfaces, env, at, &gap)) {
                if (gap < -p.pos_slop) count(C::GapViolations);
                maximum(M::GapPenetration, float(-gap), slot);
            } else count(C::UnmeasuredGaps);
            bool measured = true;
            const double displacement = AuditPointDisplacement(normal.a, points, d, &measured) +
                                        AuditPointDisplacement(normal.b, points, d, &measured);
            if (measured) maximum(M::FrozenGapPenetration, float(double(d.ucontact_depth[at]) - displacement), slot);
            continue;
        }
        count(C::Contacts);
        const uint32_t axes = normal.flags & nk::nk_row_flags::kBlockNormal ? 3u : 1u;
        const float normal_impulse = d.lambda[slot];
        if (!isfinite(normal_impulse)) count(C::InvalidRows);
        if (normal_impulse < 0.0f) count(C::NegativeNormalImpulses);
        maximum(M::NegativeNormalImpulse, -normal_impulse, slot);
        double scaled_tangent[2]{};
        bool cone_invalid = false;
        for (uint32_t axis = 0u; axis < axes; ++axis) {
            const uint32_t row = slot + axis * normal.group_normal_count;
            if (row >= (env + 1u) * p.rows_per_env) { count(C::InvalidRows); continue; }
            const auto r = rows[row];
            count(C::Rows);
            const Vec3 a = AuditLinearJacobian(r.a, points);
            const Vec3 b = AuditLinearJacobian(r.b, points);
            const double impulse = d.lambda[row];
            const double numerator = sqrt(Dot(a + b, a + b)) * fabs(impulse);
            const double denominator = sqrt(Dot(a, a)) * fabs(impulse);
            const double relative = denominator > 0.0 ? numerator / denominator : numerator == 0.0 ? 0.0 : FLT_MAX;
            if (!Finite(a) || !Finite(b) || !isfinite(impulse)) count(C::InvalidRows);
            if (relative > 1.0e-6) count(C::LinearClosureViolations);
            maximum(M::RelativeLinearClosure, float(relative), row);
            if (impulse != 0.0)
                maximum(axis == 0u ? M::NormalDisplacementRate : M::TangentDisplacementRate,
                    float(copysign(1.0, impulse) * (SideDisplacementRate(r.a, false, row, d, p) +
                                                    SideDisplacementRate(r.b, true, row, d, p))), row);
            if (axis != 0u) {
                const float mu = axis == 1u ? normal.mu : normal.friction_secondary;
                if (mu > 0.0f) scaled_tangent[axis - 1u] = impulse / mu;
                else if (impulse != 0.0) cone_invalid = true;
            }
        }
        const double tangent = hypot(scaled_tangent[0], scaled_tangent[1]);
        if (cone_invalid || tangent > fmax(double(normal_impulse), 0.0) * 1.001) count(C::ConeViolations);
        maximum(M::ConeImpulseExcess, cone_invalid ? FLT_MAX : float(tangent - normal_impulse), slot);
    }
}

// Blocks per env of the readouts: one per kThreads items of the env's longest list, within the partial rows.
uint32_t ReadoutParts(const ReadoutEnergyLedgerParams& p) {
    const uint32_t items = std::max({p.particles_per_env, p.bodies_per_env, p.links_per_env, p.vbd_elements,
                                     p.field_elements, p.rows_per_env, 1u});
    return std::min((items + kThreads - 1u) / kThreads, std::max(1u, nk::kReadoutPartialRows / p.env_count));
}

Status OpReadoutContactAudit(const ModelView& m, const DataView& d, const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const ReadoutEnergyLedgerParams*>(params);
    if (!p || p->slot >= p->substeps || !d.contact_audit_counts || !d.contact_audit_metrics)
        return Status::InvalidArgument;
    const auto surfaces = MakeSurfaceQueryView(m, p->bodies_per_env, p->mesh_geometry,
                                               p->sdf_grid_count, p->sdf_cell_total);
    if (p->env_count > 0u) {
        if (p->stage == EnergyStage::Solved)
            LaunchCuda(ClearContactAuditKernel, dim3(p->env_count), dim3(32u), 0u, stream, d, *p);
        LaunchCuda(ContactAuditKernel, dim3(p->env_count * ReadoutParts(*p)), dim3(kThreads), 0u, stream,
                   m, d, *p, surfaces);
    }
    return cudaGetLastError() == cudaSuccess ? Status::Ok : Status::Failed;
}

Status OpReadoutEnergyLedger(const ModelView& m, const DataView& d, const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const ReadoutEnergyLedgerParams*>(params);
    if (!p || !(p->dt > 0.0f) || !std::isfinite(p->dt) || p->slot >= p->substeps ||
        static_cast<uint32_t>(p->stage) >= nk::kEnergyStageCount || !d.energy_ledger || !d.energy_ledger_status ||
        (p->physics_diagnostics && !d.physics_stage_metrics))
        return Status::InvalidArgument;
    if (p->env_count == 0u) return Status::Ok;
    if (p->artics_per_env > 0u)
        LaunchCuda(EnergyLinkFramesKernel, dim3(p->env_count * p->artics_per_env), dim3(32u), 0u, stream, m, d, *p);
    if (p->stage == EnergyStage::Solved && p->rows_per_env > 0u) {
        const uint64_t rows = uint64_t{p->env_count} * p->rows_per_env;
        const uint64_t blocks = (rows + kThreads - 1u) / kThreads;
        if (blocks > std::numeric_limits<uint32_t>::max()) return Status::InvalidArgument;
        LaunchCuda(EnergyKinematicImpulseKernel, dim3(static_cast<uint32_t>(blocks)), dim3(kThreads), 0u, stream, d, *p);
    }
    const uint32_t parts = ReadoutParts(*p);
    if (p->physics_diagnostics) {
        LaunchCuda(PhysicsStageMetricsKernel, dim3(p->env_count * parts), dim3(kThreads), 0u, stream, m, d, *p);
        LaunchCuda(FinishStageMetricsKernel, dim3(p->env_count), dim3(kThreads), 0u, stream, d, *p, parts);
    }
    if (p->stage == EnergyStage::Begin)
        LaunchCuda(BeginEnergyLedgerKernel, dim3(p->env_count), dim3(kThreads), 0u, stream, d, *p);
    LaunchCuda(EnergyLedgerKernel, dim3(p->env_count * parts), dim3(kThreads), 0u, stream, m, d, *p);
    LaunchCuda(FinishEnergyLedgerKernel, dim3(p->env_count), dim3(kThreads), 0u, stream, d, *p, parts);
    return cudaGetLastError() == cudaSuccess ? Status::Ok : Status::Failed;
}

}  // namespace

void RegisterNkEnergyLedgerOps() {
    SetCudaOp(NkOp::ReadoutEnergyLedger, &OpReadoutEnergyLedger);
    SetCudaOp(NkOp::ReadoutContactAudit, &OpReadoutContactAudit);
}

}  // namespace nuka::phi
