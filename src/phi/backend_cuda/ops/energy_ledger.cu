#include <cuda_runtime.h>

#include "collision/analytical_manifold.hpp"
#include "phi/backend_cuda/ops/contact_row_layout.cuh"

#include <cmath>
#include <cfloat>
#include <limits>

#include "nk/readout/energy_ledger.hpp"
#include "nk/material/mpm_constitutive.hpp"
#include "nk/material/mpm_energy.hpp"
#include "nk/readout/physics_diagnostics.hpp"
#include "nk/solve/vertex_block.hpp"
#include "collision/ogc_geometry.hpp"
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/articulation_types.cuh"
#include "phi/backend_cuda/ops/contact_features.cuh"
#include "phi/backend_cuda/ops/kinematics.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/registry.cuh"
#include "phi/backend_cuda/ops/row_coupling.cuh"
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
static_assert(kColumns == 45u, "energy field record must match its named columns");

__device__ uint32_t Column(EnergyColumn value) { return static_cast<uint32_t>(value); }
__device__ double Dot(Vec3 a, Vec3 b) {
    return double(a.x) * b.x + double(a.y) * b.y + double(a.z) * b.z;
}
__device__ Vec3 InverseRotate(math::Quat q, Vec3 value) {
    return mg::RotateByQuatNormalized(mg::MakeQuat(q.w, -q.x, -q.y, -q.z), value);
}
__device__ nk::Spatial6 ComTwist(Vec3 angular, Vec3 linear) {
    return {{angular.x, angular.y, angular.z, linear.x, linear.y, linear.z}};
}
// Wrenches contain world force then COM torque; twists contain angular then COM linear velocity.
__device__ double LoadWork(const nk::Spatial6& wrench, const nk::Spatial6& before,
                          const nk::Spatial6& solved, float dt) {
    double work = 0.0;
    for (uint32_t k = 0u; k < 3u; ++k)
        work += double(wrench.v[k]) * (double(before.v[k + 3u]) + solved.v[k + 3u]) +
                double(wrench.v[k + 3u]) * (double(before.v[k]) + solved.v[k]);
    return 0.5 * dt * work;
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

__device__ void MpmEnergyAxes(Vec3 position, const ReadoutEnergyLedgerParams& p,
                             nk::MpmCompactAxis (&axes)[3]) {
    const float inverse_dx = 1.0f / p.grid_dx;
    const float coordinate[3] = {(position.x - p.grid_origin[0]) * inverse_dx,
        (position.y - p.grid_origin[1]) * inverse_dx, (position.z - p.grid_origin[2]) * inverse_dx};
    for (int a = 0; a < 3; ++a) axes[a] = nk::MpmCompactWeights(coordinate[a]);
}

__device__ nk::MpmMaterial EnergyMpmMaterial(const DataView& d, uint32_t particle) {
    return reinterpret_cast<const nk::MpmMaterial*>(d.mpm_material_table)[d.particle_material_id[particle]];
}

__global__ void EnergyGridClearKernel(DataView d, ReadoutEnergyLedgerParams p) {
    const size_t node = size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (node >= size_t{p.env_count} * p.grid_nodes_per_env) return;
    d.energy_grid_entry_mass[node] = 0.0;
    for (uint32_t a = 0u; a < 3u; ++a) {
        d.energy_grid_entry_momentum[3u * node + a] = 0.0;
        d.energy_grid_stress_impulse[3u * node + a] = 0.0;
        d.energy_grid_viscous_impulse[3u * node + a] = 0.0;
    }
    d.energy_grid_free_velocity[node] = d.grid_velocity[node];
    d.energy_grid_prescribed[node] = 0u;
}

// Reconstruct transfer contributions in private double sums without changing the simulation grid.
__global__ void EnergyMpmTransferKernel(DataView d, ReadoutEnergyLedgerParams p) {
    const uint32_t at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at >= p.env_count * p.mpm_particles_per_env) return;
    const uint32_t env = at / p.mpm_particles_per_env;
    const uint32_t particle = env * p.particles_per_env + at % p.mpm_particles_per_env;
    const size_t record = size_t{env} * p.substeps + p.slot;
    if (d.particle_material_id[particle] >= p.mpm_material_count) {
        atomicOr(d.energy_ledger_status + record, nk::energy_status::kGridMaterial);
        return;
    }
    const auto material = EnergyMpmMaterial(d, particle);
    const Vec3 position = d.particle_pos[particle], velocity = d.particle_vel[particle];
    const double mass = d.particle_inv_mass[particle] > 0.0f ? 1.0 / d.particle_inv_mass[particle] : 0.0;
    const float* affine = d.particle_C + size_t{particle} * 9u;
    const float* stress = d.mpm_particle_stress + size_t{particle} * 9u;
    const float* deformation = d.particle_F + size_t{particle} * 9u;
    const double j_visc = nk::material::detail::Determinant(deformation) * material.viscosity;
    nk::MpmCompactAxis axes[3];
    MpmEnergyAxes(position, p, axes);
    float inverse[6];
    if (!nk::MpmApicInverse(axes, 1.0f / p.grid_dx, inverse)) {
        atomicOr(d.energy_ledger_status + record, nk::energy_status::kGridMaterial);
        return;
    }
    if (p.mpm_implicit_stress && material.viscosity > 0.0f)
        atomicOr(d.energy_ledger_status + record, nk::energy_status::kGridMaterial);
    if (mass == 0.0)
        for (int k = 0; k < 9; ++k)
            if (stress[k] != 0.0f) atomicOr(d.energy_ledger_status + record, nk::energy_status::kGridMaterial);
    for (uint32_t lattice = 0u; lattice < nk::kMpmLattices; ++lattice)
        for (uint32_t z = 0u; z < nk::kMpmStencilWidth; ++z)
            for (uint32_t y = 0u; y < nk::kMpmStencilWidth; ++y)
                for (uint32_t x = 0u; x < nk::kMpmStencilWidth; ++x) {
                    const int64_t node = nk::MpmNodeIndex(env, lattice, axes[0].base[lattice] + x,
                        axes[1].base[lattice] + y, axes[2].base[lattice] + z, p.grid_dims, p.grid_nodes_per_env);
                    if (node < 0) continue;
                    const double weight = double(nk::kMpmLatticeShare) * axes[0].w[lattice][x] *
                        axes[1].w[lattice][y] * axes[2].w[lattice][z];
                    if (mass == 0.0 && float(weight * d.particle_vol0[particle]) > 0.0f)
                        atomicExch(d.energy_grid_prescribed + node, 1u);
                    atomicAdd(d.energy_grid_entry_mass + node, weight * mass);
                    const float shift = nk::MpmLatticeOffset(lattice);
                    const double offset[3] = {
                        p.grid_origin[0] + (float(axes[0].base[lattice] + x) + shift) * p.grid_dx - position.x,
                        p.grid_origin[1] + (float(axes[1].base[lattice] + y) + shift) * p.grid_dx - position.y,
                        p.grid_origin[2] + (float(axes[2].base[lattice] + z) + shift) * p.grid_dx - position.z};
                    const double gradient[3] = {
                        inverse[0] * offset[0] + inverse[3] * offset[1] + inverse[4] * offset[2],
                        inverse[3] * offset[0] + inverse[1] * offset[1] + inverse[5] * offset[2],
                        inverse[4] * offset[0] + inverse[5] * offset[1] + inverse[2] * offset[2]};
                    const float linear[3] = {velocity.x, velocity.y, velocity.z};
                    for (uint32_t a = 0u; a < 3u; ++a) {
                        double momentum = linear[a], impulse = 0.0, viscous = 0.0;
                        for (uint32_t b = 0u; b < 3u; ++b) {
                            momentum += affine[3u * a + b] * offset[b];
                            impulse += stress[3u * a + b] * gradient[b];
                            viscous += j_visc * (affine[3u * a + b] + affine[3u * b + a]) * gradient[b];
                        }
                        atomicAdd(d.energy_grid_entry_momentum + 3u * node + a, weight * mass * momentum);
                        if (!p.mpm_implicit_stress) {
                            const double scale = -double(p.dt) * weight * d.particle_vol0[particle];
                            atomicAdd(d.energy_grid_stress_impulse + 3u * node + a, scale * impulse);
                            atomicAdd(d.energy_grid_viscous_impulse + 3u * node + a, scale * viscous);
                        }
                    }
                }
}

// Transfer loss is the weighted least-squares residual, so subtraction of almost equal energies cannot change its sign.
__device__ double MpmTransferLoss(const DataView& d, const ReadoutEnergyLedgerParams& p,
    uint32_t particle, Vec3 position, Vec3 velocity, const float* affine, bool scatter) {
    const uint32_t env = particle / p.particles_per_env;
    const double mass = 1.0 / d.particle_inv_mass[particle];
    const float linear[3] = {velocity.x, velocity.y, velocity.z};
    nk::MpmCompactAxis axes[3];
    MpmEnergyAxes(position, p, axes);
    double loss = 0.0;
    for (uint32_t lattice = 0u; lattice < nk::kMpmLattices; ++lattice)
        for (uint32_t z = 0u; z < nk::kMpmStencilWidth; ++z)
            for (uint32_t y = 0u; y < nk::kMpmStencilWidth; ++y)
                for (uint32_t x = 0u; x < nk::kMpmStencilWidth; ++x) {
                    const int64_t node = nk::MpmNodeIndex(env, lattice, axes[0].base[lattice] + x,
                        axes[1].base[lattice] + y, axes[2].base[lattice] + z, p.grid_dims, p.grid_nodes_per_env);
                    if (node < 0) continue;
                    const double weight = double(nk::kMpmLatticeShare) * axes[0].w[lattice][x] *
                        axes[1].w[lattice][y] * axes[2].w[lattice][z];
                    const float shift = nk::MpmLatticeOffset(lattice);
                    const double offset[3] = {
                        scatter ? p.grid_origin[0] + (float(axes[0].base[lattice] + x) + shift) * p.grid_dx - position.x
                                : axes[0].offset[lattice][x] * p.grid_dx,
                        scatter ? p.grid_origin[1] + (float(axes[1].base[lattice] + y) + shift) * p.grid_dx - position.y
                                : axes[1].offset[lattice][y] * p.grid_dx,
                        scatter ? p.grid_origin[2] + (float(axes[2].base[lattice] + z) + shift) * p.grid_dx - position.z
                                : axes[2].offset[lattice][z] * p.grid_dx};
                    const Vec3 solved = d.grid_velocity[node];
                    const double grid[3] = {solved.x, solved.y, solved.z};
                    const double entry_mass = d.energy_grid_entry_mass[node];
                    for (uint32_t a = 0u; a < 3u; ++a) {
                        double fitted = linear[a];
                        for (uint32_t b = 0u; b < 3u; ++b) fitted += affine[3u * a + b] * offset[b];
                        const double reference = scatter ?
                            (d.grid_inv_mass[node] > 0.0f && entry_mass > 0.0
                                ? d.energy_grid_entry_momentum[3u * node + a] / entry_mass : 0.0) : grid[a];
                        const double error = fitted - reference;
                        loss += 0.5 * mass * weight * error * error;
                    }
                }
    return loss;
}

struct EnergyMpmArithmetic {
    __device__ static float Sum3(float a, float b, float c) { return __fadd_rn(__fadd_rn(a, b), c); }
};

__device__ bool MpmReturnEnergy(const DataView& d, const ReadoutEnergyLedgerParams& p,
    uint32_t particle, const nk::MpmMaterial& material, double* plastic_loss, double* averaging_work) {
    const size_t base = size_t{particle} * 9u;
    const float* initial = d.energy_mpm_start_F + base;
    const float* affine = d.particle_C + base;
    const float alpha = d.energy_mpm_start_plastic[particle];
    const float* plastic = d.energy_mpm_start_plastic_F ? d.energy_mpm_start_plastic_F + base : nullptr;
    nk::material::MpmMaterialTrial accepted;
    const auto status = nk::material::EvaluateMpmMaterialTrial<EnergyMpmArithmetic>(
        material, {initial, plastic, alpha}, affine, p.dt, accepted);
    if (status != nk::material::ConstitutiveStatus::Ok) return false;
    double accepted_energy = 0.0, actual_energy = 0.0;
    const float next_alpha = accepted.has_plastic_history ? accepted.equivalent_plastic_strain : alpha;
    if (nk::material::EvaluateMpmStoredEnergy(material, accepted.elastic_f, next_alpha, accepted_energy) !=
            nk::material::ConstitutiveStatus::Ok ||
        nk::material::EvaluateMpmStoredEnergy(material, d.particle_F + base,
            d.particle_plastic[particle], actual_energy) != nk::material::ConstitutiveStatus::Ok) return false;
    *averaging_work = d.particle_vol0[particle] * (actual_energy - accepted_energy);
    *plastic_loss = 0.0;
    if (material.model_kind == nk::MpmMaterial::kHenckyJ2 || material.model_kind == 4.0f) {
        float map[9], unreturned[9];
        for (int k = 0; k < 9; ++k) map[k] = p.dt * affine[k] + (k % 4 == 0 ? 1.0f : 0.0f);
        nk::material::mpm_detail::Mat3Mul<EnergyMpmArithmetic>(map, initial, unreturned);
        double trial_energy = 0.0;
        if (nk::material::EvaluateMpmStoredEnergy(material, unreturned, alpha, trial_energy) !=
            nk::material::ConstitutiveStatus::Ok) return false;
        *plastic_loss = d.particle_vol0[particle] * (trial_energy - accepted_energy);
    }
    return true;
}

// Skip cleared contact rows while retaining each thread's row order and noncontact assignments.
__device__ uint32_t LedgerRow(uint32_t row, uint32_t bound, uint32_t contacts) {
    if (row < contacts && row >= bound)
        row += ((contacts - row - 1u) / blockDim.x + 1u) * blockDim.x;
    return row;
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
        Vec3 angular{}, linear{}, no_angular{}, no_linear{};
        const bool projected = p.stage == EnergyStage::Projected && p.pos_pass;
        if (p.stage == EnergyStage::Begin) d.energy_link_start_rotation[link] = d.joint_rotation[link];
        if (parent == ~0u && type == ArticulationJointType::FloatingBase) {
            pose = unprojected = d.base_pose[art];
            if (p.stage == EnergyStage::Begin) d.energy_base_start_pose[art] = pose;
            const auto velocity = d.link_velocity[link];
            angular = mg::RotateByQuatNormalized(pose.rotation,
                {velocity.v[0], velocity.v[1], velocity.v[2]});
            linear = mg::RotateByQuatNormalized(pose.rotation,
                {velocity.v[3], velocity.v[4], velocity.v[5]});
            no_angular = angular;
            no_linear = linear;
            if (projected) {
                unprojected = d.energy_base_start_pose[art];
                unprojected.position += linear * p.dt;
                const float h = 0.5f * p.dt;
                unprojected.rotation = mg::QuatNormalizeRsqrt(mg::QuatMul(unprojected.rotation,
                    mg::MakeQuat(1.0f, h * velocity.v[0], h * velocity.v[1], h * velocity.v[2])), 1.0e-12f);
                no_angular = mg::RotateByQuatNormalized(unprojected.rotation,
                    {velocity.v[0], velocity.v[1], velocity.v[2]});
            }
        } else {
            const float coordinate = d.q[link];
            const float no_position = projected ? d.energy_link_start_q[link] + p.dt * d.qdot[link] : coordinate;
            const math::Quat rotation = d.joint_rotation[link];
            math::Quat no_turn = rotation;
            if (type == ArticulationJointType::Spherical && projected) {
                const auto velocity = d.joint_angular_velocity[link];
                const float h = 0.5f * p.dt;
                no_turn = mg::QuatNormalizeRsqrt(
                    mg::QuatMul(d.energy_link_start_rotation[link],
                        mg::MakeQuat(1.0f, h * velocity.x, h * velocity.y, h * velocity.z)), 1.0e-12f);
            }
            pose = JointRelativeFrame(type, m.joint_axis[link], m.link_local_pose[link],
                                       m.parent_offset[link], coordinate, rotation);
            unprojected = JointRelativeFrame(type, m.joint_axis[link], m.link_local_pose[link],
                                              m.parent_offset[link], no_position, no_turn);
            if (parent != ~0u) {
                const uint32_t parent_link = offset + parent;
                const auto pv = d.energy_link_velocity[parent_link];
                const auto npv = d.energy_link_unprojected_velocity[parent_link];
                angular = {pv.v[0], pv.v[1], pv.v[2]};
                no_angular = {npv.v[0], npv.v[1], npv.v[2]};
                pose = ComposeFrame(d.energy_link_pose[parent_link], pose);
                unprojected = ComposeFrame(d.energy_link_unprojected_pose[parent_link], unprojected);
                linear = Vec3{pv.v[3], pv.v[4], pv.v[5]} +
                    angular.Cross(pose.position - d.energy_link_pose[parent_link].position);
                no_linear = Vec3{npv.v[3], npv.v[4], npv.v[5]} + no_angular.Cross(
                    unprojected.position - d.energy_link_unprojected_pose[parent_link].position);
            }
            const Vec3 axis = mg::RotateByQuatNormalized(pose.rotation, JointUnitAxis(m.joint_axis[link]));
            const Vec3 no_axis = mg::RotateByQuatNormalized(unprojected.rotation, JointUnitAxis(m.joint_axis[link]));
            if (type == ArticulationJointType::Revolute) angular += axis * d.qdot[link];
            if (type == ArticulationJointType::Revolute) no_angular += no_axis * d.qdot[link];
            if (type == ArticulationJointType::Prismatic) linear += axis * d.qdot[link];
            if (type == ArticulationJointType::Prismatic) no_linear += no_axis * d.qdot[link];
            if (type == ArticulationJointType::Spherical) {
                angular += mg::RotateByQuatNormalized(pose.rotation, d.joint_angular_velocity[link]);
                no_angular += mg::RotateByQuatNormalized(unprojected.rotation, d.joint_angular_velocity[link]);
            }
        }
        d.energy_link_pose[link] = pose;
        d.energy_link_unprojected_pose[link] = unprojected;
        d.energy_link_velocity[link] = {{angular.x, angular.y, angular.z, linear.x, linear.y, linear.z}};
        d.energy_link_unprojected_velocity[link] = {{no_angular.x, no_angular.y, no_angular.z,
            no_linear.x, no_linear.y, no_linear.z}};
    }
}

struct RowPower {
    double physical = 0.0;
    double reference = 0.0;
    double rate = 0.0;
    double applied = 0.0;
    double kinematic = 0.0;
    double grid_boundary = 0.0;
};

// Row power at the interval mean velocity, so the gravity integrated before the solve does no row work.
__device__ RowPower SidePower(const NkRowSide& side, bool second, uint32_t row,
    const ModelView&, const DataView& d, const ReadoutEnergyLedgerParams& p, uint32_t material_axis = 0u) {
    RowPower out;
    if (side.kind == nk::kNkSideArtic) {
        const float* jacobian = (second ? d.chain_jacobian_b : d.chain_jacobian) +
                                size_t{row} * p.dofs_per_artic;
        const size_t tile = size_t{side.index} * p.dofs_per_artic;
        for (uint32_t k = 0u; k < p.dofs_per_artic; ++k) {
            out.physical += double(jacobian[k]) * 0.5 * (d.step_qdot_flat[tile + k] + d.qdot_flat[tile + k]);
            out.reference += double(jacobian[k]) * d.step_qdot_flat[tile + k];
        }
        out.rate = out.applied = out.physical;
    } else if (side.kind == nk::kNkSideRigid) {
        const uint32_t body = side.index;
        out.physical = 0.5 * (Dot(side.jlin, d.step_body_linear_velocity[body] + d.body_linear_velocity[body]) +
            Dot(side.jang, d.step_body_angular_velocity[body] + d.body_angular_velocity[body]));
        out.reference = Dot(side.jlin, d.step_body_linear_velocity[body]) +
            Dot(side.jang, d.step_body_angular_velocity[body]);
        out.rate = out.applied = out.physical;
    } else if (PointMassView::IsPointSide(side.kind)) {
        PointMassView points;
        points.ranges = d.point_endpoint_ranges;
        points.terms = d.point_endpoint_terms;
        for (uint32_t k = 0u; k < points.Count(side); ++k) {
            const auto term = points.At(side, k);
            if (term.kind == nk::kNkSideGrid) {
                Vec3 jacobian = term.jacobian;
                if (material_axis > 0u) {
                    float components[3];
                    nk::material::MpmDeviatorGradient(material_axis - 1u,
                        jacobian.x, jacobian.y, jacobian.z, components);
                    jacobian = {components[0], components[1], components[2]};
                }
                const uint32_t node = term.index;
                if (!(d.grid_inv_mass[node] > 0.0f) && d.energy_grid_prescribed[node] == 0u) continue;
                const double mass = d.grid_mass[node];
                const double* entry = d.energy_grid_entry_momentum + size_t{node} * 3u;
                const auto velocity = d.grid_velocity[node];
                const double power = mass > 0.0 ? 0.5 *
                    (jacobian.x * (entry[0] / mass + velocity.x) +
                     jacobian.y * (entry[1] / mass + velocity.y) +
                     jacobian.z * (entry[2] / mass + velocity.z)) : 0.0;
                out.reference += mass > 0.0 ?
                    (jacobian.x * entry[0] + jacobian.y * entry[1] + jacobian.z * entry[2]) / mass : 0.0;
                out.physical += power;
                out.rate += power;
                if (d.grid_inv_mass[node] > 0.0f) out.applied += power;
                else out.grid_boundary += power;
                continue;
            }
            if (term.kind != nk::kNkSideParticle) continue;
            const uint32_t particle = term.index;
            const double physical = 0.5 * Dot(term.jacobian,
                d.step_particle_velocity[particle] + ParticleVelocity(d, p, particle));
            out.reference += Dot(term.jacobian, d.step_particle_velocity[particle]);
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

struct CompliantShare {
    double force = 0.0;
    double conservative = 0.0;
    double damping = 0.0;
    double response = 0.0;
};

__device__ bool CompliantSpringRow(const NkRow& r, const DataView& d, uint32_t row) {
    constexpr uint32_t excluded = nk::nk_row_flags::kSpeculative | nk::nk_row_flags::kVelocityOnly |
        nk::nk_row_flags::kBlockTangent | nk::nk_row_flags::kMaterialBlock;
    return d.row_compliant_storage != nullptr && (r.flags & nk::nk_row_flags::kMuJoCo) && !(r.flags & excluded) &&
           d.row_compliant_storage[size_t{row} * 4u] > 0.0f;
}

// Negated conservative force at the midpoint of the solved displacement, with the active set
// responding to both damping and the change in spring excitation.
__device__ CompliantShare CompliantStorageForce(const NkRow& r, uint32_t row, double impulse,
                                                const DataView& d, const ReadoutEnergyLedgerParams& p) {
    CompliantShare out;
    if (!CompliantSpringRow(r, d, row) || !(p.dt > 0.0f)) return out;
    const float* stored = d.row_compliant_storage + size_t{row} * 4u;
    const double damping_impulse = stored[2];
    out.response = stored[3];
    const double inverse_mass = out.response + double(r.compliance_alpha);
    if (!(impulse > 0.0) || !(inverse_mass > 0.0)) return out;
    out.damping = damping_impulse;
    out.conservative = impulse - damping_impulse;
    out.force = -out.conservative / double(p.dt) + 0.5 * stored[1];
    return out;
}

// Rows of an env's damping split: solved rows strictly inside their bounds and friction cones.
__device__ bool DampingSplitMember(const NkRow* rows, const float* lambda, uint32_t row) {
    const NkRow& r = rows[row];
    if (!(r.flags & nk::nk_row_flags::kActive) || (r.flags & nk::nk_row_flags::kMaterialBlock)) return false;
    const float impulse = lambda[row];
    const float tolerance = 1e-6f * fmaxf(1.0f, fabsf(impulse));
    if (impulse == 0.0f || !(impulse > r.lower + tolerance) || !(impulse < r.upper - tolerance)) return false;
    if (!(r.flags & nk::nk_row_flags::kBlockTangent)) return true;
    const uint32_t count = r.group_normal_count;
    const float primary = r.mu, secondary = r.friction_secondary > 0.0f ? r.friction_secondary : r.mu;
    if (count == 0u || !(primary > 0.0f) || !(secondary > 0.0f)) return false;
    const uint32_t normal = r.group_first + (row - r.group_first) % count;
    return hypotf(lambda[normal + count] / primary, lambda[normal + 2u * count] / secondary) <
           (1.0f - 1e-4f) * lambda[normal];
}

__device__ RowForward StoredForward(const NkRow& r, const float* base, uint32_t dofs) {
    RowForward forward;
    if (r.a.kind == nk::kNkSideArtic) forward.side[0] = base;
    if (r.b.kind == nk::kNkSideArtic) forward.side[1] = base + dofs;
    return forward;
}

__device__ double SideDisplacementRate(const NkRowSide& side, bool second, uint32_t row,
    const DataView& d, const ReadoutEnergyLedgerParams& p);

// Independent response components share a factor buffer; each factor splits damping and spring excitation.
// Dependent rows carry zero force in the semidefinite solve.
__global__ void EnergyDampingSplitKernel(DataView d, ReadoutEnergyLedgerParams p) {
    constexpr uint32_t capacity = nk::kEnergySplitCapacity;
    constexpr size_t double_words = sizeof(double) / sizeof(uint32_t);
    __shared__ uint32_t members[capacity], small_members[capacity];
    __shared__ double rhs[capacity], original[capacity];
    __shared__ uint32_t dropped[capacity], warp_counts[kThreads / 32u];
    __shared__ uint32_t total, component_count, selected;
    const uint32_t env = blockIdx.x;
    if (env >= p.env_count) return;
    const auto* rows = reinterpret_cast<const NkRow*>(d.urows);
    float* const storage = d.row_compliant_storage;
    const uint32_t first = env * p.rows_per_env, dofs = p.dofs_per_artic;
    const uint32_t bound = ContactRowBound(d.contact_row_extent, env, p.contact_rows);
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    // A larger arena holds the active list at its tail, beyond a capacity-squared matrix prefix.
    uint32_t* const active = p.rows_per_env > capacity
        ? reinterpret_cast<uint32_t*>(d.energy_split_matrix + size_t{first} * capacity) +
              size_t{p.rows_per_env} * (double_words * capacity - 1u)
        : small_members;
    if (threadIdx.x == 0u) total = 0u;
    __syncthreads();
    bool spring = false;
    for (uint32_t base = 0u; base < p.rows_per_env; base += blockDim.x) {
        const uint32_t local = base + threadIdx.x, row = first + local;
        const bool scanned = local < p.rows_per_env && (local >= p.contact_rows || local < bound);
        const bool member = scanned && DampingSplitMember(rows, d.lambda, row);
        if (scanned && !member) {
            storage[size_t{row} * 4u + 1u] = 0.0f;
            storage[size_t{row} * 4u + 2u] = 0.0f;
        }
        spring |= member && CompliantSpringRow(rows[row], d, row) && d.lambda[row] > 0.0f;
        const uint32_t ballot = __ballot_sync(0xffffffffu, member);
        if (lane == 0u) warp_counts[warp] = __popc(ballot);
        __syncthreads();
        uint32_t offset = total + __popc(ballot & ((1u << lane) - 1u));
        for (uint32_t w = 0u; w < warp; ++w) offset += warp_counts[w];
        if (member) active[offset] = row;
        __syncthreads();
        if (threadIdx.x == 0u)
            for (uint32_t w = 0u; w < blockDim.x / 32u; ++w) total += warp_counts[w];
        __syncthreads();
    }
    if (!__syncthreads_or(spring)) return;
    const auto* mass_factor = static_cast<const float*>(d.m_factor);
    const auto* inv_mass = static_cast<const float*>(d.body_inv_mass);
    const auto* inv_inertia = static_cast<const math::SymmetricMat3*>(d.body_world_inv_inertia);
    const auto points = PointMasses(d);
    for (uint32_t k = threadIdx.x; k < total; k += blockDim.x) {
        const uint32_t row = active[k];
        if (dofs > 0u)
            ForwardRowSides(rows[row], row, d.chain_jacobian, d.chain_jacobian_b, mass_factor, dofs,
                            d.energy_split_forward + size_t{row} * 2u * dofs);
    }
    __syncthreads();
    for (uint32_t seed = 0u; seed < total; ++seed) {
        if (threadIdx.x == 0u) selected = active[seed];
        __syncthreads();
        if (selected == ~0u) continue;
        if (threadIdx.x == 0u) {
            members[0] = selected;
            active[seed] = ~0u;
            component_count = 1u;
        }
        __syncthreads();
        for (uint32_t front = 0u; front < component_count; ++front) {
            const uint32_t ri = members[front];
            const RowForward fi = StoredForward(rows[ri], d.energy_split_forward + size_t{ri} * 2u * dofs, dofs);
            bool finite = true;
            for (uint32_t at = threadIdx.x; at < total; at += blockDim.x) {
                const uint32_t rj = active[at];
                if (rj == ~0u) continue;
                const RowForward fj = StoredForward(rows[rj], d.energy_split_forward + size_t{rj} * 2u * dofs, dofs);
                const double coupling = ri >= rj
                    ? PairDrivenRowCoupling(rows[ri], fi, rows[rj], fj, mass_factor, inv_mass, inv_inertia, points, dofs)
                    : PairDrivenRowCoupling(rows[rj], fj, rows[ri], fi, mass_factor, inv_mass, inv_inertia, points, dofs);
                finite &= isfinite(coupling);
                if (coupling == 0.0) continue;
                const uint32_t index = atomicAdd(&component_count, 1u);
                if (index < capacity) members[index] = rj;
                active[at] = ~0u;
            }
            const bool invalid = __syncthreads_or(!finite) != 0;
            if (component_count > capacity || invalid) {
                for (uint32_t local = threadIdx.x; local < p.rows_per_env; local += blockDim.x)
                    storage[size_t{first + local} * 4u] = 0.0f;
                if (threadIdx.x == 0u)
                    atomicOr(d.energy_ledger_status + size_t{env} * p.substeps + p.slot,
                        invalid ? nk::energy_status::kNonfinite : nk::energy_status::kStorageCapacity);
                return;
            }
        }
        const uint32_t n = component_count;
        // Slot order makes the factor independent of the discovery order across threads.
        if (threadIdx.x == 0u)
            for (uint32_t i = 1u; i < n; ++i) {
                const uint32_t value = members[i];
                uint32_t j = i;
                for (; j > 0u && members[j - 1u] > value; --j) members[j] = members[j - 1u];
                members[j] = value;
            }
        __syncthreads();
        for (uint32_t k = threadIdx.x; k < n; k += blockDim.x) {
            const uint32_t row = members[k];
            rhs[k] = (rows[row].flags & nk::nk_row_flags::kMuJoCo) ? double(storage[size_t{row} * 4u + 2u]) : 0.0;
        }
        __syncthreads();
        // Each connected response block uses the environment's matrix prefix.
        double* const a = d.energy_split_matrix + size_t{first} * capacity;
        for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
            const uint32_t ri = members[i];
            const RowForward fi = StoredForward(rows[ri], d.energy_split_forward + size_t{ri} * 2u * dofs, dofs);
            for (uint32_t j = 0u; j <= i; ++j) {
                const uint32_t rj = members[j];
                const RowForward fj = StoredForward(rows[rj], d.energy_split_forward + size_t{rj} * 2u * dofs, dofs);
                double coupling = PairDrivenRowCoupling(rows[ri], fi, rows[rj], fj, mass_factor, inv_mass, inv_inertia,
                                                        points, dofs);
                if (i == j) coupling += rows[ri].compliance_alpha;
                a[size_t{i} * n + j] = coupling;
            }
            original[i] = a[size_t{i} * n + i];
        }
        __syncthreads();
        for (uint32_t k = 0u; k < n; ++k) {
            if (threadIdx.x == 0u) {
                const double pivot = a[size_t{k} * n + k];
                dropped[k] = pivot > 1e-9 * original[k] && pivot > 0.0 ? 0u : 1u;
                a[size_t{k} * n + k] = dropped[k] ? 1.0 : sqrt(pivot);
            }
            __syncthreads();
            const double diagonal = a[size_t{k} * n + k];
            for (uint32_t i = k + 1u + threadIdx.x; i < n; i += blockDim.x)
                a[size_t{i} * n + k] = dropped[k] ? 0.0 : a[size_t{i} * n + k] / diagonal;
            __syncthreads();
            for (uint32_t i = k + 1u + threadIdx.x; i < n; i += blockDim.x) {
                const double lik = a[size_t{i} * n + k];
                for (uint32_t j = k + 1u; j <= i; ++j) a[size_t{i} * n + j] -= lik * a[size_t{j} * n + k];
            }
            __syncthreads();
        }
        // The pivot reference diagonals are no longer needed after factorization.
        double* const spring_change = original;
        for (uint32_t k = threadIdx.x; k < n; k += blockDim.x) {
            const uint32_t row = members[k];
            const double displacement = double(p.dt) * (SideDisplacementRate(rows[row].a, false, row, d, p) +
                SideDisplacementRate(rows[row].b, true, row, d, p));
            spring_change[k] = CompliantSpringRow(rows[row], d, row) ? storage[size_t{row} * 4u] * displacement : 0.0;
        }
        __syncthreads();
        for (uint32_t k = 0u; k < n; ++k) {
            if (threadIdx.x == 0u) {
                rhs[k] = dropped[k] ? 0.0 : rhs[k] / a[size_t{k} * n + k];
                spring_change[k] = dropped[k] ? 0.0 : spring_change[k] / a[size_t{k} * n + k];
            }
            __syncthreads();
            for (uint32_t i = k + 1u + threadIdx.x; i < n; i += blockDim.x) {
                rhs[i] -= a[size_t{i} * n + k] * rhs[k];
                spring_change[i] -= a[size_t{i} * n + k] * spring_change[k];
            }
            __syncthreads();
        }
        for (uint32_t k = n; k > 0u; --k) {
            const uint32_t c = k - 1u;
            if (threadIdx.x == 0u) {
                rhs[c] = dropped[c] ? 0.0 : rhs[c] / a[size_t{c} * n + c];
                spring_change[c] = dropped[c] ? 0.0 : spring_change[c] / a[size_t{c} * n + c];
            }
            __syncthreads();
            for (uint32_t i = threadIdx.x; i < c; i += blockDim.x) {
                rhs[i] -= a[size_t{c} * n + i] * rhs[c];
                spring_change[i] -= a[size_t{c} * n + i] * spring_change[c];
            }
            __syncthreads();
        }
        for (uint32_t k = threadIdx.x; k < n; k += blockDim.x) {
            storage[size_t{members[k]} * 4u + 1u] = float(spring_change[k]);
            storage[size_t{members[k]} * 4u + 2u] = float(rhs[k] * double(p.dt));
        }
        __syncthreads();
    }
}

// Writes one row's solved-stage ledger terms when the per-row readout is allocated.
__device__ void RecordRowTerms(const DataView& d, uint32_t row, nk::EnergyRowKind kind, double impulse,
                               double work, double storage, double rate, const CompliantShare& share,
                               float compliance) {
    if (d.energy_row_terms == nullptr) return;
    float* terms = d.energy_row_terms + size_t{row} * nk::kEnergyRowTermCount;
    const float values[nk::kEnergyRowTermCount] = {float(static_cast<uint32_t>(kind)), float(impulse),
        float(work), float(storage), float(rate), float(share.conservative), float(share.response), compliance};
    for (uint32_t k = 0u; k < nk::kEnergyRowTermCount; ++k) terms[k] = values[k];
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
            else if (term.kind == nk::kNkSideGrid) rate += Dot(term.jacobian, d.grid_velocity[term.index]);
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

__device__ KinematicElasticResponse KinematicElasticReaction(const ModelView& m, const DataView& d,
    const ReadoutEnergyLedgerParams& p, uint32_t particle) {
    if (!IsVertex(p, particle)) return {};
    const uint32_t vertex = particle % p.particles_per_env - p.vbd_begin;
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
        if (element.damping > 0.0f) {
            nk::vbd::ElementGeometry frozen;
            Vec3 rates[4];
            for (uint32_t j = 0u; j < nk::VbdElementVertexCount(element.kind); ++j) {
                const uint32_t point = base + element.vertex[j];
                frozen.start[j] = d.particle_prev_pos[point];
                rates[j] = ParticleVelocity(d, p, point);
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

__global__ void EnergyLedgerKernel(ModelView m, DataView d, ReadoutEnergyLedgerParams p) {
    const uint32_t env = blockIdx.x;
    if (env >= p.env_count) return;
    const size_t record = size_t{env} * p.substeps + p.slot;
    float* output = d.energy_ledger + record * kColumns;
    double* sum = d.energy_ledger_sum + record * kColumns;
    uint32_t* coverage = d.energy_ledger_status + record;
    if (p.stage == EnergyStage::Begin) {
        if (threadIdx.x == 0u) {
            *coverage = p.coverage_status;
            d.vbd_force_residual_work[record] = 0.0f;
        }
        for (uint32_t c = threadIdx.x; c < kColumns; c += blockDim.x) {
            output[c] = 0.0f;
            sum[c] = 0.0;
        }
    }
    __syncthreads();
    double values[kReductionColumns]{};
    const auto add = [&](EnergyColumn column, double value) {
        const uint32_t c = Column(column);
        values[c] += value;
        if (c >= Column(EnergyColumn::DriveWork) && c <= Column(EnergyColumn::ExternalWork))
            values[Column(EnergyColumn::Throughput)] += fabs(value);
        if (c >= Column(EnergyColumn::FrictionLoss) && c <= Column(EnergyColumn::RayleighLoss))
            values[Column(EnergyColumn::Throughput)] += fabs(value);
    };
    double kinetic = 0.0, gravity = 0.0, elastic = 0.0, position = 0.0, position_kinetic = 0.0;
    const Vec3 g{p.gravity[0], p.gravity[1], p.gravity[2]};
    for (uint32_t local = threadIdx.x; local < p.particles_per_env; local += blockDim.x) {
        const uint32_t particle = env * p.particles_per_env + local;
        if (p.stage == EnergyStage::Begin) d.energy_particle_start[particle] = d.particle_pos[particle];
        const bool grid_particle = local < p.mpm_particles_per_env;
        if (grid_particle && p.stage == EnergyStage::Begin) {
            for (uint32_t k = 0u; k < 9u; ++k) {
                d.energy_mpm_start_F[size_t{particle} * 9u + k] = d.particle_F[size_t{particle} * 9u + k];
                if (d.energy_mpm_start_plastic_F)
                    d.energy_mpm_start_plastic_F[size_t{particle} * 9u + k] = d.particle_plastic_F[size_t{particle} * 9u + k];
            }
            d.energy_mpm_start_plastic[particle] = d.particle_plastic[particle];
        }
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
        if (grid_particle) {
            if (p.stage == EnergyStage::Free) {
                add(EnergyColumn::MpmP2gLoss, MpmTransferLoss(d, p, particle,
                    d.particle_pos[particle], velocity, d.particle_C + size_t{particle} * 9u, true));
                continue;
            }
            if (p.stage == EnergyStage::Solved) continue;
            if (d.particle_material_id[particle] >= p.mpm_material_count) {
                atomicOr(coverage, nk::energy_status::kGridMaterial);
                continue;
            }
            const auto material = EnergyMpmMaterial(d, particle);
            double stored = 0.0;
            if (nk::material::EvaluateMpmStoredEnergy(material, d.particle_F + size_t{particle} * 9u,
                d.particle_plastic[particle], stored) != nk::material::ConstitutiveStatus::Ok)
                atomicOr(coverage, nk::energy_status::kGridMaterial);
            else elastic += d.particle_vol0[particle] * stored;
            nk::MpmCompactAxis axes[3];
            MpmEnergyAxes(d.particle_pos[particle], p, axes);
            const float* affine = d.particle_C + size_t{particle} * 9u;
            const double affine_kinetic = nk::material::MpmAffineKinetic(affine, mass, axes, p.grid_dx);
            kinetic += affine_kinetic;
            if (p.stage == EnergyStage::Begin) {
                add(EnergyColumn::MpmBeginAffineKinetic, affine_kinetic);
            } else if (p.stage == EnergyStage::End) add(EnergyColumn::MpmEndAffineKinetic, affine_kinetic);
            else if (p.stage == EnergyStage::Projected) {
                MpmEnergyAxes(d.energy_particle_start[particle], p, axes);
                const double gathered_affine = nk::material::MpmAffineKinetic(affine, mass, axes, p.grid_dx);
                add(EnergyColumn::MpmG2pLoss, MpmTransferLoss(d, p, particle,
                    d.energy_particle_start[particle], velocity, affine, false));
                add(EnergyColumn::MpmAffineTransport, affine_kinetic - gathered_affine);
                double plastic_loss = 0.0, averaging_work = 0.0;
                if (!MpmReturnEnergy(d, p, particle, material, &plastic_loss, &averaging_work))
                    atomicOr(coverage, nk::energy_status::kGridMaterial);
                else {
                    add(EnergyColumn::MpmPlasticLoss, plastic_loss);
                    add(EnergyColumn::MpmVolumeAverageWork, averaging_work);
                }
            }
        }
        if (p.stage == EnergyStage::Solved && IsVertex(p, particle)) {
            const uint32_t vertex = local - p.vbd_begin;
            const uint32_t slot = env * p.vbd_vertices + vertex;
            VertexBlockView block;
            block.elements = m.vbd_elements;
            block.membrane_start = d.vbd_membrane_start;
            block.offsets = m.vbd_incidence_offsets;
            block.incidence = m.vbd_incidence;
            block.start = d.particle_prev_pos;
            block.free_rate = d.vbd_free_rate;
            block.free_velocity = d.vbd_free_velocity;
            block.inertia = d.vbd_inertia;
            block.effective_step = d.vbd_step;
            block.layout.begin = p.vbd_begin;
            block.layout.vertices = p.vbd_vertices;
            block.layout.elements = p.vbd_elements;
            block.layout.particles_per_env = p.particles_per_env;
            block.layout.env_count = p.env_count;
            block.dt = p.dt;
            const Vec3 displacement = d.particle_vel[particle] * p.dt;
            const Vec3 force = VertexBlockForceDouble(block, d.particle_vel, env, vertex,
                nk::vbd::InertialGradient(d.particle_vel[particle], d.vbd_free_rate[slot], d.vbd_inertia[slot], p.dt),
                d.particle_row_impulse[particle] / p.dt);
            values[kForceWorkColumn] -= Dot(force, displacement);
        }
        kinetic += 0.5 * mass * Dot(velocity, velocity);
        gravity -= mass * Dot(g, d.particle_pos[particle]);
        if (p.stage == EnergyStage::Projected && p.pos_pass) {
            const Vec3 correction = grid_particle
                ? d.particle_pos[particle] - d.energy_particle_start[particle] - velocity * p.dt
                : d.particle_pseudo_vel[particle] * p.dt;
            position -= mass * Dot(g, correction);
        }
        if (p.stage == EnergyStage::End && p.has_aero)
            add(EnergyColumn::AeroWork, Dot(d.energy_aero_impulse[particle],
                d.particle_pos[particle] - d.energy_particle_start[particle]) / p.dt);
    }
    for (uint32_t local = threadIdx.x; local < p.grid_nodes_per_env; local += blockDim.x) {
        if (p.stage != EnergyStage::Free && p.stage != EnergyStage::Solved) break;
        const uint32_t node = env * p.grid_nodes_per_env + local;
        const double mass = d.grid_mass[node];
        if (!(mass > 0.0)) continue;
        if (!(d.grid_inv_mass[node] > 0.0f) && d.energy_grid_prescribed[node] == 0u) continue;
        const Vec3 velocity = d.grid_velocity[node];
        kinetic += 0.5 * mass * Dot(velocity, velocity);
        const double* initial = d.energy_grid_entry_momentum + size_t{node} * 3u;
        const double before[3] = {initial[0] / mass, initial[1] / mass, initial[2] / mass};
        if (p.stage == EnergyStage::Free) {
            const double entry_kinetic = 0.5 * mass *
                (before[0] * before[0] + before[1] * before[1] + before[2] * before[2]);
            add(EnergyColumn::MpmGridBeginKinetic, entry_kinetic);
            continue;
        }
        const double solved_kinetic = 0.5 * mass * Dot(velocity, velocity);
        add(EnergyColumn::MpmGridSolvedKinetic, solved_kinetic);
        const Vec3 free = d.energy_grid_free_velocity[node];
        const float solved[3] = {velocity.x, velocity.y, velocity.z};
        const float predicted[3] = {free.x, free.y, free.z};
        double stress_work = 0.0, viscous_work = 0.0, boundary_work = 0.0;
        for (uint32_t a = 0u; a < 3u; ++a) {
            const size_t at = size_t{node} * 3u + a;
            const double mean = 0.5 * (before[a] + solved[a]);
            stress_work += d.energy_grid_stress_impulse[at] * mean;
            viscous_work += d.energy_grid_viscous_impulse[at] * mean;
            const double boundary = mass * (predicted[a] - before[a] - double(p.gravity[a]) * p.dt) -
                d.energy_grid_stress_impulse[at];
            if (d.energy_grid_prescribed[node] != 0u) boundary_work += boundary * mean;
        }
        add(EnergyColumn::MpmStressWork, stress_work - viscous_work);
        add(EnergyColumn::MpmViscousLoss, -viscous_work);
        add(EnergyColumn::MpmBoundaryWork, boundary_work);
    }
    for (uint32_t local = threadIdx.x; local < p.bodies_per_env; local += blockDim.x) {
        const uint32_t body = env * p.bodies_per_env + local;
        if (m.body_to_link && m.body_to_link[body] != ~0u) continue;
        const float inverse_mass = d.body_inv_mass[body];
        if (!(inverse_mass > 0.0f)) continue;
        const auto twist = ComTwist(d.body_angular_velocity[body], d.body_linear_velocity[body]);
        if (p.stage == EnergyStage::Begin) {
            const Vec3 force = d.body_force[body] - g * (d.body_gravity_compensation[body] / inverse_mass);
            const Vec3 torque = d.body_torque[body];
            d.energy_body_wrench[body] = {{force.x, force.y, force.z, torque.x, torque.y, torque.z}};
            d.energy_body_start_velocity[body] = twist;
        } else if (p.stage == EnergyStage::Solved) {
            add(EnergyColumn::ExternalWork, LoadWork(d.energy_body_wrench[body],
                d.energy_body_start_velocity[body], twist, p.dt));
        }
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
            const Vec3 pseudo = d.body_pseudo_angular_velocity[body];
            if (pseudo.LengthSq() != 0.0f) {
                const float h = -0.5f * p.dt;
                const auto no_rotation = mg::QuatNormalizeRsqrt(mg::QuatMul(
                    mg::MakeQuat(1.0f, h * pseudo.x, h * pseudo.y, h * pseudo.z),
                    d.body_pose[body].rotation), 1.0e-12f);
                const auto no_principal = mg::QuatNormalizeRsqrt(mg::QuatMul(no_rotation,
                    d.body_inertial_frame[body].rotation), 1.0e-12f);
                const Vec3 momentum = mg::RotateByQuatNormalized(principal.rotation,
                    {inverse_inertia.x > 0.0f ? angular.x / inverse_inertia.x : 0.0f,
                     inverse_inertia.y > 0.0f ? angular.y / inverse_inertia.y : 0.0f,
                     inverse_inertia.z > 0.0f ? angular.z / inverse_inertia.z : 0.0f});
                const Vec3 local = InverseRotate(no_principal, momentum);
                const double before = 0.5 * (double(local.x) * local.x * inverse_inertia.x +
                    double(local.y) * local.y * inverse_inertia.y + double(local.z) * local.z * inverse_inertia.z);
                double after = 0.0;
                if (inverse_inertia.x > 0.0f) after += 0.5 * double(angular.x) * angular.x / inverse_inertia.x;
                if (inverse_inertia.y > 0.0f) after += 0.5 * double(angular.y) * angular.y / inverse_inertia.y;
                if (inverse_inertia.z > 0.0f) after += 0.5 * double(angular.z) * angular.z / inverse_inertia.z;
                position_kinetic += after - before;
            }
        }
    }
    for (uint32_t local = threadIdx.x; local < p.links_per_env; local += blockDim.x) {
        const uint32_t link = env * p.links_per_env + local;
        if (p.stage == EnergyStage::Begin) {
            d.energy_link_start_q[link] = d.q[link];
            d.energy_link_start_qdot[link] = d.qdot[link];
            d.energy_joint_start_angular_velocity[link] = d.joint_angular_velocity[link];
            d.energy_link_wrench[link] = d.link_wrench[link];
            if (m.link_body[link] < p.bodies_per_env) {
                const uint32_t body = env * p.bodies_per_env + m.link_body[link];
                const float scale = d.body_gravity_compensation[body] * m.link_inertia[link].m[21];
                d.energy_link_wrench[link].v[0] -= scale * g.x;
                d.energy_link_wrench[link].v[1] -= scale * g.y;
                d.energy_link_wrench[link].v[2] -= scale * g.z;
            }
        }
        const auto pose = d.energy_link_pose[link];
        const auto world = d.energy_link_velocity[link];
        const Vec3 world_angular{world.v[0], world.v[1], world.v[2]};
        const Vec3 offset = mg::RotateByQuatNormalized(pose.rotation, m.link_inertial_frame[link].position);
        const auto twist = ComTwist(world_angular,
            Vec3{world.v[3], world.v[4], world.v[5]} + world_angular.Cross(offset));
        if (p.stage == EnergyStage::Begin) d.energy_link_start_velocity[link] = twist;
        else if (p.stage == EnergyStage::Solved)
            add(EnergyColumn::ExternalWork, LoadWork(d.energy_link_wrench[link],
                d.energy_link_start_velocity[link], twist, p.dt));
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
        const auto joint_type = static_cast<ArticulationJointType>(m.joint_type[link]);
        if (JointDofCountDevice(joint_type) == 1u) {
            if (p.stage == EnergyStage::Solved)
                add(EnergyColumn::ExternalWork, -0.5 * double(p.dt) * d.joint_gravity_force[link] *
                    (double(d.energy_link_start_qdot[link]) + d.qdot[link]));
            kinetic += 0.5 * double(m.joint_armature[link]) * d.qdot[link] * d.qdot[link];
            const double stretch = double(d.q[link]) - m.joint_spring_reference[link];
            const double stiffness = m.joint_stiffness[link];
            elastic += 0.5 * stiffness * stretch * stretch;
            if (p.stage == EnergyStage::Projected && p.pos_pass) {
                const float coordinate = d.energy_link_start_q[link] + p.dt * d.qdot[link];
                const double unprojected = double(coordinate) - m.joint_spring_reference[link];
                position += 0.5 * stiffness * (stretch * stretch - unprojected * unprojected);
            }
        }
        if (joint_type == ArticulationJointType::Spherical)
            kinetic += 0.5 * double(m.joint_armature[link]) *
                Dot(d.joint_angular_velocity[link], d.joint_angular_velocity[link]);
        if (joint_type == ArticulationJointType::FloatingBase && m.joint_armature[link] != 0.0f)
            atomicOr(coverage, nk::energy_status::kArmatureEnergy);
        if (p.stage == EnergyStage::Projected && p.pos_pass) {
            const auto no_pose = d.energy_link_unprojected_pose[link];
            const auto no_position = ComposeFrame(no_pose, m.link_inertial_frame[link]);
            position -= mass * Dot(g, com.position - no_position.position);
            const auto no_world = d.energy_link_unprojected_velocity[link];
            const Vec3 no_angular = InverseRotate(no_pose.rotation, {no_world.v[0], no_world.v[1], no_world.v[2]});
            const Vec3 no_linear = InverseRotate(no_pose.rotation, {no_world.v[3], no_world.v[4], no_world.v[5]});
            const float no_velocity[6] = {no_angular.x, no_angular.y, no_angular.z,
                no_linear.x, no_linear.y, no_linear.z};
            for (uint32_t r = 0u; r < 6u; ++r)
                for (uint32_t c = 0u; c < 6u; ++c)
                    position_kinetic += 0.5 * (double(velocity[r]) * inertia.m[r * 6u + c] * velocity[c] -
                        double(no_velocity[r]) * inertia.m[r * 6u + c] * no_velocity[c]);
        }
        if (p.stage == EnergyStage::End && joint_type == ArticulationJointType::Spherical) {
            const Vec3 velocity = d.joint_angular_velocity[link];
            const Vec3 damping_velocity = p.implicit_passive_damping
                ? velocity : d.energy_joint_start_angular_velocity[link];
            add(EnergyColumn::PassiveLoss, double(m.joint_damping[link]) * p.dt * Dot(damping_velocity, velocity));
        }
        if (p.stage == EnergyStage::End && JointDofCountDevice(joint_type) == 1u) {
            const double displacement = double(d.q[link]) - d.energy_link_start_q[link] -
                (p.pos_pass ? double(p.dt) * d.qdot_pseudo[link] : 0.0);
            const double effort = p.drive_rows ? d.actuator_effort[link] : 0.0;
            add(EnergyColumn::DriveWork, effort * displacement);
            add(EnergyColumn::ExternalWork, (double(d.tau[link]) - effort) * displacement);
            add(EnergyColumn::PassiveLoss, double(m.joint_damping[link]) *
                (p.implicit_passive_damping ? d.qdot[link] : d.energy_link_start_qdot[link]) * displacement);
        }
    }
    for (uint32_t i = threadIdx.x; i < p.vbd_elements; i += blockDim.x) {
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
                rates[j] = ParticleVelocity(d, p, particle);
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
    if (p.stage == EnergyStage::Solved) {
        const uint32_t first_friction = p.contact_rows + p.limit_rows;
        const uint32_t first_drive = first_friction + p.friction_rows;
        const auto* rows = reinterpret_cast<const NkRow*>(d.urows);
        const uint32_t bound = ContactRowBound(d.contact_row_extent, env, p.contact_rows);
        for (uint32_t local = LedgerRow(threadIdx.x, bound, p.contact_rows); local < p.rows_per_env;
             local = LedgerRow(local + blockDim.x, bound, p.contact_rows)) {
            const uint32_t row = env * p.rows_per_env + local;
            const auto r = rows[row];
            if (!(r.flags & nk::nk_row_flags::kActive)) continue;
            if (r.flags & nk::nk_row_flags::kMaterialBlock) {
                for (uint32_t axis = 0u; axis < nk::kMpmStressRowsPerCell; ++axis) {
                    const double impulse = d.lambda[row + axis];
                    if (impulse == 0.0) continue;
                    const auto power = SidePower(r.a, false, row, m, d, p, axis);
                    add(EnergyColumn::MpmStressWork, impulse * power.physical);
                    add(EnergyColumn::RowRateWork, impulse * power.rate);
                    add(EnergyColumn::RowPhysicalImpulseWork, impulse * power.applied);
                    add(EnergyColumn::MpmBoundaryWork, -impulse * power.grid_boundary);
                }
                RecordRowTerms(d, row, nk::EnergyRowKind::Material, d.lambda[row], 0.0, 0.0, 0.0, {}, 0.0f);
                continue;
            }
            const double impulse = d.lambda[row];
            if (impulse == 0.0) continue;
            const auto a = SidePower(r.a, false, row, m, d, p);
            const auto b = SidePower(r.b, true, row, m, d, p);
            const double work = impulse * (a.physical + b.physical);
            add(EnergyColumn::RowRateWork, impulse * (a.rate + b.rate));
            add(EnergyColumn::RowPhysicalImpulseWork, impulse * (a.applied + b.applied));
            add(EnergyColumn::MpmBoundaryWork, -impulse * (a.grid_boundary + b.grid_boundary));
            const auto record = [&](nk::EnergyRowKind kind, double storage, const CompliantShare& share) {
                RecordRowTerms(d, row, kind, impulse, work, storage, a.rate + b.rate, share, r.compliance_alpha);
            };
            if (local >= first_drive && local < first_drive + p.drive_rows) {
                record(nk::EnergyRowKind::Drive, 0.0, {});
                continue;
            }
            // Storage follows the row's own displacement; a pyramid edge splits it into normal and friction parts.
            const auto store = [&](nk::EnergyRowKind kind, double normal_rate, double normal_reference, EnergyColumn loss,
                                   EnergyColumn tangent_loss) {
                const double displacement = double(p.dt) * (SideDisplacementRate(r.a, false, row, d, p) +
                    SideDisplacementRate(r.b, true, row, d, p));
                const auto share = CompliantStorageForce(r, row, impulse, d, p);
                // Explicit damping power uses its reference velocity; its mean-velocity work defect stays in Residual.
                const double normal = share.conservative * normal_rate +
                    share.damping * (normal_rate - normal_reference);
                values[Column(loss)] += normal;
                values[Column(tangent_loss)] += share.conservative * (a.physical + b.physical) +
                    share.damping * (a.physical + b.physical - a.reference - b.reference) - normal;
                values[Column(EnergyColumn::ConstraintStorage)] += share.force * displacement;
                record(kind, share.force * displacement, share);
            };
            if (r.flags & nk::nk_row_flags::kPyramidal) {
                const uint32_t contact = nkops::ContactSlotForRow(d.ucontact_row_range,
                    env * p.contact_slots_per_env, p.contact_slots_per_env, row);
                const auto range = nkops::ContactRowRangeFor(d.ucontact_row_range, contact);
                const uint32_t first = range.first + (row - range.first) % range.points;
                const uint32_t second = first + range.points;
                const auto n0a = SidePower(rows[first].a, false, first, m, d, p);
                const auto n0b = SidePower(rows[first].b, true, first, m, d, p);
                const auto n1a = SidePower(rows[second].a, false, second, m, d, p);
                const auto n1b = SidePower(rows[second].b, true, second, m, d, p);
                const double normal_work = impulse * 0.5 * (n0a.physical + n0b.physical + n1a.physical + n1b.physical);
                add(EnergyColumn::NormalLoss, -normal_work);
                add(EnergyColumn::FrictionLoss, normal_work - work);
                store(nk::EnergyRowKind::PyramidEdge, 0.5 * (n0a.physical + n0b.physical + n1a.physical + n1b.physical),
                      0.5 * (n0a.reference + n0b.reference + n1a.reference + n1b.reference),
                      EnergyColumn::NormalLoss, EnergyColumn::FrictionLoss);
            } else if ((r.flags & nk::nk_row_flags::kBlockTangent) ||
                (local >= first_friction && local < first_drive)) {
                add(EnergyColumn::FrictionLoss, -work);
                record(nk::EnergyRowKind::Friction, 0.0, {});
            } else if (r.flags & (nk::nk_row_flags::kContactNormal | nk::nk_row_flags::kBlockNormal)) {
                add(EnergyColumn::NormalLoss, -work);
                store(nk::EnergyRowKind::Normal, a.physical + b.physical, a.reference + b.reference, EnergyColumn::NormalLoss,
                      EnergyColumn::NormalLoss);
            } else if (r.flags & nk::nk_row_flags::kJointLimit) {
                add(EnergyColumn::LimitLoss, -work);
                store(nk::EnergyRowKind::Limit, a.physical + b.physical, a.reference + b.reference, EnergyColumn::LimitLoss,
                      EnergyColumn::LimitLoss);
            }
            else {
                add(EnergyColumn::UnclassifiedRowWork, work);
                atomicOr(coverage, nk::energy_status::kUnclassifiedRows);
                record(nk::EnergyRowKind::Other, 0.0, {});
            }
        }
        add(EnergyColumn::SolvedKinetic, kinetic);
    } else if (p.stage == EnergyStage::Begin) {
        if (d.energy_row_terms != nullptr)
            for (uint32_t local = threadIdx.x; local < p.rows_per_env * nk::kEnergyRowTermCount; local += blockDim.x)
                d.energy_row_terms[size_t{env} * p.rows_per_env * nk::kEnergyRowTermCount + local] = 0.0f;
        add(EnergyColumn::BeginKinetic, kinetic);
        add(EnergyColumn::BeginGravity, gravity);
        add(EnergyColumn::BeginElastic, elastic);
    } else if (p.stage == EnergyStage::Free) add(EnergyColumn::FreeKinetic, kinetic);
    else if (p.stage == EnergyStage::Projected) {
        add(EnergyColumn::BeforeDatKinetic, kinetic);
        add(EnergyColumn::BeforeDatGravity, gravity);
        add(EnergyColumn::BeforeDatElastic, elastic);
        add(EnergyColumn::PositionPotential, position);
        add(EnergyColumn::PositionKineticWork, position_kinetic);
    } else {
        add(EnergyColumn::EndKinetic, kinetic);
        add(EnergyColumn::EndGravity, gravity);
        add(EnergyColumn::EndElastic, elastic);
    }
    __shared__ double reduction[kThreads];
    for (uint32_t c = 0u; c < kReductionColumns; ++c) {
        reduction[threadIdx.x] = values[c];
        __syncthreads();
        for (uint32_t stride = blockDim.x / 2u; stride > 0u; stride /= 2u) {
            if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0u) {
            if (!isfinite(reduction[0])) atomicOr(coverage, nk::energy_status::kNonfinite);
            if (c == kForceWorkColumn) d.vbd_force_residual_work[record] += float(reduction[0]);
            else output[c] = float(sum[c] += reduction[0]);
        }
        __syncthreads();
    }
    if (p.stage != EnergyStage::End || threadIdx.x != 0u) return;
    const auto get = [&](EnergyColumn column) { return sum[Column(column)]; };
    const auto set = [&](EnergyColumn column, double value) {
        output[Column(column)] = float(sum[Column(column)] = value);
    };
    // Reserved DAT columns stay zero in the discrete contact pipeline.
    set(EnergyColumn::DatKineticLoss, 0.0);
    set(EnergyColumn::DatPotential, 0.0);
    const double change = get(EnergyColumn::EndKinetic) + get(EnergyColumn::EndGravity) + get(EnergyColumn::EndElastic) -
        get(EnergyColumn::BeginKinetic) - get(EnergyColumn::BeginGravity) - get(EnergyColumn::BeginElastic);
    double work = 0.0, losses = 0.0;
    for (uint32_t c = Column(EnergyColumn::DriveWork); c <= Column(EnergyColumn::KinematicContactWork); ++c) {
        work += sum[c];
    }
    for (uint32_t c = Column(EnergyColumn::FrictionLoss); c <= Column(EnergyColumn::RayleighLoss); ++c) {
        losses += sum[c];
    }
    for (EnergyColumn column : {EnergyColumn::MpmP2gLoss, EnergyColumn::MpmG2pLoss,
                              EnergyColumn::MpmPlasticLoss, EnergyColumn::MpmViscousLoss})
        losses += get(column);
    work += get(EnergyColumn::MpmBoundaryWork);
    set(EnergyColumn::Throughput, get(EnergyColumn::Throughput) + fabs(get(EnergyColumn::MpmBoundaryWork)) +
        fabs(get(EnergyColumn::MpmP2gLoss)) + fabs(get(EnergyColumn::MpmG2pLoss)) +
        fabs(get(EnergyColumn::MpmPlasticLoss)) + fabs(get(EnergyColumn::MpmViscousLoss)));
    const double residual = change + get(EnergyColumn::ConstraintStorage) - work + losses +
        get(EnergyColumn::DatKineticLoss) -
        get(EnergyColumn::PositionPotential) - get(EnergyColumn::MpmAffineTransport) -
        get(EnergyColumn::MpmVolumeAverageWork) - get(EnergyColumn::PositionKineticWork);
    output[Column(EnergyColumn::Dt)] = p.dt;
    output[Column(EnergyColumn::Valid)] = 1.0f;
    if (d.env_status[env] != 0u) atomicOr(coverage, nk::energy_status::kPhysicsFailure);
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
    const uint32_t env = blockIdx.x;
    const size_t record = size_t{env} * p.substeps + p.slot;
    __shared__ uint32_t history_restart;
    if (threadIdx.x == 0u)
        history_restart = p.vbd_vertices > 0u ? d.vbd_history_ready[env] & nk::kVbdHistoryRestart : 0u;
    __syncthreads();
    if (p.stage == EnergyStage::Begin && p.vbd_vertices > 0u &&
        (d.vbd_history_ready[env] & nk::kVbdHistoryValid) != 0u)
        for (uint32_t vertex = threadIdx.x; vertex < p.vbd_vertices; vertex += blockDim.x)
            if (d.vbd_restart[env * p.vbd_vertices + vertex] != 0u &&
                d.particle_inv_mass[env * p.particles_per_env + p.vbd_begin + vertex] > 0.0f)
                atomicOr(&history_restart, nk::kVbdHistoryRestart);
    __syncthreads();
    double values[columns]{};
    values[static_cast<uint32_t>(C::MinVbdEffectiveDt)] = DBL_MAX;
    const bool restarted_history = p.stage == EnergyStage::Free && history_restart != 0u;
    double begin_discrete[3]{};
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
    for (uint32_t local = threadIdx.x; local < p.particles_per_env; local += blockDim.x) {
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
        if (local < p.mpm_particles_per_env) {
            if (p.stage == EnergyStage::Free || p.stage == EnergyStage::Solved) continue;
            nk::MpmCompactAxis axes[3];
            MpmEnergyAxes(position, p, axes);
            double spin[3];
            nk::material::MpmAffineAngularMomentum(d.particle_C + size_t{particle} * 9u,
                1.0 / inverse_mass, axes, p.grid_dx, spin);
            for (uint32_t a = 0u; a < 3u; ++a)
                values[static_cast<uint32_t>(C::AngularMomentumX) + a] += spin[a];
        }
        momentum(1.0 / inverse_mass, position, velocity, {});
        if (!IsVertex(p, particle)) continue;
        if (restarted_history) {
            const Vec3 before = d.step_particle_velocity[particle];
            begin_discrete[0] += double(before.x) / inverse_mass;
            begin_discrete[1] += double(before.y) / inverse_mass;
            begin_discrete[2] += double(before.z) / inverse_mass;
        }
        const uint32_t vertex = local - p.vbd_begin;
        const uint32_t slot = env * p.vbd_vertices + vertex;
        const bool second_order = p.stage == EnergyStage::Begin
            ? p.cloth_integrator == 0u && (d.vbd_history_ready[env] & nk::kVbdHistoryValid) != 0u &&
                history_restart == 0u
            : ParticleStep(d, p, particle) < p.dt;
        const Vec3 discrete = second_order ? velocity * 1.5f - d.vbd_history_vel[slot] * 0.5f : velocity;
        vector(C::VbdDiscreteMomentumX, discrete / inverse_mass);
        if (second_order) add(C::VbdBdfParticles, 1.0);
        if (p.stage == EnergyStage::Solved) {
            vector(C::VbdGravityImpulseX, Vec3{p.gravity[0], p.gravity[1], p.gravity[2]} * (p.dt / inverse_mass));
            vector(C::VbdRowImpulseX, d.particle_row_impulse[particle]);
        }
        add(C::VbdDynamicParticles, 1.0);
        if (p.stage == EnergyStage::Begin) continue;
        const float step = ParticleStep(d, p, particle);
        auto& minimum = values[static_cast<uint32_t>(C::MinVbdEffectiveDt)];
        minimum = fmin(minimum, double(step));
        maximum(C::MaxVbdEffectiveDt, step);
        if (p.stage != EnergyStage::Solved) continue;
        VertexBlockView block;
        block.elements = m.vbd_elements;
        block.membrane_start = d.vbd_membrane_start;
        block.offsets = m.vbd_incidence_offsets;
        block.incidence = m.vbd_incidence;
        block.start = d.particle_prev_pos;
        block.free_rate = d.vbd_free_rate;
        block.free_velocity = d.vbd_free_velocity;
        block.inertia = d.vbd_inertia;
        block.effective_step = d.vbd_step;
        block.layout.begin = p.vbd_begin;
        block.layout.vertices = p.vbd_vertices;
        block.layout.elements = p.vbd_elements;
        block.layout.particles_per_env = p.particles_per_env;
        block.layout.env_count = p.env_count;
        block.dt = p.dt;
        const Vec3 displacement = d.particle_vel[particle] * p.dt;
        const Vec3 force = VertexBlockForceDouble(block, d.particle_vel, env, vertex,
            nk::vbd::InertialGradient(d.particle_vel[particle], d.vbd_free_rate[slot], d.vbd_inertia[slot], p.dt),
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
    for (uint32_t local = threadIdx.x; local < p.grid_nodes_per_env; local += blockDim.x) {
        if (p.stage != EnergyStage::Free && p.stage != EnergyStage::Solved) break;
        const uint32_t node = env * p.grid_nodes_per_env + local;
        const uint32_t lattice_nodes = p.grid_nodes_per_env / nk::kMpmLattices;
        const uint32_t lattice = local / lattice_nodes, index = local % lattice_nodes;
        const float shift = nk::MpmLatticeOffset(lattice);
        const Vec3 position{p.grid_origin[0] + (index % p.grid_dims[0] + shift) * p.grid_dx,
            p.grid_origin[1] + ((index / p.grid_dims[0]) % p.grid_dims[1] + shift) * p.grid_dx,
            p.grid_origin[2] + (index / (p.grid_dims[0] * p.grid_dims[1]) + shift) * p.grid_dx};
        const Vec3 velocity = d.grid_velocity[node];
        const float mass = d.grid_mass[node];
        if (!Finite(velocity) || !isfinite(mass)) add(C::NonfiniteParticles, 1.0);
        if (mass > 0.0f) momentum(mass, position, velocity, {});
    }
    for (uint32_t local = threadIdx.x; local < p.bodies_per_env; local += blockDim.x) {
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
    for (uint32_t local = threadIdx.x; local < p.links_per_env; local += blockDim.x) {
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
        if (threadIdx.x == 0u && p.rows_per_env > 0u && d.solver_residual != nullptr) {
            for (uint32_t c = 0u; c < 4u; ++c)
                values[static_cast<uint32_t>(C::SolverContactNormalResidual) + c] = d.solver_residual[env * 4u + c];
            add(C::SolverResidualMeasured, 1.0);
        }
        const auto* rows = reinterpret_cast<const NkRow*>(d.urows);
        const uint32_t first_drive = p.contact_rows + p.limit_rows + p.friction_rows;
        const uint32_t bound = ContactRowBound(d.contact_row_extent, env, p.contact_rows);
        for (uint32_t local = LedgerRow(threadIdx.x, bound, p.contact_rows); local < p.rows_per_env;
             local = LedgerRow(local + blockDim.x, bound, p.contact_rows)) {
            const uint32_t row = env * p.rows_per_env + local;
            const auto r = rows[row];
            if (!(r.flags & nk::nk_row_flags::kActive)) continue;
            add(C::ActiveRows, 1.0);
            if (local >= first_drive && local < first_drive + p.drive_rows) continue;
            if (r.flags & nk::nk_row_flags::kMaterialBlock) continue;
            const auto a = SidePower(r.a, false, row, m, d, p);
            const auto b = SidePower(r.b, true, row, m, d, p);
            const double work = double(d.lambda[row]) * (a.physical + b.physical);
            if (work > 0.0) {
                add(C::EnergyInjectingPassiveRows, 1.0);
                maximum(C::MaxPassiveRowEnergyGain, work);
            }
        }
    }
    __shared__ double reduction[kThreads];
    float* output = d.physics_stage_metrics +
        (record * nk::kEnergyStageCount + static_cast<uint32_t>(p.stage)) * columns;
    for (uint32_t c = 0u; c < columns; ++c) {
        const bool minimum = c == static_cast<uint32_t>(C::MinVbdEffectiveDt);
        const bool maximum = c == static_cast<uint32_t>(C::MaxVbdMomentumVelocityError) ||
            c == static_cast<uint32_t>(C::MaxVbdForce) || c == static_cast<uint32_t>(C::MaxVbdEffectiveDt) ||
            c == static_cast<uint32_t>(C::MaxPassiveRowEnergyGain) ||
            c == static_cast<uint32_t>(C::MaxAbsoluteVertexResidualWork);
        reduction[threadIdx.x] = values[c];
        __syncthreads();
        for (uint32_t stride = blockDim.x / 2u; stride > 0u; stride /= 2u) {
            if (threadIdx.x < stride) {
                const double other = reduction[threadIdx.x + stride];
                double& held = reduction[threadIdx.x];
                held = !isfinite(held) || !isfinite(other) ? double(nanf("")) :
                    minimum ? fmin(held, other) : maximum ? fmax(held, other) : held + other;
            }
            __syncthreads();
        }
        if (threadIdx.x == 0u) {
            output[c] = c == static_cast<uint32_t>(C::Dt) ? p.dt :
                minimum && reduction[0] == DBL_MAX ? 0.0f : float(reduction[0]);
            if (!isfinite(output[c])) atomicOr(d.energy_ledger_status + record, nk::energy_status::kNonfinite);
        }
        __syncthreads();
    }
    // Contact detection resolves the integration rule after the entry sample.
    // A restarted interval uses physical entry momentum in its discrete balance.
    if (restarted_history) {
        float* begin = d.physics_stage_metrics + record * nk::kEnergyStageCount * columns;
        for (uint32_t axis = 0u; axis < 3u; ++axis) {
            reduction[threadIdx.x] = begin_discrete[axis];
            __syncthreads();
            for (uint32_t stride = blockDim.x / 2u; stride > 0u; stride /= 2u) {
                if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
                __syncthreads();
            }
            if (threadIdx.x == 0u)
                begin[static_cast<uint32_t>(C::VbdDiscreteMomentumX) + axis] = float(reduction[0]);
            __syncthreads();
        }
        if (threadIdx.x == 0u) begin[static_cast<uint32_t>(C::VbdBdfParticles)] = 0.0f;
    }
}

__device__ double3 AuditLinearJacobian(const NkRowSide& side, PointMassView points) {
    if (!PointMassView::IsPointSide(side.kind)) return make_double3(side.jlin.x, side.jlin.y, side.jlin.z);
    double3 total{};
    for (uint32_t i = 0u; i < points.Count(side); ++i) {
        const Vec3 term = points.At(side, i).jacobian;
        total.x += term.x; total.y += term.y; total.z += term.z;
    }
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

// Mixed slots place their retained contact features at the committed poses.
__device__ bool AuditMixedGap(const ModelView& m, const DataView& d, const ReadoutEnergyLedgerParams& p,
    uint32_t env, uint32_t slot, size_t at, double* gap) {
    if (d.dat_pair_kind == nullptr || d.dat_pair_kind[slot] == 0u) return false;
    ContactPointRef ra[3], rb[3];
    uint32_t na = 0u, nb = 0u;
    if (!ContactFeatureRefs(p, m, d, env, slot, ra, &na, rb, &nb)) return false;
    Vec3 a[3]{}, b[3]{};
    for (uint32_t i = 0u; i < na; ++i)
        a[i] = ContactFeaturePoint(m, d, env, p.bodies_per_env, ra[i]);
    for (uint32_t i = 0u; i < nb; ++i)
        b[i] = ContactFeaturePoint(m, d, env, p.bodies_per_env, rb[i]);
    return AuditFeatureGap(a, na, b, nb, d, at, gap);
}

// Rigid pairs query their geometry again at committed poses, without contact margins or position bias.
__device__ bool AuditBodyPairGap(const ModelView& m, const DataView& d, const ReadoutEnergyLedgerParams& p,
    uint32_t env, size_t at, double* gap) {
    if (d.ucontact_a_kind[at] != nk::kUContactSideBody ||
        d.ucontact_b_kind[at] != nk::kUContactSideBody) return false;
    const uint32_t a = d.ucontact_a[at], b = d.ucontact_b[at];
    if (a >= p.bodies_per_env || b >= p.bodies_per_env) return false;
    const auto sa = LoadPrimShape(m.shape_table, a), sb = LoadPrimShape(m.shape_table, b);
    const auto primitive = [&](uint32_t body, const PrimShapeDev& shape) {
        collision::amf::PrimParams out;
        out.frame = collision::amf::BuildPrimFrameDevice(d.body_pose[size_t{env} * p.bodies_per_env + body]);
        out.radius = shape.params[0];
        out.half_height = shape.params[1];
        out.half_extents = {shape.params[0], shape.params[1], shape.params[2]};
        return out;
    };
    constraint::ContactManifold manifold;
    if (!collision::amf::PrimitivePair(sa.kind, primitive(a, sa), sb.kind, primitive(b, sb),
                                      FLT_MAX, &manifold) || manifold.point_count == 0u) return false;
    double penetration = -DBL_MAX;
    for (uint32_t i = 0u; i < manifold.point_count; ++i) {
        if (!isfinite(manifold.points[i].penetration)) return false;
        penetration = fmax(penetration, double(manifold.points[i].penetration));
    }
    *gap = -penetration;
    return isfinite(*gap);
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

// MPM point endpoints retain their physical sample index; grid nodes carry its impulse response.
__device__ bool AuditMpmGap(const NkRow& row, PointMassView points, const ModelView& m, const DataView& d,
    const ReadoutEnergyLedgerParams& p, const SurfaceQueryView& surfaces, uint32_t env, size_t at, double* gap) {
    if (d.ucontact_a_kind[at] != nk::kUContactSidePointEndpoint) return false;
    const uint32_t sample = d.ucontact_a[at] - env * p.point_endpoints_per_env;
    if (sample >= p.mpm_particles_per_env) return false;
    const Vec3 point = d.particle_pos[env * p.particles_per_env + sample];
    const uint32_t kind = d.ucontact_b_kind[at], other = d.ucontact_b[at];
    if (kind == nk::kUContactSideBody) {
        if (other >= p.bodies_per_env) return false;
        const auto pose = d.body_pose[size_t{env} * p.bodies_per_env + other];
        const auto surface = QueryCollidableSurface(surfaces, other, LoadPrimShape(m.shape_table, other),
            PrimInverseTransformPoint(pose, point), FLT_MAX);
        *gap = surface.distance;
        return surface.valid && isfinite(*gap);
    }
    if (kind == nk::kUContactSideBoundary && other == 0u) {
        const Vec3 normal{p.floor_normal[0], p.floor_normal[1], p.floor_normal[2]};
        const double length = sqrt(Dot(normal, normal));
        *gap = (Dot(point, normal) - p.floor_d) / length;
        return length > 0.0 && isfinite(*gap);
    }
    if (kind != nk::kUContactSidePointEndpoint || points.Count(row.b) != 3u) return false;
    Vec3 vertices[3];
    uint32_t indices[3];
    for (uint32_t i = 0u; i < 3u; ++i) {
        const auto term = points.At(row.b, i);
        if (term.kind != nk::kNkSideParticle) return false;
        indices[i] = term.index - env * p.particles_per_env;
        if (indices[i] >= p.particles_per_env) return false;
        vertices[i] = d.particle_pos[term.index];
    }
    uint32_t found = ~0u;
    for (uint32_t surface = 0u; surface < p.surfaces_per_env; ++surface) {
        const auto info = m.particle_surface_info[surface];
        bool contains = true;
        for (uint32_t i = 0u; i < 3u; ++i)
            contains &= indices[i] >= info.vertex_offset && indices[i] - info.vertex_offset < info.vertex_count;
        if (!contains) continue;
        if (found != ~0u) return false;
        found = surface;
    }
    if (found == ~0u) return false;
    const auto info = m.particle_surface_info[found];
    if ((info.flags & collision::kMeshSurfaceClosed) != 0u) return false;
    const auto closest = collision::ClosestTrianglePoint(point, vertices[0], vertices[1], vertices[2]);
    double distance = sqrt(Dot(point - closest.point, point - closest.point));
    if ((info.flags & collision::kMeshSurfaceOriented) != 0u) {
        if (closest.feature < 6u) return false;
        Vec3 normal{};
        bool matched = false;
        for (uint32_t triangle = 0u; triangle < info.triangle_count; ++triangle) {
            const size_t first = size_t{info.triangle_offset + triangle} * 3u;
            uint32_t authored[3];
            bool same = true;
            for (uint32_t i = 0u; i < 3u; ++i) {
                authored[i] = info.vertex_offset + m.particle_surface_triangles[first + i];
                same &= authored[i] == indices[0] || authored[i] == indices[1] || authored[i] == indices[2];
            }
            if (!same) continue;
            if (matched) return false;
            matched = true;
            const size_t base = size_t{env} * p.particles_per_env;
            normal = (d.particle_pos[base + authored[1]] - d.particle_pos[base + authored[0]]).Cross(
                d.particle_pos[base + authored[2]] - d.particle_pos[base + authored[0]]);
        }
        if (!matched || !(normal.LengthSq() > 0.0f)) return false;
        if (Dot(point - closest.point, normal) < 0.0) distance = -distance;
    }
    const double thickness = double(closest.barycentric.x) * m.particle_half_thickness[indices[0]] +
        double(closest.barycentric.y) * m.particle_half_thickness[indices[1]] +
        double(closest.barycentric.z) * m.particle_half_thickness[indices[2]];
    *gap = distance - thickness;
    return isfinite(*gap);
}

// Impulse laws use the solved state; closest-feature gaps use the committed state.
__global__ void ContactAuditKernel(ModelView m, DataView d, ReadoutEnergyLedgerParams p, SurfaceQueryView surfaces) {
    using C = nk::ContactAuditCount;
    using M = nk::ContactAuditMetric;
    const uint32_t env = blockIdx.x;
    const size_t record = size_t{env} * p.substeps + p.slot;
    uint32_t* counts = d.contact_audit_counts + record * nk::kContactAuditCountSize;
    uint64_t* metrics = d.contact_audit_metrics + record * nk::kContactAuditMetricCount;
    if (p.stage == EnergyStage::Solved) {
        for (uint32_t i = threadIdx.x; i < nk::kContactAuditCountSize; i += blockDim.x) counts[i] = 0u;
        for (uint32_t i = threadIdx.x; i < nk::kContactAuditMetricCount; i += blockDim.x) metrics[i] = 0u;
    }
    __syncthreads();
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
    for (uint32_t local = threadIdx.x; local < bound; local += blockDim.x) {
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
                    : AuditBodyPairGap(m, d, p, env, at, &gap) ||
                      AuditBodyParticleGap(m, d, p, surfaces, env, at, &gap) ||
                      AuditMpmGap(normal, points, m, d, p, surfaces, env, at, &gap)) {
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
        const bool pyramidal = (normal.flags & nk::nk_row_flags::kPyramidal) != 0u;
        const uint32_t axes = pyramidal ? 4u : normal.flags & nk::nk_row_flags::kBlockNormal ? 3u : 1u;
        uint32_t stride = normal.group_normal_count;
        if (pyramidal) {
            const uint32_t contact = nkops::ContactSlotForRow(d.ucontact_row_range,
                env * p.contact_slots_per_env, p.contact_slots_per_env, slot);
            stride = nkops::ContactRowRangeFor(d.ucontact_row_range, contact).points;
        }
        const auto resolved = nk::ContactPointImpulse(rows, d.lambda, slot, stride);
        const float normal_impulse = resolved.x;
        if (!isfinite(normal_impulse)) count(C::InvalidRows);
        if (normal_impulse < 0.0f) count(C::NegativeNormalImpulses);
        maximum(M::NegativeNormalImpulse, -normal_impulse, slot);
        double scaled_tangent[2]{};
        bool cone_invalid = false;
        for (uint32_t axis = 0u; axis < axes; ++axis) {
            const uint32_t row = slot + axis * stride;
            if (row >= (env + 1u) * p.rows_per_env) { count(C::InvalidRows); continue; }
            const auto r = rows[row];
            count(C::Rows);
            const double3 a = AuditLinearJacobian(r.a, points);
            const double3 b = AuditLinearJacobian(r.b, points);
            const double impulse = d.lambda[row];
            const double x = a.x + b.x, y = a.y + b.y, z = a.z + b.z;
            const double numerator = sqrt(x*x + y*y + z*z) * fabs(impulse);
            const double denominator = sqrt(a.x*a.x + a.y*a.y + a.z*a.z) * fabs(impulse);
            const double relative = denominator > 0.0 ? numerator / denominator : numerator == 0.0 ? 0.0 : FLT_MAX;
            if (!isfinite(a.x) || !isfinite(a.y) || !isfinite(a.z) ||
                !isfinite(b.x) || !isfinite(b.y) || !isfinite(b.z) || !isfinite(impulse)) count(C::InvalidRows);
            if (relative > 1.0e-6) count(C::LinearClosureViolations);
            maximum(M::RelativeLinearClosure, float(relative), row);
            if (impulse != 0.0)
                maximum(axis == 0u ? M::NormalDisplacementRate : M::TangentDisplacementRate,
                    float(copysign(1.0, impulse) * (SideDisplacementRate(r.a, false, row, d, p) +
                                                    SideDisplacementRate(r.b, true, row, d, p))), row);
            if (axis != 0u && !pyramidal) {
                const float mu = axis == 1u ? normal.mu : normal.friction_secondary;
                if (mu > 0.0f) scaled_tangent[axis - 1u] = impulse / mu;
                else if (impulse != 0.0) cone_invalid = true;
            }
        }
        if (pyramidal) {
            scaled_tangent[0] = resolved.y / normal.mu;
            scaled_tangent[1] = resolved.z / normal.friction_secondary;
        }
        const double tangent = pyramidal ? fabs(scaled_tangent[0]) + fabs(scaled_tangent[1])
                                         : hypot(scaled_tangent[0], scaled_tangent[1]);
        if (cone_invalid || tangent > fmax(double(normal_impulse), 0.0) * 1.001) count(C::ConeViolations);
        maximum(M::ConeImpulseExcess, cone_invalid ? FLT_MAX : float(tangent - normal_impulse), slot);
    }
}

__global__ void IntervalStateKernel(DataView d, ReadoutEnergyLedgerParams p) {
    const uint32_t env = blockIdx.x;
    const size_t record = size_t{env} * p.substeps + p.slot;
    for (uint32_t local = threadIdx.x; local < p.particles_per_env; local += blockDim.x) {
        const size_t source = size_t{env} * p.particles_per_env + local;
        const size_t target = record * p.particles_per_env + local;
        const bool mpm = local < p.mpm_particles_per_env;
        d.interval_particle_pos[target] = d.particle_pos[source];
        d.interval_particle_vel[target] = d.particle_vel[source];
        d.interval_particle_plastic[target] = mpm && d.particle_plastic ? d.particle_plastic[source] : 0.0f;
        for (uint32_t k = 0u; k < 9u; ++k) {
            const float identity = k % 4u == 0u ? 1.0f : 0.0f;
            d.interval_particle_F[target * 9u + k] = mpm && d.particle_F
                ? d.particle_F[source * 9u + k] : identity;
            d.interval_particle_C[target * 9u + k] = mpm && d.particle_C
                ? d.particle_C[source * 9u + k] : 0.0f;
            d.interval_particle_plastic_F[target * 9u + k] = mpm && d.particle_plastic_F
                ? d.particle_plastic_F[source * 9u + k] : identity;
        }
    }
    for (uint32_t local = threadIdx.x; local < p.bodies_per_env; local += blockDim.x) {
        const size_t source = size_t{env} * p.bodies_per_env + local;
        const size_t target = record * p.bodies_per_env + local;
        d.interval_body_pose[target] = d.body_pose[source];
        d.interval_body_linear_velocity[target] = d.body_linear_velocity[source];
        d.interval_body_angular_velocity[target] = d.body_angular_velocity[source];
    }
    for (uint32_t local = threadIdx.x; local < p.links_per_env; local += blockDim.x) {
        const size_t source = size_t{env} * p.links_per_env + local;
        const size_t target = record * p.links_per_env + local;
        d.interval_q[target] = d.q[source];
        d.interval_qdot[target] = d.qdot[source];
        d.interval_joint_rotation[target] = d.joint_rotation[source];
        d.interval_joint_angular_velocity[target] = d.joint_angular_velocity[source];
        d.interval_link_velocity[target] = d.link_velocity[source];
    }
    for (uint32_t local = threadIdx.x; local < p.artics_per_env; local += blockDim.x)
        d.interval_base_pose[record * p.artics_per_env + local] =
            d.base_pose[size_t{env} * p.artics_per_env + local];
    __syncthreads();
    if (threadIdx.x == 0u) d.interval_state_dt[record] = p.dt;
}

Status OpReadoutIntervalState(const ModelView&, const DataView& d, const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const ReadoutEnergyLedgerParams*>(params);
    if (!p || !(p->dt > 0.0f) || !std::isfinite(p->dt) || p->slot >= p->substeps ||
        p->stage != EnergyStage::End || !d.interval_state_dt) return Status::InvalidArgument;
    if (p->particles_per_env > 0u && (!d.interval_particle_pos || !d.interval_particle_vel ||
        !d.interval_particle_F || !d.interval_particle_C || !d.interval_particle_plastic_F ||
        !d.interval_particle_plastic)) return Status::InvalidArgument;
    if (p->bodies_per_env > 0u && (!d.interval_body_pose || !d.interval_body_linear_velocity ||
        !d.interval_body_angular_velocity)) return Status::InvalidArgument;
    if (p->links_per_env > 0u && (!d.interval_q || !d.interval_qdot || !d.interval_joint_rotation ||
        !d.interval_joint_angular_velocity || !d.interval_link_velocity)) return Status::InvalidArgument;
    if (p->artics_per_env > 0u && !d.interval_base_pose) return Status::InvalidArgument;
    if (p->env_count > 0u)
        LaunchCuda(IntervalStateKernel, dim3(p->env_count), dim3(kThreads), 0u, stream, d, *p);
    return cudaGetLastError() == cudaSuccess ? Status::Ok : Status::Failed;
}

Status OpReadoutContactAudit(const ModelView& m, const DataView& d, const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const ReadoutEnergyLedgerParams*>(params);
    if (!p || p->slot >= p->substeps || !d.contact_audit_counts || !d.contact_audit_metrics)
        return Status::InvalidArgument;
    const auto surfaces = MakeSurfaceQueryView(m, p->bodies_per_env, p->mesh_geometry,
                                               p->sdf_grid_count, p->sdf_cell_total);
    if (p->env_count > 0u)
        LaunchCuda(ContactAuditKernel, dim3(p->env_count), dim3(kThreads), 0u, stream, m, d, *p, surfaces);
    return cudaGetLastError() == cudaSuccess ? Status::Ok : Status::Failed;
}

Status OpReadoutEnergyLedger(const ModelView& m, const DataView& d, const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const ReadoutEnergyLedgerParams*>(params);
    if (!p || !(p->dt > 0.0f) || !std::isfinite(p->dt) || p->slot >= p->substeps ||
        static_cast<uint32_t>(p->stage) >= nk::kEnergyStageCount || !d.energy_ledger || !d.energy_ledger_status ||
        (p->physics_diagnostics && !d.physics_stage_metrics) ||
        (p->bodies_per_env > 0u && (!d.energy_body_wrench || !d.energy_body_start_velocity)) ||
        (p->links_per_env > 0u && (!d.energy_link_wrench || !d.energy_link_start_velocity)))
        return Status::InvalidArgument;
    if (p->env_count == 0u) return Status::Ok;
    if (p->mpm_particles_per_env > 0u) {
        if (!(p->grid_dx > 0.0f) || !d.energy_mpm_start_F || !d.energy_mpm_start_plastic ||
            !d.energy_grid_entry_momentum || !d.energy_grid_entry_mass || !d.energy_grid_stress_impulse ||
            !d.energy_grid_viscous_impulse || !d.energy_grid_free_velocity || !d.energy_grid_prescribed)
            return Status::InvalidArgument;
        if (p->stage == EnergyStage::Free) {
            const uint64_t nodes = uint64_t{p->env_count} * p->grid_nodes_per_env;
            const uint64_t particles = uint64_t{p->env_count} * p->mpm_particles_per_env;
            LaunchCuda(EnergyGridClearKernel, dim3((nodes + kThreads - 1u) / kThreads), dim3(kThreads),
                0u, stream, d, *p);
            LaunchCuda(EnergyMpmTransferKernel, dim3((particles + kThreads - 1u) / kThreads), dim3(kThreads),
                0u, stream, d, *p);
        }
    }
    if (p->artics_per_env > 0u)
        LaunchCuda(EnergyLinkFramesKernel, dim3(p->env_count * p->artics_per_env), dim3(32u), 0u, stream, m, d, *p);
    if (p->stage == EnergyStage::Solved && p->rows_per_env > 0u) {
        const uint64_t rows = uint64_t{p->env_count} * p->rows_per_env;
        const uint64_t blocks = (rows + kThreads - 1u) / kThreads;
        if (blocks > std::numeric_limits<uint32_t>::max()) return Status::InvalidArgument;
        LaunchCuda(EnergyKinematicImpulseKernel, dim3(static_cast<uint32_t>(blocks)), dim3(kThreads), 0u, stream, d, *p);
    }
    if (p->stage == EnergyStage::Solved && p->rows_per_env > 0u && d.row_compliant_storage &&
        d.energy_split_matrix && (p->dofs_per_artic == 0u || d.energy_split_forward))
        LaunchCuda(EnergyDampingSplitKernel, dim3(p->env_count), dim3(kThreads), 0u, stream, d, *p);
    if (p->physics_diagnostics)
        LaunchCuda(PhysicsStageMetricsKernel, dim3(p->env_count), dim3(kThreads), 0u, stream, m, d, *p);
    LaunchCuda(EnergyLedgerKernel, dim3(p->env_count), dim3(kThreads), 0u, stream, m, d, *p);
    return cudaGetLastError() == cudaSuccess ? Status::Ok : Status::Failed;
}

}  // namespace

void RegisterNkEnergyLedgerOps() {
    SetCudaOp(NkOp::ReadoutIntervalState, &OpReadoutIntervalState);
    SetCudaOp(NkOp::ReadoutEnergyLedger, &OpReadoutEnergyLedger);
    SetCudaOp(NkOp::ReadoutContactAudit, &OpReadoutContactAudit);
}

}  // namespace nuka::phi
