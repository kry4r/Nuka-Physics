// ---------------------------------------------------------------------------
// nuka::runtime::gpu::CudaParticleWorld implementation
// ---------------------------------------------------------------------------

#include "runtime/gpu/cuda_particle_world.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace nuka::runtime::gpu {

namespace {

struct ParticleDiagnostics {
    uint32_t contact_count;
    uint32_t rigid_impulse_count;
    uint32_t coupling_active_slot_count;
    uint32_t coupling_warm_start_count;
    uint32_t coupling_tangent_warm_start_count;
    float max_penetration;
    float max_penetration_after_solve;
    float speed_sq;
    float kinetic_energy;
    float rigid_impulse_magnitude;
    float rigid_angular_impulse_magnitude;
    float coupling_warm_start_impulse_magnitude;
    float coupling_tangent_warm_start_impulse_magnitude;
    float max_coupling_normal_impulse;
    float coupling_force_magnitude;
    float coupling_torque_magnitude;
};

struct CouplingRowDiagnostics {
    uint32_t active_row_count;
    uint32_t impulse_count;
    uint32_t friction_impulse_count;
    float impulse_magnitude;
    float friction_impulse_magnitude;
    float angular_impulse_magnitude;
    float force_magnitude;
    float torque_magnitude;
    float max_normal_impulse;
    float max_row_residual;
};

struct ParticleRigidCouplingSchedulerInput {
    CudaConstraintRowBufferView row_buffer;
    CudaConstraintRowSchedulerConfig config;
    uint32_t particle_count = 0u;
    CudaParticleCouplingConstraintRow* rows = nullptr;
    math::Vec3* particle_velocities = nullptr;
    const float* particle_inv_masses = nullptr;
    math::Vec3* body_linear_velocities = nullptr;
    math::Vec3* body_angular_velocities = nullptr;
    const float* body_inv_masses = nullptr;
    const math::Vec3* body_inv_inertias = nullptr;
    float* coupling_normal_impulses = nullptr;
    uint32_t* coupling_shape_indices = nullptr;
    float dt = 0.0f;
    uint32_t diagnostic_iteration_base = 0u;
    CouplingRowDiagnostics* diagnostics = nullptr;
    CudaParticleStepReport* report = nullptr;
    uint32_t* host_kernel_launch_count = nullptr;
    uint32_t* host_solver_launch_count = nullptr;
};

__device__ void ResetCouplingRowSolverReport(CudaParticleStepReport* report) {
    report->coupling_row_solver_launch_count = 0u;
    report->coupling_row_solver_iteration_count = 0u;
    report->coupling_row_solver_impulse_count = 0u;
    report->coupling_row_solver_impulse_magnitude = 0.0f;
    report->coupling_row_solver_friction_impulse_count = 0u;
    report->coupling_row_solver_friction_impulse_magnitude = 0.0f;
    report->coupling_row_solver_diagnostic_slot_count = 0u;
    report->coupling_row_solver_max_iteration_normal_delta_impulse = 0.0f;
    report->coupling_row_solver_max_iteration_tangent_delta_impulse = 0.0f;
    report->coupling_row_solver_max_residual = 0.0f;
    for (uint32_t slot = 0u; slot < kCudaParticleRowSolverDiagnosticSlots; ++slot) {
        report->coupling_row_solver_iteration_normal_delta_impulses[slot] = 0.0f;
        report->coupling_row_solver_iteration_tangent_delta_impulses[slot] = 0.0f;
        report->coupling_row_solver_iteration_max_residuals[slot] = 0.0f;
    }
    report->coupling_scheduler_report = CudaConstraintRowSchedulerReport{};
}

__device__ math::Vec3 MakeVec3(float x, float y, float z) {
    math::Vec3 result;
    result.x = x;
    result.y = y;
    result.z = z;
    return result;
}

__device__ math::Vec3 Add(math::Vec3 a, math::Vec3 b) {
    return MakeVec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ math::Vec3 Sub(math::Vec3 a, math::Vec3 b) {
    return MakeVec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ math::Vec3 Scale(math::Vec3 v, float s) {
    return MakeVec3(v.x * s, v.y * s, v.z * s);
}

__device__ float Dot(math::Vec3 a, math::Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float LengthSq(math::Vec3 v) {
    return Dot(v, v);
}

__device__ math::Vec3 NormalizeOr(math::Vec3 v, math::Vec3 fallback) {
    const float len_sq = LengthSq(v);
    if (len_sq <= 1.0e-12f) {
        return fallback;
    }
    return Scale(v, rsqrtf(len_sq));
}

__device__ math::Vec3 Cross(math::Vec3 a, math::Vec3 b) {
    return MakeVec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

__device__ math::Vec3 ChooseTangent(math::Vec3 normal) {
    const math::Vec3 axis =
        fabsf(normal.x) < 0.9f ? MakeVec3(1.0f, 0.0f, 0.0f)
                               : MakeVec3(0.0f, 1.0f, 0.0f);
    return NormalizeOr(Cross(normal, axis), MakeVec3(0.0f, 0.0f, 1.0f));
}

__device__ math::Vec3 ChooseContactTangent(math::Vec3 normal,
                                           math::Vec3 relative_velocity) {
    const float normal_speed = Dot(relative_velocity, normal);
    const math::Vec3 tangent_velocity =
        Sub(relative_velocity, Scale(normal, normal_speed));
    if (LengthSq(tangent_velocity) > 1.0e-12f) {
        return NormalizeOr(tangent_velocity, ChooseTangent(normal));
    }
    return ChooseTangent(normal);
}

__device__ math::Vec3 Rotate(math::Quat q, math::Vec3 v) {
    const math::Vec3 qv = MakeVec3(q.x, q.y, q.z);
    const math::Vec3 t = Scale(Cross(qv, v), 2.0f);
    return Add(Add(v, Scale(t, q.w)), Cross(qv, t));
}

__device__ math::Quat Conjugate(math::Quat q) {
    math::Quat result;
    result.w = q.w;
    result.x = -q.x;
    result.y = -q.y;
    result.z = -q.z;
    return result;
}

__device__ math::Quat Mul(math::Quat a, math::Quat b) {
    math::Quat result;
    result.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
    result.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;
    result.y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x;
    result.z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w;
    return result;
}

__device__ math::Vec3 TransformPoint(math::Transform transform, math::Vec3 point) {
    return Add(Rotate(transform.rotation, point), transform.position);
}

__device__ math::Vec3 InverseTransformPoint(math::Transform transform, math::Vec3 point) {
    return Rotate(Conjugate(transform.rotation), Sub(point, transform.position));
}

__device__ math::Vec3 TransformDirection(math::Transform transform, math::Vec3 direction) {
    return Rotate(transform.rotation, direction);
}

__device__ math::Transform Compose(math::Transform a, math::Transform b) {
    math::Transform result;
    result.position = TransformPoint(a, b.position);
    result.rotation = Mul(a.rotation, b.rotation);
    return result;
}

__device__ float ClampNonNegative(float value) {
    return value < 0.0f ? 0.0f : value;
}

__device__ float Clamp(float value, float lower, float upper) {
    return fminf(fmaxf(value, lower), upper);
}

__device__ void ApplyFriction(math::Vec3 normal, float friction, math::Vec3* velocity) {
    if (friction <= 0.0f) {
        return;
    }

    const float normal_speed = Dot(*velocity, normal);
    const math::Vec3 normal_velocity = Scale(normal, normal_speed);
    const math::Vec3 tangent_velocity = Sub(*velocity, normal_velocity);
    const float damping = fmaxf(0.0f, 1.0f - friction);
    *velocity = Add(normal_velocity, Scale(tangent_velocity, damping));
}

__device__ void AddVelocityAtomic(math::Vec3* velocities, uint32_t body_id, math::Vec3 delta) {
    atomicAdd(&velocities[body_id].x, delta.x);
    atomicAdd(&velocities[body_id].y, delta.y);
    atomicAdd(&velocities[body_id].z, delta.z);
}

__device__ math::Vec3 ApplyInverseInertia(math::Vec3 angular_impulse,
                                          math::Vec3 inv_inertia) {
    return MakeVec3(angular_impulse.x * inv_inertia.x,
                    angular_impulse.y * inv_inertia.y,
                    angular_impulse.z * inv_inertia.z);
}

__device__ float CouplingRowNormalVelocity(
    const CudaParticleCouplingConstraintRow& row,
    math::Vec3 particle_velocity,
    math::Vec3 body_linear_velocity,
    math::Vec3 body_angular_velocity) {
    return Dot(row.particle_linear_jacobian, particle_velocity) +
        Dot(row.body_linear_jacobian, body_linear_velocity) +
        Dot(row.body_angular_jacobian, body_angular_velocity);
}

__device__ float CouplingRowTangentVelocity(math::Vec3 tangent,
                                            math::Vec3 body_angular_jacobian,
                                            math::Vec3 particle_velocity,
                                            math::Vec3 body_linear_velocity,
                                            math::Vec3 body_angular_velocity) {
    return Dot(tangent, particle_velocity) +
        Dot(Scale(tangent, -1.0f), body_linear_velocity) +
        Dot(body_angular_jacobian, body_angular_velocity);
}

__device__ float CouplingProjectedRowResidual(float velocity_error,
                                              float effective_mass,
                                              float impulse,
                                              float lower_limit,
                                              float upper_limit) {
    const float projected_impulse =
        Clamp(impulse + effective_mass * velocity_error, lower_limit, upper_limit);
    return fabsf(projected_impulse - impulse);
}

__device__ math::Vec3 ApplyCouplingVelocityImpulse(
    math::Vec3 particle_linear_jacobian,
    math::Vec3 body_linear_jacobian,
    math::Vec3 body_angular_jacobian,
    float impulse,
    float particle_inv_mass,
    float body_inv_mass,
    math::Vec3 inv_inertia,
    scene::BodyId body_id,
    math::Vec3* particle_velocity,
    math::Vec3* body_linear_velocity,
    math::Vec3* body_angular_velocity,
    math::Vec3* body_linear_velocities,
    math::Vec3* body_angular_velocities) {
    if (fabsf(impulse) <= 1.0e-12f) {
        return MakeVec3(0.0f, 0.0f, 0.0f);
    }

    const math::Vec3 particle_delta =
        Scale(particle_linear_jacobian, particle_inv_mass * impulse);
    const math::Vec3 body_linear_delta =
        Scale(body_linear_jacobian, body_inv_mass * impulse);
    const math::Vec3 body_angular_delta =
        ApplyInverseInertia(Scale(body_angular_jacobian, impulse), inv_inertia);

    *particle_velocity = Add(*particle_velocity, particle_delta);
    AddVelocityAtomic(body_linear_velocities, body_id, body_linear_delta);
    AddVelocityAtomic(body_angular_velocities, body_id, body_angular_delta);
    *body_linear_velocity = Add(*body_linear_velocity, body_linear_delta);
    *body_angular_velocity = Add(*body_angular_velocity, body_angular_delta);
    return body_angular_delta;
}

__device__ uint32_t FindCouplingSlot(uint32_t shape_index,
                                     const uint32_t* stored_shape_indices) {
    if (stored_shape_indices == nullptr) {
        return kInvalidCudaParticleCouplingShape;
    }

    uint32_t free_slot = kInvalidCudaParticleCouplingShape;
    for (uint32_t slot = 0u; slot < kCudaParticleCouplingSlotsPerParticle; ++slot) {
        const uint32_t stored_shape = stored_shape_indices[slot];
        if (stored_shape == shape_index) {
            return slot;
        }
        if (stored_shape == kInvalidCudaParticleCouplingShape &&
            free_slot == kInvalidCudaParticleCouplingShape) {
            free_slot = slot;
        }
    }
    return free_slot;
}

__device__ CudaParticleCouplingConstraintRow MakeInactiveCouplingRow() {
    CudaParticleCouplingConstraintRow row{};
    row.active = false;
    row.shape_index = kInvalidCudaParticleCouplingShape;
    row.body_id = scene::kInvalidBody;
    return row;
}

__device__ float SolvePlane(const CudaParticlePlaneCollider& plane,
                            float radius,
                            math::Vec3* position,
                            math::Vec3* velocity,
                            uint32_t* contact_count,
                            bool count_contact) {
    if (!plane.enabled) {
        return 0.0f;
    }

    const math::Vec3 normal = NormalizeOr(plane.normal, MakeVec3(0.0f, 1.0f, 0.0f));
    const float signed_distance = Dot(*position, normal) - plane.offset;
    const float penetration = radius - signed_distance;
    if (penetration <= 0.0f) {
        return 0.0f;
    }

    *position = Add(*position, Scale(normal, penetration));
    const float normal_speed = Dot(*velocity, normal);
    if (normal_speed < 0.0f) {
        *velocity = Sub(*velocity, Scale(normal, (1.0f + plane.restitution) * normal_speed));
    }
    ApplyFriction(normal, plane.friction, velocity);
    if (count_contact) {
        ++(*contact_count);
    }
    return penetration;
}

__device__ float SolveSphere(const CudaParticleSphereCollider& sphere,
                             float radius,
                             math::Vec3* position,
                             math::Vec3* velocity,
                             uint32_t* contact_count,
                             bool count_contact) {
    if (!sphere.enabled) {
        return 0.0f;
    }

    const math::Vec3 delta = Sub(*position, sphere.center);
    const float distance_sq = LengthSq(delta);
    const float target_distance = sphere.radius + radius;
    const float distance = sqrtf(distance_sq);
    const float penetration = target_distance - distance;
    if (penetration <= 0.0f) {
        return 0.0f;
    }

    const math::Vec3 velocity_normal = NormalizeOr(*velocity, MakeVec3(1.0f, 0.0f, 0.0f));
    const math::Vec3 normal = NormalizeOr(delta, velocity_normal);
    *position = Add(sphere.center, Scale(normal, target_distance));

    const float normal_speed = Dot(*velocity, normal);
    if (normal_speed < 0.0f) {
        *velocity = Sub(*velocity, Scale(normal, (1.0f + sphere.restitution) * normal_speed));
    }
    ApplyFriction(normal, sphere.friction, velocity);
    if (count_contact) {
        ++(*contact_count);
    }
    return penetration;
}

__device__ float PlaneResidual(const CudaParticlePlaneCollider& plane,
                               float radius,
                               math::Vec3 position) {
    if (!plane.enabled) {
        return 0.0f;
    }
    const math::Vec3 normal = NormalizeOr(plane.normal, MakeVec3(0.0f, 1.0f, 0.0f));
    const float signed_distance = Dot(position, normal) - plane.offset;
    return ClampNonNegative(radius - signed_distance);
}

__device__ float SphereResidual(const CudaParticleSphereCollider& sphere,
                                float radius,
                                math::Vec3 position) {
    if (!sphere.enabled) {
        return 0.0f;
    }
    const math::Vec3 delta = Sub(position, sphere.center);
    const float distance = sqrtf(LengthSq(delta));
    return ClampNonNegative(sphere.radius + radius - distance);
}

__global__ void IntegrateAndCoupleParticlesKernel(
    uint32_t particle_count,
    math::Vec3* positions,
    math::Vec3* velocities,
    const float* inv_masses,
    const float* radii,
    math::Vec3 gravity,
    float dt,
    CudaParticlePlaneCollider plane,
    CudaParticleSphereCollider sphere,
    ParticleDiagnostics* diagnostics) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= particle_count) {
        return;
    }

    ParticleDiagnostics diag{};
    const float inv_mass = inv_masses[index];
    const float radius = radii[index];
    math::Vec3 position = positions[index];
    math::Vec3 velocity = velocities[index];

    if (inv_mass > 0.0f) {
        const float initial_penetration = fmaxf(
            PlaneResidual(plane, radius, position),
            SphereResidual(sphere, radius, position));

        velocity = Add(velocity, Scale(gravity, dt));
        position = Add(position, Scale(velocity, dt));

        uint32_t contact_count = 0u;
        float max_penetration = initial_penetration;
        max_penetration = fmaxf(
            max_penetration,
            SolvePlane(plane, radius, &position, &velocity, &contact_count, true));
        max_penetration = fmaxf(
            max_penetration,
            SolveSphere(sphere, radius, &position, &velocity, &contact_count, true));
        // One extra projection pass improves rigid boundary consistency while
        // keeping the operation inside the same CUDA coupling kernel.
        (void)SolvePlane(plane, radius, &position, &velocity, &contact_count, false);

        diag.contact_count = contact_count;
        diag.max_penetration = max_penetration;
        diag.max_penetration_after_solve = fmaxf(
            PlaneResidual(plane, radius, position),
            SphereResidual(sphere, radius, position));

        positions[index] = position;
        velocities[index] = velocity;
    }

    const float speed_sq = LengthSq(velocity);
    diag.speed_sq = speed_sq;
    diag.kinetic_energy =
        inv_mass > 0.0f ? 0.5f * (1.0f / inv_mass) * speed_sq : 0.0f;
    diagnostics[index] = diag;
}

__global__ void ReduceParticleDiagnosticsKernel(uint32_t particle_count,
                                                const ParticleDiagnostics* diagnostics,
                                                CudaParticleStepReport* report) {
    __shared__ uint32_t shared_contacts[256];
    __shared__ uint32_t shared_active_slots[256];
    __shared__ float shared_penetration[256];
    __shared__ float shared_residual[256];
    __shared__ float shared_speed_sq[256];
    __shared__ float shared_energy[256];
    __shared__ float shared_coupling_impulse[256];
    __shared__ float shared_coupling_force[256];
    __shared__ float shared_coupling_torque[256];

    const uint32_t tid = threadIdx.x;
    uint32_t contact_count = 0u;
    uint32_t active_slot_count = 0u;
    float max_penetration = 0.0f;
    float max_residual = 0.0f;
    float max_speed_sq = 0.0f;
    float kinetic_energy = 0.0f;
    float max_coupling_impulse = 0.0f;
    float coupling_force = 0.0f;
    float coupling_torque = 0.0f;

    for (uint32_t index = tid; index < particle_count; index += blockDim.x) {
        const ParticleDiagnostics diag = diagnostics[index];
        contact_count += diag.contact_count;
        active_slot_count += diag.coupling_active_slot_count;
        max_penetration = fmaxf(max_penetration, diag.max_penetration);
        max_residual = fmaxf(max_residual, diag.max_penetration_after_solve);
        max_speed_sq = fmaxf(max_speed_sq, diag.speed_sq);
        kinetic_energy += diag.kinetic_energy;
        max_coupling_impulse =
            fmaxf(max_coupling_impulse, diag.max_coupling_normal_impulse);
        coupling_force += diag.coupling_force_magnitude;
        coupling_torque += diag.coupling_torque_magnitude;
    }

    shared_contacts[tid] = contact_count;
    shared_active_slots[tid] = active_slot_count;
    shared_penetration[tid] = max_penetration;
    shared_residual[tid] = max_residual;
    shared_speed_sq[tid] = max_speed_sq;
    shared_energy[tid] = kinetic_energy;
    shared_coupling_impulse[tid] = max_coupling_impulse;
    shared_coupling_force[tid] = coupling_force;
    shared_coupling_torque[tid] = coupling_torque;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared_contacts[tid] += shared_contacts[tid + stride];
            shared_active_slots[tid] += shared_active_slots[tid + stride];
            shared_penetration[tid] =
                fmaxf(shared_penetration[tid], shared_penetration[tid + stride]);
            shared_residual[tid] = fmaxf(shared_residual[tid], shared_residual[tid + stride]);
            shared_speed_sq[tid] =
                fmaxf(shared_speed_sq[tid], shared_speed_sq[tid + stride]);
            shared_energy[tid] += shared_energy[tid + stride];
            shared_coupling_impulse[tid] =
                fmaxf(shared_coupling_impulse[tid],
                      shared_coupling_impulse[tid + stride]);
            shared_coupling_force[tid] += shared_coupling_force[tid + stride];
            shared_coupling_torque[tid] += shared_coupling_torque[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0u) {
        report->particle_count = particle_count;
        report->contact_count = shared_contacts[0];
        report->rigid_impulse_count = 0u;
        report->max_penetration = shared_penetration[0];
        report->max_penetration_after_solve = shared_residual[0];
        report->max_speed = sqrtf(shared_speed_sq[0]);
        report->kinetic_energy = shared_energy[0];
        report->rigid_impulse_magnitude = 0.0f;
        report->rigid_angular_impulse_magnitude = 0.0f;
        report->coupling_active_slot_count = shared_active_slots[0];
        report->coupling_warm_start_count = 0u;
        report->coupling_tangent_warm_start_count = 0u;
        report->coupling_warm_start_impulse_magnitude = 0.0f;
        report->coupling_tangent_warm_start_impulse_magnitude = 0.0f;
        report->max_coupling_normal_impulse = shared_coupling_impulse[0];
        report->coupling_force_magnitude = shared_coupling_force[0];
        report->coupling_torque_magnitude = shared_coupling_torque[0];
        ResetCouplingRowSolverReport(report);
    }
}

__global__ void ReduceParticleDiagnosticsWithRigidKernel(
    uint32_t particle_count,
    const ParticleDiagnostics* diagnostics,
    bool reset_row_solver_report,
    CudaParticleStepReport* report) {
    __shared__ uint32_t shared_contacts[256];
    __shared__ uint32_t shared_impulse_contacts[256];
    __shared__ uint32_t shared_active_slots[256];
    __shared__ uint32_t shared_warm_start_contacts[256];
    __shared__ uint32_t shared_tangent_warm_start_contacts[256];
    __shared__ float shared_penetration[256];
    __shared__ float shared_residual[256];
    __shared__ float shared_speed_sq[256];
    __shared__ float shared_energy[256];
    __shared__ float shared_impulse_magnitude[256];
    __shared__ float shared_angular_impulse_magnitude[256];
    __shared__ float shared_warm_start_magnitude[256];
    __shared__ float shared_tangent_warm_start_magnitude[256];
    __shared__ float shared_coupling_impulse[256];
    __shared__ float shared_coupling_force[256];
    __shared__ float shared_coupling_torque[256];

    const uint32_t tid = threadIdx.x;
    uint32_t contact_count = 0u;
    uint32_t impulse_count = 0u;
    uint32_t active_slot_count = 0u;
    uint32_t warm_start_count = 0u;
    uint32_t tangent_warm_start_count = 0u;
    float max_penetration = 0.0f;
    float max_residual = 0.0f;
    float max_speed_sq = 0.0f;
    float kinetic_energy = 0.0f;
    float impulse_magnitude = 0.0f;
    float angular_impulse_magnitude = 0.0f;
    float warm_start_magnitude = 0.0f;
    float tangent_warm_start_magnitude = 0.0f;
    float max_coupling_impulse = 0.0f;
    float coupling_force = 0.0f;
    float coupling_torque = 0.0f;

    for (uint32_t index = tid; index < particle_count; index += blockDim.x) {
        const ParticleDiagnostics diag = diagnostics[index];
        contact_count += diag.contact_count;
        impulse_count += diag.rigid_impulse_count;
        active_slot_count += diag.coupling_active_slot_count;
        warm_start_count += diag.coupling_warm_start_count;
        tangent_warm_start_count += diag.coupling_tangent_warm_start_count;
        max_penetration = fmaxf(max_penetration, diag.max_penetration);
        max_residual = fmaxf(max_residual, diag.max_penetration_after_solve);
        max_speed_sq = fmaxf(max_speed_sq, diag.speed_sq);
        kinetic_energy += diag.kinetic_energy;
        impulse_magnitude += diag.rigid_impulse_magnitude;
        angular_impulse_magnitude += diag.rigid_angular_impulse_magnitude;
        warm_start_magnitude += diag.coupling_warm_start_impulse_magnitude;
        tangent_warm_start_magnitude +=
            diag.coupling_tangent_warm_start_impulse_magnitude;
        max_coupling_impulse =
            fmaxf(max_coupling_impulse, diag.max_coupling_normal_impulse);
        coupling_force += diag.coupling_force_magnitude;
        coupling_torque += diag.coupling_torque_magnitude;
    }

    shared_contacts[tid] = contact_count;
    shared_impulse_contacts[tid] = impulse_count;
    shared_active_slots[tid] = active_slot_count;
    shared_warm_start_contacts[tid] = warm_start_count;
    shared_tangent_warm_start_contacts[tid] = tangent_warm_start_count;
    shared_penetration[tid] = max_penetration;
    shared_residual[tid] = max_residual;
    shared_speed_sq[tid] = max_speed_sq;
    shared_energy[tid] = kinetic_energy;
    shared_impulse_magnitude[tid] = impulse_magnitude;
    shared_angular_impulse_magnitude[tid] = angular_impulse_magnitude;
    shared_warm_start_magnitude[tid] = warm_start_magnitude;
    shared_tangent_warm_start_magnitude[tid] = tangent_warm_start_magnitude;
    shared_coupling_impulse[tid] = max_coupling_impulse;
    shared_coupling_force[tid] = coupling_force;
    shared_coupling_torque[tid] = coupling_torque;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared_contacts[tid] += shared_contacts[tid + stride];
            shared_impulse_contacts[tid] += shared_impulse_contacts[tid + stride];
            shared_active_slots[tid] += shared_active_slots[tid + stride];
            shared_warm_start_contacts[tid] += shared_warm_start_contacts[tid + stride];
            shared_tangent_warm_start_contacts[tid] +=
                shared_tangent_warm_start_contacts[tid + stride];
            shared_penetration[tid] =
                fmaxf(shared_penetration[tid], shared_penetration[tid + stride]);
            shared_residual[tid] = fmaxf(shared_residual[tid], shared_residual[tid + stride]);
            shared_speed_sq[tid] =
                fmaxf(shared_speed_sq[tid], shared_speed_sq[tid + stride]);
            shared_energy[tid] += shared_energy[tid + stride];
            shared_impulse_magnitude[tid] += shared_impulse_magnitude[tid + stride];
            shared_angular_impulse_magnitude[tid] +=
                shared_angular_impulse_magnitude[tid + stride];
            shared_warm_start_magnitude[tid] +=
                shared_warm_start_magnitude[tid + stride];
            shared_tangent_warm_start_magnitude[tid] +=
                shared_tangent_warm_start_magnitude[tid + stride];
            shared_coupling_impulse[tid] =
                fmaxf(shared_coupling_impulse[tid],
                      shared_coupling_impulse[tid + stride]);
            shared_coupling_force[tid] += shared_coupling_force[tid + stride];
            shared_coupling_torque[tid] += shared_coupling_torque[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0u) {
        report->particle_count = particle_count;
        report->contact_count = shared_contacts[0];
        report->rigid_impulse_count = shared_impulse_contacts[0];
        report->max_penetration = shared_penetration[0];
        report->max_penetration_after_solve = shared_residual[0];
        report->max_speed = sqrtf(shared_speed_sq[0]);
        report->kinetic_energy = shared_energy[0];
        report->rigid_impulse_magnitude = shared_impulse_magnitude[0];
        report->rigid_angular_impulse_magnitude = shared_angular_impulse_magnitude[0];
        report->coupling_active_slot_count = shared_active_slots[0];
        report->coupling_warm_start_count = shared_warm_start_contacts[0];
        report->coupling_tangent_warm_start_count =
            shared_tangent_warm_start_contacts[0];
        report->coupling_warm_start_impulse_magnitude = shared_warm_start_magnitude[0];
        report->coupling_tangent_warm_start_impulse_magnitude =
            shared_tangent_warm_start_magnitude[0];
        report->max_coupling_normal_impulse = shared_coupling_impulse[0];
        report->coupling_force_magnitude = shared_coupling_force[0];
        report->coupling_torque_magnitude = shared_coupling_torque[0];
        if (reset_row_solver_report) {
            ResetCouplingRowSolverReport(report);
        }
    }
}

__device__ float DeviceWorldShapeResidual(scene::ShapeType shape_type,
                                          math::Transform shape_transform,
                                          math::Vec3 shape_half_extents,
                                          float shape_radius,
                                          float shape_half_height,
                                          float particle_radius,
                                          math::Vec3 position) {
    if (shape_type == scene::ShapeType::Plane) {
        const math::Vec3 normal = Rotate(shape_transform.rotation, MakeVec3(0.0f, 1.0f, 0.0f));
        const float signed_distance = Dot(Sub(position, shape_transform.position), normal);
        return ClampNonNegative(particle_radius - signed_distance);
    }

    if (shape_type == scene::ShapeType::Sphere) {
        const float distance = sqrtf(LengthSq(Sub(position, shape_transform.position)));
        return ClampNonNegative(shape_radius + particle_radius - distance);
    }

    if (shape_type == scene::ShapeType::Box) {
        const math::Vec3 local_position = InverseTransformPoint(shape_transform, position);
        const math::Vec3 closest = MakeVec3(
            Clamp(local_position.x, -shape_half_extents.x, shape_half_extents.x),
            Clamp(local_position.y, -shape_half_extents.y, shape_half_extents.y),
            Clamp(local_position.z, -shape_half_extents.z, shape_half_extents.z));
        const float distance = sqrtf(LengthSq(Sub(local_position, closest)));
        return ClampNonNegative(particle_radius - distance);
    }

    if (shape_type == scene::ShapeType::Capsule) {
        const math::Vec3 local_position = InverseTransformPoint(shape_transform, position);
        const float clamped_y = Clamp(local_position.y, -shape_half_height, shape_half_height);
        const math::Vec3 closest = MakeVec3(0.0f, clamped_y, 0.0f);
        const float distance = sqrtf(LengthSq(Sub(local_position, closest)));
        return ClampNonNegative(shape_radius + particle_radius - distance);
    }

    return 0.0f;
}

__device__ float SolveDeviceWorldShape(scene::ShapeType shape_type,
                                       uint32_t shape_index,
                                       scene::BodyId body_id,
                                       math::Transform body_transform,
                                       math::Transform shape_transform,
                                       math::Vec3 shape_half_extents,
                                       float shape_radius,
                                       float shape_half_height,
                                       float body_inv_mass,
                                       float particle_inv_mass,
                                       float particle_radius,
                                       float friction,
                                       float restitution,
                                       bool accumulate_rigid_impulses,
                                       math::Vec3* rigid_velocities,
                                       math::Vec3* rigid_angular_velocities,
                                       const math::Vec3* rigid_inv_inertias,
                                       math::Vec3* position,
                                       math::Vec3* velocity,
                                       uint32_t* contact_count,
                                       uint32_t* impulse_count,
                                       float* impulse_magnitude,
                                       float* angular_impulse_magnitude,
                                       bool solve_coupling_rows_on_cuda,
                                       bool enable_coupling_warm_start,
                                       float* stored_normal_impulse,
                                       uint32_t* stored_shape_index,
                                       bool* touched_slot,
                                       CudaParticleCouplingConstraintRow* coupling_row,
                                       uint32_t particle_index,
                                       uint32_t* warm_start_count,
                                       uint32_t* tangent_warm_start_count,
                                       float* warm_start_magnitude,
                                       float* tangent_warm_start_magnitude,
                                       float* max_coupling_normal_impulse,
                                       float* force_magnitude,
                                       float* torque_magnitude,
                                       float dt) {
    math::Vec3 normal = MakeVec3(0.0f, 1.0f, 0.0f);
    float penetration = 0.0f;

    if (shape_type == scene::ShapeType::Plane) {
        normal = NormalizeOr(Rotate(shape_transform.rotation, MakeVec3(0.0f, 1.0f, 0.0f)),
                             MakeVec3(0.0f, 1.0f, 0.0f));
        const float signed_distance = Dot(Sub(*position, shape_transform.position), normal);
        penetration = particle_radius - signed_distance;
    } else if (shape_type == scene::ShapeType::Sphere) {
        const math::Vec3 delta = Sub(*position, shape_transform.position);
        const float distance = sqrtf(LengthSq(delta));
        penetration = shape_radius + particle_radius - distance;
        const math::Vec3 velocity_normal = NormalizeOr(*velocity, MakeVec3(1.0f, 0.0f, 0.0f));
        normal = NormalizeOr(delta, velocity_normal);
    } else if (shape_type == scene::ShapeType::Box) {
        const math::Vec3 local_position = InverseTransformPoint(shape_transform, *position);
        const math::Vec3 closest = MakeVec3(
            Clamp(local_position.x, -shape_half_extents.x, shape_half_extents.x),
            Clamp(local_position.y, -shape_half_extents.y, shape_half_extents.y),
            Clamp(local_position.z, -shape_half_extents.z, shape_half_extents.z));
        const math::Vec3 local_delta = Sub(local_position, closest);
        const float distance_sq = LengthSq(local_delta);

        if (distance_sq > 1.0e-12f) {
            const float distance = sqrtf(distance_sq);
            penetration = particle_radius - distance;
            normal = NormalizeOr(TransformDirection(shape_transform, local_delta),
                                 MakeVec3(0.0f, 1.0f, 0.0f));
        } else {
            const float dx = shape_half_extents.x - fabsf(local_position.x);
            const float dy = shape_half_extents.y - fabsf(local_position.y);
            const float dz = shape_half_extents.z - fabsf(local_position.z);
            math::Vec3 local_normal = MakeVec3(local_position.x >= 0.0f ? 1.0f : -1.0f,
                                               0.0f,
                                               0.0f);
            float min_axis_depth = dx;
            if (dy < min_axis_depth) {
                min_axis_depth = dy;
                local_normal = MakeVec3(0.0f, local_position.y >= 0.0f ? 1.0f : -1.0f, 0.0f);
            }
            if (dz < min_axis_depth) {
                min_axis_depth = dz;
                local_normal = MakeVec3(0.0f, 0.0f, local_position.z >= 0.0f ? 1.0f : -1.0f);
            }
            penetration = particle_radius + fmaxf(min_axis_depth, 0.0f);
            normal = NormalizeOr(TransformDirection(shape_transform, local_normal),
                                 MakeVec3(0.0f, 1.0f, 0.0f));
        }
    } else if (shape_type == scene::ShapeType::Capsule) {
        const math::Vec3 local_position = InverseTransformPoint(shape_transform, *position);
        const float clamped_y = Clamp(local_position.y, -shape_half_height, shape_half_height);
        const math::Vec3 closest = MakeVec3(0.0f, clamped_y, 0.0f);
        const math::Vec3 local_delta = Sub(local_position, closest);
        const float distance_sq = LengthSq(local_delta);

        if (distance_sq > 1.0e-12f) {
            const float distance = sqrtf(distance_sq);
            penetration = shape_radius + particle_radius - distance;
            normal = NormalizeOr(TransformDirection(shape_transform, local_delta),
                                 MakeVec3(1.0f, 0.0f, 0.0f));
        } else {
            math::Vec3 local_normal = MakeVec3(local_position.x, 0.0f, local_position.z);
            if (LengthSq(local_normal) <= 1.0e-12f) {
                if (fabsf(local_position.y) > shape_half_height) {
                    local_normal = MakeVec3(0.0f, local_position.y >= 0.0f ? 1.0f : -1.0f, 0.0f);
                } else {
                    local_normal = MakeVec3(1.0f, 0.0f, 0.0f);
                }
            }
            penetration = shape_radius + particle_radius;
            normal = NormalizeOr(TransformDirection(shape_transform, local_normal),
                                 MakeVec3(1.0f, 0.0f, 0.0f));
        }
    } else {
        return 0.0f;
    }

    if (penetration <= 0.0f) {
        return 0.0f;
    }

    const math::Vec3 contact_point = *position;
    *position = Add(*position, Scale(normal, penetration));

    const math::Vec3 lever_arm = Sub(contact_point, body_transform.position);
    const math::Vec3 body_angular_velocity =
        (accumulate_rigid_impulses && body_inv_mass > 0.0f)
            ? rigid_angular_velocities[body_id]
            : MakeVec3(0.0f, 0.0f, 0.0f);
    const math::Vec3 rigid_velocity =
        (accumulate_rigid_impulses && body_inv_mass > 0.0f)
            ? Add(rigid_velocities[body_id], Cross(body_angular_velocity, lever_arm))
            : MakeVec3(0.0f, 0.0f, 0.0f);
    const math::Vec3 relative_velocity = Sub(*velocity, rigid_velocity);
    const math::Vec3 angular_jacobian = Cross(lever_arm, normal);
    const math::Vec3 inv_inertia =
        (accumulate_rigid_impulses && body_inv_mass > 0.0f)
            ? rigid_inv_inertias[body_id]
            : MakeVec3(0.0f, 0.0f, 0.0f);
    const float angular_effective_inv_mass =
        Dot(angular_jacobian, ApplyInverseInertia(angular_jacobian, inv_inertia));
    const float total_inv_mass = fmaxf(
        particle_inv_mass + body_inv_mass + angular_effective_inv_mass,
        1.0e-6f);
    const float effective_mass = 1.0f / total_inv_mass;
    const math::Vec3 tangent0 = ChooseContactTangent(normal, relative_velocity);
    const math::Vec3 tangent1 = NormalizeOr(Cross(normal, tangent0),
                                            MakeVec3(0.0f, 0.0f, 1.0f));
    const math::Vec3 tangent_angular_jacobian0 = Cross(lever_arm, tangent0);
    const math::Vec3 tangent_angular_jacobian1 = Cross(lever_arm, tangent1);
    const float tangent0_effective_inv_mass = fmaxf(
        particle_inv_mass + body_inv_mass +
            Dot(tangent_angular_jacobian0,
                ApplyInverseInertia(tangent_angular_jacobian0, inv_inertia)),
        1.0e-6f);
    const float tangent1_effective_inv_mass = fmaxf(
        particle_inv_mass + body_inv_mass +
            Dot(tangent_angular_jacobian1,
                ApplyInverseInertia(tangent_angular_jacobian1, inv_inertia)),
        1.0e-6f);

    float accumulated_normal_impulse = 0.0f;
    float accumulated_tangent_impulse0 = 0.0f;
    float accumulated_tangent_impulse1 = 0.0f;
    if (enable_coupling_warm_start &&
        stored_shape_index != nullptr &&
        stored_normal_impulse != nullptr &&
        *stored_shape_index == shape_index &&
        *stored_normal_impulse > 0.0f) {
        accumulated_normal_impulse = *stored_normal_impulse;
        ++(*warm_start_count);
        *warm_start_magnitude += accumulated_normal_impulse;
        if (coupling_row != nullptr && coupling_row->active &&
            coupling_row->shape_index == shape_index) {
            const math::Vec3 stored_tangent_impulse = Add(
                Scale(coupling_row->tangent0, coupling_row->tangent_impulse_0),
                Scale(coupling_row->tangent1, coupling_row->tangent_impulse_1));
            accumulated_tangent_impulse0 = Clamp(
                Dot(stored_tangent_impulse, tangent0),
                -fmaxf(friction, 0.0f) * accumulated_normal_impulse,
                fmaxf(friction, 0.0f) * accumulated_normal_impulse);
            accumulated_tangent_impulse1 = Clamp(
                Dot(stored_tangent_impulse, tangent1),
                -fmaxf(friction, 0.0f) * accumulated_normal_impulse,
                fmaxf(friction, 0.0f) * accumulated_normal_impulse);
            const float tangent_warm_start =
                fabsf(accumulated_tangent_impulse0) +
                fabsf(accumulated_tangent_impulse1);
            if (tangent_warm_start > 0.0f) {
                ++(*tangent_warm_start_count);
                *tangent_warm_start_magnitude += tangent_warm_start;
                *warm_start_magnitude += tangent_warm_start;
            }
        }
    }

    const float relative_normal_speed = Dot(relative_velocity, normal);
    if (!solve_coupling_rows_on_cuda && relative_normal_speed < 0.0f) {
        const float impulse = -(1.0f + restitution) * relative_normal_speed
            / total_inv_mass;
        accumulated_normal_impulse += impulse;
        *velocity = Add(*velocity, Scale(normal, impulse * particle_inv_mass));

        if (accumulate_rigid_impulses && body_inv_mass > 0.0f) {
            const math::Vec3 rigid_linear_delta = Scale(normal, -impulse * body_inv_mass);
            const math::Vec3 rigid_angular_delta =
                ApplyInverseInertia(Scale(angular_jacobian, -impulse), inv_inertia);
            AddVelocityAtomic(rigid_velocities, body_id, rigid_linear_delta);
            AddVelocityAtomic(rigid_angular_velocities, body_id, rigid_angular_delta);
            ++(*impulse_count);
            *impulse_magnitude += fabsf(impulse);
            *angular_impulse_magnitude += sqrtf(LengthSq(rigid_angular_delta));
        }
    } else if (!solve_coupling_rows_on_cuda && relative_normal_speed < 1.0e-5f) {
        const float positional_impulse = penetration * effective_mass;
        accumulated_normal_impulse += positional_impulse;
        if (accumulate_rigid_impulses && body_inv_mass > 0.0f) {
            const math::Vec3 rigid_linear_delta =
                Scale(normal, -(positional_impulse * body_inv_mass));
            const math::Vec3 rigid_angular_delta =
                ApplyInverseInertia(Scale(angular_jacobian, -positional_impulse), inv_inertia);
            AddVelocityAtomic(rigid_velocities, body_id, rigid_linear_delta);
            AddVelocityAtomic(rigid_angular_velocities, body_id, rigid_angular_delta);
            ++(*impulse_count);
            *impulse_magnitude += fabsf(positional_impulse);
            *angular_impulse_magnitude += sqrtf(LengthSq(rigid_angular_delta));
        }
    }

    if (coupling_row != nullptr) {
        CudaParticleCouplingConstraintRow row{};
        row.active = true;
        row.particle_index = particle_index;
        row.shape_index = shape_index;
        row.body_id = body_id;
        row.normal = normal;
        row.contact_point = contact_point;
        row.particle_linear_jacobian = normal;
        row.body_linear_jacobian = Scale(normal, -1.0f);
        row.body_angular_jacobian = Scale(angular_jacobian, -1.0f);
        row.tangent0 = tangent0;
        row.tangent1 = tangent1;
        row.body_tangent_angular_jacobian0 =
            Scale(tangent_angular_jacobian0, -1.0f);
        row.body_tangent_angular_jacobian1 =
            Scale(tangent_angular_jacobian1, -1.0f);
        row.rhs = solve_coupling_rows_on_cuda
            ? (relative_normal_speed < 0.0f
                   ? -restitution * relative_normal_speed
                   : 0.0f)
            : (relative_normal_speed < 0.0f
                   ? -(1.0f + restitution) * relative_normal_speed
                   : penetration);
        row.position_error = penetration;
        row.effective_mass = effective_mass;
        row.tangent_effective_mass0 = 1.0f / tangent0_effective_inv_mass;
        row.tangent_effective_mass1 = 1.0f / tangent1_effective_inv_mass;
        row.friction = fmaxf(friction, 0.0f);
        row.normal_impulse = accumulated_normal_impulse;
        row.tangent_impulse_0 = accumulated_tangent_impulse0;
        row.tangent_impulse_1 = accumulated_tangent_impulse1;
        *coupling_row = row;
    }

    if (stored_normal_impulse != nullptr && stored_shape_index != nullptr) {
        *stored_normal_impulse = accumulated_normal_impulse;
        *stored_shape_index = shape_index;
        if (touched_slot != nullptr) {
            *touched_slot = true;
        }
        *max_coupling_normal_impulse =
            fmaxf(*max_coupling_normal_impulse, accumulated_normal_impulse);
    }

    const float inv_dt = dt > 0.0f ? 1.0f / dt : 0.0f;
    *force_magnitude += accumulated_normal_impulse * inv_dt;
    *torque_magnitude += sqrtf(LengthSq(angular_jacobian)) *
        accumulated_normal_impulse * inv_dt;

    if (!solve_coupling_rows_on_cuda) {
        ApplyFriction(normal, friction, velocity);
    }
    ++(*contact_count);
    return penetration;
}

__global__ void IntegrateAndCoupleParticlesAgainstDeviceWorldKernel(
    uint32_t particle_count,
    math::Vec3* particle_positions,
    math::Vec3* particle_velocities,
    const float* particle_inv_masses,
    const float* particle_radii,
    uint32_t shape_count,
    const math::Transform* body_poses,
    math::Vec3* body_linear_velocities,
    math::Vec3* body_angular_velocities,
    const float* body_inv_masses,
    const math::Vec3* body_inv_inertias,
    const scene::ShapeType* shape_types,
    const scene::BodyId* shape_body_ids,
    const math::Transform* shape_local_transforms,
    const math::Vec3* shape_half_extents,
    const float* shape_radii,
    const float* shape_half_heights,
    math::Vec3 gravity,
    float dt,
    float friction,
    float restitution,
    bool accumulate_rigid_impulses,
    bool solve_coupling_rows_on_cuda,
    bool enable_coupling_warm_start,
    float* coupling_normal_impulses,
    uint32_t* coupling_shape_indices,
    CudaParticleCouplingConstraintRow* coupling_rows,
    ParticleDiagnostics* diagnostics) {
    const uint32_t particle_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_index >= particle_count) {
        return;
    }

    ParticleDiagnostics diag{};
    const float particle_inv_mass = particle_inv_masses[particle_index];
    const float particle_radius = particle_radii[particle_index];
    math::Vec3 position = particle_positions[particle_index];
    math::Vec3 velocity = particle_velocities[particle_index];

    if (particle_inv_mass > 0.0f) {
        float max_penetration = 0.0f;
        for (uint32_t shape_index = 0u; shape_index < shape_count; ++shape_index) {
            const scene::BodyId body_id = shape_body_ids[shape_index];
            const math::Transform shape_transform =
                Compose(body_poses[body_id], shape_local_transforms[shape_index]);
            max_penetration = fmaxf(
                max_penetration,
                DeviceWorldShapeResidual(shape_types[shape_index],
                                         shape_transform,
                                         shape_half_extents[shape_index],
                                         shape_radii[shape_index],
                                         shape_half_heights[shape_index],
                                         particle_radius,
                                         position));
        }

        velocity = Add(velocity, Scale(gravity, dt));
        position = Add(position, Scale(velocity, dt));

        uint32_t contact_count = 0u;
        uint32_t impulse_count = 0u;
        uint32_t warm_start_count = 0u;
        uint32_t tangent_warm_start_count = 0u;
        float impulse_magnitude = 0.0f;
        float angular_impulse_magnitude = 0.0f;
        float warm_start_magnitude = 0.0f;
        float tangent_warm_start_magnitude = 0.0f;
        float max_coupling_normal_impulse = 0.0f;
        float force_magnitude = 0.0f;
        float torque_magnitude = 0.0f;
        bool touched_slots[kCudaParticleCouplingSlotsPerParticle] = {};
        float* stored_normal_impulses =
            coupling_normal_impulses != nullptr
                ? &coupling_normal_impulses[
                      particle_index * kCudaParticleCouplingSlotsPerParticle]
                : nullptr;
        uint32_t* stored_shape_indices =
            coupling_shape_indices != nullptr
                ? &coupling_shape_indices[
                      particle_index * kCudaParticleCouplingSlotsPerParticle]
                : nullptr;
        CudaParticleCouplingConstraintRow* stored_rows =
            coupling_rows != nullptr
                ? &coupling_rows[particle_index * kCudaParticleCouplingSlotsPerParticle]
                : nullptr;
        for (uint32_t shape_index = 0u; shape_index < shape_count; ++shape_index) {
            const scene::BodyId body_id = shape_body_ids[shape_index];
            const math::Transform body_transform = body_poses[body_id];
            const math::Transform shape_transform =
                Compose(body_transform, shape_local_transforms[shape_index]);
            const float body_inv_mass = body_inv_masses[body_id];
            const uint32_t coupling_slot =
                FindCouplingSlot(shape_index, stored_shape_indices);
            float* stored_normal_impulse =
                (stored_normal_impulses != nullptr &&
                 coupling_slot != kInvalidCudaParticleCouplingShape)
                    ? &stored_normal_impulses[coupling_slot]
                    : nullptr;
            uint32_t* stored_shape_index =
                (stored_shape_indices != nullptr &&
                 coupling_slot != kInvalidCudaParticleCouplingShape)
                    ? &stored_shape_indices[coupling_slot]
                    : nullptr;
            bool* touched_slot =
                coupling_slot != kInvalidCudaParticleCouplingShape
                    ? &touched_slots[coupling_slot]
                    : nullptr;
            CudaParticleCouplingConstraintRow* coupling_row =
                (stored_rows != nullptr &&
                 coupling_slot != kInvalidCudaParticleCouplingShape)
                    ? &stored_rows[coupling_slot]
                    : nullptr;
            max_penetration = fmaxf(
                max_penetration,
                SolveDeviceWorldShape(shape_types[shape_index],
                                      shape_index,
                                      body_id,
                                      body_transform,
                                      shape_transform,
                                      shape_half_extents[shape_index],
                                      shape_radii[shape_index],
                                      shape_half_heights[shape_index],
                                      body_inv_mass,
                                      particle_inv_mass,
                                      particle_radius,
                                      friction,
                                      restitution,
                                      accumulate_rigid_impulses,
                                      body_linear_velocities,
                                      body_angular_velocities,
                                      body_inv_inertias,
                                      &position,
                                      &velocity,
                                      &contact_count,
                                      &impulse_count,
                                      &impulse_magnitude,
                                      &angular_impulse_magnitude,
                                      solve_coupling_rows_on_cuda,
                                      enable_coupling_warm_start,
                                      stored_normal_impulse,
                                      stored_shape_index,
                                      touched_slot,
                                      coupling_row,
                                      particle_index,
                                      &warm_start_count,
                                      &tangent_warm_start_count,
                                      &warm_start_magnitude,
                                      &tangent_warm_start_magnitude,
                                      &max_coupling_normal_impulse,
                                      &force_magnitude,
                                      &torque_magnitude,
                                      dt));
        }

        uint32_t active_slot_count = 0u;
        if (stored_normal_impulses != nullptr && stored_shape_indices != nullptr) {
            for (uint32_t slot = 0u; slot < kCudaParticleCouplingSlotsPerParticle; ++slot) {
                if (touched_slots[slot]) {
                    ++active_slot_count;
                } else {
                    stored_normal_impulses[slot] = 0.0f;
                    stored_shape_indices[slot] = kInvalidCudaParticleCouplingShape;
                    if (stored_rows != nullptr) {
                        stored_rows[slot] = MakeInactiveCouplingRow();
                    }
                }
            }
        }

        float max_residual = 0.0f;
        for (uint32_t shape_index = 0u; shape_index < shape_count; ++shape_index) {
            const scene::BodyId body_id = shape_body_ids[shape_index];
            const math::Transform shape_transform =
                Compose(body_poses[body_id], shape_local_transforms[shape_index]);
            max_residual = fmaxf(
                max_residual,
                DeviceWorldShapeResidual(shape_types[shape_index],
                                         shape_transform,
                                         shape_half_extents[shape_index],
                                         shape_radii[shape_index],
                                         shape_half_heights[shape_index],
                                         particle_radius,
                                         position));
        }

        diag.contact_count = contact_count;
        diag.rigid_impulse_count = impulse_count;
        diag.coupling_active_slot_count = active_slot_count;
        diag.coupling_warm_start_count = warm_start_count;
        diag.coupling_tangent_warm_start_count = tangent_warm_start_count;
        diag.max_penetration = max_penetration;
        diag.max_penetration_after_solve = max_residual;
        diag.rigid_impulse_magnitude = impulse_magnitude;
        diag.rigid_angular_impulse_magnitude = angular_impulse_magnitude;
        diag.coupling_warm_start_impulse_magnitude = warm_start_magnitude;
        diag.coupling_tangent_warm_start_impulse_magnitude =
            tangent_warm_start_magnitude;
        diag.max_coupling_normal_impulse = max_coupling_normal_impulse;
        diag.coupling_force_magnitude = force_magnitude;
        diag.coupling_torque_magnitude = torque_magnitude;

        particle_positions[particle_index] = position;
        particle_velocities[particle_index] = velocity;
    }

    const float speed_sq = LengthSq(velocity);
    diag.speed_sq = speed_sq;
    diag.kinetic_energy =
        particle_inv_mass > 0.0f ? 0.5f * (1.0f / particle_inv_mass) * speed_sq : 0.0f;
    diagnostics[particle_index] = diag;
}

__global__ void SolveParticleCouplingRowsKernel(
    uint32_t particle_count,
    CudaParticleCouplingConstraintRow* rows,
    math::Vec3* particle_velocities,
    const float* particle_inv_masses,
    math::Vec3* body_linear_velocities,
    math::Vec3* body_angular_velocities,
    const float* body_inv_masses,
    const math::Vec3* body_inv_inertias,
    float* coupling_normal_impulses,
    uint32_t* coupling_shape_indices,
    float dt,
    bool apply_cached_impulses,
    CouplingRowDiagnostics* diagnostics) {
    const uint32_t particle_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_index >= particle_count) {
        return;
    }

    math::Vec3 particle_velocity = particle_velocities[particle_index];
    const float particle_inv_mass = particle_inv_masses[particle_index];

    for (uint32_t slot = 0u; slot < kCudaParticleCouplingSlotsPerParticle; ++slot) {
        const uint32_t row_index =
            particle_index * kCudaParticleCouplingSlotsPerParticle + slot;
        CouplingRowDiagnostics diag{};
        CudaParticleCouplingConstraintRow row = rows[row_index];

        if (row.active && row.particle_index == particle_index) {
            diag.active_row_count = 1u;
            const scene::BodyId body_id = row.body_id;
            const float body_inv_mass = body_inv_masses[body_id];
            const math::Vec3 inv_inertia = body_inv_inertias[body_id];

            math::Vec3 body_linear_velocity = body_linear_velocities[body_id];
            math::Vec3 body_angular_velocity = body_angular_velocities[body_id];
            const float cached_normal_impulse = row.normal_impulse;
            const float cached_tangent_impulse0 = row.tangent_impulse_0;
            const float cached_tangent_impulse1 = row.tangent_impulse_1;
            if (apply_cached_impulses) {
                row.normal_impulse = 0.0f;
                row.tangent_impulse_0 = 0.0f;
                row.tangent_impulse_1 = 0.0f;

                ApplyCouplingVelocityImpulse(row.particle_linear_jacobian,
                                             row.body_linear_jacobian,
                                             row.body_angular_jacobian,
                                             cached_normal_impulse,
                                             particle_inv_mass,
                                             body_inv_mass,
                                             inv_inertia,
                                             body_id,
                                             &particle_velocity,
                                             &body_linear_velocity,
                                             &body_angular_velocity,
                                             body_linear_velocities,
                                             body_angular_velocities);
                row.normal_impulse = cached_normal_impulse;
            }

            if (apply_cached_impulses && row.friction > 0.0f &&
                cached_normal_impulse > 0.0f) {
                const float friction_limit =
                    row.friction * cached_normal_impulse;
                const float tangent_impulse0 = Clamp(cached_tangent_impulse0,
                                                     -friction_limit,
                                                     friction_limit);
                const float tangent_impulse1 = Clamp(cached_tangent_impulse1,
                                                     -friction_limit,
                                                     friction_limit);
                ApplyCouplingVelocityImpulse(row.tangent0,
                                             Scale(row.tangent0, -1.0f),
                                             row.body_tangent_angular_jacobian0,
                                             tangent_impulse0,
                                             particle_inv_mass,
                                             body_inv_mass,
                                             inv_inertia,
                                             body_id,
                                             &particle_velocity,
                                             &body_linear_velocity,
                                             &body_angular_velocity,
                                             body_linear_velocities,
                                             body_angular_velocities);
                row.tangent_impulse_0 = tangent_impulse0;
                ApplyCouplingVelocityImpulse(row.tangent1,
                                             Scale(row.tangent1, -1.0f),
                                             row.body_tangent_angular_jacobian1,
                                             tangent_impulse1,
                                             particle_inv_mass,
                                             body_inv_mass,
                                             inv_inertia,
                                             body_id,
                                             &particle_velocity,
                                             &body_linear_velocity,
                                             &body_angular_velocity,
                                             body_linear_velocities,
                                             body_angular_velocities);
                row.tangent_impulse_1 = tangent_impulse1;
            }

            const float jv = CouplingRowNormalVelocity(
                row, particle_velocity, body_linear_velocity, body_angular_velocity);

            const float old_impulse = row.normal_impulse;
            const float lambda = row.effective_mass * (row.rhs - jv);
            const float new_impulse = fmaxf(old_impulse + lambda, 0.0f);
            const float delta = new_impulse - old_impulse;
            float max_row_residual = 0.0f;

            row.normal_impulse = new_impulse;

            if (fabsf(delta) > 1.0e-12f) {
                const math::Vec3 body_angular_delta = ApplyCouplingVelocityImpulse(
                    row.particle_linear_jacobian,
                    row.body_linear_jacobian,
                    row.body_angular_jacobian,
                    delta,
                    particle_inv_mass,
                    body_inv_mass,
                    inv_inertia,
                    body_id,
                    &particle_velocity,
                    &body_linear_velocity,
                    &body_angular_velocity,
                    body_linear_velocities,
                    body_angular_velocities);

                diag.impulse_count = 1u;
                diag.impulse_magnitude = fabsf(delta);
                diag.angular_impulse_magnitude = sqrtf(LengthSq(body_angular_delta));
            }

            float friction_magnitude = 0.0f;
            float friction_angular_magnitude = 0.0f;
            if (row.friction > 0.0f && new_impulse > 0.0f) {
                const float friction_limit = row.friction * new_impulse;
                const float clamped_warm_tangent_impulse0 =
                    Clamp(row.tangent_impulse_0, -friction_limit, friction_limit);
                const float warm_tangent_delta0 =
                    clamped_warm_tangent_impulse0 - row.tangent_impulse_0;
                if (fabsf(warm_tangent_delta0) > 1.0e-12f) {
                    ApplyCouplingVelocityImpulse(row.tangent0,
                                                 Scale(row.tangent0, -1.0f),
                                                 row.body_tangent_angular_jacobian0,
                                                 warm_tangent_delta0,
                                                 particle_inv_mass,
                                                 body_inv_mass,
                                                 inv_inertia,
                                                 body_id,
                                                 &particle_velocity,
                                                 &body_linear_velocity,
                                                 &body_angular_velocity,
                                                 body_linear_velocities,
                                                 body_angular_velocities);
                    row.tangent_impulse_0 = clamped_warm_tangent_impulse0;
                }
                const float clamped_warm_tangent_impulse1 =
                    Clamp(row.tangent_impulse_1, -friction_limit, friction_limit);
                const float warm_tangent_delta1 =
                    clamped_warm_tangent_impulse1 - row.tangent_impulse_1;
                if (fabsf(warm_tangent_delta1) > 1.0e-12f) {
                    ApplyCouplingVelocityImpulse(row.tangent1,
                                                 Scale(row.tangent1, -1.0f),
                                                 row.body_tangent_angular_jacobian1,
                                                 warm_tangent_delta1,
                                                 particle_inv_mass,
                                                 body_inv_mass,
                                                 inv_inertia,
                                                 body_id,
                                                 &particle_velocity,
                                                 &body_linear_velocity,
                                                 &body_angular_velocity,
                                                 body_linear_velocities,
                                                 body_angular_velocities);
                    row.tangent_impulse_1 = clamped_warm_tangent_impulse1;
                }

                float tangent_jv = CouplingRowTangentVelocity(
                    row.tangent0,
                    row.body_tangent_angular_jacobian0,
                    particle_velocity,
                    body_linear_velocity,
                    body_angular_velocity);
                float old_tangent_impulse = row.tangent_impulse_0;
                float tangent_lambda = row.tangent_effective_mass0 * (-tangent_jv);
                float new_tangent_impulse =
                    Clamp(old_tangent_impulse + tangent_lambda,
                          -friction_limit,
                          friction_limit);
                float tangent_delta = new_tangent_impulse - old_tangent_impulse;
                row.tangent_impulse_0 = new_tangent_impulse;

                if (fabsf(tangent_delta) > 1.0e-12f) {
                    const math::Vec3 body_angular_delta =
                        ApplyCouplingVelocityImpulse(
                            row.tangent0,
                            Scale(row.tangent0, -1.0f),
                            row.body_tangent_angular_jacobian0,
                            tangent_delta,
                            particle_inv_mass,
                            body_inv_mass,
                            inv_inertia,
                            body_id,
                            &particle_velocity,
                            &body_linear_velocity,
                            &body_angular_velocity,
                            body_linear_velocities,
                            body_angular_velocities);
                    friction_magnitude += fabsf(tangent_delta);
                    friction_angular_magnitude += sqrtf(LengthSq(body_angular_delta));
                }
                tangent_jv = CouplingRowTangentVelocity(
                    row.tangent0,
                    row.body_tangent_angular_jacobian0,
                    particle_velocity,
                    body_linear_velocity,
                    body_angular_velocity);
                max_row_residual = fmaxf(
                    max_row_residual,
                    CouplingProjectedRowResidual(-tangent_jv,
                                                 row.tangent_effective_mass0,
                                                 new_tangent_impulse,
                                                 -friction_limit,
                                                 friction_limit));

                tangent_jv = CouplingRowTangentVelocity(
                    row.tangent1,
                    row.body_tangent_angular_jacobian1,
                    particle_velocity,
                    body_linear_velocity,
                    body_angular_velocity);
                old_tangent_impulse = row.tangent_impulse_1;
                tangent_lambda = row.tangent_effective_mass1 * (-tangent_jv);
                new_tangent_impulse =
                    Clamp(old_tangent_impulse + tangent_lambda,
                          -friction_limit,
                          friction_limit);
                tangent_delta = new_tangent_impulse - old_tangent_impulse;
                row.tangent_impulse_1 = new_tangent_impulse;

                if (fabsf(tangent_delta) > 1.0e-12f) {
                    const math::Vec3 body_angular_delta =
                        ApplyCouplingVelocityImpulse(
                            row.tangent1,
                            Scale(row.tangent1, -1.0f),
                            row.body_tangent_angular_jacobian1,
                            tangent_delta,
                            particle_inv_mass,
                            body_inv_mass,
                            inv_inertia,
                            body_id,
                            &particle_velocity,
                            &body_linear_velocity,
                            &body_angular_velocity,
                            body_linear_velocities,
                            body_angular_velocities);
                    friction_magnitude += fabsf(tangent_delta);
                    friction_angular_magnitude += sqrtf(LengthSq(body_angular_delta));
                }
                tangent_jv = CouplingRowTangentVelocity(
                    row.tangent1,
                    row.body_tangent_angular_jacobian1,
                    particle_velocity,
                    body_linear_velocity,
                    body_angular_velocity);
                max_row_residual = fmaxf(
                    max_row_residual,
                    CouplingProjectedRowResidual(-tangent_jv,
                                                 row.tangent_effective_mass1,
                                                 new_tangent_impulse,
                                                 -friction_limit,
                                                 friction_limit));

                if (friction_magnitude > 0.0f) {
                    diag.friction_impulse_count = 1u;
                    diag.friction_impulse_magnitude = friction_magnitude;
                    diag.angular_impulse_magnitude += friction_angular_magnitude;
                }
            } else if (row.friction > 0.0f) {
                const float tangent_delta0 = -row.tangent_impulse_0;
                if (fabsf(tangent_delta0) > 1.0e-12f) {
                    ApplyCouplingVelocityImpulse(row.tangent0,
                                                 Scale(row.tangent0, -1.0f),
                                                 row.body_tangent_angular_jacobian0,
                                                 tangent_delta0,
                                                 particle_inv_mass,
                                                 body_inv_mass,
                                                 inv_inertia,
                                                 body_id,
                                                 &particle_velocity,
                                                 &body_linear_velocity,
                                                 &body_angular_velocity,
                                                 body_linear_velocities,
                                                 body_angular_velocities);
                    row.tangent_impulse_0 = 0.0f;
                }
                const float tangent_delta1 = -row.tangent_impulse_1;
                if (fabsf(tangent_delta1) > 1.0e-12f) {
                    ApplyCouplingVelocityImpulse(row.tangent1,
                                                 Scale(row.tangent1, -1.0f),
                                                 row.body_tangent_angular_jacobian1,
                                                 tangent_delta1,
                                                 particle_inv_mass,
                                                 body_inv_mass,
                                                 inv_inertia,
                                                 body_id,
                                                 &particle_velocity,
                                                 &body_linear_velocity,
                                                 &body_angular_velocity,
                                                 body_linear_velocities,
                                                 body_angular_velocities);
                    row.tangent_impulse_1 = 0.0f;
                }
            }

            rows[row_index] = row;
            coupling_normal_impulses[row_index] = new_impulse;
            coupling_shape_indices[row_index] = row.shape_index;

            const float inv_dt = dt > 0.0f ? 1.0f / dt : 0.0f;
            diag.force_magnitude = new_impulse * inv_dt;
            diag.torque_magnitude =
                sqrtf(LengthSq(row.body_angular_jacobian)) * new_impulse * inv_dt;
            diag.max_normal_impulse = new_impulse;
            const float solved_normal_velocity = CouplingRowNormalVelocity(
                row, particle_velocity, body_linear_velocity, body_angular_velocity);
            max_row_residual =
                fmaxf(max_row_residual,
                      CouplingProjectedRowResidual(row.rhs - solved_normal_velocity,
                                                  row.effective_mass,
                                                  new_impulse,
                                                  0.0f,
                                                  3.402823466e+38f));
            diag.max_row_residual = max_row_residual;
        }

        diagnostics[row_index] = diag;
    }

    particle_velocities[particle_index] = particle_velocity;
}

__global__ void ReduceCouplingRowDiagnosticsKernel(
    uint32_t row_count,
    const CouplingRowDiagnostics* diagnostics,
    uint32_t iteration_index,
    CudaParticleStepReport* report) {
    const CudaConstraintRowSchedulerReport scheduler = report->coupling_scheduler_report;
    __shared__ uint32_t shared_impulse_counts[256];
    __shared__ uint32_t shared_friction_impulse_counts[256];
    __shared__ uint32_t shared_active_row_counts[256];
    __shared__ float shared_impulse_magnitudes[256];
    __shared__ float shared_friction_impulse_magnitudes[256];
    __shared__ float shared_angular_impulse_magnitudes[256];
    __shared__ float shared_force_magnitudes[256];
    __shared__ float shared_torque_magnitudes[256];
    __shared__ float shared_max_normal_impulses[256];
    __shared__ float shared_max_row_residuals[256];

    const uint32_t tid = threadIdx.x;
    uint32_t impulse_count = 0u;
    uint32_t friction_impulse_count = 0u;
    uint32_t active_row_count = 0u;
    float impulse_magnitude = 0.0f;
    float friction_impulse_magnitude = 0.0f;
    float angular_impulse_magnitude = 0.0f;
    float force_magnitude = 0.0f;
    float torque_magnitude = 0.0f;
    float max_normal_impulse = 0.0f;
    float max_row_residual = 0.0f;

    for (uint32_t index = tid; index < row_count; index += blockDim.x) {
        const CouplingRowDiagnostics diag = diagnostics[index];
        active_row_count += diag.active_row_count;
        impulse_count += diag.impulse_count;
        friction_impulse_count += diag.friction_impulse_count;
        impulse_magnitude += diag.impulse_magnitude;
        friction_impulse_magnitude += diag.friction_impulse_magnitude;
        angular_impulse_magnitude += diag.angular_impulse_magnitude;
        force_magnitude += diag.force_magnitude;
        torque_magnitude += diag.torque_magnitude;
        max_normal_impulse = fmaxf(max_normal_impulse, diag.max_normal_impulse);
        max_row_residual = fmaxf(max_row_residual, diag.max_row_residual);
    }

    shared_impulse_counts[tid] = impulse_count;
    shared_friction_impulse_counts[tid] = friction_impulse_count;
    shared_active_row_counts[tid] = active_row_count;
    shared_impulse_magnitudes[tid] = impulse_magnitude;
    shared_friction_impulse_magnitudes[tid] = friction_impulse_magnitude;
    shared_angular_impulse_magnitudes[tid] = angular_impulse_magnitude;
    shared_force_magnitudes[tid] = force_magnitude;
    shared_torque_magnitudes[tid] = torque_magnitude;
    shared_max_normal_impulses[tid] = max_normal_impulse;
    shared_max_row_residuals[tid] = max_row_residual;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared_impulse_counts[tid] += shared_impulse_counts[tid + stride];
            shared_friction_impulse_counts[tid] +=
                shared_friction_impulse_counts[tid + stride];
            shared_active_row_counts[tid] += shared_active_row_counts[tid + stride];
            shared_impulse_magnitudes[tid] += shared_impulse_magnitudes[tid + stride];
            shared_friction_impulse_magnitudes[tid] +=
                shared_friction_impulse_magnitudes[tid + stride];
            shared_angular_impulse_magnitudes[tid] +=
                shared_angular_impulse_magnitudes[tid + stride];
            shared_force_magnitudes[tid] += shared_force_magnitudes[tid + stride];
            shared_torque_magnitudes[tid] += shared_torque_magnitudes[tid + stride];
            shared_max_normal_impulses[tid] =
                fmaxf(shared_max_normal_impulses[tid],
                      shared_max_normal_impulses[tid + stride]);
            shared_max_row_residuals[tid] =
                fmaxf(shared_max_row_residuals[tid],
                      shared_max_row_residuals[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0u) {
        report->rigid_impulse_magnitude += shared_impulse_magnitudes[0];
        report->rigid_angular_impulse_magnitude += shared_angular_impulse_magnitudes[0];
        report->rigid_impulse_count += shared_impulse_counts[0];
        report->coupling_row_solver_impulse_count += shared_impulse_counts[0];
        report->coupling_row_solver_impulse_magnitude += shared_impulse_magnitudes[0];
        report->coupling_row_solver_friction_impulse_count +=
            shared_friction_impulse_counts[0];
        report->coupling_row_solver_friction_impulse_magnitude +=
            shared_friction_impulse_magnitudes[0];
        report->coupling_force_magnitude += shared_force_magnitudes[0];
        report->coupling_torque_magnitude += shared_torque_magnitudes[0];
        report->max_coupling_normal_impulse =
            fmaxf(report->max_coupling_normal_impulse, shared_max_normal_impulses[0]);
        report->coupling_row_solver_max_iteration_normal_delta_impulse =
            fmaxf(report->coupling_row_solver_max_iteration_normal_delta_impulse,
                  shared_impulse_magnitudes[0]);
        report->coupling_row_solver_max_iteration_tangent_delta_impulse =
            fmaxf(report->coupling_row_solver_max_iteration_tangent_delta_impulse,
                  shared_friction_impulse_magnitudes[0]);
        report->coupling_row_solver_max_residual =
            fmaxf(report->coupling_row_solver_max_residual,
                  shared_max_row_residuals[0]);
        if (iteration_index < kCudaParticleRowSolverDiagnosticSlots) {
            report->coupling_row_solver_iteration_normal_delta_impulses
                [iteration_index] = shared_impulse_magnitudes[0];
            report->coupling_row_solver_iteration_tangent_delta_impulses
                [iteration_index] = shared_friction_impulse_magnitudes[0];
            report->coupling_row_solver_iteration_max_residuals[iteration_index] =
                shared_max_row_residuals[0];
            report->coupling_row_solver_diagnostic_slot_count =
                report->coupling_row_solver_diagnostic_slot_count > iteration_index
                    ? report->coupling_row_solver_diagnostic_slot_count
                    : iteration_index + 1u;
        }
        CudaConstraintRowSchedulerIterationReport iteration_report;
        iteration_report.active_row_count = shared_active_row_counts[0];
        iteration_report.normal_impulse_count = shared_impulse_counts[0];
        iteration_report.tangent_impulse_count = shared_friction_impulse_counts[0];
        iteration_report.normal_delta_impulse_magnitude =
            shared_impulse_magnitudes[0];
        iteration_report.tangent_delta_impulse_magnitude =
            shared_friction_impulse_magnitudes[0];
        iteration_report.max_normal_delta_impulse = shared_impulse_magnitudes[0];
        iteration_report.max_tangent_delta_impulse =
            shared_friction_impulse_magnitudes[0];
        iteration_report.max_residual = shared_max_row_residuals[0];
        iteration_report.diagnostic_slot_count =
            report->coupling_row_solver_diagnostic_slot_count;
        CudaConstraintRowSchedulerReport updated_scheduler = scheduler;
        AccumulateCudaConstraintRowSchedulerIteration(updated_scheduler,
                                                     iteration_report);
        report->coupling_scheduler_report = updated_scheduler;
    }
}

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) +
                                 " failed: " +
                                 cudaGetErrorString(result));
    }
}

template <typename T>
phi::Buffer UploadVector(const std::vector<T>& values) {
    phi::Buffer buffer(values.size() * sizeof(T), phi::MemoryKind::Device);
    if (!values.empty()) {
        buffer.CopyFromHost(values.data(), values.size() * sizeof(T));
    }
    return buffer;
}

template <typename T>
std::vector<T> DownloadVector(const phi::Buffer& buffer, uint32_t count) {
    std::vector<T> values(count);
    if (count > 0u) {
        buffer.CopyToHost(values.data(), count * sizeof(T));
    }
    return values;
}

} // namespace

CudaParticleWorld::CudaParticleWorld(uint32_t particle_count,
                                     phi::Buffer positions,
                                     phi::Buffer velocities,
                                     phi::Buffer inv_masses,
                                     phi::Buffer radii,
                                     phi::Buffer phases,
                                     phi::Buffer coupling_normal_impulses,
                                     phi::Buffer coupling_shape_indices,
                                     phi::Buffer coupling_rows)
    : particle_count_(particle_count)
    , positions_(std::move(positions))
    , velocities_(std::move(velocities))
    , inv_masses_(std::move(inv_masses))
    , radii_(std::move(radii))
    , phases_(std::move(phases))
    , coupling_normal_impulses_(std::move(coupling_normal_impulses))
    , coupling_shape_indices_(std::move(coupling_shape_indices))
    , coupling_rows_(std::move(coupling_rows)) {}

std::size_t CudaParticleWorld::DeviceBytes() const {
    return positions_.Size() +
           velocities_.Size() +
           inv_masses_.Size() +
           radii_.Size() +
           phases_.Size() +
           coupling_normal_impulses_.Size() +
           coupling_shape_indices_.Size() +
           coupling_rows_.Size();
}

bool CudaParticleWorld::HasUploadedState() const {
    return particle_count_ == 0u ||
           (positions_.Data() != nullptr &&
            velocities_.Data() != nullptr &&
            inv_masses_.Data() != nullptr &&
            radii_.Data() != nullptr &&
            phases_.Data() != nullptr &&
            coupling_normal_impulses_.Data() != nullptr &&
            coupling_shape_indices_.Data() != nullptr &&
            coupling_rows_.Data() != nullptr);
}

CudaParticleState CudaParticleWorld::DownloadState() const {
    CudaParticleState state;
    state.positions = DownloadVector<math::Vec3>(positions_, particle_count_);
    state.velocities = DownloadVector<math::Vec3>(velocities_, particle_count_);
    state.inv_masses = DownloadVector<float>(inv_masses_, particle_count_);
    state.radii = DownloadVector<float>(radii_, particle_count_);
    state.phases = DownloadVector<uint32_t>(phases_, particle_count_);
    return state;
}

CudaParticleCouplingState CudaParticleWorld::DownloadCouplingState() const {
    CudaParticleCouplingState state;
    state.slot_count_per_particle = kCudaParticleCouplingSlotsPerParticle;
    const uint32_t slot_count = particle_count_ * kCudaParticleCouplingSlotsPerParticle;
    state.normal_impulses = DownloadVector<float>(coupling_normal_impulses_, slot_count);
    state.shape_indices = DownloadVector<uint32_t>(coupling_shape_indices_, slot_count);
    return state;
}

CudaParticleCouplingRowsState CudaParticleWorld::DownloadCouplingRows() const {
    CudaParticleCouplingRowsState state;
    state.slot_count_per_particle = kCudaParticleCouplingSlotsPerParticle;
    const uint32_t slot_count = particle_count_ * kCudaParticleCouplingSlotsPerParticle;
    state.rows = DownloadVector<CudaParticleCouplingConstraintRow>(coupling_rows_, slot_count);
    return state;
}

CudaConstraintRowBufferView CudaParticleWorld::ConstraintRowBuffer() {
    CudaConstraintRowBufferView view;
    view.kind = CudaConstraintRowBufferKind::ParticleRigidCoupling;
    view.layout = CudaConstraintRowLayout::ParticleRigidCouplingSlot;
    view.schedule_mode = CudaConstraintRowScheduleMode::OwnerSerialSweep;
    view.device_rows = coupling_rows_.Data();
    view.row_count = particle_count_ * kCudaParticleCouplingSlotsPerParticle;
    view.owner_count = particle_count_;
    view.rows_per_owner = kCudaParticleCouplingSlotsPerParticle;
    view.row_stride_bytes = sizeof(CudaParticleCouplingConstraintRow);
    return view;
}

math::Vec3* CudaParticleWorld::DevicePositions() {
    return static_cast<math::Vec3*>(positions_.Data());
}

const math::Vec3* CudaParticleWorld::DevicePositions() const {
    return static_cast<const math::Vec3*>(positions_.Data());
}

math::Vec3* CudaParticleWorld::DeviceVelocities() {
    return static_cast<math::Vec3*>(velocities_.Data());
}

const math::Vec3* CudaParticleWorld::DeviceVelocities() const {
    return static_cast<const math::Vec3*>(velocities_.Data());
}

const float* CudaParticleWorld::DeviceInvMasses() const {
    return static_cast<const float*>(inv_masses_.Data());
}

const float* CudaParticleWorld::DeviceRadii() const {
    return static_cast<const float*>(radii_.Data());
}

const uint32_t* CudaParticleWorld::DevicePhases() const {
    return static_cast<const uint32_t*>(phases_.Data());
}

float* CudaParticleWorld::DeviceCouplingNormalImpulses() {
    return static_cast<float*>(coupling_normal_impulses_.Data());
}

const float* CudaParticleWorld::DeviceCouplingNormalImpulses() const {
    return static_cast<const float*>(coupling_normal_impulses_.Data());
}

uint32_t* CudaParticleWorld::DeviceCouplingShapeIndices() {
    return static_cast<uint32_t*>(coupling_shape_indices_.Data());
}

const uint32_t* CudaParticleWorld::DeviceCouplingShapeIndices() const {
    return static_cast<const uint32_t*>(coupling_shape_indices_.Data());
}

CudaParticleCouplingConstraintRow* CudaParticleWorld::DeviceCouplingRows() {
    return static_cast<CudaParticleCouplingConstraintRow*>(coupling_rows_.Data());
}

const CudaParticleCouplingConstraintRow* CudaParticleWorld::DeviceCouplingRows() const {
    return static_cast<const CudaParticleCouplingConstraintRow*>(coupling_rows_.Data());
}

CudaParticleWorld UploadCudaParticleWorld(const CudaParticleSet& particles) {
    const auto count = particles.positions.size();
    if (particles.velocities.size() != count ||
        particles.inv_masses.size() != count ||
        particles.radii.size() != count ||
        particles.phases.size() != count) {
        throw std::runtime_error(
            "UploadCudaParticleWorld requires positions, velocities, inv_masses, radii, and phases to have matching lengths");
    }

    if (count > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error("UploadCudaParticleWorld particle count exceeds uint32_t range");
    }

    const uint32_t particle_count = static_cast<uint32_t>(count);
    const size_t coupling_slot_count =
        count * static_cast<size_t>(kCudaParticleCouplingSlotsPerParticle);
    std::vector<float> coupling_impulses(coupling_slot_count, 0.0f);
    std::vector<uint32_t> coupling_shape_indices(
        coupling_slot_count, kInvalidCudaParticleCouplingShape);
    std::vector<CudaParticleCouplingConstraintRow> coupling_rows(coupling_slot_count);
    for (auto& row : coupling_rows) {
        row.shape_index = kInvalidCudaParticleCouplingShape;
        row.body_id = scene::kInvalidBody;
    }

    return CudaParticleWorld(particle_count,
                             UploadVector(particles.positions),
                             UploadVector(particles.velocities),
                             UploadVector(particles.inv_masses),
                             UploadVector(particles.radii),
                             UploadVector(particles.phases),
                             UploadVector(coupling_impulses),
                             UploadVector(coupling_shape_indices),
                             UploadVector(coupling_rows));
}

CudaParticleStepReport StepCudaParticleWorld(CudaParticleWorld& particle_world,
                                             const CudaParticleStepOptions& options) {
    CudaParticleStepReport report;
    report.particle_count = particle_world.ParticleCount();

    if (options.dt <= 0.0f || options.step_count == 0u ||
        particle_world.ParticleCount() == 0u) {
        return report;
    }
    if (!particle_world.HasUploadedState()) {
        throw std::runtime_error(
            "StepCudaParticleWorld requires uploaded particle state before stepping");
    }

    phi::Buffer diagnostics(
        particle_world.ParticleCount() * sizeof(ParticleDiagnostics),
        phi::MemoryKind::Device);
    phi::Buffer report_buffer(sizeof(CudaParticleStepReport), phi::MemoryKind::Device);

    constexpr uint32_t kBlockSize = 256u;
    const uint32_t block_count =
        (particle_world.ParticleCount() + kBlockSize - 1u) / kBlockSize;

    for (uint32_t step = 0u; step < options.step_count; ++step) {
        IntegrateAndCoupleParticlesKernel<<<block_count, kBlockSize>>>(
            particle_world.ParticleCount(),
            particle_world.DevicePositions(),
            particle_world.DeviceVelocities(),
            particle_world.DeviceInvMasses(),
            particle_world.DeviceRadii(),
            options.gravity,
            options.dt,
            options.plane,
            options.sphere,
            static_cast<ParticleDiagnostics*>(diagnostics.Data()));
        CheckCuda(cudaGetLastError(), "IntegrateAndCoupleParticlesKernel launch");
        ++report.kernel_launch_count;

        ReduceParticleDiagnosticsKernel<<<1u, kBlockSize>>>(
            particle_world.ParticleCount(),
            static_cast<const ParticleDiagnostics*>(diagnostics.Data()),
            static_cast<CudaParticleStepReport*>(report_buffer.Data()));
        CheckCuda(cudaGetLastError(), "ReduceParticleDiagnosticsKernel launch");
        ++report.kernel_launch_count;
    }

    CheckCuda(cudaDeviceSynchronize(), "StepCudaParticleWorld synchronize");
    report_buffer.CopyToHost(&report, sizeof(CudaParticleStepReport));
    report.simulated_step_count = options.step_count;
    report.kernel_launch_count = options.step_count * 2u;
    return report;
}

void RunCudaParticleRigidCouplingScheduler(
    const ParticleRigidCouplingSchedulerInput& input) {
    if (input.config.iterations == 0u ||
        input.row_buffer.kind != CudaConstraintRowBufferKind::ParticleRigidCoupling ||
        input.row_buffer.layout != CudaConstraintRowLayout::ParticleRigidCouplingSlot ||
        input.row_buffer.schedule_mode !=
            CudaConstraintRowScheduleMode::OwnerSerialSweep ||
        input.row_buffer.row_count == 0u ||
        input.particle_count == 0u ||
        input.rows == nullptr ||
        input.report == nullptr) {
        return;
    }

    constexpr uint32_t kBlockSize = 256u;
    const uint32_t row_block_count =
        (input.particle_count + kBlockSize - 1u) / kBlockSize;

    for (uint32_t iteration = 0u; iteration < input.config.iterations; ++iteration) {
        const bool apply_cached_impulses =
            input.config.enable_warm_start && iteration == 0u;

        SolveParticleCouplingRowsKernel<<<row_block_count, kBlockSize>>>(
            input.particle_count,
            input.rows,
            input.particle_velocities,
            input.particle_inv_masses,
            input.body_linear_velocities,
            input.body_angular_velocities,
            input.body_inv_masses,
            input.body_inv_inertias,
            input.coupling_normal_impulses,
            input.coupling_shape_indices,
            input.dt,
            apply_cached_impulses,
            input.diagnostics);
        CheckCuda(cudaGetLastError(), "SolveParticleCouplingRowsKernel launch");
        if (input.host_kernel_launch_count != nullptr) {
            ++(*input.host_kernel_launch_count);
        }
        if (input.host_solver_launch_count != nullptr) {
            ++(*input.host_solver_launch_count);
        }

        if (input.config.reduce_diagnostics) {
            ReduceCouplingRowDiagnosticsKernel<<<1u, kBlockSize>>>(
                input.row_buffer.row_count,
                input.diagnostics,
                input.diagnostic_iteration_base + iteration,
                input.report);
            CheckCuda(cudaGetLastError(), "ReduceCouplingRowDiagnosticsKernel launch");
            if (input.host_kernel_launch_count != nullptr) {
                ++(*input.host_kernel_launch_count);
            }
        }
    }
}

CudaParticleStepReport StepCudaParticlesAgainstDeviceWorld(
    CudaParticleWorld& particle_world,
    DeviceWorld& device_world,
    const CudaParticleDeviceWorldCouplingOptions& options) {
    CudaParticleStepReport report;
    report.particle_count = particle_world.ParticleCount();

    if (options.dt <= 0.0f || options.step_count == 0u ||
        particle_world.ParticleCount() == 0u || device_world.ShapeCount() == 0u) {
        return report;
    }
    if (!particle_world.HasUploadedState()) {
        throw std::runtime_error(
            "StepCudaParticlesAgainstDeviceWorld requires uploaded particle state");
    }
    if (!device_world.HasUploadedState()) {
        throw std::runtime_error(
            "StepCudaParticlesAgainstDeviceWorld requires uploaded DeviceWorld state");
    }

    const CudaConstraintRowBufferView row_buffer = particle_world.ConstraintRowBuffer();
    CudaParticleCouplingConstraintRow* coupling_rows =
        static_cast<CudaParticleCouplingConstraintRow*>(row_buffer.device_rows);
    phi::Buffer diagnostics(
        particle_world.ParticleCount() * sizeof(ParticleDiagnostics),
        phi::MemoryKind::Device);
    phi::Buffer row_diagnostics(
        row_buffer.row_count * sizeof(CouplingRowDiagnostics),
        phi::MemoryKind::Device);
    phi::Buffer report_buffer(sizeof(CudaParticleStepReport), phi::MemoryKind::Device);
    const uint32_t row_solver_iterations = options.solve_coupling_rows_on_cuda
        ? std::max(options.coupling_row_solver_iterations, 1u)
        : 0u;

    constexpr uint32_t kBlockSize = 256u;
    const uint32_t block_count =
        (particle_world.ParticleCount() + kBlockSize - 1u) / kBlockSize;

    for (uint32_t step = 0u; step < options.step_count; ++step) {
        IntegrateAndCoupleParticlesAgainstDeviceWorldKernel<<<block_count, kBlockSize>>>(
            particle_world.ParticleCount(),
            particle_world.DevicePositions(),
            particle_world.DeviceVelocities(),
            particle_world.DeviceInvMasses(),
            particle_world.DeviceRadii(),
            device_world.ShapeCount(),
            device_world.DevicePoses(),
            device_world.DeviceLinearVelocities(),
            device_world.DeviceAngularVelocities(),
            device_world.DeviceInvMasses(),
            device_world.DeviceInvInertias(),
            device_world.DeviceShapeTypes(),
            device_world.DeviceShapeBodyIds(),
            device_world.DeviceShapeLocalTransforms(),
            device_world.DeviceShapeHalfExtents(),
            device_world.DeviceShapeRadii(),
            device_world.DeviceShapeHalfHeights(),
            options.gravity,
            options.dt,
            options.friction,
            options.restitution,
            options.accumulate_rigid_impulses,
            options.solve_coupling_rows_on_cuda,
            options.enable_coupling_warm_start,
            particle_world.DeviceCouplingNormalImpulses(),
            particle_world.DeviceCouplingShapeIndices(),
            coupling_rows,
            static_cast<ParticleDiagnostics*>(diagnostics.Data()));
        CheckCuda(cudaGetLastError(),
                  "IntegrateAndCoupleParticlesAgainstDeviceWorldKernel launch");
        ++report.kernel_launch_count;

        ReduceParticleDiagnosticsWithRigidKernel<<<1u, kBlockSize>>>(
            particle_world.ParticleCount(),
            static_cast<const ParticleDiagnostics*>(diagnostics.Data()),
            step == 0u,
            static_cast<CudaParticleStepReport*>(report_buffer.Data()));
        CheckCuda(cudaGetLastError(), "ReduceParticleDiagnosticsWithRigidKernel launch");
        ++report.kernel_launch_count;

        if (options.solve_coupling_rows_on_cuda) {
            CudaConstraintRowSchedulerConfig scheduler_config;
            scheduler_config.iterations = row_solver_iterations;
            scheduler_config.enable_warm_start = options.enable_coupling_warm_start;
            scheduler_config.reduce_diagnostics = true;

            ParticleRigidCouplingSchedulerInput scheduler_input;
            scheduler_input.row_buffer = row_buffer;
            scheduler_input.config = scheduler_config;
            scheduler_input.particle_count = particle_world.ParticleCount();
            scheduler_input.rows = coupling_rows;
            scheduler_input.particle_velocities = particle_world.DeviceVelocities();
            scheduler_input.particle_inv_masses = particle_world.DeviceInvMasses();
            scheduler_input.body_linear_velocities = device_world.DeviceLinearVelocities();
            scheduler_input.body_angular_velocities = device_world.DeviceAngularVelocities();
            scheduler_input.body_inv_masses = device_world.DeviceInvMasses();
            scheduler_input.body_inv_inertias = device_world.DeviceInvInertias();
            scheduler_input.coupling_normal_impulses =
                particle_world.DeviceCouplingNormalImpulses();
            scheduler_input.coupling_shape_indices =
                particle_world.DeviceCouplingShapeIndices();
            scheduler_input.dt = options.dt;
            scheduler_input.diagnostic_iteration_base = step * row_solver_iterations;
            scheduler_input.diagnostics =
                static_cast<CouplingRowDiagnostics*>(row_diagnostics.Data());
            scheduler_input.report = static_cast<CudaParticleStepReport*>(report_buffer.Data());
            scheduler_input.host_kernel_launch_count = &report.kernel_launch_count;
            scheduler_input.host_solver_launch_count =
                &report.coupling_row_solver_launch_count;
            RunCudaParticleRigidCouplingScheduler(scheduler_input);
        }
    }

    CheckCuda(cudaDeviceSynchronize(), "StepCudaParticlesAgainstDeviceWorld synchronize");
    report_buffer.CopyToHost(&report, sizeof(CudaParticleStepReport));
    report.simulated_step_count = options.step_count;
    report.kernel_launch_count =
        options.step_count * (2u + row_solver_iterations * 2u);
    report.coupling_row_solver_launch_count =
        options.step_count * row_solver_iterations;
    report.coupling_row_solver_iteration_count =
        options.step_count * row_solver_iterations;
    CudaConstraintRowSchedulerConfig report_scheduler_config;
    report_scheduler_config.iterations = row_solver_iterations;
    report_scheduler_config.enable_warm_start = options.enable_coupling_warm_start;
    report_scheduler_config.reduce_diagnostics = options.solve_coupling_rows_on_cuda;
    SetCudaConstraintRowSchedulerMetadata(report.coupling_scheduler_report,
                                          row_buffer,
                                          report_scheduler_config);
    report.coupling_scheduler_report.executed_iterations =
        options.step_count * row_solver_iterations;
    report.coupling_scheduler_report.solver_launch_count =
        options.step_count * row_solver_iterations;
    report.coupling_scheduler_report.diagnostic_launch_count =
        options.step_count * row_solver_iterations;
    return report;
}

} // namespace nuka::runtime::gpu
