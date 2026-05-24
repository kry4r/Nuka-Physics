// ---------------------------------------------------------------------------
// nuka::runtime::gpu::CudaParticleWorld implementation
// ---------------------------------------------------------------------------

#include "runtime/gpu/cuda_particle_world.hpp"

#include <cuda_runtime.h>

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
    float max_penetration;
    float max_penetration_after_solve;
    float speed_sq;
    float kinetic_energy;
    float rigid_impulse_magnitude;
    float rigid_angular_impulse_magnitude;
};

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

__device__ math::Vec3 Rotate(math::Quat q, math::Vec3 v) {
    const math::Vec3 qv = MakeVec3(q.x, q.y, q.z);
    const math::Vec3 t = Scale(Cross(qv, v), 2.0f);
    return Add(Add(v, Scale(t, q.w)), Cross(qv, t));
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

__device__ math::Transform Compose(math::Transform a, math::Transform b) {
    math::Transform result;
    result.position = TransformPoint(a, b.position);
    result.rotation = Mul(a.rotation, b.rotation);
    return result;
}

__device__ float ClampNonNegative(float value) {
    return value < 0.0f ? 0.0f : value;
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
    __shared__ float shared_penetration[256];
    __shared__ float shared_residual[256];
    __shared__ float shared_speed_sq[256];
    __shared__ float shared_energy[256];

    const uint32_t tid = threadIdx.x;
    uint32_t contact_count = 0u;
    float max_penetration = 0.0f;
    float max_residual = 0.0f;
    float max_speed_sq = 0.0f;
    float kinetic_energy = 0.0f;

    for (uint32_t index = tid; index < particle_count; index += blockDim.x) {
        const ParticleDiagnostics diag = diagnostics[index];
        contact_count += diag.contact_count;
        max_penetration = fmaxf(max_penetration, diag.max_penetration);
        max_residual = fmaxf(max_residual, diag.max_penetration_after_solve);
        max_speed_sq = fmaxf(max_speed_sq, diag.speed_sq);
        kinetic_energy += diag.kinetic_energy;
    }

    shared_contacts[tid] = contact_count;
    shared_penetration[tid] = max_penetration;
    shared_residual[tid] = max_residual;
    shared_speed_sq[tid] = max_speed_sq;
    shared_energy[tid] = kinetic_energy;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared_contacts[tid] += shared_contacts[tid + stride];
            shared_penetration[tid] =
                fmaxf(shared_penetration[tid], shared_penetration[tid + stride]);
            shared_residual[tid] = fmaxf(shared_residual[tid], shared_residual[tid + stride]);
            shared_speed_sq[tid] =
                fmaxf(shared_speed_sq[tid], shared_speed_sq[tid + stride]);
            shared_energy[tid] += shared_energy[tid + stride];
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
    }
}

__global__ void ReduceParticleDiagnosticsWithRigidKernel(
    uint32_t particle_count,
    const ParticleDiagnostics* diagnostics,
    CudaParticleStepReport* report) {
    __shared__ uint32_t shared_contacts[256];
    __shared__ uint32_t shared_impulse_contacts[256];
    __shared__ float shared_penetration[256];
    __shared__ float shared_residual[256];
    __shared__ float shared_speed_sq[256];
    __shared__ float shared_energy[256];
    __shared__ float shared_impulse_magnitude[256];
    __shared__ float shared_angular_impulse_magnitude[256];

    const uint32_t tid = threadIdx.x;
    uint32_t contact_count = 0u;
    uint32_t impulse_count = 0u;
    float max_penetration = 0.0f;
    float max_residual = 0.0f;
    float max_speed_sq = 0.0f;
    float kinetic_energy = 0.0f;
    float impulse_magnitude = 0.0f;
    float angular_impulse_magnitude = 0.0f;

    for (uint32_t index = tid; index < particle_count; index += blockDim.x) {
        const ParticleDiagnostics diag = diagnostics[index];
        contact_count += diag.contact_count;
        impulse_count += diag.rigid_impulse_count;
        max_penetration = fmaxf(max_penetration, diag.max_penetration);
        max_residual = fmaxf(max_residual, diag.max_penetration_after_solve);
        max_speed_sq = fmaxf(max_speed_sq, diag.speed_sq);
        kinetic_energy += diag.kinetic_energy;
        impulse_magnitude += diag.rigid_impulse_magnitude;
        angular_impulse_magnitude += diag.rigid_angular_impulse_magnitude;
    }

    shared_contacts[tid] = contact_count;
    shared_impulse_contacts[tid] = impulse_count;
    shared_penetration[tid] = max_penetration;
    shared_residual[tid] = max_residual;
    shared_speed_sq[tid] = max_speed_sq;
    shared_energy[tid] = kinetic_energy;
    shared_impulse_magnitude[tid] = impulse_magnitude;
    shared_angular_impulse_magnitude[tid] = angular_impulse_magnitude;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared_contacts[tid] += shared_contacts[tid + stride];
            shared_impulse_contacts[tid] += shared_impulse_contacts[tid + stride];
            shared_penetration[tid] =
                fmaxf(shared_penetration[tid], shared_penetration[tid + stride]);
            shared_residual[tid] = fmaxf(shared_residual[tid], shared_residual[tid + stride]);
            shared_speed_sq[tid] =
                fmaxf(shared_speed_sq[tid], shared_speed_sq[tid + stride]);
            shared_energy[tid] += shared_energy[tid + stride];
            shared_impulse_magnitude[tid] += shared_impulse_magnitude[tid + stride];
            shared_angular_impulse_magnitude[tid] +=
                shared_angular_impulse_magnitude[tid + stride];
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
    }
}

__device__ float DeviceWorldShapeResidual(scene::ShapeType shape_type,
                                          math::Transform shape_transform,
                                          float shape_radius,
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

    return 0.0f;
}

__device__ float SolveDeviceWorldShape(scene::ShapeType shape_type,
                                       scene::BodyId body_id,
                                       math::Transform body_transform,
                                       math::Transform shape_transform,
                                       float shape_radius,
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
                                       float* angular_impulse_magnitude) {
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
    const float relative_normal_speed = Dot(Sub(*velocity, rigid_velocity), normal);
    if (relative_normal_speed < 0.0f) {
        const float impulse = -(1.0f + restitution) * relative_normal_speed
            / total_inv_mass;
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
    } else if (relative_normal_speed < 1.0e-5f) {
        const float effective_mass = 1.0f / total_inv_mass;
        const float positional_impulse = penetration * effective_mass;
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

    ApplyFriction(normal, friction, velocity);
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
    const float* shape_radii,
    math::Vec3 gravity,
    float dt,
    float friction,
    float restitution,
    bool accumulate_rigid_impulses,
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
                                         shape_radii[shape_index],
                                         particle_radius,
                                         position));
        }

        velocity = Add(velocity, Scale(gravity, dt));
        position = Add(position, Scale(velocity, dt));

        uint32_t contact_count = 0u;
        uint32_t impulse_count = 0u;
        float impulse_magnitude = 0.0f;
        float angular_impulse_magnitude = 0.0f;
        for (uint32_t shape_index = 0u; shape_index < shape_count; ++shape_index) {
            const scene::BodyId body_id = shape_body_ids[shape_index];
            const math::Transform body_transform = body_poses[body_id];
            const math::Transform shape_transform =
                Compose(body_transform, shape_local_transforms[shape_index]);
            const float body_inv_mass = body_inv_masses[body_id];
            max_penetration = fmaxf(
                max_penetration,
                SolveDeviceWorldShape(shape_types[shape_index],
                                      body_id,
                                      body_transform,
                                      shape_transform,
                                      shape_radii[shape_index],
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
                                      &angular_impulse_magnitude));
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
                                         shape_radii[shape_index],
                                         particle_radius,
                                         position));
        }

        diag.contact_count = contact_count;
        diag.rigid_impulse_count = impulse_count;
        diag.max_penetration = max_penetration;
        diag.max_penetration_after_solve = max_residual;
        diag.rigid_impulse_magnitude = impulse_magnitude;
        diag.rigid_angular_impulse_magnitude = angular_impulse_magnitude;

        particle_positions[particle_index] = position;
        particle_velocities[particle_index] = velocity;
    }

    const float speed_sq = LengthSq(velocity);
    diag.speed_sq = speed_sq;
    diag.kinetic_energy =
        particle_inv_mass > 0.0f ? 0.5f * (1.0f / particle_inv_mass) * speed_sq : 0.0f;
    diagnostics[particle_index] = diag;
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
                                     phi::Buffer phases)
    : particle_count_(particle_count)
    , positions_(std::move(positions))
    , velocities_(std::move(velocities))
    , inv_masses_(std::move(inv_masses))
    , radii_(std::move(radii))
    , phases_(std::move(phases)) {}

std::size_t CudaParticleWorld::DeviceBytes() const {
    return positions_.Size() +
           velocities_.Size() +
           inv_masses_.Size() +
           radii_.Size() +
           phases_.Size();
}

bool CudaParticleWorld::HasUploadedState() const {
    return particle_count_ == 0u ||
           (positions_.Data() != nullptr &&
            velocities_.Data() != nullptr &&
            inv_masses_.Data() != nullptr &&
            radii_.Data() != nullptr &&
            phases_.Data() != nullptr);
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

    return CudaParticleWorld(static_cast<uint32_t>(count),
                             UploadVector(particles.positions),
                             UploadVector(particles.velocities),
                             UploadVector(particles.inv_masses),
                             UploadVector(particles.radii),
                             UploadVector(particles.phases));
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

    phi::Buffer diagnostics(
        particle_world.ParticleCount() * sizeof(ParticleDiagnostics),
        phi::MemoryKind::Device);
    phi::Buffer report_buffer(sizeof(CudaParticleStepReport), phi::MemoryKind::Device);

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
            device_world.DeviceShapeRadii(),
            options.gravity,
            options.dt,
            options.friction,
            options.restitution,
            options.accumulate_rigid_impulses,
            static_cast<ParticleDiagnostics*>(diagnostics.Data()));
        CheckCuda(cudaGetLastError(),
                  "IntegrateAndCoupleParticlesAgainstDeviceWorldKernel launch");
        ++report.kernel_launch_count;

        ReduceParticleDiagnosticsWithRigidKernel<<<1u, kBlockSize>>>(
            particle_world.ParticleCount(),
            static_cast<const ParticleDiagnostics*>(diagnostics.Data()),
            static_cast<CudaParticleStepReport*>(report_buffer.Data()));
        CheckCuda(cudaGetLastError(), "ReduceParticleDiagnosticsWithRigidKernel launch");
        ++report.kernel_launch_count;
    }

    CheckCuda(cudaDeviceSynchronize(), "StepCudaParticlesAgainstDeviceWorld synchronize");
    report_buffer.CopyToHost(&report, sizeof(CudaParticleStepReport));
    report.simulated_step_count = options.step_count;
    report.kernel_launch_count = options.step_count * 2u;
    return report;
}

} // namespace nuka::runtime::gpu
