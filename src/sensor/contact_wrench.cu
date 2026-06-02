// ---------------------------------------------------------------------------
// nuka::sensor -- per-link contact wrench + per-slot contact force readout
// ---------------------------------------------------------------------------
//
// See contact_wrench.hpp for the math, the torque reference point (LINK FRAME
// ORIGIN, not COM), and the D1 determinism contract (one thread per output link,
// fixed slot order, no atomics).
// ---------------------------------------------------------------------------

#include "sensor/contact_wrench.hpp"

#include "phi/device_context.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace nuka::sensor {

namespace {

using nuka::math::Transform;
using nuka::math::Vec3;

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(result));
    }
}

// Per-slot contact force = lambda / dt (contact-basis components: {Fn,Ft1,Ft2}).
// One thread per slot; pure elementwise scale of the solved impulses. inv_dt is
// host-precomputed (0 when dt <= 0) so a non-positive dt yields a defined zero.
__global__ void ContactForceKernel(const float* __restrict__ lambda,
                                    uint32_t slot_count,
                                    float inv_dt,
                                    float* __restrict__ out_contact_force) {
    const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= slot_count) {
        return;
    }
    const uint32_t base = slot * 3u;
    out_contact_force[base + 0u] = lambda[base + 0u] * inv_dt;
    out_contact_force[base + 1u] = lambda[base + 1u] * inv_dt;
    out_contact_force[base + 2u] = lambda[base + 2u] * inv_dt;
}

// Net per-link contact wrench (world frame). ONE THREAD PER OUTPUT LINK g.
//   env = g / base_link_count; the env owns slots
//     [env*max_foot_contacts_per_env, (env+1)*max_foot_contacts_per_env).
//   Loop those slots in FIXED order; accumulate ONLY slots tagged to g
//   (contact_link[slot] == g). fp32 accumulation, no atomics, fixed order -> D1.
//   F[g]   = sum F_slot,  F_slot = (n*lambda_n + t1*lambda_t1 + t2*lambda_t2)*inv_dt
//   tau[g] = sum (contact_point[slot] - link_origin_world[g]) x F_slot
// link_origin_world[g] = link_world_pose[g].position (the LINK FRAME ORIGIN).
__global__ void LinkContactWrenchKernel(const float* __restrict__ lambda,
                                        const Vec3* __restrict__ contact_point,
                                        const Vec3* __restrict__ contact_normal,
                                        const Vec3* __restrict__ tangent1,
                                        const Vec3* __restrict__ tangent2,
                                        const uint32_t* __restrict__ contact_link,
                                        const Transform* __restrict__ link_world_pose,
                                        uint32_t total_link_count,
                                        uint32_t base_link_count,
                                        uint32_t max_foot_contacts_per_env,
                                        float inv_dt,
                                        float* __restrict__ out_link_wrench) {
    const uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= total_link_count) {
        return;
    }
    const uint32_t env = g / base_link_count;
    const uint32_t slot_begin = env * max_foot_contacts_per_env;
    const uint32_t slot_end = slot_begin + max_foot_contacts_per_env;
    const Vec3 origin = link_world_pose[g].position;

    Vec3 force = Vec3::Zero();
    Vec3 torque = Vec3::Zero();
    // FIXED slot order; gate on the slot's tagged link == g. Inactive slots have
    // contact_link == ~0u (kInvalidLink) which never equals g, so they are skipped.
    for (uint32_t slot = slot_begin; slot < slot_end; ++slot) {
        if (contact_link[slot] != g) {
            continue;
        }
        const uint32_t base = slot * 3u;
        const float ln = lambda[base + 0u];
        const float lt1 = lambda[base + 1u];
        const float lt2 = lambda[base + 2u];
        // World force for this slot: n*lambda_n + t1*lambda_t1 + t2*lambda_t2, /dt.
        const Vec3 n = contact_normal[slot];
        const Vec3 t1 = tangent1[slot];
        const Vec3 t2 = tangent2[slot];
        const Vec3 f_slot = (n * ln + t1 * lt1 + t2 * lt2) * inv_dt;
        // Torque about the link frame origin: r x F, r = contact_point - origin.
        const Vec3 r = contact_point[slot] - origin;
        force += f_slot;
        torque += r.Cross(f_slot);
    }

    const uint32_t out = g * 6u;
    out_link_wrench[out + 0u] = force.x;
    out_link_wrench[out + 1u] = force.y;
    out_link_wrench[out + 2u] = force.z;
    out_link_wrench[out + 3u] = torque.x;
    out_link_wrench[out + 4u] = torque.y;
    out_link_wrench[out + 5u] = torque.z;
}

}  // namespace

void ComputeContactWrench(const phi::DeviceContext& context,
                          const float* lambda,
                          const math::Vec3* contact_point,
                          const math::Vec3* contact_normal,
                          const math::Vec3* tangent1,
                          const math::Vec3* tangent2,
                          const uint32_t* contact_link,
                          const math::Transform* link_world_pose,
                          uint32_t env_count,
                          uint32_t base_link_count,
                          uint32_t max_foot_contacts_per_env,
                          float dt,
                          float* out_contact_force,
                          float* out_link_wrench) {
    const uint32_t slot_count = env_count * max_foot_contacts_per_env;
    const uint32_t total_link_count = env_count * base_link_count;
    if (slot_count == 0u || total_link_count == 0u) {
        return;
    }
    // force = impulse / dt; a non-positive dt yields a defined zero readout.
    const float inv_dt = (dt > 0.0f) ? (1.0f / dt) : 0.0f;

    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();

    constexpr uint32_t kBlock = 128u;

    const uint32_t force_grid = (slot_count + kBlock - 1u) / kBlock;
    ContactForceKernel<<<force_grid, kBlock, 0u, stream>>>(
        lambda, slot_count, inv_dt, out_contact_force);
    CheckCuda(cudaGetLastError(), "ContactForceKernel launch");

    const uint32_t wrench_grid = (total_link_count + kBlock - 1u) / kBlock;
    LinkContactWrenchKernel<<<wrench_grid, kBlock, 0u, stream>>>(
        lambda, contact_point, contact_normal, tangent1, tangent2, contact_link,
        link_world_pose, total_link_count, base_link_count,
        max_foot_contacts_per_env, inv_dt, out_link_wrench);
    CheckCuda(cudaGetLastError(), "LinkContactWrenchKernel launch");
}

}  // namespace nuka::sensor
