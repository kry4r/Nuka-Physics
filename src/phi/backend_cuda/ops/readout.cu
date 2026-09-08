// Contact readout and environment state snapshot/restore.

#include <cuda_runtime.h>
#include <cstdio>

#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/articulation_types.cuh"
#include "phi/backend_cuda/ops/rigid_types.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/registry.cuh"
#include "phi/backend_cuda/ops/union_types.cuh"  // NkRow / kNkSideArtic (row solve)
#include "sensor/noise/philox.cuh"

namespace nuka::phi {

namespace {

using namespace ::nuka::phi::nkops;

using nuka::math::Transform;
using nuka::math::Vec3;

// Contact forces use {Fn, Ft1, Ft2}; link wrenches use {force.xyz, torque.xyz}.
constexpr uint32_t kContactForceComponents = 3u;
constexpr uint32_t kLinkWrenchComponents   = 6u;
static_assert(kLinkWrenchComponents == sizeof(::nuka::math::Vec3) * 2u / sizeof(float),
              "link wrench == force(Vec3) + torque(Vec3)");

// PairDriven per-slot row layout (nk_row.hpp): rigid slots carry 4 manifold
// points x 3 spokes, the body-particle tail 1 point x 3 spokes.
constexpr uint32_t kPdPtsPerSlot = nuka::nk::kPairDrivenPtsPerSlot;
constexpr uint32_t kPdRowsPerSlot = nuka::nk::kPairDrivenRowsPerSlot;
constexpr uint32_t kPdParticlePtsPerSlot = nuka::nk::kPairDrivenParticlePtsPerSlot;
constexpr uint32_t kPdParticleRowsPerSlot = nuka::nk::kPairDrivenParticleRowsPerSlot;

// Philox generates deterministic initial-condition perturbations.
using nuka::sensor::noise::MakeCounter;
using nuka::sensor::noise::Philox4x32_10;
using nuka::sensor::noise::Philox4x32Key;
using nuka::sensor::noise::SplitSeed;
using nuka::sensor::noise::Uint32ToUniform01;

// Each (element, seq) identifies a reproducible draw in [-half, +half].
__forceinline__ __device__ float JitterDraw(Philox4x32Key key,
                                            uint32_t element_idx, uint64_t seq,
                                            float half) {
    const uint32_t word = Philox4x32_10(MakeCounter(element_idx, seq), key).v[0];
    const float u = Uint32ToUniform01(word);  // (0, 1]
    return (2.0f * u - 1.0f) * half;
}

// Populate legacy contact geometry fields from ucontact_* (PairDriven manifold).
// Copies first valid manifold point to the per-slot contact_point/normal fields.
__global__ void LegacyContactGeometryKernel(
    const uint32_t* __restrict__ ucontact_count,
    const Vec3* __restrict__ ucontact_point,   // elem:4 per slot
    const Vec3* __restrict__ ucontact_normal,  // elem:4 per slot
    uint32_t slot_count,
    Vec3* __restrict__ out_contact_point,
    Vec3* __restrict__ out_contact_normal) {
    const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= slot_count) return;
    const uint32_t n = ucontact_count[slot];
    if (n > 0u) {
        out_contact_point[slot] = ucontact_point[slot * 4u];
        out_contact_normal[slot] = ucontact_normal[slot * 4u];
    } else {
        out_contact_point[slot] = {0.0f, 0.0f, 0.0f};
        out_contact_normal[slot] = {0.0f, 0.0f, 0.0f};
    }
}

// Sum normal and tangent impulses using each slot's row layout and divide by dt.
// Environment padding is excluded from the per-slot force readout.
__global__ void ContactForceKernel(const float* __restrict__ lambda,
                                    uint32_t slot_count,
                                    uint32_t slots_per_env,
                                    uint32_t rows_per_env,
                                    uint32_t full_row_slot_count,
                                    float inv_dt,
                                    float* __restrict__ out_contact_force,
                                    const NkRow* __restrict__ urows,
                                    const uint32_t* __restrict__ row_cj_link,
                                    const uint32_t* __restrict__ row_cj_link_b,
                                    uint32_t* __restrict__ out_a_kind,
                                    uint32_t* __restrict__ out_b_kind,
                                    uint32_t* __restrict__ out_a_index,
                                    uint32_t* __restrict__ out_b_index) {
    const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= slot_count) {
        return;
    }
    const uint32_t env = slot / slots_per_env;
    const uint32_t local = slot - env * slots_per_env;
    const bool rigid_slot = local < full_row_slot_count;
    const uint32_t row_base =
        env * rows_per_env +
        (rigid_slot ? local * kPdRowsPerSlot
                    : full_row_slot_count * kPdRowsPerSlot +
                          (local - full_row_slot_count) *
                              kPdParticleRowsPerSlot);
    const uint32_t pts = rigid_slot ? kPdPtsPerSlot
                                    : kPdParticlePtsPerSlot;
    const uint32_t t1 = rigid_slot ? kPdPtsPerSlot
                                   : kPdParticlePtsPerSlot;
    float fn = 0.0f, ft1 = 0.0f, ft2 = 0.0f;
    for (uint32_t i = 0u; i < pts; ++i) {
        fn += lambda[row_base + i];
        ft1 += lambda[row_base + t1 + i];
        ft2 += lambda[row_base + 2u * t1 + i];
    }
    const uint32_t base = slot * kContactForceComponents;
    out_contact_force[base + 0u] = fn * inv_dt;
    out_contact_force[base + 1u] = ft1 * inv_dt;
    out_contact_force[base + 2u] = ft2 * inv_dt;
    const NkRow& row = urows[row_base];
    const bool active = (row.flags & nk::nk_row_flags::kActive) != 0u;
    out_a_kind[slot] = active ? row.a.kind : nk::kNkSideStatic;
    out_b_kind[slot] = active ? row.b.kind : nk::kNkSideStatic;
    out_a_index[slot] = !active || row.a.kind == nk::kNkSideStatic ? ~0u
        : row.a.kind == nk::kNkSideArtic ? row_cj_link[row_base] : row.a.index;
    out_b_index[slot] = !active || row.b.kind == nk::kNkSideStatic ? ~0u
        : row.b.kind == nk::kNkSideArtic ? row_cj_link_b[row_base] : row.b.index;
}

// Report the first articulation side's global link, preferring side A.
// Rigid/static or inactive slots report kInvalidLink.
__global__ void ContactLinkKernel(const uint32_t* __restrict__ row_cj_link,
                                   const uint32_t* __restrict__ row_cj_link_b,
                                   uint32_t slot_count,
                                   uint32_t slots_per_env,
                                   uint32_t rows_per_env,
                                   uint32_t full_row_slot_count,
                                   uint32_t* __restrict__ out_contact_link) {
    constexpr uint32_t kInvalidLink = ~0u;
    const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= slot_count) {
        return;
    }
    const uint32_t env = slot / slots_per_env;
    const uint32_t local = slot - env * slots_per_env;
    const bool rigid_slot = local < full_row_slot_count;
    const uint32_t row_base =
        env * rows_per_env +
        (rigid_slot ? local * kPdRowsPerSlot
                    : full_row_slot_count * kPdRowsPerSlot +
                          (local - full_row_slot_count) *
                              kPdParticleRowsPerSlot);
    const uint32_t pts = rigid_slot ? kPdPtsPerSlot
                                    : kPdParticlePtsPerSlot;
    uint32_t link = kInvalidLink;
    for (uint32_t i = 0u; i < pts && link == kInvalidLink; ++i) {
        link = row_cj_link[row_base + i];
    }
    for (uint32_t i = 0u; i < pts && link == kInvalidLink; ++i) {
        link = row_cj_link_b[row_base + i];
    }
    out_contact_link[slot] = link;
}

// Gather both contact endpoints into each link's world-frame wrench.
// Fixed row order gives deterministic force and torque sums without atomics.
__global__ void LinkContactWrenchKernel(const float* __restrict__ lambda,
                                        const NkRow* __restrict__ urows,
                                        const uint32_t* __restrict__ row_cj_link,
                                        const Vec3* __restrict__ row_cj_point,
                                        const Vec3* __restrict__ row_cj_dir,
                                        const uint32_t* __restrict__ row_cj_link_b,
                                        const Vec3* __restrict__ row_cj_point_b,
                                        const Vec3* __restrict__ row_cj_dir_b,
                                        const Transform* __restrict__ link_world_pose,
                                        uint32_t total_link_count,
                                        uint32_t base_link_count,
                                        uint32_t rows_per_env,
                                        float inv_dt,
                                        float* __restrict__ out_link_wrench) {
    const uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= total_link_count) {
        return;
    }
    const uint32_t env = g / base_link_count;
    const uint32_t row_begin = env * rows_per_env;
    const uint32_t row_end = row_begin + rows_per_env;
    const Vec3 origin = link_world_pose[g].position;

    Vec3 force = Vec3::Zero();
    Vec3 torque = Vec3::Zero();
    // Only active rows whose endpoint names this link contribute.
    for (uint32_t rs = row_begin; rs < row_end; ++rs) {
        if (!(urows[rs].flags & nk::nk_row_flags::kActive)) {
            continue;
        }
        const float l = lambda[rs];
        if (row_cj_link[rs] == g) {
            const Vec3 f_slot = row_cj_dir[rs] * (l * inv_dt);
            force += f_slot;
            torque += (row_cj_point[rs] - origin).Cross(f_slot);
        }
        if (row_cj_link_b[rs] == g) {
            const Vec3 f_slot = row_cj_dir_b[rs] * (l * inv_dt);
            force += f_slot;
            torque += (row_cj_point_b[rs] - origin).Cross(f_slot);
        }
    }

    const uint32_t out = g * kLinkWrenchComponents;
    out_link_wrench[out + 0u] = force.x;
    out_link_wrench[out + 1u] = force.y;
    out_link_wrench[out + 2u] = force.z;
    out_link_wrench[out + 3u] = torque.x;
    out_link_wrench[out + 4u] = torque.y;
    out_link_wrench[out + 5u] = torque.z;
}

__device__ void ClearWarmStartPoint(
    uint32_t point_index, uint64_t* cache_pair, uint64_t* cache_feature,
    float* cache_lambda, Vec3* cache_normal, Vec3* cache_tangent1,
    Vec3* cache_tangent2, uint64_t* cache_material, uint32_t* cache_age) {
    cache_pair[point_index] = 0u;
    cache_feature[point_index] = 0u;
    for (uint32_t k = 0u; k < 3u; ++k) cache_lambda[point_index * 3u + k] = 0.0f;
    cache_normal[point_index] = {0.0f, 0.0f, 0.0f};
    cache_tangent1[point_index] = {0.0f, 0.0f, 0.0f};
    cache_tangent2[point_index] = {0.0f, 0.0f, 0.0f};
    cache_material[point_index] = 0u;
    cache_age[point_index] = 0u;
}

__global__ void ResetEnvsKernel(DataView data, ResetEnvsParams p) {
    const uint32_t slot = blockIdx.x;
    if (slot >= p.count) return;
    const uint32_t env = p.use_env_ids ? data.reset_env_ids[slot] : slot;
    if (env >= p.env_count) return;
    const Philox4x32Key key = SplitSeed(p.ic_seed ^ (static_cast<uint64_t>(p.ic_episode) << 32));
    for (uint32_t local = threadIdx.x; local < p.base_link_count; local += blockDim.x) {
        const uint32_t link = env * p.base_link_count + local;
        float q = data.snapshot_q[link];
        if (p.jitter_q != 0.0f) q += JitterDraw(key, env, 1000u + local, p.jitter_q);
        data.q[link] = q;
        data.qdot[link] = data.snapshot_qdot[link];
        data.qddot[link] = 0.0f;
        data.tau[link] = 0.0f;
        data.link_velocity[link] = data.snapshot_link_velocity[link];
        data.qdot_pseudo[link] = 0.0f;
        data.link_velocity_pseudo[link] = {};
        data.link_contact_wrench[link] = {};
    }
    for (uint32_t local = threadIdx.x; local < p.articulations_per_env; local += blockDim.x) {
        const uint32_t articulation = env * p.articulations_per_env + local;
        Transform pose = data.snapshot_base_pose[articulation];
        if (p.jitter_base_pos[0] != 0.0f)
            pose.position.x += JitterDraw(key, articulation, 1u, p.jitter_base_pos[0]);
        if (p.jitter_base_pos[1] != 0.0f)
            pose.position.y += JitterDraw(key, articulation, 2u, p.jitter_base_pos[1]);
        if (p.jitter_base_pos[2] != 0.0f)
            pose.position.z += JitterDraw(key, articulation, 3u, p.jitter_base_pos[2]);
        data.base_pose[articulation] = pose;
    }
    const uint32_t dofs = p.articulations_per_env * p.dofs_per_articulation;
    for (uint32_t local = threadIdx.x; local < dofs; local += blockDim.x)
        data.qdot_pseudo_flat[env * dofs + local] = 0.0f;
    for (uint32_t local = threadIdx.x; local < p.lambda_stride; local += blockDim.x) {
        const uint32_t row = env * p.lambda_stride + local;
        data.lambda[row] = 0.0f;
        data.row_pseudo_lambda[row] = 0.0f;
        reinterpret_cast<NkRow*>(data.urows)[row] = NkRow{};
        data.row_cj_link[row] = ~0u;
        data.row_cj_link_b[row] = ~0u;
        data.row_cj_point[row] = {};
        data.row_cj_point_b[row] = {};
        data.row_cj_dir[row] = {};
        data.row_cj_dir_b[row] = {};
    }
    for (uint32_t local = threadIdx.x; local < p.contact_slot_count; local += blockDim.x) {
        const uint32_t contact = env * p.contact_slot_count + local;
        data.ucontact_count[contact] = 0u;
        data.contact_point[contact] = {};
        data.contact_normal[contact] = {};
        data.contact_depth[contact] = 0.0f;
        data.contact_link[contact] = ~0u;
        data.contact_side_a_kind[contact] = nk::kNkSideStatic;
        data.contact_side_b_kind[contact] = nk::kNkSideStatic;
        data.contact_side_a_index[contact] = ~0u;
        data.contact_side_b_index[contact] = ~0u;
        for (uint32_t component = 0u; component < kContactForceComponents; ++component)
            data.contact_force[contact * kContactForceComponents + component] = 0.0f;
        for (uint32_t point = 0u; point < kPdPtsPerSlot; ++point) {
            ClearWarmStartPoint(contact * kPdPtsPerSlot + point, data.contact_cache_pair,
                data.contact_cache_feature, data.contact_cache_lambda, data.contact_cache_normal,
                data.contact_cache_tangent1, data.contact_cache_tangent2, data.contact_cache_material,
                data.contact_cache_age);
        }
    }
    for (uint32_t local = threadIdx.x; local < p.body_count; local += blockDim.x) {
        const uint32_t body = env * p.body_count + local;
        Transform pose = data.snapshot_body_pose[body];
        if (local == p.jitter_body_index) {
            if (p.jitter_body_xyz[0] != 0.0f)
                pose.position.x += JitterDraw(key, env, 10u, p.jitter_body_xyz[0]);
            if (p.jitter_body_xyz[1] != 0.0f)
                pose.position.y += JitterDraw(key, env, 11u, p.jitter_body_xyz[1]);
            if (p.jitter_body_xyz[2] != 0.0f)
                pose.position.z += JitterDraw(key, env, 12u, p.jitter_body_xyz[2]);
        }
        data.body_pose[body] = pose;
        data.body_linear_velocity[body] = data.snapshot_body_linear_velocity[body];
        data.body_angular_velocity[body] = data.snapshot_body_angular_velocity[body];
        data.body_world_inv_inertia[body] = BodyWorldInverseInertia(
            pose, data.body_inertial_frame[body], data.body_inv_inertia[body]);
        data.body_pseudo_linear_velocity[body] = {};
        data.body_pseudo_angular_velocity[body] = {};
        data.body_force[body] = {};
        data.body_torque[body] = {};
        data.body_gyro_residual[body] = 0.0f;
        data.body_gyro_iterations[body] = data.body_gyro_status[body] = 0u;
        data.mpm_body_reaction[body] = {};
        data.mpm_body_ang_reaction[body] = {};
    }
    for (uint32_t local = threadIdx.x; local < p.particle_count; local += blockDim.x) {
        const uint32_t particle = env * p.particle_count + local;
        data.particle_pos[particle] = data.snapshot_particle_pos[particle];
        data.particle_prev_pos[particle] = data.snapshot_particle_prev_pos[particle];
        data.particle_vel[particle] = data.snapshot_particle_vel[particle];
        data.particle_pseudo_vel[particle] = {};
        if (p.has_particle_grid != 0u) {
            data.grid_neighbor_count[particle] = data.grid_neighbor_offset[particle] = 0u;
            data.grid_neighbor_attempted[particle] = 0u;
        }
        if (data.particle_F != nullptr) {
            for (uint32_t component = 0u; component < 9u; ++component) {
                data.particle_F[particle * 9u + component] = data.snapshot_particle_F[particle * 9u + component];
                data.particle_C[particle * 9u + component] = data.snapshot_particle_C[particle * 9u + component];
            }
            data.particle_plastic[particle] = data.snapshot_particle_plastic[particle];
        }
    }
    if (threadIdx.x == 0u) {
        data.contact_count[env] = 0u;
        data.env_status[env] = 0u;
    }
}

// Export each environment's root pose and joint state to a fixed-width row.

__global__ void ExportObsKernel(const Transform* base_pose,
                                const float* q,
                                const float* qdot,
                                uint32_t env_count,
                                uint32_t base_link_count,
                                uint32_t obs_width,
                                float* obs_buffer) {
    const uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= env_count) {
        return;
    }
    float* out = obs_buffer + static_cast<size_t>(env) * obs_width;
    uint32_t w = 0u;
    const Transform pose = base_pose[env];
    const float pose7[7] = {pose.position.x, pose.position.y, pose.position.z,
                            pose.rotation.w, pose.rotation.x, pose.rotation.y,
                            pose.rotation.z};
    for (uint32_t i = 0u; i < 7u && w < obs_width; ++i) {
        out[w++] = pose7[i];
    }
    const uint32_t link_begin = env * base_link_count;
    for (uint32_t l = 0u; l < base_link_count && w < obs_width; ++l) {
        out[w++] = q[link_begin + l];
    }
    for (uint32_t l = 0u; l < base_link_count && w < obs_width; ++l) {
        out[w++] = qdot[link_begin + l];
    }
    while (w < obs_width) {
        out[w++] = 0.0f;
    }
}

Status OpReadoutContactWrench(const ModelView& /*model*/, const DataView& data,
                              const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const ReadoutContactWrenchParams*>(params);
    if (p == nullptr) {
        return Status::Failed;
    }
    const uint32_t slot_count = p->env_count * p->max_contacts_per_env;
    const uint32_t total_link_count = p->env_count * p->base_link_count;
    if (slot_count == 0u) {
        return Status::Ok;
    }
    // force = impulse / dt; a non-positive dt yields a defined zero readout.
    const float inv_dt = (p->dt > 0.0f) ? (1.0f / p->dt) : 0.0f;

    constexpr uint32_t kBlock = 128u;
    const uint32_t force_grid = (slot_count + kBlock - 1u) / kBlock;

    // Populate legacy contact geometry from ucontact_* (PairDriven manifold).
    LaunchCuda(LegacyContactGeometryKernel, dim3(force_grid), dim3(kBlock), 0u, stream,
               static_cast<const uint32_t*>(data.ucontact_count),
               static_cast<const Vec3*>(data.ucontact_point),
               static_cast<const Vec3*>(data.ucontact_normal),
               slot_count, data.contact_point, data.contact_normal);

    LaunchCuda(ContactForceKernel, dim3(force_grid), dim3(kBlock), 0u, stream,
               static_cast<const float*>(data.lambda), slot_count,
               p->max_contacts_per_env, p->rows_per_env, p->full_row_slot_count,
               inv_dt, data.contact_force, reinterpret_cast<const NkRow*>(data.urows),
               data.row_cj_link, data.row_cj_link_b,
               data.contact_side_a_kind, data.contact_side_b_kind,
               data.contact_side_a_index, data.contact_side_b_index);
    LaunchCuda(ContactLinkKernel, dim3(force_grid), dim3(kBlock), 0u, stream,
               static_cast<const uint32_t*>(data.row_cj_link),
               static_cast<const uint32_t*>(data.row_cj_link_b), slot_count,
               p->max_contacts_per_env, p->rows_per_env, p->full_row_slot_count,
               data.contact_link);
    if (total_link_count != 0u) {
        const uint32_t wrench_grid = (total_link_count + kBlock - 1u) / kBlock;
        LaunchCuda(LinkContactWrenchKernel, dim3(wrench_grid), dim3(kBlock), 0u, stream,
                   static_cast<const float*>(data.lambda),
                   reinterpret_cast<const NkRow*>(data.urows),
                   static_cast<const uint32_t*>(data.row_cj_link),
                   static_cast<const Vec3*>(data.row_cj_point),
                   static_cast<const Vec3*>(data.row_cj_dir),
                   static_cast<const uint32_t*>(data.row_cj_link_b),
                   static_cast<const Vec3*>(data.row_cj_point_b),
                   static_cast<const Vec3*>(data.row_cj_dir_b),
                   static_cast<const Transform*>(data.link_pose),
                   total_link_count, p->base_link_count, p->rows_per_env,
                   inv_dt, reinterpret_cast<float*>(data.link_contact_wrench));
    }
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

Status OpExportObs(const ModelView& /*model*/, const DataView& data,
                   const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const ExportObsParams*>(params);
    if (p == nullptr) {
        return Status::Failed;
    }
    if (p->env_count == 0u || p->obs_width == 0u) {
        return Status::Ok;
    }
    constexpr uint32_t kBlock = 128u;
    const uint32_t grid = (p->env_count + kBlock - 1u) / kBlock;
    LaunchCuda(ExportObsKernel, dim3(grid), dim3(kBlock), 0u, stream,
               static_cast<const Transform*>(data.base_pose),
               static_cast<const float*>(data.q),
               static_cast<const float*>(data.qdot),
               p->env_count, p->base_link_count, p->obs_width, data.obs_buffer);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

Status OpResetEnvs(const ModelView& /*model*/, const DataView& data,
                   const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const ResetEnvsParams*>(params);
    if (p == nullptr) return Status::Failed;
    if (p->count == 0u) return Status::Ok;
    if (p->count > p->env_count || (p->use_env_ids && data.reset_env_ids == nullptr) ||
        static_cast<uint64_t>(p->articulations_per_env) * p->env_count != p->articulation_count)
        return Status::Failed;
    constexpr uint32_t block_size = 128u;
    LaunchCuda(ResetEnvsKernel, dim3(p->count), dim3(block_size), 0u, stream, data, *p);
    return cudaGetLastError() == cudaSuccess ? Status::Ok : Status::Failed;
}

Status OpSnapshotState(const ModelView& /*model*/, const DataView& data,
                       const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const SnapshotStateParams*>(params);
    if (p == nullptr) {
        return Status::Failed;
    }
    // Each state category can exist without the others.
    if (p->total_link_count == 0u && p->total_body_count == 0u &&
        p->total_particle_count == 0u) {
        return Status::Ok;
    }
    const size_t nl = p->total_link_count;
    const size_t na = p->articulation_count;
    // Root and link state have distinct array lengths.
    if (nl > 0u &&
        (cudaMemcpyAsync(data.snapshot_base_pose, data.base_pose,
                         na * sizeof(Transform), cudaMemcpyDeviceToDevice,
                         stream) != cudaSuccess ||
         cudaMemcpyAsync(data.snapshot_link_velocity, data.link_velocity,
                         nl * 6u * sizeof(float), cudaMemcpyDeviceToDevice,
                         stream) != cudaSuccess ||
         cudaMemcpyAsync(data.snapshot_q, data.q, nl * sizeof(float),
                         cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
         cudaMemcpyAsync(data.snapshot_qdot, data.qdot, nl * sizeof(float),
                         cudaMemcpyDeviceToDevice, stream) != cudaSuccess)) {
        return Status::Failed;
    }
    // Snapshot body state in environment order.
    const size_t nb = p->total_body_count;
    if (nb > 0u &&
        (cudaMemcpyAsync(data.snapshot_body_pose, data.body_pose,
                         nb * sizeof(Transform), cudaMemcpyDeviceToDevice,
                         stream) != cudaSuccess ||
         cudaMemcpyAsync(data.snapshot_body_linear_velocity,
                         data.body_linear_velocity, nb * sizeof(Vec3),
                         cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
         cudaMemcpyAsync(data.snapshot_body_angular_velocity,
                         data.body_angular_velocity, nb * sizeof(Vec3),
                         cudaMemcpyDeviceToDevice, stream) != cudaSuccess)) {
        return Status::Failed;
    }
    // Particle position, previous position and velocity are independent state.
    const size_t np = p->total_particle_count;
    if (np > 0u &&
        (cudaMemcpyAsync(data.snapshot_particle_pos, data.particle_pos,
                         np * sizeof(Vec3), cudaMemcpyDeviceToDevice,
                         stream) != cudaSuccess ||
         cudaMemcpyAsync(data.snapshot_particle_prev_pos, data.particle_prev_pos,
                         np * sizeof(Vec3), cudaMemcpyDeviceToDevice,
                         stream) != cudaSuccess ||
         cudaMemcpyAsync(data.snapshot_particle_vel, data.particle_vel,
                         np * sizeof(Vec3), cudaMemcpyDeviceToDevice,
                         stream) != cudaSuccess)) {
        return Status::Failed;
    }
    // F, C and plastic strain are mutable; rest volumes and material IDs are model data.
    if (np > 0u && data.particle_F != nullptr &&
        (cudaMemcpyAsync(data.snapshot_particle_F, data.particle_F,
                         np * 9u * sizeof(float), cudaMemcpyDeviceToDevice,
                         stream) != cudaSuccess ||
         cudaMemcpyAsync(data.snapshot_particle_C, data.particle_C,
                         np * 9u * sizeof(float), cudaMemcpyDeviceToDevice,
                         stream) != cudaSuccess ||
         cudaMemcpyAsync(data.snapshot_particle_plastic, data.particle_plastic,
                         np * sizeof(float), cudaMemcpyDeviceToDevice,
                         stream) != cudaSuccess)) {
        return Status::Failed;
    }
    return Status::Ok;
}

Status OpRestoreState(const ModelView& model, const DataView& data,
                      const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const RestoreStateParams*>(params);
    if (p == nullptr || p->env_count == 0u) return Status::Failed;
    const uint32_t envs = p->env_count;
    if (p->total_link_count % envs || p->articulation_count % envs ||
        p->total_body_count % envs || p->total_particle_count % envs ||
        p->row_slot_count % envs || p->contact_slot_count % envs) return Status::Failed;
    ResetEnvsParams reset{};
    reset.count = envs;
    reset.env_count = envs;
    reset.articulation_count = p->articulation_count;
    reset.articulations_per_env = p->articulation_count / envs;
    reset.dofs_per_articulation = p->dofs_per_articulation;
    reset.base_link_count = p->total_link_count / envs;
    reset.body_count = p->total_body_count / envs;
    reset.particle_count = p->total_particle_count / envs;
    reset.has_particle_grid = p->has_particle_grid;
    reset.lambda_stride = p->row_slot_count / envs;
    reset.contact_slot_count = p->contact_slot_count / envs;
    return OpResetEnvs(model, data, &reset, stream);
}

} // namespace

void RegisterNkReadoutOps() {
    SetCudaOp(NkOp::ReadoutContactWrench, &OpReadoutContactWrench);
    SetCudaOp(NkOp::ExportObs, &OpExportObs);
    SetCudaOp(NkOp::ResetEnvs, &OpResetEnvs);
    SetCudaOp(NkOp::SnapshotState, &OpSnapshotState);
    SetCudaOp(NkOp::RestoreState, &OpRestoreState);
}

} // namespace nuka::phi
