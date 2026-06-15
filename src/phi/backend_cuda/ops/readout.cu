// ---------------------------------------------------------------------------
// PHI v2 CUDA backend — M3b readout / reset / snapshot ops:
//   ReadoutContactWrench / ExportObs / ResetEnvs / SnapshotState / RestoreState
//
// ReadoutContactWrench is a LINE-BY-LINE PORT of src/sensor/contact_wrench.cu
// (ContactForceKernel + LinkContactWrenchKernel). ResetEnvs is a LINE-BY-LINE
// PORT of the p03 ResetEnvsKernel (the per-env RL-autoreset primitive).
// SnapshotState / RestoreState are the device-side forms of the legacy snapshot
// D2D copies / Reset() restore (replacing the M3a host-mediated
// Data::Snapshot/Restore): flat stream-ordered cudaMemcpyAsync/MemsetAsync in
// fixed address order — trivially D1.
//
// ExportObs is the M3b minimal whole-body export (no golden gates it yet): per
// env, [base_pose(7) | q(base_link_count) | qdot(base_link_count)] packed into
// the obs_width-float obs_buffer row, deterministic truncation/zero-fill.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/articulation_types.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/registry.cuh"
#include "phi/backend_cuda/ops/union_types.cuh"  // M10: LoadUnionSlot / kUSlot*
#include "sensor/noise/philox.cuh"               // M10: deterministic IC jitter

namespace nuka::phi {

namespace {

using namespace ::nuka::phi::nkops;

using nuka::math::Transform;
using nuka::math::Vec3;

// M10 deterministic IC-jitter helpers (Philox4x32-10, host+device pure fns).
using nuka::sensor::noise::MakeCounter;
using nuka::sensor::noise::Philox4x32_10;
using nuka::sensor::noise::Philox4x32Key;
using nuka::sensor::noise::SplitSeed;
using nuka::sensor::noise::Uint32ToUniform01;

// One symmetric jitter draw in [-half, +half]: u in (0,1] -> (2u-1)*half. The
// (element, seq) pair indexes a distinct Philox stream (injective counter), so
// distinct envs / lanes get independent reproducible draws and half==0 callers
// never reach here (the call sites gate on half != 0).
__forceinline__ __device__ float JitterDraw(Philox4x32Key key,
                                            uint32_t element_idx, uint64_t seq,
                                            float half) {
    const uint32_t word = Philox4x32_10(MakeCounter(element_idx, seq), key).v[0];
    const float u = Uint32ToUniform01(word);  // (0, 1]
    return (2.0f * u - 1.0f) * half;
}

// --- src/sensor/contact_wrench.cu (verbatim) --------------------------------

// Per-slot contact force = lambda / dt (contact-basis components: {Fn,Ft1,Ft2}).
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

// --- p03 ResetEnvsKernel (per-env RL-autoreset primitive, verbatim) ----------

__global__ void ResetEnvsKernel(ArticulationDeviceState state,
                                 const uint32_t* env_ids,
                                 uint32_t id_count,
                                 uint32_t base_link_count,
                                 uint32_t lambda_stride,
                                 const Transform* snapshot_base_pose,
                                 const LinkSpatialVel* snapshot_link_velocity,
                                 const float* snapshot_q,
                                 const float* snapshot_qdot,
                                 float* lambda,
                                 // M7 T1: movable rigid-body restore arm (body
                                 // slice [env*body_count, +body_count), env-major
                                 // — matches SeedInitialState's e*B+b body fill).
                                 uint32_t body_count,
                                 Transform* body_pose,
                                 Vec3* body_linear_velocity,
                                 Vec3* body_angular_velocity,
                                 const Transform* snapshot_body_pose,
                                 const Vec3* snapshot_body_linear_velocity,
                                 const Vec3* snapshot_body_angular_velocity,
                                 // M10 RL-completion: OPTIONAL per-env IC jitter.
                                 // Each draw is gated `if (half != 0)`; an all-zero
                                 // param set reduces to the verbatim snapshot copy
                                 // (byte-identical to the pre-M10 reset).
                                 uint64_t ic_seed,
                                 uint32_t ic_episode,
                                 uint32_t cup_body_index,
                                 float jitter_cup_x,
                                 float jitter_cup_y,
                                 float jitter_base_x,
                                 float jitter_base_y,
                                 float jitter_base_z,
                                 float jitter_q) {
    const uint32_t slot = blockIdx.x;
    if (slot >= id_count) {
        return;
    }
    const uint32_t env = env_ids[slot];
    if (env >= state.articulation_count) {
        return;  // defensive: host rejects OOB ids, but never OOB-write here.
    }
    const uint32_t link_begin = env * base_link_count;
    // Per-env Philox key: seed XOR (episode << 32) so distinct episodes /seeds
    // give independent reproducible streams. Computed once per block; cheap.
    const Philox4x32Key ic_key =
        SplitSeed(ic_seed ^ (static_cast<uint64_t>(ic_episode) << 32));

    // Per-link / per-DOF live state (lanes cover the env's links in fixed order).
    for (uint32_t local = threadIdx.x; local < base_link_count; local += blockDim.x) {
        const uint32_t link = link_begin + local;
        float q = snapshot_q[link];
        // M10: optional per-DOF position jitter (gated; zero => verbatim copy).
        if (jitter_q != 0.0f) {
            q += JitterDraw(ic_key, env, 1000u + local, jitter_q);
        }
        state.q[link] = q;
        state.qdot[link] = snapshot_qdot[link];
        state.qddot[link] = 0.0f;
        state.tau[link] = 0.0f;
        state.link_velocity[link] = snapshot_link_velocity[link];
    }
    // Per-articulation root pose + per-env contact warm-start (lane 0 only).
    if (threadIdx.x == 0u) {
        Transform bp = snapshot_base_pose[env];
        // M10: optional base-position jitter (per-axis distinct seq lanes 1/2/3).
        if (jitter_base_x != 0.0f) bp.position.x += JitterDraw(ic_key, env, 1u, jitter_base_x);
        if (jitter_base_y != 0.0f) bp.position.y += JitterDraw(ic_key, env, 2u, jitter_base_y);
        if (jitter_base_z != 0.0f) bp.position.z += JitterDraw(ic_key, env, 3u, jitter_base_z);
        state.base_pose[env] = bp;
    }
    for (uint32_t i = threadIdx.x; i < lambda_stride; i += blockDim.x) {
        lambda[env * lambda_stride + i] = 0.0f;
    }
    // M7 T1: restore this env's body slice (pose + linear/angular velocity) from
    // the snapshot. body_count == 0 (no bodies) skips the loop entirely, keeping
    // the articulation-only reset byte-identical.
    for (uint32_t b = threadIdx.x; b < body_count; b += blockDim.x) {
        const uint32_t body = env * body_count + b;
        Transform pose = snapshot_body_pose[body];
        // M10: optional cup-XY jitter, ONLY on the cup body slot (per-axis
        // distinct seq lanes 10/11). Gated; zero halves => verbatim copy.
        if (b == cup_body_index) {
            if (jitter_cup_x != 0.0f) pose.position.x += JitterDraw(ic_key, env, 10u, jitter_cup_x);
            if (jitter_cup_y != 0.0f) pose.position.y += JitterDraw(ic_key, env, 11u, jitter_cup_y);
        }
        body_pose[body] = pose;
        body_linear_velocity[body] = snapshot_body_linear_velocity[body];
        body_angular_velocity[body] = snapshot_body_angular_velocity[body];
    }
}

// --- ExportObs (M3b minimal whole-body export) -------------------------------

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

// --- ReadoutUnionContactObs (M10 RL-completion, union-only, additive) --------
//
// ONE BLOCK PER ENV; lanes stride over the env's union slots. No cross-env
// atomics; each finger's normal-impulse is summed in FIXED slot/row order by a
// SINGLE lane (deterministic). Writes a per-env contact-observation slice into
// obs_buffer at obs_offset:
//   obs[obs_offset + fingertip]            = sum_p lambda[base + p]  (p in [0,max_pts))
//                                            over the finger slot's normal rows.
//   obs[obs_offset + n_fingers + foot]     = (ucontact_count[slot] > 0) ? 1 : 0.
// Slot order is FIXED feet->fingers->table (union cook): foot slots [0,n_feet),
// finger slots [n_feet, n_feet+n_fingers). The normal-row impulses live at
// [base, base+max_pts) where base = env*rows_per_env + slot.row_base (the same
// row layout AssembleRows emits: normals first, then friction spokes).
__global__ void ReadoutUnionContactObsKernel(const float* __restrict__ union_slots,
                                             const float* __restrict__ lambda,
                                             const uint32_t* __restrict__ ucontact_count,
                                             uint32_t env_count,
                                             uint32_t union_slot_count,
                                             uint32_t rows_per_env,
                                             uint32_t n_feet,
                                             uint32_t n_fingers,
                                             uint32_t obs_offset,
                                             uint32_t obs_width,
                                             uint32_t max_pts,
                                             float* __restrict__ obs_buffer) {
    const uint32_t env = blockIdx.x;
    if (env >= env_count) {
        return;
    }
    float* obs = obs_buffer + static_cast<size_t>(env) * obs_width;
    const uint32_t slice = n_fingers + n_feet;
    // Zero-fill the obs slice first (lanes cover the slice deterministically).
    for (uint32_t i = threadIdx.x; i < slice; i += blockDim.x) {
        if (obs_offset + i < obs_width) {
            obs[obs_offset + i] = 0.0f;
        }
    }
    __syncthreads();
    // One lane per union slot; the slot's class selects the obs target. Each
    // target index is unique to a slot, so distinct lanes never collide (no
    // atomics needed). Within a slot the lambda sum runs in fixed row order.
    for (uint32_t slot = threadIdx.x; slot < union_slot_count; slot += blockDim.x) {
        const UnionSlotDev u = LoadUnionSlot(union_slots, slot);
        const uint32_t base = env * rows_per_env + u.row_base;
        if (u.cls == kUSlotFingerSphereHull) {
            // Finger force-closure signal: sum the slot's normal-row impulses.
            const uint32_t fingertip = slot - n_feet;  // slot in [n_feet, n_feet+n_fingers)
            if (fingertip < n_fingers && obs_offset + fingertip < obs_width) {
                float sum = 0.0f;
                for (uint32_t p = 0u; p < max_pts; ++p) {
                    sum += lambda[base + p];
                }
                obs[obs_offset + fingertip] = sum;
            }
        } else if (u.cls == kUSlotFootSpherePlane) {
            // Foot contact state: 1.0 if the foot slot has any manifold contact.
            const uint32_t foot = slot;  // slot in [0, n_feet)
            if (foot < n_feet) {
                const uint32_t idx = env * union_slot_count + slot;
                const float in_contact = (ucontact_count[idx] > 0u) ? 1.0f : 0.0f;
                if (obs_offset + n_fingers + foot < obs_width) {
                    obs[obs_offset + n_fingers + foot] = in_contact;
                }
            }
        }
    }
}

// --- op entry points ---------------------------------------------------------

Status OpReadoutContactWrench(const ModelView& /*model*/, const DataView& data,
                              const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const ReadoutContactWrenchParams*>(params);
    if (p == nullptr) {
        return Status::Failed;
    }
    const uint32_t slot_count = p->env_count * p->max_contacts_per_env;
    const uint32_t total_link_count = p->env_count * p->base_link_count;
    if (slot_count == 0u || total_link_count == 0u) {
        return Status::Ok;
    }
    // force = impulse / dt; a non-positive dt yields a defined zero readout.
    const float inv_dt = (p->dt > 0.0f) ? (1.0f / p->dt) : 0.0f;

    constexpr uint32_t kBlock = 128u;
    const uint32_t force_grid = (slot_count + kBlock - 1u) / kBlock;
    LaunchCuda(ContactForceKernel, dim3(force_grid), dim3(kBlock), 0u, stream,
               static_cast<const float*>(data.lambda), slot_count, inv_dt,
               data.contact_force);
    const uint32_t wrench_grid = (total_link_count + kBlock - 1u) / kBlock;
    LaunchCuda(LinkContactWrenchKernel, dim3(wrench_grid), dim3(kBlock), 0u, stream,
               static_cast<const float*>(data.lambda),
               static_cast<const Vec3*>(data.contact_point),
               static_cast<const Vec3*>(data.contact_normal),
               static_cast<const Vec3*>(data.contact_tangent1),
               static_cast<const Vec3*>(data.contact_tangent2),
               static_cast<const uint32_t*>(data.contact_link),
               static_cast<const Transform*>(data.link_pose),
               total_link_count, p->base_link_count, p->max_contacts_per_env,
               inv_dt, reinterpret_cast<float*>(data.link_contact_wrench));
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

Status OpReadoutUnionContactObs(const ModelView& model, const DataView& data,
                                const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const ReadoutUnionContactObsParams*>(params);
    if (p == nullptr) {
        return Status::Failed;
    }
    // Nothing to write when there are no envs, no obs targets, or no obs row.
    if (p->env_count == 0u || (p->n_feet == 0u && p->n_fingers == 0u) ||
        p->obs_width == 0u || p->union_slot_count == 0u) {
        return Status::Ok;
    }
    // ONE BLOCK PER ENV; lanes cover the union slots (cap the block at 256).
    const uint32_t kBlock = p->union_slot_count < 256u ? p->union_slot_count : 256u;
    LaunchCuda(ReadoutUnionContactObsKernel, dim3(p->env_count), dim3(kBlock), 0u,
               stream, static_cast<const float*>(model.union_slots),
               static_cast<const float*>(data.lambda),
               static_cast<const uint32_t*>(data.ucontact_count),
               p->env_count, p->union_slot_count, p->rows_per_env, p->n_feet,
               p->n_fingers, p->obs_offset, p->obs_width, p->max_pts,
               data.obs_buffer);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

Status OpResetEnvs(const ModelView& model, const DataView& data,
                   const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const ResetEnvsParams*>(params);
    if (p == nullptr) {
        return Status::Failed;
    }
    // Proceed if there is ANYTHING per-env to restore: articulation links OR
    // movable bodies. A bodies-only world (base_link_count == 0, body_count > 0)
    // still resets its body slice; the kernel's per-link loop just no-ops then.
    if (p->count == 0u ||
        ((p->articulation_count == 0u || p->base_link_count == 0u) &&
         p->body_count == 0u)) {
        return Status::Ok;
    }
    const ArticulationDeviceState state = MakeArticulationDeviceState(
        model, data, p->articulation_count * p->base_link_count,
        p->articulation_count);
    constexpr uint32_t kResetBlock = 64u;  // covers go2's 13 links / 12 lambda.
    LaunchCuda(ResetEnvsKernel, dim3(p->count), dim3(kResetBlock), 0u, stream,
               state,
               static_cast<const uint32_t*>(data.reset_env_ids),
               p->count, p->base_link_count, p->lambda_stride,
               static_cast<const Transform*>(data.snapshot_base_pose),
               reinterpret_cast<const LinkSpatialVel*>(data.snapshot_link_velocity),
               static_cast<const float*>(data.snapshot_q),
               static_cast<const float*>(data.snapshot_qdot),
               data.lambda,
               // M7 T1: movable rigid-body restore arm.
               p->body_count,
               static_cast<Transform*>(data.body_pose),
               static_cast<Vec3*>(data.body_linear_velocity),
               static_cast<Vec3*>(data.body_angular_velocity),
               static_cast<const Transform*>(data.snapshot_body_pose),
               static_cast<const Vec3*>(data.snapshot_body_linear_velocity),
               static_cast<const Vec3*>(data.snapshot_body_angular_velocity),
               // M10 RL-completion: optional per-env IC jitter (all-zero =>
               // verbatim snapshot copy, byte-identical to the pre-M10 reset).
               p->ic_seed, p->ic_episode, p->cup_body_index,
               p->jitter_cup_xy[0], p->jitter_cup_xy[1],
               p->jitter_base_pos[0], p->jitter_base_pos[1], p->jitter_base_pos[2],
               p->jitter_q);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

Status OpSnapshotState(const ModelView& /*model*/, const DataView& data,
                       const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const SnapshotStateParams*>(params);
    if (p == nullptr) {
        return Status::Failed;
    }
    // Nothing to snapshot only when there are NEITHER links NOR bodies (a
    // bodies-only world like a settled cup-on-table still round-trips bodies).
    if (p->total_link_count == 0u && p->total_body_count == 0u) {
        return Status::Ok;
    }
    const size_t nl = p->total_link_count;
    const size_t ne = p->env_count;
    // Live -> snapshot (flat D2D in fixed order; the legacy creation-time
    // snapshot 1:1: base_pose / link_velocity / q / qdot). Guarded on nl so a
    // bodies-only world (nl == 0) skips the articulation copies but still
    // snapshots bodies below — the articulation copies stay byte-identical.
    if (nl > 0u &&
        (cudaMemcpyAsync(data.snapshot_base_pose, data.base_pose,
                         ne * sizeof(Transform), cudaMemcpyDeviceToDevice,
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
    // M7 T1: APPEND the movable rigid-body snapshot (env-major total body count).
    // Strictly additive — the articulation copies above are untouched.
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
    return Status::Ok;
}

Status OpRestoreState(const ModelView& /*model*/, const DataView& data,
                      const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const RestoreStateParams*>(params);
    if (p == nullptr) {
        return Status::Failed;
    }
    // Nothing to restore only when there are NEITHER links NOR bodies.
    if (p->total_link_count == 0u && p->total_body_count == 0u) {
        return Status::Ok;
    }
    const size_t nl = p->total_link_count;
    const size_t ne = p->env_count;
    // Snapshot -> live + clear carried accumulators (qddot / tau / lambda):
    // the legacy per-env Reset() restore 1:1. Guarded on nl so a
    // bodies-only world (nl == 0) skips the articulation restore but still
    // restores bodies below — the articulation copies stay byte-identical.
    if (nl > 0u &&
        (cudaMemcpyAsync(data.base_pose, data.snapshot_base_pose,
                         ne * sizeof(Transform), cudaMemcpyDeviceToDevice,
                         stream) != cudaSuccess ||
         cudaMemcpyAsync(data.link_velocity, data.snapshot_link_velocity,
                         nl * 6u * sizeof(float), cudaMemcpyDeviceToDevice,
                         stream) != cudaSuccess ||
         cudaMemcpyAsync(data.q, data.snapshot_q, nl * sizeof(float),
                         cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
         cudaMemcpyAsync(data.qdot, data.snapshot_qdot, nl * sizeof(float),
                         cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
         cudaMemsetAsync(data.qddot, 0, nl * sizeof(float), stream) != cudaSuccess ||
         cudaMemsetAsync(data.tau, 0, nl * sizeof(float), stream) != cudaSuccess)) {
        return Status::Failed;
    }
    if (p->row_slot_count > 0u &&
        cudaMemsetAsync(data.lambda, 0,
                        static_cast<size_t>(p->row_slot_count) * sizeof(float),
                        stream) != cudaSuccess) {
        return Status::Failed;
    }
    // M7 T1: APPEND the movable rigid-body restore (snapshot -> live, env-major
    // total body count). Strictly additive — the articulation restore above is
    // untouched. Bodies carry no qddot/tau accumulator, so nothing to clear.
    const size_t nb = p->total_body_count;
    if (nb > 0u &&
        (cudaMemcpyAsync(data.body_pose, data.snapshot_body_pose,
                         nb * sizeof(Transform), cudaMemcpyDeviceToDevice,
                         stream) != cudaSuccess ||
         cudaMemcpyAsync(data.body_linear_velocity,
                         data.snapshot_body_linear_velocity, nb * sizeof(Vec3),
                         cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
         cudaMemcpyAsync(data.body_angular_velocity,
                         data.snapshot_body_angular_velocity, nb * sizeof(Vec3),
                         cudaMemcpyDeviceToDevice, stream) != cudaSuccess)) {
        return Status::Failed;
    }
    return Status::Ok;
}

} // namespace

void RegisterNkReadoutOps() {
    SetCudaOp(NkOp::ReadoutContactWrench, &OpReadoutContactWrench);
    SetCudaOp(NkOp::ExportObs, &OpExportObs);
    SetCudaOp(NkOp::ResetEnvs, &OpResetEnvs);
    SetCudaOp(NkOp::SnapshotState, &OpSnapshotState);
    SetCudaOp(NkOp::RestoreState, &OpRestoreState);
    SetCudaOp(NkOp::ReadoutUnionContactObs, &OpReadoutUnionContactObs);
}

} // namespace nuka::phi
