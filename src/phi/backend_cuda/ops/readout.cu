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

namespace nuka::phi {

namespace {

using namespace ::nuka::phi::nkops;

using nuka::math::Transform;
using nuka::math::Vec3;

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
                                 const Vec3* snapshot_body_angular_velocity) {
    const uint32_t slot = blockIdx.x;
    if (slot >= id_count) {
        return;
    }
    const uint32_t env = env_ids[slot];
    if (env >= state.articulation_count) {
        return;  // defensive: host rejects OOB ids, but never OOB-write here.
    }
    const uint32_t link_begin = env * base_link_count;

    // Per-link / per-DOF live state (lanes cover the env's links in fixed order).
    for (uint32_t local = threadIdx.x; local < base_link_count; local += blockDim.x) {
        const uint32_t link = link_begin + local;
        state.q[link] = snapshot_q[link];
        state.qdot[link] = snapshot_qdot[link];
        state.qddot[link] = 0.0f;
        state.tau[link] = 0.0f;
        state.link_velocity[link] = snapshot_link_velocity[link];
    }
    // Per-articulation root pose + per-env contact warm-start (lane 0 only).
    if (threadIdx.x == 0u) {
        state.base_pose[env] = snapshot_base_pose[env];
    }
    for (uint32_t i = threadIdx.x; i < lambda_stride; i += blockDim.x) {
        lambda[env * lambda_stride + i] = 0.0f;
    }
    // M7 T1: restore this env's body slice (pose + linear/angular velocity) from
    // the snapshot. body_count == 0 (no bodies) skips the loop entirely, keeping
    // the articulation-only reset byte-identical.
    for (uint32_t b = threadIdx.x; b < body_count; b += blockDim.x) {
        const uint32_t body = env * body_count + b;
        body_pose[body] = snapshot_body_pose[body];
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
               static_cast<const Vec3*>(data.snapshot_body_angular_velocity));
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
}

} // namespace nuka::phi
