// ---------------------------------------------------------------------------
// PHI v2 CUDA backend — M3b readout / reset / snapshot ops:
//   ReadoutContactWrench / ExportObs / ResetEnvs / SnapshotState / RestoreState
//
// ReadoutContactWrench sums the per-link net contact wrench over the ONE general
// (PairDriven) solved row buffer (urows + per-row lambda + the row_cj_* gather).
// ResetEnvs is a LINE-BY-LINE PORT of the p03 ResetEnvsKernel (RL autoreset).
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
#include "phi/backend_cuda/ops/union_types.cuh"  // NkRow / kNkSideArtic (row solve)
#include "sensor/noise/philox.cuh"               // M10: deterministic IC jitter

namespace nuka::phi {

namespace {

using namespace ::nuka::phi::nkops;

using nuka::math::Transform;
using nuka::math::Vec3;

// Contact-basis lambda layout: each contact slot carries {Fn, Ft1, Ft2}; each
// link wrench is a Spatial6 {force.xyz, torque.xyz}. Named so a layout growth
// (e.g. a torsional-friction spoke) is a single edit, not silent stride drift.
// FLAG: AssembleRows / the row solver also encode the 3-component contact basis;
// the divergence-proof home is a shared contact-constants header.
constexpr uint32_t kContactForceComponents = 3u;
constexpr uint32_t kLinkWrenchComponents   = 6u;
static_assert(kLinkWrenchComponents == sizeof(::nuka::math::Vec3) * 2u / sizeof(float),
              "link wrench == force(Vec3) + torque(Vec3)");

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
    const uint32_t base = slot * kContactForceComponents;
    out_contact_force[base + 0u] = lambda[base + 0u] * inv_dt;
    out_contact_force[base + 1u] = lambda[base + 1u] * inv_dt;
    out_contact_force[base + 2u] = lambda[base + 2u] * inv_dt;
}

// Net per-link contact wrench (world frame) over the ONE general (PairDriven)
// solved row buffer. ONE THREAD PER OUTPUT LINK g. The solver writes one impulse
// lambda[rs] per ROW slot; each active row's articulation side carries its owning
// global link / world contact point / world row direction in the chain-J gather
// (row_cj_link/point/dir for side A, row_cj_link_b/point_b/dir_b for side B). The
// world force a link receives from a row is dir*lambda/dt (dir == the row's world
// jlin: +n / -n for the two normal sides, +-spoke for the friction spokes), so the
// per-link force is the sum over every row touching it (normal + friction spokes
// of every manifold point) -- the physically-correct net contact reaction. A row
// whose side is rigid/static carries kInvalidLink in that side's row_cj_link, so
// it never matches g (matching the legacy inactive-slot skip). Both sides are
// summed so a link<->link contact contributes to BOTH links. Fixed row order,
// fp32, no atomics -> D1.
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
    // FIXED row order; gate each side on its tagged link == g. An inactive row
    // (flags bit0 clear) carries lambda == 0 AND kInvalidLink gathers, so it never
    // contributes; explicit flag skip keeps it cheap.
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
                                 uint32_t jitter_body_index,
                                 float jitter_body_x,
                                 float jitter_body_y,
                                 float jitter_body_z,
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
        // M10: optional per-body position jitter, ONLY on the targeted body slot
        // (per-axis distinct seq lanes 10/11/12). Gated; zero halves => verbatim
        // copy. Name-agnostic: targets whichever slot jitter_body_index names.
        if (b == jitter_body_index) {
            if (jitter_body_x != 0.0f) pose.position.x += JitterDraw(ic_key, env, 10u, jitter_body_x);
            if (jitter_body_y != 0.0f) pose.position.y += JitterDraw(ic_key, env, 11u, jitter_body_y);
            if (jitter_body_z != 0.0f) pose.position.z += JitterDraw(ic_key, env, 12u, jitter_body_z);
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

// L1-c: ReadoutUnionContactObsKernel (the union-only per-env contact obs) was
// DELETED here. Grasp/union moved to RL; the general path's per-env contact
// readout is OpReadoutContactWrench over the unified contact buffer.

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

// L1-c: OpReadoutUnionContactObs was DELETED here (see kernel note above).

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
               p->ic_seed, p->ic_episode, p->jitter_body_index,
               p->jitter_body_xyz[0], p->jitter_body_xyz[1], p->jitter_body_xyz[2],
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
}

} // namespace nuka::phi
