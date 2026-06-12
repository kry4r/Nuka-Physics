// ---------------------------------------------------------------------------
// nuka::runtime::coupling::ParticleParticleContact implementation (v0.7 p11, K3)
// ---------------------------------------------------------------------------
//
// The K3 cross-system particle-particle contact CO-STEP. Concatenates the XPBD
// soft + PBF fluid particle positions (and inverse masses) into a UNION scratch,
// builds the p05 cross-system uniform grid over the union, and applies the
// mass-weighted symmetric non-penetration projection as a JACOBI double-buffer
// gather/apply (NO float atomicAdd -- one thread per union particle gathers its
// OWN half over its ascending neighbor list into a per-particle delta scratch,
// then a separate own-index apply pass commits it). The corrected positions are
// scattered back to each world's device buffer. Inert (no kernel launch) when the
// union grid finds no pairs within d_min. See the header for the full contract.
//
// D1 STRONG DETERMINISM: the gather reads only the consistent pre-correction union
// positions in the grid's ascending-by-index neighbor order (a fixed-order
// __fadd_rn reduction); the apply is a pure own-index write. No atomics, no
// thread-order dependence. (Same posture as src/runtime/fluid/pbf_world.cu and
// src/runtime/soft/xpbd_world.cu; deliberately allowlist-free.)
// ---------------------------------------------------------------------------

#include "runtime/coupling/particle_particle_contact.hpp"

#include "collision/particle_uniform_grid.hpp"
#include "math/cuda_vec_ops.cuh"
#include "phi/buffer_legacy.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace nuka::runtime::coupling {

namespace {

namespace mg = ::nuka::math::gpu;
namespace cg = ::nuka::collision::gpu;

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(result));
    }
}

constexpr uint32_t kBlockSize = 128u;

// --- build per-PBF-particle inverse mass scratch ----------------------------
// PBF mass is UNIFORM, so every PBF particle's inverse mass is the single
// 1/particle_mass (0 -> pinned for a non-positive mass). One thread per PBF
// particle, pure own-index write.
__global__ void FillPbfInvMassKernel(uint32_t pbf_count,
                                     float inv_mass,
                                     float* __restrict__ out_inv_mass) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pbf_count) {
        return;
    }
    out_inv_mass[i] = inv_mass;
}

// --- coupling, pass A: compute each union particle's HALF correction --------
// One thread per union particle i. For every penetrating neighbor j (C = |p_i -
// p_j| - d_min < 0), accumulate i's half of the mass-weighted symmetric
// projection:
//     n     = (p_i - p_j)/|p_i - p_j|        (contact normal; grad_{p_i} C)
//     dl    = (-C)/(w_i + w_j + a~)          (XPBD multiplier, lambda warm 0)
//     dp_i += w_i * n * dl
// This is i's OWN half only; particle j independently accumulates its own half
// (dp_j -= w_j * n_ij * dl with n_ij pointing from j) when its thread visits i.
// Because the detection is symmetric (the grid lists j for i AND i for j when
// untruncated), the two halves are consistent and m_i*dp_i + m_j*dp_j = 0 (mass-
// weighted centroid conserved). The reduction reads ONLY the pre-correction union
// positions in the grid's ascending neighbor order (fixed order = D1) and writes
// ONLY out_delta[i] -- no atomics, no race. __fadd_rn pins the rounding so the
// fixed-order sum is bit-reproducible.
__global__ void ComputeHalfCorrectionKernel(
    uint32_t union_count,
    const math::Vec3* __restrict__ positions,
    const float* __restrict__ inv_mass,
    float d_min,
    float alpha_tilde,
    const uint32_t* __restrict__ neighbor_offsets,
    const uint32_t* __restrict__ neighbor_counts,
    const uint32_t* __restrict__ neighbor_indices,
    math::Vec3* __restrict__ out_delta) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= union_count) {
        return;
    }
    const math::Vec3 pi = positions[i];
    const float wi = inv_mass[i];
    const uint32_t base = neighbor_offsets[i];
    const uint32_t n = neighbor_counts[i];

    float dx = 0.0f;
    float dy = 0.0f;
    float dz = 0.0f;
    for (uint32_t k = 0u; k < n; ++k) {
        const uint32_t j = neighbor_indices[base + k];
        const math::Vec3 r = mg::Sub(pi, positions[j]);  // p_i - p_j
        const float dist = sqrtf(mg::Dot(r, r));
        if (dist <= 0.0f) {
            continue;  // coincident: no contact normal (degenerate; skip).
        }
        const float c = dist - d_min;
        if (c >= 0.0f) {
            continue;  // separating: unilateral contact is inactive.
        }
        const float wj = inv_mass[j];
        const float wsum = wi + wj + alpha_tilde;
        if (wsum <= 0.0f) {
            continue;  // both pinned (w_i = w_j = 0, a~ = 0): no correction.
        }
        const float dl = (-c) / wsum;          // XPBD multiplier (lambda warm 0).
        const float inv_dist = 1.0f / dist;    // n = r/dist; fold into the scale.
        const float scale = wi * dl * inv_dist;  // i's half: + w_i * n * dl.
        dx = __fadd_rn(dx, r.x * scale);
        dy = __fadd_rn(dy, r.y * scale);
        dz = __fadd_rn(dz, r.z * scale);
    }
    out_delta[i] = math::Vec3{dx, dy, dz};
}

// --- coupling, pass B: apply the accumulated half corrections ---------------
// One thread per union particle. p_i += dp_i. Pure own-index write -> race-free,
// no atomics, D1. Separated from pass A so the gather sees the consistent pre-
// correction positions (the Jacobi barrier the launch boundary provides).
__global__ void ApplyCorrectionKernel(uint32_t union_count,
                                      math::Vec3* __restrict__ positions,
                                      const math::Vec3* __restrict__ delta) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= union_count) {
        return;
    }
    const math::Vec3 p = positions[i];
    const math::Vec3 d = delta[i];
    positions[i] = math::Vec3{p.x + d.x, p.y + d.y, p.z + d.z};
}

// Build the uniform grid over the union positions. cell_size == query_radius ==
// d_min (so the 27-cell search covers every pair within d_min; satisfies the
// grid's cell_size >= query_radius precondition). The world AABB is derived from
// the union positions on the host (a small download); this and the scratch alloc
// are the per-call perf deferral (named consumer = the 1M-particle perf pass;
// src/runtime/coupling/** is not in the lint hot_path scope).
cg::ParticleGridResult BuildUnionGrid(const phi::DeviceContext& context,
                                      const math::Vec3* device_positions,
                                      uint32_t union_count,
                                      float d_min) {
    std::vector<math::Vec3> host(union_count);
    CheckCuda(cudaMemcpy(host.data(), device_positions,
                         static_cast<std::size_t>(union_count) * sizeof(math::Vec3),
                         cudaMemcpyDeviceToHost),
              "download union positions for grid AABB");
    math::Vec3 lo = host[0];
    math::Vec3 hi = host[0];
    for (const math::Vec3& p : host) {
        lo.x = p.x < lo.x ? p.x : lo.x;
        lo.y = p.y < lo.y ? p.y : lo.y;
        lo.z = p.z < lo.z ? p.z : lo.z;
        hi.x = p.x > hi.x ? p.x : hi.x;
        hi.y = p.y > hi.y ? p.y : hi.y;
        hi.z = p.z > hi.z ? p.z : hi.z;
    }
    const cg::ParticleGridConfig cfg = cg::MakeParticleGridConfig(lo, hi, d_min);
    return cg::BuildParticleUniformGrid(context, device_positions, union_count, cfg,
                                        d_min);
}

} // namespace

ParticleParticleContactReport StepParticleParticleCoupling(
    const phi::DeviceContext& context,
    soft::XpbdWorld& xpbd_world,
    fluid::PbfWorld& pbf_world,
    const ParticleParticleContactParams& params) {
    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();

    ParticleParticleContactReport report;
    const uint32_t n_x = xpbd_world.ParticleCount();
    const uint32_t n_p = pbf_world.ParticleCount();
    const uint32_t union_count = n_x + n_p;
    report.xpbd_particle_count = n_x;
    report.pbf_particle_count = n_p;
    report.union_particle_count = union_count;

    if (params.contact_distance_d_min <= 0.0f || union_count == 0u) {
        report.inert = true;
        return report;  // nothing to couple.
    }

    const uint32_t solver_iterations =
        params.solver_iterations == 0u ? 1u : params.solver_iterations;

    // --- assemble the UNION position + inverse-mass scratch -----------------
    // positions: [ xpbd | pbf ]; inv_mass: [ xpbd inv_masses | pbf uniform ].
    phi::Buffer union_pos(static_cast<std::size_t>(union_count) * sizeof(math::Vec3),
                          phi::MemoryKind::Device);
    phi::Buffer union_inv_mass(static_cast<std::size_t>(union_count) * sizeof(float),
                               phi::MemoryKind::Device);
    phi::Buffer union_delta(static_cast<std::size_t>(union_count) * sizeof(math::Vec3),
                            phi::MemoryKind::Device);
    auto* d_pos = static_cast<math::Vec3*>(union_pos.Data());
    auto* d_inv = static_cast<float*>(union_inv_mass.Data());
    auto* d_delta = static_cast<math::Vec3*>(union_delta.Data());

    if (n_x > 0u) {
        CheckCuda(cudaMemcpyAsync(d_pos, xpbd_world.DevicePositions(),
                                  static_cast<std::size_t>(n_x) * sizeof(math::Vec3),
                                  cudaMemcpyDeviceToDevice, stream),
                  "copy xpbd positions into union");
        CheckCuda(cudaMemcpyAsync(d_inv, xpbd_world.DeviceInvMasses(),
                                  static_cast<std::size_t>(n_x) * sizeof(float),
                                  cudaMemcpyDeviceToDevice, stream),
                  "copy xpbd inverse masses into union");
    }
    if (n_p > 0u) {
        CheckCuda(cudaMemcpyAsync(d_pos + n_x, pbf_world.DevicePositions(),
                                  static_cast<std::size_t>(n_p) * sizeof(math::Vec3),
                                  cudaMemcpyDeviceToDevice, stream),
                  "copy pbf positions into union");
        const float pbf_mass = pbf_world.ParticleMass();
        const float pbf_inv_mass = pbf_mass > 0.0f ? (1.0f / pbf_mass) : 0.0f;
        const uint32_t pbf_blocks = (n_p + kBlockSize - 1u) / kBlockSize;
        FillPbfInvMassKernel<<<pbf_blocks, kBlockSize, 0, stream>>>(
            n_p, pbf_inv_mass, d_inv + n_x);
        CheckCuda(cudaGetLastError(), "FillPbfInvMassKernel launch");
    }

    // --- broadphase: union grid (built on the consistent union positions) ---
    context.stream.Synchronize();  // union scratch must be filled before the AABB read.
    cg::ParticleGridResult grid =
        BuildUnionGrid(context, d_pos, union_count, params.contact_distance_d_min);
    report.total_candidate_neighbors = grid.TotalNeighbors();
    report.truncated_particle_count = grid.TruncatedParticleCount();

    // --- INERT-WHEN-NO-PAIRS: no neighbors within d_min -> no correction ----
    // Host-side early-out (the PBF "skip the launch, don't scale-by-0" posture):
    // no kernel touches either world's positions, so the attached co-step is
    // byte-identical to running XPBD + PBF independently.
    if (grid.TotalNeighbors() == 0u) {
        report.inert = true;
        return report;
    }

    const uint32_t* d_offsets = grid.DeviceNeighborOffsets();
    const uint32_t* d_counts = grid.DeviceNeighborCounts();
    const uint32_t* d_indices = grid.DeviceNeighborIndices();
    const float alpha_tilde = params.compliance_alpha;  // a~ at dt=1 (position-based).
    const uint32_t union_blocks = (union_count + kBlockSize - 1u) / kBlockSize;

    for (uint32_t iter = 0u; iter < solver_iterations; ++iter) {
        // JACOBI: gather every half from the SAME pre-correction union positions
        // (pass A, read-only on positions), THEN apply own-index (pass B). The
        // launch boundary is the barrier that makes this race-free + D1.
        ComputeHalfCorrectionKernel<<<union_blocks, kBlockSize, 0, stream>>>(
            union_count, d_pos, d_inv, params.contact_distance_d_min, alpha_tilde,
            d_offsets, d_counts, d_indices, d_delta);
        CheckCuda(cudaGetLastError(), "ComputeHalfCorrectionKernel launch");
        ++report.kernel_launch_count;

        ApplyCorrectionKernel<<<union_blocks, kBlockSize, 0, stream>>>(
            union_count, d_pos, d_delta);
        CheckCuda(cudaGetLastError(), "ApplyCorrectionKernel launch");
        ++report.kernel_launch_count;
    }

    // --- scatter the corrected union positions back to each world -----------
    if (n_x > 0u) {
        CheckCuda(cudaMemcpyAsync(xpbd_world.DevicePositions(), d_pos,
                                  static_cast<std::size_t>(n_x) * sizeof(math::Vec3),
                                  cudaMemcpyDeviceToDevice, stream),
                  "scatter union positions back to xpbd");
    }
    if (n_p > 0u) {
        CheckCuda(cudaMemcpyAsync(pbf_world.DevicePositions(), d_pos + n_x,
                                  static_cast<std::size_t>(n_p) * sizeof(math::Vec3),
                                  cudaMemcpyDeviceToDevice, stream),
                  "scatter union positions back to pbf");
    }
    context.stream.Synchronize();  // grid + scratch free at scope end; sync first.

    return report;
}

ParticleParticleContactReport StepParticleParticleCoupling(
    soft::XpbdWorld& xpbd_world,
    fluid::PbfWorld& pbf_world,
    const ParticleParticleContactParams& params) {
    auto context = phi::MakeDefaultDeviceContext();
    return StepParticleParticleCoupling(context, xpbd_world, pbf_world, params);
}

} // namespace nuka::runtime::coupling
