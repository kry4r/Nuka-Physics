// ---------------------------------------------------------------------------
// MLS-MPM transfer ops (APIC; Hu et al. 2018) — the INERT round-trip scaffold.
//
// Three file-local kernels reachable through RegisterNkMpmOps: clear the
// env-private background grid, deterministically GATHER particles -> grid (mass +
// APIC momentum, NO float atomics), and gather grid -> particle velocity + affine
// C. No constitutive force in this scaffold (grid_force stays 0), so a clear ->
// P2G -> G2P round-trip with C=0, F=I is the identity on the velocity field.
//
// Determinism: particles are radix-sorted by env-offset MPM cell key, then each
// grid node sums its 3^3-stencil particle contributions in the sorted order. The
// __fadd_rn pins the accumulation ORDER (run-to-run deterministic); inner products
// stay FMA-contractible (compile-time deterministic). The grid is env-private
// (env-offset node keys) so replicated envs never cross-couple.
// ---------------------------------------------------------------------------

#include <climits>
#include <cstdint>

#include <cuda_runtime.h>

#include <cub/device/device_radix_sort.cuh>

#include "math/vec3.hpp"
#include "nk/model/generated/views.hpp"  // ModelView / DataView (complete types)
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/registry.cuh"
#include "phi/op_schema.hpp"

namespace nuka::phi {

namespace {

namespace m = ::nuka::math;
constexpr uint32_t kBlockSize = 128u;

// Round up to the 256B section alignment the Arena lays scratch out at, so every
// sub-region of mpm_sort_scratch is 256B-aligned device memory.
constexpr uint64_t kScratchAlign = 256u;
inline __host__ uint64_t AlignScratch(uint64_t v) {
    return (v + (kScratchAlign - 1u)) & ~(kScratchAlign - 1u);
}

// 256B-aligned partition of mpm_sort_scratch [cub temp | keys-out | idx-out].
// Sizes the segment (World construct) and partitions it (the op) identically.
struct MpmSortScratchLayout {
    uint64_t temp_bytes = 0u;     // cub radix-sort temp-storage region size.
    uint64_t keys_off   = 0u;     // byte offset of the sorted-keys out buffer.
    uint64_t idx_off    = 0u;     // byte offset of the sorted-idx out buffer.
    uint64_t total      = 0u;     // full segment byte size.
    explicit MpmSortScratchLayout(uint32_t particle_count) {
        const int n = static_cast<int>(particle_count);
        size_t sort_bytes = 0u;
        (void)cub::DeviceRadixSort::SortPairs<uint32_t, uint32_t>(
            nullptr, sort_bytes, static_cast<const uint32_t*>(nullptr),
            static_cast<uint32_t*>(nullptr), static_cast<const uint32_t*>(nullptr),
            static_cast<uint32_t*>(nullptr), n);
        temp_bytes = sort_bytes;
        const uint64_t nbytes = static_cast<uint64_t>(particle_count) * sizeof(uint32_t);
        keys_off = AlignScratch(temp_bytes);
        idx_off  = AlignScratch(keys_off + nbytes);
        total    = AlignScratch(idx_off + nbytes);
    }
};

// Quadratic B-spline base node + the 3 per-axis weights. The particle sits in the
// stencil [base, base+2]; fx is the particle offset from base in cell units.
struct Bspline {
    int32_t base;
    float   w[3];
};
__device__ __forceinline__ Bspline QuadWeights(float gx) {
    // base = floor(gx - 0.5); fx in [0.5, 1.5) is the offset from base.
    Bspline b;
    b.base = static_cast<int32_t>(floorf(gx - 0.5f));
    const float fx = gx - static_cast<float>(b.base);
    b.w[0] = 0.5f * (1.5f - fx) * (1.5f - fx);
    const float d = fx - 1.0f;
    b.w[1] = 0.75f - d * d;
    b.w[2] = 0.5f * (fx - 0.5f) * (fx - 0.5f);
    return b;
}

// Per-env grid node id from integer node coords (env-offset for env-private grids).
__device__ __forceinline__ int64_t NodeId(uint32_t env, int32_t ix, int32_t iy,
                                          int32_t iz, const uint32_t dims[3],
                                          uint32_t nodes_per_env) {
    if (ix < 0 || iy < 0 || iz < 0 || ix >= static_cast<int32_t>(dims[0]) ||
        iy >= static_cast<int32_t>(dims[1]) || iz >= static_cast<int32_t>(dims[2])) {
        return -1;
    }
    const uint32_t local = (static_cast<uint32_t>(iz) * dims[1] +
                            static_cast<uint32_t>(iy)) * dims[0] +
                           static_cast<uint32_t>(ix);
    return static_cast<int64_t>(env) * nodes_per_env + local;
}

// --- grid clear -------------------------------------------------------------
__global__ void MpmGridClearKernel(uint32_t total_nodes, float* mass,
                                   m::Vec3* momentum, m::Vec3* velocity,
                                   m::Vec3* force) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_nodes) return;
    mass[i] = 0.0f;
    momentum[i] = m::Vec3::Zero();
    velocity[i] = m::Vec3::Zero();
    force[i] = m::Vec3::Zero();
}

// Clear THIS op's grid-escape status bit per env (order-independent across ops).
__global__ void MpmClearEscapeBitKernel(uint32_t* env_status, uint32_t env_count) {
    const uint32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= env_count) return;
    env_status[e] &= ~kEnvStatusMpmGridEscape;
}

// --- cell key (the particle's base-node cell, env-offset) -------------------
__global__ void MpmCellKeysKernel(uint32_t particle_count,
                                  const m::Vec3* __restrict__ pos,
                                  uint32_t particles_per_env, float inv_dx,
                                  m::Vec3 origin, uint32_t dims_x, uint32_t dims_y,
                                  uint32_t dims_z, uint32_t cells_per_env,
                                  uint32_t* __restrict__ keys,
                                  uint32_t* __restrict__ idx,
                                  uint32_t* __restrict__ env_status) {
    const uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= particle_count) return;
    const m::Vec3 xp = pos[p];
    const uint32_t env = p / particles_per_env;
    const float gx = (xp.x - origin.x) * inv_dx;
    const float gy = (xp.y - origin.y) * inv_dx;
    const float gz = (xp.z - origin.z) * inv_dx;
    const int32_t bx = static_cast<int32_t>(floorf(gx - 0.5f));
    const int32_t by = static_cast<int32_t>(floorf(gy - 0.5f));
    const int32_t bz = static_cast<int32_t>(floorf(gz - 0.5f));
    // A base outside [0, dims-2] flags a loud grid-AABB escape. Flag non-finite
    // directly; widen base+2 to i64 so an INT_MAX-saturated huge coord still trips.
    const bool nonfinite = !isfinite(gx) || !isfinite(gy) || !isfinite(gz);
    const bool escaped = nonfinite || bx < 0 || by < 0 || bz < 0 ||
                         static_cast<int64_t>(bx) + 2 >= static_cast<int64_t>(dims_x) ||
                         static_cast<int64_t>(by) + 2 >= static_cast<int64_t>(dims_y) ||
                         static_cast<int64_t>(bz) + 2 >= static_cast<int64_t>(dims_z);
    if (escaped && env_status != nullptr) {
        atomicOr(&env_status[env], kEnvStatusMpmGridEscape);
    }
    const int32_t cx = bx < 0 ? 0 : (bx >= static_cast<int32_t>(dims_x) ?
                                     static_cast<int32_t>(dims_x) - 1 : bx);
    const int32_t cy = by < 0 ? 0 : (by >= static_cast<int32_t>(dims_y) ?
                                     static_cast<int32_t>(dims_y) - 1 : by);
    const int32_t cz = bz < 0 ? 0 : (bz >= static_cast<int32_t>(dims_z) ?
                                     static_cast<int32_t>(dims_z) - 1 : bz);
    const uint32_t local = (static_cast<uint32_t>(cz) * dims_y +
                            static_cast<uint32_t>(cy)) * dims_x +
                           static_cast<uint32_t>(cx);
    keys[p] = env * cells_per_env + local;
    idx[p] = p;
}

// --- P2G deterministic gather (one thread per grid node) --------------------
// Node i sums the contributions of every particle whose 3^3 stencil includes i,
// i.e. particles whose base cell lies in [i-2, i] per axis. The sorted particle
// stream (by env-offset cell key) is scanned per candidate cell in a fixed order
// with __fadd_rn -> bit-reproducible run-to-run (NO atomics).
__global__ void MpmP2GGatherKernel(uint32_t total_nodes,
                                   const m::Vec3* __restrict__ pos,
                                   const float* __restrict__ inv_mass,
                                   const m::Vec3* __restrict__ vel,
                                   const float* __restrict__ part_C,
                                   const uint32_t* __restrict__ sorted_keys,
                                   const uint32_t* __restrict__ sorted_idx,
                                   uint32_t particle_count, uint32_t particles_per_env,
                                   uint32_t nodes_per_env, uint32_t cells_per_env,
                                   uint32_t dims_x, uint32_t dims_y, uint32_t dims_z,
                                   float inv_dx, float dx, m::Vec3 origin,
                                   float* __restrict__ grid_mass,
                                   m::Vec3* __restrict__ grid_momentum) {
    const uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= total_nodes) return;
    const uint32_t env = node / nodes_per_env;
    const uint32_t local = node % nodes_per_env;
    const int32_t nx = static_cast<int32_t>(local % dims_x);
    const int32_t ny = static_cast<int32_t>((local / dims_x) % dims_y);
    const int32_t nz = static_cast<int32_t>(local / (dims_x * dims_y));
    const m::Vec3 xi = m::Vec3{origin.x + nx * dx, origin.y + ny * dx,
                               origin.z + nz * dx};
    float mass = 0.0f;
    m::Vec3 mom = m::Vec3::Zero();
    // Scan the 3^3 base cells that can place a particle's stencil on this node.
    for (int32_t dz = -2; dz <= 0; ++dz) {
        const int32_t cz = nz + dz;
        if (cz < 0 || cz >= static_cast<int32_t>(dims_z)) continue;
        for (int32_t dy = -2; dy <= 0; ++dy) {
            const int32_t cy = ny + dy;
            if (cy < 0 || cy >= static_cast<int32_t>(dims_y)) continue;
            for (int32_t dx_i = -2; dx_i <= 0; ++dx_i) {
                const int32_t cx = nx + dx_i;
                if (cx < 0 || cx >= static_cast<int32_t>(dims_x)) continue;
                const uint32_t ck = (static_cast<uint32_t>(cz) * dims_y +
                                     static_cast<uint32_t>(cy)) * dims_x +
                                    static_cast<uint32_t>(cx);
                const uint32_t key = env * cells_per_env + ck;
                // Binary-search the lower bound of `key` in the sorted stream.
                uint32_t lo = 0u, hi = particle_count;
                while (lo < hi) {
                    const uint32_t mid = lo + ((hi - lo) >> 1);
                    if (sorted_keys[mid] < key) lo = mid + 1u; else hi = mid;
                }
                for (uint32_t s = lo; s < particle_count && sorted_keys[s] == key; ++s) {
                    const uint32_t p = sorted_idx[s];
                    const float wp = (inv_mass[p] > 0.0f) ? (1.0f / inv_mass[p]) : 0.0f;
                    if (wp <= 0.0f) continue;
                    const m::Vec3 xp = pos[p];
                    const Bspline wxs = QuadWeights((xp.x - origin.x) * inv_dx);
                    const Bspline wys = QuadWeights((xp.y - origin.y) * inv_dx);
                    const Bspline wzs = QuadWeights((xp.z - origin.z) * inv_dx);
                    const int32_t ox = nx - wxs.base, oy = ny - wys.base,
                                  oz = nz - wzs.base;
                    if (ox < 0 || ox > 2 || oy < 0 || oy > 2 || oz < 0 || oz > 2) continue;
                    const float w = wxs.w[ox] * wys.w[oy] * wzs.w[oz];
                    const m::Vec3 vp = vel[p];
                    // APIC affine term C_p * (x_i - x_p) (row-major 3x3 in part_C).
                    const m::Vec3 dpos = xi - xp;
                    const float* C = part_C + static_cast<size_t>(p) * 9u;
                    const m::Vec3 cterm = m::Vec3{
                        C[0] * dpos.x + C[1] * dpos.y + C[2] * dpos.z,
                        C[3] * dpos.x + C[4] * dpos.y + C[5] * dpos.z,
                        C[6] * dpos.x + C[7] * dpos.y + C[8] * dpos.z};
                    const float wm = w * wp;
                    mass = __fadd_rn(mass, wm);
                    mom.x = __fadd_rn(mom.x, wm * (vp.x + cterm.x));
                    mom.y = __fadd_rn(mom.y, wm * (vp.y + cterm.y));
                    mom.z = __fadd_rn(mom.z, wm * (vp.z + cterm.z));
                }
            }
        }
    }
    grid_mass[node] = mass;
    grid_momentum[node] = mom;
}

// --- grid velocity normalize (v = (mv)/m where m > 0) -----------------------
__global__ void MpmGridVelocityKernel(uint32_t total_nodes,
                                      const float* __restrict__ mass,
                                      const m::Vec3* __restrict__ momentum,
                                      m::Vec3* __restrict__ velocity) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_nodes) return;
    const float mi = mass[i];
    velocity[i] = (mi > 0.0f) ? (momentum[i] * (1.0f / mi)) : m::Vec3::Zero();
}

// --- G2P gather (one thread per particle; race-free, naturally D1) ----------
// v_p = sum_i N_i v_i; C_p = (4/dx^2) sum_i N_i v_i (x_i - x_p)^T (the MLS fit).
__global__ void MpmG2PGatherKernel(uint32_t particle_count,
                                   const m::Vec3* __restrict__ pos,
                                   uint32_t particles_per_env, uint32_t nodes_per_env,
                                   uint32_t dims_x, uint32_t dims_y, uint32_t dims_z,
                                   float inv_dx, float dx, m::Vec3 origin,
                                   const m::Vec3* __restrict__ grid_velocity,
                                   m::Vec3* __restrict__ part_vel,
                                   float* __restrict__ part_C) {
    const uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= particle_count) return;
    const uint32_t env = p / particles_per_env;
    const m::Vec3 xp = pos[p];
    const Bspline wxs = QuadWeights((xp.x - origin.x) * inv_dx);
    const Bspline wys = QuadWeights((xp.y - origin.y) * inv_dx);
    const Bspline wzs = QuadWeights((xp.z - origin.z) * inv_dx);
    m::Vec3 vp = m::Vec3::Zero();
    float C[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    const uint32_t dims[3] = {dims_x, dims_y, dims_z};
    for (int32_t a = 0; a < 3; ++a) {
        for (int32_t b = 0; b < 3; ++b) {
            for (int32_t c = 0; c < 3; ++c) {
                const int32_t ix = wxs.base + a, iy = wys.base + b, iz = wzs.base + c;
                const int64_t id = NodeId(env, ix, iy, iz, dims, nodes_per_env);
                if (id < 0) continue;
                const float w = wxs.w[a] * wys.w[b] * wzs.w[c];
                const m::Vec3 vi = grid_velocity[static_cast<size_t>(id)];
                vp.x = __fadd_rn(vp.x, w * vi.x);
                vp.y = __fadd_rn(vp.y, w * vi.y);
                vp.z = __fadd_rn(vp.z, w * vi.z);
                const m::Vec3 xi = m::Vec3{origin.x + ix * dx, origin.y + iy * dx,
                                           origin.z + iz * dx};
                const m::Vec3 d = xi - xp;
                C[0] = __fadd_rn(C[0], w * vi.x * d.x); C[1] = __fadd_rn(C[1], w * vi.x * d.y); C[2] = __fadd_rn(C[2], w * vi.x * d.z);
                C[3] = __fadd_rn(C[3], w * vi.y * d.x); C[4] = __fadd_rn(C[4], w * vi.y * d.y); C[5] = __fadd_rn(C[5], w * vi.y * d.z);
                C[6] = __fadd_rn(C[6], w * vi.z * d.x); C[7] = __fadd_rn(C[7], w * vi.z * d.y); C[8] = __fadd_rn(C[8], w * vi.z * d.z);
            }
        }
    }
    const float scale = 4.0f * inv_dx * inv_dx;
    float* Cd = part_C + static_cast<size_t>(p) * 9u;
    for (int32_t k = 0; k < 9; ++k) Cd[k] = C[k] * scale;
    part_vel[p] = vp;
}

Status OpMpmGridClear(const ModelView& /*model*/, const DataView& data,
                      const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const MpmGridParams*>(params);
    if (p == nullptr) return Status::Failed;
    const uint64_t total = static_cast<uint64_t>(p->nodes_per_env) * p->env_count;
    if (total == 0u) return Status::Ok;
    if (total > 0xFFFFFFFFull) return Status::Failed;  // u32 node-id key overflow.
    const uint32_t n = static_cast<uint32_t>(total);
    const uint32_t blocks = (n + kBlockSize - 1u) / kBlockSize;
    LaunchCuda(MpmGridClearKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
               n, data.grid_mass, data.grid_momentum, data.grid_velocity,
               data.grid_force);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

Status OpMpmP2G(const ModelView& /*model*/, const DataView& data,
                const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const MpmGridParams*>(params);
    if (p == nullptr) return Status::Failed;
    if (p->particle_count == 0u || p->nodes_per_env == 0u || p->dx <= 0.0f) {
        return Status::Ok;
    }
    const uint64_t total_nodes64 =
        static_cast<uint64_t>(p->nodes_per_env) * p->env_count;
    const uint64_t cells_per_env =
        static_cast<uint64_t>(p->grid_dims[0]) * p->grid_dims[1] * p->grid_dims[2];
    if (cells_per_env == 0u) return Status::Ok;
    // LOUD overflow: the env-offset node/cell key must fit a u32 (never silent).
    if (total_nodes64 > 0xFFFFFFFFull) return Status::Failed;
    if (cells_per_env * p->env_count > 0xFFFFFFFFull) return Status::Failed;
    const uint32_t Np = p->particle_count;
    const uint32_t Ppe = p->particles_per_env == 0u ? Np : p->particles_per_env;
    const uint32_t cpe = static_cast<uint32_t>(cells_per_env);
    // LOUD invariants: particle_count must fit the scratch/env_status footprint and
    // cub's int num_items (the SortPairs / scratch-layout sizes static_cast to int).
    if (static_cast<uint64_t>(Np) >
        static_cast<uint64_t>(Ppe) * p->env_count) return Status::Failed;
    if (static_cast<uint64_t>(Np) > static_cast<uint64_t>(INT_MAX)) return Status::Failed;
    const float inv_dx = 1.0f / p->dx;
    const m::Vec3 origin{p->grid_origin[0], p->grid_origin[1], p->grid_origin[2]};

    // Partition the pre-allocated scratch [cub temp | keys-out | idx-out] so the
    // sort never cudaMalloc/syncs mid-capture (the gather joins the graph).
    const MpmSortScratchLayout sl(Np);
    char* sbase = reinterpret_cast<char*>(data.mpm_sort_scratch);
    void* sort_temp = sbase;
    uint32_t* keys_out = reinterpret_cast<uint32_t*>(sbase + sl.keys_off);
    uint32_t* idx_out  = reinterpret_cast<uint32_t*>(sbase + sl.idx_off);
    size_t sort_temp_bytes = static_cast<size_t>(sl.temp_bytes);
    const uint32_t pblocks = (Np + kBlockSize - 1u) / kBlockSize;
    if (data.env_status != nullptr) {  // clear THIS op's grid-escape bit per env.
        const uint32_t e = p->env_count == 0u ? 1u : p->env_count;
        const uint32_t eb = (e + kBlockSize - 1u) / kBlockSize;
        LaunchCuda(MpmClearEscapeBitKernel, dim3(eb), dim3(kBlockSize), 0u, stream,
                   data.env_status, e);
    }
    LaunchCuda(MpmCellKeysKernel, dim3(pblocks), dim3(kBlockSize), 0u, stream, Np,
               data.particle_pos, Ppe, inv_dx, origin, p->grid_dims[0],
               p->grid_dims[1], p->grid_dims[2], cpe, data.mpm_grid_cell_key,
               data.mpm_grid_part_idx, data.env_status);
    // Stable radix sort the (cell key -> particle index) pairs into the out
    // buffers (stable => byte-identical run-to-run), then a node-parallel gather.
    (void)cub::DeviceRadixSort::SortPairs(
        sort_temp, sort_temp_bytes, data.mpm_grid_cell_key, keys_out,
        data.mpm_grid_part_idx, idx_out, static_cast<int>(Np), 0, 32, stream);
    const uint32_t total_nodes = static_cast<uint32_t>(total_nodes64);
    const uint32_t nblocks = (total_nodes + kBlockSize - 1u) / kBlockSize;
    LaunchCuda(MpmP2GGatherKernel, dim3(nblocks), dim3(kBlockSize), 0u, stream,
               total_nodes, data.particle_pos, data.particle_inv_mass,
               data.particle_vel, data.particle_C, keys_out, idx_out, Np, Ppe,
               p->nodes_per_env, cpe, p->grid_dims[0], p->grid_dims[1],
               p->grid_dims[2], inv_dx, p->dx, origin, data.grid_mass,
               data.grid_momentum);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

Status OpMpmG2P(const ModelView& /*model*/, const DataView& data,
                const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const MpmGridParams*>(params);
    if (p == nullptr) return Status::Failed;
    if (p->particle_count == 0u || p->nodes_per_env == 0u || p->dx <= 0.0f) {
        return Status::Ok;
    }
    const uint64_t total_nodes64 =
        static_cast<uint64_t>(p->nodes_per_env) * p->env_count;
    if (total_nodes64 > 0xFFFFFFFFull) return Status::Failed;
    const uint32_t total_nodes = static_cast<uint32_t>(total_nodes64);
    const float inv_dx = 1.0f / p->dx;
    const m::Vec3 origin{p->grid_origin[0], p->grid_origin[1], p->grid_origin[2]};
    const uint32_t Np = p->particle_count;
    const uint32_t Ppe = p->particles_per_env == 0u ? Np : p->particles_per_env;
    // LOUD invariant: env=p/Ppe reads the env-private grid, so the count must fit
    // the per-env footprint (else the gather reads grid nodes of a missing env).
    if (static_cast<uint64_t>(Np) >
        static_cast<uint64_t>(Ppe) * p->env_count) return Status::Failed;
    // Grid momentum -> velocity, then a race-free per-particle gather.
    const uint32_t nblocks = (total_nodes + kBlockSize - 1u) / kBlockSize;
    LaunchCuda(MpmGridVelocityKernel, dim3(nblocks), dim3(kBlockSize), 0u, stream,
               total_nodes, data.grid_mass, data.grid_momentum, data.grid_velocity);
    const uint32_t pblocks = (Np + kBlockSize - 1u) / kBlockSize;
    LaunchCuda(MpmG2PGatherKernel, dim3(pblocks), dim3(kBlockSize), 0u, stream, Np,
               data.particle_pos, Ppe, p->nodes_per_env, p->grid_dims[0],
               p->grid_dims[1], p->grid_dims[2], inv_dx, p->dx, origin,
               data.grid_velocity, data.particle_vel, data.particle_C);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

}  // namespace

uint64_t MpmSortScratchBytes(uint32_t particle_count) {
    if (particle_count == 0u) return 0u;
    return MpmSortScratchLayout(particle_count).total;
}

void RegisterNkMpmOps() {
    SetCudaOp(NkOp::MpmGridClear, &OpMpmGridClear);
    SetCudaOp(NkOp::MpmP2G, &OpMpmP2G);
    SetCudaOp(NkOp::MpmG2P, &OpMpmG2P);
}

}  // namespace nuka::phi
