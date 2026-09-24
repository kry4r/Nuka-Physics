// MLS-MPM transfers APIC momentum and constitutive stress through environment-private grids.
// Stable cell sorting and ordered gathers keep particle and body accumulation deterministic.

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

#include <cub/block/block_store.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <thrust/iterator/counting_iterator.h>

#include "collision/primitive_surface.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/material/hencky_j2.hpp"
#include "nk/model/generated/views.hpp"  // ModelView / DataView (complete types)
#include "nk/solve/collidable_owner.hpp"
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/launch_grid.cuh"
#include "phi/backend_cuda/ops/articulation_types.cuh"  // chain-J helper + device state
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/prims_types.cuh"
#include "phi/backend_cuda/ops/registry.cuh"
#include "phi/backend_cuda/ops/rigid_types.cuh"
#include "phi/op_schema.hpp"
#include "runtime/sdf/sparse_sdf_query.cuh"       // SparseSdfDevice / sparse_sdf_sample
#include "phi/backend_cuda/ops/surface_query.cuh"
#include "phi/backend_cuda/ops/mpm_contacts.cuh"

namespace nuka::phi {

namespace {

namespace m = ::nuka::math;
constexpr uint32_t kBlockSize = 128u;
constexpr uint32_t kSpatialComponents = 3u;
constexpr uint32_t kStencilWidth = nk::kMpmStencilWidth;
constexpr uint32_t kStencilNodes = kStencilWidth * kStencilWidth * kStencilWidth;
constexpr uint32_t kCudaWarpThreads = 32u;
constexpr uint32_t kCellGroups = kBlockSize / kCudaWarpThreads;
constexpr uint32_t kFullWarpMask = ~uint32_t{0};
static_assert(kBlockSize % kCudaWarpThreads == 0u);
static_assert(kStencilNodes <= kCudaWarpThreads);

// NUKA_MPM_TIMING enables synchronous eager stage timing per physics interval.
// Leave it disabled for ordinary execution and graph capture.
enum class MpmStage : uint32_t {
    GridPrepare,
    CellKeys,
    RadixSort,
    CellRanges,
    ActiveSelect,
    TransferInput,
    P2GCells,
    GridFinalize,
    ContactCount,
    ContactScan,
    ContactEmit,
    ReactionReadout,
    G2P,
    UpdateF,
    Count,
};

struct MpmProfiler {
    static constexpr uint32_t kCount = static_cast<uint32_t>(MpmStage::Count);
    double ms[kCount] = {0.0};
    unsigned long long calls[kCount] = {0};
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    long interval = 0;
    long warmup = 0;
    unsigned long long measured_intervals = 0;
    bool on = false;
    bool active = false;
    bool initialized = false;

    static MpmProfiler& Get() {
        static MpmProfiler profiler;
        return profiler;
    }

    static const char* Name(uint32_t stage) {
        constexpr const char* names[kCount] = {
            "grid_prepare", "cell_keys", "radix_sort", "cell_ranges",
            "active_select", "transfer_input", "p2g_cells", "grid_finalize", "contact_count",
            "contact_scan", "contact_emit", "reaction_readout", "g2p_gather", "update_F"};
        return names[stage];
    }

    void Ensure() {
        if (initialized) return;
        initialized = true;
        const char* enabled = std::getenv("NUKA_MPM_TIMING");
        on = enabled != nullptr && enabled[0] != '0';
        const char* warm = std::getenv("NUKA_MPM_TIMING_WARMUP");
        if (warm != nullptr) warmup = std::atol(warm);
        if (!on) return;
        if (cudaEventCreate(&begin) != cudaSuccess ||
            cudaEventCreate(&end) != cudaSuccess) {
            on = false;
            return;
        }
        std::atexit(&MpmProfiler::Dump);
    }

    void BeginInterval() {
        Ensure();
        if (!on) return;
        ++interval;
        active = interval > warmup;
        if (active) ++measured_intervals;
    }

    void Start(MpmStage stage, cudaStream_t stream) {
        if (!active) return;
        (void)stage;
        (void)cudaEventRecord(begin, stream);
    }

    void Stop(MpmStage stage, cudaStream_t stream) {
        if (!active) return;
        (void)cudaEventRecord(end, stream);
        (void)cudaEventSynchronize(end);
        float elapsed = 0.0f;
        (void)cudaEventElapsedTime(&elapsed, begin, end);
        const uint32_t index = static_cast<uint32_t>(stage);
        ms[index] += static_cast<double>(elapsed);
        ++calls[index];
    }

    static void Dump() {
        MpmProfiler& profiler = Get();
        if (profiler.measured_intervals == 0) return;
        std::printf("\n[NUKA_MPM_TIMING] MPM GPU stages (intervals=%llu, warmup_intervals=%ld)\n",
                    profiler.measured_intervals, profiler.warmup);
        std::printf("  %-18s %10s %10s %12s\n", "stage", "ms/interval", "ms/call",
                    "calls");
        double total = 0.0;
        for (uint32_t i = 0; i < kCount; ++i) {
            if (profiler.calls[i] == 0) continue;
            total += profiler.ms[i];
            std::printf("  %-18s %10.3f %10.4f %12llu\n", Name(i),
                        profiler.ms[i] / profiler.measured_intervals,
                        profiler.ms[i] / profiler.calls[i], profiler.calls[i]);
        }
        std::printf("  %-18s %10.3f\n", "TOTAL/interval", total / profiler.measured_intervals);
        std::fflush(stdout);
    }
};

// A normal mass keeps momentum normalization from dividing by a denormal.
constexpr float kMinNodeMass = 1.0e-30f;

// Round up to the 256B section alignment the Arena lays scratch out at, so every
// sub-region of mpm_sort_scratch is 256B-aligned device memory.
constexpr uint64_t kScratchAlign = 256u;
inline __host__ uint64_t AlignScratch(uint64_t v) {
    return (v + (kScratchAlign - 1u)) & ~(kScratchAlign - 1u);
}

// Retain enough low bits to sort valid cell/body keys before the ~0u sentinel.
inline __host__ int RadixBitsInclusive(uint32_t max_key) {
    int bits = 0;
    do {
        ++bits;
        max_key >>= 1u;
    } while (max_key != 0u);
    return bits;
}

struct MpmTransferInput {
    m::Vec3 position;
    float mass;
    m::Vec3 velocity;
    float volume;
    int32_t base[3];
    float affine[9];
    float stress[9];
};

struct alignas(float4) MpmCellTransfer {
    float mass;
    m::Vec3 momentum;
};
static_assert(sizeof(MpmCellTransfer) == (1u + kSpatialComponents) * sizeof(float));

// Particle records and grid indexing have independent capacities; sort outputs are reused.
struct MpmSortScratchLayout {
    uint64_t temp_bytes = 0u;     // cub radix-sort temp-storage region size.
    uint64_t keys_off   = 0u;     // byte offset of the sorted-keys out buffer.
    uint64_t idx_off    = 0u;     // byte offset of the sorted-idx out buffer.
    uint64_t active_nodes_off = 0u;
    uint64_t active_flags_off = 0u;
    uint64_t cell_start_off = 0u;
    uint64_t cell_end_off = 0u;
    uint64_t node_ids_off = 0u;
    uint64_t active_count_off = 0u;
    uint64_t transfer_input_off = 0u;
    uint64_t cell_transfer_off = 0u;
    uint64_t contact_hit_mask_off = 0u;
    uint64_t total      = 0u;     // full segment byte size.
    MpmSortScratchLayout(uint32_t particle_count, uint32_t node_count,
                         uint64_t collidables_per_env) {
        const auto sort_bytes_for = [](uint32_t count) {
            size_t bytes = 0u;
            const auto status = cub::DeviceRadixSort::SortPairs<uint32_t, uint32_t>(
                nullptr, bytes, static_cast<const uint32_t*>(nullptr),
                static_cast<uint32_t*>(nullptr), static_cast<const uint32_t*>(nullptr),
                static_cast<uint32_t*>(nullptr), static_cast<int>(count));
            if (status != cudaSuccess)
                throw std::runtime_error(cudaGetErrorString(status));
            return bytes;
        };
        const size_t particle_sort_bytes = sort_bytes_for(particle_count);
        const size_t node_sort_bytes = sort_bytes_for(node_count);
        size_t select_bytes = 0u;
        thrust::counting_iterator<uint32_t> node_ids(0u);
        const auto status = cub::DeviceSelect::Flagged(
            nullptr, select_bytes, node_ids, static_cast<const uint32_t*>(nullptr),
            static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
            static_cast<int>(node_count));
        if (status != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(status));
        size_t scan_bytes = 0u;
        const auto scan_status = cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes,
            static_cast<const uint64_t*>(nullptr), static_cast<uint64_t*>(nullptr),
            static_cast<int>(std::max(particle_count, node_count)));
        if (scan_status != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(scan_status));
        temp_bytes = std::max({particle_sort_bytes, node_sort_bytes, select_bytes, scan_bytes});
        const uint64_t output_bytes =
            uint64_t{std::max(particle_count, node_count)} * sizeof(uint32_t);
        const uint64_t node_bytes = uint64_t{node_count} * sizeof(uint32_t);
        keys_off = AlignScratch(temp_bytes);
        idx_off  = AlignScratch(keys_off + output_bytes);
        active_nodes_off = AlignScratch(idx_off + output_bytes);
        active_flags_off = AlignScratch(active_nodes_off + node_bytes);
        cell_start_off = AlignScratch(active_flags_off + node_bytes);
        cell_end_off = AlignScratch(cell_start_off + node_bytes);
        node_ids_off = AlignScratch(cell_end_off + node_bytes);
        active_count_off = AlignScratch(node_ids_off + node_bytes);
        transfer_input_off = AlignScratch(active_count_off + sizeof(uint32_t));
        cell_transfer_off = AlignScratch(transfer_input_off +
                                        uint64_t{particle_count} * sizeof(MpmTransferInput));
        contact_hit_mask_off = AlignScratch(cell_transfer_off +
            uint64_t{std::min(particle_count, node_count)} * kStencilNodes * sizeof(MpmCellTransfer));
        const uint64_t mask_words = (collidables_per_env + 31u) / 32u;
        total = AlignScratch(contact_hit_mask_off + uint64_t{particle_count} *
                             mask_words * sizeof(uint32_t));
    }
};

const MpmSortScratchLayout& ScratchLayout(uint32_t particle_count, uint32_t node_count,
                                          uint64_t collidables_per_env) {
    int device = -1;
    const auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    struct Entry {
        int device;
        uint32_t particles;
        uint32_t nodes;
        uint64_t collidables;
        MpmSortScratchLayout layout;
    };
    static thread_local std::vector<Entry> entries;
    for (const auto& entry : entries)
        if (entry.device == device && entry.particles == particle_count &&
            entry.nodes == node_count && entry.collidables == collidables_per_env)
            return entry.layout;
    entries.push_back({device, particle_count, node_count, collidables_per_env,
                       MpmSortScratchLayout(particle_count, node_count, collidables_per_env)});
    return entries.back().layout;
}

// Row-major 3x3 algebra matches particle_F/C packing.

// C = A * B (row-major 3x3).
__device__ __forceinline__ void Mat3Mul(const float* A, const float* B, float* C) {
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            C[r * 3 + c] = __fadd_rn(__fadd_rn(A[r * 3 + 0] * B[0 * 3 + c],
                                               A[r * 3 + 1] * B[1 * 3 + c]),
                                     A[r * 3 + 2] * B[2 * 3 + c]);
}

// C = A * B^T (row-major 3x3).
__device__ __forceinline__ void Mat3MulT(const float* A, const float* B, float* C) {
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            C[r * 3 + c] = __fadd_rn(__fadd_rn(A[r * 3 + 0] * B[c * 3 + 0],
                                               A[r * 3 + 1] * B[c * 3 + 1]),
                                     A[r * 3 + 2] * B[c * 3 + 2]);
}

__device__ __forceinline__ float Mat3Det(const float* F) {
    return F[0] * (F[4] * F[8] - F[5] * F[7]) -
           F[1] * (F[3] * F[8] - F[5] * F[6]) +
           F[2] * (F[3] * F[7] - F[4] * F[6]);
}

// Transposed inverse F^{-T} (row-major). Returns false (and leaves out unset) for a
// near-singular F so the caller can fall back to a zero stress (degenerate cell).
__device__ __forceinline__ bool Mat3InvTranspose(const float* F, float* out, float det) {
    if (fabsf(det) < 1e-12f) return false;
    const float inv = 1.0f / det;
    // Cofactor matrix C; F^{-1} = C^T/det, so F^{-T} = C/det.
    out[0] = (F[4] * F[8] - F[5] * F[7]) * inv;
    out[1] = (F[5] * F[6] - F[3] * F[8]) * inv;
    out[2] = (F[3] * F[7] - F[4] * F[6]) * inv;
    out[3] = (F[2] * F[7] - F[1] * F[8]) * inv;
    out[4] = (F[0] * F[8] - F[2] * F[6]) * inv;
    out[5] = (F[1] * F[6] - F[0] * F[7]) * inv;
    out[6] = (F[1] * F[5] - F[2] * F[4]) * inv;
    out[7] = (F[2] * F[3] - F[0] * F[5]) * inv;
    out[8] = (F[0] * F[4] - F[1] * F[3]) * inv;
    return true;
}

// One symmetric Jacobi rotation eliminates S(p,q) and accumulates eigenvectors in V.
__device__ __forceinline__ void JacobiRotate(float* S, float* V, int p, int q) {
    const float spq = S[p * 3 + q];
    if (spq == 0.0f) return;
    const float spp = S[p * 3 + p], sqq = S[q * 3 + q];
    const float theta = (sqq - spp) / (2.0f * spq);
    const float sign = theta >= 0.0f ? 1.0f : -1.0f;
    const float t = sign / (fabsf(theta) + sqrtf(theta * theta + 1.0f));
    const float c = 1.0f / sqrtf(t * t + 1.0f);
    const float s = t * c;
    for (int k = 0; k < 3; ++k) {
        const float sik = S[k * 3 + p], siq = S[k * 3 + q];
        S[k * 3 + p] = c * sik - s * siq;
        S[k * 3 + q] = s * sik + c * siq;
    }
    for (int k = 0; k < 3; ++k) {
        const float skp = S[p * 3 + k], skq = S[q * 3 + k];
        S[p * 3 + k] = c * skp - s * skq;
        S[q * 3 + k] = s * skp + c * skq;
        const float vkp = V[k * 3 + p], vkq = V[k * 3 + q];
        V[k * 3 + p] = c * vkp - s * vkq;
        V[k * 3 + q] = s * vkp + c * vkq;
    }
}

// Eight Jacobi sweeps decompose F^T F; U = F V / sig completes the SVD.
// Near-zero singular values use the corresponding V column.
__device__ __forceinline__ void Svd3(const float* F, float* U, float* sig, float* V) {
    float A[9];
    // A := F^T F (symmetric); its eigenvectors are the right singular vectors V.
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            A[r * 3 + c] = __fadd_rn(__fadd_rn(F[0 * 3 + r] * F[0 * 3 + c],
                                               F[1 * 3 + r] * F[1 * 3 + c]),
                                     F[2 * 3 + r] * F[2 * 3 + c]);
    for (int k = 0; k < 9; ++k) V[k] = (k % 4 == 0) ? 1.0f : 0.0f;  // V = I.
    for (int sweep = 0; sweep < 8; ++sweep) {
        JacobiRotate(A, V, 0, 1);
        JacobiRotate(A, V, 0, 2);
        JacobiRotate(A, V, 1, 2);
    }
    float s2[3] = {A[0], A[4], A[8]};
    // Floor the singular values so volumetric stress stays bounded as J -> 0.
    for (int i = 0; i < 3; ++i) sig[i] = fmaxf(sqrtf(fmaxf(s2[i], 0.0f)), 0.05f);
    // U columns = F * V_col / sig (fall back to V_col when sig ~ 0).
    for (int c = 0; c < 3; ++c) {
        float fc[3];
        for (int r = 0; r < 3; ++r)
            fc[r] = F[r * 3 + 0] * V[0 * 3 + c] + F[r * 3 + 1] * V[1 * 3 + c] +
                    F[r * 3 + 2] * V[2 * 3 + c];
        if (sig[c] > 1e-8f) {
            const float inv = 1.0f / sig[c];
            for (int r = 0; r < 3; ++r) U[r * 3 + c] = fc[r] * inv;
        } else {
            for (int r = 0; r < 3; ++r) U[r * 3 + c] = V[r * 3 + c];
        }
    }
    // Reflect (not just rotate): if det(U) < 0 flip the smallest-magnitude singular
    // value so R = U V^T is the closest proper rotation (handles inverted elements).
    if (Mat3Det(U) < 0.0f) {
        int kmin = 0;
        if (fabsf(sig[1]) < fabsf(sig[kmin])) kmin = 1;
        if (fabsf(sig[2]) < fabsf(sig[kmin])) kmin = 2;
        for (int r = 0; r < 3; ++r) U[r * 3 + kmin] = -U[r * 3 + kmin];
        sig[kmin] = -sig[kmin];
    }
}

// First Piola-Kirchhoff stress for fixed-corotated and Neo-Hookean elasticity.
// mu/lambda are the Lame moduli; matrices use row-major float[9] storage.
__device__ __forceinline__ void FirstPiola(const float* F, float mu, float lambda,
                                           float model_kind, float* P) {
    const float J = Mat3Det(F);
    float FinvT[9];
    const bool ok = Mat3InvTranspose(F, FinvT, J);
    if (!ok) { for (int k = 0; k < 9; ++k) P[k] = 0.0f; return; }
    if (model_kind > 1.5f) {  // Neo-Hookean elastic (kind 2).
        const float lj = logf(fmaxf(J, 1e-8f));
        for (int k = 0; k < 9; ++k)
            P[k] = mu * (F[k] - FinvT[k]) + lambda * lj * FinvT[k];
        return;
    }
    float U[9], sig[3], V[9], R[9];
    Svd3(F, U, sig, V);
    Mat3MulT(U, V, R);  // R = U * V^T (proper rotation).
    const float coef = lambda * (J - 1.0f) * J;
    for (int k = 0; k < 9; ++k)
        P[k] = 2.0f * mu * (F[k] - R[k]) + coef * FinvT[k];
}

// Granular stress and stored elastic deformation share this Hencky-strain bound.
constexpr float kSandHenckyCap = 0.15f;

// Hencky elasticity: tau = U diag(2*mu*eps + lambda*tr(eps)) U^T.
// eps_i = ln(sig_i) uses the stored elastic deformation.
__device__ __forceinline__ void GranularKirchhoff(const float* F, float mu,
                                                  float lambda, float* stress) {
    float U[9], sig[3], V[9];
    Svd3(F, U, sig, V);
    float eps[3];
    for (int i = 0; i < 3; ++i) {
        eps[i] = logf(fmaxf(fabsf(sig[i]), 1.0e-6f));
        eps[i] = fminf(fmaxf(eps[i], -kSandHenckyCap), kSandHenckyCap);
    }
    const float tr = eps[0] + eps[1] + eps[2];
    float tau[3];
    for (int i = 0; i < 3; ++i) tau[i] = 2.0f * mu * eps[i] + lambda * tr;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            stress[r * 3 + c] = U[r * 3 + 0] * tau[0] * U[c * 3 + 0] +
                                U[r * 3 + 1] * tau[1] * U[c * 3 + 1] +
                                U[r * 3 + 2] * tau[2] * U[c * 3 + 2];
}

// Drucker-Prager return mapping of principal Hencky strains (Klar et al. 2016).
// Cohesion shifts the tensile apex; friction_deg sets the yield-cone angle.
__device__ __forceinline__ void SandReturnMap(float* F, float mu, float lambda,
                                             float friction_deg, float cohesion) {
    float U[9], sig[3], V[9];
    Svd3(F, U, sig, V);
    float eps[3], sgn[3];
    for (int i = 0; i < 3; ++i) {
        sgn[i] = sig[i] < 0.0f ? -1.0f : 1.0f;
        eps[i] = logf(fmaxf(fabsf(sig[i]), 1.0e-6f));
    }
    const float tr = eps[0] + eps[1] + eps[2];
    const float kappa = 3.0f * lambda + 2.0f * mu;             // d*lambda + 2*mu, d=3.
    const float c0 = kappa > 1.0e-9f ? cohesion / kappa : 0.0f;  // apex tensile strain.
    float dev[3];
    for (int i = 0; i < 3; ++i) dev[i] = eps[i] - tr * (1.0f / 3.0f);
    const float devn = sqrtf(dev[0] * dev[0] + dev[1] * dev[1] + dev[2] * dev[2]);
    const float sinp = sinf(friction_deg * 0.017453292519943295f);
    const float alpha = 1.632993161855452f * sinp / fmaxf(3.0f - sinp, 1.0e-6f);
    const float tr_shift = tr - c0;
    float en[3];
    if (devn < 1.0e-12f || tr_shift > 0.0f) {
        for (int i = 0; i < 3; ++i) en[i] = c0 * (1.0f / 3.0f);  // return to the apex.
    } else {
        const float dgamma = devn + (kappa / (2.0f * mu)) * tr_shift * alpha;
        if (dgamma <= 0.0f) {
            for (int i = 0; i < 3; ++i) en[i] = eps[i];           // inside the cone.
        } else {
            const float inv = 1.0f / devn;                        // radial return.
            for (int i = 0; i < 3; ++i) en[i] = eps[i] - dgamma * dev[i] * inv;
        }
    }
    float s2[3];
    for (int i = 0; i < 3; ++i) {
        // Cap the STORED elastic strain: the overflow is plastic densification, so
        // the state the next substep stresses can never spiral (bounded restoring).
        en[i] = fminf(fmaxf(en[i], -kSandHenckyCap), kSandHenckyCap);
        s2[i] = sgn[i] * expf(en[i]);
    }
    float US[9];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) US[r * 3 + c] = U[r * 3 + c] * s2[c];
    Mat3MulT(US, V, F);  // F = US * V^T = U diag(s2) V^T.
}

// MPM interval status is refreshed; shared invalid-endpoint status stays latched.
__global__ void MpmClearStatusBitsKernel(uint32_t* env_status, uint32_t env_count) {
    const uint32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= env_count) return;
    env_status[e] &= ~(kEnvStatusMpmGridEscape | kEnvStatusMpmOneWayBody | kEnvStatusGridContactOverflow);
}

// Per-body reaction probes contain this interval's linear and angular impulse.
__global__ void MpmClearBodyReactionKernel(uint32_t total_bodies, m::Vec3* reaction,
                                          m::Vec3* ang_reaction) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= total_bodies) return;
    reaction[b] = m::Vec3::Zero();
    if (ang_reaction != nullptr) ang_reaction[b] = m::Vec3::Zero();
}

// Map a compact MPM index to the low particle slice of its environment.
__device__ __forceinline__ uint32_t MpmSliceGlobal(uint32_t t, uint32_t mpm_per_env,
                                                   uint32_t ppe, uint32_t& env_out) {
    const uint32_t env = t / mpm_per_env;
    env_out = env;
    return env * ppe + (t - env * mpm_per_env);
}

// Compute a particle's Kirchhoff stress for its transfer record and observable field.
__device__ __forceinline__ bool MpmParticleStress(
    uint32_t p,
    const float* __restrict__ part_C, const float* __restrict__ part_F,
    const float* __restrict__ part_vol0, const uint32_t* __restrict__ part_mat,
    const nk::MpmMaterial* __restrict__ material_table, uint32_t material_count,
    float* __restrict__ stress) {
    const float vol0 = part_vol0 != nullptr ? part_vol0[p] : 0.0f;
    if (part_F == nullptr || vol0 <= 0.0f) {
        for (int k = 0; k < 9; ++k) stress[k] = 0.0f;
        return true;
    }
    const uint32_t mid = part_mat != nullptr ? part_mat[p] : 0u;
    if (material_count > 0u && mid >= material_count) return false;
    float youngs = 0.0f, poisson = 0.0f, kind = 0.0f;
    float bulk = 0.0f, tait_gamma = 0.0f, visc = 0.0f;
    if (material_table != nullptr && mid < material_count) {
        const nk::MpmMaterial& mr = material_table[mid];
        youngs = mr.youngs; poisson = mr.poisson; kind = mr.model_kind;
        bulk = mr.bulk_modulus; tait_gamma = mr.tait_gamma; visc = mr.viscosity;
    }
    const float* F = part_F + static_cast<size_t>(p) * 9u;
    if (kind == nk::MpmMaterial::kHenckyJ2) {
        const nk::MpmMaterial& mr = material_table[mid];
        nk::material::HenckyResponse response;
        if (nk::material::EvaluateHenckyJ2(F,
                {mr.youngs, mr.poisson, mr.yield_stress, mr.hardening_modulus}, response) !=
            nk::material::ConstitutiveStatus::Ok) return false;
        for (int k = 0; k < 9; ++k) stress[k] = response.kirchhoff[k];
    } else if (kind > 3.5f) {
        const float denom = (1.0f + poisson) * (1.0f - 2.0f * poisson);
        const float mu = youngs / (2.0f * (1.0f + poisson));
        const float lambda = (denom > 1e-9f) ? youngs * poisson / denom : 0.0f;
        GranularKirchhoff(F, mu, lambda, stress);
    } else if (kind > 2.5f) {
        const float J = Mat3Det(F);
        const float pr = fmaxf(bulk * (powf(J, -tait_gamma) - 1.0f), 0.0f);
        const float diag = -pr * J;
        stress[0] = diag; stress[1] = 0.0f; stress[2] = 0.0f;
        stress[3] = 0.0f; stress[4] = diag; stress[5] = 0.0f;
        stress[6] = 0.0f; stress[7] = 0.0f; stress[8] = diag;
        if (visc > 0.0f && part_C != nullptr) {
            const float* C = part_C + static_cast<size_t>(p) * 9u;
            const float Jv = J * visc;
            stress[0] += Jv * 2.0f * C[0];
            stress[4] += Jv * 2.0f * C[4];
            stress[8] += Jv * 2.0f * C[8];
            const float s01 = Jv * (C[1] + C[3]);
            const float s02 = Jv * (C[2] + C[6]);
            const float s12 = Jv * (C[5] + C[7]);
            stress[1] += s01; stress[3] += s01;
            stress[2] += s02; stress[6] += s02;
            stress[5] += s12; stress[7] += s12;
        }
    } else {
        const float denom = (1.0f + poisson) * (1.0f - 2.0f * poisson);
        const float mu = youngs / (2.0f * (1.0f + poisson));
        const float lambda = (denom > 1e-9f) ? youngs * poisson / denom : 0.0f;
        float P[9];
        FirstPiola(F, mu, lambda, kind, P);
        Mat3MulT(P, F, stress);
    }
    return true;
}

__global__ void MpmCellKeysKernel(uint32_t mpm_count,
                                  const m::Vec3* __restrict__ pos,
                                  uint32_t particles_per_env, uint32_t mpm_per_env,
                                  float inv_dx,
                                  m::Vec3 origin, uint32_t dims_x, uint32_t dims_y,
                                  uint32_t dims_z, uint32_t cells_per_env,
                                  uint32_t* __restrict__ keys,
                                  uint32_t* __restrict__ idx,
                                  uint32_t* __restrict__ env_status) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= mpm_count) return;
    uint32_t env = 0u;
    const uint32_t p = MpmSliceGlobal(t, mpm_per_env, particles_per_env, env);
    const m::Vec3 xp = pos[p];
    const float gx = (xp.x - origin.x) * inv_dx;
    const float gy = (xp.y - origin.y) * inv_dx;
    const float gz = (xp.z - origin.z) * inv_dx;
    // Non-finite or out-of-float-range coords are UB in the float->int cast;
    // detect before converting and park the particle on the escape path.
    const bool nonfinite = !isfinite(gx) || !isfinite(gy) || !isfinite(gz) ||
                          fabsf(gx) >= 0x1p62f || fabsf(gy) >= 0x1p62f || fabsf(gz) >= 0x1p62f;
    const int64_t bx = nonfinite ? 0 : static_cast<int64_t>(floorf(gx - 0.5f));
    const int64_t by = nonfinite ? 0 : static_cast<int64_t>(floorf(gy - 0.5f));
    const int64_t bz = nonfinite ? 0 : static_cast<int64_t>(floorf(gz - 0.5f));
    // Walled x/y faces + plane floor contain a pressed particle (flag only a base off
    // the grid there); the open +z top flags a stencil clip (base+2 past the ceiling).
    const int64_t bx64 = bx, by64 = by, bz64 = bz;
    const bool escaped = nonfinite ||
                         bx64 < 0 || bx64 >= static_cast<int64_t>(dims_x) ||
                         by64 < 0 || by64 >= static_cast<int64_t>(dims_y) ||
                         bz64 < 0 || bz64 + 2 >= static_cast<int64_t>(dims_z);
    if (escaped && env_status != nullptr) {
        atomicOr(&env_status[env], kEnvStatusMpmGridEscape);
    }
    const int64_t cx = bx < 0 ? 0 : (bx >= static_cast<int64_t>(dims_x) ?
                                     static_cast<int64_t>(dims_x) - 1 : bx);
    const int64_t cy = by < 0 ? 0 : (by >= static_cast<int64_t>(dims_y) ?
                                     static_cast<int64_t>(dims_y) - 1 : by);
    const int64_t cz = bz < 0 ? 0 : (bz >= static_cast<int64_t>(dims_z) ?
                                     static_cast<int64_t>(dims_z) - 1 : bz);
    const uint32_t local = static_cast<uint32_t>(
        (cz * dims_y + cy) * dims_x + cx);
    keys[t] = env * cells_per_env + local;
    idx[t] = p;
}

// Initialize physical grid fields and independent transfer indexing before active work.
__global__ void MpmGridPrepareKernel(uint32_t total_nodes,
                                     float* __restrict__ grid_mass,
                                     m::Vec3* __restrict__ grid_momentum,
                                     m::Vec3* __restrict__ grid_velocity,
                                     m::Vec3* __restrict__ grid_pseudo_velocity,
                                     float* __restrict__ grid_inv_mass,
                                     uint32_t* __restrict__ active_node_flags,
                                     uint32_t* __restrict__ cell_start,
                                     uint32_t* __restrict__ node_ids) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_nodes) return;
    grid_mass[i] = 0.0f;
    grid_momentum[i] = m::Vec3::Zero();
    grid_velocity[i] = m::Vec3::Zero();
    grid_pseudo_velocity[i] = m::Vec3::Zero();
    grid_inv_mass[i] = 0.0f;
    active_node_flags[i] = 0u;
    cell_start[i] = ~0u;
    node_ids[i] = i;
}

// Each sorted cell run owns its boundaries and marks its common 27-node stencil once.
__global__ void MpmBuildCellRangesKernel(
    uint32_t particle_count, const uint32_t* __restrict__ sorted_keys,
    uint32_t total_cells, uint32_t* __restrict__ cell_start,
    uint32_t* __restrict__ cell_end, uint32_t cells_per_env, uint32_t nodes_per_env,
    uint32_t dims_x, uint32_t dims_y, uint32_t dims_z,
    uint32_t* __restrict__ active_node_flags) {
    const uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= particle_count) return;
    const uint32_t key = sorted_keys[s];
    if (key >= total_cells) return;
    if (s == 0u || sorted_keys[s - 1u] != key) {
        cell_start[key] = s;
        const uint32_t env = key / cells_per_env;
        const uint32_t local = key % cells_per_env;
        const int64_t bx = local % dims_x;
        const int64_t by = (local / dims_x) % dims_y;
        const int64_t bz = local / (dims_x * dims_y);
        const uint32_t dims[3] = {dims_x, dims_y, dims_z};
        for (int64_t a = 0; a < 3; ++a)
            for (int64_t b = 0; b < 3; ++b)
                for (int64_t c = 0; c < 3; ++c) {
                    const int64_t node = nk::MpmNodeIndex(env, bx + a, by + b, bz + c, dims, nodes_per_env);
                    if (node >= 0) atomicExch(&active_node_flags[node], 1u);
                }
    }
    if (s + 1u == particle_count || sorted_keys[s + 1u] != key)
        cell_end[key] = s + 1u;
}

__global__ void MpmPrepareTransferInputKernel(
    uint32_t particle_count, const uint32_t* __restrict__ sorted_idx,
    const m::Vec3* __restrict__ pos, const float* __restrict__ inv_mass,
    const m::Vec3* __restrict__ velocity, const float* __restrict__ affine,
    const float* __restrict__ deformation, const float* __restrict__ volume,
    const uint32_t* __restrict__ material_ids, const nk::MpmMaterial* __restrict__ material_table,
    uint32_t material_count, float* __restrict__ particle_stress,
    uint32_t particles_per_env, uint32_t* __restrict__ env_status,
    m::Vec3 origin, float inv_dx, uint32_t dims_x, uint32_t dims_y, uint32_t dims_z,
    MpmTransferInput* __restrict__ transfer_input) {
    const uint32_t block_begin = blockIdx.x * blockDim.x;
    const uint32_t s = block_begin + threadIdx.x;
    MpmTransferInput cached{};
    if (s < particle_count) {
        const uint32_t p = sorted_idx[s];
        if (!MpmParticleStress(p, affine, deformation, volume, material_ids,
                               material_table, material_count, cached.stress)) {
            atomicOr(&env_status[p / particles_per_env], kEnvStatusConstitutiveFailure);
            for (int i = 0; i < 9; ++i) cached.stress[i] = 0.0f;
        }
        for (int i = 0; i < 9; ++i)
            particle_stress[static_cast<size_t>(p) * 9u + i] = cached.stress[i];
        const m::Vec3 xp = pos[p];
        const nk::MpmQuadraticBasis axes[3] = {nk::MpmQuadraticWeights((xp.x - origin.x) * inv_dx),
                                 nk::MpmQuadraticWeights((xp.y - origin.y) * inv_dx),
                                 nk::MpmQuadraticWeights((xp.z - origin.z) * inv_dx)};
        float mass = inv_mass[p] > 0.0f ? 1.0f / inv_mass[p] : 0.0f;
        const bool overlaps_grid = axes[0].base >= -2 && axes[0].base < dims_x &&
            axes[1].base >= -2 && axes[1].base < dims_y &&
            axes[2].base >= -2 && axes[2].base < dims_z;
        if (!overlaps_grid) mass = 0.0f;
        cached.position = xp;
        cached.mass = mass;
        cached.velocity = velocity[p];
        cached.volume = volume[p];
        for (int i = 0; i < 3; ++i)
            cached.base[i] = overlaps_grid ? static_cast<int32_t>(axes[i].base) : 0;
        for (int i = 0; i < 9; ++i)
            cached.affine[i] = affine[static_cast<size_t>(p) * 9u + i];
    }

    static_assert(std::is_trivially_copyable_v<MpmTransferInput>);
    static_assert(sizeof(MpmTransferInput) % sizeof(uint32_t) == 0u);
    constexpr int kWords = sizeof(MpmTransferInput) / sizeof(uint32_t);
    using Store = cub::BlockStore<uint32_t, kBlockSize, kWords, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    __shared__ typename Store::TempStorage storage;
    uint32_t words[kWords];
    memcpy(words, &cached, sizeof(cached));
    const uint32_t valid_words = min(kBlockSize, particle_count - block_begin) * kWords;
    // All lanes join the transpose; the tail block writes only complete valid records.
    Store(storage).Store(reinterpret_cast<uint32_t*>(transfer_input) +
                             static_cast<size_t>(block_begin) * kWords, words, valid_words);
}

// The cached base preserves clipped stencils while evaluating only the requested weight.
__device__ __forceinline__ float QuadWeight(float coordinate, int32_t base, int32_t offset) {
    const float fx = __fsub_rn(coordinate, static_cast<float>(base));
    if (offset == 0) return 0.5f * (1.5f - fx) * (1.5f - fx);
    if (offset == 1) {
        const float d = fx - 1.0f;
        return 0.75f - d * d;
    }
    return 0.5f * (fx - 0.5f) * (fx - 0.5f);
}

// Both keys and occupied range starts are injective; use the smaller capacity.
__device__ __forceinline__ uint32_t CellTransferSlot(
    uint32_t key, uint32_t cell_begin, uint32_t particle_count, uint32_t total_cells) {
    return particle_count < total_cells ? cell_begin : key;
}

// Each occupied cell shares particle records across its quadratic stencil nodes.
__global__ void MpmP2GCellsKernel(
    uint32_t particle_count, uint32_t mpm_particles_per_env, uint32_t total_cells,
    uint32_t cells_per_env, const uint32_t* __restrict__ active_nodes,
    const uint32_t* __restrict__ active_node_count,
    const MpmTransferInput* __restrict__ transfer_input,
    const uint32_t* __restrict__ cell_start, const uint32_t* __restrict__ cell_end,
    uint32_t dims_x, uint32_t dims_y, uint32_t dims_z,
    float inv_dx, float dx, float dt, m::Vec3 origin,
    MpmCellTransfer* __restrict__ cell_transfers) {
    constexpr uint32_t kWords = sizeof(MpmTransferInput) / sizeof(uint32_t);
    __shared__ uint32_t records[kCellGroups][kCudaWarpThreads * kWords];
    const uint32_t group = threadIdx.x / kCudaWarpThreads;
    const uint32_t lane = threadIdx.x % kCudaWarpThreads;
    const uint32_t count = min(total_cells, *active_node_count);
    for (uint32_t slot = blockIdx.x * kCellGroups + group; slot < count;
         slot += gridDim.x * kCellGroups) {
        const uint32_t key = active_nodes[slot];
        const uint32_t cell_begin = cell_start[key];
        if (cell_begin == ~0u) continue;
        const uint32_t env = key / cells_per_env;
        const uint32_t local = key % cells_per_env;
        const uint32_t env_begin = env * mpm_particles_per_env;
        const uint32_t env_end = min(env_begin + mpm_particles_per_env, particle_count);
        const uint32_t begin = max(cell_begin, env_begin);
        const uint32_t end = min(cell_end[key], env_end);
        const int64_t nx64 = static_cast<int64_t>(local % dims_x) + lane % kStencilWidth;
        const int64_t ny64 = static_cast<int64_t>((local / dims_x) % dims_y) +
                             (lane / kStencilWidth) % kStencilWidth;
        const int64_t nz64 = static_cast<int64_t>(local / (dims_x * dims_y)) +
                             lane / (kStencilWidth * kStencilWidth);
        const bool valid_node = lane < kStencilNodes &&
            nx64 < dims_x && ny64 < dims_y && nz64 < dims_z;
        const int32_t nx = valid_node ? static_cast<int32_t>(nx64) : 0;
        const int32_t ny = valid_node ? static_cast<int32_t>(ny64) : 0;
        const int32_t nz = valid_node ? static_cast<int32_t>(nz64) : 0;
        const m::Vec3 xi{origin.x + nx * dx, origin.y + ny * dx, origin.z + nz * dx};
        const float stress_scale = 4.0f * inv_dx * inv_dx;
        float mass = 0.0f;
        m::Vec3 momentum = m::Vec3::Zero();
        for (uint32_t tile = begin; tile < end; tile += kCudaWarpThreads) {
            const uint32_t tile_count = min(kCudaWarpThreads, end - tile);
            const auto* input = reinterpret_cast<const uint32_t*>(transfer_input + tile);
            for (uint32_t word = lane; word < tile_count * kWords; word += kCudaWarpThreads)
                records[group][word] = input[word];
            __syncwarp(kFullWarpMask);
            if (valid_node) {
                for (uint32_t source = 0u; source < tile_count; ++source) {
                    uint32_t words[kWords];
                    #pragma unroll
                    for (uint32_t word = 0u; word < kWords; ++word)
                        words[word] = records[group][source * kWords + word];
                    MpmTransferInput cached;
                    memcpy(&cached, words, sizeof(cached));
                    const int32_t ox = nx - cached.base[0], oy = ny - cached.base[1],
                                  oz = nz - cached.base[2];
                    if (!(cached.mass > 0.0f) ||
                        ox < 0 || ox > 2 || oy < 0 || oy > 2 || oz < 0 || oz > 2) continue;
                    const m::Vec3 xp = cached.position;
                    const float wx = QuadWeight((xp.x - origin.x) * inv_dx, cached.base[0], ox);
                    const float wy = QuadWeight((xp.y - origin.y) * inv_dx, cached.base[1], oy);
                    const float wz = QuadWeight((xp.z - origin.z) * inv_dx, cached.base[2], oz);
                    const float w = wx * wy * wz;
                    const float wm = w * cached.mass;
                    const m::Vec3 dpos = xi - xp;
                    const float* C = cached.affine;
                    const m::Vec3 affine{
                        C[0] * dpos.x + C[1] * dpos.y + C[2] * dpos.z,
                        C[3] * dpos.x + C[4] * dpos.y + C[5] * dpos.z,
                        C[6] * dpos.x + C[7] * dpos.y + C[8] * dpos.z};
                    mass = __fadd_rn(mass, wm);
                    momentum.x = __fadd_rn(momentum.x, wm * (cached.velocity.x + affine.x));
                    momentum.y = __fadd_rn(momentum.y, wm * (cached.velocity.y + affine.y));
                    momentum.z = __fadd_rn(momentum.z, wm * (cached.velocity.z + affine.z));
                    if (dt > 0.0f && cached.volume > 0.0f) {
                        const float* stress = cached.stress;
                        const float coef = -w * cached.volume * stress_scale;
                        momentum.x = __fadd_rn(momentum.x, dt * coef *
                            (stress[0] * dpos.x + stress[1] * dpos.y + stress[2] * dpos.z));
                        momentum.y = __fadd_rn(momentum.y, dt * coef *
                            (stress[3] * dpos.x + stress[4] * dpos.y + stress[5] * dpos.z));
                        momentum.z = __fadd_rn(momentum.z, dt * coef *
                            (stress[6] * dpos.x + stress[7] * dpos.y + stress[8] * dpos.z));
                    }
                }
            }
            __syncwarp(kFullWarpMask);
        }
        const uint32_t transfer_slot = CellTransferSlot(key, cell_begin, particle_count, total_cells);
        if (lane < kStencilNodes)
            cell_transfers[static_cast<size_t>(transfer_slot) * kStencilNodes + lane] = {mass, momentum};
    }
}

// Each node merges cell partials in fixed order and applies its velocity boundary conditions.
__global__ void MpmGridFinalizeKernel(
    uint32_t total_nodes, const uint32_t* __restrict__ active_nodes,
    const uint32_t* __restrict__ active_node_count,
    const MpmCellTransfer* __restrict__ cell_transfers,
    const uint32_t* __restrict__ cell_start, uint32_t particle_count,
    uint32_t nodes_per_env, uint32_t cells_per_env,
    uint32_t dims_x, uint32_t dims_y, uint32_t dims_z,
    float dx, m::Vec3 origin, m::Vec3 gravity, float dt,
    float* __restrict__ grid_mass, m::Vec3* __restrict__ grid_momentum,
    m::Vec3* __restrict__ grid_velocity, float* __restrict__ grid_inv_mass) {
    const uint32_t active_count = min(total_nodes, *active_node_count);
    for (uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x; slot < active_count;
         slot += gridDim.x * blockDim.x) {
        const uint32_t node = active_nodes[slot];
        const uint32_t env = node / nodes_per_env;
        const uint32_t local = node % nodes_per_env;
        const int32_t nx = static_cast<int32_t>(local % dims_x);
        const int32_t ny = static_cast<int32_t>((local / dims_x) % dims_y);
        const int32_t nz = static_cast<int32_t>(local / (dims_x * dims_y));
        float mass = 0.0f;
        m::Vec3 momentum = m::Vec3::Zero();
        for (int32_t dz = -2; dz <= 0; ++dz) {
            const int32_t cz = nz + dz;
            if (cz < 0 || cz >= static_cast<int32_t>(dims_z)) continue;
            for (int32_t dy = -2; dy <= 0; ++dy) {
                const int32_t cy = ny + dy;
                if (cy < 0 || cy >= static_cast<int32_t>(dims_y)) continue;
                for (int32_t dx_i = -2; dx_i <= 0; ++dx_i) {
                    const int32_t cx = nx + dx_i;
                    if (cx < 0 || cx >= static_cast<int32_t>(dims_x)) continue;
                    const uint32_t key = env * cells_per_env +
                        (static_cast<uint32_t>(cz) * dims_y + static_cast<uint32_t>(cy)) * dims_x +
                        static_cast<uint32_t>(cx);
                    const uint32_t cell_begin = cell_start[key];
                    if (cell_begin == ~0u) continue;
                    const uint32_t cell_slot = CellTransferSlot(key, cell_begin, particle_count, total_nodes);
                    const uint32_t offset = static_cast<uint32_t>(
                        ((nz - cz) * kStencilWidth + (ny - cy)) * kStencilWidth + (nx - cx));
                    const auto contribution = cell_transfers[
                        static_cast<size_t>(cell_slot) * kStencilNodes + offset];
                    mass = __fadd_rn(mass, contribution.mass);
                    momentum.x = __fadd_rn(momentum.x, contribution.momentum.x);
                    momentum.y = __fadd_rn(momentum.y, contribution.momentum.y);
                    momentum.z = __fadd_rn(momentum.z, contribution.momentum.z);
                }
            }
        }
        grid_mass[node] = mass;
        grid_momentum[node] = momentum;
        if (mass < kMinNodeMass) { grid_velocity[node] = m::Vec3::Zero(); continue; }
        const float inv_mass = 1.0f / mass;
        grid_inv_mass[node] = inv_mass;
        grid_velocity[node] = momentum * inv_mass + gravity * dt;
    }
}

__global__ void MpmG2PGatherKernel(uint32_t mpm_count,
                                   uint32_t particles_per_env, uint32_t mpm_per_env,
                                   uint32_t nodes_per_env,
                                   uint32_t dims_x, uint32_t dims_y, uint32_t dims_z,
                                   float inv_dx, float dx, float dt, m::Vec3 origin,
                                   const float* __restrict__ inv_mass,
                                   const m::Vec3* __restrict__ grid_velocity,
                                   const m::Vec3* __restrict__ grid_pseudo_velocity,
                                   m::Vec3* __restrict__ part_pos,
                                   m::Vec3* __restrict__ part_vel,
                                   float* __restrict__ part_C) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= mpm_count) return;
    uint32_t env = 0u;
    const uint32_t p = MpmSliceGlobal(t, mpm_per_env, particles_per_env, env);
    const m::Vec3 xp = part_pos[p];
    const nk::MpmQuadraticBasis wxs = nk::MpmQuadraticWeights((xp.x - origin.x) * inv_dx);
    const nk::MpmQuadraticBasis wys = nk::MpmQuadraticWeights((xp.y - origin.y) * inv_dx);
    const nk::MpmQuadraticBasis wzs = nk::MpmQuadraticWeights((xp.z - origin.z) * inv_dx);
    m::Vec3 vp = m::Vec3::Zero();
    m::Vec3 pseudo = m::Vec3::Zero();
    float C[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    const uint32_t dims[3] = {dims_x, dims_y, dims_z};
    for (int64_t a = 0; a < 3; ++a) {
        for (int64_t b = 0; b < 3; ++b) {
            for (int64_t c = 0; c < 3; ++c) {
                const int64_t ix = wxs.base + a, iy = wys.base + b, iz = wzs.base + c;
                const int64_t id = nk::MpmNodeIndex(env, ix, iy, iz, dims, nodes_per_env);
                if (id < 0) continue;
                const float w = wxs.w[a] * wys.w[b] * wzs.w[c];
                const m::Vec3 vi = grid_velocity[static_cast<size_t>(id)];
                const m::Vec3 correction = grid_pseudo_velocity[static_cast<size_t>(id)];
                pseudo.x = __fadd_rn(pseudo.x, w * correction.x);
                pseudo.y = __fadd_rn(pseudo.y, w * correction.y);
                pseudo.z = __fadd_rn(pseudo.z, w * correction.z);
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
    // Advect only a movable particle (a pinned inv_mass==0 sample holds position).
    if (inv_mass != nullptr && inv_mass[p] <= 0.0f) { part_vel[p] = vp; return; }
    m::Vec3 np = xp + (vp + pseudo) * dt;
    part_vel[p] = vp;
    part_pos[p] = np;
}

// Elastic F follows the affine map; fluids retain its divergence-driven volume.
// Plastic return maps commit history once per completed grid interval.
__global__ void MpmUpdateFKernel(uint32_t mpm_count, float dt,
                                 const float* __restrict__ part_C,
                                 const uint32_t* __restrict__ part_mat,
                                 const nk::MpmMaterial* __restrict__ material_table,
                                 uint32_t material_count, uint32_t particles_per_env,
                                 uint32_t mpm_per_env,
                                 float* __restrict__ part_F,
                                 float* __restrict__ part_plastic_F,
                                 float* __restrict__ part_plastic,
                                 uint32_t* __restrict__ env_status) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= mpm_count) return;
    uint32_t env_unused = 0u;
    const uint32_t p = MpmSliceGlobal(t, mpm_per_env, particles_per_env, env_unused);
    const float* C = part_C + static_cast<size_t>(p) * 9u;
    float* F = part_F + static_cast<size_t>(p) * 9u;
    float kind = 0.0f, youngs = 0.0f, poisson = 0.0f, dpf = 0.0f, dpc = 0.0f;
    const nk::MpmMaterial* material = nullptr;
    if (part_mat != nullptr && material_table != nullptr) {
        const uint32_t mid = part_mat[p];
        if (mid < material_count) {
            material = material_table + mid;
            youngs = material->youngs; poisson = material->poisson;
            dpf = material->dp_friction; dpc = material->dp_cohesion;
            kind = material->model_kind;
        }
    }
    if (kind == nk::MpmMaterial::kHenckyJ2) {
        if (part_plastic_F == nullptr || part_plastic == nullptr) {
            atomicOr(&env_status[p / particles_per_env], kEnvStatusConstitutiveFailure);
            return;
        }
        float map[9], trial[9];
        for (int k = 0; k < 9; ++k) map[k] = dt * C[k] + (k % 4 == 0 ? 1.0f : 0.0f);
        Mat3Mul(map, F, trial);
        if (nk::material::ReturnHenckyJ2(trial,
                {youngs, poisson, material->yield_stress, material->hardening_modulus},
                F, part_plastic_F + static_cast<size_t>(p) * 9u, part_plastic[p]) !=
            nk::material::ConstitutiveStatus::Ok)
            atomicOr(&env_status[p / particles_per_env], kEnvStatusConstitutiveFailure);
    } else if (kind > 3.5f) {  // granular: elastic predictor and Drucker-Prager return.
        float IpdtC[9];
        for (int k = 0; k < 9; ++k) IpdtC[k] = dt * C[k];
        IpdtC[0] += 1.0f; IpdtC[4] += 1.0f; IpdtC[8] += 1.0f;
        float Fn[9];
        for (int k = 0; k < 9; ++k) Fn[k] = F[k];
        Mat3Mul(IpdtC, Fn, F);  // trial elastic F = (I + dt*C) * F^n.
        const float denom = (1.0f + poisson) * (1.0f - 2.0f * poisson);
        const float mu = youngs / (2.0f * (1.0f + poisson));
        const float lambda = (denom > 1e-9f) ? youngs * poisson / denom : 0.0f;
        SandReturnMap(F, mu, lambda, dpf, dpc);
    } else if (kind > 2.5f) {  // fluid: volume-only update J *= (1 + dt*tr C), F = cbrt(J)*I.
        // Shear leaves a fluid's volume unchanged, so track J off the divergence tr(C);
        // no det of the full affine map => no shear-driven inversion at a hard impact.
        const float Jraw = Mat3Det(F) * (1.0f + dt * (C[0] + C[4] + C[8]));
        if (!(Jraw > 0.0f) || !isfinite(Jraw)) {
            if (env_status != nullptr)
                atomicOr(&env_status[p / particles_per_env], kEnvStatusMpmGridEscape);
            return;
        }
        const float s = cbrtf(Jraw);
        for (int k = 0; k < 9; ++k) F[k] = (k % 4 == 0) ? s : 0.0f;
    } else {  // elastic: F^{n+1} = (I + dt*C) F^n.
        float IpdtC[9];
        for (int k = 0; k < 9; ++k) IpdtC[k] = dt * C[k];
        IpdtC[0] += 1.0f; IpdtC[4] += 1.0f; IpdtC[8] += 1.0f;
        float Fn[9];
        for (int k = 0; k < 9; ++k) Fn[k] = F[k];
        Mat3Mul(IpdtC, Fn, F);  // (I + dt*C) * F^n.
    }
}

// Sorting outputs may be reused after P2G; node indexing and transfer inputs remain separate.
struct MpmScratch {
    void* sort_temp = nullptr;
    uint32_t* keys_out = nullptr;
    uint32_t* idx_out = nullptr;
    uint32_t* active_nodes = nullptr;
    uint32_t* active_flags = nullptr;
    uint32_t* cell_start = nullptr;
    uint32_t* cell_end = nullptr;
    uint32_t* node_ids = nullptr;
    uint32_t* active_count = nullptr;
    MpmTransferInput* transfer_input = nullptr;
    MpmCellTransfer* cell_transfers = nullptr;
    uint32_t* contact_hit_mask = nullptr;
    size_t sort_temp_bytes = 0u;
};
MpmScratch PartitionScratch(void* base, uint32_t particle_count, uint32_t node_count,
                            uint64_t collidables_per_env) {
    const auto& layout = ScratchLayout(particle_count, node_count, collidables_per_env);
    auto* bytes = static_cast<char*>(base);
    MpmScratch scratch;
    scratch.sort_temp = bytes;
    scratch.keys_out = reinterpret_cast<uint32_t*>(bytes + layout.keys_off);
    scratch.idx_out = reinterpret_cast<uint32_t*>(bytes + layout.idx_off);
    scratch.active_nodes = reinterpret_cast<uint32_t*>(bytes + layout.active_nodes_off);
    scratch.active_flags = reinterpret_cast<uint32_t*>(bytes + layout.active_flags_off);
    scratch.cell_start = reinterpret_cast<uint32_t*>(bytes + layout.cell_start_off);
    scratch.cell_end = reinterpret_cast<uint32_t*>(bytes + layout.cell_end_off);
    scratch.node_ids = reinterpret_cast<uint32_t*>(bytes + layout.node_ids_off);
    scratch.active_count = reinterpret_cast<uint32_t*>(bytes + layout.active_count_off);
    scratch.transfer_input = reinterpret_cast<MpmTransferInput*>(bytes + layout.transfer_input_off);
    scratch.cell_transfers = reinterpret_cast<MpmCellTransfer*>(bytes + layout.cell_transfer_off);
    scratch.contact_hit_mask = reinterpret_cast<uint32_t*>(bytes + layout.contact_hit_mask_off);
    scratch.sort_temp_bytes = static_cast<size_t>(layout.temp_bytes);
    return scratch;
}

enum class MpmOperation { Predict, Exchange, Commit };

// Grid and transfer scratch survive until the interval commits.
template <MpmOperation operation>
cudaError_t LaunchMpmStage(const MpmParams& p, const ModelView& model,
                         const DataView& data, const MpmScratch& scratch,
                         float dt_sub, uint32_t Ppe, uint32_t mpm_pe, uint32_t mpm_count,
                         uint32_t cpe, uint32_t total_nodes, uint32_t finalize_blocks,
                         uint32_t cell_blocks, float inv_dx,
                         const m::Vec3& origin, cudaStream_t stream) {
    MpmProfiler& profiler = MpmProfiler::Get();
    const uint32_t nblocks = (total_nodes + kBlockSize - 1u) / kBlockSize;
    const uint32_t pblocks = (mpm_count + kBlockSize - 1u) / kBlockSize;
    cudaError_t error = cudaSuccess;
    const auto launch = [&](MpmStage stage, auto kernel, uint32_t blocks, auto... args) {
        if (error != cudaSuccess) return;
        profiler.Start(stage, stream);
        LaunchCuda(kernel, dim3(blocks), dim3(kBlockSize), 0u, stream, args...);
        error = cudaPeekAtLastError();
        profiler.Stop(stage, stream);
    };
    if constexpr (operation == MpmOperation::Predict) {
        launch(MpmStage::GridPrepare, MpmGridPrepareKernel, nblocks,
               total_nodes, data.grid_mass, data.grid_momentum, data.grid_velocity,
               data.grid_pseudo_vel, data.grid_inv_mass, scratch.active_flags,
               scratch.cell_start, scratch.node_ids);
        launch(MpmStage::CellKeys, MpmCellKeysKernel, pblocks,
               mpm_count, data.particle_pos, Ppe, mpm_pe, inv_dx, origin,
               p.grid_dims[0], p.grid_dims[1], p.grid_dims[2], cpe,
               data.mpm_grid_cell_key, data.mpm_grid_part_idx, data.env_status);
        if (error != cudaSuccess) return error;
        const uint32_t total_cells = cpe * p.env_count;
        size_t temp_bytes = scratch.sort_temp_bytes;
        profiler.Start(MpmStage::RadixSort, stream);
        error = cub::DeviceRadixSort::SortPairs(
            scratch.sort_temp, temp_bytes, data.mpm_grid_cell_key, scratch.keys_out,
            data.mpm_grid_part_idx, scratch.idx_out, static_cast<int>(mpm_count), 0,
            RadixBitsInclusive(total_cells - 1u), stream);
        profiler.Stop(MpmStage::RadixSort, stream);
        if (error != cudaSuccess) return error;
        launch(MpmStage::CellRanges, MpmBuildCellRangesKernel, pblocks,
               mpm_count, scratch.keys_out, total_cells, scratch.cell_start, scratch.cell_end,
               cpe, p.nodes_per_env, p.grid_dims[0], p.grid_dims[1], p.grid_dims[2], scratch.active_flags);
        if (error != cudaSuccess) return error;
        thrust::counting_iterator<uint32_t> active_id_iter(0u);
        temp_bytes = scratch.sort_temp_bytes;
        profiler.Start(MpmStage::ActiveSelect, stream);
        error = cub::DeviceSelect::Flagged(
            scratch.sort_temp, temp_bytes, active_id_iter, scratch.active_flags,
            scratch.active_nodes, scratch.active_count, static_cast<int>(total_nodes), stream);
        profiler.Stop(MpmStage::ActiveSelect, stream);
        if (error != cudaSuccess) return error;
        launch(MpmStage::TransferInput, MpmPrepareTransferInputKernel, pblocks,
               mpm_count, scratch.idx_out, data.particle_pos, data.particle_inv_mass,
               data.particle_vel, data.particle_C, data.particle_F,
               data.particle_vol0, data.particle_material_id,
               reinterpret_cast<const nk::MpmMaterial*>(data.mpm_material_table),
               p.material_count, data.mpm_particle_stress, Ppe, data.env_status,
               origin, inv_dx, p.grid_dims[0], p.grid_dims[1], p.grid_dims[2], scratch.transfer_input);
        if (error != cudaSuccess) return error;
        launch(MpmStage::P2GCells, MpmP2GCellsKernel, cell_blocks,
               mpm_count, mpm_pe, total_cells, cpe, scratch.active_nodes, scratch.active_count,
               scratch.transfer_input, scratch.cell_start, scratch.cell_end,
               p.grid_dims[0], p.grid_dims[1], p.grid_dims[2], inv_dx, p.dx, dt_sub, origin,
               scratch.cell_transfers);
        const m::Vec3 g{p.gravity[0], p.gravity[1], p.gravity[2]};
        launch(MpmStage::GridFinalize, MpmGridFinalizeKernel, finalize_blocks,
               total_nodes, scratch.active_nodes, scratch.active_count,
               scratch.cell_transfers, scratch.cell_start, mpm_count,
               p.nodes_per_env, cpe, p.grid_dims[0], p.grid_dims[1], p.grid_dims[2],
               p.dx, origin, g, dt_sub,
               data.grid_mass, data.grid_momentum, data.grid_velocity, data.grid_inv_mass);
        if (error != cudaSuccess) return error;
    }
    if constexpr (operation == MpmOperation::Exchange) {
        const auto surfaces = nkops::MakeSurfaceQueryView(model, p.bodies_per_env,
            p.mesh_geometry, p.sdf_grid_count, p.sdf_cell_total);
        const uint32_t clear_count = p.env_count * std::max(p.contact_capacity, p.point_endpoints_per_env);
        const uint32_t contact_blocks = (clear_count + kBlockSize - 1u) / kBlockSize;
        launch(MpmStage::ContactCount, mpm_contact::ClearSlots, contact_blocks, p, data);
        launch(MpmStage::ContactCount, mpm_contact::Generate<false>, pblocks,
               p, model, data, surfaces, mpm_pe, scratch.contact_hit_mask);
        if (error != cudaSuccess) return error;
        size_t temp_bytes = scratch.sort_temp_bytes;
        profiler.Start(MpmStage::ContactScan, stream);
        error = cub::DeviceScan::ExclusiveSum(scratch.sort_temp, temp_bytes,
            data.grid_contact_count, data.grid_contact_offset, static_cast<int>(mpm_count), stream);
        profiler.Stop(MpmStage::ContactScan, stream);
        if (error != cudaSuccess) return error;
        launch(MpmStage::ContactEmit, mpm_contact::CountDiagnostics,
               (p.env_count + kBlockSize - 1u) / kBlockSize, p, data, mpm_pe);
        launch(MpmStage::ContactEmit, mpm_contact::Generate<true>, pblocks,
               p, model, data, surfaces, mpm_pe, scratch.contact_hit_mask);
    }
    if constexpr (operation == MpmOperation::Commit) {
        launch(MpmStage::ReactionReadout, mpm_contact::ReadReactions<kBlockSize>,
               p.env_count * (p.bodies_per_env + nk::kMpmBoundaryCount), p, model, data);
        launch(MpmStage::G2P, MpmG2PGatherKernel, pblocks,
               mpm_count, Ppe, mpm_pe, p.nodes_per_env, p.grid_dims[0], p.grid_dims[1],
               p.grid_dims[2], inv_dx, p.dx, dt_sub, origin, data.particle_inv_mass,
               data.grid_velocity, data.grid_pseudo_vel, data.particle_pos, data.particle_vel, data.particle_C);
        launch(MpmStage::UpdateF, MpmUpdateFKernel, pblocks, mpm_count,
               dt_sub, data.particle_C, data.particle_material_id,
               reinterpret_cast<const nk::MpmMaterial*>(data.mpm_material_table),
               p.material_count, Ppe, mpm_pe, data.particle_F, data.particle_plastic_F,
               data.particle_plastic, data.env_status);
    }
    return error;
}

template <MpmOperation operation>
Status OpMpmStage(const ModelView& model, const DataView& data,
                 const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const MpmParams*>(params);
    if (p == nullptr) return Status::Failed;
    MpmProfiler& profiler = MpmProfiler::Get();
    if constexpr (operation == MpmOperation::Predict) profiler.BeginInterval();
    if ((p->mode != kParticleModeMpm && p->mode != kParticleModeMpmXpbd) || p->particle_count == 0u) {
        return Status::Ok;
    }
    if (p->env_count == 0u || p->nodes_per_env == 0u || !(p->dx > 0.0f) ||
        !std::isfinite(p->dx) || !std::isfinite(p->dt) || p->dt < 0.0f ||
        p->grid_dims[0] < nk::kMpmStencilWidth || p->grid_dims[1] < nk::kMpmStencilWidth ||
        p->grid_dims[2] < nk::kMpmStencilWidth ||
        p->substeps > 1u)
        return Status::InvalidArgument;
    if (p->contact_capacity == 0u || p->contact_slot_base < p->full_row_slot_count ||
        uint64_t{p->contact_slot_base} + p->contact_capacity > p->contact_slots_per_env ||
        uint64_t{p->contact_slots_per_env} * p->env_count > INT_MAX ||
        !data.ucontact_law || !data.ucontact_friction || !data.grid_contact_attempted ||
        !data.grid_contact_retained || !data.grid_contact_peak || !data.grid_contact_overflow ||
        !data.mpm_boundary_impulse || !data.mpm_boundary_moment)
        return Status::InvalidArgument;
    const uint64_t total_nodes64 =
        static_cast<uint64_t>(p->nodes_per_env) * p->env_count;
    const uint64_t grid_xy = static_cast<uint64_t>(p->grid_dims[0]) * p->grid_dims[1];
    if (grid_xy > p->nodes_per_env || grid_xy * p->grid_dims[2] != p->nodes_per_env ||
        total_nodes64 > static_cast<uint64_t>(INT_MAX) ||
        static_cast<uint64_t>(p->bodies_per_env) * p->env_count > static_cast<uint64_t>(INT_MAX))
        return Status::InvalidArgument;
    const uint64_t cells_per_env = grid_xy * p->grid_dims[2];
    const uint32_t Np = p->particle_count;
    const uint32_t Ppe = p->particles_per_env == 0u ? Np : p->particles_per_env;
    if (p->mpm_particles_per_env > Ppe) return Status::InvalidArgument;
    const uint32_t mpm_pe = p->mpm_particles_per_env == 0u ? Ppe : p->mpm_particles_per_env;
    const uint32_t surface_contacts = p->particle_surfaces_per_env > 0u ? p->contact_capacity : 0u;
    if (p->point_endpoints_per_env < nk::MpmPointEndpointCount(Ppe, surface_contacts) ||
        p->point_endpoint_terms_per_env < nk::MpmPointEndpointTermCount(Ppe, surface_contacts) ||
        uint64_t{p->point_endpoints_per_env} * p->env_count > INT_MAX ||
        uint64_t{p->point_endpoint_terms_per_env} * p->env_count > INT_MAX ||
        !data.point_endpoint_ranges || !data.point_endpoint_terms ||
        (p->particle_surfaces_per_env > 0u && !data.particle_surface_max_speed))
        return Status::InvalidArgument;
    const uint64_t mpm_count64 =
        static_cast<uint64_t>(mpm_pe) * p->env_count;
    if (static_cast<uint64_t>(Np) != static_cast<uint64_t>(Ppe) * p->env_count ||
        mpm_count64 > static_cast<uint64_t>(INT_MAX)) return Status::InvalidArgument;
    if (!data.particle_pos || !data.particle_vel || !data.particle_inv_mass ||
        !data.particle_C || !data.particle_F || !data.particle_vol0 ||
        !data.grid_mass || !data.grid_momentum || !data.grid_velocity || !data.grid_pseudo_vel ||
        !data.grid_inv_mass || !data.grid_contact_count || !data.grid_contact_offset ||
        !data.mpm_sort_scratch ||
        !data.mpm_grid_cell_key || !data.mpm_grid_part_idx || !data.mpm_particle_stress ||
        !data.env_status || (p->material_count > 0u && !data.mpm_material_table))
        return Status::InvalidArgument;
    if (p->dynamic_body_bc != 0u && p->bite_disable_dynamic_bc == 0u && p->bodies_per_env > 0u) {
        if (!model.shape_table || !data.body_pose || !data.body_inertial_frame ||
            !data.body_inv_mass || !data.body_world_inv_inertia || !data.body_linear_velocity ||
            !data.body_angular_velocity || !data.mpm_body_reaction || !data.mpm_body_ang_reaction)
            return Status::InvalidArgument;
        if (!nkops::SurfaceQueryStorageValid(nkops::MakeSurfaceQueryView(model,
            p->bodies_per_env, p->mesh_geometry, p->sdf_grid_count, p->sdf_cell_total)))
            return Status::InvalidArgument;
        if (p->artic_count > 0u &&
            (p->base_link_count == 0u || p->artics_per_env == 0u ||
             static_cast<uint64_t>(p->artics_per_env) * p->env_count != p->artic_count ||
             static_cast<uint64_t>(p->base_link_count) * p->env_count > INT_MAX ||
             !data.link_pose || !data.link_velocity || !model.body_to_link ||
             !model.body_to_articulation ||
             (p->max_dof > 0u && (!data.m_inv || !data.qdot_flat)))) return Status::InvalidArgument;
    }
    const uint32_t mpm_count = static_cast<uint32_t>(mpm_count64);
    const uint32_t cpe = static_cast<uint32_t>(cells_per_env);
    const uint32_t total_nodes = static_cast<uint32_t>(total_nodes64);
    const MpmScratch scratch = PartitionScratch(data.mpm_sort_scratch, mpm_count, total_nodes,
        uint64_t{p->bodies_per_env} + p->particle_surfaces_per_env);
    uint32_t finalize_blocks = 0u, cell_blocks = 0u;
    if constexpr (operation == MpmOperation::Predict) {
        if (ResidentGridSize(MpmGridFinalizeKernel, kBlockSize, 0u,
                             (total_nodes + kBlockSize - 1u) / kBlockSize,
                             &finalize_blocks) != cudaSuccess ||
            ResidentGridSize(MpmP2GCellsKernel, kBlockSize, 0u,
                             (std::min(mpm_count, total_nodes) + kCellGroups - 1u) / kCellGroups,
                             &cell_blocks) != cudaSuccess) return Status::Failed;
    }
    const float inv_dx = 1.0f / p->dx;
    const m::Vec3 origin{p->grid_origin[0], p->grid_origin[1], p->grid_origin[2]};
    const float dt_sub = p->dt;
    if constexpr (operation == MpmOperation::Predict) {
        // The pipeline aggregates diagnostics across the shared physical intervals.
        if (data.env_status != nullptr) {
            const uint32_t e = p->env_count == 0u ? 1u : p->env_count;
            const uint32_t eb = (e + kBlockSize - 1u) / kBlockSize;
            LaunchCuda(MpmClearStatusBitsKernel, dim3(eb), dim3(kBlockSize), 0u, stream,
                       data.env_status, e);
        }
        // The reaction probe contains only this interval's impulse.
        if (p->dynamic_body_bc != 0u && p->bodies_per_env > 0u &&
            data.mpm_body_reaction != nullptr) {
            const uint32_t tb = p->bodies_per_env * p->env_count;
            const uint32_t bb = (tb + kBlockSize - 1u) / kBlockSize;
            LaunchCuda(MpmClearBodyReactionKernel, dim3(bb), dim3(kBlockSize), 0u,
                       stream, tb, data.mpm_body_reaction, data.mpm_body_ang_reaction);
        }
    }
    if (cudaPeekAtLastError() != cudaSuccess) return Status::Failed;
    if (LaunchMpmStage<operation>(*p, model, data, scratch, dt_sub, Ppe, mpm_pe, mpm_count, cpe,
                      total_nodes, finalize_blocks, cell_blocks, inv_dx, origin, stream) != cudaSuccess)
        return Status::Failed;
    if (cudaPeekAtLastError() != cudaSuccess) return Status::Failed;
    if constexpr (operation == MpmOperation::Commit) {
        // Eager completion exposes asynchronous faults once per physical interval.
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess) return Status::Failed;
        if (capture == cudaStreamCaptureStatusNone && cudaStreamSynchronize(stream) != cudaSuccess)
            return Status::Failed;
    }
    return Status::Ok;
}

}  // namespace

uint64_t MpmSortScratchBytes(uint32_t particle_count, uint32_t node_count,
                             uint64_t collidables_per_env) {
    if (particle_count == 0u || node_count == 0u ||
        particle_count > static_cast<uint32_t>(INT_MAX) || node_count > static_cast<uint32_t>(INT_MAX))
        return 0u;
    return ScratchLayout(particle_count, node_count, collidables_per_env).total;
}

void RegisterNkMpmOps() {
    SetCudaOp(NkOp::MpmPredict, &OpMpmStage<MpmOperation::Predict>);
    SetCudaOp(NkOp::MpmExchange, &OpMpmStage<MpmOperation::Exchange>);
    SetCudaOp(NkOp::MpmCommit, &OpMpmStage<MpmOperation::Commit>);
}

}  // namespace nuka::phi
