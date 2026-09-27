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
#include <thrust/iterator/transform_iterator.h>

#include "collision/primitive_surface.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/material/mpm_constitutive.hpp"
#include "nk/model/generated/views.hpp"  // ModelView / DataView (complete types)
#include "nk/solve/point_endpoint.hpp"
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
constexpr uint32_t kLatticeNodes = nk::kMpmLatticeStencilNodes;
constexpr uint32_t kStencilNodes = nk::kMpmStencilNodes;
constexpr uint32_t kAxisWeights = nk::kMpmLattices * kStencilWidth;
constexpr uint32_t kCellNodes = nk::kMpmCellStencilNodes;
constexpr uint32_t kCellLatticeWidth = kStencilWidth + 1u;
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
    StressRows,
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
            "contact_scan", "contact_emit", "stress_rows", "reaction_readout", "g2p_gather", "update_F"};
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

// Linearized stress a cell's rows solve; pressure, bulk, shear, viscosity and deviator are
// volume integrals and tension and the yield cone keep the weakest material bound.
struct MpmCellVolume {
    float volume;
    float pressure;
    float bulk;
    float shear;
    float viscosity;
    float deviator[nk::kMpmDeviatorComponents];
    float tension;
    float cone_slope;
    float cone_offset;
};

__device__ __forceinline__ MpmCellVolume EmptyCellVolume() {
    MpmCellVolume volume{};
    volume.tension = volume.cone_slope = volume.cone_offset = FLT_MAX;
    return volume;
}

__device__ __forceinline__ void AddCellVolume(MpmCellVolume& total, const MpmCellVolume& part) {
    total.volume = __fadd_rn(total.volume, part.volume);
    total.pressure = __fadd_rn(total.pressure, part.pressure);
    total.bulk = __fadd_rn(total.bulk, part.bulk);
    total.shear = __fadd_rn(total.shear, part.shear);
    total.viscosity = __fadd_rn(total.viscosity, part.viscosity);
    for (uint32_t k = 0u; k < nk::kMpmDeviatorComponents; ++k)
        total.deviator[k] = __fadd_rn(total.deviator[k], part.deviator[k]);
    total.tension = fminf(total.tension, part.tension);
    total.cone_slope = fminf(total.cone_slope, part.cone_slope);
    total.cone_offset = fminf(total.cone_offset, part.cone_offset);
}

// The half-cell fixes the particle's cell on both lattices; each axis keeps its kernel weights
// per lattice node, and D^-1 is the inverse APIC inertia as xx, yy, zz, xy, xz, yz.
struct MpmTransferInput {
    m::Vec3 position;
    float mass;
    m::Vec3 velocity;
    float volume;
    int32_t half_cell[3];
    float weight[3][kAxisWeights];
    float inverse_moment[6];
    float affine[9];
    float stress[9];
    MpmCellVolume rows;
};

// Stress cells are the lattice-0 cells; their half-cells sort as one contiguous key run.
struct MpmStressCellKey {
    __host__ __device__ uint32_t operator()(uint32_t key) const {
        return key / nk::kMpmHalfCellsPerLatticeNode;
    }
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
    uint64_t occupied_cells_off = 0u;
    uint64_t occupied_count_off = 0u;
    uint64_t stress_cells_off = 0u;
    uint64_t stress_count_off = 0u;
    uint64_t cell_gradients_off = 0u;
    uint64_t cell_volumes_off = 0u;
    uint64_t transfer_input_off = 0u;
    uint64_t cell_transfer_off = 0u;
    uint64_t contact_hit_mask_off = 0u;
    uint64_t contact_reach_off = 0u;
    uint64_t body_hit_records_off = 0u;
    uint64_t body_hit_first_off = 0u;
    uint64_t body_hit_count_off = 0u;
    uint64_t contact_targets_off = 0u;
    uint64_t reaction_heads_off = 0u;
    uint64_t reaction_next_off = 0u;
    uint64_t reaction_contributions_off = 0u;
    uint64_t total      = 0u;     // full segment byte size.
    MpmSortScratchLayout(uint32_t particle_count, uint32_t node_count,
                         uint64_t collidables_per_env, uint64_t contact_count,
                         uint64_t reaction_target_count, uint64_t stress_cell_count) {
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
        const uint64_t cell_count =
            uint64_t{node_count} / nk::kMpmLattices * nk::kMpmHalfCellsPerLatticeNode;
        const uint64_t cell_bytes = cell_count * sizeof(uint32_t);
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
        size_t unique_bytes = 0u;
        const auto unique_status = cub::DeviceSelect::Unique(nullptr, unique_bytes,
            static_cast<const uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
            static_cast<uint32_t*>(nullptr), static_cast<int>(particle_count));
        if (unique_status != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(unique_status));
        size_t stress_bytes = 0u;
        if (stress_cell_count > 0u) {
            const auto stress_status = cub::DeviceSelect::Unique(nullptr, stress_bytes,
                thrust::make_transform_iterator(static_cast<const uint32_t*>(nullptr),
                                                MpmStressCellKey{}),
                static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
                static_cast<int>(particle_count));
            if (stress_status != cudaSuccess)
                throw std::runtime_error(cudaGetErrorString(stress_status));
        }
        size_t scan_bytes = 0u;
        const auto scan_status = cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes,
            static_cast<const uint64_t*>(nullptr), static_cast<uint64_t*>(nullptr),
            static_cast<int>(std::max(particle_count, node_count)));
        if (scan_status != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(scan_status));
        temp_bytes = std::max({particle_sort_bytes, node_sort_bytes, select_bytes, unique_bytes,
                               stress_bytes, scan_bytes});
        const uint64_t output_bytes =
            uint64_t{std::max(particle_count, node_count)} * sizeof(uint32_t);
        const uint64_t node_bytes = uint64_t{node_count} * sizeof(uint32_t);
        keys_off = AlignScratch(temp_bytes);
        idx_off  = AlignScratch(keys_off + output_bytes);
        active_nodes_off = AlignScratch(idx_off + output_bytes);
        active_flags_off = AlignScratch(active_nodes_off + node_bytes);
        cell_start_off = AlignScratch(active_flags_off + node_bytes);
        cell_end_off = AlignScratch(cell_start_off + cell_bytes);
        node_ids_off = AlignScratch(cell_end_off + cell_bytes);
        active_count_off = AlignScratch(node_ids_off + node_bytes);
        occupied_cells_off = AlignScratch(active_count_off + sizeof(uint32_t));
        occupied_count_off = AlignScratch(occupied_cells_off +
                                          uint64_t{particle_count} * sizeof(uint32_t));
        const uint64_t transfer_slots = std::min(uint64_t{particle_count}, cell_count);
        const bool stress = stress_cell_count > 0u;
        stress_cells_off = AlignScratch(occupied_count_off + sizeof(uint32_t));
        stress_count_off = AlignScratch(stress_cells_off +
            (stress ? uint64_t{particle_count} * sizeof(uint32_t) : 0u));
        cell_gradients_off = AlignScratch(stress_count_off + sizeof(uint32_t));
        cell_volumes_off = AlignScratch(cell_gradients_off +
            (stress ? transfer_slots * kStencilNodes * sizeof(m::Vec3) : 0u));
        transfer_input_off = AlignScratch(cell_volumes_off +
            (stress ? transfer_slots * sizeof(MpmCellVolume) : 0u));
        cell_transfer_off = AlignScratch(transfer_input_off +
                                        uint64_t{particle_count} * sizeof(MpmTransferInput));
        contact_hit_mask_off = AlignScratch(cell_transfer_off +
            transfer_slots * kStencilNodes * sizeof(MpmCellTransfer));
        const uint64_t mask_words = (collidables_per_env + 31u) / 32u;
        contact_reach_off = AlignScratch(contact_hit_mask_off + uint64_t{particle_count} *
                                         mask_words * sizeof(uint32_t));
        body_hit_records_off = AlignScratch(contact_reach_off +
                                            uint64_t{particle_count} * sizeof(float));
        body_hit_first_off = AlignScratch(body_hit_records_off +
            uint64_t{particle_count} * sizeof(mpm_contact::BodyHit));
        body_hit_count_off = AlignScratch(body_hit_first_off +
            (uint64_t{particle_count} + 31u) / 32u * collidables_per_env * sizeof(uint32_t));
        contact_targets_off = AlignScratch(body_hit_count_off + sizeof(uint32_t));
        reaction_next_off = AlignScratch(contact_targets_off + contact_count * sizeof(uint32_t));
        reaction_heads_off = AlignScratch(reaction_next_off + contact_count * sizeof(uint32_t));
        reaction_contributions_off = AlignScratch(reaction_heads_off +
            reaction_target_count * kBlockSize * sizeof(uint32_t));
        total = AlignScratch(reaction_contributions_off +
            contact_count * sizeof(mpm_contact::ReactionContribution));
    }
};

const MpmSortScratchLayout& ScratchLayout(uint32_t particle_count, uint32_t node_count,
                                          uint64_t collidables_per_env, uint64_t contact_count,
                                          uint64_t reaction_target_count,
                                          uint64_t stress_cell_count) {
    int device = -1;
    const auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    struct Entry {
        int device;
        uint32_t particles;
        uint32_t nodes;
        uint64_t collidables;
        uint64_t contacts;
        uint64_t reaction_targets;
        uint64_t stress_cells;
        MpmSortScratchLayout layout;
    };
    static thread_local std::vector<Entry> entries;
    for (const auto& entry : entries)
        if (entry.device == device && entry.particles == particle_count &&
            entry.nodes == node_count && entry.collidables == collidables_per_env &&
            entry.contacts == contact_count && entry.reaction_targets == reaction_target_count &&
            entry.stress_cells == stress_cell_count)
            return entry.layout;
    entries.push_back({device, particle_count, node_count, collidables_per_env, contact_count,
                       reaction_target_count, stress_cell_count,
                       MpmSortScratchLayout(particle_count, node_count, collidables_per_env,
                                            contact_count, reaction_target_count,
                                            stress_cell_count)});
    return entries.back().layout;
}

struct CudaMpmArithmetic {
    __device__ __forceinline__ static float Sum3(float a, float b, float c) {
        return __fadd_rn(__fadd_rn(a, b), c);
    }
};

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
    const nk::MpmMaterial material = material_table != nullptr && mid < material_count
        ? material_table[mid] : nk::MpmMaterial{};
    const size_t offset = static_cast<size_t>(p) * 9u;
    return nk::material::EvaluateMpmKirchhoff<CudaMpmArithmetic>(material,
        part_F + offset, part_C != nullptr ? part_C + offset : nullptr, stress) ==
        nk::material::ConstitutiveStatus::Ok;
}

// Half-cell keys are cell-major, so each lattice-0 cell owns eight consecutive keys.
__device__ __forceinline__ uint32_t HalfCellLocal(const int64_t* h, uint32_t dims_x,
                                                  uint32_t dims_y) {
    const uint64_t cell = (uint64_t(h[2] >> 1) * dims_y + uint64_t(h[1] >> 1)) * dims_x +
                          uint64_t(h[0] >> 1);
    return static_cast<uint32_t>(cell * nk::kMpmHalfCellsPerLatticeNode +
                                 ((h[2] & 1) << 2) + ((h[1] & 1) << 1) + (h[0] & 1));
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
    const int64_t h[3] = {nonfinite ? 0 : nk::MpmHalfCell(gx), nonfinite ? 0 : nk::MpmHalfCell(gy),
                          nonfinite ? 0 : nk::MpmHalfCell(gz)};
    const uint32_t dims[3] = {dims_x, dims_y, dims_z};
    // A clipped transfer stencil invalidates partition of unity on every grid face.
    bool escaped = nonfinite;
    int64_t clamped[3];
    for (int axis = 0; axis < 3; ++axis) {
        const int64_t halves = 2 * static_cast<int64_t>(dims[axis]);
        escaped = escaped || nk::MpmLatticeBase(h[axis], 0u) < 0 ||
                  nk::MpmLatticeBase(h[axis], nk::kMpmLattices - 1u) < 0 ||
                  nk::MpmLatticeBase(h[axis], 0u) + 1 >= dims[axis] ||
                  nk::MpmLatticeBase(h[axis], nk::kMpmLattices - 1u) + 1 >= dims[axis];
        clamped[axis] = h[axis] < 0 ? 0 : (h[axis] >= halves ? halves - 1 : h[axis]);
    }
    if (escaped && env_status != nullptr) {
        atomicOr(&env_status[env], kEnvStatusMpmGridEscape);
    }
    keys[t] = env * cells_per_env + HalfCellLocal(clamped, dims_x, dims_y);
    idx[t] = p;
}

// Initialize physical grid fields and independent transfer indexing before active work.
__global__ void MpmGridPrepareKernel(uint32_t total_nodes, uint32_t total_cells,
                                     float* __restrict__ grid_mass,
                                     m::Vec3* __restrict__ grid_momentum,
                                     m::Vec3* __restrict__ grid_velocity,
                                     m::Vec3* __restrict__ grid_pseudo_velocity,
                                     float* __restrict__ grid_inv_mass,
                                     uint32_t* __restrict__ active_node_flags,
                                     uint32_t* __restrict__ cell_start,
                                     uint32_t* __restrict__ node_ids) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t cell = i; cell < total_cells; cell += gridDim.x * blockDim.x)
        cell_start[cell] = ~0u;
    if (i >= total_nodes) return;
    grid_mass[i] = 0.0f;
    grid_momentum[i] = m::Vec3::Zero();
    grid_velocity[i] = m::Vec3::Zero();
    grid_pseudo_velocity[i] = m::Vec3::Zero();
    grid_inv_mass[i] = 0.0f;
    active_node_flags[i] = 0u;
    node_ids[i] = i;
}

// A half-cell's node on one lattice: the lattice's cell base plus the lane's corner offset.
__device__ __forceinline__ int64_t HalfCellNode(int64_t half_cell, uint32_t lattice, uint32_t corner) {
    return nk::MpmLatticeBase(half_cell, lattice) + static_cast<int64_t>(corner);
}

__device__ __forceinline__ void DecodeHalfCell(uint32_t local, uint32_t dims_x, uint32_t dims_y,
                                               int64_t* h) {
    const uint32_t cell = local / nk::kMpmHalfCellsPerLatticeNode;
    const uint32_t sub = local % nk::kMpmHalfCellsPerLatticeNode;
    h[0] = 2 * int64_t(cell % dims_x) + (sub & 1u);
    h[1] = 2 * int64_t((cell / dims_x) % dims_y) + ((sub >> 1u) & 1u);
    h[2] = 2 * int64_t(cell / (dims_x * dims_y)) + (sub >> 2u);
}

// Each sorted half-cell run owns its boundaries and marks its common compact stencil once.
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
        int64_t h[3];
        DecodeHalfCell(key % cells_per_env, dims_x, dims_y, h);
        const uint32_t dims[3] = {dims_x, dims_y, dims_z};
        for (uint32_t lane = 0u; lane < kStencilNodes; ++lane) {
            const uint32_t lattice = lane / kLatticeNodes;
            const int64_t node = nk::MpmNodeIndex(env, lattice,
                HalfCellNode(h[0], lattice, lane & 1u), HalfCellNode(h[1], lattice, (lane >> 1u) & 1u),
                HalfCellNode(h[2], lattice, (lane >> 2u) & 1u), dims, nodes_per_env);
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
    uint32_t implicit_stress, MpmTransferInput* __restrict__ transfer_input) {
    const uint32_t block_begin = blockIdx.x * blockDim.x;
    const uint32_t s = block_begin + threadIdx.x;
    MpmTransferInput cached{};
    if (s < particle_count) {
        const uint32_t p = sorted_idx[s];
        const bool rows = implicit_stress != 0u && deformation != nullptr;
        // Stress rows add viscosity through their compliance, so they read the elastic stress.
        if (!MpmParticleStress(p, rows ? nullptr : affine, deformation, volume, material_ids,
                               material_table, material_count, cached.stress)) {
            atomicOr(&env_status[p / particles_per_env], kEnvStatusConstitutiveFailure);
            for (int i = 0; i < 9; ++i) cached.stress[i] = 0.0f;
        }
        for (int i = 0; i < 9; ++i)
            particle_stress[static_cast<size_t>(p) * 9u + i] = cached.stress[i];
        cached.rows = EmptyCellVolume();
        if (rows) {
            // The rows carry the whole stress; the transfer moves mass and momentum.
            nk::MpmMaterial material;
            if (material_ids != nullptr && material_ids[p] < material_count)
                material = material_table[material_ids[p]];
            const auto response = nk::material::EvaluateMpmStressRows(
                material, deformation + static_cast<size_t>(p) * 9u, cached.stress);
            float deviator[nk::kMpmDeviatorComponents];
            nk::material::MpmDeviatorComponents(cached.stress, deviator);
            const float v = volume[p];
            cached.rows.volume = v;
            cached.rows.pressure = v * response.pressure;
            cached.rows.bulk = v * response.bulk;
            cached.rows.shear = v * response.shear;
            cached.rows.viscosity = v * response.viscosity;
            for (uint32_t k = 0u; k < nk::kMpmDeviatorComponents; ++k)
                cached.rows.deviator[k] = v * deviator[k];
            cached.rows.tension = response.tension;
            cached.rows.cone_slope = response.cone_slope;
            cached.rows.cone_offset = response.cone_offset;
            for (float& entry : cached.stress) entry = 0.0f;
        }
        const m::Vec3 xp = pos[p];
        const float coordinate[3] = {(xp.x - origin.x) * inv_dx, (xp.y - origin.y) * inv_dx,
                                     (xp.z - origin.z) * inv_dx};
        const uint32_t dims[3] = {dims_x, dims_y, dims_z};
        nk::MpmCompactAxis axes[3];
        for (int axis = 0; axis < 3; ++axis) {
            const int64_t h = nk::MpmHalfCell(coordinate[axis]);
            const nk::MpmCompactAxis weights = nk::MpmCompactWeights(coordinate[axis], h);
            axes[axis] = weights;
            // Half-cells -3 and 2 dims + 1 touch no node on either lattice.
            const int64_t outside = 2 * static_cast<int64_t>(dims[axis]) + 1;
            cached.half_cell[axis] = static_cast<int32_t>(h < -3 ? -3 : (h > outside ? outside : h));
            for (uint32_t lattice = 0u; lattice < nk::kMpmLattices; ++lattice)
                for (uint32_t corner = 0u; corner < kStencilWidth; ++corner)
                    cached.weight[axis][lattice * kStencilWidth + corner] =
                        weights.w[lattice][corner];
        }
        if (!nk::MpmApicInverse(axes, inv_dx, cached.inverse_moment))
            for (float& entry : cached.inverse_moment) entry = 0.0f;
        cached.position = xp;
        cached.mass = inv_mass[p] > 0.0f ? 1.0f / inv_mass[p] : 0.0f;
        cached.velocity = velocity[p];
        cached.volume = volume[p];
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

// Both keys and occupied range starts are injective; use the smaller capacity.
__device__ __forceinline__ uint32_t CellTransferSlot(
    uint32_t key, uint32_t cell_begin, uint32_t particle_count, uint32_t total_cells) {
    return particle_count < total_cells ? cell_begin : key;
}

// Each occupied half-cell shares particle records across its compact dual-lattice stencil.
__global__ void MpmP2GCellsKernel(
    uint32_t particle_count, uint32_t mpm_particles_per_env, uint32_t total_cells,
    uint32_t cells_per_env, const uint32_t* __restrict__ occupied_cells,
    const uint32_t* __restrict__ occupied_count,
    const MpmTransferInput* __restrict__ transfer_input,
    const uint32_t* __restrict__ cell_start, const uint32_t* __restrict__ cell_end,
    uint32_t dims_x, uint32_t dims_y, uint32_t dims_z,
    float inv_dx, float dx, float dt, m::Vec3 origin,
    MpmCellTransfer* __restrict__ cell_transfers, m::Vec3* __restrict__ cell_gradients,
    MpmCellVolume* __restrict__ cell_volumes) {
    constexpr uint32_t kWords = sizeof(MpmTransferInput) / sizeof(uint32_t);
    __shared__ uint32_t records[kCellGroups][kCudaWarpThreads * kWords];
    const uint32_t group = threadIdx.x / kCudaWarpThreads;
    const uint32_t lane = threadIdx.x % kCudaWarpThreads;
    const uint32_t count = min(total_cells, *occupied_count);
    for (uint32_t slot = blockIdx.x * kCellGroups + group; slot < count;
         slot += gridDim.x * kCellGroups) {
        const uint32_t key = occupied_cells[slot];
        const uint32_t cell_begin = cell_start[key];
        if (cell_begin == ~0u) continue;
        const uint32_t env = key / cells_per_env;
        const uint32_t local = key % cells_per_env;
        const uint32_t env_begin = env * mpm_particles_per_env;
        const uint32_t env_end = min(env_begin + mpm_particles_per_env, particle_count);
        const uint32_t begin = max(cell_begin, env_begin);
        const uint32_t end = min(cell_end[key], env_end);
        int64_t h[3];
        DecodeHalfCell(local, dims_x, dims_y, h);
        const uint32_t lattice = min(lane / kLatticeNodes, nk::kMpmLattices - 1u);
        const int64_t nx64 = HalfCellNode(h[0], lattice, lane & 1u);
        const int64_t ny64 = HalfCellNode(h[1], lattice, (lane >> 1u) & 1u);
        const int64_t nz64 = HalfCellNode(h[2], lattice, (lane >> 2u) & 1u);
        const bool valid_node = lane < kStencilNodes && nx64 >= 0 && ny64 >= 0 && nz64 >= 0 &&
            nx64 < dims_x && ny64 < dims_y && nz64 < dims_z;
        const int32_t nx = valid_node ? static_cast<int32_t>(nx64) : 0;
        const int32_t ny = valid_node ? static_cast<int32_t>(ny64) : 0;
        const int32_t nz = valid_node ? static_cast<int32_t>(nz64) : 0;
        const float shift = nk::MpmLatticeOffset(lattice);
        const m::Vec3 xi{origin.x + (nx + shift) * dx, origin.y + (ny + shift) * dx,
                         origin.z + (nz + shift) * dx};
        const bool stress = cell_gradients != nullptr;
        float mass = 0.0f;
        m::Vec3 momentum = m::Vec3::Zero();
        m::Vec3 gradient = m::Vec3::Zero();
        MpmCellVolume volume = EmptyCellVolume();
        for (uint32_t tile = begin; tile < end; tile += kCudaWarpThreads) {
            const uint32_t tile_count = min(kCudaWarpThreads, end - tile);
            const auto* input = reinterpret_cast<const uint32_t*>(transfer_input + tile);
            for (uint32_t word = lane; word < tile_count * kWords; word += kCudaWarpThreads)
                records[group][word] = input[word];
            __syncwarp(kFullWarpMask);
            if (valid_node || (stress && lane == 0u)) {
                for (uint32_t source = 0u; source < tile_count; ++source) {
                    uint32_t words[kWords];
                    #pragma unroll
                    for (uint32_t word = 0u; word < kWords; ++word)
                        words[word] = records[group][source * kWords + word];
                    MpmTransferInput cached;
                    memcpy(&cached, words, sizeof(cached));
                    if (!(cached.mass > 0.0f)) continue;
                    if (stress && lane == 0u && cached.volume > 0.0f)
                        AddCellVolume(volume, cached.rows);
                    if (!valid_node) continue;
                    const int64_t ox = nx - nk::MpmLatticeBase(cached.half_cell[0], lattice);
                    const int64_t oy = ny - nk::MpmLatticeBase(cached.half_cell[1], lattice);
                    const int64_t oz = nz - nk::MpmLatticeBase(cached.half_cell[2], lattice);
                    if (ox < 0 || ox > 1 || oy < 0 || oy > 1 || oz < 0 || oz > 1) continue;
                    const m::Vec3 xp = cached.position;
                    const uint32_t row = lattice * kStencilWidth;
                    const float w = nk::kMpmLatticeShare * cached.weight[0][row + ox] *
                                    cached.weight[1][row + oy] * cached.weight[2][row + oz];
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
                        // MLS force -V tau D^-1 (xi - xp) with the full compact-kernel D.
                        const float* stress = cached.stress;
                        const float coef = -w * cached.volume;
                        const float* Dinv = cached.inverse_moment;
                        const m::Vec3 g{Dinv[0] * dpos.x + Dinv[3] * dpos.y + Dinv[4] * dpos.z,
                                        Dinv[3] * dpos.x + Dinv[1] * dpos.y + Dinv[5] * dpos.z,
                                        Dinv[4] * dpos.x + Dinv[5] * dpos.y + Dinv[2] * dpos.z};
                        gradient.x = __fadd_rn(gradient.x, -coef * g.x);
                        gradient.y = __fadd_rn(gradient.y, -coef * g.y);
                        gradient.z = __fadd_rn(gradient.z, -coef * g.z);
                        momentum.x = __fadd_rn(momentum.x, dt * coef *
                            (stress[0] * g.x + stress[1] * g.y + stress[2] * g.z));
                        momentum.y = __fadd_rn(momentum.y, dt * coef *
                            (stress[3] * g.x + stress[4] * g.y + stress[5] * g.z));
                        momentum.z = __fadd_rn(momentum.z, dt * coef *
                            (stress[6] * g.x + stress[7] * g.y + stress[8] * g.z));
                    }
                }
            }
            __syncwarp(kFullWarpMask);
        }
        const uint32_t transfer_slot = CellTransferSlot(key, cell_begin, particle_count, total_cells);
        if (lane < kStencilNodes)
            cell_transfers[static_cast<size_t>(transfer_slot) * kStencilNodes + lane] = {mass, momentum};
        if (stress && lane < kStencilNodes)
            cell_gradients[static_cast<size_t>(transfer_slot) * kStencilNodes + lane] = gradient;
        if (stress && lane == 0u) cell_volumes[transfer_slot] = volume;
    }
}

// Each node merges cell partials in fixed order and applies its velocity boundary conditions.
__global__ void MpmGridFinalizeKernel(
    uint32_t total_nodes, const uint32_t* __restrict__ active_nodes,
    const uint32_t* __restrict__ active_node_count,
    const MpmCellTransfer* __restrict__ cell_transfers,
    const uint32_t* __restrict__ cell_start, uint32_t particle_count, uint32_t total_cells,
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
        const uint32_t lattice_nodes = nodes_per_env / nk::kMpmLattices;
        const uint32_t lattice = (node % nodes_per_env) / lattice_nodes;
        const uint32_t local = (node % nodes_per_env) % lattice_nodes;
        const int64_t n[3] = {local % dims_x, (local / dims_x) % dims_y, local / (dims_x * dims_y)};
        const int64_t halves[3] = {2 * int64_t{dims_x}, 2 * int64_t{dims_y}, 2 * int64_t{dims_z}};
        float mass = 0.0f;
        m::Vec3 momentum = m::Vec3::Zero();
        // Half-cells 2 (n - 1) + l .. 2 n + 1 + l place node n at corner 1 or 0 of lattice l.
        for (int64_t hz = 2 * n[2] - 2 + lattice; hz <= 2 * n[2] + 1 + lattice; ++hz) {
            if (hz < 0 || hz >= halves[2]) continue;
            for (int64_t hy = 2 * n[1] - 2 + lattice; hy <= 2 * n[1] + 1 + lattice; ++hy) {
                if (hy < 0 || hy >= halves[1]) continue;
                for (int64_t hx = 2 * n[0] - 2 + lattice; hx <= 2 * n[0] + 1 + lattice; ++hx) {
                    if (hx < 0 || hx >= halves[0]) continue;
                    const int64_t h[3] = {hx, hy, hz};
                    const uint32_t key = env * cells_per_env + HalfCellLocal(h, dims_x, dims_y);
                    const uint32_t cell_begin = cell_start[key];
                    if (cell_begin == ~0u) continue;
                    const uint32_t cell_slot =
                        CellTransferSlot(key, cell_begin, particle_count, total_cells);
                    const uint32_t offset = lattice * kLatticeNodes + static_cast<uint32_t>(
                        (n[0] - nk::MpmLatticeBase(hx, lattice)) +
                        2 * (n[1] - nk::MpmLatticeBase(hy, lattice)) +
                        4 * (n[2] - nk::MpmLatticeBase(hz, lattice)));
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

// Stress row blocks occupy the tail of each environment's row span. A cell's rows are written
// together, so one thread clears a cell and skips a cell whose head row is already clear.
__global__ void MpmClearStressRowsKernel(MpmParams p, DataView data) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.env_count * p.stress_cells_per_env) return;
    const uint32_t env = i / p.stress_cells_per_env;
    const uint32_t head = env * p.rows_per_env + p.rows_per_env -
        (p.stress_cells_per_env - i % p.stress_cells_per_env) * nk::kMpmStressRowsPerCell;
    auto* const urows = reinterpret_cast<nk::NkRow*>(data.urows);
    if (urows[head].flags == 0u) return;
    for (uint32_t slot = head; slot < head + nk::kMpmStressRowsPerCell; ++slot) {
        urows[slot] = nk::NkRow{};
        data.lambda[slot] = 0.0f;
        data.row_cj_link[slot] = data.row_cj_link_b[slot] = ~0u;
        data.row_penetration[slot] = data.row_damping[slot] = 0.0f;
    }
}

__device__ __forceinline__ uint32_t LowerBound(const uint32_t* values, uint32_t count,
                                               uint32_t value) {
    uint32_t begin = 0u;
    while (begin < count) {
        const uint32_t middle = begin + (count - begin) / 2u;
        if (values[middle] < value) begin = middle + 1u;
        else count = middle;
    }
    return begin;
}

// One warp per occupied cell merges its half-cell gradients g_n into a stress block: the head has
// R = V / (h^2 bulk), lambda = h p; deviator k has R = V / (2h (h shear + visc)), lambda = -h s_k.
__global__ void MpmEmitStressRowsKernel(
    MpmParams p, DataView data, uint32_t particle_count, uint32_t total_cells,
    uint32_t cells_per_env, const uint32_t* __restrict__ cell_start,
    const uint32_t* __restrict__ stress_cells, const uint32_t* __restrict__ stress_count,
    const m::Vec3* __restrict__ cell_gradients, const MpmCellVolume* __restrict__ cell_volumes,
    uint32_t endpoint_base, uint32_t term_base) {
    const uint32_t lane = threadIdx.x % kCudaWarpThreads;
    const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) / kCudaWarpThreads;
    const uint32_t warps = gridDim.x * blockDim.x / kCudaWarpThreads;
    const uint32_t lattice_nodes = p.nodes_per_env / nk::kMpmLattices;
    const uint32_t count = *stress_count;
    for (uint32_t slot = warp; slot < count; slot += warps) {
        const uint32_t cell = stress_cells[slot];
        const uint32_t env = cell / lattice_nodes;
        const uint32_t local = cell % lattice_nodes;
        const uint32_t ordinal = slot - LowerBound(stress_cells, count, env * lattice_nodes);
        if (ordinal >= p.stress_cells_per_env) {
            if (lane == 0u) atomicOr(&data.env_status[env], kEnvStatusInvalidEndpoint);
            continue;
        }
        const int64_t base[3] = {local % p.grid_dims[0], (local / p.grid_dims[0]) % p.grid_dims[1],
                                 local / (p.grid_dims[0] * p.grid_dims[1])};
        const uint32_t first = env * p.point_endpoint_terms_per_env + term_base + ordinal * kCellNodes;
        uint32_t terms = 0u;
        for (uint32_t pass = 0u; pass * kCudaWarpThreads < kCellNodes; ++pass) {
            const uint32_t j = pass * kCudaWarpThreads + lane;
            const uint32_t lattice = j < kLatticeNodes ? 0u : 1u;
            const uint32_t k = lattice == 0u ? j : j - kLatticeNodes;
            const uint32_t width = lattice == 0u ? kStencilWidth : kCellLatticeWidth;
            const int64_t node[3] = {base[0] - lattice + k % width, base[1] - lattice + (k / width) % width,
                                     base[2] - lattice + k / (width * width)};
            m::Vec3 gradient = m::Vec3::Zero();
            bool covered = false;
            for (uint32_t sub = 0u; j < kCellNodes && sub < nk::kMpmHalfCellsPerLatticeNode; ++sub) {
                const uint32_t key = env * cells_per_env + local * nk::kMpmHalfCellsPerLatticeNode + sub;
                const uint32_t begin = cell_start[key];
                if (begin == ~0u) continue;
                uint32_t corner = 0u;
                bool inside = true;
                for (uint32_t axis = 0u; axis < 3u; ++axis) {
                    const int64_t h = 2 * base[axis] + ((sub >> axis) & 1u);
                    const int64_t offset = node[axis] - nk::MpmLatticeBase(h, lattice);
                    inside = inside && offset >= 0 && offset < kStencilWidth;
                    corner += static_cast<uint32_t>(offset) << axis;
                }
                if (!inside) continue;
                const uint32_t transfer = CellTransferSlot(key, begin, particle_count, total_cells);
                const m::Vec3 g = cell_gradients[static_cast<size_t>(transfer) * kStencilNodes +
                                                 lattice * kLatticeNodes + corner];
                gradient.x = __fadd_rn(gradient.x, g.x);
                gradient.y = __fadd_rn(gradient.y, g.y);
                gradient.z = __fadd_rn(gradient.z, g.z);
                covered = true;
            }
            const int64_t index = covered ? nk::MpmNodeIndex(env, lattice, node[0], node[1], node[2],
                                                             p.grid_dims, p.nodes_per_env) : -1;
            const uint32_t emit = __ballot_sync(kFullWarpMask, index >= 0);
            if (index >= 0) {
                nk::PointEndpointTerm term;
                term.kind = nk::kNkSideGrid;
                term.index = static_cast<uint32_t>(index);
                term.column[0] = {gradient.x, 0.0f, 0.0f};
                term.column[1] = {gradient.y, 0.0f, 0.0f};
                term.column[2] = {gradient.z, 0.0f, 0.0f};
                data.point_endpoint_terms[first + terms + __popc(emit & ((1u << lane) - 1u))] = term;
            }
            terms += __popc(emit);
        }
        if (lane != 0u) continue;
        MpmCellVolume volume = EmptyCellVolume();
        for (uint32_t sub = 0u; sub < nk::kMpmHalfCellsPerLatticeNode; ++sub) {
            const uint32_t key = env * cells_per_env + local * nk::kMpmHalfCellsPerLatticeNode + sub;
            const uint32_t begin = cell_start[key];
            if (begin == ~0u) continue;
            AddCellVolume(volume, cell_volumes[CellTransferSlot(key, begin, particle_count, total_cells)]);
        }
        if (terms == 0u || !(volume.volume > 0.0f)) continue;
        const uint32_t endpoint = env * p.point_endpoints_per_env + endpoint_base + ordinal;
        data.point_endpoint_ranges[endpoint] = {first, terms};
        const float h = p.dt;
        const float inverse_volume = 1.0f / volume.volume;
        const float pressure = volume.pressure * inverse_volume;
        const float bulk = volume.bulk * inverse_volume;
        const uint32_t row_slot = env * p.rows_per_env + p.rows_per_env -
            p.stress_cells_per_env * nk::kMpmStressRowsPerCell + ordinal * nk::kMpmStressRowsPerCell;
        nk::NkRow row;
        row.flags = nk::nk_row_flags::kActive | nk::nk_row_flags::kVelocityOnly |
                    nk::nk_row_flags::kMaterialBlock;
        row.group_first = row_slot;
        row.group_normal_count = 1u;
        row.env = env;
        row.lower = volume.tension < FLT_MAX ? -h * volume.tension : -FLT_MAX;
        row.upper = FLT_MAX;
        // The deviator impulse norm stays within mu * lambda + friction_secondary.
        row.mu = volume.cone_slope;
        row.friction_secondary = volume.cone_offset < FLT_MAX ? h * volume.cone_offset : FLT_MAX;
        const float compliance = volume.volume / (h * h * bulk);
        if (bulk > 0.0f && isfinite(compliance)) {
            row.compliance_alpha = compliance;
            row.rhs = compliance * pressure;
        } else {
            // Without a volumetric stiffness the row holds the transferred pressure.
            row.lower = row.upper = fmaxf(h * pressure, row.lower);
        }
        row.a.kind = nk::kNkSidePointEndpoint;
        row.a.index = endpoint;
        row.a.jlin = {1.0f, 0.0f, 0.0f};
        auto* const urows = reinterpret_cast<nk::NkRow*>(data.urows);
        urows[row_slot] = row;
        const float head_impulse = fminf(fmaxf(h * pressure, row.lower), row.upper);
        data.lambda[row_slot] = head_impulse;
        // Rows of infinite compliance carry no deviator.
        const float rate = 2.0f * h * (h * volume.shear + volume.viscosity) * inverse_volume;
        const float deviator_compliance = rate > 0.0f ? volume.volume / (2.0f * h *
            (h * volume.shear + volume.viscosity)) : FLT_MAX;
        float impulse[nk::kMpmDeviatorComponents];
        float norm = 0.0f;
        for (uint32_t k = 0u; k < nk::kMpmDeviatorComponents; ++k) {
            impulse[k] = deviator_compliance < FLT_MAX ? -h * volume.deviator[k] * inverse_volume : 0.0f;
            norm += impulse[k] * impulse[k];
        }
        norm = sqrtf(norm);
        const float bound = row.friction_secondary < FLT_MAX
            ? fmaxf(fmaf(row.mu, head_impulse, row.friction_secondary), 0.0f) : FLT_MAX;
        const float scale = norm > bound ? bound / norm : 1.0f;
        for (uint32_t k = 0u; k < nk::kMpmDeviatorComponents; ++k) {
            nk::NkRow member;
            member.group_first = row_slot;
            member.group_normal_count = 1u;
            member.env = env;
            member.lower = -FLT_MAX;
            member.upper = FLT_MAX;
            member.compliance_alpha = deviator_compliance;
            member.rhs = deviator_compliance < FLT_MAX
                ? -deviator_compliance * volume.deviator[k] * inverse_volume : 0.0f;
            member.a = row.a;
            urows[row_slot + 1u + k] = member;
            data.lambda[row_slot + 1u + k] = impulse[k] * scale;
        }
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
    const nk::MpmCompactAxis axes[3] = {nk::MpmCompactWeights((xp.x - origin.x) * inv_dx),
                                        nk::MpmCompactWeights((xp.y - origin.y) * inv_dx),
                                        nk::MpmCompactWeights((xp.z - origin.z) * inv_dx)};
    m::Vec3 vp = m::Vec3::Zero();
    m::Vec3 pseudo = m::Vec3::Zero();
    float B[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    const uint32_t dims[3] = {dims_x, dims_y, dims_z};
    for (uint32_t lattice = 0u; lattice < nk::kMpmLattices; ++lattice) {
        for (uint32_t c = 0u; c < kStencilWidth; ++c) {
            for (uint32_t b = 0u; b < kStencilWidth; ++b) {
                for (uint32_t a = 0u; a < kStencilWidth; ++a) {
                    const int64_t id = nk::MpmNodeIndex(env, lattice, axes[0].base[lattice] + a,
                        axes[1].base[lattice] + b, axes[2].base[lattice] + c, dims, nodes_per_env);
                    if (id < 0) continue;
                    const float w = nk::kMpmLatticeShare * axes[0].w[lattice][a] *
                                    axes[1].w[lattice][b] * axes[2].w[lattice][c];
                    const m::Vec3 vi = grid_velocity[static_cast<size_t>(id)];
                    const m::Vec3 correction = grid_pseudo_velocity[static_cast<size_t>(id)];
                    pseudo.x = __fadd_rn(pseudo.x, w * correction.x);
                    pseudo.y = __fadd_rn(pseudo.y, w * correction.y);
                    pseudo.z = __fadd_rn(pseudo.z, w * correction.z);
                    vp.x = __fadd_rn(vp.x, w * vi.x);
                    vp.y = __fadd_rn(vp.y, w * vi.y);
                    vp.z = __fadd_rn(vp.z, w * vi.z);
                    const m::Vec3 d{axes[0].offset[lattice][a] * dx, axes[1].offset[lattice][b] * dx,
                                    axes[2].offset[lattice][c] * dx};
                    B[0] = __fadd_rn(B[0], w * vi.x * d.x); B[1] = __fadd_rn(B[1], w * vi.x * d.y); B[2] = __fadd_rn(B[2], w * vi.x * d.z);
                    B[3] = __fadd_rn(B[3], w * vi.y * d.x); B[4] = __fadd_rn(B[4], w * vi.y * d.y); B[5] = __fadd_rn(B[5], w * vi.y * d.z);
                    B[6] = __fadd_rn(B[6], w * vi.z * d.x); B[7] = __fadd_rn(B[7], w * vi.z * d.y); B[8] = __fadd_rn(B[8], w * vi.z * d.z);
                }
            }
        }
    }
    // C = B D^-1 with the full compact-kernel inertia D.
    float Dinv[6];
    if (!nk::MpmApicInverse(axes, inv_dx, Dinv)) for (float& entry : Dinv) entry = 0.0f;
    float* Cd = part_C + static_cast<size_t>(p) * 9u;
    for (int32_t r = 0; r < 3; ++r) {
        const float* b = B + 3 * r;
        Cd[3 * r + 0] = b[0] * Dinv[0] + b[1] * Dinv[3] + b[2] * Dinv[4];
        Cd[3 * r + 1] = b[0] * Dinv[3] + b[1] * Dinv[1] + b[2] * Dinv[5];
        Cd[3 * r + 2] = b[0] * Dinv[4] + b[1] * Dinv[5] + b[2] * Dinv[2];
    }
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
    nk::MpmMaterial material;
    if (part_mat != nullptr && material_table != nullptr && part_mat[p] < material_count)
        material = material_table[part_mat[p]];
    float* plastic_f = part_plastic_F != nullptr ? part_plastic_F + static_cast<size_t>(p) * 9u : nullptr;
    float* plastic_strain = part_plastic != nullptr ? part_plastic + p : nullptr;
    const nk::material::MpmMaterialHistory history{F, plastic_f,
        plastic_strain != nullptr ? *plastic_strain : 0.0f};
    nk::material::MpmMaterialTrial trial;
    const auto status = nk::material::EvaluateMpmMaterialTrial<CudaMpmArithmetic>(
        material, history, C, dt, trial);
    if (status != nk::material::ConstitutiveStatus::Ok ||
        !nk::material::CommitMpmMaterialTrial(trial, F, plastic_f, plastic_strain)) {
        const uint32_t flag = material.model_kind == 3.0f &&
            status == nk::material::ConstitutiveStatus::SingularDeformation
            ? kEnvStatusMpmGridEscape : kEnvStatusConstitutiveFailure;
        if (env_status != nullptr) atomicOr(&env_status[p / particles_per_env], flag);
    }
}

// Stress rows see only a cell's mean volume change, so each particle takes the cell's mean
// V0-weighted J; its deviatoric deformation is kept.
__global__ void MpmCellVolumeAverageKernel(
    uint32_t cells_per_env, const uint32_t* __restrict__ cell_start,
    const uint32_t* __restrict__ cell_end, const uint32_t* __restrict__ sorted_idx,
    const uint32_t* __restrict__ stress_cells, const uint32_t* __restrict__ stress_count,
    uint32_t lattice_nodes, const float* __restrict__ volume, float* __restrict__ part_F) {
    const uint32_t lane = threadIdx.x % kCudaWarpThreads;
    const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) / kCudaWarpThreads;
    const uint32_t warps = gridDim.x * blockDim.x / kCudaWarpThreads;
    const uint32_t count = *stress_count;
    for (uint32_t slot = warp; slot < count; slot += warps) {
        const uint32_t cell = stress_cells[slot];
        const uint32_t first_key = (cell / lattice_nodes) * cells_per_env +
                                   (cell % lattice_nodes) * nk::kMpmHalfCellsPerLatticeNode;
        uint32_t begin = ~0u, end = 0u;
        for (uint32_t sub = 0u; sub < nk::kMpmHalfCellsPerLatticeNode; ++sub) {
            if (cell_start[first_key + sub] == ~0u) continue;
            begin = min(begin, cell_start[first_key + sub]);
            end = max(end, cell_end[first_key + sub]);
        }
        if (begin >= end) continue;
        float weighted = 0.0f, total = 0.0f;
        for (uint32_t s = begin + lane; s < end; s += kCudaWarpThreads) {
            const uint32_t p = sorted_idx[s];
            const float J = nk::material::mpm_detail::Mat3Det(part_F + static_cast<size_t>(p) * 9u);
            if (!(J > 0.0f) || !(volume[p] > 0.0f)) continue;
            weighted += volume[p] * J;
            total += volume[p];
        }
        for (uint32_t offset = kCudaWarpThreads / 2u; offset > 0u; offset /= 2u) {
            weighted += __shfl_xor_sync(kFullWarpMask, weighted, offset);
            total += __shfl_xor_sync(kFullWarpMask, total, offset);
        }
        if (!(total > 0.0f)) continue;
        const float mean = weighted / total;
        for (uint32_t s = begin + lane; s < end; s += kCudaWarpThreads) {
            float* F = part_F + static_cast<size_t>(sorted_idx[s]) * 9u;
            const float J = nk::material::mpm_detail::Mat3Det(F);
            if (!(J > 0.0f)) continue;
            const float scale = cbrtf(mean / J);
            for (uint32_t i = 0u; i < 9u; ++i) F[i] *= scale;
        }
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
    uint32_t* occupied_cells = nullptr;
    uint32_t* occupied_count = nullptr;
    uint32_t* stress_cells = nullptr;
    uint32_t* stress_count = nullptr;
    m::Vec3* cell_gradients = nullptr;
    MpmCellVolume* cell_volumes = nullptr;
    MpmTransferInput* transfer_input = nullptr;
    MpmCellTransfer* cell_transfers = nullptr;
    uint32_t* contact_hit_mask = nullptr;
    float* contact_reach = nullptr;
    mpm_contact::BodyHitCache body_hits;
    uint32_t* contact_targets = nullptr;
    uint32_t* reaction_heads = nullptr;
    uint32_t* reaction_next = nullptr;
    mpm_contact::ReactionContribution* reaction_contributions = nullptr;
    size_t sort_temp_bytes = 0u;
};
MpmScratch PartitionScratch(void* base, uint32_t particle_count, uint32_t node_count,
                            uint64_t collidables_per_env, uint64_t contact_count,
                            uint64_t reaction_target_count, uint64_t stress_cell_count) {
    const auto& layout = ScratchLayout(particle_count, node_count, collidables_per_env,
                                       contact_count, reaction_target_count, stress_cell_count);
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
    scratch.occupied_cells = reinterpret_cast<uint32_t*>(bytes + layout.occupied_cells_off);
    scratch.occupied_count = reinterpret_cast<uint32_t*>(bytes + layout.occupied_count_off);
    if (stress_cell_count > 0u) {
        scratch.stress_cells = reinterpret_cast<uint32_t*>(bytes + layout.stress_cells_off);
        scratch.stress_count = reinterpret_cast<uint32_t*>(bytes + layout.stress_count_off);
        scratch.cell_gradients = reinterpret_cast<m::Vec3*>(bytes + layout.cell_gradients_off);
        scratch.cell_volumes = reinterpret_cast<MpmCellVolume*>(bytes + layout.cell_volumes_off);
    }
    scratch.transfer_input = reinterpret_cast<MpmTransferInput*>(bytes + layout.transfer_input_off);
    scratch.cell_transfers = reinterpret_cast<MpmCellTransfer*>(bytes + layout.cell_transfer_off);
    scratch.contact_hit_mask = reinterpret_cast<uint32_t*>(bytes + layout.contact_hit_mask_off);
    scratch.contact_reach = reinterpret_cast<float*>(bytes + layout.contact_reach_off);
    scratch.body_hits.records =
        reinterpret_cast<mpm_contact::BodyHit*>(bytes + layout.body_hit_records_off);
    scratch.body_hits.first = reinterpret_cast<uint32_t*>(bytes + layout.body_hit_first_off);
    scratch.body_hits.count = reinterpret_cast<uint32_t*>(bytes + layout.body_hit_count_off);
    scratch.body_hits.capacity = particle_count;
    scratch.contact_targets = reinterpret_cast<uint32_t*>(bytes + layout.contact_targets_off);
    scratch.reaction_heads = reinterpret_cast<uint32_t*>(bytes + layout.reaction_heads_off);
    scratch.reaction_next = reinterpret_cast<uint32_t*>(bytes + layout.reaction_next_off);
    scratch.reaction_contributions = reinterpret_cast<mpm_contact::ReactionContribution*>(
        bytes + layout.reaction_contributions_off);
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
        const uint32_t total_cells = cpe * p.env_count;
        launch(MpmStage::GridPrepare, MpmGridPrepareKernel, nblocks,
               total_nodes, total_cells, data.grid_mass, data.grid_momentum, data.grid_velocity,
               data.grid_pseudo_vel, data.grid_inv_mass, scratch.active_flags,
               scratch.cell_start, scratch.node_ids);
        launch(MpmStage::CellKeys, MpmCellKeysKernel, pblocks,
               mpm_count, data.particle_pos, Ppe, mpm_pe, inv_dx, origin,
               p.grid_dims[0], p.grid_dims[1], p.grid_dims[2], cpe,
               data.mpm_grid_cell_key, data.mpm_grid_part_idx, data.env_status);
        if (error != cudaSuccess) return error;
        size_t temp_bytes = scratch.sort_temp_bytes;
        profiler.Start(MpmStage::RadixSort, stream);
        error = cub::DeviceRadixSort::SortPairs(
            scratch.sort_temp, temp_bytes, data.mpm_grid_cell_key, scratch.keys_out,
            data.mpm_grid_part_idx, scratch.idx_out, static_cast<int>(mpm_count), 0,
            RadixBitsInclusive(total_cells - 1u), stream);
        if (error == cudaSuccess) {
            temp_bytes = scratch.sort_temp_bytes;
            error = cub::DeviceSelect::Unique(scratch.sort_temp, temp_bytes, scratch.keys_out,
                scratch.occupied_cells, scratch.occupied_count, static_cast<int>(mpm_count), stream);
        }
        if (error == cudaSuccess && scratch.stress_cells != nullptr) {
            temp_bytes = scratch.sort_temp_bytes;
            error = cub::DeviceSelect::Unique(scratch.sort_temp, temp_bytes,
                thrust::make_transform_iterator(static_cast<const uint32_t*>(scratch.keys_out),
                                                MpmStressCellKey{}),
                scratch.stress_cells, scratch.stress_count, static_cast<int>(mpm_count), stream);
        }
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
               origin, inv_dx, p.grid_dims[0], p.grid_dims[1], p.grid_dims[2],
               scratch.stress_cells != nullptr ? 1u : 0u, scratch.transfer_input);
        if (error != cudaSuccess) return error;
        launch(MpmStage::P2GCells, MpmP2GCellsKernel, cell_blocks,
               mpm_count, mpm_pe, total_cells, cpe, scratch.occupied_cells, scratch.occupied_count,
               scratch.transfer_input, scratch.cell_start, scratch.cell_end,
               p.grid_dims[0], p.grid_dims[1], p.grid_dims[2], inv_dx, p.dx, dt_sub, origin,
               scratch.cell_transfers, scratch.cell_gradients, scratch.cell_volumes);
        const m::Vec3 g{p.gravity[0], p.gravity[1], p.gravity[2]};
        launch(MpmStage::GridFinalize, MpmGridFinalizeKernel, finalize_blocks,
               total_nodes, scratch.active_nodes, scratch.active_count,
               scratch.cell_transfers, scratch.cell_start, mpm_count, total_cells,
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
        if (scratch.stress_cells != nullptr) {
            const uint32_t stress_cells = p.env_count * p.stress_cells_per_env;
            launch(MpmStage::StressRows, MpmClearStressRowsKernel,
                   (stress_cells + kBlockSize - 1u) / kBlockSize, p, data);
            const uint32_t surfaces = p.particle_surfaces_per_env > 0u ? p.contact_capacity : 0u;
            launch(MpmStage::StressRows, MpmEmitStressRowsKernel,
                   (std::min(mpm_count, stress_cells) + kCellGroups - 1u) / kCellGroups,
                   p, data, mpm_count, cpe * p.env_count, cpe, scratch.cell_start,
                   static_cast<const uint32_t*>(scratch.stress_cells),
                   static_cast<const uint32_t*>(scratch.stress_count),
                   static_cast<const m::Vec3*>(scratch.cell_gradients),
                   static_cast<const MpmCellVolume*>(scratch.cell_volumes),
                   static_cast<uint32_t>(nk::MpmPointEndpointCount(Ppe, surfaces, 0u)),
                   static_cast<uint32_t>(nk::MpmPointEndpointTermCount(Ppe, surfaces, 0u)));
            if (error != cudaSuccess) return error;
        }
        launch(MpmStage::ContactCount, mpm_contact::PrepareSamples, pblocks,
               p, data, mpm_pe, scratch.contact_hit_mask, scratch.contact_reach,
               scratch.body_hits.count);
        if (error != cudaSuccess) return error;
        const bool query_bodies = p.dynamic_body_bc != 0u && p.bite_disable_dynamic_bc == 0u &&
            p.bodies_per_env > 0u && pblocks > 0u;
        constexpr uint32_t kMaxGridY = 65535u;
        const dim3 body_grid(pblocks, std::min(p.bodies_per_env, kMaxGridY));
        if (query_bodies) {
            profiler.Start(MpmStage::ContactCount, stream);
            LaunchCuda(mpm_contact::QueryBodies, body_grid,
                       dim3(kBlockSize), 0u, stream, p, model, data, surfaces, mpm_pe,
                       scratch.contact_hit_mask, static_cast<const float*>(scratch.contact_reach),
                       scratch.body_hits);
            error = cudaPeekAtLastError();
            profiler.Stop(MpmStage::ContactCount, stream);
            if (error != cudaSuccess) return error;
        }
        launch(MpmStage::ContactCount, mpm_contact::CountContacts, pblocks,
               p, model, data, mpm_pe, scratch.contact_hit_mask,
               static_cast<const float*>(scratch.contact_reach));
        if (error != cudaSuccess) return error;
        size_t temp_bytes = scratch.sort_temp_bytes;
        profiler.Start(MpmStage::ContactScan, stream);
        error = cub::DeviceScan::ExclusiveSum(scratch.sort_temp, temp_bytes,
            data.grid_contact_count, data.grid_contact_offset, static_cast<int>(mpm_count), stream);
        profiler.Stop(MpmStage::ContactScan, stream);
        if (error != cudaSuccess) return error;
        launch(MpmStage::ContactEmit, mpm_contact::CountDiagnostics,
               (p.env_count + kBlockSize - 1u) / kBlockSize, p, data, mpm_pe);
        launch(MpmStage::ContactEmit, mpm_contact::EmitContacts, pblocks,
               p, model, data, mpm_pe, scratch.contact_hit_mask,
               static_cast<const float*>(scratch.contact_reach));
        if (query_bodies && error == cudaSuccess) {
            profiler.Start(MpmStage::ContactEmit, stream);
            LaunchCuda(mpm_contact::EmitBodyContacts, body_grid,
                       dim3(kBlockSize), 0u, stream, p, model, data, surfaces, mpm_pe,
                       static_cast<const uint32_t*>(scratch.contact_hit_mask),
                       static_cast<const float*>(scratch.contact_reach), scratch.body_hits);
            error = cudaPeekAtLastError();
            profiler.Stop(MpmStage::ContactEmit, stream);
        }
    }
    if constexpr (operation == MpmOperation::Commit) {
        uint32_t classify_blocks = 0u;
        if (error == cudaSuccess)
            error = ResidentGridSize(mpm_contact::ClassifyReactions, kBlockSize, 0u,
                (p.env_count * p.contact_capacity + kBlockSize - 1u) / kBlockSize, &classify_blocks);
        launch(MpmStage::ReactionReadout, mpm_contact::ClassifyReactions, classify_blocks,
               p, model, data, scratch.contact_targets, scratch.reaction_contributions);
        launch(MpmStage::ReactionReadout, mpm_contact::IndexReactions<kBlockSize>,
               p.env_count, p, data, scratch.contact_targets, scratch.reaction_heads,
               scratch.reaction_next);
        launch(MpmStage::ReactionReadout, mpm_contact::ReadReactions<kBlockSize>,
               p.env_count * (p.bodies_per_env + nk::kMpmBoundaryCount), p, model, data,
               scratch.reaction_heads, scratch.reaction_next, scratch.reaction_contributions);
        launch(MpmStage::G2P, MpmG2PGatherKernel, pblocks,
               mpm_count, Ppe, mpm_pe, p.nodes_per_env, p.grid_dims[0], p.grid_dims[1],
               p.grid_dims[2], inv_dx, p.dx, dt_sub, origin, data.particle_inv_mass,
               data.grid_velocity, data.grid_pseudo_vel, data.particle_pos, data.particle_vel, data.particle_C);
        launch(MpmStage::UpdateF, MpmUpdateFKernel, pblocks, mpm_count,
               dt_sub, data.particle_C, data.particle_material_id,
               reinterpret_cast<const nk::MpmMaterial*>(data.mpm_material_table),
               p.material_count, Ppe, mpm_pe, data.particle_F, data.particle_plastic_F,
               data.particle_plastic, data.env_status);
        if (scratch.stress_cells != nullptr)
            launch(MpmStage::UpdateF, MpmCellVolumeAverageKernel,
                   (std::min(mpm_count, p.env_count * p.stress_cells_per_env) + kCellGroups - 1u) /
                       kCellGroups,
                   cpe, static_cast<const uint32_t*>(scratch.cell_start),
                   static_cast<const uint32_t*>(scratch.cell_end),
                   static_cast<const uint32_t*>(scratch.idx_out),
                   static_cast<const uint32_t*>(scratch.stress_cells),
                   static_cast<const uint32_t*>(scratch.stress_count),
                   p.nodes_per_env / nk::kMpmLattices, data.particle_vol0, data.particle_F);
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
        (uint64_t{p->bodies_per_env} + nk::kMpmBoundaryCount) * p->env_count > INT_MAX ||
        !data.ucontact_law || !data.ucontact_friction || !data.grid_contact_attempted ||
        !data.grid_contact_retained || !data.grid_contact_peak || !data.grid_contact_overflow ||
        !data.mpm_boundary_impulse || !data.mpm_boundary_moment)
        return Status::InvalidArgument;
    const uint64_t total_nodes64 =
        static_cast<uint64_t>(p->nodes_per_env) * p->env_count;
    const uint64_t grid_xy = static_cast<uint64_t>(p->grid_dims[0]) * p->grid_dims[1];
    const uint64_t lattice_nodes = grid_xy * p->grid_dims[2];
    const uint64_t cells_per_env = lattice_nodes * nk::kMpmHalfCellsPerLatticeNode;
    if (grid_xy > p->nodes_per_env || lattice_nodes * nk::kMpmLattices != p->nodes_per_env ||
        total_nodes64 > static_cast<uint64_t>(INT_MAX) ||
        cells_per_env * p->env_count > static_cast<uint64_t>(INT_MAX) ||
        static_cast<uint64_t>(p->bodies_per_env) * p->env_count > static_cast<uint64_t>(INT_MAX))
        return Status::InvalidArgument;
    const uint32_t Np = p->particle_count;
    const uint32_t Ppe = p->particles_per_env == 0u ? Np : p->particles_per_env;
    if (p->mpm_particles_per_env > Ppe) return Status::InvalidArgument;
    const uint32_t mpm_pe = p->mpm_particles_per_env == 0u ? Ppe : p->mpm_particles_per_env;
    const uint32_t surface_contacts = p->particle_surfaces_per_env > 0u ? p->contact_capacity : 0u;
    if (p->point_endpoints_per_env <
            nk::MpmPointEndpointCount(Ppe, surface_contacts, p->stress_cells_per_env) ||
        p->point_endpoint_terms_per_env <
            nk::MpmPointEndpointTermCount(Ppe, surface_contacts, p->stress_cells_per_env) ||
        uint64_t{p->stress_cells_per_env} * nk::kMpmStressRowsPerCell > p->rows_per_env ||
        uint64_t{p->stress_cells_per_env} * p->env_count > INT_MAX ||
        (p->stress_cells_per_env > 0u && (!data.urows || !data.lambda || !data.row_cj_link ||
            !data.row_cj_link_b || !data.row_penetration || !data.row_damping)) ||
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
        uint64_t{p->bodies_per_env} + p->particle_surfaces_per_env,
        uint64_t{p->contact_capacity} * p->env_count,
        (uint64_t{p->bodies_per_env} + nk::kMpmBoundaryCount) * p->env_count,
        uint64_t{p->stress_cells_per_env} * p->env_count);
    uint32_t finalize_blocks = 0u, cell_blocks = 0u;
    if constexpr (operation == MpmOperation::Predict) {
        if (ResidentGridSize(MpmGridFinalizeKernel, kBlockSize, 0u,
                             (total_nodes + kBlockSize - 1u) / kBlockSize,
                             &finalize_blocks) != cudaSuccess ||
            ResidentGridSize(MpmP2GCellsKernel, kBlockSize, 0u,
                             (std::min<uint64_t>(mpm_count, uint64_t{cpe} * p->env_count) +
                              kCellGroups - 1u) / kCellGroups,
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
                             uint64_t collidables_per_env, uint64_t contact_count,
                             uint64_t reaction_target_count, uint64_t stress_cell_count) {
    if (particle_count == 0u || node_count == 0u ||
        particle_count > static_cast<uint32_t>(INT_MAX) || node_count > static_cast<uint32_t>(INT_MAX) ||
        contact_count > INT_MAX || reaction_target_count > INT_MAX)
        return 0u;
    return ScratchLayout(particle_count, node_count, collidables_per_env, contact_count,
                         reaction_target_count, stress_cell_count).total;
}

void RegisterNkMpmOps() {
    SetCudaOp(NkOp::MpmPredict, &OpMpmStage<MpmOperation::Predict>);
    SetCudaOp(NkOp::MpmExchange, &OpMpmStage<MpmOperation::Exchange>);
    SetCudaOp(NkOp::MpmCommit, &OpMpmStage<MpmOperation::Commit>);
}

}  // namespace nuka::phi
