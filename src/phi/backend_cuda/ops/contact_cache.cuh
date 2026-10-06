#pragma once

#include <cub/device/device_scan.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "core/checked_size.hpp"
#include "nk/contact/contact_identity.hpp"
#include "nk/model/generated/views.hpp"
#include "nk/solve/nk_row.hpp"
#include "phi/backend.hpp"

namespace nuka::phi::contact_cache {

struct Ranks { uint32_t free, old; };
struct AddRanks {
    __host__ __device__ Ranks operator()(Ranks a, Ranks b) const {
        return {a.free + b.free, a.old + b.old};
    }
};

struct KeySource {
    const uint32_t* counts;
    const uint64_t* current_pair;
    const uint64_t* current_feature;
    const uint64_t* current_material;
    const uint64_t* cache_pair;
    const uint64_t* cache_feature;
    const uint64_t* cache_material;
    uint32_t points, points_per_env;

    __device__ uint32_t Point(uint32_t id) const { return id < points ? id : id - points; }
    __host__ __device__ bool Valid(uint32_t id) const {
        return id < points ? (id % nk::kPairDrivenPtsPerSlot < counts[id / nk::kPairDrivenPtsPerSlot])
            : cache_pair[id - points] != 0u;
    }
    __device__ uint64_t Pair(uint32_t id) const {
        return id < points ? current_pair[id] : cache_pair[id - points];
    }
    __device__ uint64_t Feature(uint32_t id) const {
        return id < points ? current_feature[id] : cache_feature[id - points];
    }
    __device__ uint64_t Material(uint32_t id) const {
        return id < points ? current_material[id / nk::kPairDrivenPtsPerSlot] : cache_material[id - points];
    }
    __device__ uint32_t Bucket(uint32_t id) const {
        uint64_t hash = nk::ContactHashWord(1469598103934665603ull, Pair(id));
        hash = nk::ContactHashWord(hash, Feature(id));
        hash = nk::ContactHashWord(hash, Material(id));
        return static_cast<uint32_t>(hash) ^ static_cast<uint32_t>(hash >> 32u);
    }
    __device__ bool Same(uint32_t a, uint32_t b) const {
        return Valid(a) && Valid(b) && Point(a) / points_per_env == Point(b) / points_per_env &&
               Pair(a) == Pair(b) && Feature(a) == Feature(b) && Material(a) == Material(b);
    }
    __host__ __device__ uint32_t Source(uint32_t index) const {
        const uint32_t env = index / (points_per_env * 2u);
        const uint32_t local = index - env * points_per_env * 2u;
        return env * points_per_env + (local < points_per_env ? local : points + local - points_per_env);
    }
};

inline KeySource Keys(const DataView& data, uint32_t points, uint32_t points_per_env) {
    return {data.ucontact_count, data.ucontact_id_pair, data.ucontact_id_feature,
            data.contact_material, data.contact_cache_pair, data.contact_cache_feature,
            data.contact_cache_material, points, points_per_env};
}

struct ValidEntry {
    KeySource keys;
    __host__ __device__ uint32_t operator()(uint32_t index) const {
        return keys.Valid(keys.Source(index)) ? 1u : 0u;
    }
};
using ValidInput = thrust::transform_iterator<ValidEntry, thrust::counting_iterator<uint32_t>>;

struct RankEntry {
    const uint32_t* owner;
    const uint32_t* keep;
    __host__ __device__ Ranks operator()(uint32_t point) const {
        return {owner[point] == 0u ? 1u : 0u, keep[point]};
    }
};
using RankInput = thrust::transform_iterator<RankEntry, thrust::counting_iterator<uint32_t>>;

// An env's table has 2n+1 slots for its n entries, so linear probing always ends.
inline uint64_t TableStride(uint32_t entries_per_env) { return uint64_t{entries_per_env} * 2u + 1u; }

// Entry loops give each env a fixed share of blocks, so launch shapes depend only on capacities.
inline uint32_t EnvBlocks(uint32_t entries_per_env, uint32_t envs) {
    constexpr uint32_t block = 128u, target = 2048u;
    const uint32_t needed = (entries_per_env + block - 1u) / block;
    const uint32_t share = envs < target ? target / envs : 1u;
    return needed < share ? needed : share;
}

// Every cache point at or past an env's previous bound holds no entry.
__device__ inline uint32_t PreviousBound(const uint32_t* extent, uint32_t env, uint32_t points_per_env) {
    if (extent == nullptr || extent[2u * env] == 0u) return points_per_env;
    return extent[2u * env] - 1u < points_per_env ? extent[2u * env] - 1u : points_per_env;
}

// Scan prefixes, first-entry tables and ranks occupy the shared region at different times.
struct Layout {
    size_t slots_offset, shared_offset, warm_offset, match_offset;
    size_t begin_offset, end_offset, temp_offset;
    static size_t Align(size_t value) { return CheckedAlignUp(value, 256u); }
    static size_t TableBytes(uint32_t points, uint32_t envs) {
        return size_t{envs} * TableStride(points / envs * 2u) * sizeof(uint32_t);
    }
    Layout(uint32_t points, uint32_t envs)
        : slots_offset(Align(size_t{points} * 2u * sizeof(uint32_t))),
          shared_offset(Align(slots_offset + size_t{points} * 2u * sizeof(uint32_t))),
          warm_offset(Align(shared_offset + TableBytes(points, envs))),
          match_offset(Align(warm_offset + TableBytes(points, envs))),
          begin_offset(Align(match_offset + size_t{points} * sizeof(uint32_t))),
          end_offset(Align(begin_offset + size_t{envs} * sizeof(uint32_t))),
          temp_offset(Align(end_offset + size_t{envs} * sizeof(uint32_t))) {}
};

inline uint64_t ScratchBytes(uint32_t points, uint32_t envs) {
    if (points == 0u) return 0u;
    if (envs == 0u || points % envs != 0u ||
        uint64_t{points} * 2u > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("contact cache exceeds scan index range");
    size_t scan_bytes = 0u, compact_bytes = 0u;
    auto status = cub::DeviceScan::ExclusiveScan(nullptr, scan_bytes,
        RankInput(thrust::counting_iterator<uint32_t>(0u), RankEntry{}),
        static_cast<Ranks*>(nullptr), AddRanks{}, Ranks{}, static_cast<int>(points));
    if (status == cudaSuccess)
        status = cub::DeviceScan::ExclusiveSum(nullptr, compact_bytes,
            ValidInput(thrust::counting_iterator<uint32_t>(0u), ValidEntry{}),
            static_cast<uint32_t*>(nullptr), static_cast<int>(points * 2u));
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    if (compact_bytes > scan_bytes) scan_bytes = compact_bytes;
    return CheckedAdd(Layout(points, envs).temp_offset, scan_bytes);
}

struct Workspace {
    uint32_t* order;
    uint32_t* slots;
    uint32_t* prefix;
    uint32_t* first;
    uint32_t* warm;
    Ranks* ranks;
    uint32_t* matches;
    uint32_t* begins;
    uint32_t* ends;
    void* temp;
    size_t temp_bytes;
    Workspace(void* base, size_t bytes, uint32_t points, uint32_t envs) {
        const Layout layout(points, envs);
        auto* data = static_cast<uint8_t*>(base);
        order = reinterpret_cast<uint32_t*>(data);
        slots = reinterpret_cast<uint32_t*>(data + layout.slots_offset);
        prefix = reinterpret_cast<uint32_t*>(data + layout.shared_offset);
        first = prefix;
        warm = reinterpret_cast<uint32_t*>(data + layout.warm_offset);
        ranks = reinterpret_cast<Ranks*>(data + layout.shared_offset);
        matches = reinterpret_cast<uint32_t*>(data + layout.match_offset);
        begins = reinterpret_cast<uint32_t*>(data + layout.begin_offset);
        ends = reinterpret_cast<uint32_t*>(data + layout.end_offset);
        temp = data + layout.temp_offset;
        temp_bytes = bytes - layout.temp_offset;
    }
};

// Adjacent exclusive prefixes give each validity flag; only the final entry has no successor.
static __global__ void CompactSourcesKernel(KeySource keys, const uint32_t* prefix,
                                             uint32_t* order, uint32_t* begins, uint32_t* ends,
                                             uint32_t* owner, uint32_t* keep, uint32_t* extent) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = keys.points * 2u;
    if (i >= total) return;
    if (i < keys.points) {
        owner[i] = 0u;
        keep[i] = 0u;
    }
    const uint32_t stride = keys.points_per_env * 2u;
    const uint32_t env = i / stride, begin = env * stride;
    const uint32_t valid = i + 1u < total ? prefix[i + 1u] - prefix[i]
                                          : (keys.Valid(keys.Source(i)) ? 1u : 0u);
    if (i == begin + stride - 1u) {
        begins[env] = begin;
        ends[env] = begin + prefix[i] + valid - prefix[begin];
        if (extent != nullptr) extent[2u * env + 1u] = 0u;
    }
    if (valid != 0u) order[begin + prefix[i] - prefix[begin]] = keys.Source(i);
}

// Only the live part of each env's table is cleared and probed.
static __global__ void ClearTableKernel(const uint32_t* begins, const uint32_t* ends,
                                        uint32_t env_blocks, uint64_t stride,
                                        uint32_t* first, uint32_t* warm) {
    const uint32_t env = blockIdx.x / env_blocks;
    const uint32_t part = blockIdx.x - env * env_blocks;
    const uint64_t live = uint64_t{ends[env] - begins[env]} * 2u + 1u;
    for (uint64_t slot = uint64_t{part} * blockDim.x + threadIdx.x; slot < live;
         slot += uint64_t{env_blocks} * blockDim.x) {
        first[env * stride + slot] = ~0u;
        warm[env * stride + slot] = ~0u;
    }
}

// Hashes choose where probing starts and complete keys choose the slot; integer minima make the
// first entry and warm point of each key independent of thread order.
static __global__ void InsertEntriesKernel(KeySource keys, const uint32_t* order,
                                           const uint32_t* begins, const uint32_t* ends,
                                           const uint32_t* age, uint32_t decay_steps,
                                           uint32_t env_blocks, uint64_t stride,
                                           uint32_t* first, uint32_t* warm, uint32_t* slots) {
    const uint32_t env = blockIdx.x / env_blocks;
    const uint32_t part = blockIdx.x - env * env_blocks;
    const uint32_t begin = begins[env], end = ends[env];
    const uint64_t live = uint64_t{end - begin} * 2u + 1u;
    uint32_t* const table = first + env * stride;
    for (uint32_t entry = begin + part * blockDim.x + threadIdx.x; entry < end;
         entry += env_blocks * blockDim.x) {
        const uint32_t id = order[entry];
        uint64_t slot = (uint64_t{keys.Bucket(id)} * live) >> 32u;
        for (;;) {
            const uint32_t held = atomicCAS(table + slot, ~0u, entry);
            if (held == ~0u || keys.Same(order[held], id)) break;
            if (++slot == live) slot = 0u;
        }
        slots[entry] = static_cast<uint32_t>(slot);
        atomicMin(table + slot, entry);
        if (id >= keys.points && age[id - keys.points] < decay_steps)
            atomicMin(warm + env * stride + slot, id - keys.points);
    }
}

// Current entries precede cache entries in each env, so a cache head has no current twin.
static __global__ void ResolveEntriesKernel(KeySource keys, const uint32_t* order,
                                            const uint32_t* begins, const uint32_t* ends,
                                            const uint32_t* age, uint32_t decay_steps,
                                            uint32_t env_blocks, uint64_t stride,
                                            const uint32_t* first, const uint32_t* warm,
                                            const uint32_t* slots, uint32_t* matches,
                                            uint32_t* owner, uint32_t* keep, uint32_t* extent) {
    const uint32_t env = blockIdx.x / env_blocks;
    const uint32_t part = blockIdx.x - env * env_blocks;
    const uint32_t end = ends[env];
    uint32_t owned = 0u;
    for (uint32_t entry = begins[env] + part * blockDim.x + threadIdx.x; entry < end;
         entry += env_blocks * blockDim.x) {
        const uint32_t id = order[entry];
        const uint64_t slot = env * stride + slots[entry];
        const bool head = first[slot] == entry;
        if (id < keys.points) {
            matches[id] = head ? warm[slot] : ~0u;
            if (head) {
                owner[id] = 1u;
                owned = max(owned, id - env * keys.points_per_env + 1u);
            }
        } else if (head && decay_steps > 1u && age[id - keys.points] < decay_steps - 1u) {
            keep[id - keys.points] = 1u;
        }
    }
    owned = __reduce_max_sync(0xffffffffu, owned);
    if (extent != nullptr && owned != 0u && (threadIdx.x & 31u) == 0u) atomicMax(extent + 2u * env + 1u, owned);
}

inline cudaError_t BuildIndex(const DataView& data, uint32_t points,
                               uint32_t points_per_env, uint32_t decay_steps,
                               Workspace workspace, cudaStream_t stream) {
    constexpr uint32_t block = 128u;
    const uint32_t blocks = (points * 2u + block - 1u) / block;
    const uint32_t envs = points / points_per_env;
    const uint32_t env_blocks = EnvBlocks(points_per_env * 2u, envs);
    const uint64_t stride = TableStride(points_per_env * 2u);
    const auto keys = Keys(data, points, points_per_env);
    size_t temp_bytes = workspace.temp_bytes;
    const auto status = cub::DeviceScan::ExclusiveSum(workspace.temp, temp_bytes,
        ValidInput(thrust::counting_iterator<uint32_t>(0u), ValidEntry{keys}), workspace.prefix,
        static_cast<int>(points * 2u), stream);
    if (status != cudaSuccess) return status;
    CompactSourcesKernel<<<blocks, block, 0u, stream>>>(keys, workspace.prefix, workspace.order,
        workspace.begins, workspace.ends, data.contact_cache_current_owner,
        data.contact_cache_old_keep, data.contact_cache_extent);
    if (const auto native = cudaGetLastError(); native != cudaSuccess) return native;
    ClearTableKernel<<<envs * env_blocks, block, 0u, stream>>>(workspace.begins, workspace.ends,
        env_blocks, stride, workspace.first, workspace.warm);
    if (const auto native = cudaGetLastError(); native != cudaSuccess) return native;
    InsertEntriesKernel<<<envs * env_blocks, block, 0u, stream>>>(keys, workspace.order,
        workspace.begins, workspace.ends, data.contact_cache_age, decay_steps, env_blocks, stride,
        workspace.first, workspace.warm, workspace.slots);
    if (const auto native = cudaGetLastError(); native != cudaSuccess) return native;
    ResolveEntriesKernel<<<envs * env_blocks, block, 0u, stream>>>(keys, workspace.order,
        workspace.begins, workspace.ends, data.contact_cache_age, decay_steps, env_blocks, stride,
        workspace.first, workspace.warm, workspace.slots, workspace.matches,
        data.contact_cache_current_owner, data.contact_cache_old_keep, data.contact_cache_extent);
    return cudaGetLastError();
}

static __global__ void RetainedSourcesKernel(uint32_t points, uint32_t points_per_env,
                                              const uint32_t* keep, const Ranks* ranks,
                                              uint32_t* sources) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= points || keep[i] == 0u) return;
    const uint32_t begin = (i / points_per_env) * points_per_env;
    sources[begin + ranks[i].old - ranks[begin].old] = i;
}

// Owners and the retained entries that fill the first free points end before the bound kept for the
// next step; the rebuild also covers the points the previous step may have filled.
static __global__ void CacheExtentKernel(uint32_t envs, uint32_t points_per_env, const uint32_t* owner,
                                         const uint32_t* keep, const Ranks* ranks, uint32_t* extent) {
    const uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= envs) return;
    const uint32_t begin = env * points_per_env, last = begin + points_per_env - 1u;
    const uint32_t unowned = ranks[last].free + (owner[last] == 0u ? 1u : 0u) - ranks[begin].free;
    const uint32_t retained = ranks[last].old + keep[last] - ranks[begin].old;
    const uint32_t next = min(max(extent[2u * env + 1u], points_per_env - unowned + retained), points_per_env);
    const uint32_t previous = PreviousBound(extent, env, points_per_env);
    extent[2u * env] = next + 1u;
    extent[2u * env + 1u] = max(previous, next);
}

// Retained sources reuse the match array once prepare has read it.
inline cudaError_t BuildRanks(const DataView& data, uint32_t points,
                               uint32_t points_per_env, Workspace workspace, cudaStream_t stream) {
    constexpr uint32_t block = 128u;
    const RankInput input(thrust::counting_iterator<uint32_t>(0u),
                          RankEntry{data.contact_cache_current_owner, data.contact_cache_old_keep});
    const auto scanned = cub::DeviceScan::ExclusiveScan(workspace.temp, workspace.temp_bytes,
        input, workspace.ranks, AddRanks{}, Ranks{}, static_cast<int>(points), stream);
    if (scanned != cudaSuccess) return scanned;
    RetainedSourcesKernel<<<(points + block - 1u) / block, block, 0u, stream>>>(points,
        points_per_env, data.contact_cache_old_keep, workspace.ranks, workspace.matches);
    if (const auto native = cudaGetLastError(); native != cudaSuccess) return native;
    if (data.contact_cache_extent != nullptr) {
        const uint32_t envs = points / points_per_env;
        CacheExtentKernel<<<(envs + block - 1u) / block, block, 0u, stream>>>(envs, points_per_env,
            data.contact_cache_current_owner, data.contact_cache_old_keep, workspace.ranks,
            data.contact_cache_extent);
    }
    return cudaGetLastError();
}
}  // namespace nuka::phi::contact_cache
