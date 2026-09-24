#pragma once

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
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
    __device__ bool Valid(uint32_t id) const {
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
    __device__ uint32_t Source(uint32_t index) const {
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

struct Layout {
    size_t alternate_offset, keys_offset, alternate_keys_offset;
    size_t ranks_offset, match_offset, begin_offset, end_offset, temp_offset;
    static size_t Align(size_t value) { return CheckedAlignUp(value, 256u); }
    Layout(uint32_t points, uint32_t envs)
        : alternate_offset(Align(size_t{points} * 2u * sizeof(uint32_t))),
          keys_offset(Align(alternate_offset + size_t{points} * 2u * sizeof(uint32_t))),
          alternate_keys_offset(Align(keys_offset + size_t{points} * 2u * sizeof(uint64_t))),
          ranks_offset(Align(alternate_keys_offset + size_t{points} * 2u * sizeof(uint64_t))),
          match_offset(Align(ranks_offset + size_t{points} * sizeof(Ranks))),
          begin_offset(Align(match_offset + size_t{points} * sizeof(uint32_t))),
          end_offset(Align(begin_offset + size_t{envs} * sizeof(uint32_t))),
          temp_offset(Align(end_offset + size_t{envs} * sizeof(uint32_t))) {}
};

// Keys hold env, an invalid-tail bit, then the bucket; the index range keeps env below 2^30.
inline int SortBits(uint32_t envs) {
    int bits = 33;
    for (uint32_t value = envs - 1u; value != 0u; value >>= 1u) ++bits;
    return bits;
}

inline uint64_t ScratchBytes(uint32_t points, uint32_t envs) {
    if (points == 0u) return 0u;
    if (envs == 0u || points % envs != 0u ||
        uint64_t{points} * 2u > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("contact cache exceeds sort index range");
    size_t sort_bytes = 0u, scan_bytes = 0u, compact_bytes = 0u;
    cub::DoubleBuffer<uint64_t> keys(nullptr, nullptr);
    cub::DoubleBuffer<uint32_t> order(nullptr, nullptr);
    auto status = cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, keys, order,
        static_cast<int>(points * 2u), 0, SortBits(envs));
    if (status == cudaSuccess)
        status = cub::DeviceScan::ExclusiveScan(nullptr, scan_bytes,
            static_cast<const Ranks*>(nullptr), static_cast<Ranks*>(nullptr),
            AddRanks{}, Ranks{}, static_cast<int>(points));
    if (status == cudaSuccess)
        status = cub::DeviceScan::ExclusiveSum(nullptr, compact_bytes,
            static_cast<const uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
            static_cast<int>(points * 2u));
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    if (scan_bytes > sort_bytes) sort_bytes = scan_bytes;
    if (compact_bytes > sort_bytes) sort_bytes = compact_bytes;
    return CheckedAdd(Layout(points, envs).temp_offset, sort_bytes);
}

struct Workspace {
    uint32_t* order;
    uint32_t* alternate;
    uint64_t* keys;
    uint64_t* alternate_keys;
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
        alternate = reinterpret_cast<uint32_t*>(data + layout.alternate_offset);
        keys = reinterpret_cast<uint64_t*>(data + layout.keys_offset);
        alternate_keys = reinterpret_cast<uint64_t*>(data + layout.alternate_keys_offset);
        ranks = reinterpret_cast<Ranks*>(data + layout.ranks_offset);
        matches = reinterpret_cast<uint32_t*>(data + layout.match_offset);
        begins = reinterpret_cast<uint32_t*>(data + layout.begin_offset);
        ends = reinterpret_cast<uint32_t*>(data + layout.end_offset);
        temp = data + layout.temp_offset;
        temp_bytes = bytes - layout.temp_offset;
    }
};

static __global__ void InitValidKernel(KeySource keys, uint32_t* valid,
                                       uint32_t* matches, uint32_t* owner, uint32_t* keep) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= keys.points * 2u) return;
    valid[i] = keys.Valid(keys.Source(i)) ? 1u : 0u;
    if (i < keys.points) {
        matches[i] = ~0u;
        owner[i] = 0u;
        keep[i] = 0u;
    }
}

static __global__ void CompactSourcesKernel(KeySource keys, const uint32_t* valid,
                                             const uint32_t* prefix, uint32_t* order,
                                             uint32_t* begins, uint32_t* ends) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= keys.points * 2u) return;
    const uint32_t envs = keys.points / keys.points_per_env;
    const uint32_t stride = keys.points_per_env * 2u;
    if (i < envs) {
        const uint32_t begin = i * stride, last = begin + stride - 1u;
        begins[i] = begin;
        ends[i] = begin + prefix[last] + valid[last] - prefix[begin];
    }
    if (valid[i] == 0u) return;
    const uint32_t begin = (i / stride) * stride;
    order[begin + prefix[i] - prefix[begin]] = keys.Source(i);
}

// Unused tail slots sort after their env's entries, so each env keeps its fixed segment.
static __global__ void BucketKeysKernel(KeySource source, uint32_t* order,
                                        const uint32_t* ends, uint64_t* keys) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= source.points * 2u) return;
    const uint32_t env = i / (source.points_per_env * 2u);
    const uint64_t prefix = uint64_t{env} << 33u;
    if (i < ends[env]) {
        keys[i] = prefix | source.Bucket(order[i]);
    } else {
        keys[i] = prefix | (uint64_t{1u} << 32u);
        order[i] = 0u;
    }
}

// Hashes index buckets only; complete keys determine matches and ownership.
static __global__ void MergeGroupsKernel(KeySource keys, const uint32_t* order,
                                          const uint64_t* buckets,
                                          const uint32_t* begins, const uint32_t* ends,
                                          const uint32_t* age, uint32_t decay_steps,
                                          uint32_t* matches, uint32_t* owner, uint32_t* keep) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= keys.points * 2u) return;
    const uint32_t env = i / (keys.points_per_env * 2u);
    if (i >= ends[env]) return;
    const uint64_t bucket = buckets[i];
    if (i != begins[env] && buckets[i - 1u] == bucket) return;
    uint32_t end = i + 1u;
    while (end < ends[env] && buckets[end] == bucket) ++end;
    for (uint32_t entry = i; entry < end; ++entry) {
        const uint32_t source = order[entry];
        bool seen = false;
        for (uint32_t prior = i; prior < entry; ++prior) {
            if (keys.Same(order[prior], source)) { seen = true; break; }
        }
        if (seen) continue;
        uint32_t current = ~0u, old = ~0u, warm = ~0u;
        for (uint32_t j = entry; j < end; ++j) {
            const uint32_t id = order[j];
            if (!keys.Same(source, id)) continue;
            if (id < keys.points) {
                if (current == ~0u) current = id;
            } else {
                const uint32_t point = id - keys.points;
                if (old == ~0u) old = point;
                if (warm == ~0u && age[point] < decay_steps) warm = point;
            }
        }
        if (current != ~0u) {
            owner[current] = 1u;
            matches[current] = warm;
        } else if (old != ~0u && decay_steps > 1u && age[old] < decay_steps - 1u) {
            keep[old] = 1u;
        }
    }
}

inline cudaError_t BuildIndex(const DataView& data, uint32_t points,
                               uint32_t points_per_env, uint32_t decay_steps,
                               Workspace workspace, cudaStream_t stream) {
    constexpr uint32_t block = 128u;
    const uint32_t blocks = (points * 2u + block - 1u) / block;
    const uint32_t envs = points / points_per_env;
    const auto keys = Keys(data, points, points_per_env);
    auto* valid = reinterpret_cast<uint32_t*>(workspace.keys);
    auto* prefix = reinterpret_cast<uint32_t*>(workspace.alternate_keys);
    InitValidKernel<<<blocks, block, 0u, stream>>>(keys, valid,
        workspace.matches, data.contact_cache_current_owner, data.contact_cache_old_keep);
    if (const auto status = cudaGetLastError(); status != cudaSuccess) return status;
    size_t temp_bytes = workspace.temp_bytes;
    auto status = cub::DeviceScan::ExclusiveSum(workspace.temp, temp_bytes,
        valid, prefix, static_cast<int>(points * 2u), stream);
    if (status != cudaSuccess) return status;
    CompactSourcesKernel<<<blocks, block, 0u, stream>>>(keys, valid, prefix,
        workspace.order, workspace.begins, workspace.ends);
    if (const auto native = cudaGetLastError(); native != cudaSuccess) return native;
    cub::DoubleBuffer<uint64_t> key_buffers(workspace.keys, workspace.alternate_keys);
    cub::DoubleBuffer<uint32_t> order_buffers(workspace.order, workspace.alternate);
    BucketKeysKernel<<<blocks, block, 0u, stream>>>(keys, order_buffers.Current(),
        workspace.ends, key_buffers.Current());
    if (const auto native = cudaGetLastError(); native != cudaSuccess) return native;
    temp_bytes = workspace.temp_bytes;
    status = cub::DeviceRadixSort::SortPairs(workspace.temp, temp_bytes, key_buffers,
        order_buffers, static_cast<int>(points * 2u), 0, SortBits(envs), stream);
    if (status != cudaSuccess) return status;
    MergeGroupsKernel<<<blocks, block, 0u, stream>>>(keys,
        order_buffers.Current(), key_buffers.Current(), workspace.begins, workspace.ends,
        data.contact_cache_age, decay_steps, workspace.matches,
        data.contact_cache_current_owner, data.contact_cache_old_keep);
    return cudaGetLastError();
}

static __global__ void RankInputKernel(uint32_t points, const uint32_t* owner,
                                       const uint32_t* keep, Ranks* output) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < points) output[i] = {owner[i] == 0u ? 1u : 0u, keep[i]};
}

static __global__ void RetainedSourcesKernel(uint32_t points, uint32_t points_per_env,
                                              const uint32_t* keep, const Ranks* ranks,
                                              uint32_t* sources) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= points || keep[i] == 0u) return;
    const uint32_t begin = (i / points_per_env) * points_per_env;
    sources[begin + ranks[i].old - ranks[begin].old] = i;
}

// The sorted permutation becomes scan input after prepare; matches become retained sources.
inline cudaError_t BuildRanks(const DataView& data, uint32_t points,
                               uint32_t points_per_env, Workspace workspace, cudaStream_t stream) {
    constexpr uint32_t block = 128u;
    auto* input = reinterpret_cast<Ranks*>(workspace.order);
    RankInputKernel<<<(points + block - 1u) / block, block, 0u, stream>>>(points,
        data.contact_cache_current_owner, data.contact_cache_old_keep, input);
    if (const auto status = cudaGetLastError(); status != cudaSuccess) return status;
    const auto scanned = cub::DeviceScan::ExclusiveScan(workspace.temp, workspace.temp_bytes,
        input, workspace.ranks, AddRanks{}, Ranks{}, static_cast<int>(points), stream);
    if (scanned != cudaSuccess) return scanned;
    RetainedSourcesKernel<<<(points + block - 1u) / block, block, 0u, stream>>>(points,
        points_per_env, data.contact_cache_old_keep, workspace.ranks, workspace.matches);
    return cudaGetLastError();
}
}  // namespace nuka::phi::contact_cache
