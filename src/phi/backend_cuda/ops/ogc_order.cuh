#pragma once

#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "core/checked_size.hpp"
#include "nk/contact/contact_identity.hpp"
#include "nk/model/generated/views.hpp"
#include "nk/solve/nk_row.hpp"
#include "nk/solve/point_endpoint.hpp"
#include "phi/op_schema.hpp"

namespace nuka::phi::ogc_order {

constexpr uint32_t kSideTerms = nk::kTriangleEndpointTerms;
constexpr uint32_t kSlotTerms = 2u * kSideTerms;

// One detected OGC point with its DAT references; endpoint sides keep their terms, not their
// ordinal-dependent indices.
struct Record {
    uint64_t pair, feature;
    math::Vec3 point, witness_a, witness_b, normal;
    float depth, friction;
    uint32_t side[2], side_kind[2], gen, law, count;
    uint32_t dat[5];
    uint32_t term_count[2];
    nk::PointEndpointTerm terms[kSlotTerms];
};

struct Layout {
    size_t alternate_offset, keys_offset, alternate_keys_offset, records_offset, temp_offset;
    static size_t Align(size_t value) { return CheckedAlignUp(value, 256u); }
    explicit Layout(uint32_t items)
        : alternate_offset(Align(size_t{items} * sizeof(uint32_t))),
          keys_offset(Align(alternate_offset + size_t{items} * sizeof(uint32_t))),
          alternate_keys_offset(Align(keys_offset + size_t{items} * sizeof(uint64_t))),
          records_offset(Align(alternate_keys_offset + size_t{items} * sizeof(uint64_t))),
          temp_offset(Align(records_offset + size_t{items} * sizeof(Record))) {}
};

// Buckets only index records that complete comparisons then order, so capacity bounds their bits.
inline int BucketBits(uint32_t capacity) {
    int bits = 1;
    while (bits < 32 && (uint64_t{1} << bits) < capacity) ++bits;
    return bits;
}

// Keys hold env, an inactive-tail bit, then the bucket.
inline int SortBits(uint32_t envs, uint32_t capacity) {
    int bits = BucketBits(capacity) + 1;
    for (uint32_t value = envs - 1u; value != 0u; value >>= 1u) ++bits;
    return bits;
}

inline uint64_t ScratchBytes(uint32_t capacity, uint32_t envs) {
    const uint64_t items = uint64_t{capacity} * envs;
    if (items == 0u) return 0u;
    if (items > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("OGC contact order exceeds sort index range");
    size_t sort_bytes = 0u;
    cub::DoubleBuffer<uint64_t> keys(nullptr, nullptr);
    cub::DoubleBuffer<uint32_t> order(nullptr, nullptr);
    const auto status = cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, keys, order,
        static_cast<int>(items), 0, SortBits(envs, capacity));
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    return CheckedAdd(Layout(static_cast<uint32_t>(items)).temp_offset, sort_bytes);
}

struct Workspace {
    uint32_t* order;
    uint32_t* alternate;
    uint64_t* keys;
    uint64_t* alternate_keys;
    Record* records;
    void* temp;
    size_t temp_bytes;
    Workspace(void* base, size_t bytes, uint32_t items) {
        const Layout layout(items);
        auto* data = static_cast<uint8_t*>(base);
        order = reinterpret_cast<uint32_t*>(data);
        alternate = reinterpret_cast<uint32_t*>(data + layout.alternate_offset);
        keys = reinterpret_cast<uint64_t*>(data + layout.keys_offset);
        alternate_keys = reinterpret_cast<uint64_t*>(data + layout.alternate_keys_offset);
        records = reinterpret_cast<Record*>(data + layout.records_offset);
        temp = data + layout.temp_offset;
        temp_bytes = bytes - layout.temp_offset;
    }
};

__device__ inline uint32_t Bucket(uint64_t pair, uint64_t feature) {
    const uint64_t hash = nk::ContactHashWord(nk::ContactHashWord(1469598103934665603ull, pair), feature);
    return static_cast<uint32_t>(hash) ^ static_cast<uint32_t>(hash >> 32u);
}

// Lexicographic order over every record word, with floats compared by bit pattern.
struct RecordOrder {
    int order = 0;
    __device__ void Add(uint64_t a, uint64_t b) { if (order == 0 && a != b) order = a < b ? -1 : 1; }
    __device__ void Add(uint32_t a, uint32_t b) { Add(uint64_t{a}, uint64_t{b}); }
    __device__ void Add(float a, float b) { Add(__float_as_uint(a), __float_as_uint(b)); }
    __device__ void Add(math::Vec3 a, math::Vec3 b) { Add(a.x, b.x); Add(a.y, b.y); Add(a.z, b.z); }
};

__device__ inline int Compare(const Record& x, const Record& y) {
    RecordOrder o;
    o.Add(x.pair, y.pair);
    o.Add(x.feature, y.feature);
    for (uint32_t s = 0u; s < 2u; ++s) {
        o.Add(x.side_kind[s], y.side_kind[s]);
        o.Add(x.side[s], y.side[s]);
        o.Add(x.term_count[s], y.term_count[s]);
    }
    o.Add(x.point, y.point);
    o.Add(x.witness_a, y.witness_a);
    o.Add(x.witness_b, y.witness_b);
    o.Add(x.normal, y.normal);
    o.Add(x.depth, y.depth);
    o.Add(x.friction, y.friction);
    o.Add(x.gen, y.gen);
    o.Add(x.law, y.law);
    o.Add(x.count, y.count);
    for (uint32_t k = 0u; k < 5u; ++k) o.Add(x.dat[k], y.dat[k]);
    for (uint32_t k = 0u; k < kSlotTerms; ++k) {
        o.Add(x.terms[k].kind, y.terms[k].kind);
        o.Add(x.terms[k].index, y.terms[k].index);
        for (uint32_t axis = 0u; axis < 3u; ++axis)
            o.Add(x.terms[k].column[axis], y.terms[k].column[axis]);
    }
    return o.order;
}

// Records are read from the detected slots before any canonical slot is written.
static __global__ void SnapshotKernel(OgcDetectParams p, DataView data, int bucket_bits,
                                      uint64_t* keys, uint32_t* order, Record* records) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= p.env_count * p.slot_capacity) return;
    const uint32_t env = item / p.slot_capacity;
    const uint32_t ordinal = item - env * p.slot_capacity;
    const uint64_t prefix = uint64_t{env} << (bucket_bits + 1);
    order[item] = item;
    if (ordinal >= data.ogc_contact_count[env]) {
        keys[item] = prefix | (uint64_t{1u} << bucket_bits);
        return;
    }
    const uint32_t slot = env * p.slot_stride + p.slot_base + ordinal;
    const size_t at = size_t{slot} * nk::kPairDrivenPtsPerSlot;
    Record r;
    r.pair = data.ucontact_id_pair[at];
    r.feature = data.ucontact_id_feature[at];
    r.point = data.ucontact_point[at];
    r.witness_a = data.ucontact_witness_a[at];
    r.witness_b = data.ucontact_witness_b[at];
    r.normal = data.ucontact_normal[at];
    r.depth = data.ucontact_depth[at];
    r.friction = data.ucontact_friction[slot];
    r.gen = data.ucontact_gen[at];
    r.law = data.ucontact_law[slot];
    r.count = data.ucontact_count[slot];
    const uint32_t sides[2] = {data.ucontact_a[at], data.ucontact_b[at]};
    const uint32_t kinds[2] = {data.ucontact_a_kind[at], data.ucontact_b_kind[at]};
    for (uint32_t s = 0u; s < 2u; ++s) {
        const bool point = kinds[s] == nk::kUContactSidePointEndpoint;
        const nk::PointEndpointRange range =
            point ? data.point_endpoint_ranges[sides[s]] : nk::PointEndpointRange{};
        if (range.count > kSideTerms) atomicOr(data.env_status + env, kEnvStatusInvalidEndpoint);
        r.side_kind[s] = kinds[s];
        r.side[s] = point ? 0u : sides[s];
        r.term_count[s] = range.count < kSideTerms ? range.count : kSideTerms;
        for (uint32_t k = 0u; k < kSideTerms; ++k)
            r.terms[s * kSideTerms + k] = k < r.term_count[s]
                ? data.point_endpoint_terms[range.first + k] : nk::PointEndpointTerm{};
    }
    // Owners and features are only defined for mixed pairs.
    const uint32_t kind = data.dat_pair_kind[slot];
    r.dat[0] = kind;
    r.dat[1] = kind != 0u ? data.dat_pair_owner_a[slot] : 0u;
    r.dat[2] = kind != 0u ? data.dat_pair_owner_b[slot] : 0u;
    r.dat[3] = kind != 0u ? data.dat_pair_feature_a[slot] : 0u;
    r.dat[4] = kind != 0u ? data.dat_pair_feature_b[slot] : 0u;
    records[item] = r;
    keys[item] = prefix | (Bucket(r.pair, r.feature) >> (32 - bucket_bits));
}

__device__ inline void WriteRecord(const OgcDetectParams& p, const DataView& data, uint32_t env,
                                   uint32_t ordinal, const Record& r) {
    const uint32_t slot = env * p.slot_stride + p.slot_base + ordinal;
    const size_t at = size_t{slot} * nk::kPairDrivenPtsPerSlot;
    const uint32_t endpoint = env * p.point_endpoints_per_env + p.point_endpoint_first + ordinal * 2u;
    const uint32_t term = env * p.point_endpoint_terms_per_env + p.point_endpoint_term_first +
                          ordinal * kSlotTerms;
    uint32_t sides[2];
    for (uint32_t s = 0u; s < 2u; ++s) {
        const bool point = r.side_kind[s] == nk::kUContactSidePointEndpoint;
        sides[s] = point ? endpoint + s : r.side[s];
        data.point_endpoint_ranges[endpoint + s] = point
            ? nk::PointEndpointRange{term + s * kSideTerms, r.term_count[s]} : nk::PointEndpointRange{};
        for (uint32_t k = 0u; k < kSideTerms; ++k)
            data.point_endpoint_terms[term + s * kSideTerms + k] = r.terms[s * kSideTerms + k];
    }
    data.ucontact_point[at] = r.point;
    data.ucontact_witness_a[at] = r.witness_a;
    data.ucontact_witness_b[at] = r.witness_b;
    data.ucontact_normal[at] = r.normal;
    data.ucontact_depth[at] = r.depth;
    data.ucontact_a[at] = sides[0];
    data.ucontact_b[at] = sides[1];
    data.ucontact_a_kind[at] = r.side_kind[0];
    data.ucontact_b_kind[at] = r.side_kind[1];
    data.ucontact_gen[at] = r.gen;
    data.ucontact_law[slot] = r.law;
    data.ucontact_friction[slot] = r.friction;
    data.ucontact_id_pair[at] = r.pair;
    data.ucontact_id_feature[at] = r.feature;
    data.ucontact_count[slot] = r.count;
    data.dat_pair_kind[slot] = r.dat[0];
    data.dat_pair_owner_a[slot] = r.dat[1];
    data.dat_pair_owner_b[slot] = r.dat[2];
    data.dat_pair_feature_a[slot] = r.dat[3];
    data.dat_pair_feature_b[slot] = r.dat[4];
}

// Equal-bucket runs rank by complete records; identical records take their sorted positions in turn.
static __global__ void WriteKernel(OgcDetectParams p, DataView data, const uint64_t* keys,
                                   const uint32_t* order, const Record* records) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= p.env_count * p.slot_capacity) return;
    const uint32_t env = item / p.slot_capacity;
    const uint32_t begin = env * p.slot_capacity;
    const uint32_t end = begin + data.ogc_contact_count[env];
    if (item >= end) return;
    const uint64_t key = keys[item];
    uint32_t first = item, last = item + 1u;
    while (first > begin && keys[first - 1u] == key) --first;
    while (last < end && keys[last] == key) ++last;
    const Record& record = records[order[item]];
    uint32_t rank = 0u;
    for (uint32_t other = first; other < last; ++other) {
        if (other == item) continue;
        const int sign = Compare(records[order[other]], record);
        rank += sign < 0 || (sign == 0 && other < item) ? 1u : 0u;
    }
    WriteRecord(p, data, env, first - begin + rank, record);
}

inline cudaError_t Canonicalize(const OgcDetectParams& p, const DataView& data,
                                Workspace workspace, cudaStream_t stream) {
    constexpr uint32_t block = 128u;
    const uint32_t items = p.env_count * p.slot_capacity;
    const uint32_t blocks = (items + block - 1u) / block;
    const int bucket_bits = BucketBits(p.slot_capacity);
    cub::DoubleBuffer<uint64_t> keys(workspace.keys, workspace.alternate_keys);
    cub::DoubleBuffer<uint32_t> order(workspace.order, workspace.alternate);
    SnapshotKernel<<<blocks, block, 0u, stream>>>(p, data, bucket_bits, keys.Current(),
                                                  order.Current(), workspace.records);
    if (const auto status = cudaGetLastError(); status != cudaSuccess) return status;
    size_t temp_bytes = workspace.temp_bytes;
    const auto status = cub::DeviceRadixSort::SortPairs(workspace.temp, temp_bytes, keys, order,
        static_cast<int>(items), 0, SortBits(p.env_count, p.slot_capacity), stream);
    if (status != cudaSuccess) return status;
    WriteKernel<<<blocks, block, 0u, stream>>>(p, data, keys.Current(), order.Current(), workspace.records);
    return cudaGetLastError();
}

}  // namespace nuka::phi::ogc_order
