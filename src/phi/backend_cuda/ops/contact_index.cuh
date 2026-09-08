#pragma once

#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_sort.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "core/checked_size.hpp"
#include "nk/model/generated/views.hpp"
#include "nk/solve/nk_row.hpp"

namespace nuka::phi::contact_index {

constexpr uint32_t kEndpointsPerRow = 2u;
struct Rank { uint32_t rows, endpoints; };
struct AddRank {
    __host__ __device__ Rank operator()(Rank a, Rank b) const {
        return {a.rows + b.rows, a.endpoints + b.endpoints};
    }
};

struct RowInput {
    const nk::NkRow* rows;
    const uint32_t* link_a;
    const uint32_t* link_b;
    uint32_t rows_per_env, links_per_env;

    __host__ __device__ bool ValidLink(uint32_t row, uint32_t link) const {
        const uint32_t first = (row / rows_per_env) * links_per_env;
        return link >= first && link - first < links_per_env;
    }
    __host__ __device__ Rank operator()(uint32_t row) const {
        if (!(rows[row].flags & nk::nk_row_flags::kActive)) return {};
        return {1u, static_cast<uint32_t>(ValidLink(row, link_a[row])) +
                    static_cast<uint32_t>(ValidLink(row, link_b[row]))};
    }
};

using Input = thrust::transform_iterator<RowInput, thrust::counting_iterator<uint32_t>>;

struct Layout {
    size_t keys, begins, ends, temp;
    Layout(uint32_t rows, uint32_t envs) {
        keys = CheckedAlignUp(CheckedProduct({rows, sizeof(Rank)}), 256u);
        begins = CheckedAlignUp(CheckedAdd(keys,
            CheckedProduct({rows, kEndpointsPerRow, sizeof(uint64_t)})), 256u);
        ends = CheckedAlignUp(CheckedAdd(begins, CheckedProduct({uint64_t{envs} + 1u, sizeof(uint32_t)})), 256u);
        temp = CheckedAlignUp(CheckedAdd(ends, CheckedProduct({uint64_t{envs} + 1u, sizeof(uint32_t)})), 256u);
    }
};

inline uint64_t ScratchBytes(uint32_t rows, uint32_t envs) {
    if (rows == 0u) return 0u;
    if (envs == 0u || rows % envs != 0u ||
        uint64_t{rows} * kEndpointsPerRow > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("contact index exceeds device sort range");
    size_t sort_bytes = 0u, scan_bytes = 0u;
    auto status = cub::DeviceSegmentedSort::StableSortKeys(nullptr, sort_bytes,
        static_cast<const uint64_t*>(nullptr), static_cast<uint64_t*>(nullptr),
        static_cast<int>(rows * kEndpointsPerRow), static_cast<int>(envs + 1u),
        static_cast<const uint32_t*>(nullptr), static_cast<const uint32_t*>(nullptr));
    const Input input(thrust::counting_iterator<uint32_t>(0u), RowInput{});
    if (status == cudaSuccess)
        status = cub::DeviceScan::ExclusiveScan(nullptr, scan_bytes, input,
            static_cast<Rank*>(nullptr), AddRank{}, Rank{}, static_cast<int>(rows));
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    return CheckedAdd(Layout(rows, envs).temp, sort_bytes > scan_bytes ? sort_bytes : scan_bytes);
}

// Prefix ranks and endpoint keys occupy distinct regions throughout compaction.
static __global__ void CompactKernel(RowInput input, const Rank* prefix, uint32_t row_count,
                                      uint32_t link_count, uint32_t envs, uint32_t* active_rows,
                                      uint32_t* active_count, uint32_t* endpoint_count,
                                      uint64_t* keys, uint32_t* begins, uint32_t* ends,
                                      uint32_t* link_begin, uint32_t* link_end) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < link_count) link_begin[row] = link_end[row] = 0u;
    if (row < envs) {
        const uint32_t first = row * input.rows_per_env;
        const uint32_t last = first + input.rows_per_env - 1u;
        const Rank last_rank = AddRank{}(prefix[last], input(last));
        active_count[row] = last_rank.rows - prefix[first].rows;
        endpoint_count[row] = last_rank.endpoints - prefix[first].endpoints;
        begins[row] = first * kEndpointsPerRow;
        ends[row] = begins[row] + endpoint_count[row];
    }
    if (row == 0u) begins[envs] = ends[envs] = row_count * kEndpointsPerRow;
    if (row >= row_count || input(row).rows == 0u) return;
    const uint32_t first = (row / input.rows_per_env) * input.rows_per_env;
    active_rows[first + prefix[row].rows - prefix[first].rows] = row;
    uint32_t destination = first * kEndpointsPerRow + prefix[row].endpoints - prefix[first].endpoints;
    const uint32_t links[kEndpointsPerRow] = {input.link_a[row], input.link_b[row]};
    for (uint32_t side = 0u; side < kEndpointsPerRow; ++side) {
        if (input.ValidLink(row, links[side]))
            keys[destination++] = (uint64_t{links[side]} << 32u) | (row * kEndpointsPerRow + side);
    }
}

// Keys order endpoints by global link, stable row ID, and side A before B.
static __global__ void LinkSpansKernel(const uint64_t* keys, const uint32_t* begins,
                                       const uint32_t* ends, uint32_t count, uint32_t stride,
                                       uint32_t* link_begin, uint32_t* link_end) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const uint32_t env = i / stride;
    if (i >= ends[env]) return;
    const auto link = static_cast<uint32_t>(keys[i] >> 32u);
    if (i == begins[env] || static_cast<uint32_t>(keys[i - 1u] >> 32u) != link) link_begin[link] = i;
    if (i + 1u == ends[env] || static_cast<uint32_t>(keys[i + 1u] >> 32u) != link) link_end[link] = i + 1u;
}

inline cudaError_t Build(const DataView& data, uint32_t rows_per_env, uint32_t links_per_env,
                         uint32_t envs, uint64_t workspace_bytes, cudaStream_t stream) {
    const uint64_t rows64 = uint64_t{rows_per_env} * envs;
    const uint64_t links64 = uint64_t{links_per_env} * envs;
    if (rows64 == 0u || rows64 > static_cast<uint64_t>(std::numeric_limits<int>::max()) / kEndpointsPerRow ||
        links64 > std::numeric_limits<uint32_t>::max() || !data.contact_index_scratch ||
        !data.urows || !data.row_cj_link || !data.row_cj_link_b ||
        !data.active_row_ids || !data.active_row_count || !data.contact_endpoint_keys ||
        !data.contact_endpoint_count || !data.link_contact_begin || !data.link_contact_end)
        return cudaErrorInvalidValue;
    const auto rows = static_cast<uint32_t>(rows64), links = static_cast<uint32_t>(links64);
    const Layout layout(rows, envs);
    if (workspace_bytes < layout.temp) return cudaErrorInvalidValue;
    auto* base = data.contact_index_scratch;
    auto* prefix = reinterpret_cast<Rank*>(base);
    auto* keys = reinterpret_cast<uint64_t*>(base + layout.keys);
    auto* begins = reinterpret_cast<uint32_t*>(base + layout.begins);
    auto* ends = reinterpret_cast<uint32_t*>(base + layout.ends);
    void* temp = base + layout.temp;
    const size_t available = static_cast<size_t>(workspace_bytes - layout.temp);
    size_t temp_bytes = available;
    const RowInput source{reinterpret_cast<const nk::NkRow*>(data.urows), data.row_cj_link,
                          data.row_cj_link_b, rows_per_env, links_per_env};
    const Input input(thrust::counting_iterator<uint32_t>(0u), source);
    auto status = cub::DeviceScan::ExclusiveScan(temp, temp_bytes, input, prefix,
        AddRank{}, Rank{}, static_cast<int>(rows), stream);
    if (status != cudaSuccess) return status;
    constexpr uint32_t block = 128u;
    const uint64_t extent = rows > links ? rows : links;
    CompactKernel<<<static_cast<uint32_t>((extent + block - 1u) / block), block, 0u, stream>>>(
        source, prefix, rows, links, envs, data.active_row_ids, data.active_row_count,
        data.contact_endpoint_count, keys, begins, ends, data.link_contact_begin, data.link_contact_end);
    if (const auto error = cudaGetLastError(); error != cudaSuccess) return error;
    temp_bytes = available;
    status = cub::DeviceSegmentedSort::StableSortKeys(temp, temp_bytes, keys,
        data.contact_endpoint_keys, static_cast<int>(rows * kEndpointsPerRow),
        static_cast<int>(envs + 1u), begins, ends, stream);
    if (status != cudaSuccess) return status;
    const uint32_t endpoints = rows * kEndpointsPerRow;
    LinkSpansKernel<<<(endpoints + block - 1u) / block, block, 0u, stream>>>(
        data.contact_endpoint_keys, begins, ends, endpoints, rows_per_env * kEndpointsPerRow,
        data.link_contact_begin, data.link_contact_end);
    return cudaGetLastError();
}

}  // namespace nuka::phi::contact_index
