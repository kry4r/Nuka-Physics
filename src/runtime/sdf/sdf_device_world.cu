// ---------------------------------------------------------------------------
// nuka::runtime::sdf::SdfDeviceWorld implementation (p08-A)
// ---------------------------------------------------------------------------
// Upload-only: a pure host->device copy of the cooked sparse SDF table. No
// kernel, no atomics -- it is in physics_path but trivially D1.
// ---------------------------------------------------------------------------

#include "runtime/sdf/sdf_device_world.hpp"

#include "phi/buffer_transfer.hpp"

#include <utility>
#include <vector>

namespace nuka::runtime::sdf {

using ::nuka::phi::UploadVector;

SdfDeviceWorld::SdfDeviceWorld(uint32_t sdf_count,
                               uint32_t piece_count,
                               uint32_t cell_count,
                               phi::Buffer cell_keys,
                               phi::Buffer cell_values,
                               phi::Buffer cell_gradients,
                               phi::Buffer piece_sdf_indices,
                               phi::Buffer views)
    : sdf_count_(sdf_count)
    , piece_count_(piece_count)
    , cell_count_(cell_count)
    , cell_keys_(std::move(cell_keys))
    , cell_values_(std::move(cell_values))
    , cell_gradients_(std::move(cell_gradients))
    , piece_sdf_indices_(std::move(piece_sdf_indices))
    , views_(std::move(views)) {}

const SparseSdfDevice* SdfDeviceWorld::DeviceViews() const {
    return static_cast<const SparseSdfDevice*>(views_.Data());
}

const uint32_t* SdfDeviceWorld::DevicePieceSdfIndices() const {
    return static_cast<const uint32_t*>(piece_sdf_indices_.Data());
}

const uint64_t* SdfDeviceWorld::DeviceCellKeys() const {
    return static_cast<const uint64_t*>(cell_keys_.Data());
}

const float* SdfDeviceWorld::DeviceCellValues() const {
    return static_cast<const float*>(cell_values_.Data());
}

const math::Vec3* SdfDeviceWorld::DeviceCellGradients() const {
    return static_cast<const math::Vec3*>(cell_gradients_.Data());
}

std::size_t SdfDeviceWorld::DeviceBytes() const {
    return cell_keys_.Size()
        + cell_values_.Size()
        + cell_gradients_.Size()
        + piece_sdf_indices_.Size()
        + views_.Size();
}

SdfDeviceWorld UploadSdfDeviceWorld(const phi::DeviceContext& context,
                                    const scene::CookedSdfTable& sdf_table) {
    phi::ScopedDeviceGuard guard(context.device_id);

    const uint32_t sdf_count = sdf_table.Count();
    const uint32_t piece_count =
        static_cast<uint32_t>(sdf_table.piece_sdf_indices.size());
    const uint32_t cell_count =
        static_cast<uint32_t>(sdf_table.cell_keys.size());

    // 1. Upload the flat cell arrays + the per-piece sdf-index map. The cooked
    //    table already stores gradients as math::Vec3, so this is a plain blob
    //    copy with no conversion.
    phi::Buffer cell_keys = UploadVector(sdf_table.cell_keys);
    phi::Buffer cell_values = UploadVector(sdf_table.cell_values);
    phi::Buffer cell_gradients = UploadVector(sdf_table.cell_gradients);
    phi::Buffer piece_sdf_indices = UploadVector(sdf_table.piece_sdf_indices);

    // 2. Build the per-SDF SparseSdfDevice views ON THE HOST, with their cell_*
    //    pointers baked to DEVICE addresses (base device pointer + per-SDF key
    //    offset). We never dereference these host-side; the pointer arithmetic on
    //    cudaMalloc'd addresses is well-defined.
    const auto* keys_base =
        static_cast<const uint64_t*>(cell_keys.Data());
    const auto* values_base =
        static_cast<const float*>(cell_values.Data());
    const auto* gradients_base =
        static_cast<const math::Vec3*>(cell_gradients.Data());

    std::vector<SparseSdfDevice> views(sdf_count);
    for (uint32_t s = 0; s < sdf_count; ++s) {
        const uint32_t offset = sdf_table.key_offsets[s];
        const uint32_t count = sdf_table.key_counts[s];
        SparseSdfDevice& v = views[s];
        v.origin = sdf_table.origins[s];
        v.voxel_size = sdf_table.voxel_sizes[s];
        v.dims[0] = sdf_table.dims_x[s];
        v.dims[1] = sdf_table.dims_y[s];
        v.dims[2] = sdf_table.dims_z[s];
        v.cell_keys = keys_base + offset;
        v.cell_values = values_base + offset;
        v.cell_gradients = gradients_base + offset;
        v.cell_count = count;
    }

    // 3. Upload the view array as a plain POD blob.
    phi::Buffer views_buf = UploadVector(views);

    return SdfDeviceWorld(sdf_count,
                          piece_count,
                          cell_count,
                          std::move(cell_keys),
                          std::move(cell_values),
                          std::move(cell_gradients),
                          std::move(piece_sdf_indices),
                          std::move(views_buf));
}

SdfDeviceWorld UploadSdfDeviceWorld(const scene::CookedSdfTable& sdf_table) {
    auto context = phi::MakeDefaultDeviceContext();
    return UploadSdfDeviceWorld(context, sdf_table);
}

} // namespace nuka::runtime::sdf
