#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::sdf::SdfDeviceWorld -- GPU upload of cooked sparse SDFs (p08-A)
// ---------------------------------------------------------------------------
// p07 cooks narrow-band SDFs into a HOST-side scene::CookedSdfTable (flat,
// concatenated SoA: per-SDF header arrays + flat cell_keys/cell_values/
// cell_gradients, and a per-convex-piece piece_sdf_indices map). There was no
// device upload, so the p08 Newton-summed-SDF contact kernel could not sample
// them on the GPU. This container closes that gap: it uploads the cooked table
// to phi::Buffer device buffers and exposes a DEVICE-RESIDENT array of
// SparseSdfDevice views (one per SDF) plus the piece->sdf mapping, so a contact
// kernel can do, for a candidate convex piece p:
//
//     const uint32_t s = device_piece_sdf_indices[p];          // kNoSdf if none
//     if (s != scene::kNoSdf) {
//         const SparseSdfDevice& sdf = device_views[s];
//         const float phi = sparse_sdf_sample(sdf, p_local, grad);  // p07 sampler
//     }
//
// HOW THE VIEWS ARE BUILT (no fill kernel needed): SparseSdfDevice is a flat
// raw-pointer POD. We UploadVector the flat cell arrays to device buffers, then
// build the per-SDF SparseSdfDevice structs ON THE HOST with their cell_* fields
// pointing into the uploaded device buffers (device-pointer arithmetic on the
// host is well-defined as long as we never dereference host-side -- we don't),
// and finally UploadVector that std::vector<SparseSdfDevice> as a plain blob.
// The result is a device array the kernel indexes directly.
//
// DETERMINISM: the upload is a pure host->device copy (no kernel, no atomics,
// no reduction), so it is trivially D1: uploading + sampling twice yields
// byte-identical device results. The sampler itself is p07's D1 code.
//
// LIFETIME: the uploaded views buffer's cell_* pointers ALIAS the cell data
// buffers. They are all members of this one object and move together; never let
// a view outlive the SdfDeviceWorld that owns the buffers it points into.
// ---------------------------------------------------------------------------

#include "phi/buffer_legacy.hpp"
#include "phi/device_context.hpp"
#include "runtime/sdf/sparse_sdf_query.cuh"
#include "scene/cooked_blob.hpp"

#include <cstddef>
#include <cstdint>

namespace nuka::runtime::sdf {

// CUDA-resident mirror of a scene::CookedSdfTable. Owns the device buffers and a
// device array of per-SDF SparseSdfDevice views + the per-piece sdf-index map.
class SdfDeviceWorld {
public:
    SdfDeviceWorld() = default;

    SdfDeviceWorld(const SdfDeviceWorld&) = delete;
    SdfDeviceWorld& operator=(const SdfDeviceWorld&) = delete;
    SdfDeviceWorld(SdfDeviceWorld&&) noexcept = default;
    SdfDeviceWorld& operator=(SdfDeviceWorld&&) noexcept = default;

    // Number of unique cooked SDFs (== device_views array length).
    uint32_t SdfCount() const { return sdf_count_; }
    // Number of convex-geometry pieces (== piece_sdf_indices array length).
    uint32_t PieceCount() const { return piece_count_; }
    // Total cooked narrow-band cells across all SDFs.
    uint32_t CellCount() const { return cell_count_; }

    // Device array of SparseSdfDevice views, one per SDF. Index it with an SDF
    // index in [0, SdfCount()); pass &views[s] (or views[s]) to a kernel that
    // calls sparse_sdf_sample. The cell_* pointers inside each view are device
    // addresses into this object's cell buffers.
    const SparseSdfDevice* DeviceViews() const;

    // Device array (length PieceCount()) mapping each convex-geometry piece to
    // its SDF index, or scene::kNoSdf if the piece has no cooked SDF.
    const uint32_t* DevicePieceSdfIndices() const;

    // Raw device pointers to the flat cell arrays (for diagnostics / advanced
    // callers; the kernel normally goes through DeviceViews()).
    const uint64_t* DeviceCellKeys() const;
    const float* DeviceCellValues() const;
    const math::Vec3* DeviceCellGradients() const;

    std::size_t DeviceBytes() const;

private:
    uint32_t sdf_count_ = 0;
    uint32_t piece_count_ = 0;
    uint32_t cell_count_ = 0;

    phi::Buffer cell_keys_;        // flat uint64 keys (concatenated, per-SDF sorted)
    phi::Buffer cell_values_;      // flat float signed distances
    phi::Buffer cell_gradients_;   // flat math::Vec3 gradients
    phi::Buffer piece_sdf_indices_; // per-piece uint32 sdf index (kNoSdf sentinel)
    phi::Buffer views_;            // device array of SparseSdfDevice (aliases above)

public:
    // Construct from already-built device buffers + counts. Public so the upload
    // factory below can hand it the finished buffers via move; not intended for
    // direct use.
    SdfDeviceWorld(uint32_t sdf_count,
                   uint32_t piece_count,
                   uint32_t cell_count,
                   phi::Buffer cell_keys,
                   phi::Buffer cell_values,
                   phi::Buffer cell_gradients,
                   phi::Buffer piece_sdf_indices,
                   phi::Buffer views);
};

// Upload a cooked SDF table to the device. Pure copy; deterministic.
SdfDeviceWorld UploadSdfDeviceWorld(const phi::DeviceContext& context,
                                    const scene::CookedSdfTable& sdf_table);
SdfDeviceWorld UploadSdfDeviceWorld(const scene::CookedSdfTable& sdf_table);

} // namespace nuka::runtime::sdf
