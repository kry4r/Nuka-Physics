#pragma once
// ---------------------------------------------------------------------------
// Host-side sampling helpers for the SDF cooker tests (p07).
// ---------------------------------------------------------------------------
// The cook tests sample the cooked SDF on the CPU through the SAME function p08
// ships on the GPU (runtime/sdf/sparse_sdf_query.cuh::sparse_sdf_sample). These
// helpers build a SparseSdfDevice view over host-side cooked data so that view
// can be handed to sparse_sdf_sample directly — the analytical test therefore
// validates the shipping sampler path, not a separate mirror.
// ---------------------------------------------------------------------------

#include "import/cooker/sparse_sdf_storage.hpp"
#include "runtime/sdf/sparse_sdf_query.cuh"
#include "math/vec3.hpp"

#include <vector>

namespace nuka::test {

// Owns the math::Vec3 gradient buffer (SparseSdfData stores gradients as flat
// floats; SparseSdfDevice wants math::Vec3*). Keep this alive while sampling.
struct HostSdfView {
    std::vector<nuka::math::Vec3> gradients;
    nuka::runtime::sdf::SparseSdfDevice device;
};

inline HostSdfView MakeHostView(const nuka::import::cooker::SparseSdfData& sdf) {
    HostSdfView view;
    const uint32_t n = sdf.CellCount();
    view.gradients.resize(n);
    for (uint32_t c = 0; c < n; ++c) {
        view.gradients[c] = {sdf.cell_gradients[3 * c + 0],
                             sdf.cell_gradients[3 * c + 1],
                             sdf.cell_gradients[3 * c + 2]};
    }
    auto& d = view.device;
    d.origin = {sdf.origin[0], sdf.origin[1], sdf.origin[2]};
    d.voxel_size = sdf.voxel_size;
    d.dims[0] = sdf.dims[0];
    d.dims[1] = sdf.dims[1];
    d.dims[2] = sdf.dims[2];
    d.cell_keys = sdf.cell_keys.data();
    d.cell_values = sdf.cell_values.data();
    d.cell_gradients = view.gradients.data();
    d.cell_count = n;
    return view;
}

inline float SampleHost(const HostSdfView& view, float x, float y, float z,
                        nuka::math::Vec3& grad) {
    return nuka::runtime::sdf::sparse_sdf_sample(view.device, {x, y, z}, grad);
}

}  // namespace nuka::test
