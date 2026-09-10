#pragma once

#include "collision/mesh_surface.hpp"
#include "collision/primitive_surface.hpp"
#include "phi/backend_cuda/ops/prims_types.cuh"
#include "runtime/sdf/sparse_sdf_query.cuh"

namespace nuka::phi::nkops {

struct SurfaceQueryView {
    collision::MeshSurfaceView mesh;
    const collision::MeshSurfaceInfo* mesh_info = nullptr;
    uint32_t shape_count = 0u;
    const float* sdf_headers = nullptr;
    const uint32_t* sdf_cell_count = nullptr;
    const uint64_t* sdf_keys = nullptr;
    const float* sdf_values = nullptr;
    const math::Vec3* sdf_grads = nullptr;
    uint32_t sdf_grid_count = 0u;
    uint32_t sdf_cell_total = 0u;
};

inline SurfaceQueryView MakeSurfaceQueryView(const ModelView& model, uint32_t shape_count,
    collision::MeshGeometryCounts mesh_counts, uint32_t sdf_grid_count, uint32_t sdf_cell_total) {
    return {{model.hull_verts, model.mesh_triangles, model.mesh_bvh_nodes, mesh_counts},
        model.mesh_surface_info, shape_count, model.sdf_headers, model.sdf_cell_count,
        model.sdf_cell_keys, model.sdf_cell_values, model.sdf_cell_gradients,
        sdf_grid_count, sdf_cell_total};
}

inline bool SurfaceQueryStorageValid(const SurfaceQueryView& view) {
    return (view.mesh.counts.triangles == 0u ||
            (view.mesh.vertices && view.mesh.triangles && view.mesh.nodes && view.mesh_info)) &&
           (view.sdf_grid_count == 0u || (view.sdf_headers && view.sdf_cell_count &&
            view.sdf_keys && view.sdf_values && view.sdf_grads));
}

__device__ inline bool HasCollidableSurface(
    const SurfaceQueryView& view, uint32_t body, const PrimShapeDev& shape) {
    if (shape.kind <= collision::kShapePlane) return true;
    if (shape.kind != collision::kShapeConvexHull && shape.kind != collision::kShapeSdfMesh)
        return false;
    if (view.mesh.counts.triangles > 0u && view.mesh_info && body < view.shape_count &&
        view.mesh_info[body].triangle_count > 0u) return true;
    return shape.sdf_grid < view.sdf_grid_count;
}

// The authored mesh supplies exact geometry; explicit SDF-only shapes use their field.
__device__ inline collision::MeshSurfacePoint QueryCollidableSurface(
    const SurfaceQueryView& view, uint32_t body, const PrimShapeDev& shape,
    math::Vec3 position, float max_distance) {
    collision::MeshSurfacePoint result;
    if (body >= view.shape_count) return result;
    if (shape.kind <= collision::kShapePlane) {
        const auto primitive = collision::QueryPrimitiveSurface(shape.kind,
            {shape.params[0], shape.params[1], shape.params[2]}, position);
        result.distance = primitive.distance;
        result.normal = primitive.normal;
        result.point = primitive.point;
        result.feature = primitive.feature;
        result.valid = primitive.valid;
        return result;
    }
    if (shape.kind != collision::kShapeConvexHull && shape.kind != collision::kShapeSdfMesh)
        return result;
    if (view.mesh.counts.triangles > 0u && view.mesh_info &&
        view.mesh_info[body].triangle_count > 0u)
        return collision::QueryMeshSurface(view.mesh, view.mesh_info[body], position, max_distance);
    if (shape.sdf_grid >= view.sdf_grid_count) return result;
    namespace sdf = ::nuka::runtime::sdf;
    const float* h = view.sdf_headers + static_cast<size_t>(shape.sdf_grid) * kSdfHeaderStride;
    const uint32_t offset = __float_as_uint(h[7]);
    const uint32_t count = view.sdf_cell_count[shape.sdf_grid];
    if (offset > view.sdf_cell_total || count > view.sdf_cell_total - offset ||
        !(h[3] > 0.0f) || !isfinite(h[3])) return result;
    sdf::SparseSdfDevice grid;
    grid.origin = {h[0], h[1], h[2]};
    grid.voxel_size = h[3];
    grid.dims[0] = __float_as_uint(h[4]);
    grid.dims[1] = __float_as_uint(h[5]);
    grid.dims[2] = __float_as_uint(h[6]);
    grid.cell_keys = view.sdf_keys + offset;
    grid.cell_values = view.sdf_values + offset;
    grid.cell_gradients = view.sdf_grads + offset;
    grid.cell_count = count;
    result.distance = sdf::sparse_sdf_sample(grid, position, result.normal);
    if (result.distance >= sdf::SparseSdfDevice::kOutsideBand) {
        result.valid = true;
        return result;
    }
    const float length = sqrtf(result.normal.LengthSq());
    if (!isfinite(result.distance) || !isfinite(length) || !(length > 0.0f)) return result;
    result.normal = result.normal / length;
    result.point = position - result.normal * result.distance;
    result.valid = true;
    return result;
}

}  // namespace nuka::phi::nkops
