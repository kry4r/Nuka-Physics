#pragma once
// ---------------------------------------------------------------------------
// PHI v2 CUDA backend — M5 pair-driven NarrowphasePrimitives shared types + the
// cross-TU launcher seam (contacts_foot.cu's OpNarrowphasePrimitives dispatches
// the pair-driven family here).
// ---------------------------------------------------------------------------

#include <cstdint>

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/views.hpp"  // ModelView / DataView (complete types)
#include "phi/backend.hpp"      // Status
#include "phi/op_schema.hpp"    // NarrowphasePrimitivesParams

namespace nuka::phi::nkops {

// Canonical shape_table row stride (packed f32 / body row). GREW 8 -> 10 (R1
// body_id/group) -> 12 (L-RECON-D hull slice). The SINGLE source of truth shared
// by every shape_table reader (LoadPrimShape, broadphase LoadShape) AND the model
// builder (model.cpp ShapeTable staging / fields.yaml max_bodies_total*12) — adding
// a lane bumps THIS one constant; the static_assert below pins the struct to it.
inline constexpr uint32_t kShapeTableRowStride = 12u;

// Device-side shape_table record. R1 GREW it 8 -> 10 packed f32 / body row
// (Model::PairDrivenShape staging in StageModelField(ShapeTable)). params:
// sphere r / capsule r,hh / box he.xyz, by kind. Lanes 8/9 = body_id (int32) +
// group (uint32), the shape->body indirection. L-RECON-D GREW it 10 -> 12,
// appending the per-shape hull slice {hull_vert_offset, hull_vert_count} into
// lanes 10/11 so the cvx narrowphase uses THIS shape's verts (0 == not a hull).
struct PrimShapeDev {
    uint32_t kind;
    float    params[4];
    uint32_t contype;
    uint32_t conaffinity;
    uint32_t sdf_grid;
    int32_t  body_id;     // owning body row, or -1 == static (R1).
    uint32_t group;       // signed collision-group filter key (R1).
    uint32_t hull_vert_offset;  // base vertex index into hull_verts/3 (L-RECON-D).
    uint32_t hull_vert_count;   // vertex count, 0 == not a hull row (L-RECON-D).
};
// All 12 lanes are 4-byte scalars, so the packed row is exactly the stride; this
// guards against a lane addition that forgets to bump kShapeTableRowStride.
static_assert(sizeof(PrimShapeDev) == kShapeTableRowStride * sizeof(float),
              "PrimShapeDev must pack exactly kShapeTableRowStride floats");

__forceinline__ __device__ PrimShapeDev LoadPrimShape(const float* table,
                                                      uint32_t row) {
    const float* q = table + static_cast<size_t>(row) * kShapeTableRowStride;
    PrimShapeDev s;
    s.kind = __float_as_uint(q[0]);
    s.params[0] = q[1]; s.params[1] = q[2]; s.params[2] = q[3]; s.params[3] = q[4];
    s.contype = __float_as_uint(q[5]);
    s.conaffinity = __float_as_uint(q[6]);
    s.sdf_grid = __float_as_uint(q[7]);
    s.body_id = static_cast<int32_t>(__float_as_uint(q[8]));
    s.group = __float_as_uint(q[9]);
    s.hull_vert_offset = __float_as_uint(q[10]);
    s.hull_vert_count = __float_as_uint(q[11]);
    return s;
}

// Quat rotate — EXACT host math::Quat::Rotate expression (matches BuildPrimFrame
// columns at the FP floor; same posture as union_types.cuh::RotateQuatHostExpr).
__forceinline__ __device__ math::Vec3 PrimRotate(math::Quat q, math::Vec3 v) {
    const math::Vec3 qv{q.x, q.y, q.z};
    const math::Vec3 t = 2.0f * qv.Cross(v);
    return v + q.w * t + qv.Cross(t);
}

// Pair-driven narrowphase launcher (contacts_foot.cu dispatches here for the
// kContactFamilyPairDriven family).
Status LaunchPairDrivenNarrowphase(const ModelView& model, const DataView& data,
                                   const NarrowphasePrimitivesParams& p,
                                   cudaStream_t stream);

}  // namespace nuka::phi::nkops
