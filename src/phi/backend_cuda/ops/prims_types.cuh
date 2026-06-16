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

// Device-side shape_table record. R1 GREW it 8 -> 10 packed f32 / body row
// (Model::PairDrivenShape staging in StageModelField(ShapeTable)). params:
// sphere r / capsule r,hh / box he.xyz, by kind. Lanes 8/9 = body_id (int32) +
// group (uint32), the shape->body indirection (INERT in Phase 0).
struct PrimShapeDev {
    uint32_t kind;
    float    params[4];
    uint32_t contype;
    uint32_t conaffinity;
    uint32_t sdf_grid;
    int32_t  body_id;     // owning body row, or -1 == static (R1).
    uint32_t group;       // signed collision-group filter key (R1).
};

__forceinline__ __device__ PrimShapeDev LoadPrimShape(const float* table,
                                                      uint32_t row) {
    const float* q = table + static_cast<size_t>(row) * 10u;
    PrimShapeDev s;
    s.kind = __float_as_uint(q[0]);
    s.params[0] = q[1]; s.params[1] = q[2]; s.params[2] = q[3]; s.params[3] = q[4];
    s.contype = __float_as_uint(q[5]);
    s.conaffinity = __float_as_uint(q[6]);
    s.sdf_grid = __float_as_uint(q[7]);
    s.body_id = static_cast<int32_t>(__float_as_uint(q[8]));
    s.group = __float_as_uint(q[9]);
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
