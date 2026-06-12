#pragma once
// ---------------------------------------------------------------------------
// PHI v2 CUDA backend — M5 NarrowphaseSdf shared types + device transform
// helpers + the cross-TU standalone launcher seam (the precision oracle).
//
// math::Transform / Quat::Rotate are host-only `inline` (NOT __host__ __device__
// tagged), so device kernels reimplement the exact host expressions (the same
// posture union_types.cuh::RotateQuatHostExpr takes — bit-for-bit the host
// math). The SDF sampler itself (runtime/sdf/sparse_sdf_query.cuh) IS HD-clean.
// ---------------------------------------------------------------------------

#include <cstdint>

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/views.hpp"  // ModelView / DataView (complete types)
#include "phi/backend.hpp"   // Status
#include "runtime/sdf/sparse_sdf_query.cuh"

namespace nuka::phi::nkops {

// Quat rotate — EXACT host math::Quat::Rotate expression (no defensive
// normalize), so device detection matches the host narrowphase at the FP floor.
__forceinline__ __device__ math::Vec3 SdfRotate(math::Quat q, math::Vec3 v) {
    const math::Vec3 qv{q.x, q.y, q.z};
    const math::Vec3 t = 2.0f * qv.Cross(v);
    return v + q.w * t + qv.Cross(t);
}

// Device math::Transform::TransformPoint (rotate then translate).
__forceinline__ __device__ math::Vec3 SdfTransformPoint(const math::Transform& xf,
                                                        math::Vec3 p) {
    return SdfRotate(xf.rotation, p) + xf.position;
}

// Device inverse-transform of a world point into the transform's local frame:
// conj(q).Rotate(world - position). (math::Transform::Inverse uses Conjugate +
// Normalized; for a unit quat conjugate==inverse — the cooked poses are unit.)
__forceinline__ __device__ math::Vec3 SdfInverseTransformPoint(
    const math::Transform& xf, math::Vec3 world) {
    math::Quat c;
    c.w = xf.rotation.w;
    c.x = -xf.rotation.x;
    c.y = -xf.rotation.y;
    c.z = -xf.rotation.z;
    const math::Vec3 d{world.x - xf.position.x, world.y - xf.position.y,
                       world.z - xf.position.z};
    return SdfRotate(c, d);
}

// One SDF narrowphase work item: a sampling body (its SAMP slice) vs a target
// body carrying a cooked SDF grid. The pair is env-resolved (env + the contact
// slot it writes); bodies are template-local rows (env-major pose lookup).
struct SdfPairDev {
    uint32_t env;
    uint32_t slot;          // ucontact slot to write.
    uint32_t sample_body;   // template-local body row of the sampling shape.
    uint32_t target_body;   // template-local body row of the SDF shape.
    uint32_t sdf_grid;      // sdf_headers index of the target's grid.
    uint32_t bodies_per_env;
};

}  // namespace nuka::phi::nkops

namespace nuka::phi {

// Standalone SDF narrowphase launcher (the precision-oracle seam). Drives the
// sampling kernel over a hand-built pair list + device SAMP/SDF buffers.
Status LaunchNarrowphaseSdf(const float* samp_points,
                            const uint32_t* samp_ranges,
                            const float* shape_table,
                            const float* sdf_headers,
                            const uint32_t* sdf_cell_count,
                            const uint64_t* sdf_keys,
                            const float* sdf_values,
                            const math::Vec3* sdf_grads,
                            const math::Transform* body_pose,
                            const nkops::SdfPairDev* pairs,
                            uint32_t pair_count,
                            uint32_t k,
                            float margin,
                            uint32_t slot_stride,
                            uint32_t* ucount,
                            math::Vec3* upoint,
                            math::Vec3* unormal,
                            float* udepth,
                            uint32_t* contact_count,
                            cudaStream_t stream);

}  // namespace nuka::phi
