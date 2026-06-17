#pragma once
// ---------------------------------------------------------------------------
// PHI v2 CUDA backend — M4 union-family (CSR compliant) shared op types.
//
// The union contact pipeline (NarrowphasePrimitives union branch ->
// AssembleRows -> SolveRowsBlockIsland) transcribes the legacy coresident
// union world step semantics 1:1:
//   * detection = the SAME HD-clean analytic handlers the legacy host /
//     grasp-GPU narrowphase ran (amf::SpherePlane / amf::BoxPlane /
//     cvx::SphereHull), now launched device-resident per (env x slot);
//   * row emission = EmitCompliantContactRows' math (solref/solimp aref + R
//     via the SAME constraint/solref_solimp.hpp HD header, ChooseTangent
//     spoke basis, normals-then-spokes group layout) onto FIXED row slots;
//   * solve = row_solver.cu's compliant branch math (side dispatch, coupled-
//     pyramid friction bounds, regularizer feedback) on the device-resident
//     island/color schedule.
//
// This header carries the device-side UnionSlot unpack, the NkRow alias, and
// the host->device launcher seams shared between the op TUs.
// ---------------------------------------------------------------------------

#include <cstdint>

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/solve/nk_row.hpp"
#include "phi/backend.hpp"   // Status / ModelView / DataView
#include "phi/backend_cuda/ops/articulation_types.cuh"
#include "phi/op_schema.hpp"

namespace nuka::phi::nkops {

using ::nuka::nk::Spatial6;
using ::nuka::nk::NkRow;
using ::nuka::nk::NkRowSide;
using ::nuka::nk::kNkSideArtic;
using ::nuka::nk::kNkSideParticle;
using ::nuka::nk::kNkSideRigid;
using ::nuka::nk::kNkSideStatic;

// Union slot classes / flags — MUST mirror nk::UnionSlot (model.hpp).
inline constexpr uint32_t kUSlotInactive       = 0u;
inline constexpr uint32_t kUSlotFootSpherePlane = 1u;
inline constexpr uint32_t kUSlotFingerSphereHull = 2u;
inline constexpr uint32_t kUSlotBodyBoxPlane   = 3u;
// M6 particle coupling: side a == a particle sphere (slot.link == LOCAL particle
// index); side b == static +Z plane (kUSlotParticleSpherePlane) or rigid box
// (slot.body; kUSlotParticleSphereBox).
inline constexpr uint32_t kUSlotParticleSpherePlane = 4u;
inline constexpr uint32_t kUSlotParticleSphereBox   = 5u;
inline constexpr uint32_t kUSlotGatedOnTable   = 1u;  // flags bit0

// The 16-f32 packed union_slots record (Model::StageModelField UnionSlots).
struct UnionSlotDev {
    uint32_t   cls;
    uint32_t   link;
    uint32_t   body;
    uint32_t   condim;
    math::Vec3 offset;
    float      radius;
    math::Vec3 box_half;
    float      plane_height;
    float      mu;
    uint32_t   flags;
    uint32_t   row_base;   // per-env row-slot base (staged reserved[14])
    uint32_t   reserved15;
};

__forceinline__ __device__ UnionSlotDev LoadUnionSlot(const float* table,
                                                      uint32_t slot) {
    const float* p = table + static_cast<size_t>(slot) * 16u;
    UnionSlotDev u;
    u.cls    = __float_as_uint(p[0]);
    u.link   = __float_as_uint(p[1]);
    u.body   = __float_as_uint(p[2]);
    u.condim = __float_as_uint(p[3]);
    u.offset = {p[4], p[5], p[6]};
    u.radius = p[7];
    u.box_half = {p[8], p[9], p[10]};
    u.plane_height = p[11];
    u.mu = p[12];
    u.flags = __float_as_uint(p[13]);
    u.row_base = __float_as_uint(p[14]);
    u.reserved15 = __float_as_uint(p[15]);
    return u;
}

// Worst-case manifold points / rows per slot (mirror UnionSlot helpers).
__forceinline__ __device__ uint32_t USlotMaxPoints(const UnionSlotDev& u) {
    return u.cls == kUSlotBodyBoxPlane ? 4u : 1u;
}
__forceinline__ __device__ uint32_t USlotRowsPerPoint(const UnionSlotDev& u) {
    return 1u + (u.condim >= 2u ? 2u * (u.condim - 1u) : 0u);
}

// Quat rotate — EXACT replication of the HOST math::Quat::Rotate expression
// (t = 2*(qv x v); v + w*t + qv x t — NO defensive normalize), so the device
// detection matches the legacy host narrowphase at the FP floor. (The fused
// family's RotateByQuat normalizes defensively — a DIFFERENT legacy function;
// do not unify them.)
__forceinline__ __device__ math::Vec3 RotateQuatHostExpr(math::Quat q,
                                                         math::Vec3 v) {
    const math::Vec3 qv{q.x, q.y, q.z};
    const math::Vec3 t = 2.0f * qv.Cross(v);
    return v + q.w * t + qv.Cross(t);
}

// L1-c: the LaunchUnionNarrowphase cross-TU launcher seam was DELETED with
// contacts_union.cu (the UnionCsr narrowphase). The PairDriven narrowphase
// (narrowphase_prims.cu) is the only live detection path.

}  // namespace nuka::phi::nkops
