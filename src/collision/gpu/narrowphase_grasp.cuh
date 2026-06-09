#pragma once
// ---------------------------------------------------------------------------
// nuka::collision::gpu -- BATCHED GRASP NARROWPHASE (v0.8 P2.4c)
// ---------------------------------------------------------------------------
// The host->GPU move of the grasp narrowphase (the P2.4b-dominant 90.9% stage).
// One __global__ thread per (env x fingertip) wraps the EXISTING HD-clean contact
// math cvx::SphereHull (sphere=fingertip x ConvexHull=cup) -- the SAME function
// the host narrowphase_dispatch.hpp grasp slot calls (sphere_is_a=true). NO new
// contact math: this is the find_sdf_contact_kernel template (sdf_contact.cu)
// applied to the SphereHull HD body (fixed-capacity stack arrays, no std::vector,
// fixed GJK/closest-point iteration caps -> D1 by construction).
//
// WHY A THIN WRAPPER. cvx::SphereHull is authored __host__ __device__ (NUKA_CVX_HD)
// over Vec3 HD ops + a baked PrimFrame, so it compiles verbatim into a kernel. The
// host-vs-device delta is therefore ULP-scale (the SAME float ops), which is why
// the batched world's A2 gate (vs the host StepGrasp oracle) stays at 1e-5 and the
// host-vs-device diagnostic confirms ~1e-6. Wiring this kernel into BOTH the batched
// world AND the test's StandaloneGraspRef keeps the A1 gate device-vs-device
// same-binary -> byte-exact, no gate edit.
//
// LAYOUT. The cup hull verts are SHARED (env-invariant, ONE upload); per-env the
// cup PrimFrame (built host-side from the live cup pose -- BuildPrimFrame/Quat::Rotate
// are host-only `inline`, NOT HD) selects the world transform. Per (env x fingertip)
// a GraspSphereInput carries the fingertip world center + radius (the sphere
// PrimFrame.t) -- the host already has these from the batched FK download, so we
// avoid a device round-trip of the FK poses. Output: one ContactManifold per
// (env x fingertip) slot, point_count=0 for a no-contact thread (the SphereHull
// separated early-return). The host launcher then applies StampSides + the param
// merge (mirroring contact_stream_driver.cpp) so each non-empty slot is a drop-in
// for what host BuildContactManifolds produced.
//
// SCOPE: sphere x ConvexHull ONLY (the grasp fingertip<->cup pair). The box x plane
// ground path stays host (off the perf critical path; its P2.2 gates are unchanged).
// ---------------------------------------------------------------------------

#include "collision/convex_narrowphase.hpp"   // cvx::SphereHull / ConvexHullView (HD math)
#include "constraint/contact_manifold.hpp"    // ContactManifold (POD output)
#include "collision/analytical_manifold.hpp"  // amf::PrimFrame (POD, host-baked)
#include "math/vec3.hpp"

#include <cstdint>

namespace nuka::collision::gpu {

// One (env x fingertip) thread's per-slot SPHERE input. POD, uploaded as a flat
// device array. The cup hull frame is per-env (NOT per-slot); we index it by
// env_index so all fingertips of an env share one frame upload. #ifndef-guarded so
// a TU that includes BOTH this .cuh and the host .hpp does not double-declare it.
#ifndef NUKA_GRASP_SPHERE_INPUT_DEFINED
#define NUKA_GRASP_SPHERE_INPUT_DEFINED
struct GraspSphereInput {
    math::Vec3 center{0.0f, 0.0f, 0.0f};  // fingertip world center (sphere PrimFrame.t)
    float      radius = 0.0f;             // fingertip sphere radius
    uint32_t   env_index = 0u;            // which per-env cup hull frame this slot reads
};
#endif

// The __global__ kernel: one thread per (env x fingertip) slot. Reads the slot's
// sphere input + its env's cup hull frame (frames[in.env_index]) + the SHARED cup
// verts (cup_verts/cup_vcount), runs cvx::SphereHull(sphere, hull, sphere_is_a=true),
// and writes the resulting ContactManifold to results[slot]. point_count=0 ==
// no-contact (the SphereHull separated early-return). Slot-indexed (no atomics,
// no reduction, fixed iteration caps) -> D1.
__global__ void grasp_sphere_hull_kernel(
    const GraspSphereInput* __restrict__ inputs,
    const amf::PrimFrame*   __restrict__ cup_frames,   // one per env
    const float*            __restrict__ cup_verts,    // shared mesh-local verts (vcount*3)
    uint32_t                             cup_vcount,
    constraint::ContactManifold* __restrict__ results,
    uint32_t                             slot_count);

}  // namespace nuka::collision::gpu
