#pragma once
// ---------------------------------------------------------------------------
// nuka::collision::gpu -- BATCHED GRASP NARROWPHASE host launcher (v0.8 P2.4c)
// ---------------------------------------------------------------------------
// The HOST-includable face of narrowphase_grasp.cu: the LaunchGraspSphereHullNarrowphase
// drop-in for the grasp fingertip<->cup narrowphase (host BuildContactManifolds). This
// header carries NO __global__ / CUDA-only syntax (the kernel lives in the .cu / .cuh),
// so plain host TUs (batched_unified_world.cpp, the test's StandaloneGraspRef) include
// THIS to call the launcher. GraspSphereInput (the POD slot input) lives in the .cuh,
// but is re-declared here as the SAME POD so a host caller can populate it without
// pulling the .cuh's kernel decl. (Both definitions are byte-identical POD; a CUDA TU
// includes the .cuh, a host TU includes this -- they agree on the layout.)
// ---------------------------------------------------------------------------

#include "collision/analytical_manifold.hpp"  // amf::PrimFrame (POD)
#include "constraint/collidable.hpp"           // CollidableRef
#include "constraint/contact_manifold.hpp"     // ContactManifold
#include "math/vec3.hpp"

#include <cstdint>
#include <vector>

namespace nuka::phi { struct DeviceContext; }

namespace nuka::collision::gpu {

// The per-(env x fingertip) sphere slot input (mirrors the .cuh POD verbatim). A
// host caller fills `center` (fingertip world center) + `radius` + `env_index` (the
// per-env cup frame this slot reads). #ifndef-guarded against the .cuh's copy so a
// CUDA TU that includes BOTH headers does not double-declare it.
#ifndef NUKA_GRASP_SPHERE_INPUT_DEFINED
#define NUKA_GRASP_SPHERE_INPUT_DEFINED
struct GraspSphereInput {
    math::Vec3 center{0.0f, 0.0f, 0.0f};  // fingertip world center (sphere PrimFrame.t)
    float      radius = 0.0f;             // fingertip sphere radius
    uint32_t   env_index = 0u;            // which per-env cup hull frame this slot reads
};
#endif

// Launch the BATCHED grasp narrowphase: one GPU thread per (env x fingertip) slot
// runs cvx::SphereHull(sphere=fingertip, cup=hull, sphere_is_a=true). `inputs` is the
// flat per-slot sphere array; `cup_frames` is the per-env cup world frame (indexed by
// each slot's env_index); `cup_verts`/`cup_vcount` is the SHARED mesh-local cup hull.
// `side_a[i]`/`side_b[i]` are the candidate pair refs for slot i (fingertip / cup) used
// for StampSides. On return `out` is SLOT-INDEXED: out->size() == inputs.size(), with
// out[i] the manifold for input slot i. A no-contact slot is left default-constructed
// (point_count==0; the SphereHull separated early-return) -- the caller skips it (mirrors
// host BuildContactManifolds, which appends only non-empty manifolds). A non-empty slot
// is fully stamped (a/b/manifold_key + the merged friction/solimp/solref), byte-equivalent
// to host BuildContactManifolds modulo the ULP-scale SphereHull host-vs-device geometry
// delta. The slot-indexed layout lets a batched caller recover (env, fingertip) from i.
// `out` is cleared first.
void LaunchGraspSphereHullNarrowphase(
    const phi::DeviceContext& context,
    const std::vector<GraspSphereInput>& inputs,
    const std::vector<amf::PrimFrame>& cup_frames,
    const float* cup_verts,
    uint32_t cup_vcount,
    const std::vector<constraint::CollidableRef>& side_a,
    const std::vector<constraint::CollidableRef>& side_b,
    std::vector<constraint::ContactManifold>* out);

}  // namespace nuka::collision::gpu
