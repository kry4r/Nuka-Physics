#pragma once
// ---------------------------------------------------------------------------
// nuka::sensor -- RGB + depth + normal + uv CAMERA SENSOR over the LIVE batched
// simulation state (v0.7 p14b, exit-crit 6, second half; p14a closed the
// tactile/F-T half).
//
// This is a SENSOR, NOT a render demo: it renders the ACTUAL simulation state.
// Each body's cooked sparse SDF (the SAME p07 narrow-band SDF the p08 SDF-contact
// path and the p13 SDF sphere-trace leaf consume) is INSTANCED at the body's LIVE
// world pose (math::Transform from the batched world's DevicePoses()), so when the
// physics moves a body the rendered image tracks it. The four AOVs are READY to be
// surfaced as DLPack-able sensor channels over batched envs ([env, H, W, C]) -- see
// the "C-ABI / DLPack ARM -- DEFERRED" note below for why that plumbing lands in
// p16 (no renderable pose+SDF source on the C-ABI handle today), not p14b.
//
// FIDELITY (the mandated decision): SDF-INSTANCED (the preferred, real-geometry
// path), NOT the AABB-proxy degradation. The primary ray is transformed into each
// body's LOCAL frame (T^-1), sphere-traced against the body's cooked SDF in local
// space (RaySdfSphereTrace, REUSED verbatim from p13's leaf), and the hit point /
// normal are transformed back to world. A rigid transform keeps the ray direction
// unit, so the returned t is invariant -> depth needs no rescale.
//
// REUSE (no rewrite of p12/p13): the p13 __host__ __device__ SDF sphere-trace
// (rt::RaySdfSphereTrace), the p12 stackless traversal (rt::TraverseRay), the
// camera (rt::PinholeCamera), the AABB/LBVH build (collision::BuildLbvhForQuery),
// and the closest-hit + NaN-guarded slab helpers (rt::ray_box.cuh). The ONLY new
// code is the per-instance ray transform (quaternion rotation inlined in the
// kernel because math::Quat::Rotate / math::Transform are host-only `inline`, NOT
// __host__ __device__ -- we do NOT add HD markers to the shared math headers) and
// the per-body-id flat-palette albedo.
//
// SENSOR CONVENTIONS (differ from p13's WORLD-geometric AOVs -- a policy wants a
// viewer-relative frame):
//   * NORMAL: emitted in CAMERA space (viewer-faced) -- {dot(n,right),
//     dot(n,up), dot(n,forward)}, then flipped so dot(n_cam, +z_cam) <= 0 (faces
//     the viewer). p13 emitted world geometric normals; that stays unchanged.
//   * DEPTH: ray/camera-space distance t (origin + t*dir), RtMissDepth() on miss.
//   * RGB: a DETERMINISTIC flat per-body palette color keyed on the hit body id
//     (cooked SDFs carry no material; full PBR is p16/G2). A miss writes black.
//   * UV: the RT byproduct this phase surfaces; the SDF leaf has no surface
//     parameterisation, so uv = (0,0) on a hit (kept for channel-shape parity and
//     a future parameterised-surface leaf). A miss writes (0,0).
//
// BATCHED, ENV-MAJOR: outputs are [env, H, W, C]; this free function renders
// whatever env_count it is given (it is the raw, world-agnostic device entry).
//
// C-ABI / DLPack ARM -- DEFERRED (not wired in p14b). The plan scopes p14b's
// AOVs to surface as NUKA_FIELD_CAMERA_* DLPack channels (field ids from 20,
// mirroring p14a; single-env -> NOT_SUPPORTED). That plumbing is BLOCKED on a
// LIVE per-body SDF geometry source behind the C-ABI handle: the cooker
// populates a CookedSdfTable ONLY for ConvexHull/TriMesh shapes (scene/cooker.cpp
// ~line 236), and every world the handle constructs today (Go2/H1 articulations)
// steps on Sphere/Capsule/Box primitives -> sdf_table is EMPTY, so an eager
// Step-end render would emit all-sentinel (a sensor that senses nothing). The
// NAMED CONSUMER is p16 (H1 from a real USD/mesh scene -> ConvexHull -> SDFs);
// the DLPack channels + the camera-config setter + eager-render hook land there
// against real geometry. This header + RenderCameraSensor are the verified
// device core that p16's plumbing calls. See the p14b report for the proposed
// C-ABI surface (nuka_world_set_camera + mirror buffers).
//
// DETERMINISM (D1): one writer per pixel, NO atomics on any AOV, no RNG / AA /
// jitter, fixed per-env loop order. Built --fmad=false (paired with the p13
// intersection code it reuses) so the fp64-internal sphere-trace stays
// bit-consistent. Two runs are memcmp-identical across all channels.
// ---------------------------------------------------------------------------

#include "math/transform.hpp"
#include "phi/device_context.hpp"
#include "rt/camera.hpp"
#include "runtime/sdf/sparse_sdf_query.cuh"

#include <cstdint>

namespace nuka::sensor {

// Sentinel for "this body has no cooked SDF" in the body_to_sdf map (matches
// scene::kNoSdf; redeclared here so this TU needs no scene/ dependency).
inline constexpr uint32_t kCameraNoSdf = 0xFFFFFFFFu;

// Number of flat palette colors. The per-body color is palette[material_id %
// kCameraPaletteSize]; the palette is a fixed deterministic table baked in the .cu.
inline constexpr uint32_t kCameraPaletteSize = 8u;

// Render the LIVE batched world into per-env RGB/depth/normal/uv AOVs.
//
// Inputs (all device pointers unless noted; raw so this TU depends only on
// nuka_core/nuka_phi + math + the rt headers, never on the runtime world types):
//   poses        [env_count * body_count_per_env] math::Transform -- LIVE per-body
//                world poses, env-major (index env*body_count_per_env + body). The
//                SDF for body b is INSTANCED at poses[...].
//   sdf_views    [sdf_count] runtime::sdf::SparseSdfDevice -- the cooked per-SDF
//                views (DeviceViews()); the body's local SDF bound is derived from
//                origin / dims / voxel_size.
//   body_to_sdf  [body_count_per_env] uint32 -- local-body -> sdf index, or
//                kCameraNoSdf if the body has no cooked SDF (that body is skipped:
//                it contributes no geometry). SHARED across envs (the batched world
//                shares one shape/SDF table; only poses vary per env).
//   body_material[body_count_per_env] uint32 -- local-body -> palette id (RGB albedo
//                source). SHARED across envs.
//   cameras      [env_count] PinholeCamera (HOST pointer) -- one camera per env. All
//                must have the SAME width/height (the output layout is [env,H,W,C]).
//   env_count, body_count_per_env, width, height: dimensions. width/height taken
//                from cameras[0]; all cameras must match.
//
// Outputs (device pointers, env-major [env, H, W, C], must be preallocated):
//   out_rgb    [env_count * H * W * 3] float -- per-body palette color (r,g,b).
//   out_depth  [env_count * H * W]     float -- ray/camera-space distance (miss =>
//              RtMissDepth()).
//   out_normal [env_count * H * W * 3] float -- CAMERA-space viewer-faced normal.
//   out_uv     [env_count * H * W * 2] float -- (0,0) (RT byproduct placeholder).
//
// A non-positive env_count / body_count / width / height, or a null required
// pointer, is a no-op.
void RenderCameraSensor(const phi::DeviceContext& context,
                        const math::Transform* poses,
                        const runtime::sdf::SparseSdfDevice* sdf_views,
                        const uint32_t* body_to_sdf,
                        const uint32_t* body_material,
                        const rt::PinholeCamera* cameras,
                        uint32_t env_count,
                        uint32_t body_count_per_env,
                        uint32_t width,
                        uint32_t height,
                        float* out_rgb,
                        float* out_depth,
                        float* out_normal,
                        float* out_uv);

}  // namespace nuka::sensor
