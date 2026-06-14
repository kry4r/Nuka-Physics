#pragma once
// ---------------------------------------------------------------------------
// nuka::render::RenderWorldToTwoLevelScene -- the HOST, CUDA-FREE adapter that
// maps the backend-agnostic RenderWorld (M8 manifest #1) into the self-written
// ray tracer's scene-desc POD rt::TwoLevelScene (M11 RT-4 / §3.10 RT row).
//
// RenderWorld is the single render data product that feeds BOTH the Vulkan raster
// renderer AND the software path-tracer (render_world.hpp design constraint). This
// adapter is the path-tracer side of that fork: it deduplicates nothing new (the
// MeshLibrary is already deduped), it simply re-expresses each library mesh as a
// BLAS of triangles, each RenderInstance as a placed rt::Instance referencing its
// mesh's BLAS, and each RenderMaterial as an rt::Material. The result is handed to
// render::RtBackendI::BuildScene / Trace (the CUDA backend) -- this header names
// NO CUDA and NO backend type, so it compiles into nuka_render (zero-CUDA red-
// line over src/render/**).
//
// GEOMETRY: MeshGeometry is positions + indices (triangle soup); each triangle
// (i0,i1,i2) becomes one rt::TrianglePrim in the mesh's LOCAL frame (the instance
// transform places it in the world, matching the two-level tracer's BLAS-local /
// TLAS-world split). Meshes with no triangles produce an empty BlasMesh (a valid,
// no-hit BLAS) so blas_id indices stay 1:1 with MeshLibrary ids.
//
// MATERIALS: rt::TwoLevelScene material is PER-INSTANCE; rt::Material is indexed
// by Instance::material_id. We emit one rt::Material per RenderWorld material (in
// order, so render_material_id maps directly) and resolve each instance's
// material_id (kNoId / out-of-range -> the appended default material).
//
// CAMERA / LIGHT: RenderCameraToPinhole builds an rt::PinholeCamera from a
// RenderCamera (its world_xform forward/up + vfov); the first RenderLight maps to
// the scene's single analytic rt::Light (the bounded sensor renderer takes one
// light; extra lights are ignored with the brightest-or-first policy documented
// below). Both are pure host transforms of the already-resolved RenderWorld pose.
// ---------------------------------------------------------------------------

#include "render/render_world.hpp"
#include "rt/camera.hpp"
#include "rt/two_level_render.hpp"

#include <cstdint>

namespace nuka::render {

// Build a path-tracer scene-desc from a RenderWorld. The returned TwoLevelScene's
// meshes are 1:1 with world.meshes (same ids), instances 1:1 with world.instances
// (using each instance's live world_xform), and materials cover every
// world.materials entry plus an appended default for unresolved instances.
rt::TwoLevelScene RenderWorldToTwoLevelScene(const RenderWorld& world);

// Build an rt::PinholeCamera from a RenderWorld camera at the given image size.
// Uses camera.world_xform (forward = -Z column convention of the render camera /
// resolved to a look direction) + vertical_fov_degrees. Pure host.
rt::PinholeCamera RenderCameraToPinhole(const RenderCamera& camera,
                                        uint32_t width,
                                        uint32_t height);

}  // namespace nuka::render
