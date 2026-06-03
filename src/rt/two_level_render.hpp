#pragma once
// ---------------------------------------------------------------------------
// nuka::rt -- the self-written TWO-LEVEL (TLAS/BLAS) ray tracer (v0.7 G1-A).
// NO OptiX / NO closed SDK. Built to make the p16 demo (H1 robot + cup + table
// + USD scene = MANY rigid mesh INSTANCES) render FAST: p13's RenderScene
// rebuilds ONE giant LBVH over EVERY triangle EVERY frame; this builds each
// UNIQUE mesh's BLAS ONCE and rebuilds only the cheap N-instance TLAS per frame.
//
// THE ALGORITHM (architect + advisor verified): nest the p12 stackless
// rt::TraverseRay TWO deep, REUSED VERBATIM at both levels.
//   * TLAS = top-level LBVH over instance WORLD-AABBs (rebuilt per frame).
//   * BLAS = per-unique-mesh LBVH over that mesh's primitives in LOCAL space
//     (built ONCE in BuildTwoLevelScene; rigid -> never refit).
//   * The TLAS leaf functor, instead of intersecting a primitive: (a) transforms
//     the ray into the instance-local frame (rt::TransformRayToLocal), (b) runs
//     an INNER rt::TraverseRay over that instance's BLAS with a p13-style prim
//     leaf, (c) the inner traversal updates the SAME best_t/best_prim by ref.
//   * Rigid -> |d_l| == |d| == 1 -> local-frame t == world t, so best_t is
//     directly comparable ACROSS instances with NO rescale, and t_min is passed
//     into the BLAS unchanged.
//
// prim_id AOV packs (instance, local-prim) via rt::PackPrimId (instance HIGH);
// the closest-hit tie-break is then a TOTAL ORDER (lowest instance, then lowest
// local) -> D1 by construction (atomic-free, one ray/pixel, two runs identical).
//
// SCOPE: RIGID two-level CORE only (p16 is rigid; soft-body BLAS refit is G1-D,
// out of scope). NO wide-BVH8 / ray-sorting / persistent threads / quantized
// nodes (G1-B, dropped/deferred). Programmatic geometry only (no mesh importer).
// World-space normal AOV (p13 convention; NOT p14b's camera-space sensor frame).
// Material is PER-INSTANCE (a shared BLAS cannot carry instance-specific
// material); the per-prim material_id in BlasMesh is unused in G1-A.
//
// HOST header (included from g++ test TUs). Device-only LBVH types stay in the
// .cu. Mirrors scene_render.hpp.
// ---------------------------------------------------------------------------

#include "math/transform.hpp"
#include "rt/camera.hpp"
#include "rt/framebuffer.hpp"
#include "rt/material.hpp"
#include "rt/scene_render.hpp"  // TrianglePrim / SpherePrim / SdfPrim (reused)

#include <cstdint>
#include <memory>
#include <vector>

namespace nuka::rt {

// One UNIQUE mesh's primitives in its OWN LOCAL space (p13 declaration-order
// layout: all triangles first, then spheres, then SDFs). A BLAS is built over
// these ONCE. The per-prim material_id fields are UNUSED in G1-A (material is
// per-instance; see Instance::material_id).
struct BlasMesh {
    std::vector<TrianglePrim> triangles;
    std::vector<SpherePrim> spheres;
    std::vector<SdfPrim> sdfs;
};

// One placed instance: which unique mesh (blas_id indexes TwoLevelScene::meshes),
// its rigid world transform, and its material (per-instance shading).
struct Instance {
    uint32_t blas_id = 0u;
    math::Transform transform;
    uint32_t material_id = 0u;
};

// A complete two-level scene. `meshes` are the unique BLAS geometries (built
// once); `instances` place them in the world (the TLAS is rebuilt per frame over
// these). `materials` is indexed by Instance::material_id.
struct TwoLevelScene {
    std::vector<BlasMesh> meshes;
    std::vector<Instance> instances;
    std::vector<Material> materials;
    Light light;
    AmbientTerm ambient;
};

// Opaque, move-only device handle that owns the per-mesh BLAS trees + uploaded
// LOCAL prim buffers. Survives across frames (rigid BLAS is never refit); the
// per-frame TLAS is built inside RenderFrame. Built once by BuildTwoLevelScene.
class TwoLevelSceneDevice {
public:
    TwoLevelSceneDevice();
    ~TwoLevelSceneDevice();
    TwoLevelSceneDevice(const TwoLevelSceneDevice&) = delete;
    TwoLevelSceneDevice& operator=(const TwoLevelSceneDevice&) = delete;
    TwoLevelSceneDevice(TwoLevelSceneDevice&&) noexcept;
    TwoLevelSceneDevice& operator=(TwoLevelSceneDevice&&) noexcept;

    struct Impl;  // defined in the .cu
    const Impl* GetImpl() const { return impl_.get(); }
    Impl* GetImpl() { return impl_.get(); }

private:
    std::unique_ptr<Impl> impl_;
};

// Build each unique mesh's BLAS ONCE (per-mesh LBVH over LOCAL prims + uploaded
// local prim buffers + per-mesh local AABB for the TLAS world-box build). Host-
// asserts the prim_id caps (instances <= 4096, per-BLAS prims <= 2^20).
TwoLevelSceneDevice BuildTwoLevelScene(const TwoLevelScene& scene);

// Render one frame: rebuild the TLAS over the CURRENT instance world-AABBs
// (read from `scene.instances` each call so moving an instance tracks), launch
// the nested-traversal kernel, download all 6 AOVs. `device` must have been
// built from a scene with the SAME meshes (the BLAS is reused); transforms /
// materials / light are read fresh from `scene`.
Framebuffer RenderFrame(TwoLevelSceneDevice& device,
                        const TwoLevelScene& scene,
                        const PinholeCamera& camera);

} // namespace nuka::rt
