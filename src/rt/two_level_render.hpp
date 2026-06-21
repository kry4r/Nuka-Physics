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

namespace nuka::phi {
struct Backend;  // opaque (phi/backend.hpp); selects the render device + stream
struct Buffer;   // opaque (phi/buffer.hpp); the caller-resident AOV target
}  // namespace nuka::phi

namespace nuka::rt {

// Per-vertex shading normals for ONE triangle (LOCAL space, unit length). Parallel
// to BlasMesh::triangles. Consumed ONLY by the beauty path for smooth shading; the
// FP64 golden/sensor kernel ignores them (it uses the flat geometric normal).
struct TriangleNormals {
    math::Vec3 n0;
    math::Vec3 n1;
    math::Vec3 n2;
};

// One UNIQUE mesh's primitives in its OWN LOCAL space (p13 declaration-order
// layout: all triangles first, then spheres, then SDFs). A BLAS is built over
// these ONCE. The per-prim material_id fields are UNUSED in G1-A (material is
// per-instance; see Instance::material_id).
//
// tri_normals, when non-empty, is 1:1 with triangles and carries smooth per-vertex
// normals for the beauty smooth-shading path; empty => flat geometric normals
// (byte-identical to a mesh that ships no normals).
struct BlasMesh {
    std::vector<TrianglePrim> triangles;
    std::vector<SpherePrim> spheres;
    std::vector<SdfPrim> sdfs;
    std::vector<TriangleNormals> tri_normals;
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

// Which host AOV channels RenderBeauty copies back from the device. The kernel
// always WRITES every channel (determinism unchanged); this only gates the D2H
// copies. Default = all true => the returned Framebuffer is byte-unchanged. The
// mp4 path consumes only color + prim, so it clears the four it never reads.
struct AovDownloadMask {
    bool color = true;
    bool depth = true;
    bool normal = true;
    bool albedo = true;
    bool uv = true;
    bool prim = true;
};

// Stochastic render controls for the beauty path (RenderBeauty): jittered MSAA,
// soft sun shadows, AO + one-bounce GI, procedural sky/fog. Same scene feeds both.
struct BeautyOptions {
    uint32_t samples = 16u;          // jittered sub-pixel samples per pixel (MSAA)
    uint32_t shadow_rays = 4u;       // penumbra samples across the sun disc
    float sun_angular_radius = 0.04f;// sun half-angle (rad) -> soft-shadow penumbra
    uint32_t gi_bounces = 1u;        // diffuse indirect bounces (0 = AO-only ambient)
    uint32_t ao_samples = 3u;        // hemisphere rays for ambient occlusion
    float ao_radius = 0.6f;          // AO ray max distance (world units)
    uint32_t seed = 0x9e3779b9u;     // base RNG seed (fixed -> reproducible frames)
    uint32_t transmit_bounces = 2u;  // dielectric reflect/refract recursion depth cap
    bool smooth_normals = false;     // opaque arm uses per-vertex smooth normals (off => flat, byte-exact)

    // Procedural sky + height fog (matches the lavapipe atmosphere); the miss
    // shader returns sky(dir), reused as sky-dome ambient for secondary misses.
    math::Vec3 sky_top{0.55f, 0.62f, 0.72f};     // zenith (up)
    math::Vec3 sky_bottom{0.86f, 0.89f, 0.93f};  // horizon
    math::Vec3 sky_ground{0.34f, 0.35f, 0.37f};  // below-horizon (down) fill
    math::Vec3 fog_color{0.84f, 0.88f, 0.93f};   // height/distance fog tint
    float fog_density = 0.0f;                     // per-metre extinction (0 = off)
    float sky_intensity = 1.0f;                   // scales the sky-dome ambient

    // Host-download gate (default all => byte-unchanged). RenderBeautyToAovs and
    // the device-resident path ignore it; only the host RenderBeauty copy uses it.
    AovDownloadMask download;
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

// Caller-resident device AOV outputs (phi v2 device buffers). The kernel writes
// each non-null channel in place on the selected backend's stream -- no host
// round-trip. Sizes mirror the Framebuffer channels (color/normal/albedo 3*float,
// uv 2*float, depth 1*float, prim 1*uint32 per pixel). null skips that channel.
struct RtDeviceAovs {
    phi::Buffer* color  = nullptr;
    phi::Buffer* depth  = nullptr;
    phi::Buffer* normal = nullptr;
    phi::Buffer* albedo = nullptr;
    phi::Buffer* uv     = nullptr;
    phi::Buffer* prim   = nullptr;
};

// Build each unique mesh's BLAS ONCE (per-mesh LBVH over LOCAL prims + uploaded
// local prim buffers + per-mesh local AABB for the TLAS world-box build). Host-
// asserts the prim_id caps (instances <= 4096, per-BLAS prims <= 2^20). `backend`
// selects the device + stream the BLAS uploads run on; null falls back to the
// registry's first device on its own backend.
TwoLevelSceneDevice BuildTwoLevelScene(const TwoLevelScene& scene,
                                       phi::Backend* backend = nullptr);

// Surface ONE SensorBlasRef per built mesh (indexed by blas_id) so the batched
// sensor path reuses the SAME once-built BLAS. Additive; the trace path unchanged.
struct SensorBlasRef;  // defined in the CUDA-side sensor_scatter.hpp
void CollectSensorBlasRefs(const TwoLevelSceneDevice& device,
                           std::vector<SensorBlasRef>* out_refs);

// Render one frame: rebuild the TLAS over the CURRENT instance world-AABBs
// (read from `scene.instances` each call so moving an instance tracks), launch
// the nested-traversal kernel, download all 6 AOVs. `device` must have been
// built from a scene with the SAME meshes (the BLAS is reused); transforms /
// materials / light are read fresh from `scene`. `backend` selects the device +
// stream; null falls back to the registry's first device.
Framebuffer RenderFrame(TwoLevelSceneDevice& device,
                        const TwoLevelScene& scene,
                        const PinholeCamera& camera,
                        phi::Backend* backend = nullptr);

// Same nested-traversal trace, writing the requested AOVs into CALLER-RESIDENT
// device buffers in place (no host download). The pixel writes are identical to
// RenderFrame -- only the destination differs (the caller's buffers vs internal
// ones). `backend` selects the device + stream + AOV buffer type.
void RenderFrameToAovs(TwoLevelSceneDevice& device,
                       const TwoLevelScene& scene,
                       const PinholeCamera& camera,
                       const RtDeviceAovs& aov,
                       phi::Backend* backend = nullptr);

// Beauty render: a SEPARATE stochastic path tracer over the SAME device scene
// (MSAA + soft shadows + AO + one-bounce GI + sky/fog) so near-black bodies read
// as solid lit volumes. Reproducible given a fixed opt.seed; `color` is the beauty
// image, the other AOVs carry the center-sample primary hit. RenderFrame untouched.
Framebuffer RenderBeauty(TwoLevelSceneDevice& device,
                         const TwoLevelScene& scene,
                         const PinholeCamera& camera,
                         const BeautyOptions& opt,
                         phi::Backend* backend = nullptr);

// Beauty trace into CALLER-RESIDENT device AOV buffers in place (no host
// download). Same pixel writes as RenderBeauty; only the destination differs.
void RenderBeautyToAovs(TwoLevelSceneDevice& device,
                        const TwoLevelScene& scene,
                        const PinholeCamera& camera,
                        const BeautyOptions& opt,
                        const RtDeviceAovs& aov,
                        phi::Backend* backend = nullptr);

} // namespace nuka::rt
