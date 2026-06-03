#pragma once
// ---------------------------------------------------------------------------
// nuka::rt -- the SENSOR renderer: camera -> LBVH -> closest-hit per-primitive
// intersection -> Lambert+GGX shade + hard shadows -> full AOV framebuffer
// (v0.7 p13). Builds DIRECTLY on the p12 stackless traversal (REUSED unchanged
// for both primary and shadow rays); the ONLY additions are the generalized leaf
// (triangle / sphere / sparse-SDF), one analytic light + hard shadow rays, and
// the widened AOV framebuffer (color/depth/normal/albedo/uv/prim_id).
//
// PROGRAMMATIC geometry only (built in-test). NO mesh-file importer (p16 orphan),
// NO photoreal path (G2). One ray per pixel, no AA/jitter/RNG -> D1 byte-exact
// (atomic-free, one writer per pixel; two runs memcmp-identical across all AOVs).
//
// HOST header (included from g++ test TUs). Device-only LBVH types stay in the
// .cu. The scene is described host-side; RenderScene uploads geometry, builds the
// tree, launches the kernel, and downloads the framebuffer.
// ---------------------------------------------------------------------------

#include "collision/aabb.hpp"
#include "math/vec3.hpp"
#include "rt/camera.hpp"
#include "rt/framebuffer.hpp"
#include "rt/material.hpp"
#include "runtime/sdf/sparse_sdf_query.cuh"

#include <cstdint>
#include <vector>

namespace nuka::rt {

// Primitive kind tag (one per scene primitive). Selects the leaf intersection
// function + attribute reconstruction.
enum class PrimKind : uint32_t {
    Triangle = 0u,
    Sphere = 1u,
    Sdf = 2u,
};

// One triangle primitive (world-space vertices).
struct TrianglePrim {
    math::Vec3 v0;
    math::Vec3 v1;
    math::Vec3 v2;
    uint32_t material_id = 0u;
};

// One sphere primitive (world-space center + radius).
struct SpherePrim {
    math::Vec3 center;
    float radius = 1.0f;
    uint32_t material_id = 0u;
};

// One sparse-SDF primitive. The cooked SDF is rendered at IDENTITY transform
// (world == local). `keys`/`values`/`gradients` are the host-side cooked arrays
// (RenderScene uploads them); `header` carries origin/voxel/dims/cell_count and
// world-space `aabb` clips the sphere-trace march.
struct SdfPrim {
    runtime::sdf::SparseSdfDevice header; // pointers FILLED on device by RenderScene
    collision::AABB aabb;                 // world bound (== local, identity)
    std::vector<uint64_t> keys;
    std::vector<float> values;
    std::vector<math::Vec3> gradients;
    uint32_t material_id = 0u;
    float surface_eps = 1.0e-3f;
    int max_iters = 256;
};

// A complete scene. The closest-hit prim id (LBVH leaf's original index) indexes
// into a FLAT primitive table built in declaration order: all triangles first,
// then all spheres, then all SDFs. RenderScene builds the per-prim AABBs + the
// kind/sub-index tables internally.
struct Scene {
    std::vector<TrianglePrim> triangles;
    std::vector<SpherePrim> spheres;
    std::vector<SdfPrim> sdfs;
    std::vector<Material> materials;
    Light light;
    AmbientTerm ambient;
};

// Render the scene for `camera` into a full AOV framebuffer. Builds per-prim
// AABBs, the LBVH (BuildLbvhForQuery, retained), launches the closest-hit +
// shade kernel (primary ray closest hit, then ONE hard shadow ray per lit hit
// via the SAME traversal), and downloads every AOV. Deterministic; no RNG;
// atomic-free.
Framebuffer RenderScene(const Scene& scene, const PinholeCamera& camera);

} // namespace nuka::rt
