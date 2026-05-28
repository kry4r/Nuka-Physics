# Nuka Physics v0.7 – Phase 13: CUDA RT — Primitive Intersections + Simple Shading

> **Master plan reference:** §3 Round 10 (self-written CUDA RT)
> **Prerequisites:** v0.7 Phase 12 (LBVH + traversal kernel)
> **Blocks:** v0.7 Phase 14 (sensors render via this RT)
> **Exit criteria gate:** v0.7
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

Complete the self-written CUDA RT pipeline by adding:
1. **Primitive intersections**: mesh triangle, sphere (particles), sparse-SDF.
2. **Simple shading**: Lambert + GGX BRDF, point lights, ambient.
3. **Framebuffer output**: RGB + depth + albedo + normal aux buffers.

After this phase, the RT pipeline can render arbitrary scenes. Phase 14 wires this into sensors. Diff-rendering (analytical adjoint of the RT pipeline) is deferred to v2.0 Phase 7.

## Tech Stack

- CUDA 12+
- Phase 12 RT infrastructure
- Phase 7 sparse SDFs (for SDF intersection)

## Files to Create

- `src/rt/intersect_triangle.cuh` — Möller-Trumbore
- `src/rt/intersect_sphere.cuh` — quadratic formula
- `src/rt/intersect_sdf.cuh` — sphere tracing through narrow-band SDF
- `src/rt/material.hpp` — Lambert + GGX material types
- `src/rt/shading.cu` — shading kernel
- `src/rt/framebuffer.hpp` — multi-aux-buffer output structure
- `src/rt/scene.cu` — scene compilation (gather primitives, materials, lights)
- `tests/rt/test_triangle_intersect.cpp`
- `tests/rt/test_sphere_intersect.cpp`
- `tests/rt/test_sdf_intersect.cpp`
- `tests/rt/test_shading_lambert.cpp`
- `tests/rt/test_render_cornell_box.cpp` — quick visual sanity

## Tasks

### Task 7.13.1 — Triangle intersection (Möller-Trumbore)

`src/rt/intersect_triangle.cuh`:

```cuda
__device__ bool intersect_triangle(const Ray& r,
                                    float3 v0, float3 v1, float3 v2,
                                    float& out_t, float2& out_barycentric)
{
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 h = cross(r.direction, e2);
    float a = dot(e1, h);
    if (fabsf(a) < 1e-8f) return false;     // parallel
    float f = 1.f / a;
    float3 s = r.origin - v0;
    float u = f * dot(s, h);
    if (u < 0.f || u > 1.f) return false;
    float3 q = cross(s, e1);
    float v = f * dot(r.direction, q);
    if (v < 0.f || u + v > 1.f) return false;
    float t = f * dot(e2, q);
    if (t < r.t_min || t > r.t_max) return false;
    out_t = t;
    out_barycentric = {u, v};
    return true;
}
```

### Task 7.13.2 — Sphere intersection

For particles rendered as spheres:

```cuda
__device__ bool intersect_sphere(const Ray& r, float3 center, float radius,
                                  float& out_t, float3& out_normal)
{
    float3 oc = r.origin - center;
    float a = dot(r.direction, r.direction);
    float b = 2.f * dot(oc, r.direction);
    float c = dot(oc, oc) - radius * radius;
    float disc = b * b - 4.f * a * c;
    if (disc < 0) return false;
    float sqrt_d = sqrtf(disc);
    float t = (-b - sqrt_d) / (2.f * a);
    if (t < r.t_min) t = (-b + sqrt_d) / (2.f * a);
    if (t < r.t_min || t > r.t_max) return false;
    out_t = t;
    out_normal = normalize(r.origin + t * r.direction - center);
    return true;
}
```

Used for fluid (PBF particle) and soft body (XPBD particle) rendering.

### Task 7.13.3 — Sphere tracing through sparse SDF

For rigid bodies represented by sparse SDFs (Phase 7), use sphere tracing — march along ray, step by `min(|φ|, max_step)`:

```cuda
__device__ bool intersect_sdf(const Ray& r, const SparseSdfDevice& sdf,
                              const Transform& world_to_local,
                              float& out_t, float3& out_normal)
{
    float t = r.t_min;
    for (int i = 0; i < 64; ++i) {       // max iterations
        float3 p_world = r.origin + t * r.direction;
        float3 grad_local;
        float phi = sparse_sdf_sample(sdf, world_to_local * p_world, grad_local);
        if (phi < 1e-4f) {
            out_t = t;
            out_normal = normalize(transform_normal(world_to_local, grad_local));
            return true;
        }
        t += phi;
        if (t > r.t_max) return false;
    }
    return false;
}
```

This unifies rendering with collision — same SDF, two consumers.

### Task 7.13.4 — Material system

`src/rt/material.hpp`:

```cpp
struct Material {
    float3 albedo;
    float  metallic;
    float  roughness;
    float3 emission;
};
```

Per-primitive material ID indexes into a flat material table.

### Task 7.13.5 — Shading kernel

Lambert + GGX shading on the hit point:

```cuda
__device__ float3 shade(const Intersection& isect, const Ray& view_ray,
                         const Material& mat, const Light* lights, uint32_t num_lights)
{
    float3 result = float3{0,0,0};
    float3 N = isect.normal;
    float3 V = -view_ray.direction;

    for (uint32_t i = 0; i < num_lights; ++i) {
        const Light& L = lights[i];
        float3 light_dir = normalize(L.position - isect_point);
        float NoL = max(dot(N, light_dir), 0.f);
        // Lambert
        float3 diffuse = mat.albedo * (1.f - mat.metallic) * (NoL / PI);
        // GGX specular (simplified)
        float3 H = normalize(V + light_dir);
        float NoH = max(dot(N, H), 0.f);
        float a = mat.roughness * mat.roughness;
        float D = (a * a) / (PI * sqr(NoH * NoH * (a*a - 1.f) + 1.f));
        float3 spec = float3{D, D, D};   // simplified; full GGX adds F and G terms
        result += L.color * (diffuse + spec * mat.metallic) * NoL;
    }
    result += mat.emission;
    result += mat.albedo * 0.05f;   // ambient
    return result;
}
```

### Task 7.13.6 — Multi-aux framebuffer

`src/rt/framebuffer.hpp`:

```cpp
struct Framebuffer {
    float3* color;        // RGB
    float*  depth;        // distance from camera
    float3* normal;       // world-space normals
    float3* albedo;       // material albedo (for denoising / sim2real)
    uint32_t* primitive_id; // for semantic / instance segmentation
    uint32_t width, height;
};
```

Each `intersect + shade` writes into all aux buffers. Sensor code (Phase 14) selectively reads the buffers it needs.

### Task 7.13.7 — Scene compilation

Per frame (or per cook):
- Gather all rigid primitives (triangle meshes from convex pieces or original meshes; or SDFs).
- Gather all particles (XPBD + PBF) as spheres.
- Build RT-LBVH (Phase 12 LBVH build).

Refit per frame is cheaper than full rebuild; rebuild every 30-100 frames.

### Task 7.13.8 — Tests

`tests/rt/test_triangle_intersect.cpp`:

```cpp
// Single triangle; ray hits center; verify t and barycentric
```

`tests/rt/test_render_cornell_box.cpp`:

```cpp
// Cornell box (6 quads, 1 light); render 640x480
// Verify image roughly matches reference (sum of pixel values bounded; specific bright/dark regions)
```

## Validation

- Triangle / sphere / SDF intersection correctness on known geometry.
- Lambert shading matches mathematical reference.
- Cornell box renders recognizably.
- RT-LBVH + intersections + shading achieves throughput target for sensor use cases.
- Determinism: same scene + same camera → bit-exact framebuffer.

## Exit Criteria for v0.7 Phase 13

1. Three primitive intersection kernels operational.
2. Lambert + GGX shading.
3. Multi-aux framebuffer (color/depth/normal/albedo/prim_id).
4. Scene compilation (mesh + particle + SDF in one LBVH).
5. Per-frame refit + periodic rebuild.
6. Cornell box sanity render passes.
7. D1 determinism preserved.

## What This Phase Does Not Do

- No reflection / refraction / multi-bounce GI (future).
- No texture sampling (uniform material only for v0.7; textures in v1.0).
- No shadow rays (penumbra simplified to NoL term).
- No diff-rendering (v2.0 Phase 7).
- No camera motion blur (sim2real N3 in v1.0).
