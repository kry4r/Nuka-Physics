# Nuka Physics v0.7 – Phase 6: V-HACD Convex Decomposition (Cooker)

> **Master plan reference:** §3 Round 7 (cooker SDF infra) + §3 Round 12 (asset pipeline)
> **Prerequisites:** v0.5 closed (independent of solver / broadphase work; can run parallel to Phase 4-5)
> **Blocks:** v0.7 Phase 7 (sparse SDF) — V-HACD output feeds per-piece SDF
> **Exit criteria gate:** v0.7
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6. **V-HACD itself runs at cook-time (offline / pre-simulation), which is host CPU code — this is permitted as it falls under "one-time scene cook." It does not violate the constraint.**

## Goal

Integrate V-HACD (Voxelized Hierarchical Approximate Convex Decomposition) into the asset cooker. Concave / complex meshes (robot fingertips, panels, decorative objects) are broken into a union of convex pieces, each of which gets its own narrow-band SDF in Phase 7.

This solves the thin-shell SDF problem identified in master plan §3 Round 7 (MuJoCo's actual workaround): rather than try to resolve thin features at full mesh resolution, decompose into pieces and resolve each at appropriate resolution.

V-HACD runs **once at cook time** (offline / host CPU); the cooked output is GPU-resident and feeds runtime queries. Per §5.6, this is permitted as cook is one-time pre-simulation work.

## Tech Stack

- V-HACD library v4.x (open source, MIT, BSD)
- Existing scene importer / cooker pipeline (`src/import/`, `src/runtime/world_builder.cpp`)
- CUDA 12+ (for storing decomposition output on device)

## Files to Create

- `src/import/cooker/convex_decomposition.hpp`
- `src/import/cooker/convex_decomposition.cpp` — V-HACD invocation + result conversion
- `src/import/cooker/convex_piece.hpp` — single convex piece structure
- `src/include/nuka/nuka_cooker.h` — C ABI extension for cook-time control
- `external/vhacd/` — third-party submodule or vendored source
- `cmake/Findvhacd.cmake` or `external/vhacd/CMakeLists.txt`
- `tests/import/test_vhacd_simple_cube.cpp` — sanity: cube → 1 piece
- `tests/import/test_vhacd_robot_finger.cpp` — Go2 fingertip / H1 hand → expected piece count
- `tests/import/test_vhacd_determinism.cpp` — same mesh + params → same decomposition

## Files to Modify

- `src/runtime/world_builder.cpp` — invoke V-HACD on meshes flagged for decomposition
- `src/import/usda_importer.cpp` / `urdf_importer.cpp` / `mjcf_importer.cpp` — recognize a `nuka:decompose` USD custom attribute or URDF/MJCF tag
- `CMakeLists.txt` — vendor V-HACD as third party

## Tasks

### Task 7.6.1 — V-HACD vendoring

Include V-HACD as either:
- Git submodule pointing to upstream
- Vendored source under `external/vhacd/` (preferred for stability)

V-HACD is a pure CPU library (~10K lines of C++). Building it adds ~2 MB to the engine binary.

### Task 7.6.2 — Cooker invocation interface

`src/import/cooker/convex_decomposition.hpp`:

```cpp
namespace nuka::import::cooker {

struct ConvexDecompositionParams {
    uint32_t max_pieces = 32;             // upper bound
    uint32_t voxel_resolution = 100000;   // V-HACD voxelization grid (default 100K)
    float    concavity_threshold = 0.001f;
    uint32_t max_vertices_per_piece = 64;
    bool     project_hull_vertices = true;
};

struct ConvexPiece {
    std::vector<float> vertices;          // x,y,z triples
    std::vector<uint32_t> indices;        // triangle indices (closed mesh)
    float volume;
    float aabb_min[3], aabb_max[3];
};

struct ConvexDecompositionResult {
    std::vector<ConvexPiece> pieces;
    bool succeeded;
    std::string error_message;
};

ConvexDecompositionResult DecomposeMesh(const float* vertices, uint32_t vertex_count,
                                         const uint32_t* indices, uint32_t triangle_count,
                                         const ConvexDecompositionParams& params);

} // namespace
```

V-HACD's API is wrapped behind this interface so we can swap implementations later.

### Task 7.6.3 — USD / URDF / MJCF authoring

Allow scene authors to specify decomposition per-mesh:

USD (custom attribute on a mesh prim):
```
custom token nuka:decompose = "auto"           # default; cooker decides
custom token nuka:decompose = "force"          # always decompose
custom token nuka:decompose = "skip"           # treat as single convex
custom int   nuka:decompose:max_pieces = 16    # override
```

URDF (link's collision element):
```xml
<collision>
  <geometry><mesh filename="finger.obj"/></geometry>
  <nuka:decompose max_pieces="8"/>
</collision>
```

MJCF (geom):
```xml
<geom mesh="finger" nuka:decompose="auto"/>
```

The importers read these attributes and pass them through to the cooker.

### Task 7.6.4 — Cooker logic

`src/runtime/world_builder.cpp`:

For each mesh in the scene graph:
1. If `nuka:decompose = "skip"`: store as single convex piece (its convex hull).
2. If mesh is already convex (concavity ≤ threshold): single piece.
3. Otherwise: invoke V-HACD with params; store result pieces.

Pieces are uploaded to GPU memory as part of the cooked scene state.

### Task 7.6.5 — Deduplication via `.nuka` cache

Per master plan §29: `.nuka` cooked binary caches decomposition output. Same mesh + same params → reuse cached pieces, avoid re-running V-HACD (which takes 0.5-5 s per complex mesh).

Cache key: SHA-256 of (mesh bytes + decomposition params bytes).

### Task 7.6.6 — Determinism

V-HACD is deterministic at the algorithm level given fixed params. Validate by running same mesh + params 10× and checking piece counts + vertex sets are identical.

### Task 7.6.7 — Tests

`tests/import/test_vhacd_simple_cube.cpp`:

```cpp
// Box mesh → exactly 1 convex piece, vertices match input within rounding
```

`tests/import/test_vhacd_robot_finger.cpp`:

```cpp
// Load Go2 finger mesh → expect 1-3 pieces (the finger is mostly convex)
// Load H1 hand → expect 5-15 pieces (more complex)
```

`tests/import/test_vhacd_determinism.cpp`:

```cpp
// 10 runs with same mesh + params → identical piece set
```

## Validation

- V-HACD library builds cleanly on Linux + Windows.
- Simple shapes (cube, sphere, capsule) → 1 piece (or recognized as already convex).
- Complex shapes (robot finger, hand) → reasonable piece count (5-20).
- Cooker caches results in `.nuka`; re-cook uses cache.
- Determinism: same mesh + params → same output.

## Exit Criteria for v0.7 Phase 6

1. V-HACD vendored + builds.
2. `DecomposeMesh()` API operational.
3. USD / URDF / MJCF decomposition attributes recognized.
4. Cooker invokes V-HACD on flagged meshes; output uploaded to GPU.
5. `.nuka` cache prevents redundant decomposition.
6. Determinism tests pass.
7. Performance: cooking Go2 (entire robot) completes in < 30 seconds (one-time cost).

## What This Phase Does Not Do

- Does not run V-HACD on every simulation step — strictly cook-time.
- Does not generate SDFs (Phase 7).
- Does not modify runtime physics — only adds geometry data structures.
- Does not implement alternate decomposition (CoACD, etc.); V-HACD suffices.
