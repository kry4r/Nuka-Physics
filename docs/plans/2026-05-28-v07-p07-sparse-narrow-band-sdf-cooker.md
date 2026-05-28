# Nuka Physics v0.7 – Phase 7: Narrow-Band Sparse SDF Cooker (OpenVDB-Style)

> **Master plan reference:** §3 Round 7 (sparse SDF for thin shells) + §3 Round 12 (cooker pipeline)
> **Prerequisites:** v0.7 Phase 6 (V-HACD output is per-piece SDF input)
> **Blocks:** v0.7 Phase 8 (Newton-on-summed-SDF contact)
> **Exit criteria gate:** v0.7
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6. **SDF generation runs at cook-time and is host-orchestrated CUDA work — permitted as one-time pre-simulation.**

## Goal

Implement narrow-band sparse SDFs for collision queries. Reference: OpenVDB-style sparse VDB hierarchy (NanoVDB on GPU).

Each rigid mesh (or each V-HACD piece) gets a narrow-band SDF: a sparse 3D grid storing signed distance only in a band of voxels surrounding the surface (typically ±4 voxels). This gives 10-50× memory savings vs dense SDF while keeping fine resolution near surfaces — the right answer to the thin-shell problem.

Cooker runs the SDF generation; runtime only queries.

## Tech Stack

- CUDA 12+
- Fast sweeping method (FSM) or fast marching method (FMM) for SDF
- Mesh sampling (point-in-mesh tests; winding-number method)
- NanoVDB-inspired sparse layout (or custom equivalent)

## Files to Create

- `src/import/cooker/sparse_sdf_cooker.hpp`
- `src/import/cooker/sparse_sdf_cooker.cu`
- `src/import/cooker/winding_number_sampler.cu` — robust inside/outside for noisy meshes
- `src/import/cooker/fast_sweeping_sdf.cu` — narrow-band SDF generation
- `src/import/cooker/sparse_sdf_storage.hpp` — runtime data layout
- `src/runtime/sdf/sparse_sdf_query.cuh` — runtime sample API
- `tests/import/test_sparse_sdf_known_shapes.cpp` — sphere / box / capsule analytical comparison
- `tests/import/test_sparse_sdf_thin_shell.cpp` — thin panel preserved within resolution
- `tests/import/test_sparse_sdf_memory.cpp` — memory savings vs dense

## Files to Modify

- `src/runtime/world_builder.cpp` — invoke SDF cooker on each mesh / piece
- `src/include/nuka/nuka_cooker.h` — expose SDF resolution config

## Tasks

### Task 7.7.1 — Sparse SDF storage layout

`src/import/cooker/sparse_sdf_storage.hpp`:

```cpp
namespace nuka::cooker {

struct SparseSdfHeader {
    float3 origin;            // world-space origin of grid
    float  voxel_size;
    uint3  dims;              // full grid dimensions (most empty)
    uint32_t narrow_band_voxel_count;  // actually stored
};

// Sparse layout: hash from (i,j,k) cell → array index in narrow_band_values
struct SparseSdfDevice {
    SparseSdfHeader header;
    uint64_t* cell_keys;        // sorted; each = pack(i,j,k) into 64 bits
    uint32_t  cell_count;
    float*    cell_values;      // SDF value at narrow band cells
    // Optional: gradient buffers
    float3*   cell_gradients;
};

} // namespace
```

Layout chosen for fast GPU query: binary search by cell key → lookup value + gradient. For very dense queries, can build perfect hash to reduce lookup to O(1).

### Task 7.7.2 — Cook pipeline (per mesh / piece)

`src/import/cooker/sparse_sdf_cooker.cu`:

Algorithm:
1. Compute mesh AABB; pad by voxel-band.
2. Voxelize: for each voxel center in narrow band candidate region, sample inside/outside using **winding number** (robust to non-watertight meshes — see Jacobson et al. 2013).
3. Initialize narrow band: voxels within ±N voxels of any surface triangle get an initial distance estimate from triangle-point distance.
4. Refine via **fast sweeping**: propagate Eikonal equation `|∇φ| = 1` outward from surface.
5. Extract narrow band (mask voxels with `|φ| ≤ N × voxel_size`); compact into sparse storage.
6. Optional: precompute gradient via central difference, store per cell.

Generation runs as CUDA kernels on the cook-time host context. Output is uploaded to the `World`'s device memory and tracked as part of cooked scene state.

### Task 7.7.3 — Runtime query API

`src/runtime/sdf/sparse_sdf_query.cuh`:

```cuda
// Sample SDF + gradient at world-space point p
__device__ float sparse_sdf_sample(const SparseSdfDevice& sdf, float3 p, float3& out_grad);
```

Implementation:
1. Convert p to grid index: `idx = (p - origin) / voxel_size`.
2. Look up the cell containing p (and 8 surrounding for trilinear interp).
3. If outside narrow band: return large distance (outside coverage; signals "particle is far from this rigid").
4. Otherwise: trilinear interpolation of SDF values + central-difference gradient (or precomputed gradient).

Performance target: < 50 ns per query on RTX 4090.

### Task 7.7.4 — Memory budget

S3 envelope: 50K rigids × narrow-band SDF.

For a 100³ AABB grid: total voxels = 1M; narrow band ~ 5% × 1M = 50K voxels per mesh; storage per voxel = 4 (value) + 12 (gradient) = 16 bytes → 800 KB per mesh.

50K rigids × 800 KB = 40 GB → exceeds H100. **Must share SDFs across instances.** Cooker deduplicates by mesh hash; instances reference shared SDF + per-instance pose transform.

Realistic: ~100 unique meshes in a scene → 100 × 800 KB = 80 MB. Plus per-instance transform 7 floats × 50K = 1.4 MB. Total well within envelope.

### Task 7.7.5 — Tests

`tests/import/test_sparse_sdf_known_shapes.cpp`:

```cpp
// Sphere mesh: SDF at sample points should equal |x - center| - radius within 0.01 × voxel_size
// Box mesh: SDF agrees with analytical box SDF
// Capsule mesh: SDF agrees with analytical capsule SDF
```

`tests/import/test_sparse_sdf_thin_shell.cpp`:

```cpp
// Generate a 0.5 mm-thick panel mesh.
// With voxel_size = 0.1 mm, expect the panel correctly represented in SDF (both sides distinguished).
// Verify SDF gradient flips sign across the panel.
```

`tests/import/test_sparse_sdf_memory.cpp`:

```cpp
// Cook a Go2 mesh; verify narrow-band storage ≤ 10% of equivalent dense grid memory.
```

## Validation

- Analytical agreement for sphere / box / capsule SDF within tolerance.
- Thin-panel preservation: 0.5 mm panel resolved with 0.1 mm voxel.
- 10-50× memory savings over dense SDF for typical robot meshes.
- Query throughput meets budget.
- Determinism: same mesh + same params → same SDF cells + values bit-exact.
- `.nuka` cache reuses cooked SDFs.

## Exit Criteria for v0.7 Phase 7

1. `SparseSdfCooker` operational; produces narrow-band SDFs for arbitrary meshes.
2. Analytical agreement for primitive shapes.
3. Thin-shell representation works at fine voxel size.
4. Memory budget met.
5. `.nuka` cache integrated.
6. Runtime query throughput < 50 ns / query on RTX 4090.
7. Per-mesh sharing across instances.

## What This Phase Does Not Do

- No Newton contact (Phase 8 builds on this).
- No analytical adjoint of SDF gradient (also Phase 8).
- No CSG / boolean operations on SDFs.
- No SDF for soft body / particles (they have their own surfaces; SDF is for rigid-as-collider only).
