# Nuka Physics v0.7 – Phase 5: Particle Uniform Grid + Cross-System Broadphase Query

> **Master plan reference:** §3 Round 7 (K2+K3 hybrid coupling) + §3 Round 9 (broadphase upgrade)
> **Prerequisites:** v0.7 Phase 4 (LBVH rigid broadphase)
> **Blocks:** v0.7 Phase 8 (SDF contact needs cross-system query) + Phase 10 (PBF needs particle grid)
> **Exit criteria gate:** v0.7
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

Add the broadphase infrastructure for **particles** and **cross-system queries**.

Two deliverables:

1. **Particle uniform-grid hash** — standard PBF / XPBD broadphase. O(1) per-particle neighbor lookup; deterministic ordering for D1.
2. **Cross-system query** — given a particle, find all nearby rigid bodies (via the LBVH from Phase 4). This is the keystone of K2 SDF coupling.

These two systems together enable: particle-particle neighbor lists (PBF density, soft-soft contact) and particle-rigid candidate pairs (rigid-soft contact, rigid-fluid contact, Featherstone-link-particle contact).

## Tech Stack

- CUDA 12+
- Phase 4 LBVH infrastructure
- Spatial hashing (cell-based)

## Files to Create

- `src/collision/particle_uniform_grid.hpp`
- `src/collision/particle_uniform_grid.cu`
- `src/collision/particle_grid_traversal.cuh` — per-particle neighbor query
- `src/collision/cross_system_query.hpp` — particle cell → rigid BVH leaves
- `src/collision/cross_system_query.cu`
- `tests/collision/test_particle_grid_correctness.cpp`
- `tests/collision/test_particle_grid_determinism.cpp`
- `tests/collision/test_cross_system_query.cpp`
- `tests/perf/test_1m_particles_grid_perf.cpp`

## Files to Modify

- `src/runtime/world_stepper.cpp` — wire particle grid build into step
- `src/runtime/gpu/cuda_particle_world.cu` (existing) — replace any ad-hoc neighbor lookup with grid

## Tasks

### Task 7.5.1 — Particle uniform grid build

`src/collision/particle_uniform_grid.cu`:

Algorithm:
1. Hash each particle position to a grid cell.
2. Sort particles by cell index (stable sort → determinism).
3. Build per-cell index ranges (start, end) via segmented prefix.

```cuda
struct ParticleGridConfig {
    float3 cell_size;       // typically 2 × particle_radius
    float3 grid_min;        // world AABB lower
    uint3  grid_dims;       // cells per axis
};

__global__ void compute_cell_keys_kernel(
    uint32_t particle_count,
    const float3* __restrict__ positions,
    ParticleGridConfig cfg,
    uint32_t* __restrict__ out_cell_keys,
    uint32_t* __restrict__ out_particle_idx);

__global__ void build_cell_ranges_kernel(
    uint32_t particle_count,
    const uint32_t* __restrict__ sorted_cell_keys,
    uint32_t* __restrict__ out_cell_start,
    uint32_t* __restrict__ out_cell_end);
```

### Task 7.5.2 — Per-particle neighbor query

`src/collision/particle_grid_traversal.cuh`:

```cuda
// Iterate neighbors in 27 surrounding cells; return up to k closest
__device__ uint32_t query_particle_neighbors(
    float3 p, float radius,
    const ParticleGridConfig& cfg,
    const uint32_t* cell_start, const uint32_t* cell_end,
    const uint32_t* particle_idx_sorted,
    const float3* particle_positions,
    uint32_t* out_neighbors, uint32_t max_count);
```

Output: list of neighbor indices. Stable order within the function (sorted by neighbor index ascending) — supports D1.

### Task 7.5.3 — Cross-system query (particle → rigid)

`src/collision/cross_system_query.cu`:

Algorithm:
1. For each particle, compute the world AABB expanded by query radius.
2. Traverse the rigid LBVH (from Phase 4) for AABB-overlap query → list of candidate rigid bodies.
3. Output: per-particle list of candidate rigid body indices.

```cuda
__global__ void cross_system_particle_to_rigid_kernel(
    uint32_t particle_count,
    const float3* particle_positions,
    float query_radius,
    const LbvhNode* rigid_lbvh_nodes,
    const float3* rigid_aabb_min, const float3* rigid_aabb_max,
    uint32_t* out_candidate_offsets,
    uint32_t* out_candidate_rigid_idx);
```

Output is CSR-like: per-particle offset + flat list of candidates. Used by Phase 8 (SDF contact) to know which rigid SDFs to query for each particle.

### Task 7.5.4 — Determinism handling

Both particle grid and cross-system query produce **sorted output** to preserve D1. Even though kernels execute in non-deterministic warp order, the final output is sorted by stable criterion (cell index, particle index, rigid index).

Test: same scene + same particle positions → bit-exact same neighbor lists across runs.

### Task 7.5.5 — Memory budget

S3 envelope: 1M particles. Grid memory:
- ~1M cells (assuming average cell occupancy ~1) → ~16 MB cell arrays.
- 1M sorted particle indices → 4 MB.
- 1M cell keys → 4 MB.
- Cross-system output: 1M × avg-30-candidates = 30M ints = 120 MB. **This is large.** Mitigate via cap: limit candidates per particle to 16 (sufficient for SDF queries; document the bound in V1 oracle tests).

### Task 7.5.6 — Tests

`tests/collision/test_particle_grid_correctness.cpp`:

```cpp
// 100 particles in known positions; query neighbors at radius r; verify against brute force O(N²)
```

`tests/collision/test_particle_grid_determinism.cpp`:

```cpp
// Build twice; result bit-exact
```

`tests/collision/test_cross_system_query.cpp`:

```cpp
// 10 rigid + 1000 particles; verify particle→rigid candidate set matches brute-force AABB test
```

`tests/perf/test_1m_particles_grid_perf.cpp`:

```cpp
// 1M particles; build + query budget < 800 µs on RTX 4090
```

## Validation

- Brute-force vs grid-based neighbor lists agree bit-exact on small scenes.
- Determinism: bit-exact across runs.
- 1M particle grid build + query within performance budget.
- Cross-system candidate list bounded by 16 per particle; oracle scenes within bound.
- No float atomics; lint clean.

## Exit Criteria for v0.7 Phase 5

1. `ParticleUniformGrid` operational; build + query passes correctness.
2. Cross-system particle→rigid query produces correct candidate lists.
3. Determinism tests pass.
4. Performance test for 1M particles meets budget.
5. Memory usage within S3 envelope.

## What This Phase Does Not Do

- No SDF queries yet (Phase 8). Phase 5 only finds candidates; Phase 8 evaluates SDF at candidate locations.
- No PBF density (Phase 10).
- No XPBD contact rows (Phase 9).
- No deformable surface broadphase beyond what particle grid provides.
