# SDF world-pipeline storage/cook optimization

Date: 2026-07-14

Branch: `t2-sdf-storage`

Status: implementation verified; integration pending

## Outcome

The BDX warning was not evidence that the live PairDriven model uploaded a
1.149 GB SDF atlas.  The live cook already disables convex-piece SDFs because
its narrowphase does not consume them.  A second, host-only cook in
`CaptureArticulationHostMirror`, used only to retain articulation inertia data
for diffsim/domain randomization, called the legacy one-argument `CookScene` and
therefore baked every convex-piece SDF anyway.  Those SDFs were discarded as
soon as the mirror had read the body/joint tables.

The fix applies the live general-path stage gates to that mirror cook:

- `bake_sdf=false`;
- `general_single_hull=true`;
- `bake_link_sdf=false`.

These options affect only collision-geometry products.  The body, joint,
inertial-frame, armature and link-inertia tables consumed by
`BuildArticulationHostState` are unchanged.  The live model's explicit
`bake_link_sdf=true` remains intact and continues to supply MPM coupling.

## Measured BDX pipeline result

The comparison uses the same training-coarse scene, warm artifact cache,
`env_count=1`, GPU1, one uploaded world and one synchronized step.

| Metric | `dc7de95` baseline | fixed | Change |
|---|---:|---:|---:|
| Process pipeline wall time | 12.963 s | 1.623 s | 7.99x faster |
| Peak host RSS | 1322.5 MiB | 157.3 MiB | -88.1% |
| Dead convex-piece SDF warning | 1,148,946,648 B; 893/1402 | absent | eliminated |
| Particle rows | 2497 | 2497 | unchanged |
| CUDA allocation delta | 596 MiB | 596 MiB | unchanged |
| Readback finite | yes | yes | unchanged |

The unchanged CUDA allocation is expected: the 1.149 GB was transient host
data from the dead mirror cook, not per-env device state.  It therefore cannot
explain or repair the remaining `42.385 MiB/env` coarse slope; that slope is
dominated by the replicated MPM grid/particle state and remains a separate T2
capacity problem.

## Existing sparse/pre-baked path, measured rather than reinvented

The live link-SDF path already has the storage properties requested for this
follow-up:

1. `CookSparseSdf` stores only a narrow band (or the requested solid interior),
   with sorted 64-bit cell keys, FP32 distance and FP32 xyz gradient.
2. `ComputeSdfCacheKey` hashes transformed mesh bytes, indices and every bake
   parameter under the versioned `nuka.sparse_sdf.v1` domain.
3. `AssetCache` serializes the exact `SDF0` payload to content-addressed
   `.nukasdf` files; a hit reconstructs byte-identical `SparseSdfData`.
4. Identical meshes/parameters deduplicate within one cook, and the uploaded
   SDF tables are model-scalar fields, shared by all batch environments.

An isolated empty-cache run of the real
`SceneBuilder -> SceneIR -> cook -> upload -> readback` pipeline produced only
the three link-SDF assets actually consumed by BDX MPM coupling:

| Link-SDF cells | Cold bake | Warm load |
|---:|---:|---:|
| 172,300 + 42,213 + 171,881 = 386,394 | 1.498 s build | 0.129 s build |
| cache payload | 9,273,552 B, 3 files | +0 B, +0 files |

Warm pre-baked startup is 11.6x faster than the isolated cold bake.  The payload
is exactly `386,394 * (8-byte key + 4-byte distance + 12-byte gradient)`.

## Why no FP16/brick-atlas rewrite in this milestone

After removing the false 1.149 GB cost, the real shared device/host link-SDF
payload is 9.27 MB per model, not per environment.  A new brick page table or
FP16 distance/normal format would change every runtime sampler and contact
consumer for a best-case saving of only a few model-level megabytes in this
scene.  It would also invalidate exact-value SDF/contact gates and require a
new error budget for distance, normal, penetration and reaction force.

That trade is currently negative.  A format rewrite should be reopened only
when a real consumer demonstrates one of these conditions:

- shared SDF payload remains material after dead-bake filtering and content
  deduplication;
- random-query bandwidth/latency, rather than MPM state, is measured as a step
  bottleneck;
- multiple concurrently resident models cannot share the content-addressed
  payload at the device-allocation layer.

When reopened, benchmark dense vs current sparse vs Morton 8^3/16^3 bricks and
FP32 vs quantized payloads, including p50/p95 query latency and contact error.
Do not regenerate frozen goldens without owner approval.

## Reusable pipeline gate

`examples/demo/sdf_world_pipeline_profile.py` accepts any `.nks` scene and runs
the real load/cook/upload/step/readback sequence.  It reports build/wall time,
peak RSS, CUDA allocation, cache file/byte deltas, particle count and finite
state, with optional RSS/build-time failure thresholds.  It is a reusable
pipeline test; no local unit test was added.

## Validation

- BDX pipeline: finite state, 2497 rows, no dead-SDF warning.
- Domain-randomization host mirror: 4/4 passed.
- `set_link_mass`/diffsim host mirror: 5/5 passed, including FD and two-run D1.
- Cloth frozen cook FNV: `777503423208024307`, unchanged.
- MPM granular gates: 3/3 fresh passed; cone `28.3 deg, esc=0`, persistent
  footprint `0.0214 -> 0.0215`, cohesion `h=0.1195, esc=0`.
- Go2 owner canary: exactly `0.923080623`, unchanged (the intentionally red
  tolerance assertion retains its established semantics).
- `CoupledWorldCAbi`: 6/8 passed; the same two pre-existing coupling tests fail
  with byte-identical metrics on the baseline and fixed builds.  This change
  neither introduces nor fixes those stored reds.
