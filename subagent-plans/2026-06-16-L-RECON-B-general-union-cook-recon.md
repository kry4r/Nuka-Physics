# L-RECON-B — tractable general cook for whole-body union scenes (H1)

Recon by the controller (the dispatched read-only agent stalled on infra). Source-grounded.

## The finding
`CookToModel(scene, env, {PairDriven})` on the whole-body H1 union scene
(`h1_cup_table.nks`: 51-DOF H1 + 30 fingertips + a 5388-vert cup hull + a table) is
**intractable (>178 s, never completes)** while the specialized
`CookSceneToUnionTemplate` (`union_cook.cpp`) cooks the same scene near-instantly. The
landing (L1) deletes the specialized union path, so the general cook must become
tractable first.

## Call graph + cost (anchors)
- `CookToModel` (`src/scene/cook/cook_to_model.cpp:204`) → `CookScene(scene)`
  (`src/scene/cooker.cpp:287`). `CookScene` does, per shape:
  1. **V-HACD convex decomposition** — `ToCookerMode(s.decompose_mode)`
     (`cooker.cpp:392`); `DecomposeMode::Skip` skips it (`:394`), else Auto/Force run
     V-HACD (content-hash cached in `.nuka_cache`, but a COLD cook on H1's meshes is
     the cost). Default per-shape mode = Auto.
  2. **Sparse-SDF bake** per unique convex piece (`cooker.cpp:91-146`,
     `CookSdfTable` → `backend.Bake`). A 5388-vert cup hull + every decomposed piece
     each get a narrow-band SDF voxelization. Expensive and content-addressed-cached
     but cold on first cook.

## Why the union cook is fast — the "dodge" (anchor)
`CookH1` (`union_cook.cpp:118-131`) deep-copies the scene then **clears
`mesh_vertices`/`mesh_indices` on every shape** before `CookScene` → no mesh → no
V-HACD ("inertia + topology only — dodge V-HACD", `:124`). The cup hull is supplied
PRE-AUTHORED (`GraspConfig.cup_hull_verts`, `:340`), not decomposed.

## THE KEY FACT: the general path does NOT consume the SDF
`OpNarrowphaseSdf` (`src/phi/backend_cuda/ops/narrowphase_sdf.cu:188-191`) is a
**deterministic no-op for every family** (`if (p->family != kContactFamilyPairDriven)
return Ok; return Ok;`). The general PairDriven narrowphase is cvx-only:
- heightfield → per-cell TRIANGLE_PRISM via cvx (`narrowphase_heightfield.cu`),
- body↔body → cvx GJK/EPA on convex hulls/primitives (`narrowphase_prims.cu`),
- fingertip↔cup → `cvx::SphereHull` on the cup's convex hull.
None reads a sparse SDF. **So the SDF bake in `CookScene` is wasted work for the
general path**, and V-HACD's N-piece decomposition is unnecessary — the cvx
narrowphase wants ONE convex hull per mesh (or the primitive directly; primitives
never decompose, `cook_to_model.cpp:548`).

## The fix (general, no entity-type special-casing)
Make the general (PairDriven) cook do what the cvx narrowphase actually needs:
1. **Skip the SDF bake** when cooking for the general path (it is a no-op consumer).
   Cleanest: a `CookScene` option / `CookToModelOptions` flag `bake_sdf=false` for
   PairDriven (or gate `CookSdfTable` on "any registered narrowphase consumes SDF",
   which for the general path is none).
2. **Single convex hull per collision mesh instead of V-HACD** for the general path:
   set each mesh shape's `decompose_mode = Skip`-equivalent that still produces ONE
   convex hull (the convex hull of the mesh verts) for the cvx support function —
   NOT an empty collision (the union dodge clears verts because it supplies the hull
   separately; the general cook must instead cook the hull). A genuinely-concave
   shape can opt INTO decomposition per-shape (rare); the default general behavior is
   one-hull-per-mesh. This is O(meshes) hull builds vs O(meshes) V-HACD+SDF.

This stays general: the predicate is "mesh has a convex collision rep → use it;
primitive → use it; only decompose a shape that explicitly asks" — no "union"/"H1"/
"grasp" branch.

## Open item before implementing — PROFILE the split
The >178s was not broken down. Add a one-shot timing probe (cook H1 union with (a)
SDF-off, (b) decompose Skip+single-hull, (c) both) to confirm which dominates (likely
the SDF bake on the 5388-vert cup hull + any decomposed H1 pieces, plus V-HACD on
concave H1 body meshes). Implement the fix that the profile shows is load-bearing;
both (1)+(2) are correct regardless.

## D1 / risk
- go2 cook is UNAFFECTED (go2 has primitive feet + simple body, never decomposes;
  no concave meshes) → the go2 golden (FUSED + the general path) does not move.
- The `UnionCookGolden` tests cook via the SPECIALIZED path (unchanged) → they hold
  until L1 deletes that path; at L1 the general cook of H1 becomes the new reference
  (owner-approved re-baseline) and must be validated for cvx-grasp correctness.
- Single-hull vs V-HACD changes the COLLISION SHAPE of any concave mesh (loses
  concavity). For H1 body parts this is acceptable (they are convex-ish / primitives);
  flag any shape where it matters.

## Priority note
L-RECON-B is required for L1 (delete UnionCsr → H1 must cook on the general path). It
is NOT on the critical path for the go2 multi-dog stairs demo (go2 cooks fast
already; the throughput bench cooked N=1024 go2 in seconds). Sequence accordingly.
