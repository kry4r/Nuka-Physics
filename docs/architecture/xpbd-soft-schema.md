# USD MaterialX soft-body schema (XPBD) — v0.7 p09-D

Status: **schema design + param→engine mapping function** (exit-crit 4). This
document defines the custom-attribute convention an authored asset uses to mark a
mesh as an XPBD soft body, and the deterministic cook-time mapping from those
attributes onto the engine's XPBD constraint rows.

**Scope boundary (read this first).** This is the SCHEMA + the MAPPING, realized
as the programmatic cooker `nuka::import::cooker::CookXpbdSoftBody`
(`src/import/cooker/xpbd_cooker.{hpp,cpp}`). **USD/MJCF FILE PARSING for soft
bodies is DEFERRED** to the p16 demo asset pipeline (folded into the mesh-loader
orphan, task #19), because the demo's soft-asset source format isn't settled yet.
The future importer will read these attributes off a stage, populate an
`XpbdSoftBodySpec`, and call `CookXpbdSoftBody` — i.e. this cooker is the
*programmatic target* the file importer feeds. Nothing here parses a `.usd` /
`.usda` / `.xml` file, and `usd_importer.cpp` / `mjcf_importer.cpp` are untouched.

---

## 1. Custom attributes

Authored on the `Mesh` (or its bound material) prim, in the `nuka:soft:`
namespace:

| USD attribute              | Type    | Default  | Meaning |
|----------------------------|---------|----------|---------|
| `nuka:soft:type`           | `token` | `cloth`  | Soft-body kind. One of `cloth`, `softbody`, `shape_match`. |
| `nuka:soft:stiffness`      | `float` | `1.0e4`  | Material stiffness. **Type-dependent meaning** — see §3. |
| `nuka:soft:damping`        | `float` | `0.01`   | Velocity damping. **RESERVED** — see §4. |

```usda
def Mesh "cloak"
{
    # ... points / faceVertexIndices ...
    custom token nuka:soft:type      = "cloth"
    custom float nuka:soft:stiffness = 1.0e4
    custom float nuka:soft:damping   = 0.01
}
```

`nuka:soft:type` selects which topology the mesh feeds and which engine rows the
cooker emits:

| `nuka:soft:type` | `SoftBodyType` | Mesh topology  | Emitted XPBD rows                       |
|------------------|----------------|----------------|-----------------------------------------|
| `cloth`          | `Cloth`        | triangle mesh  | distance (id6, per edge) + bend (id7, per interior edge) |
| `softbody`       | `SoftBody`     | tet mesh       | distance (id6, per tet edge) + volume (id8, per tet)     |
| `shape_match`    | `ShapeMatch`   | (point set)    | one shape-match cluster (id9) over all particles         |

---

## 2. Cook dispatch

`CookXpbdSoftBody(spec)` is a thin, deterministic, cook-time CPU function. It owns
no constraint math — it maps the params and dispatches to the existing p09-B/C
cook-time topology builders, returning a host `runtime::soft::XpbdConstraintSet`
that the caller uploads via `runtime::soft::UploadXpbdWorld`:

- `Cloth` → `runtime::soft::BuildClothConstraints(rest_positions, triangles, opts, out)`
  with `opts.distance_compliance_alpha = opts.bend_compliance_alpha = α` (§3) and
  `opts.emit_bend_constraints = spec.emit_bend`.
- `SoftBody` → `runtime::soft::BuildTetMeshConstraints(rest_positions, tets, opts, out)`
  with `opts.distance_compliance_alpha = opts.volume_compliance_alpha = α` (§3) and
  `opts.emit_distance_constraints = spec.emit_tet_edges`.
- `ShapeMatch` → one `XpbdShapeMatchCluster` assembled inline over **all**
  particles (there is no topology builder for the meshless cluster; cooking it is
  just rest-position + weight assembly). Each particle gets a uniform shape-match
  weight `1.0` (the spec carries rest geometry only, not per-particle masses; the
  standard Müller et al. 2005 default). `UploadXpbdWorld` cooks the rest centroid
  `c0` and the per-particle `q_i = x_i^0 − c0` from these rest positions.

Determinism: the builders emit constraints in a fixed sorted, de-duplicated order
and the cooker only maps/forwards, so two cooks of the same spec yield a
byte-identical `XpbdConstraintSet` (asserted in `tests/import/test_xpbd_cooker.cpp`).

---

## 3. `nuka:soft:stiffness` → engine (the mapping)

The mapping is **type-dependent**, because the elastic rows and the shape-match
cluster take physically different fields. This is the load-bearing design decision
of this schema; getting it wrong silently inverts the physics.

### Cloth / SoftBody — elastic rows (distance, bend, volume)

These rows carry an XPBD **compliance** `compliance_alpha` (= 1/stiffness; `0` ==
rigid/inextensible). So the mapping is the inverse:

```
compliance_alpha = (stiffness > 0) ? 1 / stiffness : 0      // stiffness <= 0 => rigid
```

`ComplianceAlphaFromStiffness(stiffness)` is the single source of truth.
Examples: `1e4 → 1e-4`, `100 → 0.01`, `0 → 0` (rigid), `−5 → 0` (rigid; guards
div-by-zero). The same `α` is applied to the stretch and the bend (cloth) or the
stretch and the volume (softbody) rows.

### ShapeMatch — id9 cluster

The cluster's `XpbdShapeMatchCluster::stiffness` is **NOT a compliance**. It is the
per-step **goal-pull fraction** in `[0, 1]` — the GPU id9 row applies
`p_i += w_i · stiffness · (goal_i − p_i)` each step (`s = 1` pulls fully to the
shape-matched goal; `s = 0` is no matching). Applying `1/stiffness` here would
**invert** the physics (a high authored stiffness like `1e4` would map to a
near-zero `1e-4` pull → a floppy cluster) and would even pass a naïve range check
(`1e-4 ∈ [0,1]`). So for `shape_match` the authored `nuka:soft:stiffness` is
interpreted **directly as the `[0,1]` fraction and clamped**:

```
cluster.stiffness = clamp(stiffness, 0, 1)      // <=0 => 0 (no matching); >=1 => 1 (full)
```

`ShapeMatchFractionFromStiffness(stiffness)` is the single source of truth.
Examples: `0.5 → 0.5` (NOT `1/0.5 = 2`), `2.0 → 1.0`, `1e4 → 1.0`, `0 → 0`,
`−1 → 0`.

> Authoring note: because the meaning of `nuka:soft:stiffness` differs by type,
> an asset authored as `shape_match` should set a value in `[0,1]`, whereas a
> `cloth`/`softbody` asset sets a physical stiffness (e.g. `1e4`). A future
> importer may surface this as a validation hint, but the clamp makes any value
> safe.

---

## 4. `nuka:soft:damping` → RESERVED

None of the four XPBD row structs (`XpbdDistanceConstraint`, `XpbdBendConstraint`,
`XpbdVolumeConstraint`, `XpbdShapeMatchCluster`) — nor the topology `…Options`
structs — carry a `damping_beta` field today. So `nuka:soft:damping` has **no
engine field to map onto**. The cooker parses and stores it on `XpbdSoftBodySpec`
but threads it nowhere: it is **reserved** for a future XPBD velocity-damping row
(Macklin et al. 2016 §3.4 `β`). The cooker neither drops it silently nor invents a
field for it. When the damping row lands, the mapping `damping → damping_beta`
slots in here, and this section becomes a live mapping.

---

## 5. Files

| File | Role |
|------|------|
| `src/import/cooker/xpbd_cooker.hpp` | `XpbdSoftBodySpec`, `SoftBodyType`, the two mapping fns, `CookXpbdSoftBody`. |
| `src/import/cooker/xpbd_cooker.cpp` | The thin cook-time dispatch + mapping (compiled into `nuka_runtime_gpu`, where the builders live). |
| `tests/import/test_xpbd_cooker.cpp` | Cook tests: counts (cloth quad/grid, tet, cluster), both mappings, two-cook determinism. |
| `src/runtime/soft/cloth_topology.{hpp,cpp}` | p09-B cloth stretch+bend builder (called by the cooker). |
| `src/runtime/soft/tetmesh_topology.{hpp,cpp}` | p09-B tet edge+volume builder (called by the cooker). |
| `src/runtime/soft/xpbd_world.hpp` | `XpbdConstraintSet` + the four row structs + the id9 cluster. |

## 6. Deferred (named)

- **USD/MJCF file parsing for soft bodies** → p16 demo asset pipeline / mesh-loader
  orphan (task #19). The future importer fills `XpbdSoftBodySpec` and calls
  `CookXpbdSoftBody`.
- **`nuka:soft:damping` → `damping_beta`** → blocked on a future XPBD
  velocity-damping row (no row carries the field yet). §4.
- **Per-particle shape-match weights** → today every particle gets weight `1.0`
  (rest geometry only in the spec). A future importer that knows per-particle
  masses can populate `XpbdShapeMatchCluster::cluster_mass` non-uniformly.
