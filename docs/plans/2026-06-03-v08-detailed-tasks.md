# v0.8 Detailed Task Design — Unified Collision/Contact/Coupling + Cross-Cutting Foundations

> **Status:** controller-authored task decomposition (2026-06-03), for owner + advisor review. Decomposes the RATIFIED architecture in `docs/plans/2026-06-03-v08-unified-collision-contact.md` (Q1–Q11, FIXED) and the v0.8 scope in `docs/plans/2026-06-03-post-v07-roadmap.md` §3/§5/§6. The Q1–Q11 decisions are NOT re-litigated here — this document turns them into granular, committable phase-tasks (one task per commit, p-task granularity).
> **Research grounding:** `docs/plans/research/2026-06-03-research-{newton-mujocowarp,genesis,breadth-solvers}.md` (cited inline with the source repo file:line, e.g. `mujoco_warp/_src/constraint.py:81–117`).
> **Scope of THIS doc:** the 12 v0.8 phases — C1, C2, C3, C4, C5, C6, C7, R1, G2, G3, G4, U1. C8+ (v0.9 breadth solvers) is explicitly OUT; this doc defines the spine v0.9 consumes.

---

## 0. SHARED-INTERFACE SPEC (the v0.8 spine — every later phase + v0.9 depends on these)

These are the load-bearing contracts. v0.9 breadth solvers (cable, MPM, FEM, garment cloth) plug into the spine through exactly these interfaces, so they are defined CONCRETELY here. Each is owned by a specific C-phase (noted), but the *contract* is frozen up front so downstream tasks can be authored against it.

### 0.1 `ContactManifold` — the unified detection→solver handoff (owned by C3; extends `src/constraint/contact_manifold.hpp`)

The current struct (`src/constraint/contact_manifold.hpp:21`) carries only `uint32 body_a/body_b`, a scalar `friction`, scalar `restitution`, and `ContactPoint points[4]` (position/normal/penetration + 3 warm-start impulses). It is rigid-maximal-only: no collidable type, no reaction-provider tag, no per-point compliant params. **Decision: EXTEND, not replace** (the field layout below is additive; `body_a/body_b` keep their meaning as the per-side collidable handle for the rigid case). The new struct:

```cpp
// src/constraint/collidable.hpp  (NEW, C1)
enum class CollidableType : uint8_t {   // the collidable-type registry's enum half
    RigidBody       = 0,   // maximal rigid body  (reaction = invM tensor)
    ArticulationLink= 1,   // articulated link    (reaction = chain Jacobian)
    Particle        = 2,   // XPBD/PBF/MPM/FEM vert(reaction = scalar invM)
    StaticWorld     = 3,   // immovable (ground/static collider; invM = 0)
    // v0.9 seam: CableSegment=4, ClothTriangle=5, MpmParticle=6, FemTet=7 …
};
enum class ReactionProviderKind : uint8_t {
    RigidInvMass = 0, ArticulationChainJ = 1, ParticleInvMass = 2, StaticNull = 3,
};
struct CollidableRef {                  // one side of a manifold
    CollidableType        type   = CollidableType::RigidBody;
    ReactionProviderKind  react  = ReactionProviderKind::RigidInvMass;
    uint32_t              handle = ~0u;  // body id / global link id / particle id
};

// src/constraint/contact_manifold.hpp  (EXTENDED, C3)
struct ContactPoint {                    // EXTENDED with compliant params + ids
    math::Vec3 position;
    math::Vec3 normal;                   // separation dir for side A (existing convention)
    float penetration        = 0.0f;     // >0 == overlap depth
    float normal_impulse     = 0.0f;     // warm start (existing)
    float friction_impulse_1 = 0.0f;
    float friction_impulse_2 = 0.0f;
    uint64_t stable_key      = 0u;       // NEW: D1 contact identity (see 0.6)
    // compliant params (resolved per-point from merged material; see 0.2):
    float solref_timeconst   = 0.02f;    // solref[0]
    float solref_dampratio   = 1.0f;     // solref[1]
};
struct ContactManifold {
    CollidableRef a;                     // NEW: was uint32 body_a
    CollidableRef b;                     // NEW: was uint32 body_b
    uint32_t point_count = 0;
    float friction       = 0.5f;         // merged (elementwise-max), isotropic μ
    float restitution    = 0.0f;
    float solimp[5]      = {0.9f,0.95f,0.001f,0.5f,2.0f}; // dmin,dmax,width,mid,power
    uint64_t manifold_key= 0u;           // NEW: (min,max handle)+type for D1 sort
    static constexpr uint32_t kMaxPoints = 4;   // KEEP 4: D1-friendly + matches MuJoCo face-clip ≤4
    ContactPoint points[kMaxPoints];
    void AddPoint(const ContactPoint& pt);
    void Clear();
};
```
**OPEN-A — RESOLVED (owner 2026-06-04): rename.** `body_a`/`body_b` are referenced by name in `row_builder.cpp:162` and `world_stepper.cpp`. Extending replaces them with `a.handle`/`b.handle`; the 4 call sites get a mechanical rename (clarity over a `body_a()` shim that hides the type). See Appendix D OPEN-A.

### 0.2 solref/solimp storage + the contact row regularizer (owned by C1 storage / C4 consumption)

**Storage decision (C1) — REFINED in C1a (advisor-confirmed 2026-06-04): contact params are PER-SHAPE, not per-material.** The MuJoCo contact params (solref/solimp/condim/priority/solmix/margin/gap) live as NEW fields on `CollisionShapeRecord` (`scene_ir.hpp:29`) — matching MuJoCo's per-geom semantics — alongside the contype/conaffinity/group filtering bitmasks. The earlier "fields on `MaterialRecord`" wording was lossy: (1) MuJoCo's `mj_contactParam` merge is per-GEOM (C1b reads per-geom, C1c merges per-geom); (2) robot-MJCF collision geoms usually carry NO material, so per-material storage has nowhere to put their params. **Per-shape strictly generalizes per-material**, so this honors owner Q8 ("per-material μ") by keeping `MaterialRecord.friction_mu` as the per-material *default source*: the cooker resolves the per-shape μ with precedence `shape-override (friction_mu≥0) → material.friction_mu → MuJoCo default 1.0` and stores the RESOLVED value. A new `CookedContactParamTable` (one row per shape, parallel to the shape table) holds the cooked params, NOT on the Row. Row-emission (C4) computes the derived `(k,b,D,R=1/D)` per `_efc_row` and writes DERIVED values into the Row.

```cpp
// scene_ir.hpp  CollisionShapeRecord  (NEW per-shape fields, C1a — LANDED)
uint32_t contype       = 1;       // collision bitmask (filtering, C2)
uint32_t conaffinity   = 1;
int32_t  collision_group = 0;
float    solref[2]     = {0.02f, 1.0f};                     // timeconst, dampratio
float    solimp[5]     = {0.9f, 0.95f, 0.001f, 0.5f, 2.0f}; // dmin,dmax,width,mid,power
float    friction_mu   = -1.0f;   // SENTINEL: <0 ⇒ inherit material μ; ≥0 ⇒ per-shape override
int32_t  priority      = 0;       // mj per-contact merge priority
float    solmix        = 1.0f;    // mj solmix weight
float    margin        = 0.0f;
float    gap           = 0.0f;
uint8_t  condim        = 3;       // {1,3,4,6}; v0.8 supports 1 & 3
// scene_ir.hpp  MaterialRecord  (NEW field, C1a — LANDED): per-material μ DEFAULT source
float    friction_mu   = 1.0f;    // resolved into the per-shape cooked μ at cook time
```

**Solver-regularizer decision (C4) — requires a solver consume-path change, NOT a new Row field (VERIFIED):** I read `src/solver/gpu/row_solver.cu`. The production maximal-contact PGS path consumes ONLY `baumgarte` (position bias at `row_solver.cu:310`, `SolveConstraints` config at `world_stepper.cpp:678`). `compliance_alpha`/`damping_beta`/`contact_softness` (`row.hpp:28-35`) are consumed ONLY by the XPBD path (`xpbd_world.cu:126` as `α/dt²`), NOT by rigid contact. So compliant solref/solimp contact CANNOT ride the existing fields without a solver change. **Decision:** reuse `compliance_alpha` to carry the dual regularizer `R = 1/D` (semantics: rigid-contact PGS adds `α̃ = R` to the row's effective-mass denominator exactly as XPBD adds `α/dt²`), and reuse `damping_beta`/`rhs` to carry the spring-damper reference accel `aref = −k·imp·pos_aref − b·vel`. This UNIFIES rigid contact and XPBD compliance through one denominator term (Q10's literal glue) and needs NO new Row field — only a new consume-path in `row_solver.cu`'s contact branch. **OPEN-B:** if reusing `compliance_alpha` for two different physical quantities (XPBD α vs contact R) proves confusing, add an explicit `float regularizer_R = 0.0f` field to `Row` (16→17 fields; codegen YAML `Row` struct + `row_class_registry.hpp` regen). Recommend reuse first; promote to a field only if a test shows aliasing bugs.

### 0.3 Row-class ID allocation — the v0.9 collision-avoidance constraint (owned by C4/C6)

Current registry (`row_class_registry.hpp:21`) ends at **id10** (`kRowClassCount = 11`). The v0.9 breadth research (`research-breadth-solvers.md:38,98`) ALREADY RESERVED **id11 = `XpbdCosseratStretchShear`, id12 = `XpbdCosseratBendTwist`**. v0.8's new row classes therefore **MUST NOT** silently take 11/12. **Decision:**
- **Compliant contact = NO new row class.** Extend id0 (`MaximalContactRow`) and id3 (`FeatherstoneContactRow`) with the solref/solimp regularizer semantics from 0.2; ids 4/5 (SDF contact) already exist and gain the same path. This keeps the contact tier within existing classes (a forward-kernel change, not a new class), so the v0.9 cable ids 11/12 stay free.
- **Coupling-row framework (C6) = NEW row class id13** `CouplingContactRow` (a generic two-system non-penetration/friction row whose two `CollidableRef`s may differ in type; the K2/K3 generalization). id11/12 are RESERVED for v0.9 Cosserat → coupling takes **id13**, NOT 11. Update the YAML + regen so `kRowClassCount` becomes 14 with 11/12 as reserved placeholders (or document them as reserved and let v0.9 fill them; **OPEN-C** below).

**OPEN-C — RESOLVED (owner 2026-06-04): (i) reserve.** Add id11/12 as reserved placeholders now (`kRowClassCount=14`) and make id13 the coupling row, so v0.9 plugs Cosserat into 11/12 with zero renumber. Costs two no-op registry entries now. See Appendix D OPEN-C.

### 0.4 Collidable-type registry (owned by C2; codegen-time enum + runtime vtable-free dispatch)

A registration table mapping `CollidableType` → its 4 providers. Following the codegen-YAML pattern that produces `row_class_registry.hpp`, the registry is **codegen-time for the enum + static metadata** (so dispatch tables are `constexpr` and D1) and **runtime for the device function pointers** (resolved once at world build, never per-step):

```cpp
// src/collision/collidable_registry.hpp  (NEW, C2)
struct CollidableTypeInfo {
    CollidableType type;
    // AABB provider:    fills a world AABB per instance (for broadphase)
    void (*compute_aabbs)(const CollidableSystemView&, collision::AABB* out);
    // accel-structure:  which per-system structure indexes this type
    AccelStructureKind accel;          // LbvhRigid | UniformGridParticle | None(static)
    // shape/proxy:      narrowphase geometry accessor (shape type + transform)
    ShapeProxyKind proxy;
    // reaction provider: how an impulse maps back to DOFs (0.5)
    ReactionProviderKind react;
};
const CollidableTypeInfo& GetCollidableTypeInfo(CollidableType);  // O(1) table lookup
```
v0.8 registers RigidBody / ArticulationLink / Particle / StaticWorld. v0.9 adds CableSegment/ClothTriangle/MpmParticle/FemTet by APPENDING a `CollidableTypeInfo` — no enum hardcoded across dispatch (the extensibility-seam mandate, Q9 / architecture §3).

### 0.5 Reaction providers (owned by C4) — impulse → DOF delta

One interface, three v0.8 implementations, matching the three existing solver math paths:

```cpp
// src/constraint/reaction_provider.hpp  (NEW, C4)
// Given a unit-impulse along a row Jacobian on collidable `ref`, return the
// effective inverse mass (J M^-1 J^T) and apply a solved impulse to the DOFs.
struct ReactionProvider {
    // RigidInvMass:       invM = m^-1 I3 (+ angular I^-1) — maximal 6-vec (row_solver.cu path)
    // ArticulationChainJ: invM = J M^-1 J^T via ComputeContactEffectiveMass
    //                     (articulation_contacts.hpp:108 — REUSE) ; chain Jacobian
    //                     from ComputeContactChainJacobians
    // ParticleInvMass:    invM = 1/particle_mass (xpbd_world.cu / pbf path)
    float    effective_inv_mass(const CollidableRef&, const RowJacobian6&) const;
    void     apply_impulse(const CollidableRef&, const RowJacobian6&, float dlambda) const;
};
```
This is the concrete object that makes TWO-WAY reaction work (C5): a single contact row between an articulation link and a rigid body uses `ArticulationChainJ` for side A and `RigidInvMass` for side B; the solver applies equal-and-opposite impulses to both via their providers. This subsumes the foot-ground path (`articulation_contacts`), whose reaction body is currently NULL (impulse to articulation DOFs only — architecture §0(b)).

### 0.6 Solver entry — Newton's `step(state_in, state_out, control, contacts, dt)` contract (owned by C5)

ADOPT Newton's solver-registry contract VERBATIM (`newton/_src/solvers/solver.py:300`): **Contacts (here: the assembled row stream incl. contact rows) is a first-class argument passed INTO the solve**, alongside state (q/qd) + control. This is the "solver consumes rows, never shape pairs" separation (architecture §2).

```cpp
// src/solver/unified_solve.hpp  (NEW, C5) — the registry contract
struct SolveContext {                    // ≈ Newton State+Control+Contacts
    RowBuffers*            rows;          // ALL row classes: joint+drive+contact+coupling
    BodyStateView          state_in;      // q, qd (rigid + articulation + particle)
    BodyStateView          state_out;     // double-buffered out
    ControlView            control;       // actuation
    const ReactionProvider* reactions;    // per-collidable-type (0.5)
    float dt;
};
void UnifiedSolve(const SolveContext&, const SolverConfig&);
```
**DELIBERATE DIVERGENCE from Newton (keep):** Newton picks ONE solver per step (no cross-solver row mixing). Nuka co-solves joint + drive + contact + coupling rows TOGETHER in one PGS (`research-newton-mujocowarp.md:13` — "keep our single co-solve … as our deliberate strengthening"). The registry SLOT (for SolverKamino later, §3 extension seam) consumes the SAME row stream.

### 0.7 Coupling-row framework (owned by C6) — Genesis `_func_collide_in_rigid_geom` as a ROW

From Genesis `legacy_coupler.py:284–347` (`research-genesis.md:27-35`): rigid carries the SDF, the deformable queries it, relative velocity is projected (normal restitution + Coulomb friction), and an equal-and-opposite reaction goes to the rigid link. Genesis applies this as an EXPLICIT force `−Δmv/dt` (conditionally stable). **Nuka makes it a solver-resolved CONSTRAINT ROW** (id13, 0.3) so it stays implicit/stable, and replaces Genesis's `influence = exp(−d/ε)` soft gate with the solref/solimp compliant schedule (0.2). The framework interface:

```cpp
// src/runtime/coupling/coupling_row_framework.hpp  (NEW, C6)
// Emits CouplingContactRow(s) for a candidate pair whose two sides may be ANY
// two CollidableTypes (the cross-system generalization that kills Genesis's
// O(N^2) static pair whitelist, research-genesis.md:96).
struct CouplingPair { CollidableRef a, b; uint64_t stable_key; };
void EmitCouplingRows(std::span<const CouplingPair>, const ContactManifold* manifolds,
                      const ReactionProvider*, RowBuffers* out);
```
Co-step ordering (Genesis `simulator.py`, `research-genesis.md:18,99`): `preprocess → pre_coupling (each system advances) → couple (emit + solve coupling rows) → post_coupling`. v0.8 ships the FRAMEWORK + the co-step bridge; SPECIFIC pairs (rigid↔MPM, cloth↔rigid) are v0.9 (R8).

---

## C1 — Filtering & contact metadata foundation

Bakes MuJoCo-parity collision filtering + contact compliance params into the IR → importer → cooked blob → a precomputed filtered system-pair matrix. Per `research-newton-mujocowarp.md:103` (MJ bakes the filtered pair set at model build, `collision_driver.py:797`), filtering is a COOK-TIME set-difference, not a per-step branch.

### C1a — SceneIR + cooked-blob contact-metadata fields
- **Objective.** Add contype/conaffinity/group + solref/solimp + per-material μ + condim/priority/solmix/margin/gap to the IR and cook them.
- **Technical approach.** Add to `CollisionShapeRecord` (`scene_ir.hpp:29`): `uint32 contype = 1`, `uint32 conaffinity = 1`, `int32 collision_group = 0`. Add the 0.2 contact-param fields **to `CollisionShapeRecord` (per-shape, the refined storage decision — see 0.2)**; add a `float friction_mu = 1.0f` per-material default to `MaterialRecord`. Add an `<exclude>` body-pair list to `SceneIR` (new `std::vector<std::pair<BodyId,BodyId>> exclude_pairs_` + `AddExcludePair`, canonicalized (min,max)). New `CookedContactParamTable` in `cooked_blob.hpp` (parallel-to-shape SoA: `contypes, conaffinities, groups, solref0, solref1, solimp[5], frictions, condims, priorities, solmix, margins, gaps`). Extend the `cooker.cpp` shape-cook loop to fill the table **in lockstep with the shape rows** (folded into `PushShapeRow` so a V-HACD-decomposed mesh's N pieces each inherit the parent geom's contact metadata — a separate loop would silently misalign on decomposed meshes); friction is RESOLVED per the 0.2 precedence at cook time. **LANDED (C1a): `CollisionShapeRecord`/`MaterialRecord`/`CookedContactParamTable`/`SceneIR::AddExcludePair` + `tests/scene/test_contact_metadata_cook.cpp` (9 tests, 47/47 scene suite green, cook-twice D1, lint 0).** **NAMED carry-over → C1c:** `scene_compose.cpp` does NOT yet propagate `exclude_pairs_` across a compose (relates to USD↔MJCF coexistence) — C1c owns it.
- **Inputs/Outputs/Interface.** In: SceneIR. Out: CookedBlob with the new table. Interface: read by C1c (matrix), C2 (filtering), C4 (compliance).
- **Dependencies.** None (pure data plumbing).
- **D1 strategy.** Pure host data copy in fixed shape/material order; no FP, no atomics. Bit-exact by construction.
- **Validation/test gates.** `tests/scene/test_contact_metadata_cook.cpp` — round-trip a hand IR with contype/conaffinity/solref/solimp, assert cooked values byte-equal. Cook-twice memcmp gate.
- **Effort.** S.
- **Risks.** Defaults must match MuJoCo (contype=conaffinity=1, solref=(0.02,1), solimp=(0.9,0.95,0.001,0.5,2.0), condim=3) or imported scenes silently mis-collide.

### C1b — MJCF importer parse (contype/conaffinity/solref/solimp/condim/priority + `<exclude>`/`<pair>`)
- **Objective.** Parse the MuJoCo collision/contact attributes the importer currently drops (`mjcf_importer.cpp:362` parses geom type/pos/quat/size/material/mesh ONLY — verified no contype).
- **Technical approach.** In the geom loop (`mjcf_importer.cpp:362-415`) read `contype`, `conaffinity`, `condim`, `group`, `priority`, `solref` (2 floats), `solimp` (5 floats), `friction` (take [0] as isotropic μ), `solmix`, `margin`, `gap`, `solreffriction`. Parse `<contact><exclude body1= body2=>` → `SceneIR::AddExcludePair`; parse `<contact><pair geom1= geom2= …>` → an explicit-pair override list. Apply MuJoCo defaults from `<default><geom contype=… conaffinity=… solref=… solimp=… friction=…>` class inheritance. **This is NEW work folded into C1b (not a verify-and-maybe):** VERIFIED the importer has only `ApplyJointDefault` + `ApplyGeneralDefault` (`mjcf_importer.cpp:161,178`) — there is NO `ApplyGeomDefault`, so geom `<default>`-class inheritance does not exist and must be added here. Wire to C1a fields.
- **Inputs/Outputs/Interface.** In: MJCF XML. Out: SceneIR with metadata. Interface: feeds C1a cook.
- **Dependencies.** C1a (fields must exist).
- **D1 strategy.** Deterministic XML walk (tinyxml2, fixed sibling order); host-only.
- **Validation/test gates.** `tests/import/test_mjcf_contact_filtering.cpp` — parse an MJCF with contype masks + `<exclude>` + a `<pair>` override; assert parsed values. Use an h1_with_hand snippet (memory: fingers are contype=0 visual-only) to confirm contype=0 is read.
- **Effort.** M (realistic-to-tight: the new geom `<default>`-class inheritance — no existing `ApplyGeomDefault` — is the part that pushes M toward its upper bound).
- **Risks.** `<default>`-class inheritance of contype/solref is easy to miss → wrong masks. MuJoCo friction is a vector (condim−1 entries); v0.8 takes only the tangential μ — document the drop.

### C1c — Filtered system-pair matrix at cook time (the MuJoCo precedence)
- **Objective.** Precompute, at cook time, the candidate set-difference per `research-newton-mujocowarp.md:78,104`: **`<pair>` override → `<exclude>` → parent-child auto-exclude → bitmask `(contype_i & conaffinity_j) || (contype_j & conaffinity_i)`** (the EXACT MuJoCo precedence). Plus the system-pair enable matrix (rigid/articulated/soft/cloth/fluid).
- **Technical approach.** New `cooker.cpp` pass `BuildFilteredPairPolicy`: emit (a) a per-shape-pair allow/deny derived from the precedence chain, stored compactly (a sorted list of explicitly-EXCLUDED canonical pairs + the bitmask arrays for the bitmask test at broadphase), and (b) a `SystemPairMatrix` (5×5 bool: which CollidableType pairs are enabled). Implement the per-contact param MERGE rule VERBATIM from `engine_collision_driver.c` `mj_contactParam` L2850–2930 (`research-newton-mujocowarp.md:79-89`): higher `priority` geom supplies all params; equal priority → `condim=max`, `friction[i]=max`, solmix-weighted solref when both `solref[0]>0` else elementwise `min`, solimp solmix-mixed, `margin/gap=max`. Store the merged per-pair params for explicit pairs.
- **Inputs/Outputs/Interface.** In: C1a cooked metadata + exclude/pair lists. Out: `CookedFilterPolicy` (bitmask arrays + sorted exclude list + merged explicit-pair params + 5×5 system matrix). Interface: C2 consumes the bitmask + exclude list at broadphase; C4 consumes merged params.
- **Dependencies.** C1a, C1b.
- **D1 strategy.** Sorted exclude list (canonical (min,max), `std::sort`); deterministic merge math (host, fixed order). The bitmask test is a pure integer AND at broadphase (order-independent).
- **Validation/test gates.** `tests/scene/test_filter_precedence.cpp` — assert `<pair>` overrides a 0-bitmask; `<exclude>` removes a colliding pair; parent-child auto-excluded; merge produces max-friction/solmix-solref. Cross-check 2-3 cases against MuJoCo's documented behavior (hand-computed).
- **Effort.** M.
- **Risks.** The precedence order is subtle (pair BEATS exclude? — MuJoCo: `<pair>` always generates even if bitmask rejects, but `<exclude>` removes a body pair entirely; resolve the pair-vs-exclude interaction explicitly). The solmix/min-on-direct-solref branch is error-prone.

## C2 — Unified broadphase dispatcher (cross-system candidate-pair MATRIX)

Generalize the per-system structures (LBVH rigid `broadphase_lbvh.hpp` + uniform grid particles `particle_uniform_grid.hpp` + `cross_system_query.hpp`) into a cross-system query MATRIX producing ONE filtered, D1-sorted candidate-pair stream. Topology Q9-B (`v08-unified-collision-contact.md:22`): per-system optimal structures, unify the OUTPUT stream not the geometry. Mirrors MJ's pluggable broadphase + staged filter (`collision_driver.py:286`).

> **D1-gate note (applies to every per-task `N≥32 cross-replica` + two-run byte-exact gate below).** These gates REUSE existing test idioms — `tests/runtime/test_batched_articulation_replication.cpp` (`NReplicasStepBitIdenticalToSingleEnv`) for the N≥32 cross-replica check and the `D1TwoRunByteExact` pattern already used across `tests/collision/` + `tests/runtime/` (e.g. `test_cross_system_query.cpp`, `test_lbvh_vs_sap_pair_set.cpp`). No separate harness-building task is needed; each task instantiates these idioms on its own buffers.

> **★ C2 RE-SCOPE (controller + advisor, 2026-06-04 — applies to C2b/C2c; faithful to Q9-B, NOT a deviation).** Code-reality check (Explore + advisor) found: (1) the production broadphase is already **SHAPE-level** (per-shape AABBs; `CollisionPair` carries shape indices, narrowphase maps shape→body) → C1c's per-shape bitmask fits natively, NO body-aggregate needed. (2) maximal rigid bodies (`WorldInstance`/CPU `world_stepper`) and articulation links (`BatchedArticulatedWorld`/GPU) are **TWO SEPARATE, non-co-resident worlds**, and there is **no per-link AABB** today. So the C2b prose "ONE union LBVH over rigid+links" was a MIS-READING — **Q9-B ratified "unify the OUTPUT candidate-pair stream, NOT the geometry"** (arch §2 / Q9 = per-system structures + cross-query dispatcher). Therefore:
> - **C2b = RIGID↔RIGID only** via LBVH self-traversal (shape-level) + C1c bitmask/exclude/parent-child cull → emit into the C2a `CandidatePairStream`. **VALIDATE** the pair SET + canonical order against the existing broadphase (reuse `test_lbvh_vs_sap_pair_set`); **DO NOT flip the production path** (build-validate-defer; consumers C3 narrowphase / C5 closer; the production rigid path is CPU `DynamicBroadphase` so the real "flip" is the CPU→GPU migration that rides with C5, exactly as p04 LBVH was validated-not-forced). The "PRODUCTION-BROADPHASE FLIP" wording in C2b below is SUPERSEDED by this note.
> - **articulation-link↔rigid = a CROSS-QUERY** (like particle↔rigid), NOT a world-merge. Its only real prerequisite is **link-AABB computation** (does not exist). Deferred; see the LONG-POLE below.
> - **★ LONG-POLE (named, blocks C7 — NOT C2b/C2c; the v0.7-integration-debt trap if left vague):** the grasp (C7) needs a **movable rigid body + an articulation link CO-RESIDENT in one steppable world with two-way reaction** — that world does NOT exist (the two worlds are separate). **World co-residence + per-link AABB computation** is real, unscoped work the plan implicitly assumed free across C2/C4c/C5b. Best-fit home = a **prerequisite step of C5b** (two-way reaction), with **per-link AABB as its own concrete step**. SIZE it + confirm owner at the C2 batch boundary (a TELL: "spine effort is back-loaded onto world co-residence for the grasp"). C2c's articulation-link-as-cross-query-source ALSO needs the link-AABB step → confirm at C2c whether C2c or the C5b-prereq owns it; do NOT let C2c grow a half-version.

### C2a — Unified candidate-pair stream type + collidable-type registry
- **Objective.** Define the registry (0.4) and the unified candidate-pair output type that every cross-query path writes into.
- **Technical approach.** Implement `collidable_registry.hpp` (0.4) with v0.8's 4 types registered. New `CandidatePair { CollidableRef a, b; uint64 stable_key; }` and `CandidatePairStream` (CSR-like device buffer, like `CrossSystemQueryResult` in `cross_system_query.hpp:42` — two-pass count→scan→private-slice fill, NO append atomics). `stable_key = pack(min(typeA,handleA),(typeB,handleB))` canonical-ordered.
- **Inputs/Outputs/Interface.** In: per-system views. Out: the stream type + registry. Interface: every C2b–C2e query writes `CandidatePair`s here.
- **Dependencies.** C1 (CollidableRef/CollidableType from 0.1, though the enum can land in C1a).
- **D1 strategy.** stable_key packing is deterministic; the stream is built by count→scan→fill (the `cross_system_query.cu` precedent), never an append atomic.
- **Validation/test gates.** `tests/collision/test_candidate_stream.cpp` — fill from a fixed pair set, assert sorted-by-stable-key, 2-run memcmp.
- **Effort.** S–M.
- **Risks.** stable_key bit-packing must not collide across types (reserve high bits for type).

### C2b — Rigid↔rigid + articulated-link↔rigid via the retained LBVH
- **Objective.** Produce filtered rigid/articulated candidate pairs from the LBVH self-traversal.
- **Technical approach.** Use `BuildLbvhForQuery` (`broadphase_lbvh.hpp:126`, retains nodes) over the UNION of rigid + articulated-link world AABBs (articulation link AABBs via `UpdateWorldLinkPoses` + per-shape AABB). The LBVH already emits canonical (a<b) sorted pairs (`broadphase_lbvh.hpp:21-27`). Add the C1c bitmask + exclude filter as a post-traversal cull (integer AND test + binary-search the sorted exclude list) BEFORE writing the `CandidatePair`. Tag each side's CollidableType (rigid vs articulation-link) from a cook-time per-leaf type array.
- **Inputs/Outputs/Interface.** In: union AABBs + LBVH + C1c filter. Out: rigid/articulated `CandidatePair`s. Interface: C3 narrowphase.
- **Dependencies.** C1c, C2a.
- **D1 strategy.** LBVH is already D1 (`stable_sort`, uint32 atomics only, final sort — `broadphase_lbvh.hpp:24-27`). Filter is a pure integer test. No new nondeterminism.
- **Validation/test gates.** `tests/collision/test_lbvh_filtered_pairs.cpp` — assert filtered set = unfiltered minus excluded; reuse `test_lbvh_vs_sap_pair_set.cpp` harness for the unfiltered set equality.
- **Effort.** M.
- **Risks.** Articulation links move every step → AABB refit each frame (LBVH refit exists: `lbvh_refit.cu`). Mixing rigid + articulated into one tree needs a unified leaf→(type,handle) map. **PRODUCTION-BROADPHASE FLIP (supersedes a p04 decision):** p04 deliberately kept the legacy O(n²) SAP (`src/collision/gpu/broadphase.cu`) as the SOLE production broadphase "so rigid-body goldens stay untouched" (`broadphase_lbvh.hpp:8-15`); LBVH was test-only. C2b makes LBVH the PRODUCTION contact broadphase — this reverses that protection. **Order analysis (checked):** SAP enumerates an `n(n-1)/2` upper-triangular slot layout (`broadphase.cu:230`), so its active-pair download is ALREADY in canonical `(body_a<body_b)` slot order — the SAME canonical order the LBVH pair list is sorted into (`broadphase_lbvh.hpp:21-27`). Same SET (guaranteed by `test_lbvh_vs_sap_pair_set.cpp`) AND same canonical order ⇒ the graph-colored-PGS row-ordering risk is mitigated for the rigid-maximal path. Nonetheless C5's compliant (solref/solimp) contact formulation already re-baselines contact dynamics independently of the broadphase source — see §C (re-baseline note).

### C2c — Particle↔rigid cross-query (generalize cross_system_query) + particle↔particle
- **Objective.** Generalize `QueryParticlesAgainstRigidLbvh` (`cross_system_query.hpp:86`) into the matrix; add particle↔particle via the grid.
- **Technical approach.** `cross_system_query` already does particle→rigid-LBVH (CSR, 16-cap, D1 — `cross_system_query.hpp:14-22`). Reframe its output as `CandidatePair`s (particle side = `CollidableType::Particle`, rigid side from LBVH leaf). Particle↔particle reuses `particle_uniform_grid` neighbor CSR (`particle_uniform_grid.hpp`) — already the K3 path (`coupling/particle_particle_contact`). Apply the C1c SYSTEM matrix (e.g. fluid↔fluid on, fluid↔cloth per matrix).
- **Inputs/Outputs/Interface.** In: particle positions, rigid LBVH, grid. Out: `CandidatePair`s. Interface: C3 (SDF tier for particle↔rigid; particle-particle row for p↔p).
- **Dependencies.** C2a, C2b (rigid LBVH).
- **D1 strategy.** Both paths are already D1 (count→scan→fill, 16/32-cap LOWEST-index deterministic subset, integer truncation counter only — `cross_system_query.hpp:18`, `particle_particle_contact.hpp`).
- **Validation/test gates.** `tests/collision/test_cross_system_matrix.cpp` — assert particle→rigid candidate set unchanged from p05 baseline; p↔p set matches K3. 2-run + N≥32 cross-replica identity.
- **Effort.** M.
- **Risks.** The 16-candidate cap (`kCrossSystemMaxCandidates`) may truncate in dense grasp contact; oracle must assert truncation==0 or raise the cap.

## C3 — Unified narrowphase + multi-point ContactManifold

The hybrid narrowphase (Q2): analytical primitives (multi-point/face-face) + convex GJK/EPA/SAT + face-clip + the SDF high-precision tier (now WIRED). Static `(typeA × typeB × tier) → handler` dispatch table, the MJ `MJ_COLLISION_TABLE` / `mjCOLLISIONFUNC` pattern (`research-newton-mujocowarp.md:42,102`).

### C3a — Extend ContactManifold + narrowphase dispatch table
- **Objective.** Land the extended `ContactManifold` (0.1) and the pluggable dispatch table.
- **Technical approach.** Extend `contact_manifold.hpp` per 0.1 (CollidableRef sides, stable_key, per-point solref, manifold solimp). New `narrowphase_dispatch.hpp`: `using NarrowphaseFn = void(*)(const CandidatePair&, const ShapeProxyView&, ContactManifold* out);` and a `constexpr` table keyed by `(ShapeType a, ShapeType b, NarrowphaseTier)` → fn, with `NarrowphaseTier ∈ {Analytical, Convex, Sdf}`. Pick tier from cooked geometry: primitive×primitive→Analytical; mesh/convex→Convex; SDF-equipped pieces→Sdf (high-precision). New entries = new table rows (extensibility seam §3).
- **Inputs/Outputs/Interface.** In: `CandidatePair` stream + shape proxies. Out: `ContactManifold` stream. Interface: C4 row emission.
- **Dependencies.** C2 (candidate stream), C1 (merged friction/solimp onto the manifold).
- **D1 strategy.** Table is `constexpr` (no runtime dispatch nondeterminism). One thread per candidate writes its own manifold slot (slot-indexed, the `sdf_contact.cu:280` precedent), no atomics.
- **Validation/test gates.** `tests/collision/test_narrowphase_dispatch.cpp` — assert each (type,type,tier) routes to the right handler; manifold byte-stable.
- **Effort.** M.
- **Risks.** The migration of `world_stepper.cpp`'s ad-hoc `GenerateContact` (`:307`, IsPlane/IsSphere/IsBoxLike branching) into the table must preserve current goldens until C5 re-baselines — keep both paths until C5.

### C3b — Analytical multi-point / face-face primitives
- **Objective.** Upgrade the existing single-point analytical contacts to multi-point/face-face (resting + grasp stability, Q2).
- **Technical approach.** Port + extend `world_stepper.cpp` primitives (`GeneratePlaneContact:196`, `GenerateSphereSphere:225`, `GenerateSphereBox:246`, `GenerateBoxBox:268`) into device narrowphase handlers that emit up to 4 points: box-vs-plane → 4 deepest corners; box-vs-box → SAT axis + face-clip (Sutherland-Hodgman polygon clip, ≤4 retained by deepest-penetration reduction); capsule→segment-vs-face. Sphere stays single-point. Fill per-point penetration/position; normal = separation dir for A (existing convention `row_builder.cpp:88`).
- **Inputs/Outputs/Interface.** In: primitive shape pairs. Out: multi-point manifolds. Interface: C3a table.
- **Dependencies.** C3a.
- **D1 strategy.** Fixed clip-vertex order, fixed-order deepest-4 reduction (sort the ≤8 clip points by a stable key, take 4), `__fadd_rn`. No data-dependent branch that reorders FP.
- **Validation/test gates.** `tests/collision/test_analytical_manifold.cpp` — box-on-plane yields 4 coplanar points; box-box face-face yields a stable 4-point patch; resting box does not jitter. Compare point SET (order-insensitive) to a hand oracle.
- **Effort.** M–L.
- **Risks.** Face-clip 4-point reduction is the classic determinism trap (tie-breaks on equal penetration). Stable key tie-break mandatory.

### C3c — Convex narrowphase (GJK/EPA/SAT + face-clip)
- **Objective.** General convex×convex / convex×primitive contact (mesh collidable via the general tier, Q2).
- **Technical approach.** Self-written GJK (Minkowski support, `support_function` over cooked convex hull vertices `CookedConvexGeometry`) → EPA for penetration depth/normal → face-clip for the multi-point manifold (incident/reference face, Sutherland-Hodgman, ≤4). Consume `CookedConvexGeometry` (`cooked_blob.hpp:59`, already exists from p06 V-HACD). Reference SAT for box-box fast path (already in C3b). Newton uses GJK+MPR (`research-newton-mujocowarp.md:17`); we use GJK+EPA+face-clip (deterministic, no MPR origin-ray randomness).
- **Inputs/Outputs/Interface.** In: convex hull pairs. Out: multi-point manifolds. Interface: C3a table.
- **Dependencies.** C3a, C3b (shares the face-clip util).
- **D1 strategy.** Fixed GJK iteration cap, fixed simplex-vertex order, deterministic EPA expansion (fixed-order priority by stable edge key, NOT a float-keyed heap that reorders on ties), fixed-iteration. The known GJK/EPA D1 hazard is the expanding-polytope priority queue — use a fixed-order array scan, not a nondeterministic heap.
- **Validation/test gates.** `tests/collision/test_gjk_epa_convex.cpp` — two tetrahedra / two boxes vs analytic; penetration depth within tol; 2-run bit-identity; N≥32 cross-replica.
- **Effort.** L.
- **Risks.** GJK/EPA numerical robustness near-degenerate (parallel faces, deep penetration). EPA termination + D1 priority ordering is the hardest determinism item in C3.

### C3d — Wire the SDF high-precision tier
- **Objective.** Wire `find_sdf_contact_newton` (`sdf_contact.hpp:136`, currently ZERO stepper callsites — architecture §0(c)) as the Sdf tier handler.
- **Technical approach.** The SDF narrowphase already exists, is D1 (fixed 32-iter, fixed tol, step clamp — `sdf_contact.hpp:95-99`), HD, and differentiable-ready (p08-C handoff `sdf_contact.hpp:101-117`). Add a dispatch-table handler that, for SDF-equipped pieces (cooked `CookedSdfTable`, `cooked_blob.hpp:83`), builds `SdfContactCandidate` (`sdf_contact.hpp:267`) from the `CandidatePair` (initial guess = midpoint of the pair AABBs), calls `find_sdf_contact_kernel` (`sdf_contact.cu`), and converts `SdfContactResult` → `ContactManifold` (single witness point → 1-point manifold; multi-point via perturbed restarts is a v0.9 refinement). Tag the manifold side reaction providers. This is the Q5 differentiable tier (rides p08 adjoint).
- **Inputs/Outputs/Interface.** In: SDF-equipped candidate pairs + `CookedSdfTable`. Out: 1-point high-precision manifolds. Interface: C3a table (Sdf tier).
- **Dependencies.** C3a, C2 (candidate stream).
- **D1 strategy.** Already D1 (`sdf_contact.hpp:25-34`). Slot-indexed kernel, no atomics. The IFT-at-convergence adjoint (gradient_mode=2, `row_class_registry.hpp:81`) is preserved.
- **Validation/test gates.** `tests/collision/test_sdf_tier_wired.cpp` — box-on-ground via SDF matches the analytical normal/penetration within tol (the `sdf_contact.hpp:236` verification); the existing SDF accuracy/D1 tests still pass; assert ≥1 stepper callsite now exists.
- **Effort.** M.
- **Risks.** Single-witness SDF gives 1 point → resting stability weaker than the analytical 4-point; acceptable for the high-precision/grasp tier where SDF is used for fingertip precision, multi-point handled by the analytical tier on the same body where applicable. Document the cardinality gap.

## C4 — Manifold→rows: compliant contact + pyramid friction + reaction providers

The glue (Q10): port `_efc_row` solref/solimp VERBATIM, linearized-pyramid isotropic friction (MJ combine), and the three reaction providers.

### C4a — Port MuJoCo `_efc_row` solref/solimp (the verbatim compliance math)
- **Objective.** Compute `(k, b, imp, D, R=1/D, aref)` from solref/solimp EXACTLY per `mujoco_warp/_src/constraint.py:81–117` (`research-newton-mujocowarp.md:52-73`).
- **Technical approach.** New `src/constraint/solref_solimp.hpp` (HD, header-only like `sdf_contact.hpp` so host tests + device kernel share one path):
```
timeconst=solref[0]; dampratio=solref[1]; dmin,dmax,width,mid,power=solimp;
if REFSAFE: timeconst = max(timeconst, 2*dt);            // clamp
dmin,dmax=clamp(.,MINIMP,MAXIMP); width=max(MINVAL,width); mid=clamp(..); power=max(1,power);
k = 1/(dmax^2 * timeconst^2 * dampratio^2);  b = 2/(dmax*timeconst);   // standard (solref[0]>0)
if solref[0]<=0: k = -solref[0]/dmax^2;       if solref[1]<=0: b = -solref[1]/dmax;  // DIRECT
x = |pos_imp|/width; a = (1/mid^(power-1))*x^power; bb = 1-(1/(1-mid)^(power-1))*(1-x)^power;
imp_y = (x<mid)?a:bb;  imp = clamp(dmin+imp_y*(dmax-dmin), dmin, dmax); if x>1: imp=dmax;
D = 1/max(invweight*(1-imp)/imp, MINVAL);   R = 1/D;     // R is the DUAL regularizer
aref = -k*imp*pos_aref - b*vel;
```
Output a `CompliantContactRow { float R, aref_bias; }` per contact point. Per 0.2: write `R` into `Row.compliance_alpha` (added to the effective-mass denominator) and `aref` into `Row.rhs`. Include `power==1 ⇒ linear`. **Do NOT swap D and R** (`research-newton-mujocowarp.md:73`) — the formula outputs D; add `1/D` to A.
- **Inputs/Outputs/Interface.** In: merged solref/solimp (C1c), penetration (pos_imp), normal velocity, invweight (from reaction provider). Out: `(R, aref)`. Interface: C4b emission.
- **Dependencies.** C1c (merged params), C3 (manifold).
- **D1 strategy.** Pure arithmetic, fixed order, `__fadd_rn`/HD; pow via fixed `expf/logf` (deterministic on the GPU for a fixed power; or integer-power unroll for common power=2). No branches that reorder FP.
- **Validation/test gates.** `tests/constraint/test_solref_solimp.cpp` — table of (solref,solimp,pos,vel) → (k,b,imp,D,R,aref) cross-checked against hand-evaluated MuJoCo formula values (incl. the negative-convention + power=1 + REFSAFE-clamp cases). HOST + device parity (call the HD fn on both).
- **Effort.** M.
- **Risks.** `pow(x,power)` and `pow(mid,power-1)` determinism across host/device (g++ vs nvcc libm). Mitigate: use the same `__powf`-free formulation or restrict to integer powers + document. This is the single most error-prone item (`research-newton-mujocowarp.md:100`).

### C4b — Row emission from manifold (compliant normal + pyramid friction)
- **Objective.** Turn a `ContactManifold` into contact rows with compliant normals + linearized-pyramid isotropic friction.
- **Technical approach.** Replace `AppendContactGroup` (`row_builder.cpp:45`) to: per point emit a normal row (id0/id3, Unilateral) with `compliance_alpha=R`, `rhs=aref` (from C4a); emit `2*(condim−1)` friction rows for the pyramid (`research-newton-mujocowarp.md:94` `nmaxpyramid = max(1,2*(condim-1))`; condim=3 → 4 edges, vs the current 2 tangent rows `row_builder.cpp:93`). Friction μ = elementwise-max combine (C1c). Tangent basis via `ChooseTangent` (`row_builder.cpp:15`, deterministic). Set CollidableRef sides on the rows so C5's reaction providers know each side's type.
- **Inputs/Outputs/Interface.** In: extended manifold + C4a compliance. Out: row stream. Interface: C5 solver.
- **Dependencies.** C4a, C3, 0.5 (reaction provider tags).
- **D1 strategy.** Fixed emission order (points in manifold order, friction edges in fixed angular order). No atomics.
- **Validation/test gates.** `tests/constraint/test_compliant_rows.cpp` — a penetrating box emits 4 normal + 16 pyramid rows with expected R/aref; friction-pyramid edges sum to the isotropic disk within tol. 2-run.
- **Effort.** M.
- **Risks.** Pyramid (4 edges) vs the current 2-tangent path changes solver behavior → contributes to the C5 golden re-baseline. condim=4/6 (torsional/rolling) explicitly deferred (Q8 v1.0 seam).

### C4c — Reaction providers (rigid invM / articulation chain-J / particle invM)
- **Objective.** Implement the three `ReactionProvider`s (0.5) so a row's effective mass + impulse application is type-correct.
- **Technical approach.** RigidInvMass: `invM = m^-1 I + I_body^-1` (the existing `row_solver.cu` maximal path, `EffectiveMass` at `:154`). ArticulationChainJ: reuse `ComputeContactEffectiveMass` (`articulation_contacts.hpp:108`, `m_eff = 1/(J M^-1 J^T)`) + `ComputeContactChainJacobians`. ParticleInvMass: `1/particle_mass` (the xpbd/pbf path, `xpbd_world.cu`). Provide `effective_inv_mass` + `apply_impulse` per type, resolved once at world build (function-pointer table, 0.4).
- **Inputs/Outputs/Interface.** In: CollidableRef + RowJacobian6. Out: effective inv-mass; impulse→DOF apply. Interface: C5 unified solve.
- **Dependencies.** C4b, existing articulation/xpbd infra.
- **D1 strategy.** Each provider reuses an already-D1 path (CRBA M-inverse is fixed-order LDLᵀ `articulation_contacts.hpp:79-85`; rigid invM is closed-form; particle is scalar). No new nondeterminism.
- **Validation/test gates.** `tests/constraint/test_reaction_providers.cpp` — unit impulse on a 1-DOF pendulum link vs hand-computed `J M^-1 J^T`; rigid invM vs analytic; particle invM scalar. 2-run.
- **Effort.** M.
- **Risks.** The articulation chain-Jacobian must be evaluated at the CONTACT point (not the link origin); reuse the foot-ground construction but generalize the contact point from foot-sphere to arbitrary manifold point.

## C5 — Solver wiring + TWO-WAY reaction

All row classes through ONE solve (0.6); articulated-link↔rigid two-way (the grasp crux); subsume foot-ground; re-baseline goldens.

### C5a — UnifiedSolve entry consuming the shared row stream
- **Objective.** Implement `UnifiedSolve` (0.6) routing joint+drive+contact+coupling rows through one graph-colored PGS, with per-row reaction via providers.
- **Technical approach.** Extend `SolveConstraints` (`world_stepper.cpp:678`, `row_solver.cu`) to: (1) accept the `SolveContext` (0.6); (2) in the contact branch consume `compliance_alpha=R` as the dual regularizer added to the denominator (`row_solver.cu:154` `EffectiveMass`) and `rhs=aref` as the bias (replacing/augmenting the Baumgarte term at `:310` for compliant rows — keep Baumgarte for hard-contact compliance→0 special case); (3) dispatch `apply_impulse` through the per-side `ReactionProvider`. Newton's contract (`solver.py:300`) — contacts passed in. Keep the co-solve (joint+drive+contact together) — the deliberate divergence (0.6).
- **Inputs/Outputs/Interface.** In: SolveContext (all rows + providers). Out: updated state. Interface: world_stepper / batched world.
- **Dependencies.** C4 (rows + providers + compliance).
- **D1 strategy.** Graph-colored PGS is already D1 (fixed colors, fixed iteration count `SolverConfig.velocity_iterations`). The regularizer is per-row deterministic. NO new atomics. Two-run bit-identity preserved.
- **Validation/test gates.** `tests/solver/test_unified_solve.cpp` — a stack of boxes settles; compliant contact matches a stiffness oracle; 2-run + N≥32 cross-replica.
- **Effort.** M.
- **Risks.** Mixing the compliant regularizer with Baumgarte double-counts position correction unless gated by compliance→0. Define the hard/soft switch explicitly.

### C5b — Articulated-link↔rigid two-way reaction (the grasp crux)
- **Objective.** A single contact row whose side A is an articulation link and side B is a movable rigid body applies equal-and-opposite impulses to BOTH (the grasp requirement — architecture §0(b) notes this is currently MISSING).
- **Technical approach.** With C4c providers, a mixed-type row uses ArticulationChainJ for A and RigidInvMass for B; `effective_inv_mass = J_A M^-1 J_A^T + invM_B` (sum of both sides); the solved `dlambda` is applied to the articulation DOFs (via chain-J) AND the rigid body (via invM) — the two-way step. Manifold from C3 (analytical or SDF tier for the fingertip).
- **Inputs/Outputs/Interface.** In: mixed-type manifold + both providers. Out: two-way impulse. Interface: C7 grasp.
- **Dependencies.** C5a, C4c.
- **D1 strategy.** Both providers are D1; the summed effective mass is fixed-order. No atomics.
- **Validation/test gates.** `tests/runtime/test_link_rigid_twoway.cpp` — a 1-DOF arm pushes a free box: box accelerates AND arm feels reaction torque; momentum balance within tol. 2-run.
- **Effort.** M–L.
- **Risks.** This is the highest-value, highest-risk task (grasp depends on it). The chain-J at an arbitrary fingertip contact point + the rigid reaction must be sign-consistent (normal convention `sdf_contact.hpp:230`).

### C5c — Subsume foot-ground + re-baseline goldens
- **Objective.** Route the GPU foot-ground contact (`articulation_contacts` / `batched_articulated_world`, currently FootShape×scalar ground, +Z hardcoded, NO reaction body — architecture §0(b)) through the unified pipeline; regenerate `go2_stand_5s.bin`.
- **Technical approach.** Replace `DetectFootGroundContacts` (`articulation_contacts.hpp:62`) + `AssembleArticulatedContactRows` (`batched_articulated_world.hpp:23`) with: feet become `ArticulationLink` collidables, ground becomes a `StaticWorld` plane collider in the scene (Q3 migration item 2), contacts flow C2→C3(analytical plane tier)→C4→C5a. The ground gains a real `StaticWorld` reaction provider (invM=0, identical dynamics to the old NULL-reaction case → goldens move only by the compliant-vs-Baumgarte formulation change). Regenerate `go2_stand_5s.bin` (see the re-baseline note, Appendix C).
- **Inputs/Outputs/Interface.** In: go2/h1 scenes + ground collider. Out: unified foot contacts. Interface: standing demo.
- **Dependencies.** C5a, C5b, C3b (plane tier).
- **D1 strategy.** Per-env fixed-order detection preserved (`articulation_contacts.hpp:28-30`). New golden regenerated under the unified path; 2-run + N≥32 on the NEW golden.
- **Validation/test gates.** `tests/oracle/test_go2_stand` (re-baselined): go2 stands 5s without sinking/jitter; compare standing height to physical expectation. Standing golden regenerated + 2-run bit-identity on the new path.
- **Effort.** L.
- **Risks.** Golden re-baseline is owner-approved (Q3 "re-baseline goldens OK") but must be CORRECT (visually stable stand), not merely self-consistent. The +Z-only ground generalizes to a real plane normal — verify go2 still stands.

## C6 — Full migration onto the unified pipeline + coupling-row framework

Migrate XPBD soft/cloth (p09) and PBF fluid (p10) to CONSUME unified manifolds via co-step; ship the coupling-row framework (0.7) + co-step bridge as FOUNDATION (specific pairs → v0.9; un-defer the K2 framework).

### C6a — XPBD soft/cloth + PBF fluid consume unified manifolds (co-step)
- **Objective.** The PBD systems read relevant manifolds from the unified narrowphase instead of their private contact paths (Q3 migration items 4,5).
- **Technical approach.** XPBD (`xpbd_world.cu`) and PBF (`pbf_world.cu`) particles register as `Particle` collidables (C2a); their particle↔rigid / particle↔particle candidate pairs (C2c) → manifolds (C3, SDF tier for particle↔rigid per Genesis "rigid carries SDF, deformables query it" `research-genesis.md:39,97`) → coupling rows (C6b). The co-step ordering (Genesis `simulator.py`, `research-genesis.md:99`): `pre_coupling` (XPBD predict / PBF density solve) → `couple` (emit + solve coupling rows) → `post_coupling` (XPBD correct). Replace the K3 private path (`coupling/particle_particle_contact`) with the unified emission (it already uses row id10 — keep id10 for p↔p, add id13 for cross-system).
- **Inputs/Outputs/Interface.** In: unified manifolds. Out: PBD systems coupled via rows. Interface: co-step bridge.
- **Dependencies.** C3, C4, C5, C6b.
- **D1 strategy.** Gather-not-scatter, `__fadd_rn`, no float atomics (the binding rule `research-breadth-solvers.md:11`; the K3 precedent `particle_particle_contact.hpp:36-48`). Per-system substep budget allowed (do NOT force one global dt — `research-genesis.md:100`).
- **Validation/test gates.** `tests/runtime/test_pbd_costep_unified.cpp` — cloth drapes on a box via unified manifolds; fluid rests in a container; V2 energy non-increasing; 2-run + N≥32.
- **Effort.** L.
- **Risks.** Co-step ordering bugs (correct-before-couple) blow up energy. The K3→unified migration must keep the existing fluid/cloth goldens (or re-baseline with justification).

### C6b — Coupling-row framework + co-step bridge (id13 CouplingContactRow)
- **Objective.** Ship the generic coupling-row framework (0.7) — the K2 framework un-deferred — so any (systemA,systemB) pair gets coupling rows without hand-written O(N²) kernels (the win over Genesis's static whitelist, `research-genesis.md:96`).
- **Technical approach.** New row class id13 `CouplingContactRow` (codegen YAML + regen; 0.3). `EmitCouplingRows` (0.7) builds a unilateral compliant non-penetration row + friction rows for a cross-type `CouplingPair`, using both sides' reaction providers (C4c). The SDF-field-in / impulse-field-out handshake (Newton `SolverImplicitMPM`, `research-newton-mujocowarp.md:28,108`) becomes: read rigid SDF (C3d) → project → emit row → solver applies equal-and-opposite via providers (Genesis `_func_collide_in_rigid_geom` as a ROW, `research-genesis.md:95`). Ship the FRAMEWORK + K2 particle↔rigid-SDF as the proving pair; rigid↔MPM, cloth↔rigid = v0.9 (R8).
- **Inputs/Outputs/Interface.** In: cross-type `CandidatePair`s + manifolds + providers. Out: id13 rows. Interface: v0.9 R8 (each new pair = a registry entry, not a new kernel).
- **Dependencies.** C4c, C5a, 0.3 (id13), 0.4 (registry).
- **D1 strategy.** Solver-resolved row (implicit, not explicit −Δmv/dt — fixes Genesis's conditional stability, `research-genesis.md:35,89`). Gather reductions, no float atomics.
- **Validation/test gates.** `tests/runtime/test_coupling_framework.cpp` — K2: a fluid particle pushes a free rigid box (two-way momentum balance); the box reacts. 2-run + N≥32. Assert id13 registered + `kRowClassCount` correct.
- **Effort.** L.
- **Risks.** This is the foundational v0.9 contract — get the interface right or every v0.9 coupling pair pays. OPEN-C (id allocation) must be resolved first.

## C7 — H1 grasp-place demo (validation, the former #16)

> **OPEN-J RESOLVED (owner, 2026-06-04): the REAL usdc-binary newton-assets cup is MANDATORY** (not a usda/primitive placeholder). The minimal usdc parse needed to load that cup is pulled into v0.8 as **C7a** (scoped to THIS cup, NOT a general usdc reader — general composition-arc hardening stays v0.9 U4a). **Critical-path risk (advisor):** C7 — the flagship grasp gate — now blocks on an **L-tier self-written binary parser (C7a)**, which sits on the C7 critical path. Schedule slack on C7a; if the cup asset turns out to exercise composition arcs, C7a escalates toward U4a scope (flag early). This is a deliberate version-inversion the owner accepted to ship the real asset.

### C7a — Scoped usdc cup reader (newton-assets cup only)
- **Objective.** Parse exactly the usdc-binary (crate) sections the newton-assets `manipulation_objects/cup` exercises, emitting the cup mesh into SceneIR — the minimal real-usdc capability the C7 grasp needs (OPEN-J). NOT general usdc support.
- **Technical approach.** Self-written usdc (crate) binary reader (no OpenUSD — the no-closed-SDK pillar) on the `usd_stage_adapter` seam (memory `v07-usd-mjcf-coexistence`: "self-written reader, no OpenUSD"). Scope to the crate structural sections + `points`/`faceVertexIndices`/xform the cup asset actually uses; reuse the #19 mesh path + the #32 USD-mesh physics-less relaxation. Anything the cup does NOT exercise (general references/payloads/sublayers/variants) is OUT — that is the v0.9 U4a hardening item. Output the same SceneIR both importers produce → `scene_compose` (#31) composes onto the table.
- **Inputs/Outputs/Interface.** In: the newton-assets cup usdc. Out: SceneIR with the cup TriMesh. Interface: C7b grasp scene; v0.9 U4a extends this to general usdc.
- **Dependencies.** #19 mesh loader + #32 USD-mesh (landed v0.7); scene_compose (#31, exists). **Independent of the C1–C6 contact spine** (importer work — can land in parallel with the spine).
- **D1 strategy.** Deterministic parse (fixed traversal order); host-only cook; cook-twice memcmp.
- **Validation/test gates.** `tests/import/test_usdc_cup.cpp` — the real newton-assets cup usdc parses to the expected mesh (vertex/index counts + bbox within tol); cook-twice byte-equal.
- **Effort.** **L** (usdc binary/crate format is involved even scoped to one asset; the C7 critical-path risk lives here — see the OPEN-J note above).
- **Risks.** usdc crate format is versioned + nontrivial; the "load-this-cup-only" scope discipline is the mitigation. If the cup uses composition arcs this escalates toward U4a scope.

### C7b — H1 hand grasps cup, places on table (on the unified system)
- **Objective.** The re-homed grasp gate as VALIDATION (not a gate): H1 hand grasps a cup → places on a table, on the unified pipeline (architecture §0, roadmap §3).
- **Technical approach.** Load h1_with_hand (MJCF, memory: fingers contype=0 visual-only → C1b reads that; the demo must ENABLE + cook fingertip collision per the honesty finding in memory `newton-assets-resource`). Cup = the REAL usdc newton-assets cup loaded via **C7a** (OPEN-J resolved) on a table; coexistence via `scene_compose` (#31). Fingertips = `ArticulationLink` collidables; cup = `RigidBody`; grasp contact = SDF high-precision tier (C3d) for fingertip precision + analytical for the table. Two-way reaction (C5b) holds the cup. Optional R1 IK to pre-position the hand. Run on the batched articulated world.
- **Inputs/Outputs/Interface.** In: h1_with_hand + cup (via C7a) + table scene. Out: a grasp-place trajectory + RT video.
- **Dependencies.** ALL of C1–C6 + **C7a (the real cup asset)**; R1 (optional, for reach); G2 (for the photoreal video) optional.
- **D1 strategy.** The whole pipeline is D1; the demo trajectory is bit-reproducible (2-run). RT video via the G1 two-level tracer (`two_level_render.hpp`).
- **Validation/test gates.** Regression (architecture §0 / roadmap §3): **no-interpenetration** (assert min separation ≥ −slop), **hold** (cup does not slip out over N steps), **D1** (2-run bit-identity of the trajectory), **V2 energy** (no spurious energy injection). RT video rendered on the G1 tracer for the homepage.
- **Effort.** L.
- **Risks.** Fingertip collision enablement (the honesty finding) + grasp stability (multi-point manifold + friction pyramid + two-way reaction all must work together). This is the integration capstone; failures here diagnose C3–C5. Plus the C7a asset-load critical-path risk (OPEN-J note).

## R1 — Inverse Kinematics (Featherstone-Jacobian DLS / Levenberg-Marquardt)

Per-frame IK for the manipulation demos (roadmap §W1 R1). NOT a constraint-row solver — runs BEFORE the dynamics step, emitting joint targets the existing drive rows (id2) track (`research-breadth-solvers.md:25`). Zero new physics row classes.

### R1a — Batched damped-least-squares IK from the Featherstone task Jacobian
- **Objective.** Solve `q` for end-effector pose objectives via DLS / Levenberg-Marquardt, batched per env.
- **Technical approach.** Build the task Jacobian `J(q) ∈ ℝ^{6m×n}` from the existing Featherstone Jacobians (`articulation_jacobian.cu` / `.hpp` — REUSE). Per iteration solve `(JᵀJ + λI)Δ = −Jᵀr` (LM = adaptive-λ DLS, Newton `IKOptimizerLM` `ik_lm_optimizer.py:83`: `lambda_initial=0.1, lambda_factor=2.0, lambda_min=1e-5, lambda_max=1e10`; λ↑ on rejected, λ↓ on accepted — trust-region). The small dense `(JᵀJ+λI)` (n≈30–50 for H1) is a per-env batched dense Cholesky/LDLᵀ fitting a thread-block (`research-breadth-solvers.md:23`). Objective list API (Newton `newton.ik`, `research-newton-mujocowarp.md:23`): `IKObjectivePosition`/`IKObjectiveRotation`/`IKObjectiveJointLimit`. ANALYTIC Jacobian (we have Featherstone J); AUTODIFF deferred.
- **Inputs/Outputs/Interface.** In: target end-effector pose(s), current q. Out: target q → drive rows (id2, `row_builder.cpp:276`). Interface: C7 grasp pre-positioning.
- **Dependencies.** Featherstone Jacobians (exist). Independent of C1–C6 (can land in parallel).
- **D1 strategy.** Trivially D1 (`research-breadth-solvers.md:24`): fixed iteration count, fixed pivot order in the small dense factorization, no atomics, fixed-order `__fadd_rn` (reuse the CG math posture `sparse_solver_cg.cu`).
- **Validation/test gates.** `tests/runtime/test_ik_dls.cpp` — H1 hand reaches a target within tol in fixed iterations; singular-config (straight arm) does not blow up (λ damping); 2-run + N≥32 cross-replica.
- **Effort.** S–M (DLS = S; multi-target/task-priority QP = M, deferred — Pink-style task-space QP is a v0.9+ seam, `research-breadth-solvers.md:20`).
- **Risks.** Per-block dense LDLᵀ for n≈50 may exceed shared memory; tile or fall back. Rotation-objective parameterization (quaternion log) is a determinism trap (branch-free).

## G2 — Photoreal path tracer (PBR + shadows + AO/soft-GI + denoise + AA + tonemap)

Offline/progressive path tracer (A1 tier) over the G1 two-level tracer (`src/rt/two_level_render.{hpp,cu}`). The current `RenderFrame` (`two_level_render.hpp:108`) does single-ray direct shade (Lambert+GGX, `shading.cuh`, ONE light + flat ambient, hard shadow by caller). The framebuffer already carries 6 AOVs (color/depth/normal/albedo/uv/prim_id, `framebuffer.hpp:36-41`). Self-written, D1, no OptiX (roadmap §6; memory `rt-self-written-vs-optix-decision`).

### G2a — Progressive path-tracing integrator (multi-bounce GI + soft shadows + area lights)
- **Objective.** Replace the single-bounce direct shade with a progressive Monte-Carlo path integrator over the TLAS/BLAS.
- **Technical approach.** New `src/rt/path_integrator.cu`: per pixel, trace N bounces (Russian-roulette terminated at a fixed depth cap for D1), importance-sample the GGX+Lambert BRDF (extend `shading.cuh` to full BRDF sampling, not just D-term `shading.cuh:32`), area-light NEE (next-event estimation) for soft shadows + soft GI. Accumulate progressively into the color AOV. Deterministic seeded sampling (per-pixel counter-based RNG, e.g. PCG/philox seeded by `(pixel, sample, bounce)` — NO global RNG state) for D1 (`research`/roadmap §6 "deterministic seeded sampling"). Reuse `TwoLevelSceneDevice` BLAS (`two_level_render.hpp:81`), `intersect_primitives.cuh`, `bvh_ray_traversal`.
- **Inputs/Outputs/Interface.** In: `TwoLevelScene` + lights + sample count. Out: converged color AOV (+ existing AOVs). Interface: C7 video, G4 benchmark.
- **Dependencies.** G1 (exists). Independent of C-phases (can land in parallel).
- **D1 strategy.** Counter-based RNG keyed by (pixel,sample,bounce) → bit-identical across runs/replicas (NO atomics, one ray/pixel per sample, fixed bounce cap — the `two_level_render.hpp:24` precedent). Fixed-order sample accumulation (`__fadd_rn`, progressive sum in fixed sample order).
- **Validation/test gates.** `tests/rt/test_path_integrator.cpp` — a Cornell-box-like scene converges to a reference image within tol at fixed spp; 2-run bit-identity at fixed seed; energy conservation (furnace test). 
- **Effort.** L.
- **Risks.** Convergence noise vs spp budget (offline, so acceptable). D1 with progressive accumulation requires fixed sample order — no atomic splatting.

### G2b — Spatiotemporal denoise + AA + tonemap
- **Objective.** Denoise the path-traced result + anti-alias + tonemap for beauty.
- **Technical approach.** New `src/rt/denoise.cu` + `tonemap.cuh`: edge-avoiding À-Trous / SVGF-style denoiser guided by the normal + albedo + depth AOVs (already present `framebuffer.hpp:38-40`) — self-written, no OptiX denoiser. AA via supersampling (multiple deterministic sub-pixel samples accumulated, reusing G2a's seeded RNG). Tonemap (ACES/Reinhard) + gamma in a final pass. All deterministic (fixed kernel taps, fixed-order reduction).
- **Inputs/Outputs/Interface.** In: noisy color + guide AOVs. Out: final LDR image. Interface: C7 video.
- **Dependencies.** G2a.
- **D1 strategy.** Fixed À-Trous tap order + fixed-order weighted sum (`__fadd_rn`); supersample accumulation in fixed sub-sample order. No temporal feedback loop unless reprojection is also deterministic (defer temporal reuse if it threatens D1).
- **Validation/test gates.** `tests/rt/test_denoise_tonemap.cpp` — denoised image PSNR vs reference; tonemap maps a known HDR ramp correctly; 2-run bit-identity.
- **Effort.** M–L.
- **Risks.** Temporal denoising introduces history-buffer state that can break D1; ship spatial-only first, temporal as a guarded fast-path (D2 seam per Q6).

## G3 — Semantic / normal AOVs (sim-sensor parity)

### G3a — Instance + semantic-class ID AOVs
- **Objective.** Add instance-segmentation + semantic-class-ID G-buffer channels (Isaac-grade sensor parity, roadmap §W2 G3); normal/uv/albedo already present (`framebuffer.hpp`).
- **Technical approach.** Extend `Framebuffer` (`framebuffer.hpp:32`) with `std::vector<uint32_t> instance_id` and `std::vector<uint32_t> semantic_id` channels. `prim_id` already packs (instance, local-prim) via `PackPrimId` (`two_level_render.hpp:22`, `prim_id.cuh`) — derive `instance_id` by unpacking the high bits. `semantic_id` from a cooker class-id map: add a per-shape/per-material semantic class id to the cooked blob (small C1-adjacent field) and a `prim_id → semantic` lookup written at hit. One write per pixel (the closest hit), slot-indexed → D1.
- **Inputs/Outputs/Interface.** In: hit prim_id + cooker class-id map. Out: instance_id + semantic_id AOVs (DLPack-exportable like the others, `framebuffer.hpp:16`). Interface: sensor consumers.
- **Dependencies.** G1 (prim_id exists); a small cooker class-id field (can reuse material id or add a dedicated semantic id).
- **D1 strategy.** Pure per-pixel closest-hit write, no atomics, fixed (one ray/pixel) → memcmp-identical (`framebuffer.hpp:19` precedent).
- **Validation/test gates.** `tests/rt/test_semantic_aov.cpp` — a 2-instance scene yields distinct instance_ids; semantic_id matches the cooker class map; 2-run identity.
- **Effort.** S–M.
- **Risks.** Instance cap (`two_level_render.hpp` asserts ≤4096 instances) bounds the id space; document. Semantic class authoring (where do class ids come from in MJCF/USD?) — OPEN below.

## G4 — Render-perf benchmark + OptiX-trigger check

### G4a — MRays/s + seconds/frame benchmark harness + honest OptiX-trigger report
- **Objective.** MEASURE render throughput on showcase scenes; report honestly; check the OptiX-trigger condition (revisit OptiX ONLY if self-written can't hit the bar — roadmap §W2 G4; memory `rt-self-written-vs-optix-decision`).
- **Technical approach.** New `src/apps/render_bench` (or a test harness): time `RenderFrame` (G1 direct) + the G2 path integrator over fixed showcase scenes (grasp, stack, fluid), report MRays/s, seconds/frame, samples-to-converge. Compare against the documented bar (and Newton/Isaac comparables qualitatively — Newton has NO photoreal RT, `roadmap §1`). Emit a markdown report. The OptiX-trigger is a MEASURED gate (G4 only measures; publish → v1.0): if seconds/frame on the showcase exceeds the target by the trigger margin, FLAG for an OptiX-hybrid spike (does not auto-adopt — preserves the no-SDK pillar pending owner decision).
- **Inputs/Outputs/Interface.** In: showcase scenes. Out: a perf table + a go/no-go OptiX-trigger flag.
- **Dependencies.** G1, G2.
- **D1 strategy.** Benchmark only (timing is non-deterministic by nature; the RENDERED output it times is D1). Report wall-clock + a fixed-seed image hash to prove the timed render is the D1 one.
- **Validation/test gates.** `tests/rt/test_render_bench_smoke.cpp` — harness runs, emits numbers, the timed image hash matches the D1 reference. The benchmark NUMBERS are not gated (hardware-dependent); the harness CORRECTNESS is.
- **Effort.** S.
- **Risks.** nvidia-smi is BROKEN in this env (memory `v07-build-run-env`); rely on CUDA event timing, not nvml. Honest reporting (vendor-number temptation) — report measured, not aspirational.

## U1 — Real-time interactive Vulkan viewport

The current Vulkan renderer is OFFSCREEN-only (`vulkan_renderer.hpp`: `RenderSceneVulkan` / `RenderDebugDrawListVulkan` / `VulkanOffscreenOptions` — no swapchain/window/input). Roadmap §W3 U1: live orbit + play/pause/step + drag-to-perturb + debug overlays, on the rasterizer (NOT path tracing). 60fps viewport.

### U1a — Windowed swapchain + present loop + orbit camera
- **Objective.** A live window with a swapchain present loop and an orbit camera, reusing the existing Vulkan rasterizer.
- **Technical approach.** Extend `src/render/vulkan_renderer.{hpp,cpp}` with a windowed path (GLFW or xcb surface → `VkSwapchainKHR` → per-frame acquire/submit/present), alongside the existing offscreen path (`RenderSceneVulkan` shares the scene draw). Orbit camera (azimuth/elevation/distance) feeding `RenderCamera` (`render_scene.hpp:35`). Reuse `BuildRenderScene` (`render_scene.hpp:80`) per frame from the live `SceneIR`/state.
- **Inputs/Outputs/Interface.** In: live world state + input events. Out: a 60fps window. Interface: U1b controls, U1c overlays.
- **Dependencies.** Vulkan renderer (exists, offscreen). Independent of C-phases.
- **D1 strategy.** N/A for the interactive viewport (it is a VIEWER of D1 sim state, not a sim producer); the SIM it displays stays D1. Keep the viewport off the determinism-gated path.
- **Validation/test gates.** `tests/render/test_vulkan_swapchain_probe.cpp` — swapchain creates + presents headless (or via the existing `ProbeVulkanRenderer` `vulkan_renderer.hpp:76`); orbit camera matrix matches a hand oracle. (Interactive bits are manual-QA.)
- **Effort.** M (windowing) within an overall L for U1.
- **Risks.** This env may have no display (headless); gate the windowed path behind a runtime probe + keep offscreen as the CI path. GLFW dependency — confirm it is acceptable (self-written pillar is about SOLVERS/SDKs, windowing libs are fine; CONFIRM with owner — OPEN).

### U1b — Play/pause/step + drag-to-perturb interaction
- **Objective.** Sim control (play/pause/single-step) + mouse drag-to-perturb (object picking → apply force).
- **Technical approach.** Pause/step gate the world_stepper loop. Object picking via a GPU prim_id read-back at the cursor (reuse the prim_id AOV idea, or a Vulkan picking pass). Drag → a transient external force/impulse on the picked body (a debug spring to the cursor ray). Gizmos for the picked body.
- **Inputs/Outputs/Interface.** In: mouse/keyboard. Out: sim control + perturbation forces. Interface: U1a window.
- **Dependencies.** U1a.
- **D1 strategy.** Interactive perturbations are non-deterministic INPUT (user-driven); when no interaction occurs the sim is bit-identical. The perturbation force enters as a normal external wrench (does not break the solver's internal D1). Document that interactive sessions are not replay-D1 unless inputs are recorded.
- **Validation/test gates.** `tests/render/test_picking.cpp` — ray-pick returns the correct body id for a known scene; pause/step advances exactly one step. 
- **Effort.** M.
- **Risks.** Picking precision; force injection must use the public external-wrench path (no solver-internal hack).

### U1c — Debug overlays (contacts / forces / SDF)
- **Objective.** Visualize contacts, contact forces, and SDF fields live (the diagnostic value for the new contact subsystem).
- **Technical approach.** Reuse the debug-draw path (`VulkanDebugDrawCommand` / `debug_draw.comp` `vulkan_renderer.hpp:39`, `src/render/shaders/debug_draw.comp`): draw contact points + normals from the unified `ContactManifold` stream (C3), contact-force arrows from the solver's per-row lambda (the `ContactForceBuffer` `batched_articulated_world.hpp:291` already exposes {Fn,Ft1,Ft2}), SDF iso-surface slices from `CookedSdfTable`. Toggleable layers.
- **Inputs/Outputs/Interface.** In: manifolds + solver forces + SDFs. Out: overlay draw commands. Interface: U1a window.
- **Dependencies.** U1a; reads C3/C5 outputs (so most useful AFTER the contact spine lands, but the overlay plumbing is independent).
- **D1 strategy.** N/A (visualization of D1 data).
- **Validation/test gates.** `tests/render/test_debug_overlay.cpp` — a known contact renders a point+normal at the expected pixel; force arrow length scales with lambda.
- **Effort.** M.
- **Risks.** Overlay reads cross-subsystem buffers; keep it read-only (no perturbation of the sim buffers).

---

## Appendix A — Dependency DAG / suggested commit order

One phase-task per commit. The SPINE (C1→C5) is strictly serial (each consumes the previous); C6/C7 close it; R1/G2/G3/G4/U1 are LARGELY INDEPENDENT of the C-spine and can interleave or run in parallel branches.

```
C1a ─▶ C1b ─▶ C1c ────────────┐  (metadata + importer + filter policy)
                              ▼
        C2a ─▶ C2b ─▶ C2c     │  (registry + LBVH-filtered + cross-system matrix)
                       │      │
                       ▼      ▼
        C3a ─▶ C3b ─▶ C3c     (dispatch + analytical + convex)
          └──▶ C3d            (SDF tier; needs C3a only)
                       │
                       ▼
        C4a ─▶ C4b ─▶ C4c     (solref/solimp + emission + reaction providers)
                       │
                       ▼
        C5a ─▶ C5b ─▶ C5c     (unified solve + two-way + foot-ground re-baseline)
                       │
                       ▼
        C6b ─▶ C6a            (coupling-row framework / id13 + PBD co-step)
                       │
        C7a ───────────┤      (scoped usdc cup reader; importer — parallel to spine)
                       ▼
        C7b                   (H1 grasp-place validation)  ◀── R1a (optional reach)

INDEPENDENT TRACKS (parallel branches, no C-spine dependency):
   C7a   (scoped usdc cup reader; importer work — can land anytime, gates C7b)
   R1a   (IK; needs only Featherstone Jacobians — can land anytime)
   G2a ─▶ G2b   (path tracer + denoise; needs only G1 — parallel)
   G3a          (semantic AOVs; needs G1 + a small cooker class-id field)
   G4a          (render bench; needs G1 + G2)
   U1a ─▶ U1b
        └▶ U1c  (U1c most useful after C3/C5 but plumbing independent)
```

**Recommended serial commit order (single-branch):**
`C1a, C1b, C1c, C2a, C2b, C2c, C3a, C3b, C3c, C3d, C4a, C4b, C4c, C5a, C5b, C5c, C6b, C6a, C7a, C7b` — then the independent tracks `R1a, G2a, G2b, G3a, G4a, U1a, U1b, U1c` (these may be reordered or parallelized; G2/U1/C7a can begin immediately — C7a only needs the v0.7 importer seams, G2/U1 only the shipped G1/Vulkan seams). 28 commit-tasks total.

**Critical path:** C1c → C2 → C3 → C4 → C5 → C6 → C7b is the contact spine; everything in C7b's grasp depends on the full chain **plus C7a (the real usdc cup asset — OPEN-J)**. C7a runs in parallel with the spine but MUST land before C7b. R1/G2/G3/G4/U1 do not block the grasp gate (R1 only assists reach; the video uses G1, G2 is an upgrade).

## Appendix B — Extension-seam checklist (roadmap-readiness; Q9 / architecture §3)

Every seam below must be a REGISTRATION point, not a hardcoded enum across dispatch. Verified against the architecture §3 directive.

- [ ] **Collidable-type registry** (0.4, `collidable_registry.hpp`, C2a): v0.8 registers RigidBody/ArticulationLink/Particle/StaticWorld. v0.9 APPENDS CableSegment/ClothTriangle/MpmParticle/FemTet — each supplies {AABB provider, accel-structure, shape/proxy provider, reaction provider}. NO type enum branched across dispatch. **Codegen-time enum + static metadata; runtime device-fn pointers resolved at world build.**
- [ ] **Narrowphase dispatch table** (C3a, `narrowphase_dispatch.hpp`): `(ShapeType a × ShapeType b × Tier) → handler`, `constexpr`. New pairs (mesh×cloth-tri, particle×fem-tet) = new table rows. SDF tier is just another table value (MJ `MJ_COLLISION_TABLE` pattern, `collision_driver.py:45`).
- [ ] **Solver registry slot** (0.6, `unified_solve.hpp`, C5a): the `UnifiedSolve` row stream is solver-agnostic (Newton `step(...,contacts,...)` contract). **SolverKamino (Proximal-ADMM, closed loops, R6 — DEFERRED "kamino先不做")** plugs in alongside the row-PGS, consuming the SAME row stream/manifolds, when revived. The slot exists now; the solver does not.
- [ ] **Coupling-pair matrix** (0.7 / C6b, id13 `CouplingContactRow`): rigid↔MPM↔cloth↔fluid two-way = new coupling-row EMISSION entries per pair (not new kernels — the win over Genesis's O(N²) whitelist). v0.8 ships the framework + K2 proving pair; v0.9 R8 fills the matrix.
- [ ] **CCD hook** (reserved, C3a-adjacent; `research-breadth-solvers.md:84-92`): discrete-only in v0.8. Reserve a `NarrowphaseTier::SweptCCD` table slot + a swept-AABB-BVH-refit entry point (the LBVH refit `lbvh_refit.cu` already exists). Primary consumers: garment cloth (R3) + thin cable (R2) in v0.9. ACCD (Li 2021) is the planned algorithm. No new row class — a detection MODE feeding TOI-aware contacts to existing contact rows.

## Appendix C — D1 re-baseline note (which goldens C5 invalidates + how to regenerate)

Current goldens (`tests/oracle/golden/`):
- **`go2_stand_5s.bin`** (62 KB, the standing trajectory) — **INVALIDATED by C5c.** It rides the foot-ground-only contact path (`articulation_contacts` FootShape×scalar-ground, NULL reaction body). C5c routes feet through the unified pipeline (analytical plane tier + StaticWorld reaction + compliant solref/solimp formulation instead of the old Baumgarte-only contact). The trajectory WILL change (compliant vs Baumgarte). **Regenerate** via the go2-stand oracle regeneration path (the test that produced it — `tests/oracle/test_go2_stand` regen mode) AFTER C5c lands and the stand is verified visually stable. Re-baseline is owner-approved (Q3 / `v08-unified-collision-contact.md:16` "re-baseline goldens OK incl. subsume foot-ground"). Gate the new golden with 2-run bit-identity + N≥32 cross-replica on the unified path.
- **`featherstone_go2_random_sample.bin`** (208 KB) + **`featherstone_h1_random_sample.bin`** (320 KB) — **NOT touched by C5.** These validate the Featherstone ABA FORWARD DYNAMICS (random q/qd → accelerations), which has NO contact. The unified contact subsystem does not change ABA. They stay byte-identical; if a refactor incidentally touches `featherstone_aba.cu` (it should not), re-run their generating tests to confirm no movement. **Expectation: unchanged.**

**C2 broadphase-flip caveat (a SECOND source of potential rigid-golden movement, distinct from C5's formulation change):** C2b flips the production contact broadphase from the legacy SAP to the LBVH (reversing the p04 "SAP stays default to protect goldens" decision, `broadphase_lbvh.hpp:8-15`). The pair SET is SAP-equal (`test_lbvh_vs_sap_pair_set.cpp`) AND, per the C2b order analysis, both emit canonical `(body_a<body_b)` order (SAP's upper-triangular slot layout `broadphase.cu:230` ≡ LBVH's sorted list), so the graph-colored-PGS row order is preserved — rigid-maximal goldens are NOT expected to move from the broadphase flip alone. **If C2b lands BEFORE C5** (it does in the serial order), run the rigid-contact regression tests (any rigid stacking/resting goldens, and the rigid-maximal solver tests) at C2b to CONFIRM no movement; if any moves, the SAP/LBVH order assumption is wrong and that golden re-baselines at C2b with justification. Treat this as a checkpoint, not an assumed null.

**Regeneration discipline:** regenerate ONLY `go2_stand_5s.bin`, ONLY at C5c, ONLY after the stand is confirmed physically correct (not merely self-consistent). Commit the regenerated golden in the C5c commit with a message documenting the formulation change (compliant contact + StaticWorld reaction). Any C6 PBD/fluid goldens (if present beyond these three — none found in `tests/oracle/golden/`) follow the same rule at C6a.

## Appendix D — OPEN questions — ALL RESOLVED (owner, 2026-06-04)

These were DECOMPOSITION/IMPLEMENTATION forks, NOT re-litigation of Q1–Q11. Owner directive 2026-06-04: "其余小型 OPEN 都解决" → every OPEN below is resolved at its recommended answer (or the owner's explicit choice for J).

- **OPEN-A (0.1) — RESOLVED: rename.** `ContactManifold` uses `a.handle/b.handle`; the 4 call sites (`row_builder.cpp:162`, `world_stepper.cpp`, …) get the mechanical rename (clarity over a `body_a()` shim).
- **OPEN-B (0.2) — RESOLVED: reuse is safe.** Contact regularizer `R=1/D` reuses `Row.compliance_alpha` (no Row growth). VERIFIED no aliasing: `src/solver/gpu/row_solver.cu` NEVER reads `compliance_alpha` (it is a separate kernel from `xpbd_world.cu`, the only consumer), so the rigid-contact branch can repurpose the field. The `float regularizer_R` field alternative stays a fallback only if a future test surfaces a concrete aliasing bug.
- **OPEN-C (0.3) — RESOLVED: reserve.** Reserve id11/id12 for v0.9 Cosserat now; the C6b coupling row takes **id13**; `kRowClassCount=14` (11/12 as reserved placeholders). v0.9 plugs Cosserat into 11/12 with zero renumber.
- **OPEN-D (0.4) — RESOLVED: codegen enum + fn-ptr table.** Collidable registry = codegen-time enum + static metadata + runtime device-fn pointers resolved at world build (matches the YAML→registry path), NOT a pure-runtime registry object.
- **OPEN-E (C6b) — RESOLVED: ship concrete.** v0.8 ships the CONCRETE id13 `CouplingContactRow` + the **K2 (particle↔rigid-SDF) proving pair** as the minimal concrete validation, not interface-only. v0.9 R8 fills the remaining pairs.
- **OPEN-F (C4a) — RESOLVED: integer powers + unroll.** The solimp two-branch sigmoid restricts the `pow(x,power)` exponent to INTEGER powers, evaluated by an unrolled integer-power helper → host/device bit-identity (no g++-libm-vs-nvcc divergence). Non-integer `power` is not used (documented constraint); if a future material needs it, it carries a documented host-vs-device tolerance — out of scope for v0.8.
- **OPEN-G (G3a) — RESOLVED: `nuka:semantic` attr + material-id fallback.** Semantic class ids come from an optional `nuka:semantic="<class>"` geom attribute in MJCF/USD import; when absent, the cooker falls back to the material id as the class id. The cooker maps both to the G3 class-id field. No heuristic guessing.
- **OPEN-H (U1a) — RESOLVED: GLFW OK + headless gate.** GLFW is acceptable under the no-closed-SDK pillar (it is a windowing lib, not a solver/render SDK). Offscreen rendering is the CI path; the windowed viewport is gated behind a runtime display probe so headless CI never opens a window.
- **OPEN-I (C3d) — RESOLVED: single-witness for the v0.8 grasp.** Single-point fingertip SDF contact + analytical multi-point elsewhere is sufficient for the C7b grasp. Perturbed-restart multi-point SDF is DEFERRED to v0.9 (OPEN-V4), where go2-on-sand foot resting stability needs it and R4f implements it.
- **OPEN-J (owner, C7) — RESOLVED: real usdc cup, scoped reader into v0.8.** The REAL usdc-binary newton-assets cup is MANDATORY (owner 2026-06-04). The minimal usdc parse is pulled into v0.8 as **C7a (scoped to this one cup)** — general usdc composition-arc hardening stays v0.9 U4a. Accepted consequence: C7 (the flagship grasp gate) now has an **L-tier self-written binary parser on its critical path** (C7a). Documented as a critical-path risk in the C7 header note + the DAG.
