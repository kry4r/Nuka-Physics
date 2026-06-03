# v0.9 Detailed Task Design — Breadth Physics Solvers + Advanced Frontend

> **Status:** controller-authored task decomposition (2026-06-03), for owner + advisor review. Decomposes the v0.9 scope in `docs/plans/2026-06-03-post-v07-roadmap.md` §3 (breadth solvers) / §6 (cross-cutting policy: full D1; **new breadth solvers FORWARD-ONLY — NOT differentiable**; no closed SDK). Every v0.9 solver **plugs into the v0.8 unified collision/contact/coupling spine** (`docs/plans/2026-06-03-v08-detailed-tasks.md`, the SHARED-INTERFACE SPEC §0). The v0.8 design is the foundation and is NOT re-litigated here.
> **Algorithm grounding:** `docs/plans/research/2026-06-03-research-breadth-solvers.md` (PRIMARY — R1 IK / R2 Cosserat / R3 garment cloth / R4 MLS-MPM+Drucker-Prager / R5 corotational→stable-Neo-Hookean / CCD=ACCD), `...-genesis.md` (per-pair coupling, `_func_collide_in_rigid_geom`), `...-newton-mujocowarp.md` (VBD/Style3D, SolverImplicitMPM, shared Contacts registry, tendon parity via SolverMuJoCo). Citations carry the source file:line (e.g. `research-breadth-solvers.md:48`).
> **Scope of THIS doc:** the v0.9 phases **R2, R3, R4, R5, R7, R8, U2, U3, U4, U5**. **R1 IK landed in v0.8** (v0.8/R1a) — referenced, not re-decomposed. **R6 Kamino closed-loops = DEFERRED** ("kamino先不做"): a single reserved-slot stub (§R6), NO task breakdown.

---

## 0. THE TWO MASTER REGISTRIES (lock these first — every v0.9 task references them)

The single largest risk in a doc this size is registry drift. Both registries are FROZEN here and every task below cites which slot it fills. They extend — never renumber — the v0.8 seams.

### 0.1 `CollidableType` registry (the collision seam — v0.8 §0.1 / §0.4 `collidable_registry.hpp`)

v0.8 defines `enum class CollidableType : uint8_t` and **already reserved the v0.9 values** (`v08-detailed-tasks.md:24`): `RigidBody=0, ArticulationLink=1, Particle=2, StaticWorld=3, // v0.9 seam: CableSegment=4, ClothTriangle=5, MpmParticle=6, FemTet=7`. v0.9 IMPLEMENTS each reserved type by APPENDING a `CollidableTypeInfo {compute_aabbs, accel, proxy, react}` (v0.8 §0.4) — no enum branched across dispatch (the Q9 extensibility mandate, `post-v07-roadmap.md:109`).

| CollidableType | id | Filled by | AABB provider | accel structure | reaction provider |
|---|---|---|---|---|---|
| CableSegment | 4 | **R2c** | per-segment capsule AABB | LBVH (rigid tree, refit) | ParticleInvMass (×2 endpoints) + angular |
| ClothTriangle | 5 | **R3d** | per-triangle AABB (swept for CCD) | LBVH (cloth BLAS, refit) | ParticleInvMass (×3 verts, bary-weighted) |
| MpmParticle | 6 | **R4f** | grid-cell AABB | UniformGridParticle | ParticleInvMass (grid-node mediated) |
| FemTet | 7 | **R5e** | per-tet AABB | UniformGridParticle (verts) | ParticleInvMass (×4 verts, bary-weighted) |

**R7 tendon/muscle is NOT a collidable** — it is an actuation/transmission layer over articulation DOFs (no narrowphase). It registers no `CollidableType`.

### 0.2 `RowClassId` registry (the constraint seam — v0.8 §0.3 / `row_class_registry.hpp`)

**This is CONTINGENT on v0.8 OPEN-C** (`v08-detailed-tasks.md:558`), whose RECOMMENDED resolution is: reserve id11/12 for v0.9 Cosserat NOW, coupling takes **id13**, `kRowClassCount=14`. v0.9 honors that resolution and FILLS the reserved slots. Current registry ends at id10 (`kRowClassCount=11`, `row_class_registry.hpp:62`); v0.8 adds id13 + reserves 11/12. v0.9's final state:

| id | name | Added by | jacobian_kind | constraint_kind | gradient_mode |
|---|---|---|---|---|---|
| 0–10 | (existing: contact/joint/drive/FS-contact/SDF×2/XPBD×4/p↔p) | v0.7 | — | — | — |
| 11 | `XpbdCosseratStretchShear` | **R2a** (fills v0.8 reservation) | maximal_6vec (2 pos + 1 quat) | equality (compliant) | **none** (forward-only) |
| 12 | `XpbdCosseratBendTwist` | **R2b** (fills v0.8 reservation) | maximal_6vec (2 quat / Darboux) | equality (compliant) | **none** |
| 13 | `CouplingContactRow` | **v0.8 C6b** (generic cross-type) | maximal_6vec | unilateral_with_friction | none |
| 14 | `XpbdStrainLimit` | **R3a** | maximal_6vec (2 pos) | unilateral (clamped) | **none** |
| 15 | `XpbdDihedralBend` | **R3b** | maximal_6vec (4 pos) | equality (compliant) | **none** |
| 16 | `FixedTendonRow` | **R7a** | featherstone_chain_scalar (variable bodies) | equality | **none** |
| 17 | `SpatialTendonRow` (routed length + limit) | **R7b** | featherstone_chain_scalar (variable) | unilateral / equality | **none** |

**R4 (MPM) and R5 (FEM) add ZERO new row classes.** MPM is a grid co-step; FEM is a CG co-step. Both expose contact to the rest of the world through the **existing manifold + the generic id13 `CouplingContactRow`** — NOT a per-solver row id. This is the deliberate divergence from `research-breadth-solvers.md:52,99` (which names `MpmRigidCoupling` as a new id): the id13 framework's whole purpose is to kill per-pair ids (the Genesis O(N²)-whitelist win, `research-genesis.md:96`). Grid-specific math (the summed-nodal-force reaction) lives in the **reaction provider** registered for `MpmParticle`, not in a new row class. **OPEN-V1** (below) flags the one case where a pair might genuinely need its own row.

All v0.9 row classes are `default_gradient_mode: none` (forward-only, `post-v07-roadmap.md:107`) — they emit a forward-only stub, no `adjoint_evaluator` block (contrast the v0.7 XPBD rows which carry `dense_adjoint`, `xpbd_distance.yaml:12`). They register the standard way: a new `tools/codegen/classes/<name>.yaml` + `python tools/codegen/regen.py` → `row_class_registry.hpp` + the hand-mirrored constant in `row.hpp` (the `kXpbd*RowClassId` pattern, `row.hpp:57-64`), guarded by the codegen-roundtrip test. **Final `kRowClassCount` after v0.9 = 18** (v0.8 takes it to 14 with id13 + reserved 11/12; v0.9 fills 11/12 and adds 14/15/16/17). v0.8 tracked this count explicitly (`row_class_registry.hpp:62`); v0.9 does too.

### 0.5 The load-bearing NEW types v0.9 introduces (concrete signatures — the v0.9 analog of the v0.8 §0 contracts)

v0.8 concentrated its concrete C++ contracts in §0 (`ContactManifold`, `CollidableRef`, `ReactionProvider`, `SolveContext`, `EmitCouplingRows`). v0.9's load-bearing NEW types — proposed signatures below; per-task field-lists (the YAML blocks) carry the rest.

```cpp
// src/runtime/soft/cosserat_world.hpp  (NEW, R2) — per-segment orientation state.
// Rod NODES are XPBD particles (reuse XpbdWorld position buffers); SEGMENTS (one
// between consecutive nodes) add the orientation half a position-only particle lacks.
struct CosseratSegmentSet {
    std::vector<math::Quat>  orientation;   // per-segment material frame quaternion
    std::vector<math::Vec3>  ang_velocity;  // per-segment angular velocity
    std::vector<float>       inv_inertia;    // per-segment scalar inverse rot. inertia (0 == pinned)
    std::vector<float>       rest_length;    // per-segment rest length l
    std::vector<math::Vec3>  rest_darboux;   // Ω_0 rest curvature/twist (0 == straight)
};
class CosseratWorld {                        // shares node positions with an XpbdWorld
    // predict/solve/correct loop shape == XpbdWorld; adds own-index quaternion
    // projection + post-sweep renormalize (the D1 pitfall: renorm own-index, not in a reduction).
    math::Quat*  DeviceOrientations();  math::Vec3* DeviceAngVelocities();
    const float* DeviceInvInertia() const;  // ... (mirrors XpbdWorld accessor pattern)
};

// src/runtime/mpm/mpm_world.hpp  (NEW, R4) — grid + particle state.
struct MpmParticleSet {
    std::vector<math::Vec3>  position, velocity;
    std::vector<math::Mat3>  affine_C;        // APIC affine momentum matrix
    std::vector<math::Mat3>  deform_grad_F;   // deformation gradient
    std::vector<float>       plastic_state;    // Drucker-Prager hardening / log-J_p
    std::vector<float>       mass, volume;
};
struct MpmGrid {                              // sparse uniform block grid (dirty-cell tracked)
    math::Vec3 origin; float cell_size; uint32_t dims[3];
    // device SoA: node_mass, node_momentum, node_velocity; + dirty-cell list.
};
struct MpmStepReport { uint32_t particle_count, active_cell_count, truncated_node_count; };
MpmStepReport StepMpmWorld(MpmWorld&, const MpmStepOptions&);  // P2G(gather)→grid→G2P→advect→return-map

// src/constraint/reaction_provider.hpp  (EXTEND, R4f) — a 5th provider kind appended
// to v0.8's {RigidInvMass, ArticulationChainJ, ParticleInvMass, StaticNull}.
// The grid-specific MPM↔rigid math (summed nodal force over covered nodes, fixed
// gather order, NO float atomics) lives HERE — NOT in a new row class (see §0.2).
enum class ReactionProviderKind : uint8_t { /* …v0.8 four… */ MpmGridNodal = 4 };
// effective_inv_mass(ref, J): grid-mediated; apply_impulse(ref, J, dλ): distribute
// over the rigid's covered grid nodes via the P2G/G2P weights, fixed-order sum.

// src/runtime/fem/fem_world.hpp  (NEW, R5) — tet element + co-step over the self-written CG.
struct FemTetElement {
    uint32_t v[4];                 // vertex indices (reuse tetmesh_topology)
    math::Mat3 rest_shape_inv;     // Dm^-1 (rest shape matrix inverse)
    float rest_volume, lame_mu, lame_lambda;
};
class FemWorld {                   // owns its implicit backward-Euler solve
    // assemble (M/dt^2 + K)Δx = −∇E (gather, one thread/vertex over incident tets,
    // sorted order, NO scatter-atomic) → SelfWrittenCgBackend (reuse, D1) → integrate.
    void Step(const FemStepOptions&);
    const math::Vec3* DeviceVertexPositions() const;  // surface verts → particle pseudo-bodies for contact
};
```

### 0.3 The other v0.8 seams v0.9 plugs into (named, with the consuming v0.9 task)

| v0.8 seam (owner task) | what it provides | v0.9 consumers |
|---|---|---|
| **Collidable-type registry** (§0.4, C2a) | append `CollidableTypeInfo` | R2c, R3d, R4f, R5e |
| **Narrowphase dispatch table** (§0.3/C3a) | `(ShapeType a × b × Tier) → handler`, constexpr | R2 cable-vs-world, R3 self-collision, R4/R5 vs-rigid |
| **`NarrowphaseTier::SweptCCD` hook** (Appendix B, RESERVED only) | a reserved tier slot + swept-AABB-refit entry; **NOT implemented in v0.8** | **R3c IMPLEMENTS ACCD on it**; R2 consumes |
| **Compliant contact (solref/solimp → `R=1/D` on `compliance_alpha`)** (§0.2/C4a) | one denominator regularizer | every v0.9 contact/coupling row |
| **Reaction providers** (§0.5/C4c) | impulse→DOF for Rigid/Articulation/Particle | every v0.9 two-way coupling; R4/R5 register new ones |
| **`CouplingContactRow` id13 + `EmitCouplingRows` framework + co-step bridge** (§0.7/C6b) | cross-type non-penetration/friction row | **R8 (all pairs)**, R4↔rigid, R5↔rigid |
| **`UnifiedSolve(SolveContext)` + solver-registry slot** (§0.6/C5a) | one co-solve consuming all rows; Kamino slot | R2/R3 rows co-solved; R6 reserved |
| **Two-way articulation↔rigid reaction** (C5b) | summed effective mass, equal-and-opposite | R8 articulated↔MPM (go2-sand), R8 cloth↔articulated |
| **Co-step ordering `pre_coupling → couple → post_coupling`** (C6a/C6b, Genesis `simulator.py`) | the bridge each system advances inside | R4 (MPM substep), R5 (FEM implicit solve), R3 (cloth predict/correct) |

### 0.4 v0.8 OPEN flags v0.9 INHERITS (do not re-decide; track to closure)

- **OPEN-C** (id allocation) — v0.9's §0.2 ASSUMES the recommended resolution (11/12 reserved → Cosserat, coupling=id13). If owner picks the alternative (coupling=id11, force renumber), R2 Cosserat moves to 13/14 and everything downstream shifts +2. **R2a must not start until OPEN-C is closed.**
- **OPEN-E** (does v0.8 ship a concrete id13 row?) — v0.9 R8 ASSUMES YES (v0.8 ships id13 + the K2 fluid↔rigid proving pair). If v0.8 ships interface-only, R8a additionally lands the first concrete id13 emission.
- **OPEN-I** (SDF tier = single witness point) — R8 articulated↔MPM and the grasp use SDF-as-grid-BC; the single-witness cardinality gap (`v08-detailed-tasks.md:564`) caps resting stability. R4/R8 must decide if perturbed-restart multi-point SDF is needed for go2-on-sand foot stability (flagged OPEN-V4).

---

## v0.8 → v0.9 DEPENDENCY MATRIX (which spine seam each phase needs)

| v0.9 phase | collidable reg | narrowphase table | SweptCCD hook | solref/solimp | reaction providers | id13 coupling framework | two-way artic↔rigid | co-step bridge | hard v0.8 task deps |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|---|
| **R2 Cable/DER** | ✅ (CableSegment) | ✅ (cable-vs-world) | consumes (knots) | ✅ | ✅ (Particle) | — | — | ✅ | C3a, C4a, C5a; OPEN-C |
| **R3 Garment cloth** | ✅ (ClothTriangle) | ✅ (self-collision) | **IMPLEMENTS** | ✅ | ✅ (Particle) | ✅ (cloth↔rigid) | (cloth↔artic) | ✅ | C2c, C3a, C4a, C5a, C6a, C6b |
| **R4 MPM** | ✅ (MpmParticle) | ✅ (MPM↔rigid SDF) | — | ✅ | ✅ (**new MpmGridNodal**) | ✅ (rigid↔MPM) | — | ✅ | C3d (SDF tier), C4c, C6a, C6b |
| **R5 Volumetric FEM** | ✅ (FemTet) | ✅ (FEM↔rigid) | (optional) | ✅ | ✅ (Particle/vert) | ✅ (FEM↔rigid) | — | ✅ | C4c, C6a, C6b; CG (exists) |
| **R7 Tendon/muscle** | — | — | — | — | (articulation only) | — | — | — | articulation infra (exists); R1 IK (optional) |
| **R8 Coupling per-pair** | (uses R2–R5 types) | ✅ | (per pair) | ✅ | ✅ | ✅ (**the core**) | ✅ (go2-sand) | ✅ | C5b, C6b + the partner solver phase |
| **U2 authoring** | — | — | — | — | — | — | — | — | C-ABI, SceneIR (exist) |
| **U3 web/log viewer** | — | — | — | — | — | — | — | — | bindings, framebuffer (exist) |
| **U4 real-USD pipeline** | — | — | — | — | — | — | — | — | importer/cooker, scene_compose (exist) |
| **U5 teleop/XR** | — | — | — | — | — | — | — | — | v0.8 U1 viewport |

---

# R2 — Cable / DER (XPBD-Cosserat rods)

**Pick (research-fixed):** XPBD-Cosserat over Bergou-DER (`research-breadth-solvers.md:36,98`). *Why:* native to Nuka's colored-XPBD scheduler + `compliance_alpha=1/stiffness`, forward-only-friendly, D1 by the existing XPBD posture; beats Newton's no-true-DER cable (`research-breadth-solvers.md:40`). Cosserat = stretch+shear+bend+twist via per-segment **position + orientation (quaternion)**. Key papers: Kugelstadt & Schömer 2016 (*Position and Orientation Based Cosserat Rods*); Deul et al. 2018 (*Direct Position-Based Solver for Stiff Rods*); Soler et al. 2018.

**State extension (the structural new work — flagged by advisor):** existing XPBD particles are position-only (`xpbd_world.hpp:55-59`, no rotational state). Cosserat segments carry a **quaternion orientation + angular velocity + inverse rotational inertia**. This is a concrete extension; **decision: a NEW sibling module `src/runtime/soft/cosserat_world.{hpp,cu}`** that owns the per-segment orientation state, rather than bloating `XpbdWorld` (whose particles stay translational). It reuses the same predict/solve/correct loop shape and the same colored-sweep D1 posture, and shares the particle position buffers (rod nodes ARE XPBD particles; segments add the orientation between consecutive nodes). The existing solve is SERIAL Gauss-Seidel (`xpbd_world.hpp:39`, colored-parallel deferred #25) — a path graph colors trivially red/black (`research-breadth-solvers.md:37`); ship serial first (D1-correct, sufficient for demo rope), color as a scaling follow-up.

### R2a — `XpbdCosseratStretchShear` row class (id11) + segment orientation state
- **Objective.** The stretch+shear constraint coupling two rod nodes' positions to the connecting segment's orientation. Fills the v0.8-reserved id11.
- **Technical approach.** Constraint (Kugelstadt 2016): `C_stretchShear = (1/l) (p_{i+1} − p_i) − d_3(q)` where `d_3` is the segment director (third column of the rotation of quaternion `q`), `l` = rest length. Project as an XPBD constraint over `{p_i, p_{i+1}, q}` with `compliance_alpha = 1/(stretch/shear stiffness)`, gradient w.r.t. the two positions = ±n/l (the distance-row pattern, `xpbd_distance.yaml:21-26`) and w.r.t. the quaternion via the analytic ∂d_3/∂q. New state in `cosserat_world`: `seg_orientation` (quaternion/segment), `seg_ang_vel`, `seg_inv_inertia`. New codegen class `tools/codegen/classes/xpbd_cosserat_stretch_shear.yaml` (`row_class_id: 11`, `jacobian_kind: maximal_6vec`, `body_count: 3` — 2 particles + 1 orientation-pseudo-body, `supports_compliance: true`, `default_gradient_mode: none`, NO `adjoint_evaluator`). Quaternion normalization is OWN-INDEX post-sweep, NOT inside a reduction (the D1 pitfall, `research-breadth-solvers.md:37`).
- **Registers into:** RowClassId id11 (fills v0.8 reservation, OPEN-C); the `UnifiedSolve` row stream (co-solved with the rest); the CollidableType `CableSegment` reaction path (R2c).
- **Inputs/Outputs/Interface.** In: rod node positions + segment quaternions + rest lengths + stiffness. Out: corrected positions + orientations. Interface: R2b bend/twist, R2c contact, R8 cable↔rigid.
- **Dependencies.** v0.8 OPEN-C closed; v0.8 C5a (UnifiedSolve); the existing XPBD predict/correct loop (`xpbd_world.cu`).
- **D1 strategy.** Forward-only. Colored (or serial) fixed-order Gauss-Seidel; own-index quaternion projection (no scatter); fixed-order `__fadd_rn`; quaternion renorm own-index post-sweep. Two-run + N≥32 gate.
- **Validation/test gates.** `tests/runtime/test_cosserat_stretch_shear.cpp` — a hanging rod under gravity reaches a **catenary** shape within tol vs the analytic catenary (the oracle); inextensible rod (α→0) does not stretch >tol; 2-run bit-identity; N≥32 cross-replica.
- **Effort.** **M**.
- **Risks.** Quaternion-derivative determinism (branch-free ∂d_3/∂q); coupling position and orientation in one row is the subtle part. The orientation-pseudo-body indexing into the maximal Jacobian must be consistent with the reaction provider (R2c).

### R2b — `XpbdCosseratBendTwist` row class (id12) via the Darboux vector
- **Objective.** The bend+twist constraint between two consecutive segment orientations. Fills the v0.8-reserved id12.
- **Technical approach.** Constraint (Kugelstadt 2016): `C_bendTwist = Im(conj(q_i) ⊗ q_{i+1}) − Ω_0` where `Ω` is the discrete Darboux vector (the imaginary part of the relative quaternion) and `Ω_0` the rest curvature/twist (0 for a straight rod). XPBD-project over `{q_i, q_{i+1}}` with `compliance_alpha = 1/(bend/twist stiffness)`; gradients are the analytic quaternion-product derivatives. New codegen class `xpbd_cosserat_bend_twist.yaml` (`row_class_id: 12`, `body_count: 2` orientation pseudo-bodies, compliant, `gradient_mode: none`).
- **Registers into:** RowClassId id12; the `UnifiedSolve` row stream.
- **Inputs/Outputs/Interface.** In: consecutive segment quaternions + rest Darboux. Out: corrected orientations. Interface: R2c, R8.
- **Dependencies.** R2a (segment orientation state must exist).
- **D1 strategy.** Forward-only; own-index quaternion projection; fixed-order sweep; post-sweep renorm. Two-run + N≥32.
- **Validation/test gates.** `tests/runtime/test_cosserat_bend_twist.cpp` — a pre-twisted rod relaxes to a helix matching the analytic Kirchhoff helix curvature within tol; a straight rod with Ω_0=0 stays straight; 2-run.
- **Effort.** **M**.
- **Risks.** Twist coupling (the holonomy term) is the historically hard part of rod sims; verify against the helix oracle, not just visually.

### R2c — `CableSegment` collidable + cable-vs-world + cable-vs-cable contact
- **Objective.** Register `CableSegment` (CollidableType=4) so cable collides with rigid/world (and itself) through the unified narrowphase.
- **Technical approach.** Append a `CollidableTypeInfo` for `CableSegment` (v0.8 §0.4): AABB provider = per-segment capsule (two nodes + radius) world AABB; accel = the rigid LBVH (cable segments inserted as capsule leaves, refit per step via `lbvh_refit.cu`); proxy = capsule; reaction = `ParticleInvMass` distributed to the two endpoint nodes (linear) + the segment angular reaction (the new orientation pseudo-body). Narrowphase: capsule-vs-{plane,box,convex,SDF} already covered by C3b/C3c/C3d tiers — cable just routes through the dispatch table as a capsule shape. Cable-vs-cable = segment-segment (capsule-capsule) candidate pairs from the LBVH self-traversal (C2b path generalized). Contact emits compliant contact rows (C4b) for cable-vs-rigid; the two-way reaction goes to both the rod nodes and the rigid via their reaction providers.
- **Registers into:** the **collidable-type registry** (CableSegment=4); the **narrowphase dispatch table** (capsule tiers, already present); the **compliant contact** path; the **SweptCCD hook** as a CONSUMER (knots/self-contact use R3c's ACCD).
- **Inputs/Outputs/Interface.** In: segment capsules + world geometry. Out: contact manifolds → contact rows. Interface: R8 cable↔rigid; the cable/rope v1.0 demo (demo 6).
- **Dependencies.** R2a/R2b (segments exist); v0.8 C2b/C3a/C3b/C3c/C3d (narrowphase tiers); v0.8 C4b (row emission); R3c for the CCD path (tight knots).
- **D1 strategy.** Forward-only; LBVH is already D1 (`broadphase_lbvh.hpp:24`); capsule narrowphase rides the D1 C3 handlers; contact rows D1 via C4/C5. Two-run + N≥32.
- **Validation/test gates.** `tests/runtime/test_cable_contact.cpp` — a rope draped over a box rests without penetration (min-sep ≥ −slop); a rope coiled on the ground does not self-intersect (with R3c CCD); 2-run.
- **Effort.** **M** (without robust self-knots; tight-knot CCD adds the R3c dependency).
- **Risks.** Cable-vs-cable self-contact with thin radius is a CCD problem (tunneling) — depends on R3c ACCD landing; without it, fast-moving thin cable tunnels. Document the discrete-only limitation until R3c.

---

# R3 — Garment-grade cloth (strain-limited XPBD + dihedral bend + robust self-collision)

The high-precision upgrade over the basic XPBD cloth (id6 distance + id7 bend, `cloth_topology.hpp`). Three pillars (`research-breadth-solvers.md:72-78`): strain limiting (Provot 1995), accurate dihedral bending (Bridson 2003 / Bergou 2006 / Grinspun discrete shells), robust self-collision + CCD (Bridson/Fedkiw/Anderson 2002). Tracks VBD (Chen et al. 2024) / Style3D-PD (Bouaziz 2014) as the SOTA upgrade target. **R3 owns the ACCD implementation** on v0.8's reserved SweptCCD tier (advisor: v0.8 only RESERVES it).

### R3a — `XpbdStrainLimit` row class (id14)
- **Objective.** Cap per-edge stretch (e.g. ≤10%) so cloth doesn't rubber-band — the visible failure of plain compliant distance constraints.
- **Technical approach.** A UNILATERAL clamped constraint per edge: `C = |p_a − p_b| − (1+ε_max)·rest_length ≤ 0` projected only when violated (Provot 1995 biphasic; `research-breadth-solvers.md:73`). New codegen class `xpbd_strain_limit.yaml` (`row_class_id: 14`, `body_count: 2`, `constraint_kind: unilateral`, compliant, `gradient_mode: none`). Emitted alongside the existing id6 distance constraint by `cloth_topology` (extend `BuildClothConstraints`, `cloth_topology.hpp:48`) with a strain-limit option. Runs in the existing XPBD solve as an extra colored sweep.
- **Registers into:** RowClassId id14; the XPBD solve loop (extra sweep after distance).
- **Inputs/Outputs/Interface.** In: cloth edges + max-strain ε. Out: clamped positions. Interface: R3 cloth showcase / robot-dressing.
- **Dependencies.** existing XPBD cloth (`xpbd_world`, `cloth_topology`).
- **D1 strategy.** Forward-only; clamped own-index projection; fixed-order sweep; no atomics. Two-run + N≥32.
- **Validation/test gates.** `tests/runtime/test_strain_limit.cpp` — a heavily loaded cloth strip's max edge strain stays ≤ ε+tol (vs plain distance which exceeds it); 2-run.
- **Effort.** **S–M**.
- **Risks.** Strain limiting + distance compliance can fight (over-constrain) → tune iteration order; document the recommended sweep order (distance → strain-limit → bend).

### R3b — `XpbdDihedralBend` row class (id15) — accurate bending
- **Objective.** Replace the cheap constant-stencil isometric bend (id7, `xpbd_world.hpp:73-78` — flat-rest only) with the accurate **dihedral-angle** bending energy for realistic wrinkles/folds and curved rest shapes.
- **Technical approach.** Dihedral bending (Bridson 2003 / Grinspun discrete shells, `research-breadth-solvers.md:74`): for the 4 vertices of two triangles sharing an edge, `C = θ − θ_0` where θ is the dihedral angle between the two face normals and θ_0 the rest angle (≠0 for curved garments — the gap id7 cannot express). Gradients = the analytic ∂θ/∂p_i (the standard discrete-shell bending gradient). New codegen class `xpbd_dihedral_bend.yaml` (`row_class_id: 15`, `body_count: 4`, `constraint_kind: equality`, compliant, `gradient_mode: none`). Cooked by an extended `cloth_topology` (the flap geometry already exists, `cloth_topology.hpp:gridded flap`; add a rest-dihedral cook path replacing the flat-only Bergou stencil for non-flat rest).
- **Registers into:** RowClassId id15; the XPBD solve loop.
- **Inputs/Outputs/Interface.** In: interior-edge flaps + rest dihedral. Out: bend-corrected positions. Interface: cloth showcase.
- **Dependencies.** existing cloth topology; can land parallel to R3a.
- **D1 strategy.** Forward-only; the ∂θ/∂p must be branch-free near θ=0/π (the determinism + numerical trap); fixed-order sweep; no atomics. Two-run + N≥32.
- **Validation/test gates.** `tests/runtime/test_dihedral_bend.cpp` — a cantilever cloth strip's tip deflection under gravity matches a bending-stiffness oracle within tol; a pre-curved rest shape (θ_0≠0) holds its curve (which id7 cannot); 2-run.
- **Effort.** **M**.
- **Risks.** Dihedral gradient singularity at θ→0/π (degenerate flap) — clamp/guard branch-free; the curved-rest cook is new (id7 was flat-only).

### R3c — ACCD self-collision (IMPLEMENTS the v0.8 SweptCCD tier) — the hard part
- **Objective.** Robust cloth self-collision + cloth-vs-world via Additive CCD on the SweptCCD tier v0.8 only RESERVED. **This is the highest-risk v0.9 cloth task** and is shared infrastructure (R2 cable knots consume it).
- **Technical approach.** ACCD (Li, Kaufman, Jiang 2021, *Codimensional IPC*; `research-breadth-solvers.md:87`): a robust conservative-advancement variant refining a TOI lower bound. Implement: (1) swept-AABB BVH over cloth triangles (the LBVH refit per substep, `lbvh_refit.cu` — exists); (2) candidate **vertex-triangle + edge-edge** pairs in deterministic stable-key order (count→scan→private-slice, the `cross_system_query.cu` D1 pattern); (3) per-pair ACCD (fixed max-iteration count, `__fadd_rn`); (4) earliest TOI per vertex via a **fixed-order min-reduction** with deterministic stable-pair-key tie-break (`research-breadth-solvers.md:89`); (5) feed TOI-aware contacts into the unified manifold → existing contact rows (NO new row class — a detection MODE, `v08-detailed-tasks.md:540`). Optional Bridson rigid-impact-zone fail-safe for guaranteed intersection-free states.
- **Registers into:** the **`NarrowphaseTier::SweptCCD`** table slot (v0.8 reserved, Appendix B); feeds existing contact rows; the ClothTriangle collidable (R3d).
- **Inputs/Outputs/Interface.** In: swept cloth-triangle geometry. Out: TOI-aware contact manifolds. Interface: R3 cloth, R2 cable knots (the shared consumer).
- **Dependencies.** v0.8 C3a (dispatch table + the reserved tier), C2c (candidate stream), `lbvh_refit` (exists), R3d (ClothTriangle collidable).
- **D1 strategy.** Forward-only; fixed ACCD iteration count; fixed-order TOI min-reduction with stable tie-break; pairs into private slices (no append atomic); LBVH refit D1. Two-run + N≥32 — **the determinism crux of all of v0.9** (CCD min-reductions are the classic nondeterminism trap).
- **Validation/test gates.** `tests/collision/test_accd_self_collision.cpp` — a cloth dropped on itself produces zero triangle intersections post-step (the intersection-free guarantee); a fast cloth through a thin obstacle does not tunnel (vs discrete-only which does); TOI matches an analytic moving-point-vs-plane oracle; 2-run bit-identity (the hardest gate).
- **Effort.** **R** (research-grade robustness; the single most uncertain v0.9 task alongside R4b P2G — ACCD robustness + the TOI min-reduction D1 carry genuine algorithmic uncertainty, hence R not L).
- **Risks.** ACCD robustness near-degenerate (coplanar EE, vertex-on-edge); the TOI min-reduction determinism; perf (VT+EE pairs explode in dense self-contact — cap + truncation-counter discipline like the particle grid). Reserve schedule slack.

### R3d — `ClothTriangle` collidable + cloth↔rigid via id13 coupling
- **Objective.** Register `ClothTriangle` (CollidableType=5) and emit cloth↔rigid coupling through the generic id13 framework.
- **Technical approach.** Append `CollidableTypeInfo` for `ClothTriangle`: AABB = per-triangle (swept for CCD), accel = cloth BLAS (LBVH refit), proxy = triangle, reaction = `ParticleInvMass` bary-weighted across the 3 verts. Cloth↔rigid candidate pairs (C2c) → manifolds (C3, SDF tier for rigid-carries-SDF, `research-genesis.md:39`) → **id13 `CouplingContactRow`** via `EmitCouplingRows` (v0.8 §0.7) — NO new row class. Two-way: rigid reacts via its provider, cloth verts via the bary-weighted Particle provider.
- **Registers into:** the **collidable-type registry** (ClothTriangle=5); the **id13 coupling framework**; the SDF narrowphase tier.
- **Inputs/Outputs/Interface.** In: cloth triangles + rigid SDFs. Out: id13 coupling rows. Interface: R8 cloth↔rigid; cloth showcase (drape on box) — note this is demo-4 territory but demo-4 uses BASIC XPBD (see capability→demo map).
- **Dependencies.** v0.8 C6b (id13 framework), C3d (SDF tier), C2c; R3a/R3b (cloth solve).
- **D1 strategy.** Forward-only; id13 rows D1 via C5; gather reductions. Two-run + N≥32.
- **Validation/test gates.** `tests/runtime/test_cloth_rigid_coupling.cpp` — garment cloth drapes on a box with two-way reaction (box feels weight, within momentum tol); 2-run.
- **Effort.** **M**.
- **Risks.** Bary-weighted reaction sign-consistency; reuses the id13 framework so most risk is upstream in v0.8 C6b.

---

# R4 — MPM (MLS-MPM + APIC + Drucker-Prager) — the go2-on-sand path

**Algorithm (research-fixed):** MLS-MPM (Hu et al. 2018) + APIC transfer (Jiang et al. 2015) + Drucker-Prager return mapping on a Hencky-strain hyperelastic predictor (Klár et al. 2016, sand) (`research-breadth-solvers.md:45`). Loop: P2G → grid update (force+gravity+BC) → G2P → advect + constitutive/return-map. **The headline hazard: P2G scatter must be GATHER-FORM (no float atomics) for D1** (`research-breadth-solvers.md:47-49`). MPM is a grid co-step (NOT row-native, `research-breadth-solvers.md:52`), exposing rigid coupling through the **id13 framework + a new reaction provider** (NOT a new row class — see §0.2). New module `src/runtime/mpm/mpm_world.{hpp,cu}`.

### R4a — Sparse grid + cell-sorted particle layout
- **Objective.** The MPM background grid + particle-to-cell sort, the D1 substrate for transfers.
- **Technical approach.** Uniform/hashed sparse block grid (Genesis dirty-cell bookkeeping, `research-genesis.md:67`); particles sorted by cell with the existing radix sort (reuse `particle_uniform_grid` `thrust::stable_sort_by_key`, `particle_uniform_grid.hpp:14`). Grid stores nodal mass + momentum + velocity. Sparse dirty-cell tracking to touch only active cells. New `MpmGrid` + `MpmParticleSet` (position, velocity, affine-C matrix, deformation gradient F, plastic state, mass, volume).
- **Registers into:** (foundation — no seam yet; feeds R4b–R4f).
- **Inputs/Outputs/Interface.** In: particle positions. Out: cell-sorted particles + grid. Interface: R4b P2G.
- **Dependencies.** `particle_uniform_grid` (exists, reuse the radix sort + CSR).
- **D1 strategy.** Forward-only; `stable_sort_by_key` (radix, D1); integer-atomic counters only. Two-run.
- **Validation/test gates.** `tests/runtime/test_mpm_grid.cpp` — particle-cell assignment + sort byte-stable; dirty-cell set correct; 2-run.
- **Effort.** **M**.
- **Risks.** Sparse-grid addressing complexity; keep it uniform first if hashing threatens D1.

### R4b — GATHER-FORM P2G (the D1 headline) + APIC transfer
- **Objective.** Particle→grid mass/momentum transfer with ZERO float atomics — the binding Nuka rule and the #1 v0.9 hazard.
- **Technical approach.** **Gather-form P2G** (`research-breadth-solvers.md:48`, the PREFERRED option): invert the loop — **one thread per grid node** gathers from the particles whose quadratic-B-spline support (3³=27 stencil) covers it. Build a node→particle adjacency from the R4a cell sort (count→exclusive-scan→private CSR slice, exactly `particle_uniform_grid.cu`); each node sums contributions in **fixed sorted-ascending particle order with `__fadd_rn`** (no reassociation, no warp-shuffle tree that reorders). APIC/MLS-MPM affine momentum via the per-particle `C` matrix (MLS-MPM makes APIC fall out, `research-breadth-solvers.md:45`). This is the deterministic replacement for Genesis's `atomicAdd(float)` P2G (`research-genesis.md:59` — explicitly non-deterministic).
- **Registers into:** (the D1 posture; feeds R4c).
- **Inputs/Outputs/Interface.** In: cell-sorted particles + node→particle CSR. Out: grid mass + momentum. Interface: R4c grid update.
- **Dependencies.** R4a (sort + CSR).
- **D1 strategy.** Forward-only; **gather-not-scatter, fixed-order `__fadd_rn`, NO float atomicAdd** (the binding rule, `research-breadth-solvers.md:11`); node→particle CSR via count→scan→private-slice. Two-run + N≥32 — this gate IS the headline deliverable.
- **Validation/test gates.** `tests/runtime/test_mpm_p2g_d1.cpp` — total grid mass = total particle mass (conservation); total grid momentum = total particle momentum; **2-run bit-identity + N≥32 cross-replica (the headline D1 gate)**; cross-check vs a known scatter-form reference VALUE (not bit, since scatter is nondeterministic) within fp tol.
- **Effort.** **R** (the headline D1 hazard; gather-form-vs-segment-reduce tradeoff + memory/perf carry research-grade uncertainty — R not L).
- **Risks.** Gather-form node→particle adjacency memory + perf (each node lists its covering particles); the cap/truncation discipline (like the 32-neighbor particle grid). If the adjacency is too big, fall back to sort-and-segment-reduce (`research-breadth-solvers.md:49`, option 2, also D1). This is the make-or-break MPM task.

### R4c — Grid update + boundary conditions + R4d return-mapping
- **Objective.** Grid velocity update (forces, gravity, BCs) and the constitutive + plasticity return map.
- **Technical approach.** Grid: `v_node = momentum/mass`, apply gravity + stress-divergence force (MLS-MPM's ~2× faster discretization, `research-breadth-solvers.md:45`), apply velocity BCs (walls/ground). **R4d return-mapping:** Drucker-Prager on a Hencky-strain hyperelastic predictor for SAND (Klár 2016, `research-breadth-solvers.md:45`); von-Mises seam for metal/clay. Per-particle, embarrassingly parallel, fixed analytic iteration (SVD of F via fixed-iteration analytic 3×3 — branch-free for D1).
- **Registers into:** (grid solver; feeds R4e G2P).
- **Inputs/Outputs/Interface.** In: grid mass/momentum + particle F. Out: updated grid velocity + plastic-corrected F. Interface: R4e.
- **Dependencies.** R4b.
- **D1 strategy.** Forward-only; per-node/per-particle own-output writes (no atomics); fixed-iteration analytic SVD/return-map (no data-dependent branch reordering FP). Two-run + N≥32.
- **Validation/test gates.** `tests/runtime/test_mpm_drucker_prager.cpp` — a sand column collapses to the **angle of repose** matching the Drucker-Prager friction angle within tol (the constitutive oracle); 2-run.
- **Effort.** **M–L** (Drucker-Prager return map + analytic SVD are the substance).
- **Risks.** Analytic 3×3 SVD determinism (singular-value ordering tie-breaks); the return-map cone projection branches must be FP-order-stable.

### R4e — G2P gather + particle advection
- **Objective.** Grid→particle velocity + affine-C gather, then advect.
- **Technical approach.** G2P is a clean gather (`research-breadth-solvers.md:46`): one thread per particle reads its 27-node stencil, reconstructs `v` + APIC affine `C`, advects position. Deformation-gradient update `F = (I + dt·C)·F`.
- **Registers into:** (closes the MPM step).
- **Inputs/Outputs/Interface.** In: grid velocity. Out: particle velocity + C + position + F. Interface: next step; R4f coupling.
- **Dependencies.** R4c.
- **D1 strategy.** Forward-only; per-particle own-output gather (naturally D1, no atomics, `research-breadth-solvers.md:46`); fixed stencil order. Two-run + N≥32.
- **Validation/test gates.** `tests/runtime/test_mpm_g2p.cpp` — a free-fall block of MPM material accelerates at g (no BC); round-trip P2G→grid→G2P preserves a uniform velocity field; 2-run.
- **Effort.** **M**.
- **Risks.** Low — G2P is the clean half. Mostly correctness of the F update.

### R4f — `MpmParticle` collidable + `MpmGridNodal` reaction provider + MPM↔rigid two-way
- **Objective.** Register MpmParticle (CollidableType=6) and a NEW reaction provider so MPM↔rigid two-way coupling rides the id13 framework.
- **Technical approach.** Two-way recipe (Hu 2018, `research-breadth-solvers.md:51`): rigid bodies act as moving velocity/Dirichlet BCs on the grid (rigid SDF read as the grid BC — C3d SDF tier, `research-genesis.md:39`); grid→rigid reaction = the **summed nodal force over the rigid's covered nodes**, accumulated in the SAME gather/fixed-order manner (no float atomics). Register `CollidableTypeInfo` for MpmParticle + a NEW `ReactionProvider` kind `MpmGridNodal` (the grid-specific summed-nodal-force math lives HERE, in the provider — per §0.2, NOT in a new row class). Coupling emits **id13 `CouplingContactRow`** (or, where the grid-BC handshake is cleaner as an external wrench, the SDF-field-in/impulse-field-out path of Newton's SolverImplicitMPM, `research-newton-mujocowarp.md:28`). The reaction goes to the rigid/articulation via the v0.8 reaction provider — this IS the articulated↔MPM coupling for go2-on-sand (R8b).
- **Registers into:** the **collidable-type registry** (MpmParticle=6); a **NEW reaction provider** (`MpmGridNodal`, appended to v0.8 §0.5's three); the **id13 coupling framework**; the **SDF narrowphase tier** (rigid SDF as grid BC).
- **Inputs/Outputs/Interface.** In: MPM grid + rigid SDFs + reaction providers. Out: id13 coupling rows / impulse field. Interface: R8b (go2-sand), the MPM go2-on-sand demo (demo 5).
- **Dependencies.** R4a–R4e (MPM step), v0.8 C3d (SDF tier), C4c (reaction providers), C5b (two-way), C6b (id13 framework + co-step).
- **D1 strategy.** Forward-only; **summed-nodal-force reaction in fixed gather order, NO float atomics** (the Genesis differentiator, `research-genesis.md:62`); id13 rows D1. Two-run + N≥32.
- **Validation/test gates.** `tests/runtime/test_mpm_rigid_coupling.cpp` — a rigid ball dropped into sand displaces it AND decelerates (two-way momentum balance within tol); 2-run + N≥32.
- **Effort.** **L**.
- **Risks.** **OPEN-V1**: if the grid-BC handshake genuinely can't express two-way reaction as an id13 row (the grid is not a point-contact manifold), it falls back to the SDF-field/impulse-field external-wrench path — STILL through the registered reaction provider, but not literally an id13 row. Flag for advisor. **OPEN-V4** (inherits v0.8 OPEN-I): single-witness SDF may be insufficient for go2 foot-on-sand resting stability — may need perturbed-restart multi-point.

---

# R5 — Volumetric FEM (corotational → stable Neo-Hookean, implicit BE on the self-written CG)

**Algorithm (research-fixed):** tetrahedral FEM, implicit backward-Euler; corotational linear (Müller & Gross 2004) → stable Neo-Hookean (Smith, de Goes, Kim 2018) (`research-breadth-solvers.md:60-62`). Best fit for Nuka's existing self-written CG (`sparse_solver_cg.hpp` — already D1: fp64 internal, fixed-order warp-shuffle butterfly reductions, no atomics, `research-breadth-solvers.md:64`). Co-step like MPM (FEM owns its implicit solve), contact via existing particle-manifold rows + id13. **Zero new row classes** (§0.2). New module `src/runtime/fem/fem_world.{hpp,cu}`. Reuses the tet topology infra (`tetmesh_topology.hpp`).

### R5a — Tet element force + corotational stiffness (per-tet, parallel)
- **Objective.** Per-tet elastic force + corotational stiffness warping.
- **Technical approach.** Corotational (Müller 2004, `research-breadth-solvers.md:60`): per-tet polar-decompose F=RS, rotate the constant linear stiffness K₀ by R → `f = RK₀(Rᵀx − x₀)`. Per-tet parallel. Reuse tet topology (`tetmesh_topology.hpp` — gives the 4-vertex tets + rest geometry). New `FemTetElement` record (rest shape matrix, K₀, material λ/μ).
- **Registers into:** (FEM element pass; feeds R5b assembly).
- **Inputs/Outputs/Interface.** In: tet vertices + rest. Out: per-tet force + stiffness. Interface: R5b.
- **Dependencies.** `tetmesh_topology` (exists).
- **D1 strategy.** Forward-only; per-tet own-output (no atomics); fixed-iteration analytic polar decomposition (branch-free, `research-breadth-solvers.md:64`). Two-run.
- **Validation/test gates.** `tests/runtime/test_fem_corotational.cpp` — a single tet under uniaxial stretch returns the analytic linear-elastic force; large-rotation rigid motion produces zero internal force (the corotational invariance test); 2-run.
- **Effort.** **M**.
- **Risks.** Polar-decomposition determinism near-degenerate F; corotational ghost-force artifacts (known limitation, acceptable for forward-only demo).

### R5b — Backward-Euler assembly + the self-written CG solve
- **Objective.** Assemble `(M/dt² + K)Δx = −∇E` per Newton iteration and solve with the existing CG.
- **Technical approach.** Backward-Euler (`research-breadth-solvers.md:62`): assemble the system (matrix-free `Ax` or CSR), solve with the self-written CG (`SelfWrittenCgBackend`, `sparse_solver_cg.hpp` — REUSE; it is exactly the SPD-system D1 solver this needs). Residual assembly into the global vector = **gather (one thread per vertex over its incident tets in sorted order)**, NEVER scatter-atomic (`research-breadth-solvers.md:64`). Fixed CG iteration count (or deterministic residual-floor early exit, which the CG already has).
- **Registers into:** the **self-written CG** (reuse, not extend); FEM owns its implicit solve (co-step, not the row solver).
- **Inputs/Outputs/Interface.** In: per-tet force/stiffness + mass + dt. Out: vertex position update. Interface: R5c, R5d.
- **Dependencies.** R5a; `sparse_solver_cg` (exists, D1).
- **D1 strategy.** Forward-only; CG is D1 (fp64, fixed-order butterfly reductions, no atomics — `research-breadth-solvers.md:64`); gather assembly (no scatter-atomic); fixed iteration. Two-run + N≥32.
- **Validation/test gates.** `tests/runtime/test_fem_implicit_be.cpp` — a **cantilever beam** under gravity deflects to the analytic Euler-Bernoulli tip deflection within tol (the FEM oracle); energy non-increasing (V2); 2-run + N≥32.
- **Effort.** **M–L**.
- **Risks.** The Delassus/SPD system the existing CG solves is per-articulation-block (`sparse_solver_cg.hpp` — one warp per ≤12×12 block); the FEM global system is bigger and different topology — may need a CG variant (matrix-free `Ax` over the FEM mesh) rather than the block-dense path. Flag: the existing CG may need a FEM-shaped front-end (OPEN-V2).

### R5c — Stable Neo-Hookean upgrade (robust nonlinear)
- **Objective.** The inversion-safe hyperelastic energy with analytic SPD Hessian projection (the modern robust nonlinear choice over corotational).
- **Technical approach.** Stable Neo-Hookean (Smith 2018, `research-breadth-solvers.md:61`): inversion-safe energy with closed-form eigensystem → analytic SPD Hessian projection (no clamp hacks). Replaces R5a's corotational stiffness in the R5b Newton loop. Per-tet eigen-projection (fixed-iteration analytic, branch-free).
- **Registers into:** (swaps the constitutive model inside R5b's solve).
- **Inputs/Outputs/Interface.** In: tet F. Out: SPD-projected element Hessian + force. Interface: R5b.
- **Dependencies.** R5a, R5b.
- **D1 strategy.** Forward-only; analytic eigen-projection (fixed-order, branch-free); same gather assembly. Two-run + N≥32.
- **Validation/test gates.** `tests/runtime/test_fem_stable_neohookean.cpp` — an inverted tet (negative-volume initial config) recovers (does NOT explode — the inversion-safety property corotational lacks); cantilever still matches the deflection oracle; 2-run.
- **Effort.** **M** (on top of R5b).
- **Risks.** Eigensystem determinism near-degenerate singular values; this is the L-tier robustness in `research-breadth-solvers.md:66`.

### R5d — Damping + dynamics integration
- **Objective.** Velocity update + (optional) Rayleigh damping for stable soft dynamics.
- **Technical approach.** Standard implicit-BE velocity update `v = (x − x_prev)/dt`; optional Rayleigh damping folded into the assembled system. Per-vertex own-output.
- **Registers into:** (FEM step closure).
- **Inputs/Outputs/Interface.** In: position update. Out: velocities. Interface: next step; R5e contact.
- **Dependencies.** R5b.
- **D1 strategy.** Forward-only; per-vertex own-output; no atomics. Two-run.
- **Validation/test gates.** `tests/runtime/test_fem_dynamics.cpp` — a soft cube dropped and bouncing conserves/dissipates energy monotonically (V2); 2-run.
- **Effort.** **S–M**.
- **Risks.** Low.

### R5e — `FemTet` collidable + FEM↔rigid via id13
- **Objective.** Register FemTet (CollidableType=7); FEM↔rigid contact through the unified manifold + id13.
- **Technical approach.** Append `CollidableTypeInfo` for FemTet: AABB = per-tet, accel = particle grid over vertices, proxy = tet (surface triangles for contact), reaction = `ParticleInvMass` bary-weighted across the 4 verts (surface verts for surface contact, `research-genesis.md:31` FEM↔rigid is surface-vertex). FEM surface verts register as particle pseudo-bodies → unified manifold → **id13 `CouplingContactRow`** (no new row class). Two-way via the providers.
- **Registers into:** the **collidable-type registry** (FemTet=7); the **id13 coupling framework**; the narrowphase (surface tris).
- **Inputs/Outputs/Interface.** In: FEM surface + rigid. Out: id13 coupling rows. Interface: R8 FEM↔rigid; soft-tissue completeness (no named v1.0 demo).
- **Dependencies.** R5a–R5d (FEM solve), v0.8 C6b (id13), C3 (narrowphase), C4c (providers).
- **D1 strategy.** Forward-only; surface-vert manifold + id13 rows D1; bary-weighted gather. Two-run + N≥32.
- **Validation/test gates.** `tests/runtime/test_fem_rigid_coupling.cpp` — a soft cube squished by a rigid plate deforms AND the plate feels reaction (two-way); no surface penetration; 2-run.
- **Effort.** **M**.
- **Risks.** FEM↔rigid surface-only misses edge/volume contact (the Genesis limitation, `research-genesis.md:82`) — acceptable for forward-only completeness; document.

---

# R7 — Tendon / muscle / actuator (MuJoCo-parity)

**Thinnest-researched phase** (the breadth-solver research skips tendon — advisor flag). Anchored to MuJoCo's documented actuation model (no per-line code citation available — cite MuJoCo docs: modeling.html#tendons, #actuators; the same `mjModel` model Newton wraps via `SolverMuJoCo`, `research-newton-mujocowarp.md:21`). Three layers: routed/fixed tendons, Hill-type muscle, transmission. Two new row classes (id16/17, §0.2) over articulation DOFs; no collidable, no narrowphase. Reuses the articulation + constraint-row infra (`src/runtime/articulation/`, `joint_constraints.hpp`).

### R7a — `FixedTendonRow` (id16): length = Σ joint coefficients
- **Objective.** Fixed tendons (a linear combination of joint coordinates) with length limits — the MuJoCo `<fixed>` tendon.
- **Technical approach.** Fixed tendon length `L = Σ_k coef_k · q_k` (a linear joint-coordinate combination — MuJoCo fixed tendon). As an EQUALITY/limit row over the participating joint DOFs: the Jacobian is the constant coefficient vector (`jacobian_kind: featherstone_chain_scalar`, `body_count_mode: variable`). New codegen class `fixed_tendon.yaml` (`row_class_id: 16`, variable bodies, equality, `gradient_mode: none`). Length limits (`<limited>` with range) emit a unilateral row when at the stop. Reuse the joint-constraint emission path (`joint_constraints.hpp`).
- **Registers into:** RowClassId id16; the `UnifiedSolve` row stream (co-solved with joint rows); the articulation Jacobian path.
- **Inputs/Outputs/Interface.** In: joint coeffs + DOF indices + limits. Out: tendon-length constraint rows. Interface: R7c muscle/transmission; manipulation demos.
- **Dependencies.** articulation infra (exists); v0.8 C5a (UnifiedSolve, for the row to be co-solved with joint rows).
- **D1 strategy.** Forward-only; constant-Jacobian row, fixed-order; no atomics. Two-run + N≥32.
- **Validation/test gates.** `tests/runtime/test_fixed_tendon.cpp` — a coupled-joint mechanism (e.g. a finger with a coupling tendon) maintains `L = Σ coef·q` within tol; the limit stops at range; 2-run. Cross-check the length law vs a hand-computed MuJoCo value.
- **Effort.** **M**.
- **Risks.** Variable-body-count row through the scheduler (the existing rows are mostly fixed 2/4-body); verify the CSR variable-body path. Low algorithmic risk (linear).

### R7b — `SpatialTendonRow` (id17): routed/via-point tendons
- **Objective.** Spatial (routed) tendons through via-points / wrapping geometry — the MuJoCo `<spatial>` tendon (length = polyline through sites/wrap surfaces).
- **Technical approach.** Spatial tendon length = sum of segment lengths through ordered via-points (sites on bodies) + wrapping-geometry detours (cylinder/sphere wrap). Length-Jacobian = the per-via-point geometric Jacobian (each segment's unit vector projected onto the site Jacobians). New codegen class `spatial_tendon.yaml` (`row_class_id: 17`, variable bodies, unilateral/equality, `gradient_mode: none`). Via-point sites cooked from MJCF `<spatial>` (importer extension — adjacent to v0.8 C1b). Wrapping geometry (the hard part) deferred behind via-points-only first.
- **Registers into:** RowClassId id17; the `UnifiedSolve` row stream; the articulation Jacobian path.
- **Inputs/Outputs/Interface.** In: ordered via-point sites + (optional) wrap geom. Out: routed-length rows. Interface: R7c.
- **Dependencies.** R7a (shares the tendon emission), articulation Jacobian (exists).
- **D1 strategy.** Forward-only; fixed via-point order; per-segment Jacobian fixed-order; no atomics. Two-run + N≥32.
- **Validation/test gates.** `tests/runtime/test_spatial_tendon.cpp` — a via-point tendon over a pulley site computes the analytic routed length within tol; 2-run.
- **Effort.** **M–L** (via-points = M; wrapping geometry = the L/R uncertainty — defer).
- **Risks.** **Highest uncertainty in R7.** Wrapping-geometry (cylinder/sphere wrap) is geometrically involved and not researched here — DEFER to via-points-only; flag wrapping as a v1.0+ seam (OPEN-V3). Routed-length derivative determinism at via-point transitions.

### R7c — Hill-type muscle + transmission (moment arm)
- **Objective.** Hill-type muscle actuator (force-length-velocity) + transmission (moment arm from tendon length gradient) — MuJoCo muscle parity.
- **Technical approach.** Hill-type muscle (MuJoCo `gainprm`/`biasprm` force-length-velocity curves): actuator force = `f(activation, L, L̇)` along the tendon; transmission moment arm = `∂L/∂q` (the tendon-length Jacobian from R7a/R7b — already computed). The muscle force enters as an actuation on the tendon's DOFs (a generalized force, NOT a constraint row — it routes through the existing drive/actuation path, like a force-mode actuator). Activation dynamics (first-order) optional.
- **Registers into:** the existing actuation/drive path (not a new row class — muscle force is a generalized force on DOFs via the moment arm); reuses the R7a/R7b length Jacobian.
- **Inputs/Outputs/Interface.** In: activation + tendon length/velocity + muscle params. Out: generalized actuation force. Interface: muscle-driven demos (no named v1.0 demo; MuJoCo-parity completeness).
- **Dependencies.** R7a (fixed) and/or R7b (spatial) for the moment arm.
- **D1 strategy.** Forward-only; per-actuator force (closed-form curves, branch-free); fixed-order. Two-run.
- **Validation/test gates.** `tests/runtime/test_hill_muscle.cpp` — the force-length curve matches the MuJoCo Hill formula at sampled lengths within tol; a 1-DOF muscle-driven joint reaches equilibrium at the expected activation; 2-run.
- **Effort.** **M**.
- **Risks.** Hill-curve parameterization fidelity vs MuJoCo (many `gainprm`/`biasprm` conventions); activation dynamics stability. Forward-only so no adjoint complexity.

---

# R8 — Coupling per-pair (instantiate concrete pairs on the v0.8 id13 framework)

The v0.8 spine ships the **generic id13 `CouplingContactRow` framework + `EmitCouplingRows` + co-step bridge** and (per OPEN-E) the K2 fluid↔rigid proving pair. R8 INSTANTIATES the remaining concrete pairs — **each as a registry entry (a registered CollidableType + reaction provider + an `EmitCouplingRows` invocation), NOT a new kernel/row id** (the Genesis O(N²)-whitelist win, `research-genesis.md:96`; advisor-confirmed default). Genesis `_func_collide_in_rigid_geom` is the per-pair impulse-exchange model (`research-genesis.md:27-35`), but made a solver-resolved row (implicit, stable) not Genesis's explicit `−Δmv/dt`.

**Honest scoping (advisor):** R8 rigid↔fluid LARGELY OVERLAPS v0.8 C6b's proving pair (a fluid particle pushing a rigid box). The genuinely-NEW pairs needing R8 work are **articulated↔MPM (go2-sand, depends R4)** and **cloth↔rigid/articulated (depends R3)**. R8 = thin per-pair instantiations that TRAIL each partner solver.

### R8a — rigid↔fluid + fluid↔articulated (harden C6b's pair → full)
- **Objective.** Complete the rigid↔fluid pair beyond C6b's proving case (multi-point, articulation side) for the fluid+rigid v1.0 demo.
- **Technical approach.** PBF particles (`pbf_world`) ↔ rigid/articulation via id13: candidate pairs (C2c) → SDF-tier manifolds → `EmitCouplingRows`. The articulation side uses the C5b two-way artic↔rigid reaction generalized to particle↔articulation. No new row class.
- **Registers into:** the **id13 framework** (a registry instantiation); the Particle + Rigid/ArticulationLink reaction providers.
- **Inputs/Outputs/Interface.** In: PBF particles + rigid/articulated bodies. Out: id13 coupling rows. Interface: **fluid+rigid demo (demo 3)**.
- **Dependencies.** v0.8 C6b (id13 + K2 proving pair), C5b (two-way), PBF (exists).
- **D1 strategy.** Forward-only; id13 rows D1; gather reductions; no float atomics. Two-run + N≥32.
- **Validation/test gates.** `tests/runtime/test_rigid_fluid_coupling.cpp` — poured water pushes a floating object; momentum balance within tol; 2-run + N≥32.
- **Effort.** **S–M** (mostly C6b reuse).
- **Risks.** Largely upstream (C6b). Multi-point fluid-vs-rigid resting stability (the OPEN-I single-witness SDF cardinality).

### R8b — articulated↔MPM (the go2-on-sand coupling)
- **Objective.** Two-way coupling between the go2 articulation and MPM sand — the headline R4+R8 demo.
- **Technical approach.** go2 feet/links (ArticulationLink collidables) interact with MPM sand: rigid links read as MPM grid BCs (rigid SDF, R4f), grid→link reaction via the `MpmGridNodal` reaction provider (R4f) summed-nodal-force, routed to the articulation DOFs via the C5b two-way artic-chain-J reaction. Emits id13 rows (or the impulse-field path per R4f OPEN-V1).
- **Registers into:** the **id13 framework** + the **MpmGridNodal reaction provider** (R4f) + the **C5b two-way artic↔X** path.
- **Inputs/Outputs/Interface.** In: go2 articulation + MPM grid. Out: two-way coupling. Interface: **MPM go2-on-sand demo (demo 5)**.
- **Dependencies.** R4f (MpmParticle collidable + reaction provider), v0.8 C5b (two-way artic), C6b (id13/co-step).
- **D1 strategy.** Forward-only; summed-nodal-force in fixed gather order (no float atomics); artic-chain-J reaction D1. Two-run + N≥32.
- **Validation/test gates.** `tests/oracle/test_go2_sand` — go2 walks on sand: feet sink+push, sand displaces, gait is stable (forward progress); 2-run bit-identity of the trajectory; N≥32. This is the integration capstone of R4+R8.
- **Effort.** **L**.
- **Risks.** Foot-on-sand resting stability (OPEN-V4 single-witness SDF); high-mass-ratio artic↔grid stability (implicit row helps vs Genesis explicit). The capstone — failures here diagnose R4.

### R8c — cloth↔rigid + cloth↔articulated (robot dressing)
- **Objective.** Two-way cloth↔rigid/articulation for the garment/dressing scenarios.
- **Technical approach.** R3d already registers ClothTriangle + cloth↔rigid id13. R8c adds the ARTICULATION side (cloth draped/grasped by an articulated hand) via the C5b two-way artic reaction + the ClothTriangle bary-weighted provider. No new row class.
- **Registers into:** the **id13 framework** + ClothTriangle (R3d) + the C5b two-way artic path.
- **Inputs/Outputs/Interface.** In: cloth + articulation. Out: id13 rows. Interface: robot-dressing (completeness; demo-4 cloth showcase uses BASIC XPBD — see map).
- **Dependencies.** R3d (ClothTriangle), v0.8 C5b, C6b.
- **D1 strategy.** Forward-only; id13 + artic-chain-J D1. Two-run + N≥32.
- **Validation/test gates.** `tests/runtime/test_cloth_artic_coupling.cpp` — an articulated gripper lifts a cloth corner (cloth follows, gripper feels weight); 2-run.
- **Effort.** **M**.
- **Risks.** Cloth↔articulation grasp is a thin-shell-vs-fingertip contact (CCD-adjacent, depends R3c); high contact density.

### R8d — FEM↔fluid / MPM↔FEM (completeness, opportunistic)
- **Objective.** The remaining cross-continuum pairs for full coverage (opportunistic — only if the demos need them).
- **Technical approach.** Each is another id13 instantiation over the already-registered collidable types + reaction providers. MPM↔FEM exchanges momentum via the MPM grid (Genesis model, `research-genesis.md:31`). NO new row classes.
- **Registers into:** the **id13 framework** (pure instantiations).
- **Inputs/Outputs/Interface.** In: the two partner systems. Out: id13 rows. Interface: completeness (no named demo).
- **Dependencies.** R4f, R5e, C6b.
- **D1 strategy.** Forward-only; id13; gather. Two-run + N≥32.
- **Validation/test gates.** `tests/runtime/test_cross_continuum_coupling.cpp` — momentum balance per pair; 2-run.
- **Effort.** **S–M each** (the framework payoff: each new pair is a registry entry).
- **Risks.** Low (the framework's whole point). Skip if no demo needs it; not on the critical path.

---

# R6 — Closed kinematic loops (Kamino) — DEFERRED

**⏸ DEFERRED** (owner 2026-06-03: "kamino先不做", `post-v07-roadmap.md:50,85`). NOT decomposed. The gap is real (Nuka's reduced-coord Featherstone is tree-only; `SolverKamino` = Disney maximal-coord + Proximal-ADMM for closed loops, arXiv:2504.19771). **Reserved seam:** the v0.8 **solver-registry slot** (v0.8 §0.6, Appendix B) is the plug point — when revived, SolverKamino consumes the SAME row stream/manifolds (`UnifiedSolve` SolveContext) alongside the row-PGS, exactly as Newton's `step(...,contacts,...)` contract allows a second solver (`research-newton-mujocowarp.md:11`). The slot exists now; the solver does not. No v0.9 work.

---

# U2 — Scene authoring / inspection

Panels over the C-ABI for hierarchy + transforms + physics/material params + live tweak (`post-v07-roadmap.md:67`). Builds on v0.8 U1's interactive Vulkan viewport.

### U2a — Scene-graph introspection + transform editing
- **Objective.** A hierarchy panel (bodies/joints/shapes) with live transform editing over the C-ABI.
- **Technical approach.** Expose SceneIR hierarchy via the C-ABI (`src/c_abi/`); a panel walks `scene_graph` (`scene/scene_graph.hpp`) showing the body/joint tree; transform edits write back through the public state path (no solver-internal hack). Reuse `BuildRenderScene` (`render_scene.hpp:80`) to reflect edits live in the U1 viewport.
- **Registers into:** the C-ABI + U1 viewport (no physics seam).
- **Inputs/Outputs/Interface.** In: SceneIR + user edits. Out: live-updated scene. Interface: U1 viewport, U2b params.
- **Dependencies.** v0.8 U1 (viewport), C-ABI + scene_graph (exist).
- **D1 strategy.** N/A (an editor of D1 sim state, not a sim producer; edits are user input — document that edited sessions aren't replay-D1 unless edits are recorded, the v0.8 U1b posture).
- **Validation/test gates.** `tests/c_abi/test_scene_introspection.cpp` — the C-ABI returns the correct hierarchy for a known scene; a transform edit round-trips to SceneIR.
- **Effort.** **M**.
- **Risks.** Editing live state during sim must use the public path; gating edits to paused state is safest.

### U2b — Physics/material parameter inspection + live tweak
- **Objective.** Introspect + edit physics params (mass, friction, solref/solimp, material color/roughness) live.
- **Technical approach.** Extend the C-ABI with param getters/setters over `MaterialRecord` (the v0.8 contact params, `scene_ir.hpp:80`) + body inertials; panel binds them. solref/solimp tweak re-cooks the affected `CookedContactParamTable` (v0.8 C1a) entry.
- **Registers into:** the C-ABI + the v0.8 cooked contact-param table (read/write).
- **Inputs/Outputs/Interface.** In: material/body params + edits. Out: re-cooked params. Interface: U1 viewport (live effect).
- **Dependencies.** U2a, v0.8 C1a (contact-param table).
- **D1 strategy.** N/A (param editing is input).
- **Validation/test gates.** `tests/c_abi/test_param_tweak.cpp` — editing friction changes the cooked param + the next-step contact behavior; round-trip.
- **Effort.** **M**.
- **Risks.** Live re-cook mid-sim consistency; restrict heavy re-cooks to pause.

---

# U3 — Web / log viewer

A lightweight shareable viewer (lower-effort than a Kit clone) — Rerun-style log viewer or three.js/WebGPU web page; Jupyter inline for notebooks (`post-v07-roadmap.md:68`).

### U3a — Frame/log recording format + writer
- **Objective.** A compact recorded-trajectory + AOV log format the viewer reads (sharable, replayable).
- **Technical approach.** A self-written log (per-frame body transforms + optional framebuffer AOVs `framebuffer.hpp:36`) written via the bindings; deterministic record so a logged run replays bit-identically (rides D1). DLPack-exportable AOV channels (`framebuffer.hpp:16`) feed the log directly.
- **Registers into:** the bindings + framebuffer (no physics seam).
- **Inputs/Outputs/Interface.** In: per-frame state + AOVs. Out: a log file. Interface: U3b viewer.
- **Dependencies.** bindings, framebuffer (exist).
- **D1 strategy.** The recorded sim is D1 (the log captures the D1 trajectory); the log format itself is deterministic byte layout. 2-run identical logs.
- **Validation/test gates.** `tests/c_abi/test_log_roundtrip.cpp` — a recorded log replays to byte-identical state; AOV channels round-trip.
- **Effort.** **M**.
- **Risks.** Log size (AOVs are heavy) — gate AOV recording behind a flag.

### U3b — Web/Jupyter viewer (three.js/WebGPU or Rerun-style)
- **Objective.** A browser/notebook viewer rendering the U3a log (transforms + optional RT frames).
- **Technical approach.** A three.js/WebGPU page (or a Rerun-style log viewer) that loads the U3a log; mesh instances from the scene + per-frame transforms; optional RT frames as image planes. Jupyter inline cell for notebook demos. No engine dependency in the browser (replays the log).
- **Registers into:** (frontend; consumes U3a log).
- **Inputs/Outputs/Interface.** In: U3a log. Out: a browser/notebook view. Interface: shareable demos / homepage.
- **Dependencies.** U3a.
- **D1 strategy.** N/A (a viewer).
- **Validation/test gates.** Manual-QA (browser); a smoke test that the log parses + the scene loads.
- **Effort.** **M**.
- **Risks.** WebGPU browser support; keep a canvas/WebGL fallback. Web tooling (three.js) is fine under the no-closed-SDK pillar (it is a viewer lib, not a solver/SDK).

---

# U4 — Real complex-USD asset pipeline

Import real complex USD scenes for demos (the p16 requirement; `post-v07-roadmap.md:69`). Hardens the hand-rolled importer + mesh-geometry retention (#19) + the usdc binary seam. Builds on the v0.7 `scene_compose` (`scene/scene_compose.hpp`) coexistence layer.

### U4a — usdc binary reader (self-written) + references-follow
- **Objective.** Read real complex USD (often usdc binary + composition `references`/`payloads`), not just ASCII usda — the gap blocking real-asset demos (memory `v07-usd-mjcf-coexistence`).
- **Technical approach.** Self-written usdc (crate) binary reader (no OpenUSD — the no-closed-SDK pillar) on the `usd_stage_adapter` seam (memory: "self-written reader, no OpenUSD"); follow `references`/`payloads`/sublayers composition. Mesh-geometry read (#19 STL/OBJ already landed in v0.7; add USD `points`/`faceVertexIndices`). Output the same SceneIR both importers produce → `scene_compose` (exists) composes.
- **Registers into:** the importer → SceneIR → `scene_compose` (no physics seam).
- **Inputs/Outputs/Interface.** In: real USD (usdc + references). Out: SceneIR. Interface: scene_compose, cooker.
- **Dependencies.** importer/cooker + scene_compose (exist); #19 mesh loader (landed).
- **D1 strategy.** Deterministic parse (fixed traversal order); host-only cook. Cook-twice memcmp.
- **Validation/test gates.** `tests/import/test_usdc_reader.cpp` — a usdc file with references parses to the expected SceneIR; mesh points/indices correct; cook-twice byte-equal.
- **Effort.** **M–L** (usdc binary format + composition is involved).
- **Risks.** usdc is a complex binary format (crate, versioned); scope to the subset the target demo assets use (newton-assets kitchen/cup, memory `newton-assets-resource`); document unsupported features. Composition arcs (variants/inherits) may be deep — scope to references/payloads first.

### U4b — Real-asset demo scene assembly (kitchen + cup coexistence)
- **Objective.** Assemble a real complex demo scene (USD cup on MJCF kitchen counter — the owner must-do, memory `v07-usd-mjcf-coexistence`) cooked + sim-ready.
- **Technical approach.** Load the newton-assets kitchen (MJCF+OBJ) + cup (USD mesh) via U4a + the existing MJCF importer; `scene_compose` (exists) merges them at a placement; cook; verify physics-ready (collision geometry retained per memory `v07-integration-debt-discipline` mesh-retention orphan). The H1 grasp (v0.8 C7) uses this scene.
- **Registers into:** scene_compose + cooker (consumes U4a output).
- **Inputs/Outputs/Interface.** In: kitchen + cup assets. Out: a cooked composed scene. Interface: the grasp + photoreal demos.
- **Dependencies.** U4a, scene_compose (exists), v0.8 C1 (cook), v0.8 C7 (grasp consumer).
- **D1 strategy.** Deterministic compose + cook (the `scene_compose.hpp:Compose` is "pure and deterministic (D1)"). Cook-twice memcmp.
- **Validation/test gates.** `tests/scene/test_real_usd_scene.cpp` — the composed kitchen+cup scene cooks; the cup has collision geometry; min-sep sane at rest; 2-run cook.
- **Effort.** **M**.
- **Risks.** Asset collision-geometry retention (the known mesh-retention orphan); asset scale/units (USD cm vs MJCF m) — verify the compose placement.

---

# U5 — Teleop / XR

Device input → robot retargeting (`post-v07-roadmap.md:70`); keyboard/spacemouse/VR abstraction over the v0.8 U1 viewport. Lower priority (later in the value order).

### U5a — Device-input abstraction (keyboard/spacemouse/VR)
- **Objective.** A device-input layer feeding teleop commands into the interactive viewport.
- **Technical approach.** An input abstraction (keyboard → spacemouse → VR controller) producing a 6-DOF target stream; routes through the v0.8 U1b interaction path (external-wrench / target, no solver hack). VR via OpenXR (an open standard — acceptable under no-closed-SDK; it is a device API, not a solver/SDK).
- **Registers into:** the v0.8 U1 viewport interaction path (no physics seam).
- **Inputs/Outputs/Interface.** In: device events. Out: 6-DOF target stream. Interface: U5b retargeting.
- **Dependencies.** v0.8 U1a/U1b (viewport + interaction).
- **D1 strategy.** N/A (input is non-deterministic by nature; the sim it drives stays D1; teleop sessions aren't replay-D1 unless inputs recorded — the U1b posture).
- **Validation/test gates.** `tests/render/test_input_abstraction.cpp` — keyboard/synthetic events produce the expected target stream; headless.
- **Effort.** **M**.
- **Risks.** VR hardware availability in CI (headless gate); OpenXR runtime dependency. Confirm OpenXR acceptable (OPEN-V5).

### U5b — Teleop retargeting to robot (IK-driven)
- **Objective.** Map the teleop 6-DOF stream to robot joint targets via IK.
- **Technical approach.** Feed the device 6-DOF target to the v0.8 R1 IK (DLS, `v08-detailed-tasks.md:399`) as the end-effector objective → joint targets → drive rows (id2). Real-time per-frame IK in the viewport loop.
- **Registers into:** v0.8 R1 IK (consumer) + the drive-row path.
- **Inputs/Outputs/Interface.** In: 6-DOF teleop target. Out: joint targets. Interface: live teleop of a robot.
- **Dependencies.** U5a, v0.8 R1a (IK).
- **D1 strategy.** N/A (teleop input); the IK + sim are D1 given recorded input.
- **Validation/test gates.** `tests/runtime/test_teleop_retarget.cpp` — a synthetic teleop path drives the H1 hand to track the target via IK within tol; 2-run given recorded input.
- **Effort.** **M**.
- **Risks.** IK singularities during teleop (the DLS λ damping handles); latency in the viewport loop.

---

## Appendix A — Dependency DAG / suggested commit order

Each phase-task is one commit. v0.9 has FOUR largely-independent solver tracks (R2, R3, R4, R5) that each END in an R8 coupling instantiation, plus R7 (independent of the spine), plus the U-frontend track. **R2a is BLOCKED on v0.8 OPEN-C** (id allocation). R3c (ACCD) is shared infra (R2 + R3 consume it).

```
                     [v0.8 SPINE COMPLETE: C1..C7, R1 IK, id13 framework]
                                          │
  ┌──────────────┬──────────────┬────────┴───────┬──────────────┬─────────────┐
  ▼              ▼              ▼                 ▼              ▼             ▼
R2 (cable)     R3 (cloth)     R4 (MPM)         R5 (FEM)       R7 (tendon)   U-track
R2a→R2b→R2c    R3a─┐          R4a→R4b→R4c→      R5a→R5b→       R7a→R7b→R7c   U2a→U2b
   │  (needs    R3b─┤            R4d→R4e→R4f      R5c→R5d→         (indep)    U3a→U3b
   │  R3c CCD)  R3c (ACCD)◀──┐      │             R5e              of spine)  U4a→U4b
   │            R3d          │      │              │                          U5a→U5b
   │             │           │      ▼              ▼
   └─────────────┴───────────┘    R8b           R8c (cloth↔artic, needs R3)
        R3c shared (R2 knots,    (go2-sand,
         R3 self-collision)       needs R4f+C5b)
                  │                  │
                  ▼                  ▼
   R8a (rigid↔fluid, ~C6b)   ──▶  [R8d FEM↔fluid/MPM↔FEM — opportunistic]
```

**Recommended serial commit order (single branch):**
`R7a, R7b, R7c` (independent, low risk, warm-up) →
`R2a, R2b, R2c` (cable; after OPEN-C) →
`R3a, R3b, R3d, R3c` (cloth; R3c ACCD last — the hard one) →
`R8a, R8c` (fluid+rigid, cloth+artic) →
`R4a, R4b, R4c, R4d, R4e, R4f` (MPM; R4b P2G is the gate) →
`R8b` (go2-sand capstone) →
`R5a, R5b, R5c, R5d, R5e` (FEM) →
`R8d` (opportunistic) →
U-track `U2a, U2b, U3a, U3b, U4a, U4b, U5a, U5b` (parallel-izable; U4 unblocks real-asset demos, prioritize for v1.0).

**Critical paths:** (1) **go2-sand** = R4a→R4b→...→R4f→R8b (the longest, highest-risk chain; **R4b gather-P2G is the determinism gate, rated R**). (2) **cable demo** = R2a→R2b→R2c (+R3c for knots). (3) **R3c ACCD (rated R)** is the joint-hardest D1 task and gates robust cloth + cable knots — schedule slack on both R-rated items. The U-track and R7 do not block any solver.

## Appendix B — Capability → v1.0 demo mapping (HONEST)

The v1.0 demos (`post-v07-roadmap.md:88-94`). **Honesty (advisor):** not every breadth solver maps to a named demo — several are parity/completeness with NO named demo, and the doc must not overclaim.

| v0.9 capability | v1.0 demo it unblocks | Notes |
|---|---|---|
| **R2 cable/DER** (R2a–c) | **Demo 6 — Cable/rope** | direct; +R3c CCD for tight knots |
| **R4 MPM + R8b articulated↔MPM** | **Demo 5 — MPM go2-on-sand** | the R4+R8 headline; longest critical path |
| **R8a rigid↔fluid** | **Demo 3 — Fluid+rigid coupling** | largely v0.8 C6b reuse; R8a hardens it |
| **R3 garment cloth** (R3a–d) | **NO named v1.0 demo** | Demo 4 (soft/cloth showcase) uses **BASIC XPBD p09, NOT garment-grade** (`post-v07-roadmap.md:92`). R3 = completeness/parity vs Newton Style3D/VBD + robot-dressing capability; the SOTA cloth gap-closer, but not a gated demo. |
| **R5 volumetric FEM** (R5a–e) | **NO named v1.0 demo** | constitutive soft-tissue / soft-robot completeness; gap-closer vs PhysX/Genesis FEM. No gated demo. |
| **R7 tendon/muscle** (R7a–c) | **NO named v1.0 demo** | MuJoCo-parity actuation completeness; enables muscle-driven characters but not a gated demo. |
| **R3c ACCD CCD seam** | (infrastructure) | enables R2 knots + R3 self-collision; no demo of its own |
| **R8c cloth↔articulation** | (robot-dressing capability) | completeness; could feed a future dressing demo, not in the v1.0 six |
| **U4 real-USD pipeline** | **Demo 1 — H1 grasp** (asset side) + homepage | real kitchen+cup scene for the grasp + photoreal showcase |
| **U2/U3/U5** | (usability, homepage shareable) | authoring + web viewer + teleop; support the homepage, no gated physics demo |

(Demo 1 H1-grasp + Demo 2 rigid-collision are v0.8-spine demos, not v0.9 — listed only where U4 feeds the grasp's assets.)

## Appendix C — D1 / forward-only discipline (the cross-cutting gate)

**Forward-only ≠ no determinism.** Every v0.9 solver is forward-only (no adjoint, `gradient_mode: none`, no `adjoint_evaluator` YAML block) per `post-v07-roadmap.md:107` — but **full D1 is non-negotiable** (`post-v07-roadmap.md:106`). Each task's D1 gate = **two-run bit-identity + N≥32 cross-replica identity**, AND an analytic oracle where one exists:

| solver | analytic oracle (the correctness anchor, beyond D1) |
|---|---|
| R2 cable | **catenary** (hanging rod) + **Kirchhoff helix** (twist) |
| R3 cloth | **cantilever deflection** (bend) + max-strain bound (strain-limit) + zero-intersection (CCD) |
| R4 MPM | **angle of repose** (Drucker-Prager sand) + mass/momentum conservation (P2G) |
| R5 FEM | **Euler-Bernoulli cantilever** deflection + inversion recovery (stable Neo-Hookean) |
| R7 tendon | MuJoCo Hill-curve values + routed-length geometry |

The binding D1 rules (verified posture, `research-breadth-solvers.md:11`): `thrust::stable_sort_by_key` (radix); **integer atomics ONLY, NO float atomicAdd**; **gather-not-scatter** with fixed-order `__fadd_rn`; two-pass count→scan→private-slice (no append atomics); own-index quaternion/position projection. The two HARDEST D1 items: **R4b gather-form P2G** (the headline hazard) and **R3c ACCD TOI min-reduction** (fixed-order min with stable-key tie-break).

## Appendix D — OPEN questions (for owner / advisor)

These are v0.9 DECOMPOSITION forks + the v0.8 OPENs v0.9 inherits.

- **OPEN-V1 (R4f / §0.2):** does the MPM↔rigid two-way reaction express as an id13 `CouplingContactRow` (the framework's intent — recommend), or does the grid-BC handshake genuinely require the SDF-field-in/impulse-field-out external-wrench path (a non-row coupling routed through the registered reaction provider)? The grid is not a point-contact manifold — this is the one place a pair might escape id13. Recommend: id13 where possible, impulse-field fallback documented per-pair.
- **OPEN-V2 (R5b):** the existing self-written CG is per-articulation-block (≤12×12, one warp/block, `sparse_solver_cg.hpp`); the FEM global system is larger/different-topology. Does R5b reuse the CG with a FEM-shaped matrix-free `Ax` front-end (recommend), or does FEM need its own CG variant? Confirm the existing CG's D1 reduction posture ports to the bigger system.
- **OPEN-V3 (R7b):** spatial-tendon **wrapping geometry** (cylinder/sphere wrap) is unresearched + geometrically involved. Defer to via-points-only in v0.9 + flag wrapping as a v1.0+ seam (recommend), or attempt wrapping now (R-effort)?
- **OPEN-V4 (R4f/R8b, inherits v0.8 OPEN-I):** single-witness SDF contact (the v0.8 cardinality gap) for go2 foot-on-sand resting stability — sufficient, or does R4f need perturbed-restart multi-point SDF for stable footing?
- **OPEN-V5 (U5a):** OpenXR for VR teleop — acceptable under the no-closed-SDK pillar (it is an open device API, not a solver/render SDK)? And the headless-CI gate for VR.
- **Inherited v0.8 OPEN-C (id allocation):** v0.9 §0.2 ASSUMES the recommended resolution (id11/12 reserved → Cosserat, coupling=id13, count=14). **Must be closed before R2a.** If owner picks coupling=id11, R2 Cosserat renumbers to 13/14 and §0.2 shifts.
- **Inherited v0.8 OPEN-E (id13 shipped):** R8 assumes v0.8 ships the concrete id13 row + the K2 fluid↔rigid proving pair. If interface-only, R8a additionally lands the first concrete id13 emission.
- **Effort honesty:** **R3c (ACCD) and R4b (gather-P2G) are rated R (research-grade)** — the two hardest D1 items in all of v0.9. R4d (Drucker-Prager + analytic SVD), R5c (stable Neo-Hookean eigen-projection), R4f/R8b (go2-sand coupling), R5b, U4a (usdc reader) are the L-tier items where the estimate carries real but bounded uncertainty. R7b wrapping-geometry is deferred (OPEN-V3) precisely because it is unbounded-R. The U-track and R7a/c are the safe, low-risk warm-ups.
