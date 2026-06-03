# v0.9 Breadth-Solver Research — IK / Cable / MPM / FEM / Garment-Cloth / CCD

> **Status:** read-only research for the v0.9 design (controller-authored 2026-06-03). NOT a protected plan.
> **Scope:** the 6 breadth solvers slated for **v0.9** (post-v0.8). Each plugs into the **v0.8 unified GPU collision/contact spine** (`docs/plans/2026-06-03-v08-unified-collision-contact.md`): compliant solref/solimp contact, row-class constraints, **full D1** (forward-only — diff deferred), GPU-resident, self-written (no closed SDK).
> **Roadmap mapping** (`docs/plans/2026-06-03-post-v07-roadmap.md` §W1): topic 1=**R1 IK**, 2=**R2 cable/DER**, 3=**R4 MPM**, 4=**R5 FEM**, 5=**R3 garment cloth**, 6=**CCD seam** (a §3 extensibility hook, not a numbered W1 row). Effort keys reconcile with the roadmap column.
> **Comparators are Newton / MuJoCo-Warp / Genesis** (NOT Isaac Lab — Isaac defers physics to Newton/PhysX). Newton = `newton-physics/newton` (Warp, LF-governed, alpha→1.0 GTC-Mar-2026); MuJoCo-Warp = `google-deepmind/mujoco_warp` (GPU MuJoCo, ≠ MJX); Genesis = `Genesis-Embodied-AI/genesis-world` (unified multi-solver: Rigid/MPM/SPH/FEM/PBD + explicit coupler + SAP).

## Nuka substrate this rides on (verified in-repo)
- **Row IR** (`src/constraint/row.hpp`): `Row{row_class_id, body_count, body_list_offset, jacobian_offset, rhs, lambda, lower, upper, compliance_alpha, damping_beta, flags, …}`; flags `Equality/Unilateral/Friction/Coupled/GradActive`. Row classes today: 0 maximal-contact, 1 joint, 2 drive, 3 Featherstone-contact, 4 rigid-SDF-contact, 5 Featherstone-SDF-contact, 6 XPBD-distance, 7 XPBD-bend, 8 XPBD-volume, 9 XPBD-shape-match, 10 particle-particle-contact. New classes register via codegen YAML + `row_class_registry.hpp`.
- **Solver** (`src/solver/gpu/row_solver.cu`, `row_scheduler.*`): graph-colored PGS over CSR rows; `compliance_alpha=α/dt²` is the XPBD/compliant glue; solref/solimp added in v0.8/C4.
- **D1 posture (the binding rule, verified `particle_uniform_grid.cu`, `pbf_world.cu`, `coupling/particle_particle_contact.cu`):** (a) sorting = `thrust::stable_sort_by_key` (radix, deterministic); (b) **integer atomics ONLY** (order-independent counters); **NO float `atomicAdd`** in production; (c) accumulation = **gather-not-scatter** — one thread owns its output cell and sums over a *sorted-ascending* neighbor list with a **fixed-order `__fadd_rn` reduction** (no reassociation, no warp-shuffle tree that reorders); (d) two-pass count→exclusive-scan→fill into **private CSR slices** (never an append atomic); (e) Jacobi double-buffer for symmetric particle interactions. Validation = 2-run bit-identity + N≥32 cross-replica identity.
- **Reuse:** Featherstone Jacobians (`articulation/articulation_jacobian.cu`); self-written CG (v0.7 p01); particle uniform grid + LBVH broadphase; SDF query; XPBD predict/correct stepper.

---

## 1. Inverse Kinematics (R1, scheduled v0.8 — listed here for completeness)
**Canonical algorithm.** Numerical IK from the Featherstone task Jacobian `J(q) ∈ ℝ^{6m×n}` (m tasks, n DOFs). Three tiers:
- **Jacobian transpose** `Δq = αJᵀe` — cheapest, no inverse, slow/biased.
- **Damped least squares (DLS / Levenberg-Marquardt)** `Δq = Jᵀ(JJᵀ+λ²I)⁻¹ e` — SVD-robust through singularities; λ trades accuracy vs stability (Buss 2004; Wampler 1986; Nakamura/Hanafusa). **Selectively-damped LS** (Buss & Kim 2005) tunes λ per singular vector.
- **Task-space QP (Pink-style):** min over `v` of Σ wₖ‖Jₖv + αₖeₖ‖² s.t. joint vel/pos limits, multiple weighted/prioritized tasks (Caron's *Pink*; Kanoun stack-of-tasks). This is the multi-target path (reach + posture + look-at).

**Key papers.** Buss 2004 *Intro to IK (transpose/pseudoinverse/DLS)*; Buss & Kim 2005 *Selectively Damped LS*; Wampler 1986 (DLS origin); Caron *Pink* (differential IK by weighted tasks, QP).
**GPU strategy.** H1 has n≈30–50; the `(JJᵀ+λ²I)` solve is a small dense 6m×6m system — solve per-env with a batched dense Cholesky/LDLᵀ (self-written, fits a thread-block); QP via a few projected-Gauss-Seidel / active-set sweeps. Embarrassingly parallel across envs.
**D1.** Trivially D1: fixed iteration count, fixed pivot order in the small dense factorization, no atomics. Reuse the existing `__fadd_rn` math posture.
**Rows / spine.** IK is **NOT a constraint-row solver** — it runs *before* the dynamics step, emitting joint targets the existing **drive rows (id2)** track. Zero new row classes; consumes Featherstone Jacobians only. (Optionally formulate as equality "soft-attractor" rows for an in-solver variant, but the pre-step DLS is the cheap H1 path.)
**Effort.** **S–M** (DLS = S; Pink-QP multi-target = M).
**Comparators.** Newton ships `newton.ik` (Jacobian-based, GPU-batched). Isaac Lab: differential-IK + Pink QP + cuRobo (NVlabs cuRobo: GPU IK, 23× over TracIK, collision-free IK >7000 q/s). MuJoCo: `mj_jac` + user DLS; Genesis: via standard kinematics (no headline IK solver).

---

## 2. Cable / DER (R2)
**Two families.**
- **Discrete Elastic Rods (DER)** — Bergou et al. 2008 (SIGGRAPH; +Bergou 2010 *Discrete Viscous Threads*). Centerline polyline + a **material frame represented as angular offset θ from the Bishop (parallel-transport) frame**; centerline dynamic, twist quasistatic; energies for stretch + **bending (curvature κ)** + **twist (holonomy m)**. Physically the gold standard for anisotropic/curl-prone cable, but it needs implicit integration + a frame-update solve → heavier, not a natural fit for a position-projection scheduler.
- **XPBD Cosserat rods** — orientation-based: each segment carries a position **and a quaternion** (Cosserat = stretch+shear+bend+twist). Kugelstadt & Schömer 2016 (*Position and Orientation Based Cosserat Rods*); Soler et al. 2018 (*Cosserat Rods with Projective Dynamics*); Deul et al. 2018 (*Direct Position-Based Solver for Stiff Rods*). Expresses directly as XPBD constraints: a **stretch/shear** constraint (position pair + segment quaternion) and a **bend/twist** constraint (quaternion pair via the Darboux vector). This is iteration-count-stiffness-correct via XPBD compliance.

**Tradeoff / pick.** For a rope/cable **demo** under the existing XPBD scheduler + D1 + GPU, **XPBD-Cosserat wins**: it slots into Nuka's exact pattern (per-constraint position/orientation projection, graph-colored Jacobi-PBD, `compliance_alpha=1/stiffness`), reuses the soft-body stepper, and is forward-only-friendly. DER's quasistatic-twist implicit solve and frame bookkeeping fight the row-class model and offer accuracy the demo doesn't need.
**GPU + D1.** Color the segment graph (a path → trivial red/black even/odd coloring) so each color is a conflict-free parallel sweep; quaternion projection done own-index (no scatter). D1 identical to existing XPBD rows. Pitfall: quaternion normalization order must be fixed (do it own-index post-sweep, not in a reduction).
**Rows / spine.** **New row classes:** id11 `XpbdCosseratStretchShear` (2 particles + 1 orientation), id12 `XpbdCosseratBendTwist` (2 orientations). Couples to rigid/contact via the same particle pseudo-body path the cloth uses; consumes unified manifolds for cable-vs-world contact (CCD seam below for tight knots).
**Effort.** **M**.
**Comparators.** Newton: **VBD** now handles "linear deformables (cables)" (and AVBD rigid) — no classic DER. MuJoCo(-Warp): the **cable elasticity plugin** = an *inextensible 1D continuum* discretizing **twist+bend** (DER-flavored; stretch negligible) — MuJoCo-Warp ships flex ELASTICITY + VERTCOLLIDE. Genesis: cables via its PBD/FEM solvers. **Nuka opportunity:** a true orientation-based Cosserat rod beats Newton's cable story.

---

## 3. MPM (R4)
**Canonical algorithm.** **MLS-MPM** (Hu et al. 2018, SIGGRAPH, *A Moving Least Squares MPM with Displacement Discontinuity and Two-Way Rigid Body Coupling*) with **APIC** transfer (Jiang et al. 2015) — MLS-MPM makes APIC fall out naturally and gives a ~2× faster stress-divergence discretization. Loop per step: **P2G** (scatter particle mass+momentum → grid via the affine `C` matrix and quadratic B-spline weights), grid update (apply forces+gravity, BCs), **G2P** (gather grid velocity → particle v + affine `C`), particle advect + **constitutive/return-mapping**. **Plasticity for SAND:** Drucker-Prager return mapping on a Hencky-strain hyperelastic predictor (**Klár et al. 2016**, *Drucker-Prager Elastoplasticity for Sand Animation*); **von-Mises** for metal/clay; foundations: Stomakhin et al. 2013 (snow) + Jiang et al. 2016 (SIGGRAPH course *The MPM for Simulating Continuum Materials*). This is the **go2-on-sand** demo path.
**GPU strategy.** Grid as a hashed/uniform sparse block grid; particles sorted by cell (reuse `particle_uniform_grid` radix sort). P2G is the cost+hazard center; G2P is a clean gather.
**D1 — the P2G scatter hazard (the headline question).** Naïve P2G does `atomicAdd(float)` of each particle's contribution into its 27 (3³) neighbor grid nodes → **non-deterministic** (float-add non-associative under racing warps). **NO float atomics is the binding Nuka rule.** Bit-exact options, in preference order:
1. **Gather-form P2G (preferred — matches the PBF/K3 precedent).** Invert the loop: **one thread per grid node** gathers from the particles whose support covers it. Build a node→particle adjacency from the cell sort (count→exclusive-scan→private CSR slice, exactly like `particle_uniform_grid.cu`), then each node sums its contributions in **fixed sorted-ascending particle order with `__fadd_rn`**. Bit-exact, no atomics, no reassociation.
2. **Sort-and-segment-reduce.** Emit (node_id, contribution) pairs, `stable_sort_by_key` on node_id, segmented-reduce per node with a fixed-order serial scan. Deterministic; more memory traffic than (1).
3. (Rejected) deterministic-float-atomic / fixed-point accumulation — fragile and off-posture.
**MPM↔rigid two-way coupling.** Hu 2018 gives the recipe: rigid bodies act as moving velocity/Dirichlet BCs on the grid (P2G respects them; G2P reads back), and grid→rigid reaction is the **summed nodal force over the rigid's covered nodes** — accumulate that sum in the *same gather/fixed-order* manner and feed it as an external wrench (or, cleaner, a **coupling row** to the rigid/articulation reaction provider in the v0.8 spine). This is the articulated↔MPM R8 coupling for go2-on-sand.
**Rows / spine.** MPM is a **grid solver, not row-native** — it **co-steps** beside the row solver (like PBF p10), and exposes its rigid coupling through the **coupling-row framework** (new id: `MpmRigidCoupling`) + the v0.8 reaction providers. It consumes the unified contact spine only at the MPM↔rigid boundary (SDF of the rigid as a grid BC is the high-precision option).
**Effort.** **L** (sparse grid + transfers + return-mapping + coupling + oracle).
**Comparators.** Newton: **SolverImplicitMPM** (implicit/iMPM, granular, rough-terrain locomotion) — direct parity target. Genesis: full **differentiable MPM** (its flagship; MPM+Tool solver are the diff ones). MuJoCo-Warp: **no MPM** (flex/FEM-soft only). **Nuka would be at parity with Newton/Genesis here and ahead of MuJoCo-Warp.**

---

## 4. Volumetric FEM (R5)
**Canonical algorithm.** Tetrahedral FEM, implicit **backward-Euler**:
- **Corotational linear** (Müller & Gross 2004, *stiffness warping*): per-tet polar-decompose F=RS, rotate the constant linear stiffness `K₀` by R → `f = RK₀(Rᵀx − x₀)`. Cheap, stable to large rotations, the standard real-time choice.
- **Stable Neo-Hookean** (Smith, de Goes, Kim 2018): inversion-safe hyperelastic energy with closed-form eigen-system → analytic **SPD Hessian projection** (no clamp hacks); the modern robust nonlinear choice.
Backward-Euler assembles `(M/dt² + K)Δx = −∇E` per Newton iteration → SPD sparse solve.
**GPU strategy + Nuka reuse.** This is the **best fit for Nuka's self-written CG**: per-tet element force + Hessian (parallel over tets), assemble into a CSR system (or matrix-free `Ax`), then **CG** (Nuka has it; PCG with Jacobi/block-diag preconditioner). Element pass = embarrassingly parallel; coloring or gather for the residual assembly.
**D1.** CG is D1 iff its inner products use a fixed-order reduction — and **Nuka's existing CG already is** (`src/diffsim/sparse_solver_cg.cu`: fp64 internal arithmetic, deterministic FMA single-rounding, **fixed-order warp-shuffle butterfly sums** that are bit-exact, no atomics, deterministic residual-floor early exit). Reuse that posture; never a non-deterministic tree reduction; fixed iteration count (or the deterministic residual threshold + hard cap). Element-force assembly into the global vector = **gather (one thread per vertex over its incident tets in sorted order)**, never scatter-atomic. Polar decomposition / eigen-projection: fixed-iteration analytic forms (no data-dependent branching that reorders FP).
**Rows / spine.** Two viable embeddings: (a) **co-step like MPM/cloth** with FEM owning its own implicit solve, exposing contact via the spine (vertices as particle pseudo-bodies → unified manifold + multi-point contact rows); or (b) express the linearized element constraints as **compliant rows** (XPBD-FEM / VBD-style) — but (a) reuses the CG and is the cleaner forward-only path. **New:** `FemTet` element record + a co-step; contact through existing particle/manifold rows.
**Effort.** **M–L** (corotational = M; stable-Neo-Hookean + robust contact = L).
**Comparators.** Newton: **VBD** for volumetric deformables (rubber) + XPBD-soft; `warp.fem` available. MuJoCo-Warp: flex-based soft (elasticity plugin, FEM-like passive forces). Genesis: dedicated **FEM solver** (implicit). PhysX (via Isaac): GPU FEM. **Nuka gap-closer.**

---

## 5. Garment-grade cloth (R3)
**Upgrade over the basic XPBD p09 (id6 distance + id7 bend).** Three pillars:
- **Strain limiting** — cap edge stretch (e.g. ≤10%) per iteration (Provot 1995; Thomaszewski/Goldenthal biphasic) so cloth doesn't rubber-band; add as a clamped post-projection or a unilateral compliant row.
- **Accurate bending** — quadratic isometric/dihedral bending (Bridson 2003; Bergou 2006; Grinspun discrete shells) replacing the cheap distance-bend; or the cotangent bending energy. Needed for realistic wrinkles/folds.
- **Robust self-collision + CCD** (the hard part) — Bridson, Fedkiw, Anderson 2002 (*Robust treatment of collisions, contact, friction for cloth*): repulsion + **CCD** + a fail-safe **rigid-impact-zone** resolution guaranteeing intersection-free states; vertex-triangle + edge-edge tests.
**Alternatives to plain XPBD.** **Projective Dynamics** (Bouaziz et al. 2014) — local constraint projection + a global SPD solve (reuses Nuka's CG), the basis of Style3D-grade cloth. **VBD / Vertex Block Descent** (Chen, Liu, Yang, Yuksel 2024, SIGGRAPH) — vertex Gauss-Seidel on the implicit-Euler variational form, unconditionally stable, GPU-friendly, converges to backward-Euler; AVBD (Giles 2025) extends to rigid/contact. VBD is the current SOTA Nuka should track.
**GPU + D1.** Strain-limit + bend rows ride the existing colored XPBD scheduler (D1 by construction). PD/VBD: PD = colored local projections + a CG global solve (D1 same as FEM-CG); VBD = colored vertex Gauss-Seidel (D1 via fixed color order + fixed-iteration). **CCD is the D1 crux → §6.**
**Rows / spine.** **New rows:** id `XpbdStrainLimit` (unilateral, clamped), id `XpbdDihedralBend` (4-vertex accurate bending). Self-collision = cloth-vertex↔cloth-triangle pairs through the **unified manifold** + the **CCD seam**; consumes contact rows from the spine.
**Effort.** **M–L** (strain-limit + better bend = M; robust self-collision/CCD = L).
**Comparators.** Newton: **Style3D** (simplified PD garment solver) + **VBD** thin deformables (the cloth-manipulation example uses Featherstone+VBD). MuJoCo-Warp: flex cloth (elasticity + VERTCOLLIDE). Genesis: PBD/FEM cloth + uipc-style. **Nuka is "basic-only" today → this is the largest cloth gap vs Newton's Style3D/VBD.**

---

## 6. CCD seam (extensibility hook — reserve, don't fully build in v0.9)
**Options.**
- **Conservative Advancement (CA)** — iteratively advance time-of-impact using a lower bound on closest-distance / max-projected-motion; converges to TOI. Per-pair, branch-light.
- **Additive CCD (ACCD)** — Li, Kaufman, Jiang 2021 (*Codimensional IPC*, C-IPC): a robust CA variant that refines a TOI lower bound for deforming primitives; the modern, numerically-safe, GPU-amenable choice (used across IPC-family solvers). **Preferred seam.**
- **Swept-BVH / exact CCD** — Bridson 2002 vertex-triangle + edge-edge swept tests; Brochu 2012 / Tang Bernstein-Sign-Classification for exact roots. Higher cost, exactness when needed.
**GPU + D1.** Reserve a hook that takes (swept-AABB BVH from the existing LBVH, refit per substep) → candidate VT/EE pairs (deterministic stable-key order) → **ACCD per pair** (fixed max-iteration count, `__fadd_rn` arithmetic) → earliest TOI per vertex selected by a **fixed-order min-reduction** (deterministic tie-break by stable pair key). No append atomics; pairs into private slices. CA/ACCD are naturally per-pair → trivially parallel and D1 if iteration count is fixed.
**Rows / spine.** CCD is a **detection-stage hook** in the v0.8 narrowphase/broadphase (the spine already reserves a "CCD seam"); it feeds **time-of-impact-aware contacts** into the unified manifold → existing contact rows. No new row class — a new *detection mode*. Primary consumers: garment cloth (R3) self-collision, thin cable (R2) knots, fast rigid.
**Effort.** **R→M** (reserve the hook = R/S in v0.9; full ACCD self-collision = M, lands with R3).
**Comparators.** Newton/VBD: barrier/repulsion + CCD for thin deformables. MuJoCo-Warp: discrete flex VERTCOLLIDE (no full CCD). Genesis: uipc/IPC-style CCD. IPC/C-IPC = the ACCD reference.

---

## PER-SOLVER NUKA v0.9 RECOMMENDATION
- **R1 IK — DLS (Buss 2004) pre-step, Pink-QP multi-target later.** *Why:* cheapest H1-manipulation path, reuses Featherstone Jacobians, trivially D1 (small batched dense solve), zero new physics rows. **Rows:** none new — targets feed existing **drive rows (id2)**. **Effort: S–M.**
- **R2 Cable/DER — XPBD-Cosserat rods (Kugelstadt 2016 / Deul 2018), NOT DER.** *Why:* native to Nuka's colored XPBD scheduler + `compliance_alpha`, forward-only-friendly, D1 by the existing path; beats Newton's no-true-DER cable. **Rows:** id11 `XpbdCosseratStretchShear`, id12 `XpbdCosseratBendTwist`. **Effort: M.**
- **R4 MPM — MLS-MPM + APIC + Drucker-Prager (Hu 2018 / Klár 2016) for go2-on-sand.** *Why:* parity with Newton SolverImplicitMPM + Genesis; ahead of MuJoCo-Warp. **D1 crux solved via gather-form P2G** (node-owns-output, sorted-particle `__fadd_rn`, no float atomics — the PBF/K3 precedent). **Rows:** co-step + `MpmRigidCoupling` coupling row into the v0.8 reaction providers. **Effort: L.**
- **R5 Volumetric FEM — corotational linear (Müller 2004) → stable Neo-Hookean (Smith 2018), implicit BE + self-written CG.** *Why:* best fit for Nuka's existing CG; D1 if CG inner-products use fixed-order `__fadd_rn`. **Rows:** `FemTet` element + co-step; contact via existing particle-manifold rows. **Effort: M–L.**
- **R3 Garment cloth — strain-limited XPBD + accurate dihedral bending + robust self-collision/CCD; track VBD (Chen 2024) / PD (Bouaziz 2014) as the SOTA upgrade.** *Why:* closes the biggest cloth gap vs Newton Style3D/VBD; XPBD rows reuse the scheduler. **Rows:** `XpbdStrainLimit` (unilateral), `XpbdDihedralBend` (4-vertex); self-collision via unified manifold + CCD. **Effort: M–L.**
- **CCD seam — reserve an ACCD (Li 2021) hook** in the v0.8 narrowphase. *Why:* GPU-friendly conservative advancement, D1 with fixed iteration count + fixed-order TOI min-reduction; the right primitive for R3/R2. **Rows:** none — a detection mode feeding TOI-aware contacts to existing contact rows. **Effort: R (reserve) → M (full, with R3).**

## Sources
- Buss 2004, *Introduction to IK with Jacobian Transpose, Pseudoinverse and DLS* (UCSD); Buss & Kim 2005, *Selectively Damped LS*; Wampler 1986 (DLS); Caron, *Pink* (github.com/stephane-caron/pink); cuRobo (arXiv:2310.17274, NVlabs).
- Bergou et al. 2008, *Discrete Elastic Rods* (SIGGRAPH, cs.columbia.edu/cg/rods); Bergou et al. 2010, *Discrete Viscous Threads*; Kugelstadt & Schömer 2016, *Position and Orientation Based Cosserat Rods*; Soler et al. 2018, *Cosserat Rods with Projective Dynamics* (SCA); Deul et al. 2018, *Direct Position-Based Solver for Stiff Rods*.
- Hu et al. 2018, *MLS-MPM …Two-Way Rigid Body Coupling* (SIGGRAPH, ACM 10.1145/3197517.3201293); Jiang et al. 2015 *APIC*; Stomakhin et al. 2013 *MPM snow*; Klár et al. 2016, *Drucker-Prager Elastoplasticity for Sand* (ACM 10.1145/2897824.2925906); Jiang et al. 2016 SIGGRAPH course, *The MPM for Simulating Continuum Materials*.
- Müller & Gross 2004, *stiffness warping / corotational* (matthias-research warp.pdf); Smith, de Goes, Kim 2018, *Stable Neo-Hookean Flesh Simulation* (ACM 10.1145/3180491); Bouaziz et al. 2014, *Projective Dynamics*; Chen, Liu, Yang, Yuksel 2024, *Vertex Block Descent* (SIGGRAPH, arXiv:2403.06321); Giles 2025, *Augmented VBD*.
- Macklin & Müller 2016, *XPBD* (ACM 10.1145/2994258.2994272); Müller et al. 2007, *Position Based Dynamics*; Provot 1995 (strain limiting); Bridson, Fedkiw, Anderson 2002, *Robust treatment of collisions… for cloth* (SIGGRAPH); Bridson 2003 / Bergou 2006 / Grinspun (discrete shells, bending).
- Li, Kaufman, Jiang 2021, *Codimensional IPC* (ACCD; arXiv:2012.04457, ipc-sim.github.io/C-IPC); Brochu 2012 / Tang, *exact CCD via Bernstein Sign Classification*.
- Engines: `newton-physics/newton` solver list (SolverImplicitMPM, SolverVBD, Style3D, XPBD, MuJoCo-Warp, Featherstone — newton-physics.github.io/newton/api/newton_solvers.html, Discussion #639); `google-deepmind/mujoco_warp` README + MuJoCo elasticity/cable plugin docs; `Genesis-Embodied-AI/genesis-world` (genesis-world.readthedocs.io — unified Rigid/MPM/SPH/FEM/PBD + coupler + SAP, differentiable MPM).
- Nuka in-repo: `src/constraint/row.hpp`, `src/solver/gpu/row_*.{cu,cuh}`, `src/collision/particle_uniform_grid.cu`, `src/runtime/fluid/pbf_{kernels.cuh,world.cu}`, `src/runtime/coupling/particle_particle_contact.cu`, `docs/plans/2026-06-03-v08-unified-collision-contact.md`, `docs/plans/2026-06-03-post-v07-roadmap.md`.
