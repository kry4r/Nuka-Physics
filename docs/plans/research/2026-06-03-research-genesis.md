# Genesis (Genesis-Embodied-AI) — Unified Collision/Contact/Coupling Research

Date: 2026-06-03. Read-only research for Nuka's UNIFIED collision/contact/coupling subsystem.
Target: Genesis "universal" GPU physics engine (Rigid + MPM + SPH + FEM + PBD + StableFluid + Tool, differentiable).
Method: Genesis docs (readthedocs v0.3.3 / latest), DeepWiki, and primary source (`legacy_coupler.py`, `mpm_solver.py`, `simulator.py` on `main`).

Nuka anchor (don't redesign — map to this): GPU-first; per-system broadphase (LBVH rigid + uniform grid particles) + cross-system query dispatcher → unified multi-point **ContactManifold** → constraint **ROWS (row-class)** → solver; hybrid narrowphase (analytical + convex + SDF tier); compliant contact (MuJoCo solref/solimp); PBD bridged via **co-step coupling rows**; full **D1 determinism**; diff only on SDF tier.

---

## 1. Multi-solver / multi-material architecture

- **Hierarchy: Scene → Simulator → Solver → Entity.** Scene is the orchestrator (entities, solvers, vis, sensors). Material-based routing: an Entity's `material` arg decides which solver owns it. [DeepWiki overview; `genesis/engine/`]
- **Simulator owns 8 solver instances** as attributes and a unified list `self._solvers`, plus `self._active_solvers` populated at `build()`: `tool_solver, rigid_solver, kinematic_solver, mpm_solver, sph_solver, pbd_solver, fem_solver, sf_solver`. [`genesis/engine/simulator.py`]
  - RigidSolver (articulated robots + free/fixed bodies, constraints), FEMSolver (tet FEM, implicit/explicit), MPMSolver (MLS-MPM deformables/granular/snow), SPHSolver (WCSPH/DFSPH liquids), PBDSolver (cloth/soft/particles), SFSolver (grid stable-fluid smoke/gas), ToolSolver (kinematic background geometry). [DeepWiki "Physics Solver Integration"]
- **Per-material state is a per-solver global pool**, batched over envs `B`. FEM example: `elements_v.{pos,vel}` shape `(substeps_local+1, n_vertices, B)`; static `elements_i.{el2v,mu,lam,B,V}`; entities hold index ranges `v_start/v_end, el_start/el_end`. MPM: `self.grid[f, cell_ijk, i_b]` + `self.particles[f, i_p, i_b]`, indexed by frame `f`. [`fem_solver.py`, `mpm_solver.py`]
- **Substepping is global, shared by ALL solvers.** `_dt = options.dt`; `_substep_dt = dt / substeps`; `step()` runs `for _ in range(substeps): substep(f)`. There is NO per-solver independent substep clock — every solver advances on the same `substep_dt`. [`simulator.py`]
- **Per-substep order (the spine):** `coupler.preprocess(f)` → `substep_pre_coupling(f)` (each solver advances itself) → `coupler.couple(f)` (momentum exchange) → `substep_post_coupling(f)`. [`simulator.py`; v0.3.3 coupling doc]

→ Nuka: ADAPT. Genesis's "Simulator owns N solvers + one Coupler, shared global substep, pre→couple→post" is exactly Nuka's co-step spine. ADOPT the per-solver state pool / index-range pattern. ADOPT the rigid `substep_pre_coupling` boundary as where Nuka emits coupling rows. AVOID forcing all solvers onto one global substep_dt — Nuka's row-class lets stiff PBD/MPM sub-cycle while rigid stays on its dt; keep that flexibility.

## 2. Cross-material COUPLING (most load-bearing for Nuka)

- **One Coupler instance holds pointers to every solver; pairs are wired statically at `Coupler.build()`.** "If a combination is not in the table it is currently unsupported." Three coupler implementations selected by option: **LegacyCoupler** (force/impulse, Rigid↔MPM/SPH/PBD/FEM), **SAPCoupler** (Semi-Analytic Primal — Drake-style hydroelastic, Rigid↔FEM & FEM↔FEM), **IPCCoupler** (Incremental Potential Contact via `libuipc` — Newton + CCD, FEM/cloth/articulated). [latest coupling doc; `genesis/options/solvers.py`; `genesis/engine/couplers/`]
- **Coupling is NOT a single unified grid and NOT one unified constraint solver. It is per-pair impulse/force exchange written as Taichi `@kernel`s that touch the memory of BOTH solvers with no data copies** (all solvers share Quadrants/Taichi fields). [DeepWiki; coupling doc; `legacy_coupler.py`]
- **LegacyCoupler.couple(f) dispatch** (primary source `legacy_coupler.py:944`): MPM active → `mpm_grid_op(...)` (MPM↔everything on the grid); `_rigid_sph` → `sph_rigid(...)`; `_rigid_pbd` → `kernel_pbd_rigid_collide(...)` + 1-way `kernel_pbd_rigid_solve_animate_particles_by_link`; FEM active → `fem_surface_force(...)` + `fem_rigid_link_constraints()`.
- **THE pattern Nuka cares about — two-way rigid↔particle via SDF velocity projection (`_func_collide_in_rigid_geom`, lines 284–347):**
  1. Query rigid SDF: `signed_dist = sdf.sdf_func_world(...)`; soft influence `influence = min(exp(-signed_dist / coup_softness), 1)`; act only if `influence > 0.1`. (`_func_collide_with_rigid_geom`, 167–217)
  2. Relative vel to rigid surface point `rvel = vel - vel_rigid` (rigid surface velocity from `_func_vel_at_point`). If `rvel·n < 0` (approaching): apply **Coulomb friction** on tangential + **restitution** on normal (`coup_friction`, `coup_restitution`), blend by `influence`: `vel = vel_rigid + rvel_new*influence + rvel*(1-influence)`.
  3. **Reaction back to rigid (the two-way step):** `delta_mv = mass*(vel - vel_old); force = -delta_mv / substep_dt; rigid_solver._func_apply_coupling_force(pos, force, link_idx, i_b, ...)`.
- **Coupling table (latest doc):** MPM↔Rigid = impulse on grid nodes (supports CPIC); MPM↔SPH = average SPH particle vels within an MPM cell; MPM↔PBD = like SPH but skip pinned PBD particles; FEM↔Rigid = collision on surface vertices only; FEM↔MPM = exchange momentum via MPM P2G/G2P weights; FEM↔SPH = experimental, normal projection only; SPH↔Rigid = robust side-flip normal handling; PBD↔Rigid = positional correction then velocity projection; Tool↔MPM = delegated to each Tool entity's `collide()`.
- **Params governing all of it:** `coup_softness` ε (contact-zone thickness via `exp(-d/ε)`), `coup_restitution` e ∈[0,1], `coup_friction` μ (Coulomb). These are per-rigid-geom (`geoms_info.coup_*`). [coupling doc; `legacy_coupler.py`]
- **Direction matters:** MPM/SPH/PBD↔Rigid is "rigid drives particles, particles push back as a force" — symmetric momentum, but resolved on the *particle/grid* side then reflected to rigid. FEM↔Rigid is surface-vertex only. PBD↔Rigid is positional (PBD is positional, so correction then velocity projection).

→ Nuka: ADOPT the core mechanism wholesale — it IS Nuka's co-step coupling row. `_func_collide_in_rigid_geom` = a coupling row: read rigid SDF + surface velocity, project the PBD/particle constraint, then push equal-and-opposite impulse to the rigid link's velocity DOFs. Genesis applies it as an explicit force `−Δmv/dt`; Nuka should make it a **constraint ROW** (solver-resolved, not explicit force) so it stays implicit/stable. ADAPT: replace Genesis's `influence = exp(-d/ε)` soft gate with Nuka's **solref/solimp compliant** schedule (same goal: graded contact zone, but inside the solver). AVOID Genesis's static `Coupler.build()` whitelist of hard-coded pairs and "not-in-table = unsupported" — Nuka's cross-system query dispatcher → row-class generalizes this so any new solver gets coupling for free.

## 3. Collision across materials

- **NOT a single unified collision representation. Two regimes:** (a) rigid↔rigid runs the rigid narrowphase; (b) rigid↔everything-else (MPM/SPH/PBD/FEM particles & vertices) is done by **querying the rigid geometry's SDF** at each particle/vertex world position — never a symmetric pair test. [`legacy_coupler.py`; coupling doc "Coupler queries the signed distance function `sdf(p)` of the rigid geometry"]
- **Rigid is the SDF source for ALL cross-material contact.** `rigid_solver.collider._sdf._sdf_info` is passed into every coupling kernel. SDFs are **pre-baked offline** for triangle meshes / decomposed convex clusters. [coupling doc; collision doc]
- **CPIC (Compatible PIC) for thin rigid objects vs MPM** (`mpm_solver.py` ~167–215): during P2G compare particle vs grid-node sides of a thin object via SDF normals; set `cpic_flag` if separated; during G2P replace grid velocity with rigid collision-response velocity. **CPIC not supported in gradient mode.**
- **MPM↔rigid attach constraint:** particles can be pinned to rigid links (`particle_constraints.{is_constrained,target_pos,stiffness,link_idx,link_local_pos}`) — a stiffness-based soft attachment, separate from contact. [`mpm_solver.py:167–191, 533–536`]
- Particle↔particle across continua (MPM↔SPH, MPM↔PBD) is resolved on the **shared MPM grid** (velocity averaging in a cell), not via SDF.

→ Nuka: ADAPT — Genesis confirms "rigid carries the SDF, deformables query it" is a sound, fast unified contact primitive, and matches Nuka's **SDF narrowphase tier**. ALREADY-HAVE: Nuka's unified ContactManifold + hybrid narrowphase (analytical/convex/SDF) is strictly more general than Genesis's "SDF-only for cross-material." ADOPT pre-baked mesh SDFs for the diff/SDF tier. NOTE: CPIC (thin-shell handling) is worth stealing for cloth/thin-rigid vs particles, but Genesis loses gradients there — Nuka should design thin-shell contact to stay differentiable.

## 4. Differentiability

- **Per-solver, NOT global. Built on Quadrants/Taichi reverse-mode autodiff** (DiffTaichi lineage: two-scale AD, megakernels, source-to-source tape), with **custom-gradient checkpointing** to recompute grid state in backward (one grid copy vs O(n)). [DiffTaichi paper arXiv:1910.00935; Taichi diff docs]
- **Status (DeepWiki table):** MPMSolver = full diff; ToolSolver = diff; RigidSolver = partial (forward dynamics only); FEMSolver = under development; SPH/PBD/SF = planned.
- **Coupler has a backward path** but only for the differentiable solvers: `LegacyCoupler.couple_grad(f)` calls `fem_surface_force.grad(...)` and `mpm_grid_op.grad(...)` — i.e. rigid↔MPM and rigid↔FEM coupling gradients exist; SPH/PBD coupling gradients do not. [`legacy_coupler.py:998`]
- **Known limits:** CPIC has no gradient (`mpm_solver.py:212–215`); rigid-body collision gradients are notoriously misleading under naive autodiff ("naively differentiating leads to completely misleading gradients due to rigid body collisions" — DiffTaichi). MPM backward forces `qd.static` p2g path "so backward stays exactly as upstream" (`mpm_solver.py:232`).

→ Nuka: ADAPT/already-aligned. Genesis confirms Nuka's "diff only on the SDF tier" is the pragmatic line — even Genesis can't cleanly differentiate rigid contact, and it ring-fences diff to MPM/FEM. ADOPT custom-gradient grid checkpointing for the MPM/FEM breadth solvers (v0.9). AVOID claiming differentiable rigid-contact; AVOID a single monolithic global tape — Genesis's per-solver `.grad()` kernels (incl. `couple_grad`) are the realistic granularity.

## 5. Determinism (UNDOCUMENTED — inferred; FLAG as unverified)

- Genesis docs/DeepWiki **do not address determinism or seeding**; targeted issue/doc search returned nothing explicit. (Honest null.)
- **Primary-source evidence of NON-determinism in particle/coupling solvers:** MPM P2G is GPU **atomic scatter** — `qd.atomic_add(self.grid[f, cell_ijk, i_b].mass, mass_contrib)` and `grid[...].vel_in += weight*(...)` accumulate many particles into one grid node with thread-order-dependent float adds (`mpm_solver.py:473,477`). Coupling reaction forces likewise accumulate onto rigid links across many particles. Float add is non-associative, so atomic-scatter results vary with thread arrival order → **not bitwise-reproducible** on GPU for MPM/coupling. [`mpm_solver.py`; HPC FP-nonassociativity arXiv:2408.05148; PyTorch scatter_reduce nondeterminism #33394]
- Rigid-only solver uses sorted broadphase + (likely) ordered constraint assembly and may be more reproducible, but this is not documented.

→ Nuka: AVOID Genesis-style atomic-scatter accumulation if D1 is a hard pillar. ADOPT fixed-order / segmented (sort-by-cell then deterministic reduction) accumulation for particle→grid and for coupling-force aggregation onto rigid DOFs. This is a genuine DIFFERENTIATOR for Nuka — Genesis trades determinism for raw scatter speed; Nuka can keep determinism with a sort+segment-reduce at modest cost.

## 6. Broadphase / spatial structures per material

- **Rigid:** tight per-geom world AABB each frame (`geoms_state.aabb_min/max`) → **Sweep-and-Prune**, N·logN insertion-sort variant, projected on a single (X) axis, warm-started from last frame's near-sorted order, active-interval set; filtered by `contype`/`conaffinity` bitmasks, same/adjacent links, both-fixed pairs. [collision doc]
- **MPM:** background **uniform Eulerian grid** (`self.grid[f, cell_ijk, i_b]`, P2G/G2P), with sparse dirty-cell bookkeeping (`grid_dirty_flag`, `grid_dirty_count`) to touch only active cells. [`mpm_solver.py`]
- **SPH:** spatial-hash neighbor search (WCSPH/DFSPH); **PBD:** particle grid/neighbor list; cross-continuum coupling reuses the MPM grid cell as the meeting structure. [DeepWiki; coupling doc]

→ Nuka: ALREADY-HAVE / ADOPT. Genesis = per-material structures (rigid SAP, particle grid/hash), no global broadphase — exactly Nuka's "per-system optimal broadphase + cross-system query dispatcher." ADOPT sparse dirty-cell tracking for Nuka's particle grids. NOTE: Genesis uses SAP, Nuka chose LBVH for rigid — keep LBVH (better for many-moving-body GPU + parallel build); SAP's X-axis projection degrades on clustered scenes.

## 7. Performance

- Headline: **~43M FPS** rigid locomotion on one **RTX 4090**, ~430,000× real-time; 4,096 parallel envs ≈ 100k FPS/env; locomotion policy trained in ~26 s; claimed **10–80× faster than Isaac Gym/Sim/Lab and MuJoCo MJX**. (Vendor/blog figures — rigid-only, batched RL; not multi-material coupling.) [Genesis README/blog; DeepWiki ">43M FPS RTX 4090"; DataCamp; Medium/DeepNewz]
- Design drivers: massive env batching (`B` dim on every field); JIT megakernels (Quadrants/Taichi → CUDA/ROCm/Metal); zero-copy PyTorch interop; **no-copy coupling kernels** over shared fields; warm-started SAP; sparse MPM grid; hibernation (skip sleeping bodies/contacts). [DeepWiki; collision doc]

→ Nuka: ADAPT. The 43M-FPS number is rigid-only batched RL — set expectations: multi-material coupling is far slower, headline ≠ coupled-scene throughput. ADOPT batch-everything field layout, zero-copy CUDA↔host/PyTorch, no-copy coupling kernels over shared GPU buffers (matches Nuka's CUDA↔Vulkan interop direction), hibernation, sparse grids.

## 8. Known limitations / pain points (what to AVOID)

- **Static hard-coded coupling whitelist** — pairs fixed at `Coupler.build()`, "not in the table = unsupported." Adding a solver means hand-writing every pair kernel. Brittle, O(N²) authoring. [coupling doc]
- **Asymmetric / incomplete pairs:** FEM↔SPH "experimental, normal projection only"; FEM↔Rigid surface-vertex only (misses edge/volume); PBD↔Rigid is positional-correction (energy not conserved cleanly). [coupling doc]
- **Explicit force coupling** (`force = −Δmv/substep_dt`) is conditionally stable — stiff/high-mass-ratio rigid↔soft can blow up; mitigated only by small `substep_dt`. [`legacy_coupler.py`]
- **Three incompatible couplers** (Legacy / SAP / IPC) — you pick ONE per scene; their pair coverage differs (e.g. IPC for high-quality FEM/cloth but heavy Newton+CCD; SAP for Rigid-FEM hydroelastic; Legacy for particles). No single coupler does everything well. [DeepWiki; `options/solvers.py`]
- **Gradient gaps:** CPIC no-grad; rigid contact gradients misleading; SPH/PBD coupling no backward. [`mpm_solver.py`; `legacy_coupler.py`; DiffTaichi]
- **Determinism not guaranteed** for particle/coupling solvers (atomic scatter; §5). [`mpm_solver.py`]
- **One global substep_dt** forces the whole scene to the stiffest solver's timestep — wasteful when only one material is stiff. [`simulator.py`]

→ Nuka: AVOID all six. Nuka's row-class + cross-system dispatcher kills the O(N²) whitelist; implicit constraint rows (vs explicit `−Δmv/dt`) fix the stability; one unified contact/constraint solver (vs 3 couplers) avoids the pick-one trap; per-row solref/solimp + per-system substep budget avoids the global-dt waste; deterministic reductions fix §5.

---

## TOP RECOMMENDATIONS FOR NUKA — coupling spine (v0.8) + breadth solvers (v0.9)

1. **The coupling row IS `_func_collide_in_rigid_geom`.** Adopt its shape directly: SDF-query rigid surface + surface-point velocity → project the other system's velocity (normal restitution + Coulomb friction) → emit equal-and-opposite impulse on the rigid link DOFs. Make it a solver-resolved **constraint ROW**, not an explicit `−Δmv/dt` force, for implicit stability. (v0.8 spine)
2. **Generalize past Genesis's static whitelist.** Nuka's cross-system query dispatcher + row-class must let ANY (system A, system B) pair produce coupling rows without hand-written O(N²) pair kernels — the single biggest architectural win over Genesis.
3. **Rigid carries the SDF; deformables query it.** Adopt as Nuka's cross-material narrowphase = the SDF tier. Pre-bake mesh/convex-cluster SDFs offline. Keep analytical+convex tiers for rigid↔rigid (Nuka already more general than Genesis here).
4. **Replace `influence = exp(-d/ε)` soft gate with solref/solimp compliant contact** inside the solver — same graded-contact-zone intent, but stable and implicit.
5. **Keep co-step ordering** `preprocess → pre_coupling (each solver advances) → couple (emit+solve rows) → post_coupling`. This maps 1:1 to Nuka's co-step bridge; rigid `pre_coupling` is where Nuka assembles the coupling rows.
6. **Do NOT force one global substep_dt.** Let row-class carry per-system timestep budgets so stiff PBD/MPM can sub-cycle while rigid stays on its dt — fixes Genesis's stiffest-solver-taxes-everyone flaw.
7. **One unified contact/constraint solver, not 3 couplers.** Genesis splits Legacy/SAP/IPC by scene; Nuka's single row-solver spine should subsume impulse (Legacy), hydroelastic (SAP), and high-quality CCD (IPC-like, optional) as row-classes/tiers.
8. **Determinism is a real differentiator.** Replace Genesis's atomic-scatter P2G + coupling-force accumulation with sort-by-cell + segmented deterministic reduction. Modest cost, buys D1 — Genesis cannot claim this.
9. **Diff only on the SDF/continuum tier.** Mirror Genesis: full diff for MPM/FEM breadth solvers (v0.9) with custom-gradient grid checkpointing; partial/none for rigid contact. Don't promise differentiable rigid contact.
10. **Steal CPIC for thin shells (cloth/thin-rigid vs particles) but keep it differentiable** — Genesis's CPIC drops gradients; design Nuka's thin-shell contact row to retain them.

## Sources
- Coupling (latest): https://genesis-world.readthedocs.io/en/latest/user_guide/advanced_topics/solvers_and_coupling.html
- Coupling (v0.3.3): https://genesis-world.readthedocs.io/en/v0.3.3/user_guide/advanced_topics/solvers_and_coupling.html
- Rigid collision detection (v0.3.3): https://genesis-world.readthedocs.io/en/v0.3.3/user_guide/advanced_topics/collision_contacts_forces.html
- Simulator source: https://github.com/Genesis-Embodied-AI/Genesis/blob/main/genesis/engine/simulator.py
- LegacyCoupler source: https://github.com/Genesis-Embodied-AI/Genesis/blob/main/genesis/engine/couplers/legacy_coupler.py
- MPMSolver source (P2G atomics, CPIC): https://github.com/Genesis-Embodied-AI/Genesis/blob/main/genesis/engine/solvers/mpm_solver.py
- Couplers dir: https://github.com/Genesis-Embodied-AI/Genesis/tree/main/genesis/engine/couplers
- DeepWiki architecture: https://deepwiki.com/Genesis-Embodied-AI/Genesis
- DeepWiki rigid body: https://deepwiki.com/Genesis-Embodied-AI/Genesis/3.1-rigid-body-dynamics
- API/engine ref: https://genesis-world.readthedocs.io/en/latest/api_reference/engine/index.html
- DiffTaichi (autodiff lineage): https://ar5iv.labs.arxiv.org/html/1910.00935 ; Taichi diff docs: https://docs.taichi-lang.org/docs/differentiable_programming
- FP non-associativity / nondeterminism: https://arxiv.org/html/2408.05148v3 ; PyTorch scatter_reduce: https://github.com/pytorch/pytorch/issues/33394
- Performance figures: https://github.com/Genesis-Embodied-AI/Genesis (README) ; https://www.datacamp.com/blog/genesis-physics-engine ; https://deepnewz.com/robotics/new-physics-simulation-achieves-430000x-speed-43-million-fps-trains-robots-26-8a59df0e
