# Nuka Physics – Long-Term Master Plan

> **Status:** Project constitution. Single source of truth for all top-level design and process decisions. Supersedes any conflicting earlier plan.
>
> **AI agent boundary:** This file is in the **AI-protected list** (see §5.4). AI agents may read it freely. AI agents must not modify it. Only the human owner edits this file.
>
> **Owner edit log:**
> - 2026-05-28 – initial draft after design-grilling convergence (13 rounds, 16+ dimensions resolved).
> - 2026-05-28 – added §5.6 GPU-only simulation hard constraint; project-wide ban on CPU simulation in production paths. Added decision row #35 to §2 summary. Propagated constraint reference into all 16 v0.1 / v0.3 / v0.5 phase specs.
> - 2026-05-31 – AMENDMENT: dropped cuDSS entirely. v0.5 ships a self-written deterministic sparse linear solver (CG + Jacobi/Block-Jacobi, fixed-order reductions, D1 bit-exact) for the IFT path from the start; MINRES/ILU/GMRES/AMG remain v0.7+ extensions of this core. Reverses the Round 13 "cuDSS de-risks v0.5" reserve (see §3 Round 3/13 + §7). Propagated into v0.5 / v0.7 phase specs.

---

## Table of Contents

1. Top-Line Targets
2. Decision Summary (quick-scan table)
3. **Design Decision Log** (all 13 grilling rounds, full options + rationale + decision)
4. Universal Row IR Specification
5. Workflow Hard Constraints
6. Validation Architecture (V1–V5)
7. Phase Exit Criteria (the constitution)
8. Risk Register
9. Engineering Task Dependency Graph
10. First-Week Concrete Plan
11. Per-Subsystem Documentation Discipline
12. Out of Scope
13. Amendment Process

---

## 1. Top-Line Targets

Build a CUDA-first, full-GPU physics engine that simultaneously achieves:

1. **Large-scale parallel simulation** for RL training (4096+ envs / single GPU).
2. **Multi-body rigid + soft + fluid coupling** for embodied robotics tasks.
3. **Effects on par with NVIDIA Isaac Lab** across physics quality, sensor pipeline, and toolchain ergonomics.
4. **Seamless embedding into arbitrary C++ rendering engines** (UE5 / Unity native / Godot 4 / bespoke Vulkan/D3D12).

Phased anchors (strict order, no skipping):

- **S1** – 4096 Unitree Go2 PPO locomotion on a single RTX 4090, sub-ms per env-step.
- **S2** – Unitree H1 dexterous manipulation: grasp cup, pour water (fluid), wring towel (cloth + coupling).
- **S3** – 64-robot warehouse with fluid spill + deformable packaging on a single H100.
- **S5** – Vision-Language-Action data factory: 4096 envs with RGB + depth + semantic + tactile.

**S4** (differentiable simulation) is a horizontal capability woven through S1–S5, not a sequential phase.

---

## 2. Decision Summary

| # | Dimension | Final Decision |
|---|---|---|
| 1 | Project scope | Path D – all four pillars (parallel + coupling + diff-sim + custom RT) |
| 2 | Phasing | S1 → S2 → S3 strict; S4 horizontal; S5 visual / VLA final |
| 3 | Differentiability | Required from day 1; full analytical adjoint |
| 4 | Rendering target | Self-written CUDA ray tracer; no OptiX |
| 5 | Solver math model | Path B variant: Featherstone ABA + maximal PGS/TGS + XPBD soft + PBF fluid, unified through CSR row scheduler |
| 6 | Articulation | Dual-track: Featherstone for joint dynamics, rows for constraints |
| 7 | Row format | Option 3 – unified CSR-like universal row |
| 8 | Migration discipline | 2-3 month infrastructure refactor accepted; new format with diff-test bridge |
| 9 | Adjoint kernel pipeline | Codegen (not function pointers, not runtime dispatch) |
| 10 | Contact discontinuity (diff-sim) | β – stop-gradient at events; α retained as per-env switch |
| 11 | Diff-sim tape | Checkpointing (K≈50–100) + IFT hybrid; full tape only in debug |
| 12 | Sparse linear solver | Self-written deterministic solver from v0.5 (CG + Jacobi/Block-Jacobi, fixed-order reductions, D1 bit-exact) for the IFT path. v0.7+ extends the same self-written core with MINRES/ILU(0)/GMRES/AMG + factorization. No cuDSS / no closed-source SDK, ever |
| 13 | Determinism contract | D1 strong (bit-exact); island/coloring v1 must-have; D2 weak as RL training fallback |
| 14 | Multi-GPU | M1 single-GPU + M2 multi-GPU data-parallel; M3 explicitly out of scope |
| 15 | Single-env envelope | 80 GB H100; particles<1M, tets<200K, rigids<50K |
| 16 | Cross-system coupling | K2 (SDF Newton) + K3 (cross-system rows) hybrid |
| 17 | Thin-shell SDF | Sparse narrow-band SDF + V-HACD convex decomposition + Newton-on-summed-SDF |
| 18 | Broadphase | SAP → LBVH + particle uniform grid + cross-system query |
| 19 | C++ ABI | Pure C ABI core + thin C++20 wrapper; caller-owned cudaStream_t; no exceptions or STL across boundary |
| 20 | Renderer interop | CUDA↔Vulkan ext-mem (S1) + CUDA↔D3D12 ext-mem (S2) + CPU staging fallback + CUDA-RT framebuffer for sensors |
| 21 | Platform scope | NVIDIA only; macOS/AMD/Intel intentionally deferred via PHI layer |
| 22 | Python bindings | nanobind + DLPack + caller-supplied PyTorch CUDA stream |
| 23 | Autograd integration | PyTorch autograd.Function: v0.1 skeleton, v1.0 complete; JAX custom_vjp: v0.5 |
| 24 | Isaac Lab compat | RL training path API only (no UI/editor/Omniverse); ManagerBasedRLEnv drop-in |
| 25 | Reward/observation | v1: Python (Isaac Lab parity); v2: codegen DSL → fused CUDA kernel |
| 26 | Sensor matrix | S1: IMU + joint + tactile + F/T + lidar + basic depth; S2: + RGB + depth; v1.0: + semantic + normal/uv; v2.0: + event camera + diff render |
| 27 | Sim-to-real noise | v1: N1 + N2; v2: N3 key items (lens distortion, rolling shutter, lidar beam divergence) |
| 28 | License | Apache 2.0 + CLA; repo private through v0.3, public at v0.5 |
| 29 | Validation strategy | 5-layer V1-V5: Oracle + invariants + adjoint FD + demo suite + AI guardrails |
| 30 | Oracles | MJX + Pinocchio (Featherstone); Bullet + MuJoCo (rigid); Vellum (cloth); Flex paper (fluid); MuJoCo 3.0 SDF plugin; finite-diff (adjoint) |
| 31 | Asset pipeline | USD authoritative (.usda + .usdc/.usdz via OpenUSD SDK at S1 end); URDF + MJCF retained; glTF at S5; FBX skipped; .nuka cooked binary day 1 |
| 32 | MaterialX schema | USD MaterialX custom-attribute schema as official authoring contract |
| 33 | Time discipline | No calendar deadlines; rhythm-based (weekly/monthly/quarterly); stagnation triggers (4w review, 8w kill switch) |
| 34 | Phase discipline | No skipping; exit criteria satisfied at close, not "later" |
| 35 | **Execution target (project-wide)** | **GPU-only simulation. No CPU physics simulation in any production code path.** CPU is permitted only for one-time import/cook, host orchestration, kernel launching, C ABI marshalling, and external-engine validation (V1 oracles call out to MJX/Pinocchio/Bullet/Vellum/Flex). The PHI `CpuReference` backend remains as a validation-only tool — never invoked from production handles. See §5.6 for the full enforcement contract. |

---

## 3. Design Decision Log

This section documents the full grilling conversation: each round's question, all options considered, the recommendation with its reasoning, and the final decision. This is the long-form record of *why* the project is shaped the way it is. Read this before re-litigating any decision.

### Round 1 – Scope: What does "对标 Isaac Lab" mean concretely?

**Context:** Isaac Lab is a thin orchestration layer built on PhysX 5 GPU (rigid + joints + tendons + soft) + NVIDIA Warp (particles / SDF / differentiable custom kernels) + Omniverse RTX (sensor rendering) + Python RL gym interface. "Match Isaac Lab" can mean very different things depending on which layer.

**Options:**

- **A. Match the user experience + ecosystem.** URDF/USD import → RL task config → one-click parallel → tensorboard curves. Implies rebuilding RL gym wrapping, Python bindings, reward/observation buffers; physics itself can be modest.
- **B. Match physics simulation breadth.** Rigid + soft + fluid coupling + differentiable + robot tendons/muscles + cloth + granular + multi-phase fluid. Implies head-on competition with PhysX 5 + Warp; algorithmic depth-first.
- **C. Match large-scale parallel throughput.** Single GPU thousands of envs at sub-ms per step, RL data factory. All data GPU-resident, zero host sync, batched-kernel design; single-env complexity can be low.
- **D. All of the above.** Implies 3–5 years, 10+ engineers in industry terms.

**Recommendation:** C first (closest to current code), then B (true differentiation), finally A (engineering more than research). Concrete first milestone should target Isaac *Gym* throughput (pure GPU parallel rigid + simple contact), not Isaac *Lab* breadth.

**Decision:** **D – all of the above, as a long-term plan.** User explicitly accepted multi-year horizon.

---

### Round 2 – Flagship Demo: What is the 3-year anchor scene?

**Context:** With "D, all of it" accepted, scope is unbounded unless anchored to a concrete demo. Isaac Lab has 100+ tasks; we cannot replicate all. One specific scene anchors every architecture decision (does this design serve this demo?).

**Options:**

- **S1 · RL Data Factory:** 4096 Go2 quadrupeds parallel-learning gait on RTX 4090, sub-0.5 ms/step, 48 h to stable policy. Dominates throughput; no soft/fluid.
- **S2 · Humanoid Dexterous Manipulation:** H1 + dexterous hand grasping cloth towel, pouring water (fluid), wringing wet rag (coupling). 1× realtime sufficient; camera + tactile streamed to policy. Dominates coupling realism.
- **S3 · Warehouse Multi-Robot:** 64 mobile robots + arms in warehouse with liquid spill + crushable boxes. Per-env complex but parallel-trained.
- **S4 · Diff-Sim System Identification:** Robot + soft object; differentiate loss over 2 s sim → identify material parameters. Full pipeline must be differentiable. Architecture fundamentally different from non-diff.
- **S5 · VLA Training Data Source:** Thousands of envs each with RGBD + semantic + tactile feeding a Vision-Language-Action model. Rendering and physics throughput equally critical.

**Recommendation:** S1 → S2 → S3 in order; S4 (diff-sim) as an architectural constraint across all phases (decide now: yes or no); S5 as final phase adding visual pipeline. Strongly recommended dropping S4 if possible because PhysX 5 itself is not differentiable, so this is a Brax/MJX competitive zone, not Isaac Lab.

**Decision:**
- Phase order: **S1 → S2 → S3** as recommended.
- **Differentiable simulation: YES.** Must be built in from day 1. Architecture is forward+adjoint everywhere.
- **S5 rendering: self-written CUDA RT or direct OptiX.** Final pick deferred to Round 10.

**Implications locked here:**
- Every CUDA kernel must support a paired adjoint path.
- State checkpointing infrastructure required.
- Contact discontinuity handling required (see Round 5).
- Memory ~2× baseline due to tape storage.

---

### Round 3 – Solver Math Model: The architectural keystone

**Context:** Choosing the unified solver formulation is the foundation under every later feature. Three paths in the literature. Wrong choice = rewrite half the engine in year 3.

**Options:**

- **Path A · XPBD-everything (NVIDIA Flex / PhysX 5 unified particles).** Everything modeled as point masses + constraints: rigid = shape-matched particle clusters; articulation = distance + angle constraints; cloth = distance + bending; soft = volume constraint; fluid = density constraint (PBF). One kernel loop projects all constraints. Coupling is free (everything is in the same particle pool). Naturally differentiable. Pros: simplicity, unification, diff-friendly. Cons: rigid precision suffers (robot joints jitter, end-effector accuracy low), high-stiffness materials are numerically stiff, articulated robots (H1/Go2) underperform Featherstone/MuJoCo accuracy. **Loses to MuJoCo on robot control.**
- **Path B · Subsystem-optimal + unified coupling layer (PhysX 5 / Isaac Sim actual).** Rigid via Featherstone ABA (reduced coord, robot-grade accuracy) + TGS. Soft via FEM-tet (corotated linear, industry standard). Fluid via PBF particles. Coupling via SDF contact + Lagrangian multipliers at boundaries. Each subsystem uses its best algorithm → physics quality maxed. Pros: best-in-class per domain. Cons: 4 kernel sets + 4 data layouts + 4 gradient paths + 4 BVHs + 4 schedulers; 3-5× engineering volume; coupling stability is a research problem (PhysX 5 hasn't fully solved it); differentiability becomes very hard.
- **Path C · MPM-unified (Taichi / MPM-NG).** Everything as MPM continuum medium: rigid = ultra-stiff elastoplastic, fluid = von Mises yield, snow/sand = elastoplastic, cloth = anisotropic MPM. Grid + particle dual representation, natural coupling. Pros: physics-rich. Cons: rigid contact and robot accuracy are disasters (MPM is wrong for precise articulated dynamics); grid memory explodes; performance bottleneck at P2G/G2P scatter; differentiable but gradient-noisy. **Wrong bet for robotics.**

**Recommendation: Path B variant — subsystem-optimal but sharing one CSR-like constraint-row scheduler.**

| Subsystem | Solver | Representation | Diff strategy |
|---|---|---|---|
| Articulated robot | Featherstone ABA + TGS | reduced coord | Analytical adjoint (MJX-equivalent) |
| Free rigid | PGS / TGS | maximal coord | Analytical adjoint |
| Soft / cloth / muscle | XPBD with compliance | particles + constraints | Naturally differentiable |
| Fluid / granular | PBF (MPM optional future) | particles | Naturally differentiable |
| Coupling | Unified constraint-row scheduler | SDF contact → row → shared Jacobi/SOR | Shared adjoint channel |

Key insight: agent.md's "reusable CUDA constraint-row scheduler" is exactly the central hub of Path B. All subsystems compile their constraints (Featherstone joint, PBF density, XPBD distance, rigid contact, SDF coupling) into the same row format. Scheduler is subsystem-agnostic.

**Decision:** **B variant, with dual-track articulation:** Featherstone reduced coordinates for articulated chains + maximal coordinates for free rigid; bridged at the contact row layer.

**Implications locked here:**
- Row format must support both maximal-coord and Featherstone-generalized Jacobians (see Round 4).
- Featherstone ABA forward dynamics must be implemented on CUDA (no current Featherstone state in code; ArticulationGraph is only topology).
- Coupling rows must handle rigid (maximal) + Featherstone-link + particles in the same scheduler iteration.

---

### Round 4 – Row Format: The IR that everything compiles to

**Context:** Current `ConstraintBlock` (in `src/constraint/constraint_block.hpp`) is hardcoded for maximal coordinates: 2 bodies, 6 rows max, 4 dense 3-vector Jacobians per row, no compliance, no adjoint hooks, no path for N-body rows. The Path B variant requires a redesign. Four directions.

**Options:**

- **Option 1 · Tagged union.** Keep `ConstraintBlock`, add enum values for `XPBDDistance`, `PBFDensity`, `FeatherstoneJoint`, `Coupling`. Switch-case dispatch in solver. Pros: cheap migration. Cons: every new physics form grows a branch; GPU SoA dies; adjoint table sprawls; becomes Swiss-army-knife type. **Maintenance nightmare in 3 years.**
- **Option 2 · SoA per row class.** Fully split by row family: `MaximalContactRow[]`, `FeatherstoneJointRow[]`, `XPBDDistanceRow[]`, `PBFDensityRow[]`, `CouplingRow[]`. Each has its own SoA layout and dedicated CUDA kernel. Pros: peak SoA performance. Cons: 5+ classes × (forward + reverse + warmstart + island) = ~30+ kernels; solo+AI cannot keep consistent; each new physics form requires full new kernel set. **PhysX-internal style; requires 20-person physics team.**
- **Option 3 · Unified compressed sparse row (CSR-like universal row).** Single row format; each row holds variable-length body list + CSR Jacobian + compliance + bounds + flags + adjoint_kernel_id + row_class_id. All constraints (maximal contact, Featherstone joint, XPBD distance, PBF density, rigid-soft coupling) compile to the same format. Scheduler is class-blind. Adding a new physics form = one IR entry + one evaluator + one adjoint, scheduler untouched. Pros: one mental model; AI agents can hold it; one scheduler, one warmstart, one island, one tape; **matches agent.md vision of "reusable scheduler"**. Cons: 10–15% perf cost from indirection (PhysX 5 / Warp pay the same tax); requires rewriting `ConstraintBlock` and migrating existing contact/joint/drive code.
- **Option 4 · Two parallel formats.** Keep `ConstraintBlock` for existing maximal stuff; new `GeneralizedRow` for Featherstone/XPBD/PBF/coupling. Scheduler iterates both formats in sequence. Pros: zero migration of existing code. Cons: double everything (warmstart, island, tape); coupling between formats needs conversion. **Compromise trap.**

**Recommendation: Option 3.** Reasoning compounded by solo + AI constraint:
1. AI is great at code, weak at architectural consistency. One format leaves AI no room to drift.
2. Other options have "mental-surface-area explosion" failure mode for solo maintenance.
3. Differentiability mandates Option 3 — adjoint kernel registry as flat lookup; other options scatter adjoints across many files.
4. Existing 1244-line `cuda_constraint_solver.cu` doesn't account for XPBD compliance / Featherstone Jacobian / diff tape anyway; clean rewrite is honest sunk cost.

**Decision:**
- **Row format: Option 3 (CSR-like Universal Row).**
- **Migration accepted: 2–3 months infrastructure refactor with no visible new demos.** S1 work pauses during this period. Diff-test bridge runs old PGS and new CSR side-by-side until tolerance.
- **Featherstone strategy: ABA recursion for joint dynamics; rows only for joint *constraints* and contacts.** Forward integration is not row-based (would degrade Featherstone's O(n) recursion to O(n²) full Jacobi).
- **Adjoint mechanism: codegen.** Not CUDA function pointers (no inlining, SM perf loss) and not runtime dispatch (slow). Codegen produces specialized forward + adjoint kernels per row_class_id.

**Implications locked here:**
- Codegen pipeline becomes part of the build (CMake step).
- Generated kernels carry `// GENERATED – DO NOT EDIT` header.
- Row IR YAML schema is the source of truth.
- Adding a new row class = author IR + regenerate. AI agents author IR but never touch generated kernels.

---

### Round 5 – Diff-Sim: Contact discontinuity + tape storage

**Context:** Differentiable simulation through rigid contact is non-trivial because contact is discontinuous (in-contact vs not-in-contact, Coulomb stick/slip boundary). Both the discontinuity strategy and the tape storage strategy enter the codegen IR for every row evaluator.

**Sub-question 5a – Contact discontinuity:**

- **α · Smooth contact (Brax / dflex / Tiny Diff Sim).** Replace hard contact with always-active spring-damper; distance < ε → continuous force. All gradients everywhere differentiable. Cost: physics departs from reality — robot walking jitters, impact forces absorbed by springs; sim-to-real gap widens; Brax-trained policies generally need RMA / PPO + heavy domain randomization to transfer.
- **β · Stop-gradient at events (DiffTaichi / Warp default).** Forward runs true hard contact; reverse treats contact events as parameter-independent, only backprops through smooth segments. Biased gradient → slower training but terminal physics is correct; contact-mode-switch sensitivities ("should foot land here or there") not captured. **MuJoCo MJX uses this and it works for robot control.**
- **γ · Randomized smoothing / analytic policy gradient (Suh et al. 2022).** Add noise to contact parameters, take expectation → naturally smooth. Maintains hard physics. Cost: N samples per step (4–16×), training throughput cut; implementation complex; not yet industrialized.

**Recommendation: β with α as per-env opt-in switch.** Flagship target is Go2/H1 real robots → β's sim-to-real is far better; β implementation simplest (`if lambda > 0 then dλ else 0` in adjoint); α retained for S2 (cloth/water naturally suits soft contact); γ too research-stage for v1.

**Decision: β (stop-gradient at events) with α retained as runtime switch per-env. γ not pursued in v1.**

**Sub-question 5b – Tape storage:**

- **Path 1 · Full tape (DiffTaichi / Warp default).** Push lambda, position, velocity, J·v, contact list per step; pop on reverse. Memory = step_count × bytes_per_step. S1 case (4096 Go2 × 1000-step episode × ~10KB/step) ≈ 40 GB. **Out of single-card memory.**
- **Path 2 · Gradient checkpointing (Chen 2016 / PyTorch).** Save state every K steps; reverse recomputes from nearest checkpoint. Memory ≈ (step_count/K) × state + K × tape. With K = √step_count, memory scales √N. Reverse 2–3× slower than forward. Implementation: codegen IR adds "recompute" mode.
- **Path 3 · Implicit function theorem at convergence (PhysX 5 DiffSim / Theseus / JAX MD).** When solver converges, exploit IFT to backprop only through the final KKT system; no intermediate state stored. Memory zero between checkpoints. Requires solver actually converges (current PGS is fixed-iteration, not converged); contact-mode-switch gradients lost; needs sparse LU/CG reverse solver.

**Recommendation: Path 2 + Path 3 hybrid.** Default checkpointing K = 50–100; IFT only on guaranteed-convergent subsystems (Featherstone ABA inner loop, PBF density iteration). Full tape (Path 1) only in debug mode, banned in training.

**Decision: Path 2 + Path 3 hybrid. Path 1 banned in training.**

**Sub-question 5b extension – Sparse linear solver for IFT:**

Three options for the required sparse linear solver (IFT reverse solves KKT system):

- Self-written CUDA CG / MINRES / GMRES + preconditioners (full control, deterministic, 4–6 months work).
- NVIDIA cuDSS — **considered and rejected** (closed-source SDK; its reduction order is not bit-exact, which would force the IFT path to D2).
- Ginkgo / AMGX (open source CUDA sparse libraries).

**Initial recommendation:** cuDSS now + self-written CG fallback for the deterministic deep path.

**Decision (amended 2026-05-31; supersedes the Round 13 cuDSS resolution):** **No cuDSS — no closed-source SDK, anywhere, ever.** v0.5 ships a **self-written deterministic CG + Jacobi / Block-Jacobi solver with fixed-order tree reductions (D1 bit-exact)** as the IFT backend from the start; the v0.5 scope (rigid + Featherstone) produces SPD KKT/Schur systems, so CG + Jacobi is the minimal sufficient solver. The broader self-written suite — **MINRES / GMRES + ILU(0) / AMG + factorization** — extends this same core in **v0.7+**. Caller-facing API is stable, so adding methods later is transparent.

**Sub-question 5b extension – IR fields:**

Codegen IR must add four fields to support these decisions:
- `gradient_mode`: { dense_adjoint, stop_grad_on_event, ift_at_convergence, none }
- `recompute_mode`: { tape, checkpoint, recompute_always }
- `event_flag_field`: bit position within row for "discontinuous event fired"
- `contact_softness`: { hard, soft_alpha } per-row switch

**Decision: All four IR fields accepted.**

---

### Round 6 – Determinism contract + multi-GPU strategy

**Context:** Determinism affects every kernel template (atomics, reduction order, RNG). Multi-GPU affects memory layout from day 1. Both decisions must enter the codegen template up front; retrofitting later means rewriting kernels.

**Sub-question 6a – Determinism contract:**

- **D1 · Strong.** Bit-exact across runs on same GPU + driver. Forbids float `atomicAdd`; cross-row body writes must go through graph coloring or segment reduction; reduction trees use fixed order; Featherstone chain recursion cannot use warp shuffle. Cost: 10–20% perf, +30% engineering complexity, but island/coloring scheduler is forced into v1 instead of "future work."
- **D2 · Weak.** Same seed in same process repeats; cross-process / cross-GPU not guaranteed. Allows atomics but same-stream timing must be fixed. Cost: minimal perf hit; cross-worker trajectory comparison impossible.
- **D3 · Non-deterministic.** PhysX / Bullet default. RL training works (PPO tolerates ulp noise) but bug-hunting with same-input different-output is hell.

**Recommendation: D1 as default contract.** Reasons:
1. Flagship targets are RL + diff-sim. Diff-sim has zero tolerance for non-determinism (gradient direction shifts → training non-convergence + unreproducible bugs).
2. agent.md already calls out "shared rigid writes still rely on atomics" as a gap; D1 is the entry ticket to closing this.
3. Codegen templates built on D1 assumptions (must declare write set + graph-color) propagate the discipline to every new row class automatically.
4. The 10–20% perf cost is recoverable through per-class kernel specialization.
5. **Strong determinism is the cleanest differentiator vs PhysX / Isaac Lab.** Academic diff-sim users will value it.

D1's required cost: island/coloring scheduler upgraded from "future" to "v1 must-have"; no coloring/segment-reduction = D1 cannot run. Also: self-written deterministic CG dot-product (fixed-order tree reduction; cannot use cuBLAS `cublasDdot`).

**Sub-question 6b – Multi-GPU:**

- **M1 · Single GPU N envs.** All envs in one GPU. Simple. S1 (4096 Go2) fits on RTX 4090. Limitation: S3 doesn't fit; no scale-out.
- **M2 · Multi-GPU data-parallel envs (Isaac Gym/Lab actual).** Each GPU runs an independent env batch. NCCL only at policy gradient aggregation. No physical communication between GPUs. Cost: per-card linear scale; but single env still cannot cross cards, so S3 hard if single env exceeds 80 GB.
- **M3 · Single scene partitioned across GPUs (Omniverse / ANSYS).** Spatial domain decomposition; NCCL boundary sync. Cost: PhD-thesis level complexity; diff-sim cross-card gradients much harder; not feasible for solo + AI in 3 years.

**Recommendation: M2 default + M1 initial milestone + M3 explicitly abandoned.** S1/S2/S3 should all fit on a single RTX 4090 / H100 — this is the design anchor. Multi-robot = multi-env. M2 needs ~no codegen changes; only outer batch loop gains a device dimension. M3's complexity is deferred to RL framework (rl_games / TorchRL handles data-parallel).

Hard envelope: single-env memory cap is GPU VRAM. S3 (warehouse + fluid + 64 robots) must fit 80 GB. Constraints: particles < 1 M, soft tets < 200 K, total rigids < 50 K.

**Decision:**
- **6a: D1 strong determinism.** Island/coloring scheduler is v1 must-have. Self-written deterministic CG accepted.
- **6b: M1 + M2 both supported and selectable at runtime. M3 explicitly out of scope.**
- **80 GB single-card envelope accepted.** Particles < 1 M, tets < 200 K, rigids < 50 K per env.

---

### Round 7 – Cross-system coupling

**Context:** S2/S3 demos depend on rigid ↔ soft ↔ fluid coupling. PhysX 5 / Isaac Lab haven't fully solved this; it's a real differentiation opportunity. Eight coupling-pair types arise across S2/S3.

**Coupling pair inventory (from S2/S3):**

| Pair | S2 example | S3 example |
|---|---|---|
| Featherstone link ↔ free rigid | H1 grasps cup | Arm grabs box |
| Featherstone link ↔ soft particle | H1 grabs towel | Foot on soft packaging |
| Featherstone link ↔ fluid particle | H1 hand into water | Robot wades |
| Free rigid ↔ soft | Cup on towel | Box crushed |
| Free rigid ↔ fluid | Cup with water | Bucket sloshing |
| Soft ↔ fluid | Towel absorbs water | Box soaked |
| Soft ↔ soft (incl. self) | Towel self-folds | Boxes stacked |
| Fluid ↔ fluid | (PBF internal) | (PBF internal) |

**Options:**

- **K1 · All-particle (NVIDIA Flex).** Everything is particles; coupling = particle-particle contact, one row type. Pros: codegen minimal. Cons: rigid precision fails; **conflicts with Featherstone decision; out**.
- **K2 · SDF boundary + particle query (PhysX 5 / Houdini Flex).** Each rigid (including Featherstone link) gets an SDF at cook time; soft / fluid particles query the SDF for distance + normal; contact emitted as a row. Pros: per-particle local query, GPU-friendly; rigid receives reaction impulse via row's reverse Jacobian. Cons: every rigid mesh must cook an SDF (memory + time); thin shells lost at low SDF resolution; Featherstone links need pose-aware queries (point transformed to link's local frame each step).
- **K3 · Cross-system direct constraint rows (natural extension of the row scheduler).** Each coupling candidate pair → narrow phase → row with Jacobian spanning two systems. Scheduler doesn't distinguish. Pros: codegen IR barely changes; coupling row = two-body contact row variant. Cons: cross-system broadphase is a research problem (rigid BVH vs particle grid); 1 M particle × 50 K rigid candidate explosion; row counts of 10⁵–10⁶ per env per step.

**Recommendation: K2 + K3 hybrid, routed per pair type:**

| Pair | Path | Row class |
|---|---|---|
| rigid ↔ rigid (maximal + Featherstone) | K3 (existing) | `MaximalContactRow`, `FeatherstoneContactRow` |
| rigid ↔ soft particle | **K2** SDF query | `RigidSDFContactRow` (1 rigid + 1 particle) |
| rigid ↔ fluid particle | **K2** SDF query | `RigidSDFContactRow` |
| Featherstone link ↔ particle | **K2** pose-aware SDF | `FeatherstoneSDFContactRow` |
| soft ↔ fluid particle | **K3** cross-system row | `ParticleParticleContactRow` + density correction |
| soft ↔ soft (incl. self) | **K3** same-system row | `ParticleParticleContactRow` |
| fluid ↔ fluid | PBF internal (not emitted) | `PBFDensityRow` |

Why K2 + K3 hybrid: K2 reduces rigid-particle candidate explosion from 5 × 10¹⁰ pairs/step (S3 worst case) to O(N_particles). K3 only between particle-particle, where spatial hash makes it O(N_particle × avg_neighbors=30) — controllable.

**Sub-decisions (consequences):**

1. **SDF infrastructure (cooker work):**
   - Offline mesh → SDF generated in cooker (USD/URDF/MJCF import path adds this stage).
   - Same-mesh instances share one SDF (essential, else 50 K rigid × 128³ grid blows VRAM).
   - LOD: thin-shell / complex meshes use higher resolution; simple convex use lower.

2. **Broadphase upgrade:**
   - Current SAP only does rigid-rigid.
   - Add: particle uniform grid hash (PBF standard); rigid LBVH (replaces SAP — SAP scales poorly for large dynamic sets); cross-system query (particle grid cell → rigid BVH leaves).

3. **Two-way reaction force:**
   - K2 path: particles receive SDF gradient force; rigid receives reverse Jacobian impulse. To preserve D1, rigid accumulation must use segment reduction per rigid body id, not float atomics.

4. **Thin-shell SDF — the MuJoCo research:**

   MuJoCo 3.0 (Dec 2023) introduced SDF collision via plugins, using Newton gradient descent on the sum of two SDFs to find the contact point. This is more accurate than nearest-point sampling and is gradient-friendly. **However, MuJoCo does not magically solve thin-shell.** It mitigates via:
   - Higher local resolution (narrow-band / sparse SDF, OpenVDB-style; 10–50× memory savings vs dense)
   - V-HACD convex decomposition per piece (each piece carries its own SDF)
   - Manual primitive replacement (capsule/box for thin parts)

   MJX (MuJoCo GPU) currently supports SDF only for analytical primitives; mesh SDF on GPU remains limited. Genesis (2024–25) extended to sparse VDB SDF for high local resolution.

   Implication for Nuka: **default to narrow-band sparse SDF (OpenVDB-style, self-written CUDA), V-HACD convex decomposition in the cooker, Newton-on-summed-SDF contact (MuJoCo-style, analytical-adjoint friendly).** True zero-thickness cloth (towel) bypasses SDF and uses a `TrianglePointContactRow` class.

5. **Coupling-row adjoint:**

   Decision: full analytical adjoint required, no finite-difference shortcuts. SDF gradient w.r.t. rigid pose computed analytically (chain rule through rigid transform); for Featherstone links, chain rule through articulation kinematics back to joint position. ~doubles codegen kernel complexity but unlocks complete diff-sim across coupling.

**Decision:** **K2 + K3 hybrid as recommended. New cooker SDF pipeline accepted. Broadphase upgrade (SAP → BVH + particle grid + cross-system query) accepted. Thin-shell handled via narrow-band sparse SDF + V-HACD + triangle-particle row. Full analytical adjoint, no shortcuts.**

---

### Round 8 – C++ ABI shape + Python bindings

**Context:** Top-level user goal: "seamless integration into any C++." Each public type's design is constrained by this.

**Sub-question 8a – C++ ABI shape:**

- **Path a · Header-only + CUDA in headers.** Forces every user to install NVCC + matching stdlib. UE5/Unity native plugins dead; ROS2 painful. **Out.**
- **Path b · Pure C ABI core + thin C++20 wrapper headers (Vulkan / D3D12 model).** Core compiles into `.so`/`.dll`; cross-boundary surface = `extern "C"` + POD + opaque handles. C++ wrappers add RAII / `std::expected<T, Err>` / `std::span` for ergonomics. Any C/C++ compiler can link; PyTorch / Rust / Go can FFI. Cons: writing C ABI is tedious (no templates / inheritance across boundary); double bookkeeping (C interface + C++ wrapper). **Wins on portability.**
- **Path c · Stable C++ ABI via PIMPL.** Exposes C++ classes (forward declarations + impl pointer). CUDA hidden. Cons: cross-stdlib (libstdc++ vs MSVC STL), cross-exception-policy still incompatible; UE5/Unity still unhappy. **Doesn't truly solve.**
- **Path d · gRPC / shared-memory IPC.** Cross-process. Cost: sub-ms per step latency unacceptable for S1.

**Recommendation: Path b.** Concrete design:

```c
typedef struct nuka_device_t* nuka_device_handle;
typedef struct nuka_world_t*  nuka_world_handle;
typedef struct nuka_buffer_t* nuka_buffer_handle;

nuka_result_t nuka_device_create(uint32_t gpu_index, cudaStream_t user_stream, nuka_device_handle* out);
nuka_result_t nuka_world_build_from_usd(nuka_device_handle dev, const char* usd_path, uint32_t env_count, nuka_world_handle* out);
nuka_result_t nuka_world_step(nuka_world_handle w, float dt);
nuka_result_t nuka_world_read_state(nuka_world_handle w, nuka_buffer_handle dst, nuka_state_field_t field);
void          nuka_world_destroy(nuka_world_handle w);
```

```cpp
namespace nuka {
class World {
    nuka_world_handle h_;
public:
    World(const Device& dev, std::string_view usd_path, uint32_t env_count);
    ~World() noexcept;
    void Step(float dt);
    std::expected<StateView, Error> ReadState(StateField field);
};
}
```

Constraints:
- **Caller-owned `cudaStream_t`.** No global CUDA context (avoids fighting PyTorch / UE5 / Omniverse).
- **No STL across boundary.** `nuka_string_view` (const char* + size_t), `nuka_span` (ptr + size).
- **No exceptions across boundary.** `nuka_result_t` everywhere.
- **Opaque handles.** User cannot deref.
- **Buffer protocol = device pointer + size + dtype + layout descriptor.** Zero-copy to PyTorch via DLPack.

Consequences for existing code:
- `phi/device.hpp` `SetDevice` is thread-global; acceptable. But `phi/stream.hpp` `Stream` currently constructs its own; needs refactor to a non-owning view wrapping caller-supplied `cudaStream_t`. ~2 weeks work.
- `phi/platform_contract.hpp` `PlatformContract` is singleton-style; needs per-`Device` handle config. ~1 week.
- All `throw`s in current code must become `nuka_result_t` returns.

**Sub-question 8b – Python bindings:**

- **i · nanobind.** Compile 5× faster than pybind11; auto stub generation; C++20 friendly; built-in DLPack support; built-in `nb::ndarray<>` with dtype/shape checks; smaller ecosystem.
- **ii · pybind11.** Largest ecosystem but slow compiles and template bloat.
- **iii · ctypes / cffi.** Zero extra dependency but hand-wrap on Python side.
- **iv · No Python.** Cede the RL ecosystem.

**Recommendation: i (nanobind), and package the Python layer as a drop-in for Isaac Lab's `ManagerBasedRLEnv` API.**

Strategic point: mimicking Isaac Lab API shapes = leveraging their ecosystem. This is critical leverage for a solo project against an NVIDIA team.

**Decision:**
- **8a: Path b.** Pure C ABI + thin C++20 wrapper.
- **8a consequences accepted:** caller-owned CUDA stream; PHI Stream refactor; PlatformContract de-singletonize; no exceptions across boundary; everything via `nuka_result_t`.
- **8b: nanobind.** Isaac Lab drop-in scoped to RL training path API only (no UI / editor / Omniverse integration).

---

### Round 9 – C++ rendering engine integration (the real "seamless C++" goal)

**Context:** User clarified the primary use case for "seamless C++ integration" is **embedding into existing C++ rendering engines** (UE5 / Unity native / Godot 4 / custom Vulkan/D3D12) and visualizing physics there. Python training is secondary. This pivots the C ABI design from "hide CUDA" to "CUDA ↔ graphics API interop."

**Options:**

- **I1 · CUDA external memory interop (NVIDIA, zero-copy).** `cudaImportExternalMemory` + `cudaImportExternalSemaphore` map D3D12/Vulkan buffers/textures into CUDA address space. Physics writes directly into renderer-owned buffer.
  - Vulkan: `VK_KHR_external_memory_fd` (Linux) / `VK_KHR_external_memory_win32` (Windows).
  - D3D12: `ID3D12Resource` shared handle → CUDA map.
  - Metal: CUDA has no Metal interop → macOS out.
  - Pros: zero copy; CUDA semaphore signal + graphics wait gives clean frame sync.
  - Cons: per-API glue; D3D12 resource state transitions error-prone; no AMD/Intel.

- **I2 · CPU staging buffer.** Each frame, physics DMA-copies transforms / particles back to CPU; renderer uploads to its own GPU buffer.
  - Pros: universal across GPUs and graphics APIs.
  - Cons: S3 (10⁵ rigid transforms + 10⁶ particle positions) ≈ 50–100 MB GPU↔CPU↔GPU per frame; ~5–15 ms overhead per frame, kills real-time.

- **I3 · Engine consumes our CUDA RT framebuffer.** Since user committed to self-written CUDA RT (Round 10), let it also output the final image; external engine just receives a `cudaArray` as background.
  - Pros: one-line consumer.
  - Cons: loses engine's own pipeline (materials, post-processing, lights, UI overlays); not real integration.

**Recommendation: I1 primary + I2 universal fallback + I3 reserved for sensor / dataset modes.**

Priority order:
1. **Vulkan interop first** (existing Vulkan backend in `src/render/vulkan_renderer.cpp`; Linux training environment); S1 done in **Windows + Vulkan**.
2. **D3D12 interop second** (UE5 Windows = biggest game/robotics demo market); S2 phase.
3. **CPU staging fallback** (non-NVIDIA + legacy engines); 1–2 weeks any time.
4. **CUDA RT framebuffer** (merges with sensor path); S5 phase.

Frame pacing:
- Physics fixed step (1/120 or 1/240 s).
- Render reads latest state asynchronously.
- C ABI exposes `nuka_world_get_render_view(world, frame_index, sync_semaphore)` returning the "last completed step" buffer view + semaphore; renderer waits on semaphore.
- Training disables rendering: 4096 envs × 200 Hz cannot also render; render attaches only in "debug attach" / "demo" mode.

Threading model:
- Each `nuka_world_handle` binds one CUDA stream + one device.
- The thread that calls `nuka_world_step()` advances physics.
- Render thread is independent, calls `nuka_world_get_render_view()` for buffer + semaphore.
- Engine does not spawn threads internally.

Render-data granularity exposed via C ABI:
- `RigidBodyTransform[]` – 7-component pose per body.
- `ArticulationLinkPose[]` – per-chain per-link pose from Featherstone FK.
- `SoftBodyVertex[]` – soft vertices from XPBD particle pool.
- `FluidParticle[]` – PBF particle positions + optional density field for surface reconstruction.
- `ContactDebugViz[]` – contact points + normals (debug only).

Each is exposed as a DLPack descriptor so Python can also read it directly.

**Decision:**
- **I1 + I2 + I3 + I4 layered approach accepted.**
- **S1 development happens on Windows, priority Vulkan interop.**
- **Platform market limited to NVIDIA accepted** (macOS / AMD / Intel deferred; PHI layer preserves future extension).
- **UE5 / Unity / Godot reference plugin work deferred** — user has their own simple renderer and will rely on it through early phases.
- **JAX support: yes, second nanobind module, ~2 weeks work, introduced at v0.5.**
- **PyTorch autograd: v0.1 skeleton (just signature + tape entry), v1.0 complete adjoint path** (so tape design is constrained by autograd interface from day 1).

---

### Round 10 – Sensor matrix + RT renderer scope

**Context:** S1 needs only IMU + joint encoders + maybe basic depth. S5 needs full RGB + semantic + tactile at scale. Between these endpoints sit S2 (RGB + depth + tactile for manipulation) and S3 (lidar + semantic for navigation). Three interlocked decisions.

**Sub-question 10a – Sensor matrix by phase:**

| Sensor | S1 (Go2) | S2 (H1) | S3 (Warehouse) | S5 (VLA) | Difficulty |
|---|---|---|---|---|---|
| IMU | ✅ done | needed | needed | needed | done |
| Joint encoder | needed | needed | needed | needed | from Featherstone state |
| Tactile (per-contact normal/tangential) | optional | **required** | optional | needed | read row lambda |
| Force/torque 6-axis | optional | **required** | needed | needed | accumulated contact lambda |
| Lidar (line scan / spinning) | ✅ done | optional | **required** | needed | CUDA ray query |
| Depth camera | debug | **required** | **required** | needed | RT required |
| RGB camera | debug | **required** | needed | **required** | RT + shading |
| Semantic mask | – | optional | **required** | **required** | RT + instance ID buffer |
| Normal / UV | – | optional | needed | **required** | RT byproduct |
| Event camera | – | – | optional | optional | different model |
| Proximity / ultrasonic | – | optional | optional | – | same as lidar |
| GPS / odometry | optional | – | needed | optional | state + noise |

Key takeaway: S2 onward requires industrial RT pipeline (depth + RGB + tactile + F/T). S1 only needs lidar + IMU.

**Sub-question 10b – RT backend:**

| Path | OptiX 7+ | Self-written CUDA BVH | OptiX + self-written fallback |
|---|---|---|---|
| RT Cores | ✅ 5–10× | ❌ | ✅ primary |
| D1 determinism | ⚠️ not guaranteed across drivers | ✅ | primary loses, diff path keeps |
| Differentiability (ray gradients) | very hard (OptiX internals opaque) | ✅ natural | diff goes self-written |
| Engineering cost | 1–2 months integration | 6–8 months self-write | 2–3 months hybrid |
| Driver lock | NVIDIA closed SDK | pure CUDA | NVIDIA lock (accepted) |
| MAS perf | 4096 env × 64-line lidar real-time | 4096 env × 16-line strained | matches OptiX |

**Sub-question 10c – Sim-to-real noise:**

Depth:
- N1 · Geometric noise (Gaussian / Poisson on sensor readings).
- N2 · Physical noise (lighting / material / reflectance / camera intrinsics randomized per episode).
- N3 · Sensor physical modeling (lidar beam divergence, rolling shutter, lens distortion, motion blur, dark current, ISP pipeline).

Trigger frequency: per-step / per-episode / static randomization.

**Recommendation table:**

| Stage | Sensor matrix | RT impl | Noise |
|---|---|---|---|
| v0.1 (S1) | IMU, joint, lidar, debug depth | current CUDA ray query + OptiX basic integration | N1 |
| v0.5 (S2 entry) | + tactile + F/T + RGB + full depth | OptiX primary ready | N1 + N2 |
| v1.0 (S2 polish) | + semantic + normal/uv | OptiX optimized + self-written BVH skeleton for diff | N1 + N2 + N3 key items |
| v2.0 (S3 / S5) | + event camera + high-throughput RGB | OptiX max + self-written diff path | N3 complete |

**Decision (departing from recommendation):**
- **10a sensor matrix: as recommended, but tactile + force/torque must ship in S1 too** (RL uses joint mechanics readings from the start; reading row lambda is essentially free).
- **10b RT backend: pure self-written CUDA RT. No OptiX dependency.** Trade-off: 5–10× slower than OptiX with RT Cores, so 4096-env high-res RGB has a lower ceiling; but full determinism + full differentiability + zero closed-SDK dependency. Consistent with full-analytical-adjoint commitment from Round 7.
- **10c noise: v1 N1 + N2, v2 N3 key items.** Confirmed.

**Implications:**
- Tasks: self-write LBVH (Karras), per-frame refit, stackless traversal with persistent threads, mesh-triangle + particle-sphere + sparse-SDF intersection, simple GGX/Lambert shading, analytical adjoint for all primitives. ~6–8 months.
- OptiX integration tasks dropped from the engineering graph.

---

### Round 11 – Validation strategy (the solo + AI survival layer)

**Context:** AI agents will write physics kernels that compile, pass unit tests, but are physically wrong. This is not a question of *if* but *when*, and it happens every week. Without strong validation infrastructure, the engine accumulates **silent physics errors** over years (energy leaks per row class, occasional penetrations, Featherstone drift on long chains). Sim-to-real failure becomes unattributable. agent.md already directs against TDD-as-completion-proof — but the replacement infrastructure must be built.

**Five layers, all required:**

**Layer V1 – Oracle regression (golden trajectories).** For each subsystem, pick a community-accepted reference engine, fix inputs, record reference trajectory in Git LFS, diff per commit.

| Subsystem | Oracle | Tolerance |
|---|---|---|
| Featherstone ABA (Go2/H1 joint dynamics) | **MuJoCo MJX** primary + **Pinocchio** secondary | < 1e-4 joint angle over 1 s |
| Free rigid contact (PGS) | **Bullet** or MuJoCo | < 1e-3 position over 1 s |
| Static / slip friction | MuJoCo (strictest contact model) | slip-event frame error ≤ 1 |
| XPBD soft / cloth | **Houdini Vellum** or NVIDIA Flex paper case | < 1% vertex position |
| PBF fluid | **NVIDIA Flex paper reconstruction** (ball-into-water) | volume conservation ±2%; surface waveform qualitative |
| SDF contact | **MuJoCo 3.0 SDF plugin** | normal angular error < 0.5° |
| Adjoint per row class | **finite difference** | relative error < 1e-3 |

Mechanism: `tests/oracle/` directory, one scene per subsystem; oracle trajectory file + engine output + diff report; CI runs fast subset, nightly runs full set.

**Layer V2 – Physics invariant monitoring (every step / sampled).** No oracle required; engine self-checks. Sampled every N steps; trace stored; trends analyzed.

Quantities watched:
- Energy (kinetic + potential + elastic), respecting external work.
- Linear / angular momentum (no-external-force case).
- Constraint residual after PGS/TGS convergence: `|J·v + b| < threshold`.
- Mass conservation (PBF particle count invariant).
- Joint range bounds.
- Featherstone link length stability.
- NaN / Inf detection.
- Velocity / position envelope.

Mechanism: lightweight `core/diagnostics/` module, dev-mode on, prod-mode off. Training dashboard shows curves per env. Drift triggers auto trace dump for diagnosis.

**Layer V3 – Adjoint vs finite-difference check (per row class, CI).** Every codegen-produced (forward, adjoint) pair runs FD verification on representative perturbations. Failures (>1e-3 relative error) block merge.

Checks:
- Forward gradient: perturb each parameter, compute (engine forward at p+δ − engine forward at p)/δ, compare to analytical adjoint.
- Jacobian-vector product (`J · v`) vs FD.
- Vector-Jacobian product (`vᵀ · J`) vs FD (the backward direction).
- Chain correctness: composing two rows, each passes individually, composition also passes.

**Layer V4 – Per-S-phase demo suite.** Each phase defines representative demos run on a schedule, producing PPM sequences + metrics.

| Phase | Demo | Pass criteria |
|---|---|---|
| S1 | Go2 stand 5 s | No fall; energy drift < 2% |
| S1 | Go2 PPO 4096 envs 100-step ep | Convergence; step-time bar met |
| S2 | H1 grasp cup + place on table | No interpenetration; grasp success |
| S2 | H1 pour water into cup | Volume conserved; stable surface |
| S2 | H1 wring towel | Cloth large deformation stable; no blow-up |
| S3 | 64 robots + fluid spill | System stable; fits 80 GB envelope |
| S5 | RGB + depth + semantic + VLA prototype | Render quality + throughput targets |

**Layer V5 – AI agent guardrails (solo + AI specific).** AI-specific failure modes:

- **Mandatory review protocol:** any `*.cu` change requires running oracle subset + adjoint check before merge.
- **Codegen IR is source of truth.** AI may not edit generated kernels; only IR + regenerate.
- **physics-smell lint:** banned patterns enforced in CI (`atomicAdd<float>`, hot-path `cudaMallocAsync`, `__shared__` across row classes within dispatch).
- **Diff-test bridge:** new row format runs alongside old PGS until diff stays under tolerance, then switch.

Build order:
1. V5 (AI guardrails) – W1 of the project, cheapest.
2. V2 (invariant monitoring) – month 1; zero-oracle, runs immediately.
3. V1 (oracle regression) – months 1–6, parallel to CSR row refactor.
4. V3 (adjoint check) – bound to codegen pipeline, every row class.
5. V4 (demo suite) – bound to S phase, evolves naturally.

**Decision:** **All five layers accepted.** Oracle choices: **MJX primary + Pinocchio secondary for Featherstone; NVIDIA Flex paper reconstruction for fluid.** **Git LFS for golden trajectories accepted.** **AI cannot edit generated kernels accepted.** **Sim-to-real real-hardware validation deferred to post-S5 (when VLA training is operational and there is a real reason to deploy).**

---

### Round 12 – License + asset pipeline

**Sub-question 12a – License:**

Options:

- **Apache 2.0** – broad permissive + explicit patent grant. Industry default for robotics/AI: Isaac Lab, PyTorch, TensorFlow, Bullet, MuJoCo, Drake. Reduces user legal review friction.
- **MIT** – simpler, no patent grant. Riskier with patented algorithms (XPBD, PBF, MPM).
- **MPL 2.0** – file-level copyleft; partial.
- **GPLv3** – strong copyleft; commercial / engine integrations refuse. **Conflicts with "seamless integration into C++ engines".**
- **BSL / FUTO** – source-available + commercial license fee.
- **Closed proprietary** – no source release. Death sentence for a solo project (no community bug-finding, no AI-agent code visibility, no reference plugins).

**Recommendation: Apache 2.0 + CLA (Contributor License Agreement).** CLA preserves future re-licensing optionality (could move to dual-license commercial later if needed).

Timing of open-sourcing:
- v0.1–v0.3 phase (first 1–2 years): repo private, avoid public scrutiny of half-built work.
- v0.5 (first real demo): open + recruit community.
- This path matches Genesis / Brax history.

**Sub-question 12b – Asset pipeline:**

| Format | Current | Strategic value | Effort |
|---|---|---|---|
| USD (.usd/.usda/.usdc/.usdz) | ✅ .usda text | Core (Isaac Lab / Omniverse) | Binary needs OpenUSD SDK, ~1 month |
| URDF | ✅ | Robotics legacy | Maintain |
| MJCF | ✅ | MuJoCo migration | Maintain |
| glTF 2.0 | ❌ | Visual assets + game engines | 1–2 months |
| FBX | ❌ | Autodesk + games, closed SDK | Skip |
| OBJ | ? | Simple mesh | Few days |
| `.nuka` cooked binary | ❌ | Startup speed + cooked SDF/BVH cache | **Required**, 1 month |

**Recommendation:**
- USD all formats (binary via OpenUSD SDK) as authoritative — Isaac Lab parity.
- URDF + MJCF retained — robotics user base.
- glTF added at S5 — VLA needs visual assets.
- FBX skipped — closed SDK + heavy.
- `.nuka` cooked binary day 1 — without it, 4096-env training startup takes 10+ seconds each time.

MaterialX schema: document USD MaterialX custom attributes as engine's authoring contract. Soft body params (Young's modulus, damping, compliance) → USD MaterialX custom attrs. Fluid params (viscosity, surface tension, density) → same. Contact friction / restitution → extended from MJCF/URDF to USD.

**Decision:**
- **License: Apache 2.0 + CLA. Open at v0.5.**
- **Asset pipeline: as recommended. OpenUSD SDK integration at end of S1. `.nuka` cooked binary day 1. glTF added at S5. MaterialX schema documented.**

---

### Round 13 – Roadmap + phase gates + scope discipline + risk reserves (meta)

**Context:** All technical decisions locked. Now the question is **how to avoid the multi-year tar pit**. Solo + AI on a 5-8 year project has well-known failure modes: scope drift, bog-down in a subsystem, motivation loss, three half-finished subsystems never reaching demo.

**Initial recommended roadmap (calendar-bound):**

| Version | Initial duration | Anchor |
|---|---|---|
| v0.1 Foundation refactor | 6–9 months | Go2 stand on new infrastructure, MJX oracle passes |
| v0.3 S1 sprint | 6 months | 4096-env Go2 PPO video |
| v0.5 Open source + diff-sim | 6 months | gradient-based system ID on Go2 |
| v0.7 S2 entry | 9 months | H1 grasp cup |
| v1.0 S2 polish | 6 months | H1 pour + wring |
| v1.5 S3 entry | 9 months | warehouse multi-robot |
| v2.0 S5 entry | 12 months | VLA data generation |
| v3.0 Sim-to-real | open | real Go2 + H1 deployment |

Total to v2.0 ≈ 4.5 years. Plus v3.0 ≈ 5–6 years total.

**Initial recommended discipline rules:**
- Each phase's exit criteria must all pass before next phase begins.
- Stagnation 1.5× → architecture review; 2× → kill switch (cut demo, drop subsystem, or roll back architecture).
- No phase skipping. No "remember to fix later."
- Weekly primary objective, monthly oracle report, quarterly external output, per-subsystem two living docs (design + ops).
- AI cannot modify: codegen IR schema, oracle golden trajectories, threshold constants, license, CLA, this plan.

**Initial recommended risk register:**

| Risk | Probability | Impact | Reserve |
|---|---|---|---|
| Featherstone ↔ XPBD coupling instability | Medium | S2 blocks | Semi-implicit Featherstone + soft-contact spring fallback |
| Self-written sparse solver delays | High | v0.5 delay | **Real, accepted risk** (no cuDSS reserve): scope v0.5 to the minimal deterministic CG + Jacobi for IFT, defer MINRES/ILU/GMRES/AMG to v0.7, keep diff-sim episodes short, validate against V3 FD + a dense reference solve |
| Codegen IR over-engineering | High | v0.1 delay | Strict: "v0.1 IR supports only existing 4 row classes" |
| Diff-sim long-episode convergence | Medium | v0.5 partial fail | Limit episode length + tunable checkpoint interval + per-row grad skip |
| D1 strong determinism perf deficit | Medium | v0.3 gate fail | D2 weak mode as training fallback; D1 stays for oracle/debug |
| OpenUSD SDK integration blocks | Low | S2 entry delay | Keep .usda text path; defer binary to S3 |
| Single H100 cannot fit S3 envelope | Medium | v1.5 demo shrink | v1.0 end memory stress test; trim demo if overflow |
| **Loss of motivation / burnout** | **High** | **Project stall** | **Mandatory quarterly external output forces reflection.** |

**User responses (in order):**

1. **No calendar-bound timeline because of AI workflow.** Calendar dates removed.
2. **Phase skipping not allowed.** Strict rule confirmed.
3. **Stagnation triggers accepted.** Reformulated as effort-based instead of time-based: "no progress for 4 weeks → review; 8 weeks → kill switch."
4. **D1 → D2 RL-training fallback accepted.** Oracle / debug / diff-sim remain D1.
5. **Self-written solver:** initial decision was no fallback (strict self-write); a Round 13 resolution temporarily admitted a closed-SDK reserve. **Amended 2026-05-31 back to strict self-write: the self-written deterministic solver is now IN v0.5 (CG + Jacobi for IFT); no closed-SDK reserve.** The v0.5 schedule risk this re-introduces is **real and accepted** — mitigated by the minimal CG+Jacobi scope (MINRES/ILU/GMRES/AMG deferred to v0.7) + short episodes + FD/dense-reference validation, **not** eliminated.
6. **No additional risks identified.**

**Pushback on "no calendar time":** Strong AI workflow argument has merit (calendar doesn't reflect AI-accelerated production rate) but the *rhythm* (weekly / monthly / quarterly) and *anti-scope-drift constraints* (no skipping, no "fix later") must remain. Calendar dates dropped; relative rhythm + exit criteria kept; stagnation triggers measured in elapsed-weeks-without-progress (not calendar weeks of total project time). All accepted by user.

**Final decisions:**

- **No calendar deadlines.** Phases complete when exit criteria pass.
- **Rhythm mandatory:** weekly primary objective, monthly oracle report, quarterly external output (blog / preprint / talk). Non-negotiable; this is the project's heartbeat and burnout protection.
- **Stagnation triggers retained, measured in weeks without milestone progress.**
- **No phase skipping. No "fix later."**
- **AI-protected file list (6 categories) accepted.**
- **physics-smell lint as CI hard fail accepted.**
- **Five-gate row-class merge requirement accepted.**
- **Risk register as documented; strict self-write resolution applied (amended 2026-05-31: no closed-SDK reserve; the self-written solver ships in v0.5 and the residual schedule risk is accepted, not eliminated).**

---

## 4. Universal Row IR Specification

The single format every constraint compiles to. Codegen consumes this; AI agents author IR but never the generated CUDA kernels.

```
Row {
    row_class_id          uint32   // dispatches to evaluator + adjoint
    body_count            uint32   // CSR variable neighborhood
    body_list_offset      uint32   // index into CSR body table
    jacobian_offset       uint32   // index into CSR Jacobian table
                                   //   maximal: 6-vec (linear+angular) per body
                                   //   Featherstone: scalar dq derivative along chain
    rhs                   float    // constraint residual
    lambda                float    // accumulated multiplier (warm start + tape entry)
    lower                 float    // lower bound
    upper                 float    // upper bound
    compliance_alpha      float    // XPBD compliance; 0 → hard constraint (PGS)
    damping_beta          float    // Baumgarte / XPBD damping
    flags                 uint16   // bitfield: Equality | Unilateral | Friction | Coupled | GradActive
    adjoint_kernel_id     uint16   // diff-sim reverse evaluator id
    gradient_mode         enum     // dense_adjoint | stop_grad_on_event | ift_at_convergence | none
    recompute_mode        enum     // tape | checkpoint | recompute_always
    event_flag_field      uint8    // bit position; set when discontinuous event fires (stop-grad)
    contact_softness      enum     // hard | soft_alpha
}
```

**Initial row class catalog (v0.1):**
- `MaximalContactRow`
- `MaximalJointRow`
- `MaximalDriveRow`
- `FeatherstoneContactRow`

**Catalog extensions (added as needed, explicit IR-extension review required):**
- `SparseSDFContactRow` (K2 rigid↔particle)
- `FeatherstoneSDFContactRow` (K2 articulated-link↔particle)
- `XPBDDistanceRow`
- `XPBDBendRow`
- `XPBDVolumeRow`
- `XPBDShapeMatchRow`
- `TrianglePointContactRow` (cloth zero-thickness)
- `ParticleParticleContactRow` (soft-soft, soft-fluid coupling)
- *PBF density rows are internal to fluid subsystem; not emitted into the universal scheduler.*

**Hard rule:** v0.1 IR supports only the four base classes. Adding a row class requires explicit IR extension review. No speculative IR design.

---

## 5. Workflow Hard Constraints

### 5.1 Time and pacing

- **No calendar deadlines.** Versions complete when their exit criteria pass.
- **Rhythm is mandatory:**
  - Weekly: one primary functional objective.
  - Monthly: oracle + invariant report generated automatically; reviewed by owner.
  - Quarterly: external output (blog post / arXiv preprint / internal talk). Non-negotiable — burnout protection and architectural reflection.
- **Stagnation triggers:**
  - 4 weeks without milestone progress → architecture review.
  - 8 weeks without progress → kill switch (cut demo, drop subsystem, roll back architecture).

### 5.2 Scope discipline

- **No phase skipping.** v0.1 exit criteria must all pass before v0.3 work begins.
- **No "remember to fix later."** Exit criteria are satisfied at the time the phase closes, or the phase is not closed.
- **No speculative architecture.** v0.1 IR supports v0.1 row classes only.

### 5.3 Per-row-class quality gate (5 gates)

Every new row class must pass all five before merge:

1. **Forward oracle** – matches reference engine within tolerance (see §6.V1).
2. **Adjoint vs finite-difference** – relative error < 1e-3.
3. **Determinism (D1)** – two identical runs produce bit-exact output.
4. **Energy invariant** – under conservative external work, energy drift bounded.
5. **Demo non-regression** – existing demos still pass.

### 5.4 AI agent boundary

AI agents may freely write, edit, and refactor most code. They **must not** modify the following files except by explicit owner request:

1. This master plan.
2. Row IR schema files (`tools/codegen/schema/*.yaml`).
3. Oracle golden-trajectory files (`tests/oracle/golden/`, Git LFS).
4. Numeric threshold constants (tolerances, residual bounds, time-step caps).
5. License files (`LICENSE`, `NOTICE`, `CLA.md`).
6. Generated CUDA kernels (`*.cu` files marked `// GENERATED – DO NOT EDIT`).

### 5.5 Physics-smell lint (CI hard fail)

Banned patterns in physics-path code:

- `atomicAdd<float>` or any float atomic in row updates (breaks D1).
- `cudaMallocAsync` / `cudaMalloc` in hot path.
- `__shared__` memory shared across row classes within a single dispatch.
- Exceptions thrown across the C ABI boundary.
- STL container types in public headers.
- **CPU simulation in production paths** (see §5.6 enforcement).

Enforced by `tools/lint/physics_smell.py` in CI.

### 5.6 GPU-Only Simulation (project-wide hard constraint)

**All physics simulation runs entirely on the GPU. No CPU physics simulation path is permitted in the production code path.**

This constraint is non-negotiable. It exists because:
- D1 strong determinism is defined relative to the GPU execution model; mixing CPU+GPU paths creates ulp drift that breaks reproducibility.
- The throughput targets (4096+ envs sub-ms) are unreachable on CPU; supporting a CPU step path bifurcates engineering effort with no payoff.
- The differentiable simulation tape lives in GPU memory; CPU step would require GPU↔CPU sync per step, killing diff-sim throughput.
- Cross-system coupling (rigid ↔ soft ↔ fluid) lives in shared GPU memory; CPU detour breaks the coupling row scheduler.

**What this allows (CPU is fine here):**

- One-time scene import (URDF / MJCF / USD parsing) and cook (SDF generation, BVH build, articulation cooking) before the simulation loop starts.
- Host-side orchestration: CUDA kernel launches, stream management, memory transfer scheduling, C ABI handle marshalling.
- C ABI inputs / outputs: caller hands in stream, receives device buffer views, optionally reads back state via explicit `nuka_world_get_buffer_view` calls.
- **V1 oracle validation only**: calling external engines (MuJoCo MJX, Pinocchio, Bullet, Houdini Vellum, NVIDIA Flex paper code) from CPU test harnesses for trajectory comparison — these run as separate processes / libraries, not as in-engine simulation paths.

**What this forbids:**

- Any per-step CPU simulation routine inside the engine.
- Any `phi::PhysicsBackend::CpuReference` invocation from a production C ABI handle, from Python bindings, from RL training, or from the differentiable simulation paths.
- Any "fall back to CPU when CUDA unavailable" branch inside the step loop. CUDA is a hard requirement; missing CUDA = engine refuses to construct a `Device` handle.
- Any CPU residency of constraint rows, Featherstone state, contact manifolds, particles, or SDF query results during the simulation step loop. All such state must be in CUDA device memory between cook completion and engine destruction.
- Any GPU→CPU readback inside the per-step hot path. Post-step diagnostic readback (V2 invariant sampling, sensor output marshalling) is the only allowed readback, and even then it is opt-in via configuration.

**Enforcement:**

1. `phi::PlatformContract::CpuReference` remains as a validation-only flag. The C ABI `nuka_device_create` rejects any descriptor that requests a CPU backend in production (returns `NUKA_RESULT_NOT_SUPPORTED`). A separate `tests/reference/` build path is allowed to invoke CPU code paths for cross-validation only and is not linked into `libnuka.so`.
2. `tools/lint/physics_smell.py` includes a `production_cpu_sim` pattern that flags any reference to `CpuReference` outside of `tests/oracle/**`, `tests/reference/**`, `tools/oracle/**`, and files matching `*reference*.cpp`.
3. Per-PR review must confirm no new CPU simulation entry point is introduced.
4. Any phase spec that appears to require CPU simulation work is treated as a master-plan amendment proposal — owner-only edit, surfaces the contradiction explicitly.

This constraint propagates into every v0.1 / v0.3 / v0.5 / v0.7+ phase spec via a "🔒 HARD CONSTRAINT" callout at the top of each file referencing this section.

---

## 6. Validation Architecture

### V5 – AI Agent Guardrails (week 1)

Codegen IR locked. Generated kernels marked DO-NOT-EDIT. physics-smell lint in CI. Diff-test bridge between old PGS and new CSR row paths.

### V2 – Invariant Monitoring (month 1)

Per-step sampling of: total energy, linear/angular momentum, constraint residuals, joint range bounds, link length stability, particle count conservation, NaN/Inf, velocity/position envelope. Lightweight `core/diagnostics/`. Dashboard.

### V1 – Oracle Regression (months 1–6)

Per subsystem, as Round 11 table. Golden trajectories in Git LFS under `tests/oracle/golden/`.

### V3 – Adjoint vs Finite-Difference Check (CI per row class)

Every codegen invocation produces forward + adjoint kernels; CI runs FD check on adjoint with representative perturbations.

### V4 – Per-Phase Demo Suite (per S phase)

Per Round 11 table.

---

## 7. Phase Exit Criteria (the constitution)

Phases complete when **all** exit criteria pass. No calendar dates. No partial credit. No "fix later."

### v0.1 – Foundation Refactor
- PHI Stream refactored to non-owning view; PlatformContract de-singletonized.
- C ABI v0.1 working.
- CSR Universal Row format landed; Contact / Joint / Drive paths migrated through new scheduler via diff-test bridge.
- Island / graph-coloring scheduler operational.
- V5 + V2 validation infrastructure operational.
- Featherstone ABA forward dynamics CUDA-resident; advances Go2 correctly.
- Codegen pipeline produces forward kernels for the four base row classes.
- **Demo: Go2 stand 5 s, MJX oracle within tolerance.**

### v0.3 – S1 Sprint
- Step time < 1 ms per env-step on RTX 4090.
- Energy drift < 2% over 1000 steps.
- 4096-env Go2 PPO converges to stable walking gait.
- PyTorch `autograd.Function` skeleton wired (diff-sim not active).
- V1 oracle for Featherstone fully passing.
- **Demo: 4096-env Go2 locomotion video.**

### v0.5 – Open Source + Diff-Sim
- GitHub repo public under Apache 2.0 + CLA.
- Diff-sim end-to-end through rigid + Featherstone.
- `torch.autograd.Function` adjoint FD check passing for all base row classes.
- JAX `custom_vjp` operational.
- Sim-to-real noise N1 + N2.
- Self-written deterministic sparse linear solver (CG + Jacobi/Block-Jacobi, fixed-order reductions, D1 bit-exact) integrated for IFT — no cuDSS / no closed-source SDK.
- **Demo: gradient-based system identification on Go2.**

### v0.7 – S2 Entry
- XPBD soft / cloth rows operational; oracle vs Vellum/Flex passing.
- PBF fluid + internal density rows operational; oracle vs Flex paper passing.
- Sparse SDF cooker + Newton contact + analytical adjoint working.
- V-HACD convex decomposition in cooker.
- SAP → LBVH broadphase upgrade complete; particle uniform grid; cross-system query.
- RGB + depth + tactile + force/torque sensors operational.
- **Demo: H1 grasp cup + place on table.**

### v1.0 – S2 Polish
- K2 + K3 cross-system coupling working for rigid↔soft, rigid↔fluid, soft↔fluid.
- Sim-to-real noise N3 key items.
- Self-written CUDA RT pipeline operational (LBVH + traversal + intersection + shading).
- Vulkan ↔ CUDA external memory interop (Windows priority).
- **Demo: H1 pour water + wring towel.**

### v1.5 – S3 Entry
- 64 robots + 500K particles + 100K rigids stable on H100.
- 80 GB memory envelope holds.
- D3D12 ↔ CUDA external memory interop.
- **Demo: warehouse scene playable.**

### v2.0 – S5 Entry
- 4096-env RGB + semantic + depth pipeline.
- Differentiable rendering prototype.
- **Demo: VLA training data generation prototype.**

### v3.0 – Sim-to-Real (open scope)
- Real Go2 and H1 hardware deployment.
- Quantified sim-to-real metrics (gap analysis, transfer rate).

---

## 8. Risk Register

| Risk | Probability | Impact | Reserve / mitigation |
|---|---|---|---|
| Featherstone ↔ XPBD coupling instability | Medium | S2 blocks | Semi-implicit Featherstone + soft-contact spring fallback |
| Self-written sparse solver delays | High | Diff-sim / v0.5 delay | **Real, accepted (no cuDSS reserve):** v0.5 scoped to minimal deterministic CG + Jacobi for IFT; MINRES/ILU/GMRES/AMG deferred to v0.7; short diff-sim episodes; gradients validated vs V3 FD (<1e-3 rel) + a dense reference solve (Eigen LDLT, <1e-6) |
| Codegen IR over-engineering | High | v0.1 delay | Hard rule: v0.1 IR supports only existing 4 row classes |
| Diff-sim long-episode convergence | Medium | v0.5 partial fail | Limit episode length; tunable checkpoint; per-row grad skip |
| D1 strong determinism misses S1 perf bar | Medium | v0.3 gate fail | D2 weak mode for training; D1 for oracle/debug/diff-sim |
| OpenUSD SDK integration blocks | Low | S2 entry delay | Keep .usda text path; defer binary if needed |
| Single H100 cannot fit S3 envelope | Medium | v1.5 demo shrink | v1.0 memory stress test; trim demo if overflow |
| **Loss of motivation / burnout** | **High** | **Project stall** | **Mandatory quarterly external output.** Private v0.1–v0.3 acceptable. |

Burnout is the single largest threat. Quarterly external output is the only protection and is not optional.

---

## 9. Engineering Task Dependency Graph

```
                   [V5: AI agent guardrails]                  ← W1
                            |
                            v
                   [Row IR YAML schema]                       ← W2-3
                            |
                            v
         [Codegen pipeline skeleton (YAML→CUDA stub)]         ← W4-6
                            |
        +-------------------+-------------------+
        |                   |                   |
        v                   v                   v
[CSR row rewrite       [Featherstone     [PHI Stream caller-
 + diff-test bridge]    ABA forward       owned + PlatformCtx
                        dynamics]         de-singleton]
        |                   |                   |
        +-------------------+-------------------+
                            |
                            v
        [Island/coloring scheduler]
                            |
                            v
           ============ v0.1 EXIT ============
                            |
                            v
                  [S1 sprint: 4096-env Go2]
                            |
                            v
           ============ v0.3 EXIT ============
                            |
                            v
   [self-written CG/Jacobi sparse solver]
   [Sim-to-real N1+N2 sensors]
   [PyTorch autograd full + JAX]
                            |
                            v
           ============ v0.5 EXIT (public) ====
                            |
        +-------------------+-------------------+
        v                   v                   v
[Sparse SDF cooker    [Broadphase           [XPBD + PBF row
 + V-HACD]             LBVH+grid]            class catalog]
        |                   |                   |
        +-------------------+-------------------+
                            |
                            v
   [Self-written CUDA RT pipeline]
   [extend self-written solver: MINRES/ILU/GMRES/AMG]
                            |
                            v
           ============ v0.7 EXIT ============
                            |
                            v
   [Vulkan ↔ CUDA external memory]
   [Cross-system K2+K3 coupling rows]
                            |
                            v
           ============ v1.0 EXIT ============
                            |
                            v
   [D3D12 interop, S3 demo, envelope stress]
                            |
                            v
           ============ v1.5 EXIT ============
                            |
                            v
   [S5 RGB pipeline + diff rendering]
                            |
                            v
           ============ v2.0 EXIT ============
```

---

## 10. First-Week Concrete Plan

**W1 primary objective: V5 AI Agent Guardrails operational.**

Deliverables:
1. `tools/lint/physics_smell.py` – scans `*.cu` files in physics path; blocks merge on banned patterns. CI-integrated.
2. `tools/codegen/README.md` – documents codegen entry point, IR location, regeneration command, DO-NOT-EDIT convention.
3. `tools/codegen/schema/row_v0_1.yaml` – first cut of Row IR schema (structure only; populated W2).
4. This master plan committed under `docs/plans/`.

**W2–3 primary objective: Row IR for the four base classes.**

Author IR YAML for `MaximalContactRow`, `MaximalJointRow`, `MaximalDriveRow`, `FeatherstoneContactRow`. Each entry declares: body_count semantics, Jacobian layout, friction/equality flags, adjoint kernel reference, default gradient mode, compliance support.

**W4–6 primary objective: Codegen skeleton.**

YAML → CUDA forward stub (no adjoint yet). Must build. Stub kernel matches existing PGS contact behavior on a trivial scene. Adjoint generator deferred until V3 infrastructure is ready.

**W7+ primary objective: Migrate existing CUDA solver paths to codegen-produced kernels via diff-test bridge.**

Run old kernel and new kernel side-by-side. Tolerance must hold. When all four base classes pass diff-test, retire the old kernels.

---

## 11. Per-Subsystem Documentation Discipline

Every subsystem maintains two living documents under `docs/architecture/`:

- `*-design.md` – why this subsystem exists, what invariants it preserves, what the boundaries are, what the known limitations are.
- `*-ops.md` – how to debug it, how to extend it, known issues, last review date.

The owner reviews the task list monthly and deletes / merges stale entries. Task list growth beyond ~40 active items is an early warning of scope drift.

---

## 12. Out of Scope

Explicitly not covered by this plan:

- Specific algorithm-level choices inside a subsystem (e.g., which constitutive model for soft body): decided when that subsystem's design doc is authored.
- Programming-language style guidelines: existing `agent.md` rules apply.
- Build / CI infrastructure details: existing scripts evolve as needed.
- Pricing / monetization / community policy: deferred to post-v0.5.
- Platform support for macOS / AMD / Intel: deferred behind PHI; not v1 scope.

---

## 13. Amendment Process

Changes to this plan require:

1. Owner-only edit (not AI agent).
2. New row in the owner edit log at the top of this file with date and one-sentence rationale.
3. If the change invalidates an in-flight exit criterion, the affected phase resets to in-progress.

This plan is the project constitution. It can change, but changes are visible and intentional, not silent drift.
