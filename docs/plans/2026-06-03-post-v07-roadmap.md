# Post-v0.7 Roadmap — Gap Analysis vs Newton / Isaac Lab

> **Status:** working roadmap (controller-authored 2026-06-03 from a source-backed Newton/Isaac research pass). NOT the protected master plan; a proposal for the owner to steer.
> **★ VERSION RESTRUCTURE (owner, 2026-06-03):** **v0.7 CLOSES NOW** (asset pipeline #19/#31/#32 + the shipped sim/RT subsystems; the H1 grasp gate is RE-HOMED to v0.8). **v0.8 = the engine-completeness release** = the **unified collision/contact/coupling subsystem** (see [[v08-unified-collision-contact]] / `docs/plans/2026-06-03-v08-unified-collision-contact.md`) + this roadmap's W1/W2/W3 engine work + the former-v1.0 ENGINE items pulled forward (GPU SDF cook backend #21, AMG #24, SDF reverse-mode #23, coupling wire-in #27). **v1.0 = a FEW DEMOS ONLY** (grasp, photoreal showcase, homepage) built on the v0.8 engine. The collision subsystem is the SPINE every W1 physics solver consumes — sequence it FIRST in v0.8.
> **Owner framing (2026-06-03):** sim2sim only — DEFER sim2real (N1/N2/N3 noise + v3.0 real-hardware). Priorities = **完备 (feature completeness)** + **美观 (photoreal beauty)** + **速度 (speed)** + **易用 (usability/frontend)**. RT stays self-written (preserve D1 + differentiability + no-closed-SDK pillars), build **TLAS/BLAS**, go FAST; OptiX only on a measured trigger. See memory [[sim2sim-focus-completeness-beauty]], [[rt-self-written-vs-optix-decision]], [[demo-homepage-readme-directives]], [[v10-exit-conditions-expansion]].

## 1. Scorecard — Nuka vs Newton vs Isaac Lab (verified 2026-06-03)

Newton = open-source GPU engine on Warp (NVIDIA+DeepMind+Disney, LF-governed, 1.0 @ GTC Mar-2026); 8 solvers; OpenUSD-native; diff via Warp autodiff; rendering = OpenGL/USD/Rerun viewers (**no photoreal RTX of its own** — defers to Isaac Sim). Isaac Lab = PhysX5 + RTX + full sensor/RL/teleop platform on Omniverse Kit.

| Capability | Nuka today | Newton | Isaac Lab |
|---|---|---|---|
| Articulated rigid (tree) | ✅ Featherstone ABA | ✅ Featherstone + MuJoCo | ✅ PhysX5 |
| **Closed kinematic loops** | ❌ tree-only | ✅ **SolverKamino** (beta) | ✅ PhysX |
| Soft/cloth (basic) | ✅ XPBD (4 rows) | ✅ XPBD/VBD | ✅ PBD/FEM |
| **Garment / high-precision cloth** | ⚠️ basic only | ✅ Style3D (PD) / VBD | ✅ PBD cloth |
| Fluid | ✅ **PBF (ahead)** | ❌ (Warp-side only) | ✅ PBD liquid |
| **MPM (granular/sand/snow)** | ❌ | ✅ SolverImplicitMPM | ✅ (via Newton) |
| **Cable / rope / DER** | ❌ | ⚠️ XPBD chains only (no true DER) | ⚠️ |
| **Volumetric FEM** | ❌ (XPBD only) | ⚠️ via warp.fem | ✅ PhysX FEM |
| Tendon / muscle / actuator | ❌ | ✅ via SolverMuJoCo | ✅ |
| Differentiable sim | ✅ **IFT adjoint (rigorous)** | ✅ Warp autodiff | ⚠️ (Newton path) |
| SDF / sparse-SDF contact | ✅ | ✅ (Warp geom) | ✅ |
| **Inverse Kinematics** | ❌ | ✅ newton.ik | ✅ diff-IK + Pink QP + cuRobo |
| Sensors: RGB/depth/lidar/contact | ✅ (RGB/depth WIP) | via Isaac | ✅ full |
| Sensors: **semantic/normal AOV** | ⚠️ normal/uv WIP, no semantic | via Isaac | ✅ |
| **Photoreal RT renderer** | 🔨 self-written CUDA RT (WIP) | ❌ (defers to Isaac) | ✅ RTX |
| **Real-time interactive viewport** | ❌ (offscreen only) | ⚠️ GL/Rerun | ✅ RTX viewport |
| **Authoring GUI / frontend** | ❌ | minimal | ✅ Omniverse Kit |
| Teleop / XR | ❌ | ❌ | ✅ VisionPro/CloudXR |
| USD/MJCF/URDF import | ✅ hand-rolled (tinyxml2) | ✅ OpenUSD | ✅ OpenUSD |
| Bindings | ✅ C-ABI + Python | ✅ Python | ✅ Python |

**Where Nuka is already AHEAD / differentiated:** dedicated PBF fluid (Newton ships none), rigorous IFT-at-convergence adjoint (Newton just uses Warp autodiff), and — once the render arm lands — a **self-written photoreal RT** which Newton itself does not have. **Where Nuka is BEHIND:** MPM, closed-loops (Kamino), garment cloth, cable/DER, volumetric FEM, IK, tendon/muscle, semantic AOV, and the entire real-time-interactive + frontend layer.

**"Kamino" (owner-flagged) = resolved:** `newton.solvers.SolverKamino` (Disney Research, arXiv:2504.19771) — a maximal-coordinate + **Proximal-ADMM** solver for **closed kinematic loops** (parallel manipulators/linkages), hard frictional contact + restitution. The gap is real: Nuka's reduced-coord Featherstone is tree-only and cannot close loops.

## 2. Three workstreams (decomposed, post-v0.7)

Effort key: S=small, M=medium, L=large, R=research. Each phase reuses the noted existing Nuka subsystem.

### W1 — Physics completeness (close the solver gaps)
| Phase | Goal | Approach | Effort | Reuses |
|---|---|---|---|---|
| **R1 IK** | per-frame inverse kinematics for the H1/manipulator demos | Jacobian (from Featherstone) → DLS / pseudo-inverse; later task-space QP (Pink-style) | S–M | Featherstone Jacobians |
| **R2 Cable / DER** | ropes, tend90s, cables (and beat Newton, which has no true DER) | Discrete Elastic Rods (Bergou) or XPBD-rod Cosserat (stretch+bend+twist) as new row classes | M | XPBD row scheduler + codegen |
| **R3 Garment cloth** | robot-dressing-grade cloth | strain-limited XPBD + robust self-collision/CCD + accurate bending; or projective-dynamics / VBD | M–L | XPBD cloth + particle grid + RT-BVH for CCD |
| **R4 MPM** | granular / sand / snow / elastoplastic | MLS-MPM + APIC transfer; implicit MPM for stiff; Drucker-Prager/von-Mises return map | L | particle grid, SDF, D1 reductions |
| **R5 Volumetric FEM** | constitutive soft-tissue / soft-robot | corotational / Neo-Hookean tet FEM, implicit (backward-Euler + self-written CG) | M–L | tet topology (p09), CG solver |
| **R6 Closed kinematic loops** ⏸ **DEFERRED (owner 2026-06-03: "kamino先不做")** | parallel/linkage robots (the Kamino gap) | maximal-coord + Proximal-ADMM (or Lagrangian loop-closure) — a NEW solver alongside Featherstone; plugs into the v0.8 solver registry when revived | R | maximal-coord contact, solver suite |
| **R7 Tendon / muscle / actuator** | MuJoCo-parity actuation | routed/fixed tendons (length=Σ joint coeffs), Hill-type muscle, transmission | M | articulation + constraint rows |
| **R8 Coupling breadth** | rigid↔MPM↔cloth↔fluid two-way | extend K2/K3 rows per new solver pair (MPM-rigid grid forces, cloth-fluid) | M (grows) | p11 coupling rows |

### W2 — Rendering: beauty + speed (self-written, fast)
| Phase | Goal | Approach | Effort | Reuses |
|---|---|---|---|---|
| **G0 (in flight)** p12/p13 | box ray tracer foundation → triangle/sphere/SDF intersect + GGX/Lambert shading + framebuffer (depth/normal/albedo/prim_id/uv) | self-written CUDA RT, D1, single-level LBVH | — | p04 LBVH |
| **G1 Fast 2-level tracer** | **TLAS/BLAS + go fast** | self-written two-level: BLAS per-mesh (refit deformables, rebuild on topo-change), TLAS over instance transforms; **wide BVH (BVH8)** + **ray sorting (Morton)** + **persistent-thread queue** + quantized nodes; all D1-preserving | M–L | G0 traversal (= BLAS), p04 build |
| **G2 Photoreal path (A1)** | beauty: PBR + shadows + AO/soft-GI + denoise + AA + tonemap | offline/progressive path tracer over the fast tracer; multi-sample + spatiotemporal denoise; deterministic seeded sampling | L | G1, sensor RT |
| **G3 Semantic/normal AOVs** | sim sensor parity (Isaac-grade) | extra G-buffer channels: instance/semantic IDs (cooker class-id → prim_id map), normals/uv already present | S–M | framebuffer, cooker |
| **G4 Render-perf benchmark + OptiX trigger check** | MEASURE MRays/s + seconds/frame on showcase scenes; report honestly; revisit OptiX-hybrid ONLY if it can't hit the bar | benchmark harness | S | G1/G2 |

### W3 — Usability / frontend (易用)
| Phase | Goal | Approach | Effort | Reuses |
|---|---|---|---|---|
| **U1 Real-time interactive viewport** | live orbit camera + play/pause/step + drag-to-perturb + debug overlays (contacts/forces/SDF) | use the **Vulkan rasterizer** (`src/render/`) for the 60fps viewport (NOT path tracing); object picking + gizmos | L | Vulkan renderer, C-ABI render-view |
| **U2 Scene authoring / inspection** | hierarchy + transforms + physics/material params + live tweak | panels over the C-ABI; param introspection | M | C-ABI, scene graph |
| **U3 Web / log viewer** | lightweight shareable viewer (lower-effort than a Kit clone) | Rerun-style log viewer or three.js/WebGPU web front page; Jupyter inline for notebooks | M | bindings, render-view |
| **U4 Real complex-USD asset pipeline** | import real complex USD scenes for demos (p16 requirement) | harden the hand-rolled importer or adopt a real USD path; mesh-geometry retention (#19) | M–L | importer, cooker |
| **U5 Teleop / XR** (later) | device input → robot retargeting | keyboard/spacemouse/VR abstraction | M | U1 |

## 3. Version structure & sequencing (owner-ratified via /grill-me, 2026-06-03)

Dependency-driven split: **foundation (v0.8) → breadth solvers (v0.9) → demos (v1.0)**. Cross-cutting policy (§6): full D1 everywhere; new breadth solvers forward-only (diff deferred).

**v0.7 — CLOSED.** p12 ✅ → p13 ✅ → p14b ✅ → G1 TLAS/BLAS fast tracer ✅(#28) → p15 ✅ → asset pipeline ✅ (#19 mesh loader, #31 SceneIR compose, #32 USD mesh). The H1 grasp gate moves to v0.8/C7 (validation, not gate).

**v0.8 — SPINE + cross-cutting foundations** (every breadth solver / demo depends on these):
1. **Unified collision/contact/coupling subsystem** (C1–C8, see [[v08-unified-collision-contact]]) — THE spine. Includes the **coupling-row framework + co-step bridge** as foundation (specific coupling pairs → v0.9). C7 grasp = first validation consumer. Pulls in adjacent former-v1.0 engine items: GPU SDF cook backend (#21), SDF reverse-mode (#23), coupling wire-in (#27), AMG (#24) at its consumer.
2. **R1 IK** — Featherstone-Jacobian DLS/pseudo-inverse; demo-enabling for manipulation.
3. **W2 render beauty+speed** — G2 photoreal A1 path + G3 semantic/normal AOVs + G4 perf-benchmark/OptiX-trigger (the "MEASURE" half; publish → v1.0).
4. **W3 U1** — real-time interactive Vulkan viewport.

**v0.9 — BREADTH solvers (plug into the spine) + advanced frontend:**
1. **W1 breadth**, value-ordered: R2 cable/DER → R3 garment-grade cloth (CCD seam) → R4 MPM (granular) → R5 volumetric FEM → R7 tendon/muscle. **R6 Kamino closed-loops ⏸ DEFERRED** (owner "先不做"; revived later into the solver registry). R8 coupling **per-pair** (rigid↔fluid, articulated↔MPM, cloth↔rigid …) trails each, via the v0.8 coupling framework.
2. **W3 advanced frontend**: U2 authoring/inspection → U3 web/log viewer → U4 real-complex-USD pipeline → U5 teleop/XR.

**v1.0 — DEMOS ONLY** + release-grade wrapper (§5). No new engine work. The **8 showcase demos** (all photoreal-RT A1 on the GitHub homepage, [[v10-exit-conditions-expansion]], [[demo-homepage-readme-directives]]):
1. **H1 dexterous grasp-and-place** (collision spine + R1 IK; real usdc cup via v0.8 C7a + full kitchen scene via v0.9 U4)
2. **Rigid-body collision showcase** (stack/dominoes/pile — general rigid contact + multi-point manifold)
3. **Fluid + rigid coupling** (pour water, objects pushed — PBF + rigid↔fluid R8)
4. **Garment cloth showcase** (drape/dressing with self-collision — **R3 garment-grade**, ACCD + cloth↔rigid/articulation R3d/R8c)
5. **MPM go2-on-sand** (granular walking — R4 MPM + articulated↔MPM coupling R8; multi-point SDF footing per OPEN-V4)
6. **Cable / rope** (R2 cable/DER)
7. **FEM volumetric soft-body** (squish/elastic recovery — **R5** corotational→stable-Neo-Hookean)
8. **Tendon / musculotendon actuation** (muscle-driven limb — **R7** Hill-type + fixed/spatial tendons, via-points; wrapping deferred)

> **Demo-set decision (owner 2026-06-04, advisor-flagged dedup):** R3 garment / R5 FEM / R7 tendon EACH get a demo. The former generic "soft/cloth showcase" (basic XPBD p09) is **upgraded + split**, not kept alongside: its cloth half becomes the garment-grade **Demo 4 (R3)** and its volumetric-soft half becomes **Demo 7 (R5)** — avoiding two visually-redundant soft demos. Tendon is the new **Demo 8 (R7)**. Net 6 → 8. The XPBD p09 path still exists in the engine (it backs the R3 mass-spring base + the v0.8 PBD co-step), it just no longer fronts a standalone demo.

## 5. v1.0 release-grade (non-demo) requirements — owner-approved 2026-06-03
1. **Packaging/dist (MUST)**: pip-installable Python wheel (extend `scripts/build_python_wheel.sh`) + prebuilt CUDA wheel; optional Docker image.
2. **Documentation (MUST)**: modernized README + C-ABI/Python API reference + getting-started + the 8 demo scenes as runnable examples + architecture overview.
3. **Performance baselines**: MEASURE in v0.9 (W2 G4 + sim steps/s batched + D1 two-run/cross-replica gates); **PUBLISH** the honest benchmark table (incl. Newton/Isaac comparables) in **v1.0**.
4. **License (MUST)**: maintain Apache-2.0; add third-party asset licenses (newton-assets / mujoco_menagerie) + NOTICE.
5. **API stability (v1.0)**: freeze the C-ABI + Python surface (semver guarantee) + deprecation policy — a 1.0-appropriate commitment.
6. **Release hygiene (MUST)**: CHANGELOG + GitHub release/tag + full CI green + all goldens regenerated/passing (incl. the C5 re-baselined standing golden).
7. **Homepage (MUST)**: GitHub homepage with RT-rendered demo videos + logo (kry4r/Nuka assets/logo.png).

## 6. Cross-cutting engineering policies (all versions)
- **Determinism**: FULL D1 on every new subsystem/solver (two-run bit-identity + N≥32 cross-replica identity). D2/weak = optional fast-path seam only. Non-negotiable pillar.
- **Differentiability**: rides only where it exists — the SDF contact tier (p08 adjoint) + the existing rigid/articulated IFT adjoint. New breadth solvers (cable/MPM/FEM/IK) are **forward-only**; diff = post-v1.0 research seam.
- **No closed SDK**: self-written throughout (no OptiX/cuBLAS/OpenUSD); OptiX revisited only on a measured G4 trigger.
- **Extensibility/maintainability** (Q9 directive): registration-based seams — new collidable types, narrowphase pairs, solvers (Kamino slot), coupling pairs, and a reserved CCD hook — so v0.9 breadth plugs in without touching the v0.8 spine.

## 4. Stays deferred (per sim2sim focus)
Sim-to-real noise N1/N2/N3 (geometric/physical noise, beam divergence, rolling shutter, lens distortion, dark-current, ISP) and v3.0 real-hardware deployment + transfer metrics. Kept on the long-term map; not current priorities.
