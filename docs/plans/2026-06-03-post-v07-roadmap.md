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
| **R6 Closed kinematic loops** | parallel/linkage robots (the Kamino gap) | maximal-coord + Proximal-ADMM (or Lagrangian loop-closure) — a NEW solver alongside Featherstone | R | maximal-coord contact, solver suite |
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

## 3. Sequencing (owner-set, 2026-06-03 — REVISED for the version restructure)
**v0.7 — CLOSED.** p12 ✅ → p13 ✅ → p14b ✅ → G1 TLAS/BLAS fast tracer ✅(#28) → p15 ✅ → asset pipeline ✅ (#19 mesh loader, #31 SceneIR compose, #32 USD mesh). The H1 grasp gate moves to v0.8/C7.

**v0.8 — engine completeness (the collision spine FIRST, then physics + render-beauty + usability):**
1. **Unified collision/contact/coupling subsystem** (C1–C7, see [[v08-unified-collision-contact]]) — the SPINE. Closes the grasp demo (C7) as its first validation consumer. Pulls in former-v1.0 engine items where adjacent: GPU SDF cook backend (#21, behind p07 seam), SDF reverse-mode (#23, p08 debt), coupling wire-in (#27), AMG (#24, at its large-sparse-stiff consumer).
2. **Physics completeness, value-ordered** (W1, each consumes the new contact spine): IK (R1, cheap, demo-useful) → cable/DER (R2, leapfrogs Newton) → garment cloth (R3, needs the CCD seam) → MPM (R4) → FEM (R5) → closed-loops/Kamino SolverKamino-ADMM (R6, plugs into the solver registry) → tendon/muscle (R7); coupling (R8) trails each via the coupling matrix.
3. **Render beauty + speed** (W2 G2→G4 + G3): photoreal A1 path (G2) + benchmark/OptiX-trigger (G4) + semantic/normal AOVs (G3).
4. **Usability** (W3 U1→U2→U3 + U4 real-complex-USD), once the fast render exists.

**v1.0 — DEMOS ONLY.** A few showcase demos on the v0.8 engine: H1 grasp-place, fluid/soft/coupling scenes, all photoreal-RT-rendered (A1 tier) on the GitHub homepage ([[v10-exit-conditions-expansion]], [[demo-homepage-readme-directives]]). No new engine work — purely demos + polish.

## 4. Stays deferred (per sim2sim focus)
Sim-to-real noise N1/N2/N3 (geometric/physical noise, beam divergence, rolling shutter, lens distortion, dark-current, ISP) and v3.0 real-hardware deployment + transfer metrics. Kept on the long-term map; not current priorities.
