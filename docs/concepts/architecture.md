# Architecture overview

This is the high-level map of Nuka Physics. For the runtime internals see
[Runtime overview](../architecture/runtime-overview.md); for the long-term
architecture and version roadmap see the
[master plan](../plans/2026-05-28-nuka-physics-master-plan.md).

## Design stance

- **GPU-resident, CUDA-only production backend.** Simulation state lives on the
  GPU and steps on the GPU. CUDA is the preferred and default production physics
  backend, selected through the PHI (Platform Hardware Interface) layer.
  High-level production APIs do **not** silently fall back to CPU; the CPU path
  exists only as a reference/validation stepper and must be explicitly opted into
  at the API boundary. There is no CPU production physics path.
- **Strong (D1) determinism by default.** No floating-point atomics, fixed
  reduction order in every physics path → bit-exact two-run reproducibility. A
  physics-smell lint enforces the no-float-atomics rule on every PR. An opt-in
  weak-determinism (D2) toggle is reserved at the C ABI for workloads that would
  trade reproducibility for speed.
- **Batched-by-template.** One cooked `WorldTemplate` is shared across thousands
  of environments (e.g. 4096 Go2) on a single GPU, with GPU-resident broadphase,
  contact generation, constraint solve, Featherstone ABA, and PD drives.
- **Differentiable by construction.** The forward physics has a paired analytical
  reverse-mode adjoint (see [diff-sim](diff-sim.md)).

## Layered modules

| Layer | Modules | Description |
|-------|---------|-------------|
| **Core / Math** | `core`, `math` | Spatial algebra, vectors, quaternions, transforms |
| **Scene** | `scene`, `import` | Scene IR, cooker, SceneGraph pipeline; MJCF / URDF / text-USD importers |
| **Runtime** | `runtime`, `rigid`, `articulation` | CUDA single/batched world containers, batched contact/joint/drive solve, integrator, Featherstone ABA, PD drives |
| **Collision** | `collision` | Broadphase (sweep-and-prune), narrow-phase (GJK/SAT), raycasting |
| **Constraints / Solver** | `constraint`, `solver` | CSR universal constraint rows, contact manifolds, PGS solver |
| **Diff-sim** | `diffsim` | Tape, gradient checkpointing, reverse-mode step adjoint, IFT contact gradient, self-written deterministic CG solver |
| **Sensors** | `sensor` | CUDA single-world + batched IMU / state / lidar query paths (CPU helpers are reference-only) |
| **PHI** | `phi` | GPU abstraction / backend selection layer (CUDA backend) |
| **Rendering** *(optional)* | `render`, `apps`, `debug_draw` | Vulkan offscreen renderer + debug-draw overlays. Not part of the shipped Python runtime |

## Scene import

Scenes import from three text formats:

- **MJCF** (`.xml`) and **URDF** — parsed via tinyxml2 (statically linked).
- **Text USD** (`.usda`) — parsed by a **hand-written, standard-library-only**
  parser.

There is **no OpenUSD SDK dependency**, and **no binary `.usdc` / `.usdz`**
support. The importers compile a scene into a SceneGraph / PhysicsWorld /
RenderScene representation, which the cooker turns into the immutable
`WorldTemplate` that batched simulation shares. See
[Scene compiler pipeline](../architecture/scene-compiler.md).

## The constraint-row spine

All constraints (Featherstone joint, contact, drive) compile to a single
CSR-like universal row format and run through a class-blind scheduler. The same
row format carries the adjoint kernel identity, so the reverse pass is a property
of the row catalog, not a parallel code path. This is the design hub that v0.7+
extends with XPBD soft, PBF fluid, and cross-system coupling rows — see the
master plan §3 (Rounds 3–7).

## What is and isn't here in v0.5

- **Shipped:** rigid + articulated dynamics, batched RL forward sim, full
  analytical adjoint through rigid + Featherstone, IFT contact gradient,
  tape + checkpointing, sim-to-real noise (N1 + N2), self-written deterministic
  sparse solver, PyTorch + JAX interop.
- **Roadmap (v0.7+):** soft body (XPBD), fluid (PBF), rigid↔soft↔fluid coupling,
  SDF cooker + Newton contact, LBVH broadphase, RGB/depth/tactile/F-T sensors.

## Embedding

Nuka is a C++ engine with a stable C ABI and nanobind Python bindings. It is
designed to embed directly in a C++ host (the engine library is `libnuka`),
rather than being a Python-only framework. The Python package is a thin
re-export layer over the `_nuka_ext` extension. See the master plan §3 Round 8
for the ABI / binding shape.
