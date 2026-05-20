# Nuka Physics Engine Design

Date: 2026-04-26
Status: Approved for implementation planning

## 1. Project Intent

Build a self-written, CUDA-first physics engine and embodied-intelligence extension stack for robotics-focused simulation. The engine should support rigid bodies, robot articulations, collision, constraints, scene queries, basic sensor simulation, debug rendering, and a path toward large-scale offline synthetic data generation. The physics core must be written primarily in C++ and CUDA, with Python used as a secondary language for tooling and workflows.

This project is not an adapter around an existing physics runtime. External systems such as MuJoCo, PhysX, Isaac Lab, OpenUSD, and Kokkos are reference points for architecture, workflow, or data organization, but the core runtime, data model, and solver stack are self-authored.

## 2. Primary Constraints

1. The physics engine is self-written.
2. The execution model is CUDA first.
3. CPU is retained as orchestration, validation, tooling, and fallback infrastructure, but not as the primary performance target for P0.
4. The runtime must be architected for future multi-backend execution through a PHI layer, with CUDA first and future MUSA-class backends in mind.
5. The initial validation target is single-scene correctness, but the architecture must preserve a clean expansion path to batched world execution.
6. External asset formats are inputs to compilation, not runtime-native formats.
7. XML support is specifically MJCF support, not arbitrary XML.
8. USD authoring and scene organization should later align with Isaac Lab and Isaac Sim practices, without making USD the runtime format.

## 3. Product Scope

### P0 scope

P0 focuses on the engine foundation:

1. Canonical scene compilation pipeline.
2. CUDA-first physics runtime.
3. Rigid body simulation.
4. Robot articulation support through constraints.
5. Collision, contact generation, and scene queries.
6. Constraint solving.
7. Basic sensor simulation.
8. A lightweight debug visualization shell.

### Deferred from P0

1. Reduced-coordinate articulation fast path.
2. Full-featured editor.
3. High-fidelity offline path tracing inside the runtime.
4. Full soft body, cloth, and fluid solvers.
5. Multi-card distributed scheduling.

## 4. System Architecture

The top-level architecture is fixed as three layers:

1. Rendering and Tool Shell Layer
2. Embodied Runtime Layer
3. Physics Engine Layer

This is combined with two compilation chains:

1. Authoring chain: external scene formats to canonical IR
2. Execution chain: canonical IR to cooked runtime data and device state

### 4.1 Rendering and Tool Shell Layer

P0 uses Vulkan as the production rendering backend. The native shell should be
built around Vulkan rendering and may use Dear ImGui for controls, while the
headless PPM debug rasterizer remains a deterministic CI artifact path. The
shell is not the source of truth for physics. It consumes exported runtime
state, debug buffers, and sensor packets.

The debug shell should visualize:

1. Body frames
2. Joint frames
3. AABBs
4. Broadphase bins or BVH nodes
5. Contact manifolds
6. Constraint impulses
7. Sensor frusta and ray hits
8. Basic depth or segmentation previews

Future rendering paths can extend to Web viewers or Hydra-based rendering
integration, but those are layered on top of the same exported runtime state.

### 4.2 Embodied Runtime Layer

This layer hosts robotics-facing APIs and task logic:

1. Robot handles and control APIs
2. Sensor graph orchestration
3. Data collection and episode management
4. Batch environment management
5. Scene editing entry points

It communicates with the physics runtime through stable handles such as `SceneInstance`, `RobotHandle`, `SensorHandle`, and `QueryHandle`. It must treat the physics runtime as a compiled execution backend, not as a mutable object graph.

### 4.3 Physics Engine Layer

The physics engine layer is split into six major subsystems:

1. Importers
2. Scene compiler
3. Asset cookers
4. Runtime core
5. Simulation modules
6. PHI

The runtime core owns the authoritative execution model. Simulation modules plug into it through stable domain and interaction interfaces.

## 5. Canonical Scene Pipeline

External scene formats are supported through a three-stage scene representation:

1. Authoring Model
2. Canonical Scene IR
3. Cooked Runtime IR

### 5.1 Authoring Model

The authoring model preserves source semantics and debugging metadata. It contains:

1. Node graph and transform hierarchy
2. Physics semantics
3. Asset bindings
4. Source annotations

This layer is allowed to be human-readable and source-faithful. It is not required to be GPU-friendly.

### 5.2 Canonical Scene IR

The canonical IR is the engine's format-normalized representation. It is versioned, strongly typed, and independent from USD, URDF, or MJCF object models.

Core schemas:

1. Scene schema
2. Transform schema
3. Rigid schema
4. Collision schema
5. Joint schema
6. Robot schema
7. Sensor schema
8. Render schema
9. Extension schema

Future soft body, cloth, and fluid inputs should be carried through extension schemas so the compiler and runtime contracts are established before those solvers exist.

### 5.3 Cooked Runtime IR

Cooked runtime data is an execution-oriented binary form built from canonical IR:

1. Body tables
2. Shape tables
3. Joint tables
4. Collision proxy blobs
5. BVH blobs
6. Inertia blobs
7. Articulation graph blobs
8. Sensor graph descriptors
9. Debug and render proxy blobs

Cooked runtime data is designed for fast load and device upload. It does not need to be author-friendly.

### 5.4 Import Format Strategy

Supported external formats:

1. USD
2. URDF
3. MJCF

Import priority:

1. MJCF and URDF for early robotics validation
2. USD for authoring alignment and long-term pipeline organization

USD is not the canonical runtime format. Instead, its scene organization, sensor authoring patterns, and asset structuring can be used as upstream references, especially following Isaac Lab and Isaac Sim conventions.

## 6. Physics Runtime Design

The runtime must be designed as a multi-domain orchestrator from day one, even though P0 only implements rigid-body and articulation domains.

### 6.1 First-class runtime objects

1. `WorldTemplate`
2. `WorldInstance`
3. `DomainModule`
4. `InteractionModule`
5. `BatchContext`

#### WorldTemplate

Read-only topology and cooked resources:

1. Body tree
2. Joint graph
3. Shape ownership
4. Collision filters
5. Material tables
6. Sensor descriptors
7. Render and debug proxies
8. Future soft and fluid topology descriptors

#### WorldInstance

Mutable per-instance state:

1. Pose
2. Velocity
3. Force accumulators
4. Warm-start cache
5. Contact manifold cache
6. Sleep state
7. Random state
8. Sensor packet state
9. Debug counters

#### DomainModule

Each physical domain owns its state, preprocess logic, solver integration, and post-step handling. P0 includes:

1. RigidDomain
2. ArticulationDomain

Deferred domains:

1. SoftBodyDomain
2. ClothDomain
3. FluidDomain

#### InteractionModule

Cross-domain handling is a dedicated subsystem, not a side effect of rigid narrowphase. It is reserved for:

1. Rigid-rigid contact
2. Rigid-soft contact
3. Rigid-fluid coupling
4. Sensor interaction

#### BatchContext

BatchContext allows the architecture to treat `N = 1` as a special case of `N >= 1`, rather than hard-coding a single world assumption. P0 keeps single-scene validation, but API and memory layout must preserve the future batched-world model.

## 7. PHI: Physics Hardware Interface

PHI is the abstraction boundary for compute execution. It should not mimic a thin graphics RHI. Instead, it must expose the primitives required by simulation kernels and memory management.

Required PHI concepts:

1. Device
2. Queue or Stream
3. Fence and Event
4. Buffer and View
5. Arena and Scratch allocator
6. Kernel and Dispatch entry points
7. Graph launch support
8. Capability query

Capability query must expose backend-specific properties such as:

1. Subgroup or warp width
2. Atomic capabilities
3. Async copy support
4. Graph execution support
5. Sparse allocation support
6. Peer-to-peer support
7. Optional vendor library hooks

This abstraction is the basis for backend-aware execution policies. On this
workstation, the default production policy is CUDA. CPU execution is retained as
an explicitly selected reference/validation path and must not become the P0
performance target.

## 8. Data Layout and Memory Model

### 8.1 Hard constraints

1. Hot-path runtime data uses SoA layouts.
2. Topology and mutable state are split.
3. Persistent caches and per-step scratch storage are split.
4. Upper layers use stable handles while runtime storage remains free to reorder and compact.

### 8.2 Device residency

The simulation hot path is device-resident:

1. Body state
2. Broadphase state
3. Contact candidates
4. Contact manifolds
5. Constraint blocks and rows
6. Solver scratch
7. Query acceleration structures
8. Sensor execution buffers

Host-side ownership is limited to orchestration, compile products, selective debugging, validation, and external integration.

## 9. Simulation Pipeline

The frame loop is expressed as a staged execution graph:

1. Apply commands
2. Update kinematics
3. Broadphase
4. Narrowphase and CCD
5. Assemble constraints
6. Domain solve
7. Cross-domain coupling
8. Integrate, sleep, and emit events
9. Scene queries
10. Sensors
11. Debug export

P0 may leave cross-domain coupling effectively empty, but the execution slot must exist so future soft and fluid modules do not require a runtime rewrite.

## 10. Rigid Bodies, Articulations, and Constraints

### 10.1 Body model

P0 uses maximal-coordinate rigid bodies. Robot links are represented as rigid bodies plus joint constraints. This unifies:

1. Free rigid bodies
2. Robot links
3. Joint drives
4. Contact constraints
5. Future attachments and grasp constraints

Reduced-coordinate articulation can be added later as an optimization path, but it is not the P0 representation.

### 10.2 Constraint model

The first-class abstraction is `ConstraintBlock`, not raw scalar rows. Blocks contain:

1. Type tag
2. Involved body or domain handles
3. Row count
4. Jacobian build logic
5. Effective mass build logic
6. Limits, drives, and friction parameters
7. Warm-start cache
8. Solve policy tag

This supports:

1. Multi-row joints
2. Contact manifolds
3. Cross-domain couplings
4. Future block-level scheduling or coloring

### 10.3 Solver strategy

P0 uses a two-stage constraint solve:

1. Velocity solve via a block iterative solver in the PGS or TGS family
2. Position stabilization via bias or projection mechanisms

This choice is driven by:

1. GPU friendliness
2. Stable robotics contact handling
3. Warm-start support
4. Compatibility with a future solver family approach

Future solvers may include:

1. XPBD-style projector for soft or cloth domains
2. More implicit or high-robustness contact solvers for offline validation modes

## 11. Collision and Queries

### 11.1 Supported P0 shape proxies

1. Sphere
2. Capsule
3. Box
4. Plane
5. Convex hull
6. Static triangle mesh
7. Static heightfield

Dynamic arbitrary triangle-mesh rigid collision is explicitly excluded from P0. Dynamic complex meshes should be cooked into convex proxies or proxy sets.

### 11.2 Broadphase

Use a dual structure:

1. StaticSceneBVH for static world geometry
2. DynamicBodyBroadphase for dynamic colliders

Dynamic broadphase remains algorithm-swappable as long as the interface supports:

1. Build or update
2. Candidate pair generation
3. Pair compaction

### 11.3 Narrowphase

P0 should support:

1. Analytic primitive-primitive tests
2. Convex-convex narrowphase
3. Convex-static-mesh narrowphase
4. Shared geometry query kernels for raycast, overlap, and sweep

### 11.4 Contact representation

Contacts are stored as persistent manifolds:

1. Contact candidates
2. Contact patches
3. Contact manifolds with stable support points and friction frames

This is required for:

1. Warm start
2. Stable stacking
3. Better robot contact behavior
4. Tactile or contact sensor semantics

## 12. Sensor and Rendering Strategy

Sensors are a first-class runtime subsystem organized around a `SensorGraph`. Sensor types are represented as:

1. SensorTemplate
2. SensorInstance
3. SensorBackend

Sensor backends:

1. StateQueryBackend
2. RayQueryBackend
3. RenderBackend

P0 focuses on:

1. IMU-style state sensors
2. Joint and pose sensors
3. Contact summary sensors
4. Depth or lidar via ray queries

Rendering is split into three paths:

1. Debug viewport
2. Fast synthetic sensor path
3. High-fidelity offline rendering service

The debug viewport is native and interactive. The fast synthetic path is headless and CUDA oriented. High-fidelity rendering is planned as a separate service, not as a hard dependency of the physics runtime.

## 13. Scene Editing and Authoring

Scene editing belongs in the authoring and canonical layers. The editor does not directly mutate low-level device runtime buffers. Instead:

1. Authoring data is modified
2. Canonical IR is regenerated or patched
3. Runtime blobs are rebuilt or incrementally patched
4. Device state is refreshed through explicit rebuild or hot-patch APIs

This preserves deterministic asset compilation, undo-redo semantics, and clean dataset snapshotting.

## 14. Scaling Strategy

The long-term target includes large-scale concurrent simulation and sensor generation. The design supports this through:

1. BatchContext
2. Headless sensor backends
3. Decoupled data generation scheduling
4. World template reuse
5. Runtime packet outputs instead of view-specific APIs

A dedicated data-generation scheduler should eventually manage:

1. World step queue
2. Sensor update queue
3. Render job queue
4. Readback and encode queue
5. Dataset writer queue

## 15. Roadmap

### P0

1. PHI v1
2. Canonical scene compiler v1
3. Rigid runtime v1
4. Collision and query v1
5. Constraint system v1
6. Iterative rigid solver v1
7. Articulation v1
8. Sensor v1
9. Debug shell v1

### P1

1. Batched world execution
2. Robot control API and reset randomization
3. Broader camera and annotation support
4. Headless dataset pipeline
5. Better USD-based authoring flow
6. Initial editor workflows

### P2

1. First soft-body or cloth domain
2. First fluid domain
3. Cross-domain coupling
4. Higher-fidelity offline rendering service
5. Additional backend support through PHI

## 16. Validation Strategy

Validation is mandatory from P0 onward.

### 16.1 Test layers

1. Kernel and math unit tests
2. Closed-form scene tests
3. Differential reference checks against established simulators
4. Regression scene library
5. Performance and scaling tests

### 16.2 Key metrics

1. Constraint drift
2. Maximum penetration
3. Energy drift
4. Contact count stability
5. Sensor packet correctness
6. Per-stage GPU timings
7. Memory growth and fragmentation

## 17. Main Risks and Mitigations

### Risk: CUDA-first debugging complexity

Mitigation:

1. Host-side validation helpers
2. Selective readback tooling
3. Rich debug visualization
4. Replayable kernel inputs

### Risk: articulation and contact complexity

Mitigation:

1. Maximal-coordinate articulation in P0
2. Narrow joint-type scope in early milestones
3. Unified constraint block architecture

### Risk: importer complexity

Mitigation:

1. Importers feed one normalization path
2. MJCF and URDF land first
3. USD support starts with essential schema mapping

### Risk: soft and fluid scope collapsing rigid progress

Mitigation:

1. Reserve interfaces now
2. Do not ship those solvers in P0
3. Require domain and interaction boundaries before new solvers land

### Risk: high-fidelity rendering derailing physics progress

Mitigation:

1. Debug-first rendering shell
2. Headless fast sensors before beauty rendering
3. Offline renderer as a separate service boundary

## 18. Final Design Statement

This project will be implemented as a CUDA-first, self-written, multi-backend-ready physics engine centered on canonical scene compilation, a PHI abstraction, and a domain-oriented runtime. P0 establishes a rigid-body and articulation foundation with collision, constraints, scene queries, basic sensors, and debug visualization, while the architecture keeps explicit expansion points for batched execution, USD-centric authoring workflows, future MUSA support, soft-body and fluid domains, and high-concurrency offline sensor generation.

## 19. Reference Baselines

These references informed the architecture and should remain part of the research baseline during implementation:

1. MuJoCo programming and modeling guides: `https://mujoco.readthedocs.io/en/stable/programming/index.html`
2. MuJoCo computation guide: `https://mujoco.readthedocs.io/en/stable/computation/index.html`
3. OpenUSD UsdPhysics: `https://openusd.org/release/api/usd_physics_page_front.html`
4. OpenUSD Hydra docs: `https://openusd.org/24.08/api/_page__hydra__getting__started__guide.html`
5. PhysX GPU rigid bodies: `https://nvidia-omniverse.github.io/PhysX/physx/5.4.0/docs/GPURigidBodies.html`
6. PhysX constraint formulation: `https://nvidia-omniverse.github.io/PhysX/physx/5.4.0/_downloads/f27bad5e4b631dc274a41ecf77568a49/constraintFormulation.pdf`
7. Kokkos programming model: `https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/ProgrammingModel.html`
8. Kokkos execution spaces: `https://kokkos.org/kokkos-core-wiki/API/core/execution_spaces.html`
9. MUSA SDK docs: `https://docs.mthreads.com/en/musa-sdk/musa-sdk-doc-online/introduction/`
10. Isaac Lab sensor docs: `https://isaac-sim.github.io/IsaacLab/v2.1.1/source/overview/core-concepts/sensors/index.html`
11. XPBD paper: `https://matthias-research.github.io/pages/publications/XPBD.pdf`
12. MLS-MPM paper and project page: `https://yuanming.taichi.graphics/publication/2018-mlsmpm/`
13. IPC project: `https://ipc-sim.github.io/`
14. Rigid-IPC project: `https://ipc-sim.github.io/rigid-ipc/`
