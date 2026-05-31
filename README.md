# Nuka Physics Engine

A high-performance, CUDA-first physics engine designed for robotics simulation,
reinforcement learning, and real-time applications.

## Showcase — 4096-env Go2 locomotion, trained in-engine

![Go2 walking — batched environments driven by an in-engine PPO policy](docs/media/go2_walk_4096env.gif)

*16 of 4096 batched Go2 environments (debug-skeleton render of the headless
batched path), driven by a PPO policy trained end-to-end on Nuka.* The same
cooked `WorldTemplate` runs all 4096 environments on one GPU; the policy was
trained from scratch with [rl_games](examples/training/) PPO and converges to a
command-conditioned walking gait (forward-velocity tracking across
`-0.5 … +1.0 m/s`).

### v0.3 (S1) highlights

- **4096 parallel articulated environments** on a single GPU from one cooked
  template — GPU-resident broadphase, contact generation, PGS contact/joint
  solve, Featherstone ABA, and PD drives.
- **Comfortably under the 1 ms / env-step target** at 4096 envs under strong
  (D1) determinism on a dev **RTX 4000 Ada** (≈3× below a 4090; the absolute
  master-plan gate is validated on the owner's RTX 5080 — this box reports
  relative numbers). Median GPU step time is ≈ 0.93 µs / env-step by in-engine
  CUDA-event timers. See
  [the on-box test & perf report](docs/architecture/2026-05-31-v03-test-and-perf-report.md).
- **Byte-exact strong determinism (D1)** by default — no float atomics, fixed
  reduction order, bit-identical re-runs — with an **opt-in weak-determinism
  (D2)** toggle exposed through the C ABI (`nuka_world_desc_t.determinism`) for
  workloads that trade reproducibility for speed.
- **Opt-in, bit-exact CUDA-graph step path** (`BatchedArticulatedWorld::StepGraph`)
  — replays the captured step kernel sequence **bit-for-bit identical** to the
  reference `Step()` (asserted by a byte-compare regression test), removing
  per-launch overhead while staying D1-safe. (At 4096-env production batches the
  step is compute-bound, so the launch-overhead win is within run-to-run noise;
  the value is the determinism-preserving mechanism, not a speedup.)
- **In-engine RL stack**: zero-copy PyTorch (DLPack) buffer views, a gymnasium
  vec-env, an rl_games adapter, and an engine-side per-env reset primitive — see
  [`examples/training/`](examples/training/).

## Features

- Rigid body dynamics with symplectic Euler integration
- Iterative constraint solver (PGS) for contacts and joints
- Articulated body support (revolute, prismatic, fixed joints) with PD drives
- Collision detection: broadphase (sweep-and-prune) and narrow-phase (GJK/SAT)
- Sensor simulation: CUDA IMU/state and lidar query kernels on GPU-resident
  scenes, with CPU sensor helpers kept as reference-only validation code
- Scene import from MJCF, URDF, and USDA/text USD formats with SceneGraph / PhysicsWorld / RenderScene conversion
- Isolated USD stage adapter with explicit `.usd`/`.usda`/`.usdc`/`.usdz` routing and an OpenUSD SDK backend boundary for binary USD/USDZ
- CUDA batched world state for parallel environments sharing one cooked
  `WorldTemplate`, including GPU-resident batched broadphase, contact
  generation, contact solve for plane/box/sphere contact scenes, cooked joint
  projection, velocity drive solve, and batched CUDA IMU/lidar observation
  queries; CPU batching remains reference/orchestration metadata
- CUDA is the preferred/default production physics backend through the PHI
  (Platform Hardware Interface) backend selection layer; high-level production
  APIs reject silent CPU fallback, and CPU stepping must be explicitly enabled
  as a validation/reference run
- Vulkan is the required production rendering backend; it now has executable
  offscreen paths for both physics debug commands and materialized `RenderScene`
  mesh instances, while the headless CPU PPM rasterizer remains a deterministic
  CI/reference artifact path
- Debug draw and visualization utilities for collision proxies, AABBs, joints, contacts, centers of mass, and constraint errors

## Quick Start

### Prerequisites

- CMake 3.20+
- C++20 compiler (MSVC, GCC, or Clang)
- CUDA Toolkit for the preferred production physics backend
- Vulkan SDK for the production rendering backend

### Configure

```powershell
# PowerShell
.\scripts\configure.ps1

# Or directly with CMake
cmake -S . -B build -DNK_BUILD_TESTS=ON

# This workstation defaults to the CUDA production path and native GPU kernels.
# Override only when intentionally cross-compiling or reproducing another GPU.
cmake -S . -B build -DNK_BUILD_TESTS=ON -DNK_CUDA_ARCHITECTURES=native
```

### Build

```powershell
cmake --build build --config Release
```

### Test

```powershell
# PowerShell
.\scripts\run-tests.ps1

# Or directly with CTest
ctest --test-dir build -C Release --output-on-failure

# CUDA production physics and Vulkan production rendering checks
ctest --test-dir build -C Release --output-on-failure -R "CudaWorldStepper|CudaDeviceWorld|CudaBatchedWorld|CudaContacts|CudaConstraintSolver|CudaSensor|CudaStepTiming|CudaBatchTiming|CudaBatchContactTiming|CudaBatchJointDriveTiming|CudaBatchSensorTiming|CudaContactTiming|CudaSolverTiming|CudaSensorTiming|CudaSceneDemoTiming|VulkanRenderer|VulkanSceneDemoTiming|BatchedVulkanSceneDemoTiming|VulkanRenderSceneTiming"
```

### Contributing / Lint

Physics-smell lint is part of the v0.1 guardrail layer:

```bash
python tools/lint/physics_smell.py
cmake --build build --target nuka_run_lint
```

See [Physics-Smell Lint](tools/lint/README.md) for the banned patterns, scoped
allowlist, and extension rules.

### Imported Scene Debug Render Demo

```powershell
cmake --build build --config Release --target nuka_scene_demo
.\build\src\Release\nuka_scene_demo.exe examples\scenes\complete_robot.xml out\complete_robot_debug.ppm 640 360 60 0.0166667
.\build\src\Release\nuka_scene_demo.exe examples\scenes\complete_robot.usda out\complete_robot_usd_debug.ppm 640 360 60 0.0166667
.\build\src\Release\nuka_scene_demo.exe examples\scenes\complete_robot.usda out\complete_robot_batched_debug.ppm 640 360 60 0.0166667 8
.\build\src\Release\nuka_scene_demo.exe examples\scenes\complete_robot.usda out\complete_robot_material.ppm 640 360 60 0.0166667 renderscene
```

The demo imports MJCF/URDF/USD text scenes, compiles them into
`SceneGraph`/`PhysicsWorld`/`RenderScene`, resolves physics through the PHI
backend selection layer, and on this workstation advances the scene through the
CUDA production path: device world upload, fixed-step integration, CUDA
broadphase/contact generation, and CUDA joint/contact/drive constraint solving.
Imported IMU/frame-pose sensors are sampled from that same CUDA `DeviceWorld`;
lidar depth queries use CUDA ray kernels against cooked sphere, box-style, and
plane geometry.
It then synchronizes the simulated body poses back into render/debug views,
generates physics debug overlays, renders those overlays through the Vulkan
offscreen compute renderer, and writes a deterministic PPM image for quick
validation or CI artifacts. The CLI prints physics backend, render backend,
CUDA constraint row counts, joint/drive/contact block counts, sensor sample
counts, and maximum position error so global demo runs prove more than file
output. The optional `renderscene` output mode keeps the same import and CUDA
simulation path but renders the synchronized material `RenderScene` mesh
instances through Vulkan instead of physics debug overlays.

CPU simulation is not a production fallback. To run the imported-scene demo
against the CPU reference stepper for differential validation, callers must set
both `physics_backend_policy = ForceCpuReference` and
`allow_cpu_reference_validation = true`; otherwise the app rejects CPU physics
selection.

When the optional final `instance_count` argument is greater than one, the CLI
runs the batched imported-scene workflow: one cooked template is reused across
multiple CUDA `BatchedDeviceWorld` instances, fixed-step contacts/joints/drives
and batched IMU observations run on CUDA, and the final multi-environment debug
view is rendered through Vulkan into a single PPM artifact.

## Architecture

The engine is organized into layered modules:

| Layer | Modules | Description |
|-------|---------|-------------|
| **Core** | `math`, `core` | Spatial algebra, vectors, quaternions, transforms |
| **Scene** | `scene`, `import` | Scene IR, cooker, SceneGraph pipeline, MJCF/URDF/USD importers |
| **Rendering** | `render` | RenderScene metadata, materials, cameras, lights, debug proxies, Vulkan offscreen debug and RenderScene rendering |
| **Runtime** | `runtime`, `rigid`, `articulation` | CUDA single/batched world containers, batched contact/joint/drive solve, integrator, joint drives |
| **Collision** | `collision` | Broadphase, narrow-phase, raycasting |
| **Constraints** | `constraint`, `solver` | Contact manifolds, constraint blocks, PGS solver |
| **Sensors** | `sensor` | CUDA single-world and batched IMU/state and lidar query paths, plus CPU reference helpers |
| **PHI** | `phi` | GPU abstraction layer (CUDA backend) |
| **Apps** | `apps`, `debug_draw` | Debug draw command buffers and compiled-scene visualization overlays |

For detailed architecture documentation see:
- [Runtime Overview](docs/architecture/runtime-overview.md)
- [Scene Compiler Pipeline](docs/architecture/scene-compiler.md)
- [Imported Scene Debug Render Demo](docs/architecture/debug-render-demo.md)

## Tech Stack

- **Language**: C++20
- **Build System**: CMake
- **Testing**: Google Test
- **Physics backend**: CUDA preferred by default via PHI backend selection;
  CPU reference is kept for validation/orchestration only and requires explicit
  opt-in at app/API boundaries
- **Rendering backend**: Vulkan production backend with offscreen debug and
  RenderScene material output for CI artifacts; CPU raster output is
  reference-only
- **CI**: GitHub Actions

## Project Structure

```
nuka-physics/
  CMakeLists.txt          # Root build file
  src/                    # Engine source code
    core/                 # Core utilities
    math/                 # Spatial math library
    scene/                # Scene IR and cooker
    import/               # Format importers
    runtime/              # World containers and integrator
    collision/            # Collision detection
    constraint/           # Constraint blocks
    solver/               # Iterative solver
    phi/                  # GPU abstraction
    sensor/               # Sensor simulation
    apps/                 # Debug draw and tools
  examples/               # Runnable imported-scene fixtures
  tests/                  # Test suite
    regression/           # Regression tests
    perf/                 # Performance benchmarks
  tools/                  # Utility scripts
  docs/                   # Documentation
  scripts/                # Build helper scripts
```

## License

TBD
