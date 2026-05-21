# Nuka Physics Engine

A high-performance, CUDA-first physics engine designed for robotics simulation,
reinforcement learning, and real-time applications.

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
  (Platform Hardware Interface) backend selection layer
- Vulkan is the required production rendering backend; the current headless PPM
  debug rasterizer remains a deterministic CI/reference artifact path
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
ctest --test-dir build -C Release --output-on-failure -R "CudaWorldStepper|CudaDeviceWorld|CudaBatchedWorld|CudaContacts|CudaConstraintSolver|CudaSensor|CudaStepTiming|CudaBatchTiming|CudaBatchContactTiming|CudaBatchJointDriveTiming|CudaBatchSensorTiming|CudaContactTiming|CudaSolverTiming|CudaSensorTiming|CudaSceneDemoTiming|VulkanRenderer|VulkanSceneDemoTiming"
```

### Imported Scene Debug Render Demo

```powershell
cmake --build build --config Release --target nuka_scene_demo
.\build\src\Release\nuka_scene_demo.exe examples\scenes\complete_robot.xml out\complete_robot_debug.ppm 640 360 60 0.0166667
.\build\src\Release\nuka_scene_demo.exe examples\scenes\complete_robot.usda out\complete_robot_usd_debug.ppm 640 360 60 0.0166667
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
output.

## Architecture

The engine is organized into layered modules:

| Layer | Modules | Description |
|-------|---------|-------------|
| **Core** | `math`, `core` | Spatial algebra, vectors, quaternions, transforms |
| **Scene** | `scene`, `import` | Scene IR, cooker, SceneGraph pipeline, MJCF/URDF/USD importers |
| **Rendering** | `render` | RenderScene metadata, materials, cameras, lights, debug proxies, Vulkan offscreen rendering |
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
  CPU reference is kept for validation/orchestration only
- **Rendering backend**: Vulkan production backend with offscreen debug-render
  output for CI artifacts; CPU raster output is reference-only
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
