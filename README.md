# Nuka Physics Engine

A high-performance, GPU-ready physics engine designed for robotics simulation,
reinforcement learning, and real-time applications.

## Features

- Rigid body dynamics with symplectic Euler integration
- Iterative constraint solver (PGS) for contacts and joints
- Articulated body support (revolute, prismatic, fixed joints) with PD drives
- Collision detection: broadphase (sweep-and-prune) and narrow-phase (GJK/SAT)
- Sensor simulation: IMU, lidar, force/torque
- Scene import from MJCF, URDF, and USDA/text USD formats with SceneGraph / PhysicsWorld / RenderScene conversion
- Batched simulation via WorldTemplate/Instance for parallel environments
- Optional GPU acceleration through the PHI (Platform Hardware Interface) layer
- Debug draw and visualization utilities

## Quick Start

### Prerequisites

- CMake 3.20+
- C++17 compiler (MSVC, GCC, or Clang)
- (Optional) CUDA Toolkit for GPU acceleration

### Configure

```powershell
# PowerShell
.\scripts\configure.ps1

# Or directly with CMake
cmake -S . -B build -DNK_BUILD_TESTS=ON
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
```

## Architecture

The engine is organized into layered modules:

| Layer | Modules | Description |
|-------|---------|-------------|
| **Core** | `math`, `core` | Spatial algebra, vectors, quaternions, transforms |
| **Scene** | `scene`, `import` | Scene IR, cooker, SceneGraph pipeline, MJCF/URDF/USD importers |
| **Rendering** | `render` | RenderScene metadata, materials, cameras, lights, debug proxies |
| **Runtime** | `runtime`, `rigid`, `articulation` | World containers, integrator, joint drives |
| **Collision** | `collision` | Broadphase, narrow-phase, raycasting |
| **Constraints** | `constraint`, `solver` | Contact manifolds, constraint blocks, PGS solver |
| **Sensors** | `sensor` | IMU, lidar, force/torque sensor simulation |
| **PHI** | `phi` | GPU abstraction layer (CUDA backend) |
| **Apps** | `apps`, `debug_draw` | Debug visualization, shell utilities |

For detailed architecture documentation see:
- [Runtime Overview](docs/architecture/runtime-overview.md)
- [Scene Compiler Pipeline](docs/architecture/scene-compiler.md)

## Tech Stack

- **Language**: C++17
- **Build System**: CMake
- **Testing**: Google Test
- **GPU**: CUDA (optional, via PHI abstraction)
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
  tests/                  # Test suite
    regression/           # Regression tests
    perf/                 # Performance benchmarks
  tools/                  # Utility scripts
  docs/                   # Documentation
  scripts/                # Build helper scripts
```

## License

TBD
