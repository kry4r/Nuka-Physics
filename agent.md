# Agent Guidance

This repository is being built as a CUDA-first, full-GPU physics simulation platform for embodied robotics. Keep this file in mind before choosing work slices, validation strategy, or architecture direction.

## Current Mainline

- Prioritize CUDA backend solvers, GPU-resident data structures, robot rigid-body constraints, rigid-soft-fluid coupling, diagnostics, benchmarks, demos, and documentation.
- Preserve the API/backend selection layer, but CUDA is the default production physics path.
- Keep CPU code limited to import/cook, host orchestration, reference validation, or fallback scaffolding. Do not let CPU simulation become the main architecture.
- Do not expand Vulkan rendering unless it is directly needed for physics validation, debug visualization, or demo boundaries.
- Do not treat passing unit tests as completion. Report progress against GPU data residency, solver invariants, robot/coupling coverage, benchmarks, demos/docs, and remaining platform risks.
- Avoid framing work as a minimum loop or small demo. Choose coherent slices that move the platform toward full simulator capability.

## Workflow Preference

- Use design -> implementation -> integrated validation as the normal workflow for substantial physics work.
- Do not default to strict TDD or spend most of the session writing narrow unit tests.
- Prefer complete feature slices, reusable integration/runtime checks, scene-level validation, demos, benchmarks, and diagnostics.
- Unit tests can still be useful for stable low-level contracts, but they are secondary to integrated CUDA runtime evidence for this project.
- Avoid isolated trivial tests that do not exercise meaningful engine behavior or robot/world workflows.

## Robot Simulation Focus

- Primary robot targets: Unitree Go2 and Unitree H1.
- Use real robot assets and scene-level workflows where possible instead of synthetic toy fixtures.
- Go2 USD source:
  - https://github.com/unitreerobotics/unitree_model/tree/main/Go2/usd
- The Go2 USD directory should be treated as the first known source for Go2 robot USD import/cook/runtime work.
- Future work should search for Unitree H1 USD assets and useful scene USDs, then build scene-level CUDA validation around robot/world simulation.
- Robot validation should cover import/cook, articulated rigid structure, constraints/drives, contacts, solver diagnostics, GPU residency, and runtime performance.

## CUDA Constraint Scheduler Direction

- The current particle/rigid coupling work is not the end state.
- Move toward a reusable CUDA constraint-row scheduler that can serve rigid contacts, joints, drives, robot articulation, cloth/deformable, particle fluid, and rigid-soft/fluid coupling.
- Existing gaps to keep visible:
  - coupling rows are still too close to fixed per-particle slots
  - shared rigid writes still rely on atomics
  - no island/coloring scheduler yet
  - no XPBD/compliance path yet
  - robot articulation, cloth/deformable, particle fluid, and coupled scenes are not unified under one scheduler yet
- Do not add empty abstractions. Every abstraction should connect to a runnable CUDA scenario.

## Workstation Build And Verification

- Build through the Visual Studio 2022 `vcvars64` environment and the existing `C:\Softwares\code\nuka-physics\build` tree unless newer repo evidence replaces this.
- Avoid parallel MSBuild in the shared build tree; use `/m:1`.
- Known build pattern:

```cmd
cmd.exe /d /s /c ""C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul && cmake --build C:\Softwares\code\nuka-physics\build --config Release --target <target> -- /m:1 /nodeReuse:false /p:TrackFileAccess=false /p:CudaToolkitDir="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\\" /v:m"
```

- Treat intermittent `Permission denied` around generated intermediates or googletest outputs as likely shared-build concurrency fallout first.
- CRLF warnings may appear in `git diff --check`; avoid unrelated line-ending churn.
- Do not commit untracked `.claude/` files.

## Git Workflow

- After every commit, push that commit immediately.
- If no remote is configured, report that as the blocker instead of leaving the commit silently local.
