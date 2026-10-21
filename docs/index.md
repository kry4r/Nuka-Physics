# Nuka Physics

**GPU-resident CUDA physics for robot learning, with an editor to build, run and inspect it.**
One block-descent solver drives articulated robots, rigid bodies, VBD cloth and MLS-MPM solids, across
thousands of environments on one GPU.

![The Nuka editor with an imported Unitree G1](media/editor/interface.webp)

This manual covers both halves of Nuka:

- **The engine**, used from Python or C: cook a scene into a GPU world, step thousands of environments,
  read and write their state through DLPack, render, train.
- **The editor**, a desktop app over the same engine: import robots and environments, pose them, simulate
  and push them around, train locomotion policies and watch them, record sensors and datasets, look inside
  the solver, and let scripts and AI agents do all of it.

## Where to start

<div class="grid cards" markdown>

- **[Build the engine](getting-started.md)**: the Linux source build, a first batched step from
  PyTorch, a first authored scene.
- **[Install the editor](editor/install.md)**: build and launch the editor on Windows, or use the
  packaged app.
- **[A first session](editor/first-session.md)**: import a humanoid, pose it, simulate it, train it to
  walk and play the policy, in about fifteen minutes.
- **[Editor manual](editor/index.md)**: every panel, tool and command, one topic per page.
- **[Actions](reference/actions/index.md)**: the full list of the editor's 300-odd actions, the same
  commands its menus, scripts and agents run.
- **[Agents and MCP](editor/agents.md)**: connect Claude or another MCP client to a running editor.

</div>

## What the editor does

| Task | Where |
| --- | --- |
| Open, save, package and recover scenes | [Documents and files](editor/documents.md) |
| Import URDF, MJCF, USD and meshes, with their materials | [Importing assets](editor/import.md) |
| Add primitives, move them, edit physics fields, generate colliders, lay terrain | [Building scenes](editor/scene.md) |
| Create PBR materials and assign them per part | [Materials](editor/materials.md) |
| Navigate, split the view, shade, path-trace, record video | [Viewport and camera](editor/viewport.md) |
| Play, step, reset, change the time step, clone environments, push bodies | [Simulation](editor/simulation.md) |
| Joints, drives, poses, IK, gain tuning, system identification, assembly | [Robots](editor/robots.md) |
| Teleoperate, run policies, record demonstrations, key joint clips | [Teleop, policies, demos and clips](editor/motion.md) |
| Cameras, lidars, IMUs, labels, randomization, dataset export | [Sensors and datasets](editor/sensors.md) |
| Configure a task, train it, watch the run, play its checkpoints | [Training](editor/training.md) |
| Contacts, colliders, solver residuals, the timeline, Rerun | [Inspecting physics](editor/inspect.md) |
| Python inside the editor | [Scripting](editor/scripting.md) |
| External programs and AI agents | [Agents and MCP](editor/agents.md) |
| New panels, actions, overlays and teleop devices | [Extensions](editor/extensions.md) |

## Conventions

- Units are SI: metres, kilograms, seconds, newtons, radians (the editor shows degrees where it says so).
- The world is **Z-up**. Quaternions are written `[w, x, y, z]`.
- Keys are written ++ctrl+s++; on the editor's menus they show beside each command.
- *Action* ids such as `file.import` name editor commands. The [action reference](reference/actions/index.md)
  lists every one with its arguments.

## License

Nuka Physics is available under AGPL-3.0 or a commercial license; see the
[repository](https://github.com/kry4r/Nuka-Physics).
