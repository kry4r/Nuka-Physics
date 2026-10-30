# Python: the engine

The `nuka` package is a nanobind binding over the engine's C ABI. A `World` keeps its state on the GPU,
and its buffers reach PyTorch (or JAX, or anything that speaks DLPack) without a copy. This page covers
the core API; [the editor](editor.md) has its own page, and [Getting started](../getting-started.md)
builds and installs the package.

## The packages

| Module | Holds |
| --- | --- |
| `nuka` | `Device`, `World`, `Field`, `Scene`, `SceneBuilder`, `Recorder`, `sync()`, `torch_stream_ptr()` |
| `nuka.author` | The authoring layer: `morphs` (geometry), `materials`, `surfaces`, `SimOptions` and `Scene` |
| `nuka.assets` | Writers for baked tetrahedral meshes and MPM particle sets |
| `nuka.editor` | The editor from Python: launch it, script it, extend it ([Python: the editor](editor.md)) |
| `nuka.mcp` | The editor as an MCP server for AI agents ([Agents](../editor/agents.md)) |
| `nuka.train` | The training tasks the editor configures, and the run process (`python -m nuka.train job.json`) |
| `nuka.gym`, `nuka.rl_games` | A vectorized Gymnasium env over a `World`, and its rl_games 1.6.5 registration |
| `nuka.tasks` | RL task definitions and on-GPU observation builders |
| `nuka.isaaclab_compat` | A drop-in for the RL subset of Isaac Lab's manager-based API ([Isaac Lab compatibility](../concepts/isaaclab-compat.md)) |
| `nuka.energy`, `nuka.diagnostics` | Energy records over the physics intervals, and physics evidence and audits |
| `nuka.noise` | Observation errors and per-episode domain randomization |
| `nuka.jax_state_pytree` | A JAX pytree over the state arrays, loaded on first use |

`import nuka` never imports torch; `torch_stream_ptr()` loads it when called.

## Baked continuum media

Asset writers return absolute `.nka#TETS/0` or `.nka#PSET/0` references. Arrays keep their
order and use metres, kilograms and seconds. A tetrahedral mesh can carry its initial and
rest shapes separately:

```python
import numpy as np
from nuka.assets import write_tet_mesh, write_particle_set
from nuka.author import Scene, SimOptions, morphs, materials

q = np.array([[0, 0, 1], [.1, 0, 1], [0, .1, 1], [0, 0, 1.1]], dtype=np.float32)
asset = write_tet_mesh("solid.nka", q, [[0, 1, 2, 3]],
                       rest_positions=q, inverse_masses=[0, 1, 1, 1])
scene = Scene(SimOptions(dt=1/240, gravity=(0, 0, -9.81)))
scene.add_entity(morphs.TetMesh(asset), materials.Soft.VBD(youngs=1e5, poisson=.3))
```

Use `velocities`, `rest_inverse` and `material_ids` for additional tetrahedral streams.
`morphs.TetMesh(materials=(...))` supplies rows indexed by per-tet IDs; contact settings
come from the entity's material. `morphs.TetSphere` also accepts `materials.Soft.VBD`.

MPM particle sets require explicit inverse masses and rest volumes:

```python
asset = write_particle_set("points.nka", q, inverse_masses=np.full(4, 1000),
                           rest_volumes=np.full(4, 1e-6), velocities=np.zeros((4, 3)))
scene.add_entity(morphs.ParticleSet(asset, particle_radius=.005),
                 materials.Soft.MPM(dx=.02, substeps=20))
```

Optional `elastic_deformation`, `affine_velocity`, `plastic_deformation` and
`equivalent_plastic_strain` preserve continuum state through creation and reset. Matrices
have shape `(N,3,3)` and row-major entries. Plastic histories require `Soft.ElastoPlastic`
materials. `morphs.ParticleSet(materials=(...))` provides the per-particle material table;
`grid_origin` and `grid_dimensions` optionally set the domain. Default zero dimensions choose
a padded domain; moving media need enough domain headroom for their full trajectory.
`Field.PARTICLE_AFFINE_VELOCITY`, `PARTICLE_REST_VOLUME` and `PARTICLE_MATERIAL_ID` expose
these arrays alongside `PARTICLE_DEFORMATION_GRADIENT` for state inspection. The affine
velocity has nine float32 entries per particle in s⁻¹, rest volume is float32 in m³, and
material IDs are uint32 indices into the world's merged material table.

For lower-level authoring, `SceneBuilder.set_media_asset`, `set_tet_materials`,
`set_mpm_materials` and `set_mpm_grid` use the same cook path. A builder's `save(path)` packages
the binary chunks beside its `.nks`. See the [stream layout](../reference/file-formats.md#scenes).

## A world

```python
import torch
import nuka

with nuka.Device.create(0) as device:
    world = nuka.World.create_from_scene(device, "examples/scenes/go2_locomotion.usda",
                                         env_count=4096, dt=1.0 / 200.0)
    q = torch.from_dlpack(world.buffer_view(nuka.JOINT_POSITION))       # (4096, 13), read only
    target = torch.from_dlpack(world.buffer_view(nuka.DRIVE_TARGET))    # (4096, 13), writable
    target[:, 1:] = q[:, 1:]
    world.step_n(100)
    nuka.sync()
    world.destroy()
```

`World.create_from_scene(device, scene_path, env_count, dt=1/240, ...)` cooks a scene file into
`env_count` copies. Other keywords pick the control mode, the procedural terrain, the solver budget and
gravity; each left at zero keeps the engine's default.

| Member | Does |
| --- | --- |
| `step()`, `step_n(n)` | Advances one or `n` fixed steps |
| `reset()` | Every environment back to its initial physical state; control targets and views stay |
| `env_count`, `base_link_count`, `dt` | The layout: slots per environment and the step |
| `buffer_view(field)` | A zero-copy CUDA array over a state buffer (below) |
| `view(field)` | The same as a `FieldView` that raises `StaleViewError` once the world was rebuilt |
| `capture_checkpoint()` / `restore_checkpoint(c)` | Snapshot and rewind every persistent byte, the step counter and noise state |
| `read_checkpoint(path)` | Reads a checkpoint written with `checkpoint.write(path)` |
| `state_hash()` | A 64-bit hash of the state: equal hashes, identical worlds |
| `forward_kinematics()` | Re-poses links from joint state without stepping |
| `solve_ik(targets, root=None, env=0, ...)` | Damped least-squares IK; `write_targets=True` makes the answer the drive targets |
| `param_infos()`, `get_param(p, element)`, `set_param(p, element, value)` | Tunable parameters; `live` ones apply at the next step |
| `set_gravity_z(g)` | Uniform gravity along Z |
| `rebuild_from_scene(path)`, `rebuild_with_envs(path, n)` | Recooks in place, carrying body and joint state over by name |
| `export_scene(path)`, `export_usd(path)` | Writes the scene as NKS, or as UsdPhysics (returns what USD could not carry) |
| `destroy()` | Frees the world; `with World... as world:` does it for you |

Engine work is asynchronous. Call `nuka.sync()` before reading results on the host, or create the device
with `nuka.Device.create(0, stream_ptr=nuka.torch_stream_ptr())` so the engine runs on PyTorch's current
stream and needs no sync between them.

## State buffers

`buffer_view(field)` returns an array whose `__dlpack__` aliases the engine's own device buffer, so
`torch.from_dlpack` costs nothing and writes land in the engine. Indexing is environment-major: slot 0
of a joint field is the root, the actuated joints follow.

| Field | Shape | Writable | Holds |
| --- | --- | --- | --- |
| `JOINT_POSITION`, `JOINT_VELOCITY` | `(env, links)` | | q and qd |
| `ARTICULATION_LINK_POSE` | `(env, links, 7)` | | World pose `[px, py, pz, qw, qx, qy, qz]` |
| `LINK_VELOCITY` | `(env, links, 6)` | | `[wx, wy, wz, vx, vy, vz]`; the root slot is the base velocity in the root body's frame |
| `DRIVE_TARGET` | `(env, links)` | yes | PD targets; also `TORQUE_INPUT` in torque modes |
| `DRIVE_STIFFNESS`, `DRIVE_DAMPING` | `(env, links)` | yes | Per-joint kp and kd |
| `DRIVE_FORCE_LIMIT` | `(env, links)` | yes | Bound on the actuator's effort (N·m or N) |
| `VELOCITY_TARGET`, `ACCELERATION_TARGET` | `(env, links)` | yes | Inputs of the velocity and computed-torque modes |
| `TASK_TARGET`, `TASK_ROTATION_TARGET` | per articulation | yes | The OSC mode's goal |
| `LINK_CONTACT_WRENCH` | `(env, links, 6)` | | Net contact wrench per link, world frame |
| `ACTUATOR_EFFORT`, `ACTUATOR_SATURATED` | per joint | | What the actuators applied, and where they hit their limit |
| `BODY_FORCE`, `BODY_TORQUE` | per body | yes | World loads at the COM, used by the next step and cleared |
| `ENV_TERRAIN_TYPE`, `ENV_TERRAIN_DIFFICULTY` | per environment | yes | Procedural terrain kind and the scale of its features |
| `ENV_STATUS` | per environment | | Failure flags, such as a control input that was not finite |

Physical state and control fields use float32; indices, counts and status flags use integer types.
`nuka.Field` lists them all, including the solver and contact diagnostics. Downloads and DLPack views
retain each field's dtype.

Contact geometry can be captured through `DiagnosticSession(state_fields=...)` using
`UCONTACT_COUNT`, `UCONTACT_POINT`, `UCONTACT_NORMAL`, `UCONTACT_DEPTH`,
`UCONTACT_WITNESS_A`, `UCONTACT_WITNESS_B`, `UCONTACT_LAW` and `UCONTACT_FRICTION`.
Each slot reserves four points; only its first `UCONTACT_COUNT` entries are valid.
Normals point toward side A and depths are signed metres (positive for overlap). OGC witnesses are
world positions on each side's geometry; other detectors do not define these witness fields.
Law values are available as `nuka.ContactLaw`: `COMPLIANT` (`0`), `VELOCITY` (`1`) and
`SPECULATIVE` (`2`). Detector friction overrides are defined for OGC and MPM contacts; the solver
can instead obtain friction from a mixed or authored contact profile.
These records describe the geometry used to assemble the step's constraints, before integration.
Check contact sampling, force laws and time integration before judging a reference trajectory;
a scene's final-state check alone does not establish matching physical inputs.

## Control modes

`control_mode` at creation picks how joints are driven:

| Mode | Reads | Law |
| --- | --- | --- |
| `0` PD position | `DRIVE_TARGET`, `VELOCITY_TARGET` | `kp·(q* − q) + kd·(v* − v)` |
| `1` torque | `TORQUE_INPUT` | The effort itself |
| `2` velocity | `VELOCITY_TARGET` | `kp·(v* − v)` |
| `3` computed torque | position, velocity, `ACCELERATION_TARGET` | Inverse dynamics with feedforward |
| `4` OSC | `TASK_TARGET`, optional `TASK_ROTATION_TARGET` | Operational-space control of `osc_task_link` |
| `5` actuator | `TORQUE_INPUT`, `ACTUATOR_NOLOAD_SPEED` | Effort inside a torque-speed envelope |

Control inputs persist across resets. The full rules, units and the actuator envelope are in
[`python/README.md`](https://github.com/kry4r/Nuka-Physics/blob/master/python/README.md#control-modes).

## Authoring a scene

[Prescribed motion](prescribed-motion.md) provides named joint FK and COM twists
for independent kinematic boundaries, plus bounded cone mesh conversion.

`nuka.author` builds a world from parts instead of a file: geometry from `morphs`, physics from
`materials`, looks from `materials.Render` and `surfaces`, put together in a `Scene` whose
`build(device)` cooks one GPU world. [Getting started](../getting-started.md#author-a-demo-scene) has a
complete example. `morphs.NKS("scene.nks")` starts from a saved scene, so you can add cloth or soft bodies
to a robot authored in the editor.

## Training outside the editor

```bash
python examples/training/train_go2_ppo.py --num_actors 4096
python -m nuka.train <run folder>/job.json       # a run the editor configured
```

The first trains through `nuka.gym` and `nuka.rl_games`. The second is how the editor starts its own
runs, so you can rerun a job the [Training panel](../editor/training.md#run-folders) wrote from a shell.
