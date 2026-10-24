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

## Cloth attachments

An attachment keeps a cloth vertex's finite mass and adds the potential
`0.5 * stiffness * |position - target|²`. Stiffness is in N/m. Both `Cloth.VBD`
and `Cloth.Field` use the same implicit vertex solve for these springs.

```python
from nuka.author import Scene, SimOptions, materials, morphs, surfaces

scene = Scene(SimOptions(dt=1 / 240, gravity=(0, 0, -9.81)))
scene.add_entity(
    morphs.Grid(11, 11, 0.01, origin=(0, 0, 0.5), attachments=(
        morphs.ClothAttachment(vertex=0, stiffness=100.0),
        morphs.ClothAttachment(vertex=10, stiffness=100.0),
    )),
    materials.Cloth.Field(density=150, thickness=0.0001, youngs=2e7),
    surfaces.Cloth(free=True),
)
world = scene.build(device)
targets = world.download_field(nuka.PARTICLE_KINEMATIC_TARGET).reshape(-1, 3).copy()
targets[[0, 10], 2] += 0.001
world.upload_field(nuka.PARTICLE_KINEMATIC_TARGET, targets)
world.step()
```

Targets initially equal the cooked positions. `PARTICLE_ATTACHMENT_STIFFNESS`
exposes each vertex's stiffness; zero disables its spring. Values must be finite
and nonnegative. A prescribed target is an external actuator: its work and the
spring's stored energy are included in the energy ledger. It does not apply a
reaction to a robot whose pose the caller uses to construct the target.

The lower-level builder provides `add_cloth_attachment(media_id, vertex, stiffness)`.
NKS stores the same records under `cloth_mesh.attachments`, as objects with
`vertex` and `stiffness` members. Attachments cannot be cooked on pinned vertices.
Fracture divides an attachment's stiffness with its vertex mass; descendants inherit
the target. Use `PARTICLE_ROOT` to update descendant targets when driving fractured cloth.
Reset restores the initial attachment stiffness and targets for the selected environments.

`Cloth.Field` also accepts `shear_strength` in Pa for the mixed traction criterion
at existing crack tips, and `weibull_reference_area` in m² for the tensile-strength
specimen's gauge area. Area scaling applies to nucleation; the local cohesive peak
and crack-tip threshold retain the material strength. Zero uses the tensile strength
for shear or disables area scaling, respectively. These parameters are also available in an NKS `field` record.
`tear_energy` remains the cohesive separation energy in J/m².
Existing crack tips use local mixed traction to choose their next edge; cohesive
traction and separation work remain active on every newly split edge. New cracks
require tensile loading averaged over the material's process zone.

`fracture_law="mixed_mode"` (fracture contract version 2) gives the split edge its own
sliding response. Opening along the crack normal peaks at `tear_strength` and spends
`tear_energy`; sliding peaks at `shear_strength` and spends `shear_tear_energy`. Damage
starts on (Tn/σ)² + (Ts/τ)² = 1. Toughness follows G_I + (G_S − G_I) b^`mixity_exponent`
at the sliding share b of the elastic energy. Damage is irreversible across changes of
mode, and compression is undamaged until the edge parts. The crack normal is taken from
the faces at the start of each step. A relative rotation of the two faces still loads the
bending hinge, not the cohesive opening through the thickness. Zero `shear_tear_energy`
uses `tear_energy`, and zero `mixity_exponent` uses 1. The state layout lists
`softening_knee`, `inserted`, `sliding_energy_fraction` and `crack_normal` per seam.

`MF_CORNER` contains 12 uint32 slots per element and environment. Triangle elements
use the first three as their live vertices. Bend elements store their own triangle
followed by each adjacent triangle, so opening a crack keeps both faces' geometry.
Use `PARTICLE_SURFACE_CORNERS` for the live render and collision triangle indices.

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

Every field is float32 except `CONTACT_LINK` (uint32). `nuka.Field` lists them all, including the solver
and contact diagnostics.

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

`nuka.author` builds a world from parts instead of a file: geometry from `morphs`, physics from
`materials`, looks from `materials.Render` and `surfaces`, put together in a `Scene` whose
`build(device)` cooks one GPU world. [Getting started](../getting-started.md#author-a-demo-scene) has a
complete example. `morphs.NKS("scene.nks")` starts from a saved scene, so you can add cloth or soft bodies
to a robot authored in the editor.

`world.step_report(env=0)` returns the last public step's outcome, status flags, state version,
topology epoch, simulated time, and executed iteration count for each internal substep. It also
reports the requested and effective iteration budgets and transaction memory. A step waits for
backend completion. A zero requested budget means automatic selection. A failure rolls back the entire batch,
including material history, topology, controls, and state sensor noise, queues, and time; its flags
remain available in the report.
`physical_convergence_measured` distinguishes these lifecycle checks from full physical acceptance.
Checkpoint restore, reset, host field uploads, gravity or link parameter changes, and explicit
forward kinematics invalidate diagnostics from the previous state.
After writing through a device view, call `world.invalidate_diagnostics()` before reading reports.
Pending body force and torque are persistent inputs, so checkpoints retain them. Checkpoint files
with a different persistent layout are rejected; NKS scenes migrate through their versioned material contract.

`FieldMaterial` uses contract schema 1 and rejects unknown versions. Material version 1 is the
synthetic small-strain adapter: plane stress J2 plasticity and two Maxwell branches with a 5% surface
strain range. Material version 2, `Cloth.Field(law="log_strain_orthotropic", ...)`, measures membrane
strain as the Lagrangian log strain (1/2) ln C in the grain frame, with orthotropic plane stress
(`youngs`, `poisson` along `grain`, `youngs_cd`, `shear_modulus`), Hill yield (`yield_stress`,
`yield_stress_cd`, `yield_shear`, `yield_biaxial`) and the same Maxwell branches, valid through
`strain_range` in principal log strain; plates bend with the orthotropic section stiffness Q t³/12.
NKS files record material, section, fracture, and history versions. The existing cohesive law does
not yet provide verified mixed-mode or thickness opening behavior.

`world.material_state_layouts()` describes the flat, environment-major
`nuka.MATERIAL_STATE` readout. Each element has a schema version, state offset and
count, plus named fields with SI units. State schema 2 uses 10 float32 values for
plane stress, 13 for a plate with hinge damage, and 3 for the existing cohesive
law. Offsets refer to one environment's span; multiply the environment index by
`state_scalars_per_env` to locate that environment. Trial spans append two
per-element dissipation values and remain internal scratch.

Known version 1 parameter descriptions migrate to the current state allocation
when cooked. This does not migrate version 1 checkpoints; their history offsets
and persistent layout differ, so restore rejects them. The C equivalents are
`nuka_world_get_material_state_layout`, `nuka_material_state_field_info`, and
`NUKA_FIELD_MATERIAL_STATE`.

## Training outside the editor

```bash
python examples/training/train_go2_ppo.py --num_actors 4096
python -m nuka.train <run folder>/job.json       # a run the editor configured
```

The first trains through `nuka.gym` and `nuka.rl_games`. The second is how the editor starts its own
runs, so you can rerun a job the [Training panel](../editor/training.md#run-folders) wrote from a shell.
