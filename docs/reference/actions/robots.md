<!-- Generated from assets/editor/actions.toml by tools/wiki_actions.py; edit those instead. -->

# Robots: joints, poses, gains, identification, IK

33 actions. Run one from the command palette (++ctrl+p++), from Python with `app.actions.run(id, **args)` or through MCP with `action_run`.

## `robot.joints`

**Kind:** [read](index.md#kinds)

The moving joints of the scene's robots with their limits and drives, and the shown environment's live values; the roots and whether the base floats.

**Returns:** {joints: \[{name, link, path, type, lower, upper, drive, kp, force\_limit, damping, q, qd, target, effort, saturated}\], roots, floating, env, watch}

## `robot.set_targets`

**Kind:** [sim](index.md#kinds)

Sets drive targets for the shown environment by joint name, in rad or m; the next step reads them. Not an edit of the document: robot.pose\_\* and the initial\_position field change what a reset starts from.

**Arguments**

- `targets`: object, joint name to target in rad (revolute) or m (prismatic)

**Returns:** {set}: how many targets were written

**Example**

```json
{"id": "robot.set_targets", "args": {"targets": {"FL_thigh_joint": 0.9, "FR_thigh_joint": 0.9}}}
```

## `robot.reset_targets`

**Kind:** [sim](index.md#kinds)

Puts every joint's drive target back at its initial position.

**Returns:** {set}

## `robot.radians`

**Kind:** [view](index.md#kinds)

Revolute joints read in radians while on, else degrees, in the Joints panel; without `on` it toggles.

**Arguments**

- `on`: bool, radians, default: toggles

**Returns:** {on}

## `robot.watch`

**Kind:** [sim](index.md#kinds)

While on, every step samples the shown environment's joints for robot.samples and the Joints panel's curves; without `on` it toggles.

**Arguments**

- `on`: bool, sample, default: toggles

**Returns:** {on}

## `robot.samples`

**Kind:** [read](index.md#kinds)

The newest `count` samples of one joint while robot.watch is on, oldest first; `total` counts the watch's samples.

**Arguments**

- `joint`: string, the joint's name
- `count`: int, how many of the newest samples, default all kept

**Returns:** {joint, count, total, watch, tick, t, q, target, effort}: parallel lists

## `robot.drive`

**Kind:** [edit](index.md#kinds)

One undoable edit gives `joints` (all by default) a drive of `type`; a position drive takes kp.

**Arguments**

- `type`: string, the drive type: position, motor, velocity or force, default "position"
- `kp`: number, a position drive's stiffness, default the Joints panel's kp
- `joints`: list of strings, joint names, default every joint

**Returns:** {focus}

## `robot.fix_base`

**Kind:** [edit](index.md#kinds)

Static roots are fixed bases: one undoable edit makes every robot root static (fixed to the world) or free; without `on` it toggles.

**Arguments**

- `on`: bool, fix the base, default: toggles

**Returns:** {focus}

## `robot.poses`

**Kind:** [read](index.md#kinds)

The joint poses kept with the document, each a name and its joints' positions (rad or m).

**Returns:** a list of {name, joints: {joint: position}}

## `robot.pose_capture`

**Kind:** [edit](index.md#kinds)

Keeps the shown environment's joint positions as pose `name`, replacing a pose of that name; without one it is "Pose N".

**Arguments**

- `name`: string, the pose's name, default "Pose N"

**Returns:** {name, joints: {joint: position}}

## `robot.pose_apply`

**Kind:** [sim](index.md#kinds)

Blends the shown environment's drive targets to pose `name` over `duration` simulated seconds, 0 at once; joints the pose lacks keep their targets.

**Arguments**

- `name`: string, the pose
- `duration`: float, seconds of simulated time the blend takes, default 0

**Returns:** {name, set, missing, unknown, steps}: joints set, the robot's joints the pose lacks, the pose's joints the robot lacks

**Example**

```json
{"id": "robot.pose_apply", "args": {"name": "stand", "duration": 1.0}}
```

## `robot.pose_initial`

**Kind:** [edit](index.md#kinds)

Writes pose `name` into its joints' initial positions as one undoable edit; sim.reset starts there.

**Arguments**

- `name`: string, the pose

**Returns:** {focus}

## `robot.pose_rename`

**Kind:** [edit](index.md#kinds)

Renames a pose.

**Arguments**

- `name`: string, the pose
- `to`: string, its new name, not taken

**Returns:** {name}

## `robot.pose_remove`

**Kind:** [edit](index.md#kinds)

Removes a pose.

**Arguments**

- `name`: string, the pose

**Returns:** {count}: the poses left

## `robot.poses_save`

**Kind:** [file](index.md#kinds)

Writes the poses to a JSON file as {"poses": \[{name, joints}\]}; without `path` a file dialog asks.

**Arguments**

- `path`: string, the .json file, default: a file dialog

**Returns:** {path, count}

## `robot.poses_load`

**Kind:** [file](index.md#kinds)

Adds the poses of a JSON file written by robot.poses\_save, replacing those of the same name; without `path` a file dialog asks.

**Arguments**

- `path`: string, the .json file, default: a file dialog

**Returns:** {path, read, count}

## `robot.gain_test`

**Kind:** [sim](index.md#kinds)

Drives one joint's target through a step or a sine from where it stands, in a copy of the world the test throws away, and fits rise, overshoot, settling, damping ratio and more. Unset arguments keep the Gain Tuner's; the joint needs a position drive. One test runs at a time; robot.gain\_status reports it.

**Arguments**

- `joint`: string, the joint, default the tuner's, else the focused one
- `kind`: string, step or sine, default the tuner's
- `amplitude`: float, the step or sine amplitude in rad or m, nonzero, default the tuner's
- `frequency`: float, a sine's frequency in Hz, default the tuner's
- `duration`: float, seconds simulated, in (0, 10\], default the tuner's

**Returns:** {joint, kind, steps}

**Example**

```json
{"id": "robot.gain_test", "args": {"joint": "FL_calf_joint", "kind": "step", "amplitude": 0.2, "duration": 1.0}}
```

## `robot.gain_status`

**Kind:** [read](index.md#kinds)

Whether a gain test runs, how many runs the tuner keeps and the newest one's figures.

**Returns:** {running, runs, last: {joint, kind, amplitude, frequency, duration, steps, kp, damping, force\_limit, rise, overshoot, peak\_time, settling, steady\_error, zeta, omega, gain\_db, lag, rmse, saturation} or null}

## `robot.gain_runs`

**Kind:** [read](index.md#kinds)

The gain test runs the tuner keeps, oldest first; `trace` adds each one's samples.

**Arguments**

- `trace`: bool, add time, target, q, qd and effort per step, default false

**Returns:** a list of runs, as robot.gain\_status's last

## `robot.gain_clear`

**Kind:** [view](index.md#kinds)

Forgets the gain test runs the tuner keeps.

**Returns:** {count}: 0

## `robot.gain_tuner`

**Kind:** [view](index.md#kinds)

Opens or closes the Gain Tuner window; `on` sets it, else it toggles.

**Arguments**

- `on`: bool, open it, default: toggles

**Returns:** {on}

## `sysid.run`

**Kind:** [sim](index.md#kinds)

Identifies joint damping, armature, friction loss or a link mass scale from a demonstration episode: CMA-ES searches each quantity between its min and max, and every candidate replays the episode's drive targets from its first step in a copy of the world, scored by the mean squared error against the joint positions it recorded. Candidates run one at a time because the registry values are shared by every environment; the world is put back after each. One search runs at a time; sysid.status reports it and sysid.apply writes the result.

**Arguments**

- `episode`: int, the demonstration episode, default the newest
- `params`: list of {param, joints, min, max}: param is joint\_damping, joint\_armature, joint\_frictionloss or link\_mass\_scale; joints default to every joint of the demonstrations; min and max default to 0 and twice the largest value in force, or 0.5 and 2 for a mass scale
- `generations`: int, 1..1000, default the window's (20)
- `population`: int, candidates per generation, 2..256, default the window's (8)
- `sigma`: float, the initial step as a share of each range, in (0, 1\], default 0.3
- `seed`: int, the sampler's seed, default 0

**Returns:** {episode, steps, params, evaluations}: evaluations counts the replays the search may run

**Example**

```json
{"id": "sysid.run", "args": {"episode": 0, "params": [{"param": "joint_damping", "min": 0.0, "max": 2.0}], "generations": 15}}
```

## `sysid.status`

**Kind:** [read](index.md#kinds)

The identification's progress and its best fit so far: each quantity's mean value before the search and the best found; the loss is the mean squared joint error in rad² or m², and the curve holds the best loss after each generation.

**Returns:** {running, generation, generations, population, evaluations, steps, start\_loss, best\_loss, params: \[{param, joints, nominal, value}\], curve, error}

## `sysid.stop`

**Kind:** [sim](index.md#kinds)

Ends the running identification with the best values found so far.

**Returns:** as sysid.status

## `sysid.apply`

**Kind:** [edit](index.md#kinds)

Writes the finished identification's best values to its joints and links in one undoable edit; a mass scale multiplies each link's mass from before the search. Fails when the search did not improve on the values in force.

**Returns:** {focus}

## `sysid.panel`

**Kind:** [view](index.md#kinds)

Opens or closes the System Identification window; `on` sets it, else it toggles.

**Arguments**

- `on`: bool, open it, default: toggles

**Returns:** {on}

## `robot.links`

**Kind:** [read](index.md#kinds)

The shown environment's robot links in link order, each with its parent link, incoming joint and mass, and where it and its centre of mass are; the robot's total mass, centre of mass and degrees of freedom.

**Returns:** {links: \[{path, parent, joint, type, mass, origin, com}\], mass, com, dof}

## `robot.asset_links`

**Kind:** [read](index.md#kinds)

Reads an asset file without importing it and lists its links, its roots first: the mounts robot.attach\_asset takes. Without `path` a file dialog asks.

**Arguments**

- `path`: string, the asset file, default: a file dialog
- `profile`: string, how a URDF reads: strict or legacy, default the Assembler's

**Returns:** {path, links, roots}

## `robot.attach`

**Kind:** [edit](index.md#kinds)

Hangs the body `child`, and what its joints carry, off the link `parent` through a new joint, as one undoable edit. `frame` puts `mount` (default the child) at that pose in the parent's frame; without it the child stays where it is.

**Arguments**

- `child`: string, the body to attach
- `parent`: string, the link it hangs off
- `joint`: string, fixed, revolute or prismatic, default "fixed"
- `axis`: \[x, y, z\], the joint axis, default \[0, 0, 1\]
- `frame`: {pos: \[x, y, z\], rot: \[x, y, z\] degrees or quat: \[w, x, y, z\]}, the mount's pose in the parent's frame, optional
- `mount`: string, the link of what is attached that meets the parent, default its root
- `joint_name`: string, the new joint's name, optional

**Returns:** {focus, joint}: the attached root and its new joint

## `robot.attach_asset`

**Kind:** [edit](index.md#kinds)

Adds the asset at `path` and hangs its link `mount` (default its root) off the link `parent` in one undoable edit; without `frame` the mount takes the parent's frame. Takes file.import's options.

**Arguments**

- `path`: string, the asset file
- `parent`: string, the link it hangs off
- `name`: string, the new node's name, default the file's stem
- `joint`: string, fixed, revolute or prismatic, default "fixed"
- `axis`: \[x, y, z\], the joint axis, default \[0, 0, 1\]
- `frame`: {pos: \[x, y, z\], rot: \[x, y, z\] degrees or quat: \[w, x, y, z\]}, the mount's pose in the parent's frame, default the parent's frame
- `mount`: string, the asset's link that meets the parent (see robot.asset\_links), default its root
- `joint_name`: string, the new joint's name, optional
- `profile`: string, how a URDF reads: strict or legacy, default "strict"

**Returns:** {focus, joint}

**Example**

```json
{"id": "robot.attach_asset", "args": {"path": "C:/assets/gripper/gripper.urdf", "parent": "/go2/base", "frame": {"pos": [0.2, 0, 0.1]}}}
```

## `robot.assembler`

**Kind:** [view](index.md#kinds)

Opens or closes the Robot Assembler; opened on a selected link, that link becomes the parent. `on` sets it, else it toggles.

**Arguments**

- `on`: bool, open it, default: toggles

**Returns:** {on}

## `robot.ik`

**Kind:** [sim](index.md#kinds)

Moves `link` (node path or joint name) to world `pos`, and `quat` or XYZ-degree `rot`, keeping `pins` put. `apply` (default) writes the solution as a gizmo drag does, else it only reports it.

**Arguments**

- `link`: string, the link's Stage path or its joint's name
- `pos`: \[x, y, z\], the world position to reach
- `quat`: \[w, x, y, z\], the world orientation, optional
- `rot`: \[x, y, z\], the world orientation as XYZ degrees, optional
- `position_only`: bool, ignore the orientation, default false
- `pins`: list of strings, links held where they are, default the IK settings' pins
- `root`: string, the chain's root link, default the IK settings' root
- `rotation_weight`: number, metres per radian of orientation error, default the IK settings' weight
- `apply`: bool, write the solution to the joints, default true

**Returns:** {q: joint to value, position\_error, rotation\_error, iterations, joints, at\_limit, applied}

**Example**

```json
{"id": "robot.ik", "args": {"link": "/World/arm/hand", "pos": [0.4, 0.0, 0.3], "rot": [180, 0, 0]}}
```

## `robot.ik_settings`

**Kind:** [view](index.md#kinds)

The gizmo's IK: the chain root (empty for the articulation's), the links a drag pins, whether a move locks the orientation and the rotation weight \[m/rad\]; an argument given replaces its setting.

**Arguments**

- `root`: string, the chain's root link, "" for the articulation's root, default kept
- `pins`: list of strings, links a drag holds, default kept
- `lock_orientation`: bool, a move keeps the link's orientation, default kept
- `rotation_weight`: number, metres per radian, default kept

**Returns:** {on, root, pins, lock\_orientation, rotation\_weight}
