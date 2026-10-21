<!-- Generated from assets/editor/actions.toml by tools/wiki_actions.py; edit those instead. -->

# Motion: clips, demonstrations, teleoperation, policies

36 actions. Run one from the command palette (++ctrl+p++), from Python with `app.actions.run(id, **args)` or through MCP with `action_run`.

## `teleop.devices`

**Kind:** [read](index.md#kinds)

The devices teleop reads (keyboard, gamepads, the virtual device, extension devices) and the modes it maps them through, with the active ones.

**Returns:** {devices: \[{id, label, kind, owner, axes, buttons}\], modes: \[{id, label, owner, quantities}\], device, mode}

## `teleop.use`

**Kind:** [sim](index.md#kinds)

Switches teleop on or off and picks the device, the mode, the joint mode's joint, the ik mode's link ("" follows the selection), PD hold and the command baseline; bare, it toggles.

**Arguments**

- `on`: bool, teleop on, default kept (bare: toggles)
- `device`: string, a device id from teleop.devices, default kept
- `mode`: string, a mode id: command (vx, vy, wyaw for a policy), joint, ik or an extension's, default kept
- `joint`: string, the joint mode's joint, default kept
- `link`: string, the ik mode's link, "" follows the selection, default kept
- `hold`: bool, hold the pose with a PD law, default kept
- `kp`: number, the hold's stiffness, default kept
- `kd`: number, the hold's damping, default kept
- `command_base`: \[vx, vy, wz\], the command when the sticks rest, default kept

**Returns:** the teleop status, as teleop.status

**Example**

```json
{"id": "teleop.use", "args": {"on": true, "device": "virtual", "mode": "command"}}
```

## `teleop.input`

**Kind:** [sim](index.md#kinds)

Sets the virtual device's axes (\[-1, 1\]) and buttons by name; the others keep their values. This is how an agent steers teleop without hardware.

**Arguments**

- `axes`: object, axis name (leftx, lefty, rightx, righty, lefttrigger, righttrigger) to a value in \[-1, 1\], optional
- `buttons`: object, button name (a, b, x, y, start, dpup, ...) to true, false or a number (nonzero is down), optional

**Returns:** {axes, buttons}: every axis and button now

**Example**

```json
{"id": "teleop.input", "args": {"axes": {"lefty": 0.6}}}
```

## `teleop.map`

**Kind:** [sim](index.md#kinds)

A mode's bindings \[{input, quantity, gain, invert}\]: given ones replace them and `reset` restores the defaults; `deadzone` in \[0, 1) is shared by every axis.

**Arguments**

- `mode`: string, the mode, default the active one
- `bindings`: list of {input, quantity, gain, invert}, the new bindings, optional
- `reset`: bool, restore the mode's default bindings first, default false
- `deadzone`: number, in \[0, 1), default the current one

**Returns:** {mode, bindings, quantities, deadzone}

## `teleop.pad`

**Kind:** [sim](index.md#kinds)

A virtual gamepad on SDL's joystick path: `attach` names a new one; `id` with `axes` and `buttons` sets its inputs, and with `detach` removes it.

**Arguments**

- `attach`: string, the new gamepad's name, optional
- `id`: string, the gamepad's id from attach
- `axes`: object, SDL axis name to value, optional
- `buttons`: object, SDL button name to pressed, optional
- `detach`: bool, remove the gamepad, default false

**Returns:** {id} (and detached)

## `teleop.status`

**Kind:** [read](index.md#kinds)

The active device's reading, the mode's quantities and what they drove last frame.

**Returns:** {on, device, connected, mode, axes, buttons, quantities, command, command\_base, controller, joint, link, goal, hold, kp, kd, deadzone, error}

## `policy.load`

**Kind:** [process](index.md#kinds)

Attaches an NKML locomotion policy to the loaded robots in place of the one before, and sends it teleop's command; the dialog asks without `path`, `host` runs the host MLP. A policy with a `<name>.contract.json` beside it, as a training run writes one, runs on every robot with the contract's joints through its observation terms and gains; any other runs under the Go2 contract.

**Arguments**

- `path`: string, the policy file (.nkml or .bin), default: a file dialog
- `host`: bool, run the MLP on the CPU instead of the GPU, default false

**Returns:** the policy status, as policy.status

## `policy.unload`

**Kind:** [sim](index.md#kinds)

Detaches the loaded policy; the drive targets hold where it left them.

**Returns:** the policy status

## `policy.enable`

**Kind:** [sim](index.md#kinds)

Runs or stops the loaded policy; stopped, the drive targets hold where it left them. Bare, it toggles.

**Arguments**

- `on`: bool, run it, default: toggles

**Returns:** the policy status

## `policy.status`

**Kind:** [read](index.md#kinds)

The loaded policy, its contract, the joints it drives and the command it reads.

**Returns:** {loaded, on, path, error, device, robots, obs\_dim, act\_dim, decimation, rate\_hz, action\_scale, joints, command, action\_norm, contract, roots, arrows\_on, arrows: \[{at, command, velocity}\]}

## `policy.arrows`

**Kind:** [view](index.md#kinds)

Shows or hides the arrows of the command and the measured velocity over the policy's robots. Bare, it toggles.

**Arguments**

- `on`: bool, show them, default: toggles

**Returns:** the policy status

## `demo.record`

**Kind:** [sim](index.md#kinds)

Records the shown environment, with `all` every displayed one, each step at the head: drive targets, joint states and the teleop device's axes. The stop cuts each environment's consecutive steps into an episode. Bare, it toggles.

**Arguments**

- `on`: bool, recording on, default: toggles
- `all`: bool, record every environment the grid shows, default kept

**Returns:** the demonstrations' status, as demo.episodes

## `demo.episodes`

**Kind:** [read](index.md#kinds)

The recorded episodes, the recording and the replay in progress.

**Returns:** {recording, samples, playing, left, joints, axes, episodes: \[{env, start, steps, seconds}\]}

## `demo.play`

**Kind:** [sim](index.md#kinds)

Writes an episode's recorded drive targets into its environment a row a step from the state now, the controller waiting meanwhile; `play` (the default) starts the transport.

**Arguments**

- `episode`: int, the episode's index, default the selected one
- `play`: bool, start the simulation, default true

**Returns:** {episode, env, steps}

## `demo.stop`

**Kind:** [sim](index.md#kinds)

Stops a demonstration replay.

**Returns:** the demonstrations' status

## `demo.save`

**Kind:** [file](index.md#kinds)

The episodes as robomimic-style HDF5 through nuka.editor.demos; the dialog asks without `path`.

**Arguments**

- `path`: string, the .hdf5 to write, default: a file dialog

**Returns:** {path, episodes, steps}

## `demo.load`

**Kind:** [file](index.md#kinds)

Replaces the episodes with an HDF5 file's, its joints matched by name to the scene's; the dialog asks without `path`.

**Arguments**

- `path`: string, the .hdf5 to read, default: a file dialog

**Returns:** the demonstrations' status with {path}

## `demo.discard`

**Kind:** [sim](index.md#kinds)

Drops episode `episode`, without it all of them.

**Arguments**

- `episode`: int, the episode's index, default every episode

**Returns:** the demonstrations' status

## `clip.list`

**Kind:** [read](index.md#kinds)

The joint clips saved beside the document, the edited one with its playhead and selection, and the clip playing.

**Returns:** {clips, edited, open, playhead, auto\_key, interp, curve, selected, playing, loop, left, can\_undo, can\_redo}

## `clip.get`

**Kind:** [read](index.md#kinds)

A clip's tracks and keys, the edited clip's without `clip`.

**Arguments**

- `clip`: string, the clip's name, default the clip being edited

**Returns:** {name, duration, loop, tracks: \[{joint, keys: \[{time, value, interp, in, out}\]}\]}

## `clip.new`

**Kind:** [edit](index.md#kinds)

A clip of `duration` seconds (1 by default) named `name`, else "Clip N"; the editor edits it.

**Arguments**

- `name`: string, the clip's name, default "Clip N"
- `duration`: number, seconds, default 1
- `loop`: bool, play it over and over, default false

**Returns:** the new clip, as clip.get

## `clip.set`

**Kind:** [edit](index.md#kinds)

Sets a clip's `duration` or `loop`; keys past a shorter end stay but no longer play.

**Arguments**

- `clip`: string, the clip's name, default the clip being edited
- `duration`: number, seconds, default kept
- `loop`: bool, default kept

**Returns:** the clip, as clip.get

## `clip.rename`

**Kind:** [edit](index.md#kinds)

Renames a clip.

**Arguments**

- `clip`: string, the clip's name, default the clip being edited
- `to`: string, the new name

**Returns:** {name}

## `clip.remove`

**Kind:** [edit](index.md#kinds)

Removes a clip.

**Arguments**

- `clip`: string, the clip's name, default the clip being edited

**Returns:** the clips' status, as clip.list

## `clip.key`

**Kind:** [edit](index.md#kinds)

Sets one key from the arguments or each of `keys`, replacing any at their times. A key is {joint, time, value, interp, in, out}: time defaults to the playhead, value to the joint's drive target, interp to the editor's.

**Arguments**

- `clip`: string, the clip's name, default the clip being edited
- `joint`: string, the joint's name, without `keys`
- `time`: number, seconds into the clip, default the playhead
- `value`: number, rad or m, default the joint's drive target
- `interp`: string, step, linear or bezier, default the editor's
- `in`: number, a bezier key's incoming slope, default 0
- `out`: number, a bezier key's outgoing slope, default 0
- `keys`: list of keys as above, in place of one, optional

**Returns:** {clip, set, keys}

**Example**

```json
{"id": "clip.key", "args": {"keys": [{"joint": "FL_thigh_joint", "time": 0, "value": 0.8}, {"joint": "FL_thigh_joint", "time": 0.5, "value": 1.2}]}}
```

## `clip.key_pose`

**Kind:** [edit](index.md#kinds)

Keys every joint at `time` (the playhead) from library pose `pose`, else from the shown environment's positions, or its drive targets with `from` "target".

**Arguments**

- `clip`: string, the clip's name, default the clip being edited
- `time`: number, seconds into the clip, default the playhead
- `pose`: string, a pose library entry, optional
- `from`: string, q (joint positions) or target (drive targets), default "q"

**Returns:** {clip, time, set}

## `clip.remove_keys`

**Kind:** [edit](index.md#kinds)

Removes `keys` \[{joint, time}\], else every key of `joint` and/or at `time`, else the selection.

**Arguments**

- `clip`: string, the clip's name, default the clip being edited
- `keys`: list of {joint, time}, optional
- `joint`: string, every key of this joint, optional
- `time`: number, every key at this time, optional

**Returns:** {removed, keys}

## `clip.move_keys`

**Kind:** [edit](index.md#kinds)

Shifts `keys` \[{joint, time}\], else the selection, by `dt` seconds and `dv` in value; keys stay within the clip and replace those where they land.

**Arguments**

- `clip`: string, the clip's name, default the clip being edited
- `keys`: list of {joint, time}, default the selection
- `dt`: number, seconds, default 0
- `dv`: number, rad or m, default 0

**Returns:** {keys}: where the keys landed

## `clip.sample`

**Kind:** [read](index.md#kinds)

Each keyed joint's value at `time` (the playhead); with `apply` they become the shown environment's drive targets at once.

**Arguments**

- `clip`: string, the clip's name, default the clip being edited
- `time`: number, seconds into the clip, default the playhead
- `apply`: bool, write them as drive targets, default false

**Returns:** {time, values: joint to value}

## `clip.play`

**Kind:** [sim](index.md#kinds)

Drives the shown environment through a clip a step a row from now, over and over with `loop` (the clip's own by default); `play` (the default) starts the transport.

**Arguments**

- `clip`: string, the clip's name, default the clip being edited
- `loop`: bool, default the clip's own
- `play`: bool, start the simulation, default true

**Returns:** the clips' status

## `clip.stop`

**Kind:** [sim](index.md#kinds)

Stops the clip playing.

**Returns:** the clips' status

## `clip.save`

**Kind:** [file](index.md#kinds)

Writes a clip as {"format": "nkclip", "version": 1, "clip": ...}; the dialog asks without `path`.

**Arguments**

- `clip`: string, the clip's name, default the clip being edited
- `path`: string, the .nkclip.json to write, default: a file dialog

**Returns:** {path, clip}

## `clip.load`

**Kind:** [file](index.md#kinds)

Adds a clip file's clip, replacing one of its name; the editor edits it.

**Arguments**

- `path`: string, the .nkclip.json to read, default: a file dialog

**Returns:** {path, clip, keys}

## `clip.editor`

**Kind:** [view](index.md#kinds)

Opens or closes the Clip Editor (bare, it toggles) and sets what it shows: the `clip`, the `playhead`, `auto_key`, the `interp` of new keys, the `curve` joint and the `selected` keys.

**Arguments**

- `on`: bool, open it, default kept (bare: toggles)
- `clip`: string, the clip to edit, default kept
- `playhead`: number, seconds, default kept
- `auto_key`: bool, dragging a joint keys it at the playhead, default kept
- `interp`: string, step, linear or bezier for new keys, default kept
- `curve`: string, the joint whose curve shows, default kept
- `selected`: list of {joint, time}, the selected keys, default kept

**Returns:** the clips' status

## `clip.undo`

**Kind:** [edit](index.md#kinds)

Steps back through the clip edits; scene edits keep their own history.

**Returns:** the clips' status

## `clip.redo`

**Kind:** [edit](index.md#kinds)

Steps forward again through undone clip edits.

**Returns:** the clips' status
