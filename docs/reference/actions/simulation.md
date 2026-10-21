<!-- Generated from assets/editor/actions.toml by tools/wiki_actions.py; edit those instead. -->

# Simulation, timeline, solver and Rerun

31 actions. Run one from the command palette (++ctrl+p++), from Python with `app.actions.run(id, **args)` or through MCP with `action_run`.

## `sim.play`

**Kind:** [sim](index.md#kinds)

Starts the simulation. Playing takes the world over from a watched training run, from the last frame it showed.

**Returns:** {playing}

## `sim.pause`

**Kind:** [sim](index.md#kinds)

Pauses the simulation.

**Returns:** {playing}

## `sim.toggle_play`

**Kind:** [sim](index.md#kinds)

Plays the simulation when it is paused, else pauses it.

**Returns:** {playing}

## `sim.step`

**Kind:** [sim](index.md#kinds)

Advances the paused simulation by `steps` physics steps.

**Arguments**

- `steps`: int, how many steps, default 1

**Returns:** nothing

## `sim.reset`

**Kind:** [sim](index.md#kinds)

Resets every environment to the scene's initial state at tick 0 and pauses; an active randomization draws again (with the next seed unless the draw stays).

**Returns:** nothing

## `sim.speed`

**Kind:** [sim](index.md#kinds)

Sets the playback speed: 1 is real time, below 1 slow motion, above 1 faster when the machine keeps up.

**Arguments**

- `value`: number, the speed factor, default 1

**Returns:** {speed}

## `sim.envs`

**Kind:** [sim](index.md#kinds)

{count} rebuilds the world with that many environments, all from the scene's initial state at tick 0; loads and edits keep the count. Without it, the grid as view.envs reports it.

**Arguments**

- `count`: int, how many environments, 1 up to the editor's limit, default the current count

**Returns:** {count, current, show, spacing, auto\_spacing, cells: \[{env, offset}\], copies}

**Example**

```json
{"id": "sim.envs", "args": {"count": 64}}
```

## `sim.perturb`

**Kind:** [sim](index.md#kinds)

A world force and torque at a world point (the COM by default), plus a spring toward `target`; no load and no target releases the body. The load holds until released, on the shown environment.

**Arguments**

- `path`: string, the Stage path of a moving link or body
- `force`: \[x, y, z\], N, default none
- `torque`: \[x, y, z\], N m, default none
- `point`: \[x, y, z\], the world point the force acts at, default the COM
- `target`: \[x, y, z\], a world point the body is pulled toward by a spring, optional
- `gain`: number, scales the spring's stiffness and damping, default 1

**Returns:** {held, stiffness, damping}

**Example**

```json
{"id": "sim.perturb", "args": {"path": "/World/go2/base", "force": [0, 80, 0]}}
```

## `sim.grab_particle`

**Kind:** [sim](index.md#kinds)

Holds one environment particle (cloth or MPM) kinematic at `target`, where it is by default.

**Arguments**

- `particle`: int, the particle's index in the shown environment
- `target`: \[x, y, z\], where to hold it, default where it is

**Returns:** {particle, target}

## `sim.release`

**Kind:** [sim](index.md#kinds)

Lets go of the load on `path`, the held `particle`, or with neither every load.

**Arguments**

- `path`: string, the body whose load ends, optional
- `particle`: int, the particle to let go, optional

**Returns:** nothing

## `sim.loads`

**Kind:** [read](index.md#kinds)

The loads held on the shown environment as its last frame has them.

**Returns:** a list of {at, target, particle} or {at, target, path, force, torque, stiffness, damping}

## `solver.telemetry`

**Kind:** [sim](index.md#kinds)

Records solver telemetry each step: per-step residuals, sweep counts and capped steps. Without a mask the action toggles every channel; the alert defaults to the current one.

**Arguments**

- `mask`: int, the channels as NUKA\_TELEMETRY\_\* bits, 0 stops recording, default: toggles all
- `alert`: number, the residual above which a step counts as an alert, default the current alert

**Returns:** {mask, alert}

## `solver.heatmap`

**Kind:** [view](index.md#kinds)

Colours the solver's per-vertex residuals over the scene; the heatmap needs the per-vertex channel and turns it on. `on` sets it, else it toggles.

**Arguments**

- `on`: bool, heatmap on, default: toggles

**Returns:** {on}

## `solver.set_iterations`

**Kind:** [sim](index.md#kinds)

Sets the solver's sweep budget per step, live.

**Arguments**

- `n`: int, sweeps per step, at least 1

**Returns:** {n}

## `sim.dt`

**Kind:** [sim](index.md#kinds)

{seconds} rebuilds the world to step that often, its state carried over; without it, the current step. A loaded policy keeps the step it was trained at, so unload it first.

**Arguments**

- `seconds`: number, the physics step in s, 0.0001 to 0.05, default the current one

**Returns:** {seconds, rate\_hz}

**Example**

```json
{"id": "sim.dt", "args": {"seconds": 0.005}}
```

## `solver.status`

**Kind:** [read](index.md#kinds)

The solver telemetry's state: channels, alert, sweep budget, heatmap, records kept and capped, the newest record and the located marker.

**Returns:** {mask, alert, iterations, dt, heatmap, records, caps, recent\_caps, recent\_records, dropped, first, next, applied, last, marker}

## `solver.records`

**Kind:** [read](index.md#kinds)

Pages forward through the kept telemetry records: `since` on, at most `limit`, oldest first.

**Arguments**

- `since`: int, the first record number, default the oldest kept
- `limit`: int, the most records returned, default 64
- `capped`: bool, only steps that hit the sweep cap, default false

**Returns:** {records, first, next}: pass next as since to read on

## `solver.locate`

**Kind:** [view](index.md#kinds)

Frames and selects the worst element of a kept record: the given one, else the newest capped one, else the newest; {"clear": true} drops the marker.

**Arguments**

- `record`: int, the record number, default the newest capped or newest
- `clear`: bool, drop the marker instead, default false

**Returns:** {record, label, env, position, entity, residual, capped}

## `timeline.status`

**Kind:** [read](index.md#kinds)

The timeline: the playhead tick, the start and head of the current branch, whether it replays, every branch, the keyframes, the ghost, the keyframe settings and memory, and a running or finished video export.

**Returns:** {tick, start, head, replaying, branch, branches: \[{id, parent, fork\_tick, head, label, exact}\], keys, pinned, ghost, ghost\_on, ghost\_shown, interval, keyframes, bytes, budget\_mb, applied, export, exported}

## `timeline.seek`

**Kind:** [sim](index.md#kinds)

Moves the playhead and pauses: {tick}, {delta} ticks from the playhead, or {to: "start" | "head"}. Ticks before the head replay the recorded history exactly.

**Arguments**

- `tick`: int, the tick to show, optional
- `delta`: int, ticks from the playhead, negative goes back, optional
- `to`: string, start or head, optional

**Returns:** nothing

**Example**

```json
{"id": "timeline.seek", "args": {"delta": -60}}
```

## `timeline.fork`

**Kind:** [sim](index.md#kinds)

Starts a new branch at the playhead: what runs from here on is kept beside the old history.

**Arguments**

- `label`: string, the branch's name, default empty

**Returns:** nothing

## `timeline.branch`

**Kind:** [sim](index.md#kinds)

Switches to another branch and pauses at its playhead.

**Arguments**

- `id`: int, the branch's id from timeline.status

**Returns:** nothing

## `timeline.rename`

**Kind:** [sim](index.md#kinds)

Renames a branch.

**Arguments**

- `id`: int, the branch's id, default the current branch
- `label`: string, the new name

**Returns:** nothing

## `timeline.ghost`

**Kind:** [view](index.md#kinds)

{on, branch}: draws another branch's bodies as a ghost over this one at the same tick; without `on` it toggles; `branch` picks what it shows, else the last parent.

**Arguments**

- `on`: bool, ghost on, default: toggles
- `branch`: int, the branch to show, default the parent

**Returns:** nothing

## `timeline.configure`

**Kind:** [sim](index.md#kinds)

Sets how often the timeline keeps a keyframe and how much memory its history may use.

**Arguments**

- `interval`: int, ticks between keyframes, default the current interval
- `budget_mb`: number, the history's memory budget in MB, default the current budget

**Returns:** nothing

## `timeline.clear`

**Kind:** [sim](index.md#kinds)

Drops the recorded history and branches, keeping the world where it is.

**Returns:** nothing

## `timeline.hash`

**Kind:** [read](index.md#kinds)

The world's state hash once every command queued before it applied; equal hashes mean bit-identical states.

**Returns:** {hash, tick}

## `timeline.save`

**Kind:** [file](index.md#kinds)

The current branch's line of history with what it plays in, as a .nkrec; the dialog asks without `path`.

**Arguments**

- `path`: string, the .nkrec to write, default: a file dialog

**Returns:** {path, start, head, keyframes, inputs}

## `timeline.load`

**Kind:** [file](index.md#kinds)

A recording of this document and world layout becomes the history, back where it was saved; the dialog asks without `path`.

**Arguments**

- `path`: string, the .nkrec to read, default: a file dialog

**Returns:** {path, start, head, tick}

## `timeline.export`

**Kind:** [file](index.md#kinds)

An mp4 of ticks `from` to `to` (the start and head) at `fps` frames a simulated second, drawn by the viewport and encoded by ffmpeg from the PATH; timeline.status reports its progress.

**Arguments**

- `path`: string, the .mp4 to write, default: a file dialog
- `fps`: number, frames per simulated second, default 30
- `from`: int, the first tick, default the start
- `to`: int, the last tick, default the head

**Returns:** {path, frames}

## `rerun.stream`

**Kind:** [process](index.md#kinds)

Logs the shown environment's link and body poses to Rerun every `every` ticks at the head, named as nuka.editor.nkrec names them (world/links/&lt;node&gt;, world/bodies/&lt;node&gt;) over the `tick` and `sim_time` timelines: into the .rrd `file`, else to the viewer at `addr`, else to a new local viewer. Needs rerun-sdk in the editor's Python and fails saying so without it; loading another scene ends the stream.

**Arguments**

- `on`: bool, start or stop, default: toggles
- `every`: int, ticks between samples, default about 60 a simulated second
- `file`: string, an .rrd file to write instead of sending to a viewer, default none
- `addr`: string, a running viewer's gRPC address such as rerun+http://127.0.0.1:9876/proxy, default: spawn a local viewer

**Returns:** {on, every, links, bodies} when it starts; {on: false, samples} when it stops

**Example**

```json
{"id": "rerun.stream", "args": {"on": true, "file": "C:/runs/walk.rrd", "every": 2}}
```
