<!-- Generated from assets/editor/actions.toml by tools/wiki_actions.py; edit those instead. -->

# Windows, panels and tools

41 actions. Run one from the command palette (++ctrl+p++), from Python with `app.actions.run(id, **args)` or through MCP with `action_run`.

## `window.minimize`

**Kind:** [app](index.md#kinds)

Minimizes the editor's window.

**Returns:** nothing

## `window.maximize`

**Kind:** [app](index.md#kinds)

Maximizes the editor's window, or restores it when it is maximized.

**Returns:** nothing

## `window.restore`

**Kind:** [app](index.md#kinds)

Restores the editor's window from minimized or maximized.

**Returns:** nothing

## `window.state`

**Kind:** [read](index.md#kinds)

The editor window's state: maximized, minimized, focused, position and size in pixels, and the title bar's height and drag spans.

**Returns:** {maximized, minimized, focused, x, y, width, height, title\_bar: {height, drag}}

## `tool.perturb`

**Kind:** [view](index.md#kinds)

Makes Perturb the viewport tool (dragging a body pulls it with a spring), or goes back to Select; `on` sets it.

**Arguments**

- `on`: bool, true picks the tool, false goes back to Select, default: toggles

**Returns:** {tool}

## `tool.pin`

**Kind:** [view](index.md#kinds)

Makes Pin the viewport tool (clicking a body holds it where it is), or goes back to Select; `on` sets it.

**Arguments**

- `on`: bool, true picks the tool, false goes back to Select, default: toggles

**Returns:** {tool}

## `tool.measure`

**Kind:** [view](index.md#kinds)

`a` and `b` set the Measure tool's ends (a link's node path, a joint's name, or "ground" for b) and return the measurement; `at` takes link ends at their origin or com. Else it is the tool.

**Arguments**

- `a`: string, the first end: a link's Stage path or a joint's name, optional
- `b`: string, the second end: a link, a joint or ground, default "ground"
- `at`: string, origin or com for link ends, default the current choice
- `on`: bool, without `a`: true picks the tool, false goes back to Select, default: toggles

**Returns:** {distance, delta, from, to} with ends, else {tool}

**Example**

```json
{"id": "tool.measure", "args": {"a": "/World/go2/base", "b": "ground"}}
```

## `tool.measure_clear`

**Kind:** [view](index.md#kinds)

Clears the Measure tool's ends.

**Returns:** nothing

## `ui.pointer`

**Kind:** [app](index.md#kinds)

Shows a drawn mouse pointer and plays a move with it, for demos and recordings: it glides to an item found by its label, or to a window point, then clicks, double-clicks, drags, holds or turns the wheel there. While it is shown it replaces the real mouse. Moves queue behind the ones still playing; ui.pointer\_status says when they are done and why a label was not found.

**Arguments**

- `on`: bool, false hides the pointer and drops the queued moves, default true
- `to`: string or \[x, y\], a visible item's label (as shown or its ## id; an exact match first, else one containing it) or a window point, default where it is
- `window`: string, only items in windows whose name contains this, default any
- `offset`: \[dx, dy\], window points from the item's centre, default \[0, 0\]
- `seconds`: number, the glide, default from the distance (0.3 to 0.9 s)
- `click`: bool, click at the target, default false
- `double`: bool, double-click at the target, default false
- `button`: string, left, middle or right, default "left"
- `drag`: string or \[dx, dy\], press at the target and drag onto this item or by this many window points, default none
- `drag_seconds`: number, the drag's glide, default from the distance
- `hold`: number, seconds the button stays down before a drag moves, or a long press without one, default 0
- `wheel`: int, wheel notches at the target, + away from the user, default 0
- `wait`: number, seconds to pause after the move, default 0

**Returns:** {on, queued}

**Example**

```json
{"id": "ui.pointer", "args": {"to": "Play", "click": true}}
```

## `ui.pointer_status`

**Kind:** [read](index.md#kinds)

The demo pointer: whether it is shown and still playing moves, the steps queued, where it is, whether a button is down, and why the last label was not found.

**Returns:** {on, busy, queued, x, y, pressed, error}

## `tool.ik`

**Kind:** [view](index.md#kinds)

While on, the gizmo on a jointed link poses the robot through IK; without `on` it toggles.

**Arguments**

- `on`: bool, IK drag on, default: toggles

**Returns:** {on}

## `window.layout_authoring`

**Kind:** [view](index.md#kinds)

Arranges the panels for building a scene: Stage and File left, Properties and Camera right, Assets, Console, Script, Stats, Timeline and Sensors below.

**Returns:** nothing

## `window.layout_debug`

**Kind:** [view](index.md#kinds)

Arranges the panels for debugging the solver: Stage left, Solver, Debug, Stats and Properties right, Console, Timeline, Script and Assets below.

**Returns:** nothing

## `window.layout_robot`

**Kind:** [view](index.md#kinds)

Arranges the panels for a robot: Stage left, Joints, Poses, Policy, Demos and Properties right, Console, Timeline, Stats, Assets and Sensors below.

**Returns:** nothing

## `window.layout_data`

**Kind:** [view](index.md#kinds)

Arranges the panels for making data: File and Stage left, Sensors, Randomize, Stats, Camera and Properties right, Assets, Console, Script and Timeline below.

**Returns:** nothing

## `window.layout_training`

**Kind:** [view](index.md#kinds)

Arranges the panels for training: Stats and Stage left, a wide Training, Policy and Camera column right, Console, Script, Assets and Timeline below.

**Returns:** nothing

## `window.reset_layout`

**Kind:** [view](index.md#kinds)

Puts the panels of the current layout back where it places them.

**Returns:** nothing

## `window.focus`

**Kind:** [view](index.md#kinds)

Opens a panel and brings it to the front.

**Arguments**

- `panel`: string, the panel id: stage, property, file, assets, stats, debug, solver, timeline, camera, joints, poses, policy, demos, sensors, randomize, training, script, console, nui, extensions or an extension's

**Returns:** nothing

**Example**

```json
{"id": "window.focus", "args": {"panel": "training"}}
```

## `help.about`

**Kind:** [view](index.md#kinds)

Shows the About window: the nui API and Dear ImGui versions and the open-source credits.

**Returns:** nothing

## `ui.toast`

**Kind:** [view](index.md#kinds)

Shows a short notice in the corner of the editor.

**Arguments**

- `text`: string, the notice
- `level`: string, info, ok, warn or error, default "info"

**Returns:** nothing

**Example**

```json
{"id": "ui.toast", "args": {"text": "Training started", "level": "ok"}}
```

## `window.toggle_stage`

**Kind:** [view](index.md#kinds)

Opens or closes the Stage panel: the scene tree, to select, rename, reparent and spawn.

**Returns:** nothing

## `window.toggle_property`

**Kind:** [view](index.md#kinds)

Opens or closes the Properties panel: transform, material and physics fields of the selection.

**Returns:** nothing

## `window.toggle_file`

**Kind:** [view](index.md#kinds)

Opens or closes the File panel: open, save and load scenes.

**Returns:** nothing

## `window.toggle_assets`

**Kind:** [view](index.md#kinds)

Opens or closes the Assets panel: importable files to drop into the scene or open.

**Returns:** nothing

## `window.toggle_stats`

**Kind:** [view](index.md#kinds)

Opens or closes the Stats panel: frame and physics timings.

**Returns:** nothing

## `window.toggle_debug`

**Kind:** [view](index.md#kinds)

Opens or closes the Debug Draw panel: collider and contact overlays.

**Returns:** nothing

## `window.toggle_solver`

**Kind:** [view](index.md#kinds)

Opens or closes the Solver panel: solver residuals, sweep budget alerts and the residual heatmap.

**Returns:** nothing

## `window.toggle_timeline`

**Kind:** [view](index.md#kinds)

Opens or closes the Timeline panel: seek, replay and fork the run, and compare branches with a ghost.

**Returns:** nothing

## `window.toggle_camera`

**Kind:** [view](index.md#kinds)

Opens or closes the Camera panel: camera pose, framing and controls.

**Returns:** nothing

## `window.toggle_joints`

**Kind:** [view](index.md#kinds)

Opens or closes the Joints panel: joint states, drive targets and gains, the fixed base and teleop.

**Returns:** nothing

## `window.toggle_poses`

**Kind:** [view](index.md#kinds)

Opens or closes the Poses panel: named joint poses to blend to, start from, import and export.

**Returns:** nothing

## `window.toggle_policy`

**Kind:** [view](index.md#kinds)

Opens or closes the Policy panel: load a locomotion policy, run or stop it and watch its actions.

**Returns:** nothing

## `window.toggle_demos`

**Kind:** [view](index.md#kinds)

Opens or closes the Demos panel: record teleoperated demonstrations, replay them and save them as HDF5.

**Returns:** nothing

## `window.toggle_sensors`

**Kind:** [view](index.md#kinds)

Opens or closes the Sensors panel: preview each sensor's images, ranges and readings as the simulation runs.

**Returns:** nothing

## `window.toggle_randomize`

**Kind:** [view](index.md#kinds)

Opens or closes the Randomize panel: randomize looks, masses, gravity and parameters from a seed, again on reset.

**Returns:** nothing

## `window.toggle_training`

**Kind:** [view](index.md#kinds)

Opens or closes the Training panel: configure a task, start and watch runs, plot them and play their saves.

**Returns:** nothing

## `window.toggle_script`

**Kind:** [view](index.md#kinds)

Opens or closes the Script panel: the Python editor.

**Returns:** nothing

## `window.toggle_console`

**Kind:** [view](index.md#kinds)

Opens or closes the Console panel: script output and a Python prompt.

**Returns:** nothing

## `window.toggle_nui`

**Kind:** [view](index.md#kinds)

Opens or closes the UI Inspector: the nui component tree, for extension authors.

**Returns:** nothing

## `tools.nui_inspector`

**Kind:** [view](index.md#kinds)

Opens or closes the UI Inspector; the same as window.toggle\_nui.

**Returns:** nothing

## `window.toggle_agents`

**Kind:** [view](index.md#kinds)

Opens or closes the Agents panel: the programs driving the editor from outside, their access, requests and calls.

**Returns:** nothing
