<!-- Generated from assets/editor/actions.toml by tools/wiki_actions.py; edit those instead. -->

# Actions

Every menu item, toolbar button, shortcut and panel command in the editor runs an *action*: an id such as `file.import` with JSON arguments. The same actions are what scripts, extensions, smoke scenarios and agents run, so this reference covers everything the editor does. It lists 317 actions; extensions add their own.

| Group | Actions |
| --- | --- |
| [Documents: new, open, import, save, export](documents.md) | 16 |
| [Edits and undo](edits.md) | 6 |
| [Scene: add, collision, nodes](scene.md) | 19 |
| [Assets](assets.md) | 8 |
| [Materials](materials.md) | 11 |
| [Robots: joints, poses, gains, identification, IK](robots.md) | 33 |
| [Motion: clips, demonstrations, teleoperation, policies](motion.md) | 36 |
| [Simulation, timeline, solver and Rerun](simulation.md) | 31 |
| [Sensors, labels, datasets and randomization](sensors.md) | 16 |
| [Training](training.md) | 16 |
| [View and camera](view.md) | 47 |
| [Windows, panels and tools](windows.md) | 41 |
| [Scripts and console](scripts.md) | 18 |
| [The editor itself, agents and extensions](editor.md) | 19 |

## Running an action

- **In the editor**: menus, toolbar, shortcuts, or the command palette (++ctrl+p++), which finds any action by name.
- **Python in the editor** (Script panel, Console): `app.actions.run("sim.step", steps=10)`.
- **Another process**: `nuka.editor.connect()` returns a client whose `actions.run` takes the same arguments; see [Agents and MCP](../../editor/agents.md).
- **MCP**: `action_run {id, args}`; `action_help {id}` returns an entry of this reference.

An argument without a default is required. A path argument left out opens a file dialog, which waits for the user. An action that changes the document is one undoable step.

## Kinds

Each action's kind says what it touches, and with it what an agent's access mode lets through.

| Kind | Touches |
| --- | --- |
| `read` | Reads state and changes nothing. |
| `view` | The camera, panels, overlays and other display state; never the document. |
| `edit` | The document and what is saved beside it: poses, clips, randomization, labels. |
| `sim` | Runs, steps, resets or drives the simulation, or records it. |
| `file` | Reads or writes files. |
| `process` | Runs code, scripts or other programs. |
| `app` | The editor itself: settings, extensions, windows, exit. |
