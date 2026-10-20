# Checking your work

An action that returns without an error did what it says, but not always what the task needed. Look at
the editor after each step that changes something, and before you report a result.

## What to look at

- `document_get`:
  - `revision` (counts changes), `dirty`, `undo`/`redo` depths, `hash` (the document's content);
  - `selection`, `playing`, `tick`, `last_error` (the newest failure);
  - `agent` (your access mode and what waits on the user).

  A changed `hash` means the document changed; an unchanged one after an edit means the edit did nothing.
- `scene_tree {under, kind, depth}`: the nodes, to confirm an import or a spawn landed where expected.
- `scene.fields {path}`: a field's value after `edit.apply`.
- `action_log {since}`: what ran, from where (`agent:mcp` is you) and whether it failed.
- `console_read {since}`: the editor's console, with warnings from imports and scripts.
- `screenshot {viewport: true}`: what the user sees. Aim first:
  - to look at one node, `scene.select {path}` and then `view.frame_selected`;
  - `view.frame_all` fits everything, a 40 m floor included, so a robot on it shrinks to a dot;
  - `view.camera {target, distance, yaw, pitch}` aims by numbers.

  `view.colliders {on: true}` and `view.contacts {on: true}` draw what the physics uses.

## Simulation results

- Pause, then `sim.step {steps}` for an exact number of steps; read the state with `robot.joints`,
  `sim.loads`, `sensor.read` or `timeline.status`.
- `timeline.hash` after the same steps from the same state must be equal; a difference means something
  changed the inputs.
- `sim.reset` returns to tick 0 and the initial state.

## Background jobs

Use `wait_for {id, args, until, timeout_s}` on the job's status action:

| Job | Status action | Done when |
| --- | --- | --- |
| `dataset.export` | `dataset.status` | `{running: false}`; check `error` and `frame` = `frames` |
| `view.render` | `view.render_status` | `{running: false}`; `last` describes the result |
| `train.start` | `train.status {name}` | `{alive: false}`; `state` is done or failed |
| `script.run`, `script.eval` | `script.result {seq}` | `{done: true}`; `failed` and `output` |

Training runs for a long time: check `train.status` now and then instead of one long wait.

## Checks in bulk

`scenario_run {steps}` runs a list of checks and reports each:

```json
{"steps": [
  {"action": "file.import", "args": {"path": "C:/assets/go2/go2.urdf", "mode": "add"}},
  {"expect": {"dirty": true}},
  {"action": "edit.delete", "args": {"path": "no/such/node"}, "expect_error": true},
  {"wait": "dataset.status", "until": {"running": false}, "timeout_s": 300},
  {"screenshot": "after"}
]}
```

`expect_value` compares keys of an action's returned object. `passed` is true only when every step
passed.
