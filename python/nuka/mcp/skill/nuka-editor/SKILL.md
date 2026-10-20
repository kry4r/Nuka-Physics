---
name: nuka-editor
description: Work the Nuka physics editor through its MCP server (nuka-editor tools such as action_run, scene_tree, screenshot). Use it to build or import scenes (USD, URDF, MJCF, meshes), edit bodies, materials, sensors and robots, simulate and record, make datasets, train and replay locomotion policies, or export USD, and to check each result in the editor.
---

# Nuka editor

The Nuka editor is a desktop app for physics scenes and robots. Its MCP server (`nuka-mcp`, see
[connect](references/connect.md)) lets you do anything a user can do in it. Every menu item, button and
command is an *action* with an id such as `file.import` or `sim.step`, and the tools run actions and
look at the result.

## The loop

1. `editor_status`: is an editor running, and what does its document hold? If none runs, start one with
   `editor_launch` (optionally on a scene file).
2. Find the action: [the action index](references/actions.md) groups all of them; `actions_find` searches
   by words and kind. [Workflows](references/workflows.md) has the usual sequences.
3. `action_help {id}` before using an action for the first time: its arguments, their defaults, what it
   returns and an example.
4. `action_run {id, args}`, or `actions_batch` for a known sequence.
5. Check what happened before going on ([verify](references/verify.md)): `document_get`, `scene_tree`,
   `scene.fields`, a `screenshot`.

## Rules

- **Use actions, not Python.** `python_eval` runs code inside the editor and is for what no action
  covers; it is a `process` action, so the user is often asked first.
- **Always pass file paths.** An action that reads or writes a file opens a dialog when `path` is
  missing, and a dialog waits for the user. Use absolute paths.
- **Node paths come from the editor.** Take them from `scene_tree` or from an action's `focus`, exactly
  as returned (`go2/base`, `ground/ground_collision`); `""` is the world.
- **Field names come from `scene.fields`.** It lists what the Properties panel edits for a node: each
  group's `targets` and each field's op name, unit, range and value. Change them with `edit.apply`
  `{label, ops: [{target, field, after}]}`, one undoable step per call.
- **Unsaved work belongs to the user.** `file.new`, `file.open`, `file.close` and `app.exit` return
  `{prompt}` when the document has unsaved changes. Pass `discard: true` only when the user said the
  changes can go; otherwise save first (`file.save {path}`) or ask.
- **Every edit can be undone**: `edit.undo`, and `edit.redo`. Label your edits so the user can tell
  them apart in the history.
- **Long jobs run in the background.** `train.start`, `dataset.export`, `view.render` and
  `timeline.export` return at once; follow them with `wait_for` on their status action.
- **The user decides what you may do.** They choose an access mode; read `document_get`'s `agent.mode`.
  A call can come back as `{request, state: "pending"}` while the user decides; poll `proposal_get`. In
  any mode but read, `edit_propose` offers a change for the user to accept. A refusal says why; do not
  try to get around it. See [safety](references/safety.md).
- **The simulation is deterministic.** `sim.step {steps}` advances a paused world exactly;
  `timeline.hash` tells whether two states are identical.

## References

- [connect.md](references/connect.md): installing the server and this skill, MCP client configuration,
  several editors, the Python client.
- [workflows.md](references/workflows.md): scenes, imports, collision, materials, robots, simulation and
  recording, sensors and datasets, training and replay, export.
- [actions.md](references/actions.md): every action by group, with its arguments and kind.
- [verify.md](references/verify.md): checking each step, and `scenario_run` for checks in bulk.
- [safety.md](references/safety.md): access modes, proposals, approvals and undo.

The same pages are MCP resources: `nuka://guide/workflows` and so on.
