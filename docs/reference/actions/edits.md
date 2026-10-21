<!-- Generated from assets/editor/actions.toml by tools/wiki_actions.py; edit those instead. -->

# Edits and undo

6 actions. Run one from the command palette (++ctrl+p++), from Python with `app.actions.run(id, **args)` or through MCP with `action_run`.

## `edit.undo`

**Kind:** [edit](index.md#kinds)

Undoes the last edit of the document.

**Returns:** {focus}: the node the restored step selects

## `edit.redo`

**Kind:** [edit](index.md#kinds)

Redoes the last undone edit of the document.

**Returns:** {focus}: the node the restored step selects

## `edit.apply`

**Kind:** [edit](index.md#kinds)

Applies one edit transaction: its ops change fields of Stage nodes, all as one undoable step. Fields: transform ({pos, quat}), name, reparent, delete, spawn, static, drive, initial\_position, pin, attach, sensor, collision, script, import, reimport, material and material\_\* (new, rename, delete, assign, unique, friction), plus any live physics parameter by name. A null `before` is read from the document.

**Arguments**

- `label`: string, the step's name in the history, default empty
- `ops`: list of {target, field, before, after}: target is a Stage path ("" is the world), after the new value

**Returns:** {focus}: the node the edit selects

**Example**

```json
{"id": "edit.apply", "args": {"label": "Lift", "ops": [{"target": "/World/box", "field": "transform", "after": {"pos": [0, 0, 1], "quat": [1, 0, 0, 0]}}]}}
```

## `edit.rename`

**Kind:** [edit](index.md#kinds)

Renames the node at `path`, or the selection.

**Arguments**

- `name`: string, the new name
- `path`: string, the node's Stage path, default the selection

**Returns:** {focus}: the renamed node's path

## `edit.delete`

**Kind:** [edit](index.md#kinds)

Deletes `paths`, `path` or the selection in one step; a node under another goes with it. With no node selected it removes the selected sensor.

**Arguments**

- `paths`: list of strings, the Stage paths to delete, optional
- `path`: string, one Stage path to delete, default the selection

**Returns:** {focus}

## `edit.reparent`

**Kind:** [edit](index.md#kinds)

Moves the node at `path`, or the selection, under another node.

**Arguments**

- `parent`: string, the new parent's Stage path
- `path`: string, the node to move, default the selection

**Returns:** {focus}: the moved node's new path
