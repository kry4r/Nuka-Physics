<!-- Generated from assets/editor/actions.toml by tools/wiki_actions.py; edit those instead. -->

# The editor itself, agents and extensions

19 actions. Run one from the command palette (++ctrl+p++), from Python with `app.actions.run(id, **args)` or through MCP with `action_run`.

## `app.exit`

**Kind:** [app](index.md#kinds)

Closes the editor. With unsaved changes and no `discard` it asks first. A script's world outlives the window: closing only hides it and ends app.run().

**Arguments**

- `discard`: bool, drop unsaved changes without asking, default false

**Returns:** nothing, or {prompt} while it asks

## `app.answer_prompt`

**Kind:** [file](index.md#kinds)

Answers the open save question that file.new, file.open, file.close or app.exit raised: save saves first (to `path` for an untitled document) and then reruns the action, discard reruns it dropping the changes, cancel drops it.

**Arguments**

- `choice`: string, save, discard or cancel
- `path`: string, where to save an untitled document when the choice is save, optional

**Returns:** {choice}, or the rerun action's value

**Example**

```json
{"id": "app.answer_prompt", "args": {"choice": "discard"}}
```

## `app.check`

**Kind:** [app](index.md#kinds)

Reports a named check from a script or smoke: it goes to the log and the start page's checks.

**Arguments**

- `name`: string, the check's name
- `value`: string, what was measured
- `ok`: bool, whether it passed, default true

**Returns:** nothing

## `action.help`

**Kind:** [read](index.md#kinds)

Everything about one action: what it does, the arguments it reads, what it returns, an example, its kind and whether it can run now. The action list carries the summary and kind of all of them.

**Arguments**

- `id`: string, the action

**Returns:** {id, label, category, shortcut, icon, hidden, enabled, checked, owner, summary, kind, args, returns, example}

**Example**

```json
{"id": "action.help", "args": {"id": "file.import"}}
```

## `agent.mode`

**Kind:** [app](index.md#kinds)

Sets what programs driving the editor from outside (agents, through nuka.editor.connect or MCP) may do. full runs every action; propose turns their edits into proposals and asks the user before file, process and app actions; read lets only read and view actions through; off closes the endpoint and drops every client. `allow` runs actions without asking, `deny` refuses them; both take ids or wildcards such as file.\*. Only the user runs it, never an agent.

**Arguments**

- `mode`: string, off, read, propose or full, default unchanged
- `allow`: list of strings, actions agents run without asking, default unchanged
- `deny`: list of strings, actions agents may never run, default unchanged

**Returns:** {mode, allow, deny, clients}: clients is how many are connected

**Example**

```json
{"id": "agent.mode", "args": {"mode": "propose", "deny": ["file.*"]}}
```

## `agent.proposals`

**Kind:** [read](index.md#kinds)

The requests agents made that wait for the user, or were answered: proposals of edits and actions to approve, the pending ones first, then the newest.

**Returns:** a list of {id, kind, state, client, action, args, label, note, changes: \[{target, field, before, after}\], value, error}; kind is proposal or approval, state pending, accepted, approved, rejected or failed

## `agent.propose`

**Kind:** [edit](index.md#kinds)

Offers an edit for the user to review in the Agents panel, with each field's value before and after; accepting applies it as one undoable step named after `label`. Agents may propose in every access mode but read and off.

**Arguments**

- `label`: string, what the edit does, default "Edit"
- `ops`: list of {target, field, before, after}, as edit.apply takes them
- `note`: string, why, shown to the user, default empty

**Returns:** {request, kind, state, note}: the request's id, and pending

**Example**

```json
{"id": "agent.propose", "args": {"label": "Lower ground friction", "ops": [{"target": "ground/ground_collision", "field": "shape.friction_mu", "after": 0.8}]}}
```

## `agent.accept`

**Kind:** [edit](index.md#kinds)

Accepts an agent's proposal: applies its edit. Only the user runs it, never an agent.

**Arguments**

- `id`: int, the proposal

**Returns:** the request as agent.proposals lists it, now accepted, or failed with the edit's error

## `agent.approve`

**Kind:** [app](index.md#kinds)

Runs the action an agent asked to run. With `always` its id joins the access settings' allow list, so it runs without asking from then on. Only the user runs it, never an agent.

**Arguments**

- `request`: int, the request
- `always`: bool, allow this action from now on, default false

**Returns:** the request as agent.proposals lists it, now approved with the action's value, or failed with its error

## `agent.reject`

**Kind:** [app](index.md#kinds)

Turns down an agent's proposal or request; the agent sees it rejected with `reason`. Only the user runs it, never an agent.

**Arguments**

- `id`: int, the request
- `reason`: string, why, default "rejected by the user"

**Returns:** the request as agent.proposals lists it, now rejected

## `agent.disconnect`

**Kind:** [app](index.md#kinds)

Drops every connected agent; each finds its connection closed at its next call. With `off` agent access turns off too, so none can connect again until it is turned back on. Only the user runs it, never an agent.

**Arguments**

- `off`: bool, also turn agent access off, default false

**Returns:** {dropped, mode}: how many were connected, and the access mode now

## `app.command_palette`

**Kind:** [view](index.md#kinds)

Opens or closes the command palette, which finds and runs any action by name.

**Returns:** nothing

## `extensions.manager`

**Kind:** [view](index.md#kinds)

Opens or closes the Extensions panel.

**Returns:** nothing

## `extensions.rescan`

**Kind:** [app](index.md#kinds)

Looks for extensions in the extension folders again; new ones stay disabled until enabled.

**Returns:** how many were found

## `extensions.list`

**Kind:** [read](index.md#kinds)

The extensions found: their state and error, and the actions, panels, overlays and teleop devices and modes each one adds.

**Returns:** a list of {id, title, version, description, kind, state, error, auto\_disabled, dir, actions, panels, overlays, teleop\_devices, teleop\_modes}

## `extensions.enable`

**Kind:** [process](index.md#kinds)

Enables an extension, and what it depends on, which loads and runs its code; the choice is kept in the profile.

**Arguments**

- `id`: string, the extension

**Returns:** the extension, as extensions.list

## `extensions.disable`

**Kind:** [app](index.md#kinds)

Disables an extension, and what depends on it; the choice is kept in the profile.

**Arguments**

- `id`: string, the extension

**Returns:** the extension, as extensions.list

## `extensions.reload`

**Kind:** [process](index.md#kinds)

Re-reads an extension's manifest and restarts it with its code reloaded, and the dependents that went down with it.

**Arguments**

- `id`: string, the extension

**Returns:** the extension, as extensions.list

## `extensions.log`

**Kind:** [read](index.md#kinds)

An extension's log text.

**Arguments**

- `id`: string, the extension

**Returns:** the log text
