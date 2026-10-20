# Access, proposals and undo

The editor belongs to its user. They decide what agents may do, see every call in the Agents panel and
can disconnect agents at any time.

## Access modes

`document_get` reports the mode as `agent.mode`. Every action has a kind, shown in
[actions.md](actions.md) and by `action_help`, and the mode decides per kind:

| Kind | full | propose | read |
| --- | --- | --- | --- |
| read, view | runs | runs | runs |
| sim | runs | runs | refused |
| edit | runs | becomes a proposal | refused |
| file, process, app | runs | waits for approval | refused |

- *off*: no agent can connect.
- The user's allow list runs actions without asking; the deny list refuses them in every mode. Both take
  ids and wildcards (`file.*`).
- Some actions are the user's alone, whatever the mode: `agent.mode`, `agent.accept`, `agent.approve`,
  `agent.reject` and `agent.disconnect`. You cannot widen your own access; ask the user.
- `screenshot`, `document_get`, `scene_tree`, `console_read` and `action_log` only observe and work in
  every mode but off.

## Waiting for the user

In propose mode a call to an `edit` action returns `{request, kind: "proposal", state: "pending"}`, and
a call to a file, process or app action returns `{request, kind: "approval", state: "pending"}`. The
user sees the request in the Agents panel; an approval also opens a dialog.

- `proposal_get {id: request, wait_s: 30}` reports the state: `pending`, `accepted` or `approved` (with
  the action's `value`), `rejected` (with the user's reason in `error`), or `failed`.
- Tell the user what you are waiting for and why. Do not repeat a request the user rejected.
- `python_eval` returns `{pending: request}` when it needs approval; once approved, the request's
  `value` holds `{seq}` for `action_run script.result {seq}`.

## Proposing instead of doing

`edit_propose {label, ops, note}` offers an edit in any mode but read, even in full. The user sees each
field's value before and after and accepts or rejects it. Use it when the user wants to review a change
first, or when you are unsure. On acceptance it applies as one undo step named `Agent: <label>`.

## Undo

- Every edit, yours or the user's, is a step in one history: `edit.undo`, `edit.redo`.
- Label edits (`edit.apply {label}`) so the history reads well.
- Clip edits have their own history: `clip.undo`, `clip.redo`.
- Files written, scripts run and processes started cannot be undone. Say what you are about to write
  or start before you do it.

## Records

Every agent call goes into the editor's action log (`action_log`, source `agent:<client>`) and into the
profile's audit file `logs/agent-<date>.jsonl`, with arguments, outcome and time.
