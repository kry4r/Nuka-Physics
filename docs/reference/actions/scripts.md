<!-- Generated from assets/editor/actions.toml by tools/wiki_actions.py; edit those instead. -->

# Scripts and console

18 actions. Run one from the command palette (++ctrl+p++), from Python with `app.actions.run(id, **args)` or through MCP with `action_run`.

## `script.new`

**Kind:** [view](index.md#kinds)

Opens a new untitled script tab holding `text`.

**Arguments**

- `text`: string, the tab's text, default empty

**Returns:** {tab, title, path}

## `script.open`

**Kind:** [file](index.md#kinds)

Opens a Python file in a script tab, at `line` if given; without `path` a file dialog asks.

**Arguments**

- `path`: string, the .py file, default: a file dialog
- `line`: int, the line to show, optional

**Returns:** {tab, title, path}

## `script.save`

**Kind:** [file](index.md#kinds)

Saves a script tab (`tab`, else the active one) to `path`, else its own file, else where a file dialog says; with Rerun on Save it then runs.

**Arguments**

- `tab`: int, the tab, default the active one
- `path`: string, the file to save to, default the tab's

**Returns:** {tab, title, path}

## `script.close`

**Kind:** [view](index.md#kinds)

Closes a script tab (`tab`, else the active one); one with unsaved edits needs `discard`.

**Arguments**

- `tab`: int, the tab, default the active one
- `discard`: bool, drop unsaved edits, default false

**Returns:** nothing

## `script.run`

**Kind:** [process](index.md#kinds)

Runs Python in the editor, where `app`, `world`, `nuka` and `np` are bound: `code`, else the file at `path`, else a tab (`tab`, else the active one), or with `selection` that tab's selection or caret line, printing a final expression's value. Answers at once; script.result reports the run by its seq and script.console holds its output.

**Arguments**

- `code`: string, Python source, optional
- `name`: string, the name tracebacks give `code`, default "&lt;run&gt;"
- `path`: string, a .py file to run, optional
- `tab`: int, the tab to run, default the active one
- `selection`: bool, run only the tab's selection or caret line, default false

**Returns:** {seq, name}

**Example**

```json
{"id": "script.run", "args": {"code": "print(len(app.actions.run(\"scene.nodes\")))"}}
```

## `script.eval`

**Kind:** [process](index.md#kinds)

Runs one console line: echoes it, then runs it and prints the value of a final expression.

**Arguments**

- `code`: string, Python source

**Returns:** {seq}

## `script.stop`

**Kind:** [process](index.md#kinds)

Interrupts the running script; enabled while one runs.

**Returns:** {stopped}

## `script.result`

**Kind:** [read](index.md#kinds)

A script run's state by the seq script.run or script.eval returned: whether it is done or failed, and its output.

**Arguments**

- `seq`: int, the run

**Returns:** {done, failed, output, name}

## `script.complete`

**Kind:** [read](index.md#kinds)

Completions for Python `code` at a 1-based line and 0-based column.

**Arguments**

- `code`: string, Python source
- `line`: int, 1-based, default 1
- `column`: int, 0-based, default 0

**Returns:** a list of {name, type, signature}

## `script.tabs`

**Kind:** [read](index.md#kinds)

The open script tabs: their files, the /script node each came from, whether they are dirty or changed on disk, the caret and the error markers.

**Returns:** a list of {tab, title, name, path, node, active, dirty, disk\_changed, line, column, lines, markers: \[{line, message}\]}

## `script.goto`

**Kind:** [view](index.md#kinds)

Shows a file at a line in the Script panel: an open tab of that name, else the file opened in a new one.

**Arguments**

- `path`: string, the tab's name or the file
- `line`: int, 1-based, default 1

**Returns:** {tab, title, path}

## `script.console`

**Kind:** [read](index.md#kinds)

Console text from byte `since` on (0: all that is kept); `next` is the `since` of the following call and `dropped` says text before `since` is gone.

**Arguments**

- `since`: int, the byte to read from, default 0

**Returns:** {text, next, dropped}

## `script.clear_console`

**Kind:** [view](index.md#kinds)

Clears the console.

**Returns:** nothing

## `script.reload`

**Kind:** [file](index.md#kinds)

A script tab (`tab`, else the active one) whose file changed on disk reloads it, or with `keep` keeps its own text.

**Arguments**

- `tab`: int, the tab, default the active one
- `keep`: bool, keep the tab's text, default false

**Returns:** {tab, title, path}

## `script.toggle_rerun`

**Kind:** [app](index.md#kinds)

Rerun on Save: whether saving a script tab runs it; `on` sets it, else it toggles. Kept in the profile.

**Arguments**

- `on`: bool, rerun on save, default: toggles

**Returns:** whether it is on

## `script.set_budget`

**Kind:** [app](index.md#kinds)

Seconds a script may run before it is stopped, 0 for no limit; kept in the profile. Without `seconds` it reports.

**Arguments**

- `seconds`: float, the budget, 0 for none, default kept

**Returns:** the budget in seconds

## `script.attach`

**Kind:** [edit](index.md#kinds)

Keeps a script tab (`tab`, else the active one), or `source`, in the scene as a /script node, as one undoable edit: the node the tab came from, else a selected script node, else a new one.

**Arguments**

- `tab`: int, the tab, default the active one
- `source`: string, Python source instead of the tab's text, optional

**Returns:** {focus}

## `script.load`

**Kind:** [view](index.md#kinds)

Opens a /script node's source in a script tab: `path`, else the selection.

**Arguments**

- `path`: string, the /script node, default the selection

**Returns:** {tab, title, path}
