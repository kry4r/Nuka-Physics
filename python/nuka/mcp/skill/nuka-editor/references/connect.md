# Connecting to the editor

## The server

The MCP server ships with the editor's Python package `nuka` and with the packaged app. It speaks MCP
over stdio and reaches an editor running on the same machine, for the same user.

| Installed as | Command |
| --- | --- |
| Python package | `nuka-mcp`, or `python -m nuka.mcp` |
| Packaged app (Windows) | `<app folder>\nuka-editor.exe --mcp` |

Options: `--profile DIR` picks the editor profile (default: the user's), `--pid PID` one editor of
several. The environment variables `NUKA_EDITOR_PROFILE` and `NUKA_EDITOR_PID` do the same.

The server answers protocol revision 2026-07-28 statelessly (`server/discover`) and 2025-11-25,
2025-06-18 and 2025-03-26 through `initialize`.

## Client configuration

Claude Code, from the project folder:

```
claude mcp add nuka-editor -- nuka-mcp
```

or in `.mcp.json`:

```json
{"mcpServers": {"nuka-editor": {"command": "nuka-mcp", "args": []}}}
```

With the packaged app, use `"command": "C:/Tools/Nuka Editor/nuka-editor.exe", "args": ["--mcp"]`, with
the app's own folder. Other MCP clients (desktop apps, IDEs) take the same command and arguments in
their stdio server settings.

## This skill

```
python -m nuka.mcp install-skill                 # into ~/.claude/skills/nuka-editor
python -m nuka.mcp install-skill --dest DIR      # another agent's skills folder
```

## The editor side

- An editor listens for agents while its agent access (Agents panel) is not *off*. An editor with
  access off is invisible: `editor_status` lists only the reachable ones.
- `editor_status` lists the running editors, newest first, and connects to the newest (or `--pid`'s).
  `editor_launch {scene}` starts another one in its own window and connects to it.
- A call that fails with "the editor closed the connection" means the editor exited or the user
  disconnected the agents. Stop and ask the user before connecting again; `editor_status` shows whether
  the editor still runs.
- The Agents panel shows the user who is connected, what each agent ran and what waits for them.

## Without MCP

Python programs on the same machine can use the same endpoint directly:

```python
from nuka.editor import connect
with connect(client="my-tool") as editor:
    editor.actions.run("file.new", template="ground", discard=True)
    print(editor.document()["revision"])
```

`editor.actions.run/list/log`, `document()`, `console()`, `screenshot()` and `wait()` match the tools.
