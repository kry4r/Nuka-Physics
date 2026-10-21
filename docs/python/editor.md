# Python: driving the editor

`nuka.editor` opens the editor from Python, over a world your script owns or one the editor loads. The
same objects serve the editor's own [Script panel](../editor/scripting.md), where `app` is already
defined, and a separate process can drive an editor that is already open.

## Launching

```bash
python -m nuka.editor                               # the start page
python -m nuka.editor scene.nks                     # a scene in a window
python -m nuka.editor scene.nks --offscreen --frames 60
python -m nuka.editor scene.nks --headless --steps 600
python -m nuka.editor --mcp                         # an MCP server on stdio
```

The packaged app's `nuka-editor` takes the same arguments. A scene given while an editor already runs on
the same profile opens in that editor; `--new-window` starts another one. See
[Command line](../reference/command-line.md) for every option.

From a script, over a world it built itself:

```python
import argparse
import nuka
import nuka.editor

parser = argparse.ArgumentParser()
nuka.editor.add_launcher_args(parser)           # --headless and --steps
opts, rest = parser.parse_known_args()

with nuka.Device.create(0) as device:
    world = nuka.World.create_from_scene(device, "examples/scenes/go2_locomotion.usda", env_count=16)
    app = nuka.editor.launch(rest, world=world, headless=opts.headless)

    @app.on_tick
    def control(app):
        ...                                     # read and write world buffers here

    app.run(opts.steps)
    app.close()
    world.destroy()
```

`launch(args, world=None, scene=None, dt=None, headless=False)` returns an `EditorApp`, or with
`headless=True` a `HeadlessApp`, which steps the same world with no window, Vulkan or editor library.
Both offer the loop below, so one script runs on a desktop and on a training server alike.

## The loop

| Member | `EditorApp` | `HeadlessApp` |
| --- | --- | --- |
| `run(steps=None)` | Frames with the transport until the window closes, or `steps` frames | Steps until stopped, or `steps` updates |
| `update(step=False)` | One frame; `step=True` lets the transport advance physics | `step=True` advances one step |
| `on_tick(fn)` / `off_tick(fn)` | `fn(app)` at each tick boundary; usable as a decorator | `fn(app)` after each update |
| `stop()` | Ends `run()` after this frame; safe inside scripts | the same |
| `running`, `playing`, `dt`, `world` | State of the app and the world on screen | the same |
| `console(text)` | Writes to the Console | prints |
| `close()` | Destroys the editor, and the world if it made one | the same |

A world you pass in stays yours. Without one, the editor creates its own at the first scene load, and
`app.world` borrows it until `close()`. `stop_physics()` joins the editor's physics thread, so the world
is yours to touch until the next `update(True)`.

## What an `EditorApp` holds

| Member | Is |
| --- | --- |
| `app.actions` | Every editor command: `run(id, **args)`, `list()`, `log(since)` |
| `app.edit(*ops, label=...)` | One undoable transaction of field edits |
| `app.undo()`, `app.redo()` | The history |
| `app.document()` | Revision, dirty flag, hash, undo depths, tick, timeline branches, selection, last error |
| `app.ui` | Panels in the editor shell (see [Panels](../editor/extensions.md#panels)) |
| `app.teleop` | `devices()`, `use(...)`, `input(axes, buttons)`, `map(...)`, `pad(...)`, `status()` |
| `app.policy` | `load(path, host=False)`, `unload()`, `enable(on)`, `status()` |
| `app.demos` | `record(on, all)`, `episodes()`, `play(episode)`, `save(path)`, `load(path)`, `discard()` |
| `app.clips` | `new`, `key`, `key_pose`, `sample`, `play`, `save`, `load` and the rest of the Clip Editor |

### Actions

```python
app.actions.run("file.open", path="examples/scenes/go2.nks")
app.actions.run("sim.step", steps=240)
for a in app.actions.list():
    print(a["id"], a["shortcut"], a["summary"])
```

`run` returns the action's value and raises `nuka.editor.ActionError` with the editor's reason when it
fails. Every action is listed, with its arguments and an example, in the
[Actions reference](../reference/actions/index.md). `log(since)` returns the runs recorded after a
sequence number: which action, where it came from (the UI, a script, `agent:<client>`), its arguments and
its outcome.

### Edits

```python
app.edit({"target": "/World/box", "field": "transform", "after": {"pos": [0, 0, 1], "quat": [1, 0, 0, 0]}},
         label="Lift")
```

Each op names a Stage path (`""` for the world), a field and its new value; the whole call is one step in
the history. Fields include `transform`, `name`, `static`, `drive`, `initial_position`, `material` and any
live physics parameter by name; `edit.apply` in the [Edits reference](../reference/actions/edits.md)
lists them. `edit` returns `False` when an op failed, and `document()["last_error"]` says why.

### Script helpers

Inside the editor, scripts also see these shortcuts:

| Helper | Does |
| --- | --- |
| `action(id, **args)` | `app.actions.run` |
| `nodes(under="", kind="")` | The Stage nodes under a path, optionally of one kind |
| `find(name)` | The path of the first node with that name |
| `select(*paths)` | Selects nodes |
| `move(path, pos, quat=(1, 0, 0, 0))` | One undoable move |
| `log(*parts, level="info")` | To the Console |

## An editor that is already open

`nuka.editor.remote` reaches an editor running in another process on this machine:

```python
from nuka.editor import connect, editors, spawn

print(editors())                 # pid, address and start time of each, newest first
ed = connect()                   # the newest; connect(pid=1234) for one in particular
ed.actions.run("file.import", path="robot.urdf", mode="add")
ed.actions.run("train.start", name="overnight")
done = ed.wait("train.status", until={"state": "done"}, timeout_s=3600)   # {met, value, polls}
print(ed.console()["text"])
ed.screenshot("frame.ppm")       # the next frame as a PPM, and where the viewport sits in it
ed.close()

ed = spawn("scene.nks")          # starts a new editor window and connects to it
```

A connection counts as an agent: the editor's [access mode](../editor/agents.md#access-modes) decides
what it may run, and it shows in the Agents panel. `connect` and `spawn` take `client="name"`, the name
the panel shows, and `with connect() as ed:` closes the connection for you.

`RemoteEditor` offers `actions`, `document()`, `console(since)`, `screenshot(path)`,
`wait(action, until, ...)` and `call(op, ...)`; failures raise `nuka.editor.RemoteError`.
