# Scripting

The editor runs Python inside itself. Every menu item, button and panel is an
[action](../reference/actions/index.md), so a script can do anything the UI does.

## The Script panel

The Script panel is a code editor with tabs.

| Button | Does | Key |
| --- | --- | --- |
| **Run** | Runs the active tab | ++f5++ |
| Run selection | Runs the selection, or the line under the caret | ++ctrl+enter++ |
| Stop | Interrupts the running script | ++shift+f5++ |
| New, Open, Save | A new tab, a `.py` file, save the tab | ++ctrl+s++ saves |
| Attach | Keeps the tab in the scene as a `/script` node (one undoable edit) | |
| Load | Opens the selected `/script` node in a tab | |
| Rerun on Save | Saving a tab also runs it | |

The keys act while the code has the keyboard; elsewhere ++f5++ and ++shift+f5++ are global shortcuts for
Run and Stop. The status beside the buttons shows how long a script has been running, or how many
script nodes the scene holds.

- Tabs can be reordered; **+** opens a new one. A dot marks unsaved edits, and closing such a tab asks to
  **Save**, **Discard** or **Cancel**.
- When a tab's file changes on disk, a bar offers **Reload** or **Keep mine**.
- Typing completes names (Jedi, against the console's namespace).
- Errors mark their lines; a traceback's file links in the console jump to the line.

A script that runs longer than its time budget is stopped: 10 s by default, set with
`script.set_budget {seconds}` (0 for no limit) and kept in the profile.

## The Console

The Console shows what scripts print and what the editor logs.

- **All**, **Info**, **Warn** and **Error** filter by level; the filter box by text; the trash button
  clears it.
- Click a file link (`file.py:12`) to open that line in the Script panel.
- The `>>>` prompt at the bottom runs one Python line: ++enter++ runs it, ++tab++ completes, ++up++ and
  ++down++ walk the history. A final expression's value prints, and `_` holds it.

The console and the script tabs share one namespace, so a name defined in one is there in the other.

## What a script sees

| Name | What it is |
| --- | --- |
| `app` | The editor: `app.actions`, `app.ui`, `app.document()`, `app.on_tick` ([Python: the editor](../python/editor.md)) |
| `world` | The world on screen, the [World API](../python/index.md) |
| `ui` | `app.ui`, for panels and toasts |
| `nuka`, `np` | The `nuka` package and NumPy |
| `sel` | The selected node's path when a console run starts |
| `action(id, **args)` | Runs an action and returns its value; raises `ActionError` when it fails |
| `nodes(under="", kind="")` | The scene's nodes as `{path, name, kind, pose}` |
| `find(name)` | The path of the first node with that path or name |
| `select(*paths)` | Makes them the selection; no path clears it |
| `move(path, pos, quat)` | Places a node, as one undoable edit |
| `log(*parts, level="info")` | Prints at a level: info, warn or error |

```python
for node in nodes(kind="body"):
    print(node["path"], node["pose"]["pos"])
select(find("box"))
action("view.frame_selected")
```

Code runs on the editor's physics thread, between steps, so `world` holds still while a script reads it.

## Script nodes

A `/script` node keeps Python in the scene, saved with the document. Each node runs in its own
namespace when the scene loads and may define:

| Callback | Runs |
| --- | --- |
| `on_ready()` | Once, after the scene is ready |
| `on_step()` | Every physics step |
| `on_reset()` | Every reset |

A callback that fails 3 times in a row is turned off, with a line in the console.

```python
# A /script node: push the box up whenever it falls below 0.2 m.
def on_step():
    z = nodes(under="/World/box")[0]["pose"]["pos"][2]
    if z < 0.2:
        action("sim.perturb", path="/World/box", force=[0, 0, 40])
    else:
        action("sim.release", path="/World/box")
```

Make one with **Attach** in the Script panel, or `script.attach {source}`; edit it again with **Load**.

## Panels from scripts

`app.ui` builds panels in the editor's own style:

```python
panel = app.ui.panel("tools", "Robot Tools", dock="right", icon="robot")
gain = panel.float_field("Gain", value=1.0, min=0.0, max=4.0, slider=True,
                         on_change=lambda v: print("gain", v))
row = panel.hstack()
row.button("Reset", icon="reset", on_click=lambda: app.actions.run("sim.reset"))
app.ui.toast("Tools ready", "ok")
```

Widgets include labels, stats, badges, progress bars, plots, tables, trees, logs, code boxes, buttons,
toggles, segmented controls, combos, number, vector, color, text and path fields, inside stacks, grids,
collapsible sections, tabs and scroll areas. Callbacks run at the top of the next frame; what they print
goes to the console. A panel stays until removed.

## Running scripts from outside

- `nuka-editor.exe scene.nks --run-script tool.py` runs a file once against the loaded scene.
- `script.run {code}` or `{path}` runs code from an [agent](agents.md) or another process;
  `script.result {seq}` reports it and `script.console` reads the output.
- `python -m nuka.editor` and `nuka.editor.launch()` run the editor from your own Python, where the
  script is the host ([Python: the editor](../python/editor.md)).
