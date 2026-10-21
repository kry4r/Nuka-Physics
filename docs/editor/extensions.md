# Extensions

An extension is a folder holding an `extension.toml`. It adds panels, actions, viewport overlays, teleop
devices and teleop modes, written in Python or in C/C++ against a plain C ABI. Everything an extension
registers belongs to it and goes away when it is disabled or reloaded.

Two samples ship in `examples/editor_extensions/`: `hello_py` and `hello_cpp`. Each adds a panel, an action
and an overlay; `hello_cpp` also paints a panel with Dear ImGui.

## The Extensions panel

**Extensions → Extension Manager** opens the panel (`extensions.manager`). The header counts how many are enabled and
how many failed; its button rescans the folders. One card per extension shows:

- a status dot: green enabled, red failed, grey disabled;
- its title and a badge for its kind: **python**, **native**, **native + python** or **data** (a manifest
  and string tables only);
- **Reload** (not on built-ins) and the power button, which enables or disables it;
- its id and version, its description and, in red, the last error.

Click a card to show that extension's log under the cards. What it logs also reaches the Console.

- Enabling an extension enables what it depends on first. Disabling one disables what depends on it first.
- The choice is kept in the profile (`extensions.toml`), so it holds on the next start.
- **Rescan** (`extensions.rescan`) finds new folders. New extensions stay disabled until you enable them.
- While an extension is enabled, the editor checks its folder once a second. When a file changes, it
  reloads the extension and the dependents that went down with it.
- A callback that fails three times in a row disables its extension. A toast says so, and the card
  shows the reason.

## Where the editor looks

The editor searches these folders in order, so the first folder holding an id wins:

1. each `--ext-folder DIR` given on the command line, so a working copy wins over an installed one;
2. `extensions/` next to `nuka_editor.dll`;
3. `editor/extensions/` beside it, which in a Python install is inside the `nuka.editor` package;
4. `extensions/` in the profile folder (`%APPDATA%\Nuka\editor\extensions` on Windows,
   `~/.config/nuka/editor/extensions` on Linux).

A folder that holds an `extension.toml` is one extension. Otherwise each subfolder that holds one is an
extension. A folder that repeats an id already found is skipped, with a note in the Console.

`--ext ID` enables that extension for this start, on top of the saved choices.

```text
nuka-editor --ext-folder D:\work\my_tools --ext my_tools
```

## The manifest

```toml
[package]
id = "hello_py"            # unique; the folder name by default
version = "1.0.0"
title = "Hello Python"
description = "A sample Python extension: a panel, an action and a viewport overlay."

[dependencies]
robot_tools = "^1.2"       # enabled first

[python]
module = "hello_py"        # imported from this folder

[native]
library = "bin/${platform}/hello_cpp"

[settings]
greeting = "Hello from Python"
```

| Table | Holds |
| --- | --- |
| `[package]` | `id`, `version` (default `0.0.0`), `title` (default the id), `description` |
| `[dependencies]` | `id = "requirement"`: `*`, `1.2.0` or `=1.2.0`, `>=1.2`, or `^1.2` (same major, at least that); a missing one or one out of range fails the chain |
| `[python]` | `module`, the Python module holding one `Extension` subclass |
| `[native]` | `library`, the shared library path relative to the folder; `${platform}` becomes `windows-x64` or `linux-x64`, and `.dll` or `.so` is added |
| `[settings]` | any values, handed to the code as JSON: Python reads `ctx.settings`, native code `host->settings_json` |

An extension may have both `[python]` and `[native]`, or neither. An `i18n/` folder translates its strings
the same way the editor's own tables do: one `<locale>.toml` per language, with a `[meta]` table (`locale`,
`name`, `tag`) and a `[strings]` table mapping English text to the translation.

```toml
# i18n/zh-CN.toml
[meta]
locale = "zh-CN"
name = "简体中文"
tag = "中"

[strings]
"Hello Python" = "你好 Python"
"Greet" = "问候"
```

## Python extensions

The module defines exactly one subclass of `nuka.editor.Extension`. The editor calls `on_startup(ctx)`
when the extension is enabled and `on_shutdown()` when it is disabled or reloaded.

```python
from nuka.editor import Extension


class HelloPy(Extension):
    def on_startup(self, ctx):
        self.ctx = ctx
        self.count = 0
        panel = ctx.ui.panel("hello_py", "Hello Python", dock="right", icon="file_py")
        panel.label("Edit this file and the extension reloads.", tone="muted", wrap=True)
        self.counter = panel.stat("Greetings", "0")
        panel.button("Greet", icon="file_py", primary=True, on_click=self.greet)
        ctx.actions.register("hello_py.greet", "Greet from Python", lambda args: self.greet(),
                             category="tools",
                             doc={"summary": "Shows a toast and counts the greetings.", "kind": "view",
                                  "returns": "{count}"})
        ctx.overlays.register("hello_py.axes", self.draw)

    def greet(self):
        self.count += 1
        self.counter.set(value=str(self.count))
        self.ctx.ui.toast(self.ctx.settings.get("greeting", "hi"), "ok")
        return {"count": self.count}

    def draw(self, frame):
        frame.line((0, 0, 0), (1, 0, 0), 0xFF5A5AFF, 2.5)
        frame.text((1.05, 0, 0), 0xE8ECF0FF, "x")

    def on_shutdown(self):
        self.ctx.log("stopped")
```

`ctx` holds:

| Member | Is |
| --- | --- |
| `ctx.id`, `ctx.path`, `ctx.settings` | the extension's id, its folder, its `[settings]` as a dict |
| `ctx.app` | the running editor, as in a script ([Python: the editor](../python/editor.md)) |
| `ctx.ui` | `panel(id, title, dock, icon)`, `toast(text, level)`, `progress(id, label, fraction)`, `done(id)` |
| `ctx.actions.register(...)` | adds an action (below) |
| `ctx.overlays.register(id, draw)` | `draw(frame)` runs every frame over the viewport |
| `ctx.teleop.register_device(...)` | a teleop input device |
| `ctx.teleop.register_retarget(...)` | a teleop mode |
| `ctx.log(text, level)` | to the extension's log and the Console; `info`, `ok`, `warn` or `error` |

### Panels

`ctx.ui.panel` returns the root of a dockable panel (`"left"`, `"right"`, `"bottom"` or `"float"`), listed
under **Window**. Calling it again with the same id returns the same panel. Children stack vertically,
and each builder returns a node you can `set(...)`, `clear()` or `remove()`:

| Kind | Builders |
| --- | --- |
| Layout | `vstack`, `hstack`, `grid(columns)`, `collapsible(title)`, `tabs` / `tab(title)`, `scroll`, `separator`, `spacer` |
| Text and state | `label`, `stat`, `badge`, `status`, `progress`, `log`, `code` |
| Data | `plot(series)`, `table(columns, rows)`, `tree(items, on_select)`, `property_grid(schema)` |
| Input | `button`, `icon_button`, `toggle`, `segmented`, `combo`, `float_field`, `int_field`, `bool_field`, `vec3_field`, `color_field`, `text_field`, `path_field` |

Inputs take `on_click`, `on_change` or `on_commit` callbacks; `node.value` reads the current value. The same
panel API is open to scripts as `app.ui`, see [Scripting](scripting.md).

### Actions

```python
ctx.actions.register(action_id, label, fn, category=None, shortcut=None, icon=None, hidden=False, doc=None)
```

`fn(args)` gets the arguments as a dict (or `fn()` takes none) and returns any JSON value; raising fails
the call with the message. `ctx.actions.run(id, **args)` runs any other editor action.

An action shows in its `category`'s menu (**Extensions** by default) and in the command palette, unless
it is `hidden` (an action that needs arguments). Scripts, the RPC pipe and [agents](agents.md) can call it
like any built-in action.

`doc` takes the shape of an `actions.toml` entry: `summary`, `kind`, `args`, `returns`, `example`. Agents
read it from the catalog, and [access modes](agents.md#access-modes) use `kind`. An action without a
`kind` counts as `process`: in Propose mode it waits for your approval, in Read mode it is refused. See
[action kinds](../reference/actions/index.md#kinds).

### Overlays

`draw(frame)` runs once a frame over the scene viewport, with world-space points:

| Call | Draws |
| --- | --- |
| `frame.line(a, b, rgba, thickness)` | a segment |
| `frame.circle(center, radius_px, rgba, thickness)` | a screen-size ring |
| `frame.text(at, rgba, text)` | a label |

Colours are `0xRRGGBBAA`; thickness defaults to 1.5 px. `frame.tick` counts frames, `frame.view` and
`frame.proj` hold the camera (column-major) and `frame.viewport` is x, y, width, height in window pixels.
A `draw` that raises or returns `False` counts as a failure.

### Teleop devices and modes

A device feeds the [Teleop](motion.md#teleop) panel; a mode turns its inputs into joint targets.

```python
def read():
    return [x, y, z, rx, ry, rz], [left, right]   # axes in [-1, 1], buttons true/false

ctx.teleop.register_device("spacemouse", "SpaceMouse", read,
                           axes=("x", "y", "z", "rx", "ry", "rz"), buttons=("left", "right"),
                           kind="spacemouse")

def retarget(state):
    # state: {"device": {"id", "axes", "buttons"}, "quantities", "joints": [{"name", "q", "target"}], "dt"}
    return {"left_elbow_joint": state["device"]["axes"]["z"]}

ctx.teleop.register_retarget("elbow_only", "Elbow Only", retarget)
```

The device is read once a frame while it is the active one. The Teleop mapping binds its axes and buttons
by name.

## Native extensions

A native extension is a shared library exporting the functions in `nuka/nuka_editor_plugin.h`. The ABI is
plain C with sized, append-only structs, so a plugin built with another compiler or runtime still loads.
`nuka/editor/plugin.hpp` is a header-only C++ layer over it, without exceptions or RTTI:

```cpp
#include "nuka/editor/plugin.hpp"

class Hello : public nuka::editor::Plugin {
public:
    bool OnStartup(nuka::editor::Host& host) override {
        const uint32_t root = host.Panel("hello_cpp", "Hello C++", "right");
        const uint32_t greet = host.Add(root, "button", R"({"label":"Greet","primary":true})");
        host.OnEvent(greet, [this](uint32_t, const std::string&) { ++count_; });
        host.AddAction({"hello_cpp.greet", "Greet from C++", "tools", nullptr, nullptr, false,
                        R"({"summary": "Counts the greetings.", "kind": "view", "returns": "{count}"})"},
                       [this](const std::string&) {
                           return nuka::editor::Reply::Value("{\"count\":" + std::to_string(++count_) + "}");
                       });
        return true;
    }

private:
    int count_ = 0;
};

NUKA_EDITOR_PLUGIN(Hello)
```

Build it with the CMake helper; it lays the library out as `bin/<platform>/` next to a copy of
`extension.toml` and `i18n/`:

```cmake
nuka_add_editor_extension(hello_cpp SOURCES hello_cpp.cpp IMGUI)
```

The C exports are:

| Export | Does |
| --- | --- |
| `nuka_editor_plugin_abi()` | returns `NUKA_EDITOR_PLUGIN_ABI` |
| `nuka_editor_plugin_load(host, &state)` | registers what the plugin adds through the host table |
| `nuka_editor_plugin_unload(state)` | frees the plugin's state |
| `nuka_editor_plugin_imgui(&check)` | optional: the Dear ImGui build it uses, for `register_imgui_panel` |

The host table (`nuka_editor_host_api_t`) carries the extension's id, folder and settings JSON, the NUI
panel API, and `register_action`, `register_overlay`, `register_teleop_device`, `register_teleop_retarget`,
`run_action` (any editor action) and `log`.

- The editor loads a copy of the library, so you can rebuild it while the editor runs; the change
  reloads the extension.
- Callbacks run on the UI thread.
- A plugin that paints with Dear ImGui must match the editor's ImGui version and struct sizes. If it
  doesn't, the editor refuses it and the card says which version it was built against.

## From scripts and agents

| Action | Does |
| --- | --- |
| `extensions.list` | every extension with its state, error and what it adds |
| `extensions.enable {id}` / `extensions.disable {id}` | as the power button |
| `extensions.reload {id}` | re-reads the manifest and restarts it |
| `extensions.log {id}` | its log text |
| `extensions.rescan` | looks for new folders |

```python
app.actions.run("extensions.enable", id="hello_py")
app.actions.run("hello_py.greet")
print(app.actions.run("extensions.log", id="hello_py"))
```
