# Command line

## The editor

```text
nuka-editor [scene] [options]
python -m nuka.editor [scene] [options]
```

`scene` is an `.nks` to open, or a URDF, MJCF, USD or mesh file to import into a new document. With none,
the editor shows the start page. `--help` prints the options.

A scene given while an editor already runs on the same profile opens in that editor, and the new process
exits. `--new-window` starts a separate editor instead.

### Starting up

| Option | Does |
| --- | --- |
| `--template empty\|ground` | Starts a new scene from a template |
| `--empty` | Opens a blank workspace instead of the start page |
| `--layout authoring\|debug\|robot\|data\|training` | Picks the panel [layout](../editor/index.md#layouts) |
| `--locale en\|zh-CN` | The UI language |
| `--ui-scale X` | The UI scale |
| `--window WxH` | The window size, 1280x720 by default |
| `--profile DIR` | The folder for settings, recent scenes, logs and autosaves ([The profile folder](../editor/install.md#the-profile-folder)) |
| `--log FILE` | Writes the session log there |
| `--console` | Shows a console window for the application's output |
| `--new-window` | Starts a new editor even when one runs on the profile |

### The scene and the simulation

| Option | Does |
| --- | --- |
| `--play` | Starts playing once the scene loads |
| `--dt SECONDS` | The physics step |
| `--speed X` | The transport speed |
| `--policy FILE` | Attaches a trained locomotion policy (`.nkml` or `.bin`) to each load |
| `--host-policy` | Runs the policy's network on the CPU |
| `--cam tx,ty,tz,dist,yaw,pitch` | Places the camera (degrees) instead of framing the scene |
| `--cams FILE.json` | Loads camera bookmarks for the scene |
| `--show-colliders`, `--show-contacts` | Turns on the collider and contact overlays |
| `--publish auto\|host\|zero-copy` | How poses reach the renderer |

### Extensions and scripts

| Option | Does |
| --- | --- |
| `--ext ID` | Enables an extension for this start ([Extensions](../editor/extensions.md)) |
| `--ext-folder DIR` | Adds a folder to search for extensions, ahead of the installed ones |
| `--run-script FILE.py` | Runs a script once against the loaded scene |

### Automation and capture

| Option | Does |
| --- | --- |
| `--offscreen` | Renders to a display-less surface: no window |
| `--frames N` | Quits after N frames |
| `--capture-frame N --capture-out FILE.ppm` | Saves frame N as an image |
| `--spawn box\|sphere\|capsule\|cylinder\|plane` | Adds a primitive body after the load |
| `--reset-at N` | Resets the simulation at frame N |
| `--select PATH` | Selects a Stage node after the load |
| `--perturb FX,FY,FZ` | Holds a world force (N) on the `--select`ed link or body |
| `--teleop-cmd VX,VY,WYAW` | Feeds a velocity command to a policy |
| `--teleop-nudge DOF,DELTA` | Holds the robot's pose and nudges one drive target |
| `--undo`, `--redo` | Fires one undo or redo after a headless load edit |

### Python launcher options

These belong to `python -m nuka.editor` and the packaged app:

| Option | Does |
| --- | --- |
| `--headless SCENE [--dt S]` | Steps the scene without the editor, Vulkan or a window |
| `--steps N` | Stops after N updates |
| `--mcp [--profile DIR] [--pid PID]` | Runs the MCP server on stdio instead of the editor ([Agents](../editor/agents.md)) |

```bat
python -m nuka.editor examples\scenes\go2.nks --layout robot --play
python -m nuka.editor scene.nks --offscreen --frames 300 --capture-frame 300 --capture-out shot.ppm
python -m nuka.editor --headless scene.nks --steps 6000
```

## Other tools

| Command | Does |
| --- | --- |
| `python -m nuka.mcp [--profile DIR] [--pid PID]` (or `nuka-mcp`) | The MCP server; it connects to the newest editor on the profile, or the one with that pid |
| `python -m nuka.mcp install-skill [--dest DIR]` | Installs the `nuka-editor` agent skill |
| `python -m nuka.train FOLDER/job.json` | Runs a training job the editor configured ([Training](../editor/training.md#run-folders)) |
| `python tools/wiki_actions.py [--check]` | Regenerates the [Actions reference](actions/index.md) from `assets/editor/actions.toml` |
