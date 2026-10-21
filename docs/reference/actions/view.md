<!-- Generated from assets/editor/actions.toml by tools/wiki_actions.py; edit those instead. -->

# View and camera

47 actions. Run one from the command palette (++ctrl+p++), from Python with `app.actions.run(id, **args)` or through MCP with `action_run`.

## `view.frame_all`

**Kind:** [view](index.md#kinds)

Points the camera so the whole shown world fits the viewport.

**Returns:** nothing

## `view.frame_selected`

**Kind:** [view](index.md#kinds)

Points the camera at the selection with everything under it.

**Returns:** nothing

## `view.reset_camera`

**Kind:** [view](index.md#kinds)

Puts the camera back at its default view of the world.

**Returns:** nothing

## `view.colliders`

**Kind:** [view](index.md#kinds)

Draws collider proxies over the scene; `on` sets it, else it toggles.

**Arguments**

- `on`: bool, show them, default: toggles

**Returns:** nothing

## `view.contacts`

**Kind:** [view](index.md#kinds)

Draws contact points and normals; `on` sets it, else it toggles.

**Arguments**

- `on`: bool, show them, default: toggles

**Returns:** nothing

## `view.hide_overlays`

**Kind:** [view](index.md#kinds)

Turns off the collider and contact overlays.

**Returns:** nothing

## `view.shadows`

**Kind:** [view](index.md#kinds)

Casts the sun's shadows in the Lit view; `on` sets it, else it toggles.

**Arguments**

- `on`: bool, cast them, default: toggles

**Returns:** nothing

## `view.fog`

**Kind:** [view](index.md#kinds)

Fades distant geometry into the sky in the Lit view; `on` sets it, else it toggles.

**Arguments**

- `on`: bool, fog on, default: toggles

**Returns:** nothing

## `view.ground_grid`

**Kind:** [view](index.md#kinds)

The Lit view's ground grid: mode auto (under scenes without a floor), on or off; a bare call flips what is shown.

**Arguments**

- `mode`: string, auto, on or off, default: flips what is shown

**Returns:** {mode, shown}

## `view.screenshot`

**Kind:** [file](index.md#kinds)

Writes the next drawn frame, the whole window, as a PPM.

**Arguments**

- `path`: string, the .ppm to write, default "nuka\_screenshot.ppm"

**Returns:** {path}

## `view.record`

**Kind:** [file](index.md#kinds)

Records the whole window, the UI included, into an mp4: each presented frame goes to ffmpeg from the PATH and stays on until the next one is due, so the video keeps wall-clock time. Turning it off waits for ffmpeg to finish the file. Fails on a surface that cannot be read back.

**Arguments**

- `on`: bool, start or stop, default: toggles
- `file`: string, the .mp4 to write, default recordings/nuka-&lt;time&gt;.mp4 in the editor's profile
- `fps`: number, video frames a second, 1 to 120, default 30
- `codec`: string, nvenc (NVIDIA hardware) or x264, default "nvenc"
- `quality`: int, constant quality 0 to 51, lower is better, default 18

**Returns:** {on, file} when it starts; {on: false, file, frames, dropped, seconds, width, height} when it stops

**Example**

```json
{"id": "view.record", "args": {"on": true, "file": "C:/videos/walk.mp4", "fps": 30}}
```

## `view.record_status`

**Kind:** [read](index.md#kinds)

The recording in progress, or the last one: its file, the video frames written so far, presented frames dropped, seconds of video and the frame size.

**Returns:** {on, file, frames, dropped, seconds, width, height, error}; error only when ffmpeg failed

## `view.project`

**Kind:** [read](index.md#kinds)

Where a world point shows in the window, in the window points ui.pointer takes, and whether it falls inside the viewport.

**Arguments**

- `point`: \[x, y, z\], the world point in metres

**Returns:** {x, y, inside}

**Example**

```json
{"id": "view.project", "args": {"point": [0.3, -0.2, 0.9]}}
```

## `view.stats`

**Kind:** [read](index.md#kinds)

The editor's frame timings: frame count, frames per second, frame and UI time percentiles, the physics tick, how snapshots reach the renderer, whether it plays, the view mode and the viewport's pixel rectangle.

**Returns:** {frame, measured, fps, frame\_ms\_p50, frame\_ms\_p95, ui\_ms\_p50, ui\_ms\_p95, age\_ms\_p95, tick, snapshot\_drops, publish, playing, view\_mode, viewport: {x, y, width, height}}

## `view.mode`

**Kind:** [view](index.md#kinds)

The shading the viewport draws with; without a mode it reports the current one.

**Arguments**

- `mode`: string, lit, unlit, normals, material, residual, contact or wireframe, default the current one

**Returns:** {mode}

## `view.layout`

**Kind:** [view](index.md#kinds)

The panes: one, two side by side or four; without a layout it reports the current one.

**Arguments**

- `layout`: string, single, split or quad, default the current one

**Returns:** {layout, active}

## `view.layout_single`

**Kind:** [view](index.md#kinds)

Shows one viewport pane.

**Returns:** {layout, active}

## `view.layout_split`

**Kind:** [view](index.md#kinds)

Shows two viewport panes side by side.

**Returns:** {layout, active}

## `view.layout_quad`

**Kind:** [view](index.md#kinds)

Shows four viewport panes.

**Returns:** {layout, active}

## `view.maximize`

**Kind:** [view](index.md#kinds)

Shows one pane alone, the active one without an index, and goes back to the layout it left.

**Arguments**

- `index`: int, the pane to show alone, default the active one

**Returns:** {layout, active}

## `view.view`

**Kind:** [view](index.md#kinds)

A pane's mode and scene camera ("" is free), each kept when left out; activate makes it the active pane.

**Arguments**

- `index`: int, the pane, 0 to 3, default the active one
- `mode`: string, lit, unlit, normals, material, residual, contact or wireframe, default the pane's
- `camera`: string, a scene camera's Stage path, "" for the free camera, default the pane's
- `activate`: bool, make it the active pane, default false

**Returns:** the pane: {index, mode, camera, active, shown, rect}

## `view.views`

**Kind:** [read](index.md#kinds)

The viewport panes: the layout, the active pane, whether one is maximized, the scene cameras and each pane's mode and camera.

**Returns:** {layout, active, maximized, cameras, panes}

## `view.pick`

**Kind:** [view](index.md#kinds)

What the viewport shows at x, y: fractions of its width and height from the top left; `select` clicks there, making the environment it hits current and selecting the body.

**Arguments**

- `x`: number, 0 at the left edge to 1 at the right, default 0.5
- `y`: number, 0 at the top to 1 at the bottom, default 0.5
- `select`: bool, select what is hit, default false

**Returns:** {path, node, instance, env}; path is null over the background

## `view.envs`

**Kind:** [view](index.md#kinds)

The environment grid: `show` cells of the current environment's page (up to 16), `current` makes one current, `spacing` sets the metres between cells (0 sizes them from the robot).

**Arguments**

- `show`: int, how many environments to draw, 1 to 16, default the current number
- `current`: int, the environment the editor shows and edits, default the current one
- `spacing`: number, metres between cells, 0 sizes them from the robot, default the current spacing

**Returns:** {count, current, show, spacing, auto\_spacing, cells: \[{env, offset}\], copies}

## `view.camera`

**Kind:** [view](index.md#kinds)

The active pane's orbit camera: `target`, `distance` and `yaw`, `pitch` in degrees, each kept when left out.

**Arguments**

- `target`: \[x, y, z\], the point it orbits, default the current target
- `distance`: number, metres from the target, default the current distance
- `yaw`: number, degrees about z, default the current yaw
- `pitch`: number, degrees above the horizon, default the current pitch

**Returns:** {target, eye, distance, yaw, pitch}

**Example**

```json
{"id": "view.camera", "args": {"target": [0, 0, 0.4], "distance": 3, "yaw": 135, "pitch": 25}}
```

## `view.select_rect`

**Kind:** [view](index.md#kinds)

Selects what the viewport shows inside x0, y0 to x1, y1 in the same fractions as view.pick; add keeps the selection.

**Arguments**

- `x0`: number, the left edge, default 0
- `y0`: number, the top edge, default 0
- `x1`: number, the right edge, default 1
- `y1`: number, the bottom edge, default 1
- `mode`: string, replace or add, default "replace"

**Returns:** {path, selection}

## `view.rt_preview`

**Kind:** [view](index.md#kinds)

Path-traced frames of the view over the raster one, refined up to max\_spp samples while nothing moves; `on` sets it, else it toggles.

**Arguments**

- `on`: bool, preview on, default: toggles
- `max_spp`: int, the samples per pixel it stops at, default the current limit

**Returns:** {on, max\_spp}

## `view.rt_status`

**Kind:** [read](index.md#kinds)

How far the RT preview has refined: samples so far, whether it converged and is shown, pending work, its size and any error.

**Returns:** {on, spp, max\_spp, converged, shown, pending, done, width, height, error}

## `view.rt_save`

**Kind:** [file](index.md#kinds)

The RT image the viewport shows, as a PNG; the dialog asks without `path`.

**Arguments**

- `path`: string, the .png to write, default: a file dialog

**Returns:** {path, width, height, spp}

## `view.render`

**Kind:** [process](index.md#kinds)

A path-traced PNG of the view at {width, height, spp}, rendered by another process from a copy of the scene and its state; view.render\_status reports it.

**Arguments**

- `path`: string, the .png to write, default: a file dialog
- `width`: int, pixels, default 1920
- `height`: int, pixels, default 1080
- `spp`: int, samples per pixel, default 64

**Returns:** {path, process}

## `view.render_status`

**Kind:** [read](index.md#kinds)

Whether a view.render job runs, how long it has, and the last finished one.

**Returns:** {running, path, seconds, last}

## `view.bookmarks`

**Kind:** [read](index.md#kinds)

The document's camera bookmarks.

**Returns:** a list of {name, target, distance, yaw, pitch, fov}

## `view.bookmark_add`

**Kind:** [view](index.md#kinds)

Keeps the view as `name`, replacing a bookmark of that name; without one it is "View N". Bookmarks are saved beside the document.

**Arguments**

- `name`: string, the bookmark's name, default "View N"

**Returns:** {name, count}

## `view.bookmark_goto`

**Kind:** [view](index.md#kinds)

Moves the camera to a bookmark.

**Arguments**

- `name`: string, the bookmark's name

**Returns:** {name}

## `view.bookmark_rename`

**Kind:** [view](index.md#kinds)

Renames a camera bookmark.

**Arguments**

- `name`: string, the bookmark's name
- `to`: string, the new name

**Returns:** {name}

## `view.bookmark_remove`

**Kind:** [view](index.md#kinds)

Removes a camera bookmark.

**Arguments**

- `name`: string, the bookmark's name

**Returns:** {count}

## `view.bookmarks_save`

**Kind:** [file](index.md#kinds)

Writes the bookmarks in the format --cams reads; the dialog asks without `path`.

**Arguments**

- `path`: string, the .json to write, default: a file dialog

**Returns:** {path, count}

## `view.bookmarks_load`

**Kind:** [file](index.md#kinds)

Replaces the bookmarks with a file's; the dialog asks without `path`.

**Arguments**

- `path`: string, the .json to read, default: a file dialog

**Returns:** {path, count}

## `view.robot_overlays`

**Kind:** [view](index.md#kinds)

The robot overlays: link frames, joint axes, centres of mass, inertia boxes and translucent links. Each switch given sets it; the result holds them all.

**Arguments**

- `on`: bool, overlays on, default kept
- `frames`: bool, link frames, default kept
- `axes`: bool, joint axes, default kept
- `com`: bool, centres of mass, default kept
- `inertia`: bool, inertia boxes, default kept
- `translucent`: bool, translucent links, default kept

**Returns:** {on, frames, axes, com, inertia, translucent}

## `view.sensor_overlays`

**Kind:** [view](index.md#kinds)

The sensor overlays: camera frustums, lidar fans, sensor frames and labels. Each switch given sets it; the result holds them all.

**Arguments**

- `on`: bool, overlays on, default kept
- `frustums`: bool, camera frustums, default kept
- `fans`: bool, lidar fans, default kept
- `frames`: bool, sensor frames, default kept
- `labels`: bool, sensor names, default kept

**Returns:** {on, frustums, fans, frames, labels}

## `view.ui_scale`

**Kind:** [view](index.md#kinds)

Sets the UI scale, 0.5 to 3, on top of the display's.

**Arguments**

- `value`: float, the scale, 0.5 to 3

**Returns:** the scale

## `view.ui_scale_up`

**Kind:** [view](index.md#kinds)

Makes the UI 10% larger, up to 3.

**Returns:** the scale

## `view.ui_scale_down`

**Kind:** [view](index.md#kinds)

Makes the UI 10% smaller, down to 0.5.

**Returns:** the scale

## `view.ui_scale_reset`

**Kind:** [view](index.md#kinds)

Returns the UI scale to 1.

**Returns:** the scale

## `view.toggle_language`

**Kind:** [view](index.md#kinds)

Switches the UI to the next language with a string table, English first.

**Returns:** the locale now shown

## `view.language`

**Kind:** [view](index.md#kinds)

Shows the UI in `locale`: en, or one with a string table (zh-CN ships with the editor).

**Arguments**

- `locale`: string, the locale, such as en or zh-CN

**Returns:** the locale

## `view.missing_strings`

**Kind:** [read](index.md#kinds)

UI strings the current language has no translation for, as shown so far.

**Returns:** a list of strings
