<!-- Generated from assets/editor/actions.toml by tools/wiki_actions.py; edit those instead. -->

# Sensors, labels, datasets and randomization

16 actions. Run one from the command palette (++ctrl+p++), from Python with `app.actions.run(id, **args)` or through MCP with `action_run`.

## `sensor.list`

**Kind:** [read](index.md#kinds)

The scene's sensors with their mounts and settings.

**Returns:** a list of {index, name, type, mount, frame, offset, rate\_hz, update\_period, aov\_mask, camera, lidar, tactile}; each kind has its own block

## `sensor.add`

**Kind:** [edit](index.md#kinds)

Adds a sensor of `type` riding the body at `mount` (default the selected sensor's body, else the selected node's body; "" mounts a camera or lidar in the world) as one undoable edit, and selects it. Other fields set it up as sensor.set.

**Arguments**

- `type`: string, imu, contact, force\_torque, frame\_pose, joint\_state, camera, depth, lidar, range\_scan, linear\_velocity or tactile
- `mount`: string, the body it rides, "" for the world, default: from the selection
- `name`: string, its name, default the type
- `position`: \[x, y, z\], its offset from the mount, optional
- `rotation_deg`: \[x, y, z\], its rotation from the mount in degrees, optional
- `rate_hz`: float, samples per second, optional
- `camera`: {width, height, vfov, near, far}, a camera's image, optional
- `lidar`: {az\_count, el\_count, az\_min, az\_max, el\_min, el\_max, min\_range, max\_range}, a lidar's beams, optional
- `tactile`: {shape: box, sphere, ellipsoid, capsule or cylinder, size: \[x, y, z\]}, a contact region, optional

**Returns:** the new sensor, as sensor.list

**Example**

```json
{"id": "sensor.add", "args": {"type": "camera", "mount": "/go2/base", "position": [0.3, 0, 0.05], "camera": {"width": 640, "height": 480}}}
```

## `sensor.set`

**Kind:** [edit](index.md#kinds)

Edits a sensor (`index` or `sensor`, else the selected one) as one undoable edit: every other argument is a field, as sensor.add takes them, plus offset, update\_period and aov\_mask.

**Arguments**

- `index`: int, the sensor's index, optional
- `sensor`: string, the sensor's name, optional

**Returns:** the sensor, as sensor.list

**Example**

```json
{"id": "sensor.set", "args": {"sensor": "camera", "camera": {"vfov": 70}}}
```

## `sensor.remove`

**Kind:** [edit](index.md#kinds)

Removes a sensor (`index` or `sensor`, else the selected one) as one undoable edit.

**Arguments**

- `index`: int, the sensor's index, optional
- `sensor`: string, the sensor's name, optional

**Returns:** {focus}

## `sensor.select`

**Kind:** [view](index.md#kinds)

Selects a sensor, which drops the node selection; `index` -1 selects none.

**Arguments**

- `index`: int, the sensor's index, -1 for none, optional
- `sensor`: string, the sensor's name, optional

**Returns:** the selected index

## `sensor.preview`

**Kind:** [view](index.md#kinds)

The Sensors panel's live preview: `on`, the sensor it shows (`index` or `sensor`, else the selected one) and a camera's `channel`. Reports them with the newest reading's tick and error.

**Arguments**

- `on`: bool, show the preview, default kept
- `index`: int, the sensor's index, optional
- `sensor`: string, the sensor's name, optional
- `channel`: string, color, depth, normal, albedo, instance or semantic, default kept

**Returns:** {on, index, channel, read, tick, error}

## `sensor.read`

**Kind:** [read](index.md#kinds)

Renders and reads a sensor of environment `env` now: a camera's `channel`, a lidar's ranges or a state sensor's values. Returns their shape and statistics, the values when there are at most 4096, and with `path` writes all of them to a .npy file. A semantic plane holds 1 + the index of each pixel's class in `classes`, 0 for none; an instance plane names its ids in `instances`.

**Arguments**

- `index`: int, the sensor's index, optional
- `sensor`: string, the sensor's name, default the selected one
- `channel`: string, a camera's plane: color, depth, normal, albedo, instance or semantic, default "color"
- `env`: int, the environment, default the shown one
- `path`: string, a .npy file for all the values, optional

**Returns:** {index, name, channel, env, tick, sim\_time, shape, dtype, min, max, mean, values, classes or instances, path}

**Example**

```json
{"id": "sensor.read", "args": {"sensor": "camera", "channel": "depth", "path": "C:/data/depth.npy"}}
```

## `dr.get`

**Kind:** [read](index.md#kinds)

The domain randomization of the document: the seed, whether a reset draws again, the render jitters, the physics ranges and the live parameters' factor ranges.

**Returns:** {seed, reset\_draws, render: {on, color, roughness, metallic, light\_dir, light\_intensity, light\_color, ambient}, physics: {on, mass, armature, gravity}, params: {key: {on, range}}}

## `dr.set`

**Kind:** [edit](index.md#kinds)

Merges `args` in the shape dr.get reports and draws again when anything randomizes or did; saved beside the document. Render randomization needs a camera sensor, physics randomization an articulated body.

**Arguments**

- `seed`: int, the draw's seed, optional
- `reset_draws`: bool, each reset draws with the next seed, optional
- `render`: object, {on, and jitters &gt;= 0: color, roughness, metallic, light\_dir, light\_intensity, light\_color, ambient}, optional
- `physics`: object, {on, mass: \[lo, hi\] factor, armature: \[lo, hi\] offset, gravity: \[lo, hi\] offset}, optional
- `params`: object, a live parameter's key to {on, range: \[lo, hi\] factor}, optional

**Returns:** the randomization config

**Example**

```json
{"id": "dr.set", "args": {"physics": {"on": true, "mass": [0.8, 1.2]}, "reset_draws": true}}
```

## `dr.sample`

**Kind:** [sim](index.md#kinds)

Draws the randomization again with `seed`, else the next one.

**Arguments**

- `seed`: int, the seed, default the next one

**Returns:** {seed}

## `label.list`

**Kind:** [read](index.md#kinds)

The semantic labels by node path and their classes; with `path`, the label that node has and the node it comes `from`, itself or an ancestor.

**Arguments**

- `path`: string, a node to look up, optional

**Returns:** {labels: path to label, classes} (with label and from for `path`)

## `label.set`

**Kind:** [edit](index.md#kinds)

Gives the nodes at `paths` (or `path`, else the selected ones) the semantic `label` that datasets write; "" clears theirs, so they inherit their parent's again. Saved beside the document.

**Arguments**

- `label`: string, the class name, "" clears
- `paths`: list of strings, the nodes, optional
- `path`: string, one node, default the selection

**Returns:** {labels, classes}

**Example**

```json
{"id": "label.set", "args": {"path": "/World/go2", "label": "robot"}}
```

## `dataset.export`

**Kind:** [file](index.md#kinds)

Steps `frames` frames of `every` ticks from a reset (unless `reset` is false), drawing the `randomize` kinds again for each from `seed` + frame, and writes the `cameras`' `channels`, the lidars and the state sensors under `out` (a folder dialog without it); dataset.status follows it.

**Arguments**

- `out`: string, the folder to write, default: a folder dialog
- `frames`: int, how many frames, default 100
- `every`: int, ticks between frames, default 1
- `seed`: int, the first frame's seed, default the randomization's
- `randomize`: object, {render, physics}: what draws again each frame, default the document's randomization
- `channels`: list of strings, rgb, depth, instance, semantic, normal, albedo, default rgb, depth, instance, semantic
- `cameras`: list of strings, camera sensor names, default every camera
- `lidar`: bool, write the lidars, default true
- `state`: bool, write the state sensors, default true
- `reset`: bool, start from a reset, default true

**Returns:** the export's status, as dataset.status

**Example**

```json
{"id": "dataset.export", "args": {"out": "C:/data/run1", "frames": 50, "channels": ["rgb", "depth"]}}
```

## `dataset.status`

**Kind:** [read](index.md#kinds)

The dataset export running or last finished: frames written of how many, where, and any error.

**Returns:** {running, frame, frames, out, cancelled, error}

## `dataset.cancel`

**Kind:** [sim](index.md#kinds)

Stops after the frame being read and writes the frames read so far.

**Returns:** the export's status

## `dataset.exporter`

**Kind:** [view](index.md#kinds)

Opens or closes the Export Dataset dialog; `on` sets it, else it toggles.

**Arguments**

- `on`: bool, open it, default: toggles

**Returns:** {on}
