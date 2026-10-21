<!-- Generated from assets/editor/actions.toml by tools/wiki_actions.py; edit those instead. -->

# Scene: add, collision, nodes

19 actions. Run one from the command palette (++ctrl+p++), from Python with `app.actions.run(id, **args)` or through MCP with `action_run`.

## `scene.spawn`

**Kind:** [edit](index.md#kinds)

Adds a primitive to the scene. Without a pose, a spawn drops in above the selection, else above the origin.

**Arguments**

- `kind`: string, box, sphere, capsule, cylinder or plane, default "box"
- `pose`: {pos: \[x, y, z\], quat: \[w, x, y, z\]}, the world pose, optional

**Returns:** {focus}: the new node's path

**Example**

```json
{"id": "scene.spawn", "args": {"kind": "sphere", "pose": {"pos": [0, 0, 1], "quat": [1, 0, 0, 0]}}}
```

## `scene.select`

**Kind:** [view](index.md#kinds)

Selects Stage nodes: replace makes `path` the selection (none clears it), add and toggle grow or shrink the selection, range takes `paths` with `path` leading.

**Arguments**

- `path`: string, the node's Stage path, default none
- `mode`: string, replace, add, toggle or range, default "replace"
- `paths`: list of strings, the nodes of a range

**Returns:** {path, selection}: the leading node and every selected path

## `scene.set_visible`

**Kind:** [view](index.md#kinds)

Shows or hides Stage nodes in the viewport; a hidden node keeps simulating. Without `value` the first node's state flips for all.

**Arguments**

- `value`: bool, true shows and false hides, default: flips
- `paths`: list of strings, the nodes, optional
- `path`: string, one node, default the selection

**Returns:** {paths, visible}

## `scene.set_locked`

**Kind:** [view](index.md#kinds)

Locks or unlocks Stage nodes against picking and gizmo drags in the viewport. Without `value` the first node's state flips for all.

**Arguments**

- `value`: bool, true locks, default: flips
- `paths`: list of strings, the nodes, optional
- `path`: string, one node, default the selection

**Returns:** {paths, locked}

## `scene.collision_preview`

**Kind:** [read](index.md#kinds)

What scene.generate\_collision would do with the same arguments, body by body, without changing anything; the Generate Collision dialog shows it.

**Arguments**

- `paths`: list of strings, the nodes: a body, a shape or joint in one, or a group for every body under it, optional
- `path`: string, one node, default the selection
- `method`: string, hull, vhacd (convex decomposition), sdf or mesh, default "hull"
- `max_pieces`: int, the most convex pieces vhacd makes per body, default 16
- `max_triangles`: int, the most triangles mesh keeps per body by merging nearby vertices on a grid, 0 keeps them all, default 0
- `replace`: bool, drop each body's colliders first, default true
- `only_missing`: bool, leave bodies that already have a collider, default false
- `subtree`: bool, a body's descendants too, default false

**Returns:** a list of {path, triangles, pieces, vertices, had\_collider, skipped}

## `scene.generate_collision`

**Kind:** [edit](index.md#kinds)

Builds colliders for bodies from their visual meshes, as one undoable edit: a convex hull, a convex decomposition, an SDF or the triangle mesh itself. Use it to make an imported environment collide.

**Arguments**

- `paths`: list of strings, the nodes: a body, a shape or joint in one, or a group for every body under it, optional
- `path`: string, one node, default the selection
- `method`: string, hull, vhacd (convex decomposition), sdf or mesh, default "hull"
- `max_pieces`: int, the most convex pieces vhacd makes per body, default 16
- `max_triangles`: int, the most triangles mesh keeps per body by merging nearby vertices on a grid, 0 keeps them all, default 0; use it so a large environment fits the contact budget
- `replace`: bool, drop each body's colliders first, default true
- `only_missing`: bool, leave bodies that already have a collider, default false
- `subtree`: bool, a body's descendants too, default false

**Returns:** {focus}

**Example**

```json
{"id": "scene.generate_collision", "args": {"path": "/World/warehouse", "method": "mesh"}}
```

## `scene.collision_dialog`

**Kind:** [view](index.md#kinds)

Opens the Generate Collision dialog on the selection with its preview.

**Returns:** the preview, as scene.collision\_preview returns it

## `scene.terrain`

**Kind:** [edit](index.md#kinds)

Sets the scene's terrain field, which the cook bakes into a static heightfield, or removes it with kind none. Each size defaults to the bump field g1\_rough trains on; undo restores the previous field.

**Arguments**

- `kind`: string, flat, bumps, slope, stairs, mixed, or none to remove the terrain
- `size`: \[x, y\] in m, the field's extent, default \[20, 20\]
- `center`: \[x, y\] in m, the field's center, default \[0, 0\]
- `cell`: number in m, the grid spacing, 0.02 or more, default 0.1
- `base`: number in m, the floor's height, default 0
- `height`: number in m, bumps and mixed: the tallest bump, default 0.05
- `bump`: number in m, bumps and mixed: a bump's footprint, default 0.4
- `grade`: number, slope: the rise per m along +X, default 0.1
- `rise`: number in m, stairs and mixed: each step's rise, negative digs a pit, default 0.08
- `width`: number in m, stairs and mixed: each step's width, default 0.3
- `platform`: number in m, stairs and mixed: the central platform's side, default 1
- `count`: int, stairs and mixed: steps from the floor, 1 to 64, default 8
- `tile`: number in m, mixed: a tile's side, default 5
- `margin`: number in m, mixed: the flat rim of each tile, default 0.6
- `seed`: int, mixed: picks each tile's features, default 0

**Returns:** the field's arguments as scene.terrain\_info returns them, or null after a removal

**Example**

```json
{"id": "scene.terrain", "args": {"kind": "bumps", "size": [30, 30], "height": 0.06}}
```

## `scene.terrain_info`

**Kind:** [read](index.md#kinds)

The scene's terrain field as scene.terrain arguments, every size filled in; null when the scene has none.

**Returns:** {kind, size, center, cell, base, height, bump, grade, rise, width, platform, count, tile, margin, seed}, plus image for a field read from a height map

## `scene.terrain_dialog`

**Kind:** [view](index.md#kinds)

Opens the Terrain dialog on the scene's field, or on the default bump field when there is none.

**Returns:** the arguments the dialog starts from

## `scene.nodes`

**Kind:** [read](index.md#kinds)

The scene's nodes in tree order: all, those under `under`, or those of one `kind`.

**Arguments**

- `under`: string, a node whose subtree to list, default the whole scene
- `kind`: string, artic, soft, cloth, fluid, body, static, joint, geom, visual, camera, light, script or group, optional

**Returns:** a list of {path, name, kind, pose}; pose is the world {pos, quat} the node stands at now, live for what moves

**Example**

```json
{"id": "scene.nodes", "args": {"under": "/go2", "kind": "body"}}
```

## `scene.fields`

**Kind:** [read](index.md#kinds)

The fields the Properties panel edits for `paths` (else `path`, else the selection; "" or no selection gives the world options), in groups: each group's `targets` are what an edit.apply op names as its target, each field's `field` the op's field. A field applies to the running world when `live`, else at the next rebuild.

**Arguments**

- `paths`: list of strings, node paths, optional
- `path`: string, one node path, default the selection

**Returns:** a list of {title, kind (body, joint, drive, collision or world), detail, targets, fields: \[{field, label, unit, value, live, integer, range, mixed, inherited, saved}\]}

**Example**

```json
{"id": "scene.fields", "args": {"path": "ground/ground_collision"}}
```

## `scene.add_box`

**Kind:** [edit](index.md#kinds)

Adds a box above the selection, else above the origin, as one undoable edit; scene.spawn with kind box.

**Returns:** {focus}

## `scene.add_sphere`

**Kind:** [edit](index.md#kinds)

Adds a sphere above the selection, else above the origin, as one undoable edit; scene.spawn with kind sphere.

**Returns:** {focus}

## `scene.add_capsule`

**Kind:** [edit](index.md#kinds)

Adds a capsule above the selection, else above the origin, as one undoable edit; scene.spawn with kind capsule.

**Returns:** {focus}

## `scene.add_cylinder`

**Kind:** [edit](index.md#kinds)

Adds a cylinder above the selection, else above the origin, as one undoable edit; scene.spawn with kind cylinder.

**Returns:** {focus}

## `scene.add_plane`

**Kind:** [edit](index.md#kinds)

Adds a plane as one undoable edit; scene.spawn with kind plane.

**Returns:** {focus}

## `scene.expand_all`

**Kind:** [view](index.md#kinds)

Expands every node in the Stage panel.

**Returns:** nothing

## `scene.collapse_all`

**Kind:** [view](index.md#kinds)

Collapses every node in the Stage panel.

**Returns:** nothing
