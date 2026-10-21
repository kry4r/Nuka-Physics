<!-- Generated from assets/editor/actions.toml by tools/wiki_actions.py; edit those instead. -->

# Materials

11 actions. Run one from the command palette (++ctrl+p++), from Python with `app.actions.run(id, **args)` or through MCP with `action_run`.

## `material.list`

**Kind:** [read](index.md#kinds)

The scene's materials: each one's fields as material.set takes them, its friction and how many shapes use it.

**Returns:** a list of {name, base\_color, metallic, roughness, emissive, opacity, sheen, transmission, ior, absorption, albedo\_map, roughness\_map, metallic\_map, normal\_map, emissive\_map, roughness\_channel, metallic\_channel, uv\_scale, triplanar, friction, users}

## `material.get`

**Kind:** [read](index.md#kinds)

One material: `material` by name, else the one the drawn parts at `path` (else the Material section's part) share.

**Arguments**

- `material`: string, the material's name, optional
- `path`: string, a node whose drawn parts share one material, default the Material section's part

**Returns:** the material, as material.list

## `material.slots`

**Kind:** [read](index.md#kinds)

The parts the world draws at or under `path` (else the selection) and their materials, "" for the default one.

**Arguments**

- `path`: string, the node, default the selection

**Returns:** a list of {path, material}

## `material.create`

**Kind:** [edit](index.md#kinds)

Makes a material as one undoable edit: a copy of `from`, else `preset`'s look, else the default, with `fields` over it. `assign` gives it to those nodes in the same edit.

**Arguments**

- `name`: string, its name, made unique, default from `from` or `preset`, else "Material"
- `from`: string, a material to copy, optional
- `preset`: string, plastic, metal, rubber, glass, emissive or textured, optional
- `fields`: object, material fields as material.set takes them, optional
- `assign`: string or list of strings, nodes that take it, optional

**Returns:** {material}: its name

**Example**

```json
{"id": "material.create", "args": {"preset": "metal", "fields": {"base_color": [0.8, 0.6, 0.2, 1.0]}, "assign": "/robot/base"}}
```

## `material.set`

**Kind:** [edit](index.md#kinds)

Sets fields of a material (`material`, else the one at `path`) as one undoable edit. Fields are named as material.get reports them; a map is an image file path ("" for none) and `friction` is the contact friction.

**Arguments**

- `material`: string, the material's name, optional
- `path`: string, a node whose drawn parts share one material, default the Material section's part
- `base_color`: \[r, g, b, a\], linear, optional
- `metallic`: float, 0 to 1, optional
- `roughness`: float, 0 to 1, optional
- `emissive`: \[r, g, b\], optional
- `albedo_map`: string, an image file, "" for none, optional
- `roughness_map`: string, an image file, "" for none, optional
- `metallic_map`: string, an image file, "" for none, optional
- `normal_map`: string, an image file, "" for none, optional
- `emissive_map`: string, an image file, "" for none, optional
- `uv_scale`: float, texture repeats, optional
- `triplanar`: bool, project textures from three axes, optional
- `friction`: float, the contact friction, optional

**Returns:** {focus}

**Example**

```json
{"id": "material.set", "args": {"material": "Floor", "roughness": 0.7, "albedo_map": "C:/textures/concrete.png", "uv_scale": 4}}
```

## `material.choose_map`

**Kind:** [edit](index.md#kinds)

Picks an image for a material's `map` slot in a file dialog.

**Arguments**

- `map`: string, albedo\_map, roughness\_map, metallic\_map, normal\_map or emissive\_map
- `material`: string, the material's name, optional
- `path`: string, a node whose drawn parts share one material, default the Material section's part

**Returns:** {focus}

## `material.rename`

**Kind:** [edit](index.md#kinds)

Renames a material (`material`, else the one at `path`) as one undoable edit.

**Arguments**

- `name`: string, the new name
- `material`: string, the material's name, optional
- `path`: string, a node whose drawn parts share one material, optional

**Returns:** {focus}

## `material.delete`

**Kind:** [edit](index.md#kinds)

Deletes a material (`material`, else the one at `path`) as one undoable edit; the shapes that used it take `replace`, which a material in use needs.

**Arguments**

- `material`: string, the material's name, optional
- `path`: string, a node whose drawn parts share one material, optional
- `replace`: string, the material its users take, "" for the default one, required while it is in use

**Returns:** {focus}

## `material.assign`

**Kind:** [edit](index.md#kinds)

Every shape at or under `paths`, else `path`, else the selection, takes `material` ("" for the default one), as one undoable edit.

**Arguments**

- `material`: string, the material, "" for the default one
- `path`: string, a node, optional
- `paths`: list of strings, nodes, optional

**Returns:** {focus}

**Example**

```json
{"id": "material.assign", "args": {"material": "Rubber", "paths": ["/go2/FL_foot", "/go2/FR_foot"]}}
```

## `material.make_unique`

**Kind:** [edit](index.md#kinds)

The part at `path` (else the Material section's) takes a copy of its material of its own, as one undoable edit.

**Arguments**

- `path`: string, the part, default the Material section's

**Returns:** {material}: the copy's name

## `material.select_users`

**Kind:** [view](index.md#kinds)

Selects the shapes that draw with `material`, and lists them.

**Arguments**

- `material`: string, the material

**Returns:** {paths}
