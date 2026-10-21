<!-- Generated from assets/editor/actions.toml by tools/wiki_actions.py; edit those instead. -->

# Documents: new, open, import, save, export

16 actions. Run one from the command palette (++ctrl+p++), from Python with `app.actions.run(id, **args)` or through MCP with `action_run`.

## `file.new`

**Kind:** [file](index.md#kinds)

Replaces the document with a new untitled scene from a template (assets/editor/templates). With unsaved changes and no `discard` it asks first and returns {prompt}; app.answer\_prompt answers.

**Arguments**

- `template`: string, the template's name: empty or ground, default "empty"
- `discard`: bool, drop unsaved changes without asking, default false

**Returns:** {path, title} of the new document, or {prompt} while it asks

**Example**

```json
{"id": "file.new", "args": {"template": "ground", "discard": true}}
```

## `file.open`

**Kind:** [file](index.md#kinds)

Opens a scene file as the document: .nks opens in place, other formats (URDF, MJCF, USD, meshes) import into an untitled document. Without `path` a file dialog asks. With unsaved changes and no `discard` it asks first.

**Arguments**

- `path`: string, the file to open, default: a file dialog
- `discard`: bool, drop unsaved changes without asking, default false

**Returns:** {path, title}, or {prompt} while it asks

## `file.open_recent`

**Kind:** [file](index.md#kinds)

Opens a file from the recent list; the same as file.open.

**Arguments**

- `path`: string, the file to open
- `discard`: bool, drop unsaved changes without asking, default false

**Returns:** {path, title}, or {prompt} while it asks

## `file.forget_recent`

**Kind:** [app](index.md#kinds)

Removes a file from the recent list on the start page.

**Arguments**

- `path`: string, the file to forget

**Returns:** nothing

## `file.open_window`

**Kind:** [process](index.md#kinds)

Starts another editor in a window of its own, on `path` or on an empty document; it shares this editor's profile.

**Arguments**

- `path`: string, the scene to open there, default: a file dialog
- `empty`: bool, open no scene and skip the dialog, default false

**Returns:** {spawned}: the new editor's process id

## `file.revert`

**Kind:** [file](index.md#kinds)

Reloads the document from its file, dropping unsaved changes; enabled only for a saved document with changes.

**Returns:** {path, title}

## `file.close`

**Kind:** [file](index.md#kinds)

Closes the document and leaves the editor on the start page. With unsaved changes and no `discard` it asks first.

**Arguments**

- `discard`: bool, drop unsaved changes without asking, default false

**Returns:** nothing, or {prompt} while it asks

## `file.recover`

**Kind:** [file](index.md#kinds)

Opens an autosave left by an editor that did not close cleanly, as the document it belonged to with its changes unsaved.

**Arguments**

- `path`: string, the autosave .nks (from the start page's recovery list)
- `discard`: bool, drop unsaved changes without asking, default false

**Returns:** {path, title}, or {prompt} while it asks

## `file.autosave`

**Kind:** [file](index.md#kinds)

Writes the autosave of the document now.

**Returns:** {path} of the autosave

## `file.unload`

**Kind:** [file](index.md#kinds)

Unloads the scene from the world, keeping the editor open with no scene.

**Returns:** nothing

## `file.save`

**Kind:** [file](index.md#kinds)

Saves the document to its file, or to `path`; an untitled document asks where with a file dialog.

**Arguments**

- `path`: string, where to save, default the document's own file

**Returns:** {path}

## `file.save_as`

**Kind:** [file](index.md#kinds)

Saves the document to a new file, which becomes its file; without `path` a file dialog asks.

**Arguments**

- `path`: string, where to save, default: a file dialog

**Returns:** {path}

## `file.package`

**Kind:** [file](index.md#kinds)

Saves a copy at `path` that carries every image the scene uses in its own .nka, so it opens with no other files; the document keeps its path. `missing` lists the images that could not be read.

**Arguments**

- `path`: string, where to write the packaged .nks, default: a file dialog

**Returns:** {path, packed, missing}

## `file.export_usd`

**Kind:** [file](index.md#kinds)

Writes the scene, or the Stage node `node` with what hangs from it, as UsdPhysics in the format the path's extension names (.usd, .usda, .usdc, .usdz). `warnings` lists what USD could not carry; they go to the Console too.

**Arguments**

- `path`: string, the USD file to write, default: a file dialog
- `node`: string, the Stage path of the subtree to export, default the whole scene
- `light_units`: string, isaac (intensity x1000, as Isaac Sim reads it), nits or raw, default "isaac"

**Returns:** {path, warnings: \[{code, prim, message}\]}

**Example**

```json
{"id": "file.export_usd", "args": {"path": "C:/tmp/stage.usda"}}
```

## `file.import_analyze`

**Kind:** [read](index.md#kinds)

Reads an asset file (URDF, MJCF, USD, mesh, NKS) without importing it: whether it imports, its diagnostics and its size.

**Arguments**

- `path`: string, the asset file
- `profile`: string, how a URDF reads: strict or legacy, default "strict"
- `fix_base`: bool, fix the robot's base to the world, default false
- `package_paths`: list of strings, folders to search for package:// files, optional
- `light_units`: string, how a USD stage's light intensities read: isaac, nits or raw, default "isaac"
- `link_colliders`: string, a URDF or MJCF robot's triangle-mesh colliders on moving links: hull (their convex hulls, as MuJoCo collides them) or mesh (kept, far slower contact), default "hull"

**Returns:** {path, kind, ok, loaded, name, profile, link\_colliders, diagnostics: \[{severity, code, element, owner, message, line}\], stats: {links, joints, colliders, mesh\_colliders (triangle meshes), visual\_meshes, mass, lo, hi}, package\_paths}

## `file.import`

**Kind:** [edit](index.md#kinds)

Imports an asset file (URDF, MJCF, USD, OBJ/STL/GLB meshes, NKS). Without `mode` the import wizard opens on it; "auto" adds a clean asset to the open document at once and opens the wizard otherwise, "add" adds it as one undoable edit, "open" opens it as a new document.

**Arguments**

- `path`: string, the asset file, default: a file dialog
- `mode`: string, auto, add or open, default: the wizard
- `name`: string, the new node's name, default the file's stem
- `pose`: {pos: \[x, y, z\], quat: \[w, x, y, z\]}, where it lands, optional
- `ground`: bool, rest the lowest shape on z = 0 unless the asset brings its own floor, default true
- `profile`: string, how a URDF reads: strict or legacy, default "strict"
- `fix_base`: bool, fix the robot's base to the world, default false
- `package_paths`: list of strings, folders to search for package:// files, optional
- `light_units`: string, how a USD stage's light intensities read: isaac, nits or raw, default "isaac"
- `link_colliders`: string, a URDF or MJCF robot's triangle-mesh colliders on moving links: hull or mesh, default "hull"
- `discard`: bool, with mode open: drop unsaved changes without asking, default false

**Returns:** {focus} after add, {path, title} after open, the analysis while the wizard shows

**Example**

```json
{"id": "file.import", "args": {"path": "C:/assets/go2/go2.urdf", "mode": "add"}}
```
