<!-- Generated from assets/editor/actions.toml by tools/wiki_actions.py; edit those instead. -->

# Assets

8 actions. Run one from the command palette (++ctrl+p++), from Python with `app.actions.run(id, **args)` or through MCP with `action_run`.

## `asset.sources`

**Kind:** [read](index.md#kinds)

The files the document's nodes were imported from, with the options each import used.

**Returns:** a list of {node, file, kind, profile, fix\_base, package\_paths, light\_units, ground, exists}

## `asset.reimport`

**Kind:** [edit](index.md#kinds)

Imports the source of the node at `path` (the selection by default) again in its place, as one edit to undo; the node "" is an asset opened whole, which opens again as a new document.

**Arguments**

- `path`: string, the imported node's Stage path, default the selection
- `discard`: bool, for the node "": drop unsaved changes without asking, default false

**Returns:** {focus}

## `asset.show_source`

**Kind:** [process](index.md#kinds)

Shows the source file of the node at `path` (the selection by default) in the system's file browser.

**Arguments**

- `path`: string, the imported node's Stage path, default the selection

**Returns:** {file}

## `assets.add_folder`

**Kind:** [app](index.md#kinds)

Adds a folder to the Assets panel's roots, saved in the profile; without `path` a folder dialog asks.

**Arguments**

- `path`: string, the folder, default: a folder dialog

**Returns:** {path}

## `assets.remove_folder`

**Kind:** [app](index.md#kinds)

Removes a folder from the Assets panel's roots.

**Arguments**

- `path`: string, the folder

**Returns:** nothing

## `assets.rescan`

**Kind:** [view](index.md#kinds)

Scans the Assets panel's folders again for new and removed files.

**Returns:** nothing

## `assets.page`

**Kind:** [view](index.md#kinds)

Shows the Assets panel's `page`: files or materials.

**Arguments**

- `page`: string, files or materials

**Returns:** nothing

## `assets.list`

**Kind:** [read](index.md#kinds)

The asset cards as the Assets panel lists them: the bundled examples, the document's folder and the user's folders; `pictures` asks for their thumbnails as the panel does.

**Arguments**

- `pictures`: bool, render thumbnails for the cards, default false

**Returns:** {roots, cards: \[{path, name, root, kind, thumbnail}\]}
