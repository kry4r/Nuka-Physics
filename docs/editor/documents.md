# Documents and files

The editor works on one **document** at a time: a scene, saved as an `.nks` file with its meshes and
images in a companion `.nka` archive. Several editors can run side by side, each with its own document.

## The start page

With no scene open, the editor shows the start page:

- **New**: *Empty Scene* (nothing at all) or *Ground Plane* (a static floor, a sun and a sky to build on).
- **Open** a scene, or **Import** a robot or environment (URDF, MJCF, USD, meshes) into a new document.
- **Recent**: the last twelve scenes you opened or saved, with thumbnails. A file that is gone is marked
  **MISSING**; right-click to remove it from the list.
- **Recover**: autosaves left by an editor that did not close cleanly. See [Autosave](#autosave).
- **Examples**: scenes that ship with the editor, such as the gripper, the arena and the Nuka Dynamics Lab.
- The engine and GPU the editor runs on: Nuka's version, the GPU, Vulkan and CUDA, how poses reach the
  renderer, and the Python in use.

Drop a scene file anywhere on the window to open it, or an asset to add it to the open document.

## Opening and creating

| Command | Shortcut | What it does |
| --- | --- | --- |
| **File → New Scene** | ++ctrl+n++ | A new untitled document from a template: `empty` or `ground`. |
| **File → Open Scene...** | ++ctrl+o++ | Opens an `.nks` in place. Other formats import into a new untitled document. |
| **File → Open Recent** | | The recent list. |
| **File → Open in New Window...** | | Starts a second editor on a file, sharing your profile. |
| **File → Import Asset...** | ++ctrl+i++ | Adds an asset to the open document through the [import wizard](import.md). |
| **File → Close Scene** | | Back to the start page. |

Starting the editor with a file while another editor runs on the same profile hands the file to the
running editor, which opens it; `--new-window` starts a separate one instead.

## Saving

| Command | Shortcut | What it does |
| --- | --- | --- |
| **File → Save** | ++ctrl+s++ | Saves to the document's file; an untitled document asks where. |
| **File → Save As...** | ++ctrl+shift+s++ | Saves to a new file, which becomes the document's file. |
| **File → Package Scene...** | | Saves a copy whose `.nka` carries every image the scene uses, so it opens on another machine with no other files. The document keeps its own path. |
| **File → Export USD...** | | Writes the scene, or one Stage node and what hangs from it, as UsdPhysics (`.usd`, `.usda`, `.usdc`, `.usdz`). See [Exporting USD](import.md#exporting-usd). |
| **File → Revert** | | Reloads the saved file, dropping the changes since. |

Besides the scene itself, a document keeps its joint poses, joint clips, camera bookmarks, domain
randomization and semantic labels with it, and the editor remembers its training configuration in your
profile.

## Unsaved changes

The **UNSAVED** chip on the toolbar shows that the document has changes. Anything that would drop them
(New, Open, Close, Exit) first asks **Save changes to …?** with **Save**, **Don't Save** and **Cancel**.

## Autosave

About once a minute, while a document has unsaved changes, the editor writes an autosave to the
profile's `autosave/` folder. If the editor ends without closing cleanly, the next start lists the
autosave under **Recover** on the start page; recovering opens it as the document it belonged to, with
its changes still unsaved, so you decide whether to save them.

## Unloading

The `file.unload` action (in the command palette as **Unload**) takes the scene out of the world but keeps
the window, for scripts that want to load their own world afterwards.

## Several editors

Each editor is a separate process with its own document and world. They share the profile, so recent
files, settings and asset folders are common. **File → Open in New Window...** or `--new-window` starts
another one; agents can address each by process id (see [Agents and MCP](agents.md)).
