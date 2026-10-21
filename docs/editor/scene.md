# Building scenes

A scene is a tree of **nodes**: robots (articulations), bodies, static geometry, cloth, soft bodies,
joints, shapes, visuals, cameras, lights, scripts and groups. The **Stage** panel shows the tree, the
**Properties** panel edits the selection, and the viewport's gizmo moves it.

## The Stage panel

Each row shows a node's name and its kind (`artic`, `body`, `static`, `cloth`, `soft`, `joint`, `geom`,
`visual`, `camera`, `light`, `script`, ...).

- **Select** a node by clicking its row or the object in the viewport. ++ctrl++-click adds or removes a
  node, ++shift++-click selects a range. Double-click a row to frame it in the viewport.
- **Filter** narrows the tree to names that match.
- **+** adds a primitive or a sensor; the arrows expand or collapse the whole tree.
- **Drag** a row onto another to reparent it. Dropping a sensor on a row mounts it on that body; dropping
  a material from the Assets panel assigns it to every part under the row.
- ++f2++ renames the selected row in place; ++enter++ commits.
- The eye and lock icons at the end of a row hide or lock the node.
- The **context menu** (right-click) renames the node and holds **Add primitive**, **Add Sensor**, the
  semantic label, **Frame**, **Generate Collision...**, **Reimport...** and **Show Source in Folder** (for
  imported nodes), **Export USD...**, **Import Asset...** and **Delete subtree**.

**Hide** (also ++h++ over the viewport) takes a node out of the picture while it keeps simulating.
**Lock** keeps a node from being picked or dragged in the viewport, which helps when a floor or a wall
is in the way.

## Adding objects

**Create** holds the primitives: **Box, Sphere, Capsule, Cylinder** and **Plane**. A new primitive drops
in above the selected node, or above the origin, clear of moving bodies. Robots and environments come
in through [import](import.md); terrain through [Terrain](#terrain); sensors through the Stage's **+**
menu ([Sensors and datasets](sensors.md)).

## Moving things: the gizmo

Select a node and the gizmo appears on it. The toolbar sets how it works:

| Button | Effect |
| --- | --- |
| **Gizmo** | Shows or hides the gizmo. |
| **Move** / **Rotate** | Translate along or rotate about the axes. |
| **World space** / **Local space** | Axes aligned with the world or with the node. |
| **Snap** | Moves in steps; right-click the button to set the step (metres or degrees). |
| **IK Drag** | On a robot's link, the drag poses the robot through inverse kinematics; see [Robots](robots.md#inverse-kinematics). |

A drag is one undoable step. Dragging a body while the simulation plays moves it there at the next step
(a teleport), and it carries on simulating from the new place.

## The Properties panel

**Properties** shows the selection's fields in sections. What appears depends on the node:

| Section | Fields |
| --- | --- |
| **Source** | For imported nodes: the file, its format and import options, **Reimport**. |
| **Transform** | Position and rotation, in the world or relative to the parent (**Local**). |
| **Materials** / **Material** | The material each drawn part uses; see [Materials](materials.md). |
| **Body** | Mass, inertia, damping. |
| **Joint** | Lower and upper limit, damping, stiffness, armature, friction loss. |
| **Drive** | Gain, force limit. |
| **Collision** | Friction and the shape's size (radius, half height, half extents), margin and gap. |
| **World** | With nothing selected: gravity and the solver's velocity iterations, among the world's options. |

Every field shows its unit. A field marked ⚡ is **live**: the change reaches the running world at the
next physics step. Other fields change the world's structure; their edits wait as **pending** (the panel
says *N pending*) until **Apply & Rebuild**, which rebuilds the world once with all of them and carries
the bodies' and joints' state over. **Discard** drops them.

With several nodes selected, Properties edits the fields they share; a field whose values differ shows
so, and an edit sets them all. Right-click a field for **Revert to Saved**, **Copy Value** and **Paste
Value**.

The **Semantic** box at the top sets the node's class for segmentation images; see
[Sensors and datasets](sensors.md#semantic-labels).

!!! note "Scripts and agents"
    `scene.fields {path}` lists exactly what Properties shows for a node, with each field's op name, unit,
    range and whether it is live. `edit.apply {label, ops}` changes any of them as one undoable step.

## Deleting, renaming, reparenting

- **Delete** (++delete++) removes the selection and everything under it, as one undoable step.
- **Rename** with ++f2++ or the Stage context menu; **reparent** by dragging a row onto another.

## Generating colliders

Imported environments often bring visual meshes and no collision. **Create → Generate Collision...**
opens on the selection:

| Method | Builds |
| --- | --- |
| **Convex Hull** | One convex hull around each body's visuals. Fast and robust; good for props. |
| **V-HACD** | Convex pieces from a V-HACD decomposition (**Max pieces**, 16 by default); good for concave movable objects. |
| **SDF** | The visual mesh as a signed-distance contact surface. |
| **Triangle Mesh** | The visual triangles as they are, for static scenery such as buildings and floors. **Max triangles** merges nearby vertices so a large environment fits the contact budget. |

**Include descendants** takes every body under the selection. **Only bodies without colliders** leaves
bodies that have one; **Replace the body's colliders** drops existing ones first. The **Preview** lists,
body by body, the triangles, pieces and vertices it would make and whether it replaces, adds to or makes a
new collider. **Generate** applies it as one undoable step. Turn on **View → Collider Proxies** to see the
result.

## Terrain

**Create → Terrain...** lays a height field under the scene, baked into a static collider:

| Kind | Shape |
| --- | --- |
| **Flat** | A level field. |
| **Bumps** | Blocks of random height up to the bump height (**Bump height**, **Bump size**). |
| **Slope** | A ramp rising along +X (**Grade**). |
| **Stairs** | Square steps up to a central platform (**Step rise**, **Step width**, **Platform**, **Steps**); a negative rise digs a pit. |
| **Mixed** | Tiles that mix steps, pits and bumps (**Tile**, and the stairs and bump sizes), from a seed. |

**Cell** sets the grid spacing (0.1 m by default) and the field covers 20 × 20 m around the origin unless
you change it. The defaults are the rough field the G1 walking task trains on. **Remove Terrain** takes it
away; both are undoable.

## Visibility, overlays and selection in the viewport

- Click selects what is under the pointer.
- ++f++ frames the selection, ++home++ frames everything.
- **View → Collider Proxies** and **Contact Points** draw the physics shapes and contacts over the scene.

See [Viewport and camera](viewport.md) for navigation and display.
