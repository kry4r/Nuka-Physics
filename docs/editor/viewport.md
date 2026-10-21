# Viewport and camera

## Navigating

| Input | Does |
| --- | --- |
| Left mouse drag | Orbit around the target |
| Middle mouse drag, or ++shift++ + drag | Pan |
| Wheel | Zoom toward the target |
| ++w++ ++a++ ++s++ ++d++, ++q++ ++e++ | Fly forward, left, back, right, down, up; hold ++shift++ to go four times faster |
| ++f++ | Frame the selection |
| ++home++ | Frame everything |
| ++shift+space++ | Show the active pane alone, and back |

Keys act while the pointer is over the viewport and no text field has focus. While teleop drives a
robot from the keyboard, ++w++ ++a++ ++s++ ++d++ steer the robot instead of the camera.

The **Camera** panel shows the camera's eye, target, yaw, pitch and distance, sets the **field of view**
(20° to 90°), and has **Frame all**, **Selected** and **Reset**. `view.camera {target, distance, yaw,
pitch}` sets the camera exactly; `--cam tx,ty,tz,dist,yaw,pitch` does so at start.

## Camera bookmarks

The row of chips at the viewport's bottom-left holds the document's camera bookmarks. **+** keeps the
current view; click a chip to fly back to it; right-click it to rename or remove it. Bookmarks are saved
with the document and can be exported to and imported from JSON (`view.bookmarks_save`,
`view.bookmarks_load`), the same format `--cams FILE.json` loads at start.

## Panes

**View Layout** (the viewport's top-right buttons, or **View** menu) shows one pane, two side by side, or
four. Each pane has its own:

- **camera**: the free camera, or one of the scene's cameras (the camera button at the pane's top-left);
- **view mode** (the menu beside it).

Click a pane to make it the active one, outlined in green; the camera keys and the toolbar act on it.
++shift+space++ shows one pane alone and goes back to the layout it left.

## View modes

| Mode | Shows |
| --- | --- |
| **Lit** | The scene with its materials, sun, shadows and sky |
| **Unlit** | Base colors without lighting |
| **Normals** | Surface normals as colors |
| **Material ID** | Each material in its own flat color |
| **Residual** | The solver's per-vertex residual on cloth and soft bodies |
| **Contact** | Contact state on the surfaces |
| **Wireframe** | Triangle edges |

**View → Sun Shadows**, **Ambient Occlusion**, **Distance Fog** and **Ground Grid** adjust the Lit view. The ground grid's
*auto* setting draws it only under scenes without a floor.

Ambient occlusion is enabled by default and darkens nearby corners and contact areas in indirect light.
Use `view.ao {on: false}` to turn it off. It uses visible depth, so geometry outside the view cannot occlude it.

A scene's `environment.hdri` also lights the Lit viewport, supplies reflections and replaces the sky background.
The viewport honors `yaw_deg` and `intensity`. Diffuse lighting uses a global environment probe
until local lighting is baked.

### Local lighting

**View → Bake Local Lighting** samples static geometry, lights and the environment around the camera's
target. The Lit viewport interpolates this field to add diffuse bounce lighting and room occlusion.
Distance samples reduce light leaking through walls. Moving objects receive the baked light.
**View → Local Lighting** toggles the result.

For a larger room, set the volume explicitly from the Python console:

```python
app.actions.run("view.probes.bake", center=[0, 0, 1.5], spacing=1.0,
                count=5, face_size=16, samples=16)
```

`center` and `spacing` use scene metres. `count` is the number of probes on each axis (2–16);
`face_size` controls visibility resolution (4–32), and `samples` controls bake quality (1–64).
The default is a 3 × 3 × 3 grid with one metre spacing around the camera target.
Place probes throughout the room's free space and include its receiving surfaces within the volume.
Lighting fades to the global environment outside the volume.

The bake is kept for the current editor session. Changes to static geometry, materials, lights or
the environment invalidate it; bake again after editing. Moving objects do not contribute occlusion
or bounce light to this static field. Specular reflections continue to use the environment map.

## Overlays

| Overlay | Where |
| --- | --- |
| Collider proxies, contact points and normals | **View → Collider Proxies**, **Contact Points**; the Debug panel |
| Link frames, joint axes, centers of mass, inertia boxes, translucent links | **View → Robot Overlays** |
| Camera frustums, lidar fans, sensor frames and labels | `view.sensor_overlays` |
| Velocity command and measured velocity arrows | The Policy panel's **Arrows** |
| The residual heatmap | The Solver panel's **Heatmap** |

**View → Hide Overlays** turns the collider and contact overlays off together.

## Several environments

A world can hold many environments: copies of the scene that simulate side by side, as training runs
do. The **Environments** button over the viewport sets:

- **Count**, then **Rebuild**: how many environments the world holds, up to 4096. Rebuilding starts them
  all from the scene's initial state.
- **Shown**: how many the viewport draws, up to 16, laid out in a grid.
- **Spacing**: the distance between grid cells; *auto* sizes the cells from the robot.
- The cells of the current page: click one to make it the **current** environment, the one Properties,
  Joints and the sensors show and edit. The arrows page through the rest.

The status bar shows **ENV i / N** while there is more than one.

## Ray-traced preview and renders

- **RT Preview** (the sparkle button) path-traces the view over the raster image and refines it while
  nothing moves; the chip shows the samples per pixel so far. **Save RT Image...** writes it as a PNG.
- **Render This View...** (the camera button) renders a still in a separate process from a copy of the
  scene: choose the PNG, the size (1920 × 1080 by default) and the samples (64). The editor stays usable
  while it renders; `view.render_status` reports it.

## Screenshots and recordings

- **View → Screenshot** writes the next frame of the whole window as a PPM image.
- `view.record {on: true, file}` records the whole window, UI included, to an MP4 through `ffmpeg` on the
  PATH, with NVIDIA's hardware encoder or x264, at the frame rate you ask. Turning it off finishes the file.
  `view.record_status` reports frames written and dropped. Recordings go to the profile's `recordings/`
  folder unless you name a file.
- The Timeline panel exports a *simulation* to video, frame by simulated frame; see
  [Inspecting physics](inspect.md#the-timeline).

For demos and tutorials, `ui.pointer` shows a drawn mouse pointer that glides to a button by its label and
clicks it, so a script can drive the editor visibly while it records.
