# File formats

## Scenes

**`.nks`** is the engine's scene: a JSON document with `nks_version` 1. Its main tables are `tree` (the
Stage hierarchy of bodies, articulations, joints, shapes, visuals, sensors, lights and scripts),
`physics_materials` and `render_materials`, `environment` (sky, shadows, grading), `imports` (assets it
pulls in by path) and the contact filters `exclude_pairs` and `contact_pairs`.

**`.nka`** sits beside an `.nks` with the same name and holds its binary payload: meshes and, in a
packaged scene, every image ([Package Scene](../editor/documents.md#saving)). Keep the two together.

VBD cloth and cable media may set `vbd.contact_pairs_per_vertex` to reserve mesh contact pairs per surface
vertex. The cook keeps at least one slot per vertex and surface edge; `0` uses that estimate. Larger budgets
help dense folds and layered surfaces, at the cost of more contact storage.

For baked cloth, `cloth_mesh.hinge_rest_angles` accepts `[a, b, angle]` entries: the two vertex indices of an
interior edge and its signed rest dihedral in radians under the mesh's face winding. Each edge may appear
once. These angles override the rest angles derived from the mesh, including the flat angles otherwise used
with `cloth_mesh.material_mesh`. Omitted edges keep their derived angles.

The editor and the engine also read:

| Format | Read | Written |
| --- | --- | --- |
| URDF | Import, with the Strict or Legacy [profile](../import/urdf_profiles.md) and `package://` search paths | |
| MJCF | Import | |
| USD (`.usd`, `.usda`, `.usdc`, `.usdz`) | Import of UsdPhysics, visuals, materials (UsdPreviewSurface, OmniPBR) and lights | **File → Export USD...**, `World.export_usd` ([Exporting USD](../editor/import.md#exporting-usd)) |
| OBJ, STL, GLB meshes | Import | |

## Kept with a document

The editor keeps a document's extras in your [profile](../editor/install.md#the-profile-folder), one JSON
file per document, named by a hash of the document's path:

| Folder | Holds |
| --- | --- |
| `poses/` | Joint poses ([Poses](../editor/robots.md#poses)) |
| `clips/` | Joint clips |
| `bookmarks/` | Camera bookmarks |
| `dr/` | Domain randomization |
| `labels/` | Semantic labels |
| `train/` | The training configuration |
| `runs/` | Training runs ([Run folders](../editor/training.md#run-folders)) |

The files below carry them to another document or another machine.

## Poses

**Robot → Save Poses** writes, and **Load Poses** reads, a JSON file:

```json
{"poses": [
  {"name": "crouch", "joints": {"fl_thigh_joint": 1.1, "fl_calf_joint": -2.0}},
  {"name": "stand", "joints": {"fl_thigh_joint": 0.8, "fl_calf_joint": -1.5}}
]}
```

Positions are in radians (revolute) or metres (prismatic). A pose holds only the joints it names; loading
replaces poses of the same name.

## Clips

**`.nkclip.json`** holds one joint clip, written by `clip.save` and read by `clip.load`:

```json
{
  "format": "nkclip",
  "version": 1,
  "clip": {
    "name": "wave", "duration": 2.0, "loop": true,
    "tracks": [{
      "joint": "right_shoulder_roll_joint",
      "keys": [
        {"time": 0.0, "value": -0.2, "interp": "bezier", "in": 0.0, "out": 0.0},
        {"time": 1.0, "value": 0.6, "interp": "linear"}
      ]
    }]
  }
}
```

## Camera bookmarks

**View → Export Camera Bookmarks** writes the file `--cams FILE.json` reads at start:

```json
{"cameras": [{"name": "front", "target": [0, 0, 0.4], "distance": 3.0, "yaw": 35, "pitch": 20, "fov": 45}]}
```

The target is in metres; yaw, pitch and the vertical field of view are in degrees.

## Demonstrations

`demo.save` writes the recorded episodes as robomimic-style HDF5 ([Demonstrations](../editor/motion.md#demonstrations)):

```text
data                  attrs: total, env_args (JSON: scene, dt, joints, links, axes)
data/demo_<k>         attrs: num_samples, env, start_tick
    actions           (T, J)     the drive targets each step ran with
    obs/q, obs/qd     (T, J)     joint positions and velocities after the step
    obs/root_pose     (T, R, 7)  each root link's x, y, z, qw, qx, qy, qz
    obs/root_vel      (T, R, 6)  its body-frame wx, wy, wz, vx, vy, vz
    obs/command       (T, 3)     the controller's vx, vy, wyaw
    obs/axes          (T, A)     the teleop device's axes
    obs/time          (T,)       the simulated time after the step
```

`nuka.editor.demos.read(path)` returns them as NumPy arrays; reading and writing need `h5py`.

## Recordings

**`.nkrec`** holds one line of the [timeline](../editor/inspect.md#the-timeline)'s history, written by
`timeline.save`: a JSON manifest, the keyframes, the inputs between them, the shown environment's link and
body poses, and the solver records. Loading one needs the same document and world.

```bash
python -m nuka.editor.nkrec info run.nkrec          # what it holds
python -m nuka.editor.nkrec rrd run.nkrec run.rrd   # convert for the Rerun viewer (needs rerun-sdk)
```

**`.rrd`** is a [Rerun](https://rerun.io) recording: `rerun.stream` writes one live, and `nkrec rrd`
converts a recording. `timeline.export` renders a span of ticks to **`.mp4`** with ffmpeg.

## Policies

**`.nkml`** is a trained network the editor and engine run (`.bin` is the older form). A training run
writes **`<name>.contract.json`** beside each one: the joints it drives, its observation terms and its
gains. A policy with a contract runs on any robot that has the contract's joints
([Policies](../editor/motion.md#policies)). Runs also keep rl_games `.pth` checkpoints.

## Datasets

**File → Export Dataset...** writes a folder per camera with its planes (`rgb`, `depth`, `normal`, `albedo`,
`instance`, `semantic`), plus `lidar`, `state.npz`, COCO `annotations.json` and `meta.json`
([Exporting a dataset](../editor/sensors.md#exporting-a-dataset)). `nuka.editor.dataset.read(path)` loads
one.

## Settings and extensions

| File | Holds |
| --- | --- |
| `settings.toml` (profile) | UI, asset folders, script budget, trainer Python, agent access |
| `recent.json` (profile) | The recent scenes |
| `extensions.toml` (profile) | Which extensions you enabled or disabled |
| `extension.toml` (an extension's folder) | Its manifest ([Extensions](../editor/extensions.md#the-manifest)) |
| `logs/agent-<date>.jsonl` (profile) | The agents' audit log ([Agents](../editor/agents.md)) |
