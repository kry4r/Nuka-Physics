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

Baked volumetric VBD media use `kind: "soft_tet"`, `method: "vbd"` and a
`baked: "solid.nka#TETS/0"` reference. `tet_materials` contains volumetric `youngs` (Pa),
`poisson`, `density` (kg/m³) and Rayleigh `damping` (s). Contact properties remain in `vbd`.
The isotropic energy is stable Neo-Hookean; its full Hessian and the positive semidefinite
curvature used by the solver are distinct. Damping measures the Green strain rate at the interval start.

Baked MLS-MPM media use `method: "mlsmpm"` and `baked: "particles.nka#PSET/0"`.
The `mpm` block holds grid and stepping settings; optional `mpm_materials` rows hold constitutive
parameters indexed by each particle's material ID. Multiple media share a grid with the same `dx`,
floor and implicit-stress setting. `grid_origin` and `grid_dims` specify an explicit domain;
three zero dimensions derive a padded domain from positions, floor and `loft_headroom`.
`particle_radius` controls rendering only. Rest volume is always supplied by the asset.

Both chunks use little-endian integers and float32 arrays, preserve authored order, and reject
unknown flags, trailing/truncated streams, invalid indices and nonfinite values. Matrix streams
are row-major. Save packages the chunks into the scene's sidecar without changing their payloads.

| Chunk version 1 | Required streams after its four uint32 header words | Optional streams, in flag-bit order |
| --- | --- | --- |
| `TETS`: version, vertex count, tet count, flags | Positions `(N,3)`; tet indices `(T,4)` follow optional vertex streams | Bits 0–2: rest positions `(N,3)`, velocities `(N,3)`, inverse masses `(N)` before indices. Bits 3–4: rest inverse `(T,3,3)`, material IDs `(T)` after indices |
| `PSET`: version, particle count, flags, reserved zero | Positions `(N,3)`, inverse masses `(N)`, rest volumes `(N)` | Bits 0–5: velocities `(N,3)`, material IDs `(N)`, elastic F `(N,3,3)`, affine velocity C `(N,3,3)`, plastic Fp `(N,3,3)`, equivalent plastic strain `(N)` |

For `TETS`, rest inverse takes precedence over rest positions; absent both, initial positions
define the rest shape. Absent inverse masses, density and rest volume determine lumped vertex
masses; zero inverse masses pin vertices. `PSET` requires nonnegative inverse mass and positive
rest volume and determinants of F/Fp. Missing F/Fp are identity, velocities/C and plastic strain
are zero. Nontrivial Fp and equivalent plastic strain require Hencky J2 material rows. Zero-mass
MPM points do not impose a Dirichlet condition on neighboring grid nodes.

Use [`nuka.assets`](../python/index.md#baked-continuum-media) to write these assets. Importing a
particle set preserves geometry and state; comparing another engine also requires matching its
constitutive law, transfer, boundary conditions and control schedule. A scalar plastic volume
alone does not specify a full plastic deformation gradient or equivalent strain.

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
