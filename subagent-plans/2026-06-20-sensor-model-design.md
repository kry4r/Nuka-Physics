# Sensor Model + Python API for `src/rhi/offline`

Decided sensor model for the offline RHI (controller-reviewed, grounded). Feeds the
P1 section of `2026-06-20-offline-rhi-IMPL-spec.md`.

## 1. Isaac Lab sensor suite (reference)

Render (need a tracer) vs analytic (pose/physics math). Sources:
isaaclab.sensors API, camera concept, add-sensors tutorial.

| Sensor | Render? | Outputs / AOVs | Key cfg | Mount | Cadence |
|---|---|---|---|---|---|
| **Camera** | yes | `rgb`,`rgba`,`depth`/`distance_to_image_plane`,`distance_to_camera`,`normals`,`motion_vectors`,`semantic_segmentation`,`instance_segmentation_fast`,`instance_id_segmentation_fast` | `width`,`height`,`data_types[]`,`spawn`(focal_length,clipping_range,aperture) | `prim_path`+`offset{pos,rot,convention}` | `update_period` s |
| **TiledCamera** | yes (batched) | same AOVs, one render_product for all clones | Camera cfg + tiling | same | `update_period` |
| **RayCaster** | tracer-class | `pos_w`,`quat_w`,`ray_hits_w` (height/depth scan) | `mesh_prim_paths[]`,`pattern_cfg`,`max_distance`,`attach_yaw_only`,`drift_range` | body+`offset` | `update_period` |
| **RayCasterCamera** | tracer-class | RayCaster + projected `depth`/`rgb`/`normals` | + `data_types`,`pattern_cfg` | body+`offset` | `update_period` |
| **ContactSensor** | analytic | `net_forces_w(_history)`,`force_matrix_w`,`current/last_air_time` | `track_pose`,`track_air_time`,`force_threshold`,`filter_prim_paths_expr` | `prim_path` regex | `update_period` |
| **Imu** | analytic | `lin_acc_b`,`ang_vel_b`,`quat_w`,`lin_vel_b` | `offset`,`gravity_bias` | body+`offset` | `update_period` |
| **FrameTransformer** | analytic | `target_pos/quat_(source\|w)`,`source_pos/quat_w` | `source_frame_offset`,`target_frames[]` | source+target prim paths | `update_period` |

TiledCamera batched tensor: `cam.data.output["rgb"]`→`(N,H,W,3)` uint8;
`["depth"]`→`(N,H,W,1)` f32; `["instance_id_segmentation_fast"]`(non-colorized)→
`(N,H,W,1)` int32. One GPU render_product, annotator wrapped as torch CUDA tensor,
sliced per-tile — the layout Nuka must match.

## 2. USD sensor representation (parser targets)

Cameras = `UsdGeomCamera` prims (`"Camera"`). FOV derived:
`vFOV = 2·atan(verticalAperture/(2·focalLength))` (tenths of a unit);
`clippingRange=(near,far)`; transform from inherited `xformOp:*`; mount = nearest
body/Xform ancestor. RTX lidar = `OmniLidar`/RTX prim (config:
`horizontal/verticalFov`,`horizontal/verticalResolution`,`min/maxRange`). Nuka already
reads `Camera` + custom `NukaSensor` prims (`usd_importer.cpp:1104-1145`) and derives
vFOV (`:948-955`, `CameraVFovDegreesFromUsd`). Extend the `NukaSensor` branch
(`:1134`) for `nuka_type∈{lidar,depth}` + fov/resolution/range; mount via
`FindNearestBodyAncestor` (`:1108`).

## 3. MJCF `<sensor>` (mapping)

MJCF sensors mount on site/body/geom/joint/camera; intrinsics on `<camera fovy>`.

| MJCF | mount ref | Nuka kind |
|---|---|---|
| `<camera>` (`pos`,`quat/euler`,`fovy`,`resolution`) | body | render Camera |
| `rangefinder` | site (1 ray) | render RangeScan |
| `accelerometer`,`gyro`,`velocimeter` | site | analytic Imu |
| `touch`,`force`,`torque` | site | analytic Contact/ForceTorque |
| `framepos/framequat/framelinvel/frameangvel` | objtype/objname | analytic FramePose |
| `jointpos`,`jointvel` | joint | analytic JointState |

Nuka already maps `rangefinder→Lidar`,`camera→Camera`,`framepos→FramePose`,
`touch→Contact`, default→Imu (`mjcf_importer.cpp:347-363`), parses `<camera fovy>`
(`:700-714`), `<sensor>` (`:783-806`). Gaps: site→frame resolution + `pos/quat` capture.

## 4. Code grounding

- **Two sensor IRs to unify (don't fork):** runtime
  `sensor::SensorType{IMU,JointState,ContactSummary,Lidar,Depth}` +
  `SensorDescriptor{name,type,attached_body,local_transform,ray_count,ray_range}`
  (`sensor_graph.hpp:16-31`); scene `scene::SensorType{Imu,Lidar,Camera,ForceTorque,
  Contact,FramePose}` (`canonical_types.hpp:42-49`), `SensorRecord` (`scene_ir.hpp:128-135`),
  `CameraRecord{...vertical_fov_degrees,near_clip,far_clip}` (`:151-159`),
  `AddSensor/AddCamera` (`:225-228`).
- **Importer hooks present:** MJCF `ParseSensors`/`<camera>` (`mjcf_importer.cpp:783-806,
  700-714`); USD `Camera`+`NukaSensor` (`usd_importer.cpp:1104-1145`).
- **Mount seam = FK fields:** device FK pose in `FieldId::{LinkPose,BodyPose,BasePose}`
  (`field_ids.hpp:15-17`), 7-float `[px,py,pz,qw,qx,qy,qz]` env-major
  (`dlpack_table.hpp:78-101`). Renderer already resolves `(FieldId,row)` per instance
  and composes `pose ∘ local` (`render_world.hpp:24,142-149`). A sensor world pose is
  the SAME op.
- **Shared render core:** `RenderWorld` feeds raster + path-tracer (`render_world.hpp:3-17`),
  RT adapter `RenderWorldToTwoLevelScene` (`rt_adapter.hpp`) — the core `RenderProfile`
  switches.
- **Zero-copy Python seam:** `nuka_buffer_view_t{device_ptr,element_count,
  element_stride_bytes,dtype}` + `nuka_world_get_buffer_view` (`nuka.h:500-511`);
  nanobind `nb::ndarray` `nb::pytorch` → `torch.from_dlpack` (`nuka_ext.cpp:539-621`);
  `World.buffer_view(field)` (`__init__.py:66-108`).

## 5. The decision

### (a) One `SensorDesc` IR — unify
Collapse the two enums into one `scene::SensorType`; promote `SensorRecord`/`CameraRecord`
into one `SensorDesc` (keep `CameraRecord` as the intrinsics payload to not break
serialization). `sensor::SensorGraph`'s enum is deleted and typedef'd to
`scene::SensorType`. Cover Isaac's set:

```cpp
namespace nuka::scene {
enum class SensorType : uint8_t { Imu, Contact, ForceTorque, FramePose,
                                  JointState, Camera, Depth, Lidar, RangeScan };
enum class MountFrame : uint8_t { Link, Body, Base };          // -> FieldId
struct CameraIntrinsics { uint16_t width=0,height=0; float vfov_deg=45.f,
    near_clip=0.01f,far_clip=1000.f; uint8_t distortion=0; float k1=0,k2=0; };
struct LidarPattern { uint16_t az_count=0, el_count=0; float az_min,az_max,
    el_min,el_max,min_range=0.f,max_range=100.f; uint32_t dir_table_off=~0u; };
struct SensorDesc {
    std::string name; SensorId id=0; SensorType type=SensorType::Imu;
    MountFrame  mount=MountFrame::Link;  uint32_t mount_index=0;   // link/body/base row
    math::Transform local_offset = math::Transform::Identity();
    CameraIntrinsics cam; LidarPattern lidar;
    uint32_t aov_mask=0;          // Rgb|Depth|DistCam|Normal|SemSeg|InstSeg|InstId|MotionVec
    uint32_t update_period=1;     // step decimation
};
}
```
`dir_table_off` lets a real lidar direction table reuse the pattern slot — ONE path
for grid/lidar/range, no per-scene branch.

### (b) Mount (device-side, batched, zero-copy)
Resolve each `SensorDesc` once to `(FieldId,row)` like `render_world.hpp:24`
(Link→LinkPose, Body→BodyPose, Base→BasePose). Add `FieldId::SensorWorldPose`
(per-sensor, persistent, 7 floats, env-major) to `fields.yaml` + regen. A device
kernel each step (gated by `update_period`) reads `FieldPtr(mount)[env,row] ∘
local_offset` → `SensorWorldPose[env,sensor]`, on-device, batched, no host round-trip.
Render sensors use it as per-env camera/ray origin.

### (c) USD + MJCF → `SensorDesc`
Extend existing hooks. MJCF (`mjcf_importer.cpp:783-806`): site→(body,pos/quat)→mount;
`<camera fovy resolution>`→`CameraIntrinsics`; `rangefinder`→`RangeScan` 1-ray;
`accelerometer/gyro/velocimeter`→`Imu`; `touch/force/torque`→`Contact/ForceTorque`;
`frame*`→`FramePose`. USD (`usd_importer.cpp:1104-1145`): `Camera`→intrinsics+clips+
parent mount; extend `NukaSensor` for `nuka_type∈{lidar,depth}` + fov/resolution/range.

### (d) Python API (Isaac cfg pattern, reuse DLPack seam)
`*Cfg` dataclasses lower to `SensorDesc`; `.data(aov)` returns `torch.from_dlpack` of
the engine AOV buffer via a new `nuka_world_get_sensor_view(world,sensor_id,aov,&view)`
returning the existing `nuka_buffer_view_t` — no new interop machinery.

```python
cam = w.add_sensor(nuka.CameraCfg(name="wrist", mount="link", link="right_hand",
        offset=((0.05,0,0.02),(1,0,0,0)), width=160, height=120, vfov_deg=60,
        data_types=["rgb","depth","instance_id_segmentation"]))
lidar = w.add_sensor(nuka.LidarCfg(name="head", mount="link", link="head",
        az=(-1.57,1.57,180), el=(-0.26,0.26,16), max_range=20.0))
w.step()
rgb   = torch.from_dlpack(cam.data("rgb"))     # (envs,120,160,3) uint8 GPU zero-copy
depth = torch.from_dlpack(cam.data("depth"))   # (envs,120,160,1) f32
scan  = torch.from_dlpack(lidar.data("range")) # (envs,16,180) f32
```
Output shapes match Isaac TiledCamera with `num_cameras → envs`.

### (e) Render vs analytic (one model, two paths)
- **Render** (`Camera,Depth,Lidar,RangeScan`): traced by `src/rhi/offline` against the
  shared core, switched by `RenderProfile`. AOVs device-resident, zero-copy. **D1:**
  depth/seg/range are exact closest-hit; only RGB shading may be stochastic. Supersedes
  the CPU `QueryLidarSensor` (`ray_sensor.hpp`) — same output contract.
- **Analytic** (`Imu,Contact,ForceTorque,FramePose,JointState`): stay on `src/sensor`
  (`state_sensor`, GPU `LINK_CONTACT_WRENCH`, FK `SensorWorldPose`). They read the same
  `SensorDesc.mount`/`SensorWorldPose` so mounting is uniform.

**Net:** one `SensorDesc` IR, one mount mechanism (FK `FieldId ∘ offset`, device/batched),
one parse surface (extend MJCF/USD hooks), one zero-copy Python seam; render sensors via
the offline RHI, analytic via `src/sensor`; no per-scene paths.
