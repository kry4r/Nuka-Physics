# Workflows

Each step is `action id {args}` for `action_run`. Arguments not shown keep their defaults; `action_help`
lists them all. Check each step as [verify.md](verify.md) describes before the next one.

## A new scene

1. `file.new {template: "ground"}`: a floor, a sun and a sky (`"empty"` has none). With unsaved
   changes it returns `{prompt}`; see the unsaved-work rule in SKILL.md.
2. `scene.spawn {kind: "box", pose: {pos: [0, 0, 0.5], quat: [1, 0, 0, 0]}}`: kinds box, sphere,
   capsule, cylinder and plane. The result's `focus` is the new node's path.
3. `edit.rename {path, name}`, `edit.reparent {path, parent}`, `edit.delete {paths}`.
4. `file.save {path: "C:/work/scene.nks"}`. A scene saves as .nks; `file.package {path}` makes a copy
   that carries its images.

## Importing assets

Formats: URDF, MJCF (.xml), USD (.usd, .usda, .usdc, .usdz), OBJ/STL/GLB meshes and .nks.

1. `file.import_analyze {path}`: whether it imports, its diagnostics, and its links, joints, colliders
   and mass. Read the diagnostics before importing.
2. `file.import {path, mode: "add", name: "go2"}` adds it to the open scene as one undoable edit.
   `mode: "open"` opens it as a new document instead. Without `mode` a wizard waits for the user.
   - `ground: true` (default) rests the lowest shape on the floor; `pose` places it.
   - `fix_base: true` bolts a robot's base to the world.
   - URDF: `profile: "legacy"` for loose files; `package_paths` lists the folders that hold
     `package://` meshes.
   - USD: `light_units` is `isaac` (default, intensities as Isaac Sim reads them), `nits` or `raw`.
3. `scene_tree {under: focus, depth: 1}` shows what came in; `scene.select {path: focus}` and
   `view.frame_selected` bring it into view for a screenshot.
4. `asset.sources` lists each import's file and options; `asset.reimport {path}` imports a changed
   file again in place.

## Making an environment collide

Imported environments (rooms, warehouses, terrain) often bring visual meshes only.

1. `scene.collision_preview {path, subtree: true, method}` reports what would be built, body by body.
2. `scene.generate_collision {path, subtree: true, method}`: `hull` (one convex piece per body), `vhacd`
   (convex decomposition, `max_pieces`), `sdf`, or `mesh` (the triangles as they are, for static
   bodies such as buildings). `only_missing: true` keeps colliders that already exist.
3. `view.colliders {on: true}` draws them; take a `screenshot` to check.

## Changing fields: mass, friction, joints, world options

1. `scene.fields {path}` lists the node's groups (body, joint, drive, collision, world). Each group has
   the `targets` to name and each field its op name, unit, range and current value.
2. `edit.apply {label: "Lower friction", ops: [{target, field: "shape.friction_mu", after: 0.8}]}`.
   Several ops in one call make one undo step.
3. `scene.fields {path: ""}` gives the world options (gravity, solver iterations, ...); their target is
   `""`.

## Materials

- `material.list`; `material.slots {path}` lists the parts under a node and the material each draws with.
- `material.create {preset: "metal", fields: {base_color: [0.8, 0.6, 0.2, 1]}, assign: path}`; presets
  plastic, metal, rubber, glass, emissive, textured.
- `material.set {material, roughness, albedo_map: "C:/tex/floor.png", uv_scale: 4, friction: 0.9}`:
  maps are image files, `friction` is the contact friction of every shape using it.
- `material.assign {material, paths}`; `material.make_unique {path}` gives one part its own copy.

## Robots

- `robot.joints`: joints with limits, drive, gains and live state.
- `robot.drive {type: "position", kp: 40, joints}`; `robot.fix_base {on}`.
- Targets: `robot.set_targets {targets: {"FL_hip_joint": 0.1}}` (rad or m), then step the simulation.
- Poses: `robot.pose_capture {name}`, `robot.pose_apply {name, duration}`, `robot.pose_initial {name}`
  (the start state after `sim.reset`).
- `robot.gain_test {joint, kind: "step"}` fits rise, overshoot and settling, then read `robot.gain_status`.
- IK: `robot.ik {link, pos, quat, apply: true}`.
- Assembly: `robot.attach_asset {path, parent, mount}` mounts another asset (a hand, a sensor) on a link.
- Clips: `clip.new {name, duration}`, `clip.key_pose {time, pose}`, `clip.play {loop: true}`.

## Simulating and recording

- `sim.play`, `sim.pause`, `sim.step {steps: 100}` (paused), `sim.reset`, `sim.speed {value}`.
- `sim.envs {count: 64}` rebuilds the world with that many environments; `view.envs {show: 16}`.
- Pushing a body: `sim.perturb {path, force: [0, 0, 50]}`, then `sim.release {path}`.
- Timeline: `timeline.status`, `timeline.seek {tick}`, `timeline.fork {label}` to try a variation,
  `timeline.export {path: "C:/out/run.mp4", fps: 30}`; its progress shows in `timeline.status`.
- Rerun: `rerun.stream {on: true, every: 4}` opens a local Rerun viewer and logs every link and body
  pose over the `tick` timeline; `file: "C:/out/run.rrd"` writes a file instead, `addr` sends to a
  running viewer, and `rerun.stream {on: false}` ends it and reports the sample count. It needs
  rerun-sdk in the editor's Python.
- Demonstrations: `teleop.use {on: true}`, `demo.record {on: true}`, ..., `demo.record {on: false}`,
  `demo.save {path: "C:/out/demos.hdf5"}`.

## Sensors, labels and datasets

1. `sensor.add {type: "camera", mount: "go2/base", position: [0.3, 0, 0.05], camera: {width: 640,
   height: 480}}`; other types: imu, contact, force_torque, frame_pose, joint_state, depth, lidar,
   range_scan, linear_velocity, tactile.
2. `sensor.read {sensor, channel: "color"}` renders one reading now (`path` saves it as .npy).
3. `label.set {label: "robot", path}`: semantic classes for the segmentation channels.
4. `dr.set {render: {on: true, color: 0.2}, physics: {on: true, mass: [0.8, 1.2]}}`.
5. `dataset.export {out: "C:/data/run1", frames: 100, channels: ["rgb", "depth", "semantic"]}`, then
   `wait_for {id: "dataset.status", until: {running: false}, timeout_s: 600}`.

## Training a locomotion policy

1. `train.tasks` lists the tasks, `train.config {task: "go2_velocity"}` switches the document to one and
   reports its configuration and `errors`.
2. `train.set {values: {"run.num_envs": 2048, "run.max_epochs": 500}}`; `train.validate` must be `ok`.
3. `train.python` must report `ready` (torch, rl_games, a GPU); `train.python {path}` picks another
   Python.
4. `train.start {name: "go2-walk-1"}` trains in another process. Training takes minutes to hours; check
   `train.status {name}` (state, epoch, reward) now and then rather than waiting in one call.
5. `train.observe {on: true, envs: 16, follow: true}` shows the run's environments in the viewport with
   arrows of commanded and actual velocity; `train.panel {page: "dashboard"}` plots it;
   `train.metrics {name}` returns the numbers.
6. `train.stop {name}` ends it after the epoch with a final checkpoint.
7. `train.checkpoints {name}`, then `train.play {name, kind: "best"}` loads the policy onto the scene's
   robots and plays. `policy.status` shows what drives them.

## Rendering and export

- `screenshot {viewport: true}` (tool) for a quick look. Aim with `scene.select {path}` and
  `view.frame_selected`, or `view.camera {target, distance, yaw, pitch}`.
- `view.render {path: "C:/out/shot.png", width: 1920, height: 1080, spp: 64}` path-traces a still in
  another process; `wait_for {id: "view.render_status", until: {running: false}}`.
- `file.export_usd {path: "C:/out/stage.usda"}` writes UsdPhysics; `node` exports one subtree.
