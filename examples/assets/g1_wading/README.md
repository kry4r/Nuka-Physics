# G1 clothed water and granular course

The course combines an unfixed G1 EDU with 29 driven joints, the original Newton/Style3D jacket, shallow water, and layered sand and gravel. Walking uses finite motor effort and the common contact solver. The jacket retains all 33,703 vertices and 66,930 triangles. Sand and gravel use Drucker–Prager MPM; rendered grains represent material samples.

The robot descends into the pool, crosses a flat pool bottom, climbs the exit stairs, and then enters the sand and gravel bed. The pool bottom has no intermediate raised step. Water uses MLS-MPM with a Tait material; sand and gravel occupy the separate bed beyond the exit landing.

Stable traversal and moving garment contact are still under validation. A dry training rollout does not demonstrate the complete coupled scene.

## Prepare the assets

Install the CUDA library and Python binding using the repository build instructions. The asset builder also needs `usd-core`, `mujoco`, `trimesh`, `scipy`, `numpy`, and `pyyaml`. Obtain the pinned G1 Moves source and its `mjlab` submodule using the [G1 asset instructions](../../demo/README.md#g1-dance).

```bash
python tools/assets/fetch_g1_locomotion.py
python tools/assets/fetch_newton_garment.py
python tools/assets/build_g1_wading.py
```

The builder creates a new `.nuka-assets/generated/g1_coupled_course/` directory. Use `--out` to preserve an existing build. Its manifest records geometry, exclusions, policy parameters, garment provenance, and course dimensions. [robot_collision.json](robot_collision.json) describes the connected ankle housings; [cameras.json](cameras.json) defines observation and recording cameras.

| Scene | Contents |
| --- | --- |
| `flat.nks` | G1 on a flat rigid floor |
| `dry.nks` | G1 and the rigid course, with a solid cover over the granular region |
| `clothed.nks` | The dry course plus the free jacket |
| `wading.nks` | G1, jacket, water, sand, and gravel |

## Train the clothed water and granular course with onboard perception

```bash
python examples/training/train_g1_wading.py \
  --config examples/training/g1_wading_ppo_cfg.yaml \
  --out out/g1_training
```

The default configuration uses `wading.nks`: the free jacket, MLS-MPM water, and sand and gravel are present together throughout training. Episodes allow 35 seconds to reach `finish_x: 8.25`. The water depth is 0.24 m, covering 80% of the 0.30 m depression. It submerges the 0.06 m, 0.12 m and 0.18 m treads on both staircases and reaches the 0.24 m treads. The jacket's physical contact and reaction forces are not replaced by a visual attachment.

The separate `clothed.nks` dry course remains available for diagnosis, not as a substitute for coupled training or full-course acceptance.

The actor receives joint histories, IMU measurements, leg motor effort feedback, and a local elevation grid from one onboard depth camera. The grid contains minimum and maximum visible height, forward and lateral height differences, and a validity mask. It covers 0.15–1.75 m forward and 0.6 m to either side at 0.1 m spacing. Three frames retain their acquisition ages. Depth is sampled at 25 Hz; control runs at 50 Hz. The full-course configuration also supplies onboard RGB-D to the image encoder.

Camera placement comes from the authored `terrain_rgbd` mount. Its root-relative pose is computed from immutable robot calibration and measured waist encoders; gravity alignment uses the IMU estimate. A frame is projected when acquired and delivered with its configured latency and dropout. Unknown cells remain masked. Recording cameras, world poses, scene geometry and contact truth do not enter this actor. Scene truth is confined to training rewards, completion checks, the critic and diagnostics.

`load_source: motor_effort` models onboard motor effort feedback after actuator limits. The Unitree SDK2 [G1 low-state interface](https://github.com/unitreerobotics/unitree_sdk2/blob/9754cd153af3da471b0fe5f3aa535e426fb11db3/include/unitree/idl/hg/LowState_.hpp) exposes IMU and motor state. Sensor noise and the camera mount are illustrative and require hardware calibration. The legacy `foot_wrench` option assumes added six-axis foot sensors; it is not the default hardware assumption for this configuration.

The default `camera.image_observation: true` alongside `terrain` includes onboard RGB-D for water. Appearance, missing depth returns and temporal motor feedback can then be learned together. A first return from water describes its visible surface, not the submerged floor or proof of support. Water recognition and stable wading still require separate validation.

The default onboard RGB-D stream is 192 x 144 for both color and depth. The adaptive image encoder keeps the policy feature shape independent of this capture resolution.

Forward progress is capped at the commanded speed. The reward penalizes velocity error, tilt, angular motion, sliding, effort, and abrupt action changes. Completion requires both feet to cross, the body to be upright, and speed to return within tolerance. Training saves checkpoint weights, optimizer state, seeds, reward terms, episode outcomes, source identity, and a TorchScript actor.

Use `--resume path/to/policy_00010.pt` to continue a checkpoint in a new output directory. Resume restores the optimizer and policy and starts fresh physical episodes. `--initialize path/to/proprioceptive.pt` adds terrain inputs to a proprioceptive checkpoint with a fresh optimizer. Added input weights start at zero; changing the load sensor contract also clears the old load-input weights. A perception checkpoint requires its trained sensor contract.

## Evaluate and render

Evaluate a held-out sensor seed across independent environments before recording a single rollout:

```bash
python examples/training/evaluate_g1_wading.py \
  --policy out/g1_training/policy_00300.pt --envs 32 --seed 20260916 \
  --out out/g1_batch_evaluation
```

Each environment contributes one episode. Falls, leaving the course, timeouts, speed error, attitude error, and saturation are reported separately. The default acceptance requires every environment to complete; a first fall does not stop the remaining environments.

```bash
python examples/demo/g1_wading_demo.py \
  --scene .nuka-assets/generated/g1_coupled_course/clothed.nks \
  --manifest .nuka-assets/generated/g1_coupled_course/manifest.json \
  --policy out/g1_training/policy_00300.pt \
  --seconds 12 --finish-x 2.1 \
  --capture --render --frame-stride 5 --out out/g1_clothed_stairs
```

`--no-sensors` is available for proprioceptive checkpoints. Terrain and image checkpoints require onboard acquisition. `--render` is independent: it records the scene without feeding the recording camera into the actor. The `follow` recording camera shows the robot close up; `--camera wide` shows the water and complete course.

For a complete-scene initial view:

```bash
python examples/demo/g1_wading_demo.py \
  --scene .nuka-assets/generated/g1_coupled_course/wading.nks \
  --seconds 0 --hold --render --camera wide \
  --width 1600 --height 1000 --out out/g1_coupled_initial
```

Select `wading.nks` with the trained policy for a coupled rollout. Initial renders and short finite rollouts do not establish successful traversal. The jacket hem stays above the water; direct cloth–water interaction is checked separately.

Each evaluation saves `metrics.json`, `trajectory.npz`, optional state captures, and rendered frames. The trajectory includes estimated and true gravity for diagnosis, foot poses and loads, joint speeds, actions, effort, and motor saturation. Failed rendered rollouts retain their preview video and return a nonzero exit code. Keep these raw captures under `out/`.

Set `NUKA_CONTACT_SOLVER_DIAGNOSTICS=1` before creating the world to include contact solver errors in each saved sample. The readout measures the last Coulomb block solve before integration, including inactive-force contacts, with maximum errors and global row IDs. Velocity, impulse and work errors retain separate units. It does not measure scalar joint rows, geometric penetration after integration, or the worst error across a whole control interval. The extra GPU work must be enabled in both runs of any timing comparison. Reset and checkpoint restore clear these disposable diagnostics; stepping refreshes them.

RGB-D remains available through `--sensors` and a non-null training camera configuration. Existing camera checkpoints load with their visual encoder and measured image history.
