# Demos

Pretrained policies and material experiments run in Nuka. Video frames come from live simulated worlds or replay of their saved states. The [homepage gallery](../../README.md) links to the complete recordings.

The material examples load [Nuka Dynamics Lab](../../docs/nuka-stage.md), a shared NKS/NKA asset with a layered experiment deck, warm grey panels, avocado green edges and inset signage. Use `--environment` to select a layout; edit its materials, lights and cameras in NKS. The guide covers asset editing and replay rendering.

| Demo | Entry point | Recording |
|---|---|---|
| π0.5 inference | [libero_pi05_play.py](libero_pi05_play.py) | [12 s, 1280 × 720](https://github.com/kry4r/Nuka-Physics/raw/master/docs/media/pi05_libero.mp4) |
| G1 Shuffle dance | [g1_dance_play.py](g1_dance_play.py) | [20 s, 1920 × 1080](https://github.com/kry4r/Nuka-Physics/raw/master/docs/media/g1_dance.mp4) |
| G1 clothed water course | [g1_wading_demo.py](g1_wading_demo.py) | [Setup and training](../assets/g1_wading/README.md); traversal under validation |
| Elastoplastic compression | [elastoplastic_compression_demo.cpp](elastoplastic_compression_demo.cpp) | [24.04 s, 1600 × 1000](https://github.com/kry4r/Nuka-Physics/raw/master/docs/media/elastoplastic_compression.mp4) |
| Bunny elastoplastic impact | [elastoplastic_bunny_demo.cpp](elastoplastic_bunny_demo.cpp) | [3.63 s, 1920 × 1080](https://github.com/kry4r/Nuka-Physics/raw/master/docs/media/elastoplastic_bunny.mp4) |
| Bunny water drop | [bunny_water_demo.cpp](bunny_water_demo.cpp) | [3.07 s at 2× slow motion, 1920 × 1080](https://github.com/kry4r/Nuka-Physics/raw/master/docs/media/bunny_water_drop.mp4) |
| Dynamic elastoplastic gripper | [robot_elastoplastic_demo.cpp](robot_elastoplastic_demo.cpp) | [Full view](https://github.com/kry4r/Nuka-Physics/raw/master/docs/media/robot_elastoplastic.mp4) · [Contact close-up](https://github.com/kry4r/Nuka-Physics/raw/master/docs/media/robot_elastoplastic_close.mp4) |

Run commands from the repository root after the [CUDA build and Python installation](../../README.md#quick-start). Use Python 3.11, a CUDA-compatible PyTorch installation, Pillow, NumPy, and `ffmpeg` on `PATH`. Model weights and source assets stay in the ignored `.nuka-assets/` and `.nuka_cache/` directories; recordings and metrics go to `out/`.

## Go2 cloth drape

Run `python examples/demo/go2_cloth_drape.py --drape 800 --stride 32 --out out/go2_cloth` to render the coupled robot and VBD cloth. The demo loads [go2_cloth_drape_contact3.nks](../scenes/go2_cloth_drape_contact3.nks), which shares the source Go2 mesh archive and uses 3D tangential Coulomb contact at its four feet. The source [go2.nks](../scenes/go2.nks) retains its authored 6D contact declaration. The current row solver does not implement the source model's 0.02 torsional and 0.01 rolling friction coefficients, so the demo asset is a declared contact-model difference, not a 6D validation result.

## pi0.5 inference

The Franka Panda uses two rendered camera streams and an 8-value robot state to pick up the black bowl and place it on the plate in LIBERO Spatial task 2. Inference, physics, and camera rendering share one local process. The showcased run uses the full 3.62B-parameter checkpoint, OSC control, 500 Hz physics, and 20 Hz actions. It executes eight actions from each predicted 50-action chunk before updating its observation.

The [asset and inference manifest](../vla/libero_spatial_pi05_manifest.json) pins the checkpoint, source scene, normalization, and action contract. Prepare the source repositories:

```bash
mkdir -p .nuka-assets/src .nuka_cache
git clone https://github.com/google-deepmind/mujoco_menagerie.git .nuka-assets/src/mujoco_menagerie
git -C .nuka-assets/src/mujoco_menagerie checkout da76818e269b82289eba39808e2fb91d679d6994
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git .nuka_cache/LIBERO
git -C .nuka_cache/LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
```

Install the LeRobot revision used for the recording, then download the checkpoint and PaliGemma tokenizer. The tokenizer repository requires the corresponding Hugging Face access.

```bash
pip install "lerobot[pi] @ git+https://github.com/huggingface/lerobot.git@0b067df57d21d3a02d6c511f1609172fa39ac29b"
hf download lerobot/pi05_libero_finetuned_v044 \
  --revision 8e174154ef5f6c60a8da12ae99c303d8963138c1 \
  --local-dir .nuka_cache/pi05-libero
hf download google/paligemma-3b-pt-224 \
  --include tokenizer.json tokenizer_config.json special_tokens_map.json \
  --local-dir .nuka_cache/paligemma-tokenizer
python tools/assets/convert_libero_pi05.py
python examples/demo/libero_pi05_play.py \
  --control-backend osc --seconds 12 --execute-steps 8 \
  --seed 20260828 --render-quality high --fps 20 \
  --out out/libero/pi05_black_bowl
```

The demo imports the generated XML with the running engine, records the rollout and policy chunks, and writes `summary.json`, `rollout.npz`, camera images, and `libero_pi05.mp4`. `--skip-video` disables recording. Success requires sustained bilateral grasp contact, transport without excessive slip, policy release, and stable upright support on the plate.

The showcased seed completed the task with these measured results:

| Measurement | Result |
|---|---:|
| Bowl lift | 12.17 cm |
| Sustained bilateral grasp | 1.512 s |
| Maximum drift in the gripper frame during transport | 0.845 mm |
| Plate support over the final 0.5 s | 100% |
| Final placement XY error | 23.95 mm |
| Peak / settled bowl–plate collision-box overlap | 2.086 / 0.322 mm |

These measurements describe the recorded seed. They are not a LIBERO benchmark success rate. The [recording metadata](../../docs/media/demo_recordings.json) includes the model and video hashes. The transport and collision-box audits can also be run on a new recording:

```bash
python python/nuka/tasks/manipulation_metrics.py --run out/libero/pi05_black_bowl
python tools/validation/libero_overlap_audit.py \
  --run out/libero/pi05_black_bowl \
  --scene .nuka-assets/generated/libero/libero_spatial_black_bowl.xml \
  --body-a akita_black_bowl_1_main --body-b plate_1_main
```

## G1 dance

The Unitree G1 runs the pretrained `J_Dance17_Shuffle` ONNX actor from [G1 Moves](https://github.com/experientialtech/g1-moves). Its 160-value observation produces 29 joint actions at 50 Hz; Nuka applies PD control with 200 Hz physics. The reference motion is sampled at 60 Hz. This is inference with an externally trained policy.

The [robot manifest](../assets/g1_mode15/manifest.json) and [motion manifest](../motions/g1/manifest.json) record asset revisions, collision geometry, joint conventions, and policy hashes.

```bash
pip install onnxruntime pillow huggingface_hub
mkdir -p .nuka-assets/src
git clone https://github.com/experientialtech/g1-moves.git .nuka-assets/src/g1-moves
git -C .nuka-assets/src/g1-moves checkout 475fdae98dcf18f96ebd3d0566d12fdefbaf2f0f
git -C .nuka-assets/src/g1-moves submodule update --init mjlab
hf download exptech/g1-moves --repo-type dataset \
  --revision 895d064b2385725e6aabf540461c219afc3e0916 \
  --include 'dance/J_Dance17_Shuffle/training/J_Dance17_Shuffle.npz' \
            'dance/J_Dance17_Shuffle/policy/J_Dance17_Shuffle_policy.onnx' \
  --local-dir .nuka-assets/src/g1-moves
python tools/assets/convert_embodied_assets.py --asset g1
python examples/demo/g1_dance_play.py \
  --seconds 20 --video --fps 30 --width 1920 --height 1080 --spp 32 \
  --environment examples/assets/nuka_lab/stage.nks --out out/g1_lab
```

The entry point creates the NKS/NKA scene bundle and writes `g1_dance.mp4`, `summary.json`, and `metrics.jsonl`. With `--environment`, the lab deck replaces the studio set and is the floor collider. The showcased 20-second run passed the finite-state, upright, and visible-motion checks: minimum root height was 0.645 m, mean joint motion span was 1.131 rad, and mean joint tracking RMSE was 0.211 rad.

## Elastoplastic compression

A single 80 × 80 × 100 mm specimen undergoes light compression, unloading, strong compression, and unloading between prescribed rigid platens. Both loads use the same continuous simulation. Hencky J2 plasticity in the production MLS-MPM solver gives elastic recovery after the light load and permanent deformation after the strong load.

Both material recordings use E = 50 kPa, Poisson ratio 0.30, density 1000 kg/m³, yield stress 12 kPa, and linear hardening 3 kPa. The grid spacing is 5 mm and the particle spacing is 2.5 mm. Physics runs at 7680 Hz; saved states at 120 Hz play at 24 fps, giving **5× slow motion without state interpolation**. The path tracer uses 128 samples per pixel. The line below the specimen shows the contact force history, without labels.

```bash
cmake --build build-cuda128 --target \
  nuka_elastoplastic_compression_demo nuka_elastoplastic_bunny_demo -j
export LD_LIBRARY_PATH="$PWD/build-cuda128/src:${LD_LIBRARY_PATH:-}"
build-cuda128/tests/nuka_elastoplastic_compression_demo \
  --no-render --out-dir out/elastoplastic/compression
build-cuda128/tests/nuka_elastoplastic_compression_demo \
  --replay out/elastoplastic/compression \
  --out-dir out/elastoplastic/compression_render \
  --width 1600 --height 1000 --samples 128 --render-stride 1
python examples/demo/compose_elastoplastic_compression.py \
  --capture out/elastoplastic/compression \
  --frames out/elastoplastic/compression_render/frames \
  --out-dir out/elastoplastic/compression_video
```

The capture includes configuration, particle/body states, per-frame physical measurements, and reset results. Composition checks the physical acceptance criteria before encoding, then verifies the decoded video frame count, size, rate, and duration. Use `--analyze-only` to check a capture without rendering. To reproduce timestep and grid comparisons, capture again with `--steps-per-frame 128`, then with `--dx 0.004 --steps-per-frame 128`, and pass both directories as repeated `--compare` arguments. Compare the two runs at 128 steps per frame to isolate the grid change.

The recorded light-load recovery error is 0.0648%; strong loading leaves 33.13% height compression. Peak particle-center penetration reaches 4.007 mm, and geometric convergence remains open. The prescribed platens demonstrate material response; they do not validate dynamic gripper coupling.

## Bunny elastoplastic impact

A freely falling 2.5 kg Stanford bunny drops 180 mm onto a 240 × 200 × 60 mm material pad. Rendering, collision, center of mass, and inertia use the same closed triangle mesh. The original Stanford scan has five openings; the asset preparation tool caps its boundary loops and writes a separate mesh plus a hash and mass-properties manifest.

Obtain the [Stanford bunny scan](https://graphics.stanford.edu/data/3Dscanrep/) (`bunny/reconstruction/bun_zipper.ply`) and convert its vertex positions and triangle indices to OBJ without remeshing. Place it at `.nuka-assets/stanford/bunny.obj`. The recorded source has 35,947 vertices and 69,451 triangles; its identity is included in [recording metadata](../../docs/media/demo_recordings.json).

```bash
python tools/assets/prepare_solid_mesh.py \
  .nuka-assets/stanford/bunny.obj .nuka-assets/generated/bunny_solid_120mm.obj \
  --extent 0.12 --y-up
build-cuda128/tests/nuka_elastoplastic_bunny_demo \
  --no-render --out-dir out/elastoplastic/bunny
build-cuda128/tests/nuka_elastoplastic_bunny_demo \
  --replay out/elastoplastic/bunny \
  --out-dir out/elastoplastic/bunny_render \
  --width 1920 --height 1080 --samples 128 --render-stride 4
python examples/demo/compose_elastoplastic_bunny.py \
  --capture out/elastoplastic/bunny \
  --frames out/elastoplastic/bunny_render/frames \
  --out-dir out/elastoplastic/bunny_video --fps 30 --frame-stride 4
```

The recording renders every fourth 120 Hz state at 30 fps, so it plays in **real time** without state interpolation. The pad carries 80 Pa·s of Kelvin–Voigt viscosity on top of the Hencky J2 stress, so the bunny's rocking on the soft pad dies out within about 2.5 s instead of ringing through the lossless elastic range. The impact peaks at 270 N and settles to the bunny's 24.5 N weight. The bunny indents the pad by 26.2 mm and stays on it; the pad center settles 13.9 mm below its initial surface. This is deformation under load. Particle-center penetration peaks at 0.72 mm with a 5 mm contact envelope; the final bunny speed is 1.9 mm/s. Linear and angular momentum balance, non-increasing energy and the reset check all pass. The bunny, the material and the support share the common contact solve.

## Bunny water drop

A freely falling 0.41 kg Stanford bunny (density 1200 kg/m³, same closed mesh as above) drops 150 mm into a 360 × 280 mm glass tank holding 180 mm of water. The water is weakly compressible MLS-MPM: Tait pressure with bulk modulus 200 kPa and γ = 7, viscosity 1 mPa·s, 669,600 particles at 3 mm spacing on a 6 mm grid. Water keeps only its volume change, carries no tension and no negative pressure, so it splits into cavities and droplets instead of stretching like a gel. Physics runs at 14,400 Hz. The bunny, tank walls and water share the common contact solve, so the Worthington jet, the spray over the rim and the bunny's settling on the floor come from the same two-way coupling. The grid extends 0.5 m past the walls, and spray that lands on the deck grips it with friction 0.5. States saved at 60 Hz play back at 30 fps, so the recording runs in **2× slow motion**; it shows the first 1.53 s, from the drop through the jet and the spray settling.

```bash
cmake --build build-cuda128 --target nuka_bunny_water_demo -j
build-cuda128/tests/nuka_bunny_water_demo \
  --no-render --duration 3.5 --out-dir out/bunny_water/capture
build-cuda128/tests/nuka_bunny_water_demo \
  --replay out/bunny_water/capture --out-dir out/bunny_water/render \
  --width 1920 --height 1080 --samples 128
ffmpeg -framerate 30 -i out/bunny_water/render/frames/frame_%06d.ppm \
  -vf hqdn3d=1.2:1.2:2.0:2.0 -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p \
  out/bunny_water/bunny_water_drop.mp4
```

The capture writes `completion.json` with its physics checks: free-fall velocity error 1.9e-5 m/s, maximum particle volume-ratio error 0.18%, no particle escapes the grid, and a passing reset. The bunny enters the water at 0.18 s at 1.6 m/s, the jet peaks 0.60 m above the floor, and the bunny rests on the tank floor by 0.5 s with a final speed of 3.2 mm/s. The water surface is reconstructed from anisotropic particle kernels, reflected across the walls so it meets the glass flat, smoothed, and path traced as a dielectric nested inside the glass walls.

## Dynamic elastoplastic gripper

A Franka Panda applies light compression, opens, applies strong compression, and releases the same 24 × 36 × 28 mm specimen. Both fingers and the arm use articulated dynamics with finite PD forces and sampled gravity feedforward. The controller updates joint targets; contact forces determine the actual joint motion. The specimen has no pinned particles, and its material history is retained between loads.

The production MLS-MPM solver uses Hencky J2 plasticity with E = 50 kPa, Poisson ratio 0.25, density 1000 kg/m³, yield stress 12 kPa, and linear hardening 7.5 kPa. Grid nodes, rigid bodies, and articulated links exchange impulses through the common contact solver.

The published videos are a **visual preview** at 4 mm grid spacing. Maximum particle-center penetration is 3.996 mm against a 2 mm acceptance budget; grid, timestep, and iteration convergence remain pending. The other recorded physical checks pass, including elastic recovery, persistent plastic deformation, reaction balance, and reset. Publication retains the failed spatial check in the [recording metadata](../../docs/media/demo_recordings.json).

Prepare the Panda assets from the Menagerie revision in the [asset manifest](../assets/panda/manifest.json), using the source checkout described above:

```bash
python tools/assets/convert_embodied_assets.py --asset panda
python - <<'PY'
import nuka
scene = nuka.Scene.load('.nuka-assets/generated/panda/panda_pick_place.xml')
try:
    scene.save('.nuka-assets/generated/panda/panda_pick_place.nks')
finally:
    scene.destroy()
PY
cmake --build build-cuda128 --target nuka_robot_elastoplastic_demo -j
export LD_LIBRARY_PATH="$PWD/build-cuda128/src:${LD_LIBRARY_PATH:-}"
build-cuda128/tests/nuka_robot_elastoplastic_demo \
  --no-render --dx 0.004 --substeps 64 --execution graph \
  --contact-capacity 32768 \
  --out-dir out/elastoplastic/gripper
build-cuda128/tests/nuka_robot_elastoplastic_demo \
  --replay out/elastoplastic/gripper \
  --out-dir out/elastoplastic/gripper_render \
  --width 1280 --height 800 --samples 48
python examples/demo/compose_robot_elastoplastic.py \
  --capture out/elastoplastic/gripper \
  --frames out/elastoplastic/gripper_render/frames \
  --out-dir out/elastoplastic/gripper_video --visual-preview
```

Captures contain the scene bundle, particle and robot states, physical measurements, configuration, and reset result. The composer checks physical acceptance before encoding both views. `--visual-preview` explicitly permits encoding despite failed physical checks, retains every check in `analysis.json`, and labels playback as a preview. Capture integrity, finite state, execution mode, environment status, contact capacity, and reset must still pass. `--analyze-only` checks a completed capture without rendering; repeated `--compare` arguments check runs that change one of `--dx`, `--substeps`, or `--velocity-iterations`. Full physical acceptance requires all checks and all three convergence comparisons. A contact-pool overflow invalidates a capture even when the step call succeeds.

State capture saves two alternating checkpoints every 12 samples. To continue an interrupted capture, repeat the capture command with `--resume`, using the same executable, engine library, scene, and physical arguments. Keep both `scene.nks` and `scene.nka` with the recording. Recovery checks file integrity and restores material history, contact caches, controller inputs, and curve integrals; incomplete output after the last valid checkpoint is recomputed. `--stop-after N` ends a capture segment after sample N without marking the complete demo as finished. `--execution eager` selects the same operator sequence without graph replay.

The two cameras replay the same saved states. A 120 Hz capture plays at 24 fps: **5× slow motion without state interpolation**, covering 2.8 seconds of simulated motion. The line below the scene shows mean finger reaction. Penetration is measured from particle centers to the authored collision geometry, including Panda pad boxes. The rendered material surface is reconstructed from particles. Actuator work and gravity torque are sampled estimates; the recording does not claim a closed total-energy budget or validate lifting and transport.

## Go2 locomotion capture

The existing batched Go2 capture uses an externally trained TorchScript policy. It records 16 simulated environments and renders their link poses with the Python skeleton renderer:

```bash
examples/demo/render_video.sh
```

The output is `out/go2_demo/go2_locomotion_16env.mp4`. See [go2_demo_capture.py](go2_demo_capture.py), [go2_demo_render.py](go2_demo_render.py), and the [policy validation notes](../sim_val/go2_policy_drive_README.md) for the observation contract and validation. This capture is separate from the mesh-rendered skill videos in the homepage gallery.
