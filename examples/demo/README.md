# Demos

Pretrained policies and material experiments run in Nuka. Video frames come from live simulated worlds or replay of their saved states. The [homepage gallery](../../README.md) links to the complete recordings.

| Demo | Entry point | Recording |
|---|---|---|
| π0.5 inference | [libero_pi05_play.py](libero_pi05_play.py) | [12 s, 1280 × 720](https://github.com/kry4r/Nuka-Physics/raw/master/docs/media/pi05_libero.mp4) |
| G1 Shuffle dance | [g1_dance_play.py](g1_dance_play.py) | [20 s, 960 × 540](https://github.com/kry4r/Nuka-Physics/raw/master/docs/media/g1_dance.mp4) |
| Elastoplastic compression | [elastoplastic_compression_demo.cpp](elastoplastic_compression_demo.cpp) | [24.04 s, 1600 × 1000](https://github.com/kry4r/Nuka-Physics/raw/master/docs/media/elastoplastic_compression.mp4) |
| Bunny elastoplastic impact | [elastoplastic_bunny_demo.cpp](elastoplastic_bunny_demo.cpp) | [18.04 s, 1600 × 1000](https://github.com/kry4r/Nuka-Physics/raw/master/docs/media/elastoplastic_bunny.mp4) |

Run commands from the repository root after the [CUDA build and Python installation](../../README.md#quick-start). Use Python 3.11, a CUDA-compatible PyTorch installation, Pillow, NumPy, and `ffmpeg` on `PATH`. Model weights and source assets stay in the ignored `.nuka-assets/` and `.nuka_cache/` directories; recordings and metrics go to `out/`.

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
  --seconds 20 --video --fps 25 --width 960 --height 540 \
  --out out/g1_shuffle
```

The entry point creates the NKS/NKA scene bundle and writes `g1_dance.mp4`, `summary.json`, and `metrics.jsonl`. The showcased 20-second run passed the finite-state, upright, and visible-motion checks: minimum root height was 0.688 m, mean joint motion span was 1.145 rad, and mean joint tracking RMSE was 0.213 rad.

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

The recorded light-load recovery error is 0.0648%; strong loading leaves 33.13% height compression. The [physical report](../../docs/research/2026-09-11-elastoplastic-compression-demo-zh.md) records convergence and energy measurements. The prescribed platens demonstrate material response; they do not validate dynamic gripper coupling.

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
  --width 1600 --height 1000 --samples 128 --render-stride 1
python examples/demo/compose_elastoplastic_bunny.py \
  --capture out/elastoplastic/bunny \
  --frames out/elastoplastic/bunny_render/frames \
  --out-dir out/elastoplastic/bunny_video
```

The bunny remains on the pad, whose center settles 20.82 mm below its initial surface. This is deformation under load. The [physical report](../../docs/research/2026-09-11-elastoplastic-bunny-demo-zh.md) includes timestep/grid comparisons, momentum balance, plastic dissipation, and measured penetration. The current grid contact projects velocity and returns reaction impulses each physics interval; a shared finite-mass solve for multiple contacting owners remains unfinished.

## Go2 locomotion capture

The existing batched Go2 capture uses an externally trained TorchScript policy. It records 16 simulated environments and renders their link poses with the Python skeleton renderer:

```bash
examples/demo/render_video.sh
```

The output is `out/go2_demo/go2_locomotion_16env.mp4`. See [go2_demo_capture.py](go2_demo_capture.py), [go2_demo_render.py](go2_demo_render.py), and the [policy validation notes](../sim_val/go2_policy_drive_README.md) for the observation contract and validation. This capture is separate from the mesh-rendered skill videos in the homepage gallery.
