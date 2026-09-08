# Demos

Pretrained policies run in Nuka, with video frames captured from the live simulated worlds. The [homepage gallery](../../README.md) links to the complete recordings.

| Demo | Entry point | Recording |
|---|---|---|
| π0.5 inference | [libero_pi05_play.py](libero_pi05_play.py) | [12 s, 1280 × 720](https://github.com/kry4r/Nuka-Physics/raw/master/docs/media/pi05_libero.mp4) |
| G1 Shuffle dance | [g1_dance_play.py](g1_dance_play.py) | [20 s, 960 × 540](https://github.com/kry4r/Nuka-Physics/raw/master/docs/media/g1_dance.mp4) |

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

## Go2 locomotion capture

The existing batched Go2 capture uses an externally trained TorchScript policy. It records 16 simulated environments and renders their link poses with the Python skeleton renderer:

```bash
examples/demo/render_video.sh
```

The output is `out/go2_demo/go2_locomotion_16env.mp4`. See [go2_demo_capture.py](go2_demo_capture.py), [go2_demo_render.py](go2_demo_render.py), and the [policy validation notes](../sim_val/go2_policy_drive_README.md) for the observation contract and validation. This capture is separate from the mesh-rendered skill videos in the homepage gallery.
