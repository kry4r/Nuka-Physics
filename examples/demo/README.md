# Go2 locomotion demo video (v0.3 exit-criterion #6)

This directory produces the v0.3 **exit-criterion #6 deliverable -- a Go2
locomotion video** -- by **replaying the ALREADY-PROVEN trained Go2 policy**
(`motion.pt`, from unitree_rl_gym PR#62) in Nuka and rendering the rollout.

## HONESTY (read this first)

- This video shows an **externally-trained policy** that was then **validated in
  Nuka** (sim-val #41/#43: dx +2.93 m / 6 s, vx 0.488 ~= cmd 0.5, tilt < 5.4 deg
  at native dt=0.005). The capture here reproduces those numbers
  (16-env mean net dx +2.78 m / 6 s, mean vx +0.46 m/s, final tilt max 5.5 deg).
- It **satisfies exit #6** ("a Go2 locomotion video").
- It **does NOT satisfy exit #3** ("from-scratch PPO convergence to a stable
  gait"). That is a **separate gate**. Nothing here is evidence of in-house PPO
  convergence; do not present it as a self-trained policy.
- The **video frames** are drawn by a **self-contained Python rasterizer**
  (`go2_demo_render.py`, side-view stick skeleton), **NOT** the offscreen-Vulkan
  path. The Vulkan offscreen path *does* work on this box (verified with a 1-frame
  and a 16-env Go2 smoke via lavapipe), but `nuka_scene_demo` runs its own
  default-drive physics with no way to ingest the policy poses without a C++
  rebuild, and its debug view is top-down (hides the gait). So we render the
  captured world link poses ourselves. The **PPM -> MP4 ffmpeg tail** is the
  documented headless path (`docs/architecture/headless-rendering.md`).

## Pipeline

1. `go2_demo_capture.py` -- imports the PROVEN policy-driving logic from
   `examples/sim_val/go2_policy_drive.py` (so the captured motion is provably the
   same validated obs->action->drive pipeline), runs the policy **batched** across
   16 envs (per-env forward command vx in [0.35, 0.60] -- a spread **around** the
   one command validated in Nuka, cmd=[0.5,0,0]; the off-0.5 speeds are **not**
   separately pre-validated and we make **no** claim about the policy's training
   command range -- each shown env is instead verified to walk by the renderer's
   per-env check, and any that doesn't is flagged), **gates on the walk** (aborts
   if the rollout does not advance forward + stay upright), and writes per-frame
   `ARTICULATION_LINK_POSE` to a compact `.npz`.
2. `go2_demo_render.py` -- rasterizes the `.npz` to PPM frames: a 4x4 grid, each
   cell a side-view (X right, Z up) stick skeleton drawn directly from the world
   link poses, with a follow-cam + **world-anchored scrolling ground ticks** so
   the forward traversal reads. It re-checks each env per-env (forward + upright)
   and flags any that did not walk.
3. `render_video.sh` -- runs 1+2 then encodes PPM -> MP4 via ffmpeg.

## Run

```bash
export CUDA_VISIBLE_DEVICES=0
examples/demo/render_video.sh            # out/go2_demo/go2_locomotion_16env.mp4
# or step-by-step:
python examples/demo/go2_demo_capture.py --envs 16 --seconds 6 --out out/go2_demo/go2_rollout.npz
python examples/demo/go2_demo_render.py  --npz out/go2_demo/go2_rollout.npz --frames-dir out/go2_demo/frames
ffmpeg -y -framerate 30 -i out/go2_demo/frames/frame_%05d.ppm -c:v libx264 -pix_fmt yuv420p out/go2_demo/go2_locomotion_16env.mp4
```

`out/` is gitignored (scratch); the MP4 + `.npz` regenerate from the command
above. Small committed evidence lives in `examples/demo/sample_frames/`.

## 4096-env throughput claim

We render a representative **16-env grid**, not 4096 robots. **No 4096-env run was
performed for this deliverable.** The 4096-env throughput story rests entirely on
the **pre-existing** frozen perf baseline
(`out/perf/baseline_rtx4000ada_4096_frozen.json`); the validation harness
(`examples/sim_val/go2_policy_drive.py`) additionally contains a 4096-env finite
smoke that can be run separately, but it was not executed here.

## Environment caveat (at time of authoring)

A plain `import nuka` was broken in the working tree by an **uncommitted edit** to
`python/nuka/__init__.py` (`BASE_POSE = Field.BASE_POSE`, a Field the built
`_nuka_ext` does not export). That is **not** part of this deliverable. The
capture was run against the committed `__init__.py` via a temporary restore;
`render_video.sh` does the same guard automatically (and restores the working-tree
file byte-for-byte afterward).
