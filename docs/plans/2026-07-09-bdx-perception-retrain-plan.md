# BDX Perception Retrain & Full-Media Training — Task Design

Date: 2026-07-09
Status: **APPROVED-PENDING-GO** — execution of every track below is gated on the owner's
explicit go order. This document is the deliverable of the planning phase.

---

## 1. Goal

A closed-loop BDX (open_duck_mini_v2) policy traverses the one-shot corridor under real
physics — platform → two 5 cm steps + door-frame cloth (D01) → gravel bed (D02) → debris
field (D03) → hanging slab (D04) — using terrain **perception** (depth camera, RGB
co-designed), with the **full-media world as the training end state**: the duck learns to
walk *on/in* the gravel (feet submerged), leaving persistent footprints. Afterwards: scene
visual upgrade and the final one-shot film.

## 2. Owner decisions this plan encodes

1. **Perception retrain** — blind-policy geometry workarounds are abandoned.
2. **Depth-first, RGB co-designed** — training vision starts as low-res depth (the
   legged-RL standard), but the RGB pipeline (device-realistic 640×480 render →
   resize 224² → frozen pretrained encoder, the VLA convention) is designed now and
   built as a distillation phase, not deferred indefinitely.
3. **The sensor must see MPM media** — gravel/debris must appear in depth (and RGB).
4. **Full-media training is the end state** — phased is acceptable (rigid first), env
   count may be reduced; the largest medium-induced step cost must be attacked.
5. **Robot stays BDX** — plus imitation-reward gating and exactly two honest model
   fixes (true TPU sole collider, mass audit). No invented DOF, no above-spec torque.
6. **Order: BDX modeling (T1) and MPM/media speedup (T2) first.** Nothing starts
   before the go order. This plan is committed and pushed.

## 3. Evidence base (5 recon tracks, 2026-07-09)

| # | Finding | Evidence |
|---|---------|----------|
| F1 | Rigid-only real corridor batched training is feasible **today**: N=4096 = 112 ms/step = **36,452 env-steps/s** (faster than the v9 heightfield run). True sharp box step edges — the exact contact problem that defeated v9. | measured, GPU probe scripts under `/data/.../activate/tmp/bdx_*probe.py` |
| F2 | Media-on from-scratch PPO is infeasible: 419 ms/step at N=1; **67% (283 ms) is `ReadoutContactWrench`** — a readout BDX never consumes, scanning ~50k contact rows inflated by MPM particles that never emit rows (`src/phi/backend_cuda/ops/readout.cu:95,123,341`; slots reserved at `src/collision/cross_system_query.hpp:42`, early-return at `src/phi/backend_cuda/ops/narrowphase_body_particle.cu:426`). MPM solve itself is 116 ms (27%), P2G gather over a 15×-oversized grid (`mpm.cu:433,478`; sizing `cook_to_model.cpp:2536`). VRAM ≈ 18 MB/env → 75 GB at N=4096. | measured + file:line |
| F3 | Batched media cook is fully supported (per-env replication, `SceneBuilder.build`), but the RL entry point `create_from_scene` rejects media scenes (`src/c_abi/world.cpp:508`) — small plumbing. | code |
| F4 | Heightfield collidables cannot represent a vertical step edge at any resolution (bilinear grid → one-cell ramp; v9 trained on ~31° ramps). Mechanistic root cause of the v9 transfer failure. Height-scan is blind to rigid boxes; no general raycast exists (dead code only). | `heightfield_sample.hpp:28`, `convex_narrowphase.hpp:95` |
| F5 | Batched sensor core: per-env base/link-mounted cameras already follow each env's robot **on-device, zero host traffic** (`sensor_scatter.cu`, `batched_sensor_render.cu:814`); DEPTH AOV exists; **no depth-only fast path** (full shading + shadow ray always run, `batched_sensor_render.cu:196-373`); zero-copy DLPack is default. **The sensor scene contains only rigid `RenderInstance`s — media particles are invisible to it** (`src/c_abi/sensor.cpp:51-90`). Per-env instance hard cap 4096 (`prim_id.cuh:42`) forbids per-particle instances (~15k gravel). | code |
| F6 | VLA practice: device cameras are VGA+, but every frontier system encodes at 224²–384² via frozen SigLIP/DINOv2-class ViTs (all BC, none RL). Legged vision RL universally trains on ~50–90 px **depth** + small CNN, usually privileged-teacher → DAgger student. Our compute: 64²-class depth at N=4096 fits (~10 ms/step); SigLIP-B-class encoding fits only N≤256 — enough for distillation, not for RL exploration. | R5 report w/ URLs |
| F7 | BDX: 5 cm riser = 31% of the 0.162 m stand height (2× Go2's relative step). The imitation reference is flat-ground-only (240 velocity bins, no terrain dim) and `W_JOINT_POS = 15` dominates — it actively fights step-descent adaptation (`bdx_rewards.py:45`, `bdx_reference_motion.py:29`). The box foot plate is a stand-in for a mis-cooked TPU sole mesh (`bdx_author.py:9,69`). Total modeled mass 2.107 kg with `head_assembly` 0.4066 kg an outlier. The real robot carries a camera → giving the policy vision is honest. | code + MJCF |

---

## 4. Tracks and tasks

### T1 — BDX modeling fixes (Fable) — **first, with T2**

- **T1.1 True TPU sole collider.** Root-cause and fix the mesh-sole cook failure
  (currently produces a garbage-AABB placeholder; a flat box stands in). General
  mesh-cook path fix — no BDX special case. If the general fix is disproportionate,
  the fallback is an authored multi-primitive sole (heel/toe boxes) matching the TPU
  footprint — still authored data, not engine special-casing.
  *Acceptance:* sole cooks to a sane collider; stand gate and propulsion gate hold or
  improve; frozen canaries byte-identical (Go2 `0.923080623`, cloth FNV
  `777503423208024307`, MPM exact-value gates).
- **T1.2 Mass/inertia audit.** Verify the 2.107 kg total and the 0.4066 kg head against
  the vendor spec / MuJoCo reference; determine whether it is an import defect (fix
  generally) or faithful (document and close).
- Explicitly rejected: ankle-roll DOF, torque/kp beyond sts3215 spec, toe/heel joints.

### T2 — Media/MPM step-cost reduction (Fable; T2.1 delegable to Opus) — **first, with T1**

Target: media-on corridor step ≤ ~80 ms at N=1 (from 419) and ≤ ~8 MB/env, making
N=256 media-on fine-tuning practical (~2 GB state, ~10–20 ms/step amortized).

- **T2.1 Gate `ReadoutContactWrench` on consumption** (−283 ms when unconsumed; zero
  risk — consumers get the unchanged op). Emission site `src/nk/pipeline/pipeline.cpp:623`.
- **T2.2 Stop reserving body-particle contact slots for MPM slices** under MpmXpbd
  (rows ~50k → rigid-only; shrinks readout/assemble/solve and ~5–6 MB/env). Gate on the
  MpmXpbd slice; pure-XPBD/soft/fluid row layouts must stay byte-identical.
- **T2.3 Right-size the MPM grid + coarse training scene.** Grid is 15× oversized for
  the particle set (loft headroom dominated); training scene uses coarser spacing
  (0.013 → ~0.026) and reduced substeps (14 → 7) — authoring-level, film keeps the fine
  grid. Validate granular stability at the coarse setting.
- **T2.4 (contingent) P2G gather rewrite** (cell-linked-list / skip empty cells).
  Only if Phase B is still too slow after T2.1–T2.3. HIGH risk: moves MPM exact-value
  gates → default-off flag or provably identical, regen only with owner sign-off.

*Acceptance for T2:* measured step-time table before/after at N∈{1,64,256}; all frozen
canaries byte-identical (T2.1/T2.2 do not touch pure-media or particle-free paths;
T2.3 is per-scene).

### T3 — Sensor: depth + RGB co-design (Opus per controller spec; T3.2 Fable if BLAS work)

- **T3.1 Depth-only fast path.** General flag: when only DEPTH/PRIM are requested,
  return after the primary hit — skip shadow ray, `ReconstructHit`, `ShadeDirect`
  (`batched_sensor_render.cu:196-373`). Roughly halves trace cost; benefits every
  sensor consumer.
- **T3.2 Media visible to sensors** (owner requirement). The sensor path shares
  `BuildRenderScene`'s rigid instances only. Design options, one general path shared
  with beauty where possible: (a) per-medium particle-sphere BLAS rebuilt/refit from
  live particle positions per render (respects the 4096-instance cap: one instance per
  medium, not per particle), or (b) a granular surface proxy (grid-derived surface for
  depth). Preferred: (a) — it serves depth *and* the RGB/film path with one mechanism.
  *Acceptance:* depth image of the corridor shows the gravel bed and debris; beauty
  render unchanged where it already drew particles.
- **T3.3 BDX head camera.** Attach at the real robot's camera mount pose. Training
  camera: depth, ~64×48 (Extreme-Parkour-class; deployment = real camera downsampled to
  match). Eval/film/distill camera: RGB 640×480, same mount. Frame-skip knob (render
  every k control steps, hold last) at the env level.
- **T3.4 Obs wiring.** Depth patch (normalized, clamped) concatenated into the flat obs
  (44 + patch); plain MLP first; small-CNN NetworkBuilder as a follow-up only if the
  flat patch underfits. Obs-dim change ⇒ cold start (Go2 precedent).
- **T3.5 RL media entry plumbing.** Route the batched RL env through the
  `SceneBuilder.build` path so media scenes load (F3).
- **Pre-flight (execution day 1):** re-bench the sensor on the real corridor scene
  (7.6 ms figure is from a synthetic 36-box bench); audit instance count vs the 4096
  cap; pair `corridor_nomedia.nks` with its `.nka`.

### T4 — Training v10 (Opus runs; controller writes the recipe and holds the gates)

- **Phase A — rigid corridor, depth policy, N=4096.** The real corridor geometry
  (media off), spawn spread along the track, forward-biased commands. Imitation gating:
  scale `W_JOINT_POS` down near step edges / off flat ground (training-side reward
  shaping; the flat-gait prior must not fight descent — F7). Cold start with depth obs.
  *Gate:* realsim closed-loop — clears both 5 cm steps, no fall, ≥0.12 m/s average,
  reaches x ≥ 3.8 m with media present at eval.
- **Phase B — media-on fine-tune, N=64–256** (needs T2, T3.2/T3.5). Warm-start from
  Phase A. Gravel emphasis: deepen the bed per storyboard (feet visibly sink), footprint
  persistence is already MPM behavior (CapsuleFootprint gate). Randomize spawn inside
  the gravel span.
  *Gate:* stable gait crossing the full gravel bed with submerged feet; tracks persist;
  no regression on the rigid segments.
- **Phase C — RGB student (co-designed VLA-style).** Privileged/depth teacher →
  DAgger-distilled student consuming 640×480 render → resize 224² → **frozen**
  SigLIP-B (or ViT-S) embedding, N=128–256. No RL exploration in this phase, so the
  encoder cost fits. The same pipeline (render → resize → frozen encoder) is the
  reusable substrate for the π0.5 zero-shot track and the IHI inference engine.
  *Gate:* student matches the teacher's corridor traversal in realsim.
- Training on GPU 1 only; no concurrent GPU jobs; checkpoints under
  `/data/.../activate/out/bdx_walk_v10*`.

### T5 — Film & scene visual upgrade (after T4 Phase A/B gates)

Known art debt: irregular gravel grain shapes, rope as a smooth tube (not bead chain),
a better-looking hanging slab, lift the dark far-end lighting, de-plastic the duck
materials, close the sky strip over the far wall. Then the continuous one-shot render
(sweep2 camera plan), frame-by-frame review, owner acceptance.

---

## 5. Sequencing

```
[owner GO]
   ├── T1 (Fable)  ──┐            BDX modeling          — first, per owner
   ├── T2 (Fable)  ──┤            media/MPM speedup     — first, per owner
   │                 ▼
   ├── T3 spec → T3.1/3.3/3.4/3.5 (Opus) ── T3.2 (Fable/Opus)
   │                 ▼
   ├── T4-A rigid+depth N=4096  (≈6 h/run)
   │                 ▼
   ├── T4-B media fine-tune N=64–256   (needs T2, T3.2, T3.5)
   │                 ▼
   ├── T4-C RGB distill student        (needs T4-A/B teacher)
   │                 ▼
   └── T5 film + visual upgrade
```

T1/T2 lead; T3 implementation may overlap once T1/T2 are dispatched (different code
areas; GPU stays free for T2 measurements). Every engine-touching milestone ends with
an adversarial (refute-first) review before merge; the still-unreviewed scene-branch
engine deltas (`c04e0dc`, `28bb0db`, `91d1834`, Phase B set, uncommitted SDF-bake
contact fix) join the same review queue before any merge to `windows-editor`.

## 6. Model tiering (owner 2026-07-09)

Controller: planning, specs, gates, image review. Sonnet: simple recon. Opus: training
runs and spec'd script/plumbing changes. Fable: T1 modeling, T2 speedup, T3.2 if it
becomes BLAS-level work, T2.4 if triggered.

## 7. Frozen canaries (byte-identity unless owner signs a regen)

Go2 `0.923080623` · cloth FNV `777503423208024307` · MPM ConeRepose sand35
h 0.0395 / r 0.0740 / angle 28.3 · CapsuleFootprint depth0 0.0214 · Cohesion coh15
h 0.1195 · OpaqueBeautyNoOpChecksum `15275657819673905153`.
