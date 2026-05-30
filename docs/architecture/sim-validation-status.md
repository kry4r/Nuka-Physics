# Nuka Sim-Correctness Validation Status (v0.3 sim-val track)

> Date: 2026-05-30 · Branch: `v03` · Scope: the sim-val track (#32 PD standing fix, #33 C ABI
> drive/readback, #34 policy conventions, #35 g1/h1 pipeline, #36 headless render). Reviewed by an
> independent skeptical pass (read-only) before sign-off.
> **Status: validated for the static standing case + the drive/readback pipeline; explicit gaps below.**

## What IS validated (strongest honest claim)

A free-floating 6.92 kg Go2 trunk (`examples/scenes/go2_float.usda`, `FloatingBase` root, 18 DOF),
PD-driven (Kp=60 / Kd=4) to the **real** Go2 crouch (hip ±0.1 / thigh 0.8 / calf −1.5), **holds that
crouch** on Nuka's batched ABA + PGS-contact stepper. Measured at the settled tail
(`tests/runtime/test_go2_pd_standing.cpp`, re-run by the controller):

| Gate | Measured | Bound | Role |
| --- | --- | --- | --- |
| (A) worst joint tracking | **0.10 rad** | < 0.35 | **the discriminator** — excludes a sprawl |
| (B) trunk-above-feet / drift / seat-err | **0.303 m / 0.0 / 8.9e-5** | >0.20 / <5e-3 / <2e-2 | excludes collapse |
| (C) Σλ_n/dt ÷ Mg | **1.0000** | ±10% | weight support (see note) |
| (C) per-foot N | 34.8/34.8/39.8/39.8 | ≈Mg/4 ±60% | coarse lifted-foot catch |
| (D) tilt / base ω / v_z | **0.0044 / ~0 / ~0** | <0.20 / <0.10 / <0.05 | no tipping |
| (E) determinism (2-run + cross-replica) | **0 / 0** | exact | reproducibility (not correctness) |
| (F) 4096-env non-finite | **0** | 0 | runnability + 2-run/replica bit-exact |

**The discriminating evidence is joint tracking (A) + height (B).** Σλ≈Mg (C) is deliberately the *weak*
gate: it holds for **any** vertical equilibrium — including the collapsed sprawl that the soft-gain T8b
test settles into — so it cannot, alone, distinguish standing from sprawl. Determinism (E) proves
reproducibility (no float atomics, fixed reduction order), **not** physical correctness. The 4096-env
gate (F) certifies standing only via "every replica reproduces env0 bit-for-bit" + env0's tracking
gate; the 4096 test's own standing spot-check (`trunk z > ground`) is weak by itself.

**Drive + readback pipeline** (`#33` C ABI + `#35` Python harness) is validated end-to-end:
`torch → ctypes/C ABI → Nuka → state readback`, 8-rung diagnostic ladder all green, closed loop
bit-exact deterministic. The `DRIVE_TARGET` write→solver path is real (sign-correct q response,
not coincidental). Layout: env-major, `base_link_count = 13` for Go2 (slot 0 = root, slots 1–12 =
the 12 actuated leg joints), link pose = 7 floats `[px,py,pz, qw,qx,qy,qz]` — **quaternion is wxyz /
w-first**. Caveats: `ARTICULATION_LINK_POSE` lags `JOINT_POSITION` by one step; the C ABI drive/readback
is exercised on the fixed-base `go2_stand` scene by its unit test (the live floating-base pose path is
covered by the C++ floating-base tests, and by the `#35` harness on `go2_float`).

## What is NOT validated (honest gap list — do not assume these are covered)

1. **Energy drift < 2% over 1000 steps** — a **v0.3 §7 EXIT criterion**. Currently measured only on v0.1
   rigid-body smoke scenes (`tests/data/smoke/*.yaml`), **never on the Go2 articulated + contact + PD
   stepper**. Open for v0.3 closure.
2. **Walking / locomotion dynamics** — none. The track is purely the **static** stance. (The g1/h1
   harness is plumbing-only; under it the Go2 base intentionally collapses.)
3. **Friction / tangential contact (λ_t)** — μ=0.8 exists in the solver, but a straight-down stance on
   flat ground imposes ~no tangential demand → the friction cone is not meaningfully exercised.
4. **Joint-limit enforcement** — there is **no limit-clamp code** in the articulation path; USDA joint
   limits are declarative only. The crouch happens to stay in range; limits are unvalidated/unenforced.
5. **Restitution / bounce** — not implemented in the articulated contact path; resting inelastic only.
6. **Varied contact configs / non-flat ground / >4 contacts** — only 4 feet on one flat plane.
7. **No external oracle** for the floating-base *contact* equilibrium (T8a's momentum oracle is
   contact-free and drive-free).
8. **Base linear/angular velocity is not exposed by the C ABI** (it lives in `link_velocity[root]`). A
   real Go2 48-d observation needs it → a `NUKA_FIELD_LINK_VELOCITY` follow-up before policy inference.
9. **Fixed-base `go2_stand` load-bearing** is never tested (base pinned). Note `go2_stand.usda` still
   carries the latent foot-at-knee geometry (inert there); the two Go2 scenes now have **different foot
   geometry** — a latent trap if anyone cross-compares them. Left as an owner proposal.

## Provenance note (foot offset)

The foot-sphere offset added in `#32` is `(0,0,-0.213)` — the correct **real-Go2 calf length**. It was
read from the Unitree URDF `FL_foot_joint` origin in the cloned repo
`/root/third_party/unitree_rl_gym/resources/robots/go2/urdf/go2.urdf` (an out-of-tree clone, documented
in `docs/plans/2026-05-30-v03-unitree-policy-config.md`). The commit-#32 message's bare "go2.urdf"
refers to that external clone, **not** an in-repo file — the only in-tree URDF, `examples/scenes/go2_stand.urdf`,
is a degenerate fixed-strut "stand" model and is **not** the source. Treat the USDA scene as ground truth.

## Pointers

- Standing test: `tests/runtime/test_go2_pd_standing.cpp` (commit 8d10156). Floating-base contact/ABA it
  builds on: `tests/runtime/test_floating_base_contact.cpp` (T8b), `test_floating_base_aba.cpp` (T8a).
- C ABI: `src/c_abi/buffer.cpp`, `src/include/nuka/nuka.h`, test `tests/c_abi/test_drive_target_io.cpp` (0a6a951).
- Pipeline harness: `examples/sim_val/` (d652973). Policy conventions: `docs/plans/2026-05-30-v03-unitree-policy-config.md`.
