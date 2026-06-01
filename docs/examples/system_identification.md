# Example: gradient-based system identification on Go2 (v0.5)

This is the v0.5 headline demo for differentiable simulation: **recover an unknown
link mass by gradient descent through a differentiable physics rollout.** It is
the end-to-end proof that the diff-sim infrastructure works.

Source: [`examples/demos/go2_system_id.py`](../../examples/demos/go2_system_id.py).
Regression: [`python/tests/test_go2_system_id_convergence.py`](../../python/tests/test_go2_system_id_convergence.py).

## What this is — and isn't (honest framing)

This is a **contact-free, fixed-base, direct-dynamics mass-recovery** demo:

- The Go2's base is **rigidly fixed** (kinematic root) and there is **no ground
  contact** — no foot forces, no walking.
- A constant rest-pose PD target is held while the robot sags under gravity. The
  actuated joint velocities `qdot` after a short transient are a function of the
  link masses through `qddot = M⁻¹ · (τ_PD − τ_gravity)`.
- Minimizing the MSE between simulated `qdot` and a ground-truth `qdot`
  (generated with a known "true" mass) recovers that true mass by gradient
  descent.

**This is mass system identification, not a locomotion / gait demo.** The
gradient does not recover a walking gait — there is no contact in this rollout.
The fixed-base + `obs="qdot"` configuration is the engine's *clean* differentiable
mass channel (a separate gradient-check measured a thigh/calf mass gradient
rel-err of ~1.8e-5 on exactly this setup). Floating-base / contact mass channels
are deferred to a later phase.

## The setup

| Item | Value |
|------|-------|
| Scene | `examples/scenes/go2_system_id.usda` (single-env, fixed-base) |
| Gravity | `g_z = -9.81` |
| Tunable link | one thigh link, **GLOBAL index 2** |
| Rollout length | `K = 30` (the transient regime, where mass sensitivity is largest) |
| True mass | `1.4 kg` |
| Start guess | `0.9 kg` (~0.64× of true — inside the convex basin) |
| Optimizer | Adam, `lr = 0.03`, fixed budget of 80 iterations |
| Success bar | `|recovered − true| / true < 0.01` (1%) |

The gradient is the engine's own deterministic tape adjoint, wrapped as a
`torch.autograd.Function` in `nuka.autograd.differentiable_rollout`. The mass
gradient comes from the engine's `grad_parameters` (a single `tape.backward`),
host→device copied and D1-deterministic — **not** from autograd-tracing
`set_link_mass`. The mass is therefore passed as a **tensor** via `params=`;
calling `.item()` would sever the graph and `mass_param.grad` would stay `None`.

## How the loop works (key engineering facts)

- **The same fixed, deterministic action sequence** (the rest-pose PD target
  repeated `K` times) drives both the ground-truth rollout and every optimization
  rollout.
- **A fresh world + tape every iteration.** `world.reset()` is not supported and
  `tape.reset()` alone is not bit-identical to a fresh rollout (measured
  diff-norm ~0.45), so the demo rebuilds world + tape each iteration (cheap,
  ~6 ms/iter). The world + tape stay alive until **after** `loss.backward()` —
  the adjoint reads the live tape; destroying it first raises
  `Tape.backward: empty tape`. They are destroyed only after `optimizer.step()`.
- **The result is the best-LOSS iterate**, not the final iterate. Adam oscillates
  around the minimum, so an arbitrary cutoff samples that oscillation at an
  arbitrary phase. Best-loss is monotone in iteration count and *answer-blind*
  (loss is the only observable a real system-ID has — selecting by rel-error
  would peek at the unknown being recovered).

## Running it

```bash
CUDA_VISIBLE_DEVICES=0 python examples/demos/go2_system_id.py
```

## Results (real numbers from a run)

| Quantity | Value |
|----------|-------|
| Recovered mass | **1.400448 kg** (true: 1.4) |
| Relative error | **3.2015e-04** — ~31× under the 1% bar |
| Best-loss iterate | iteration **74** of 80 |
| Loss | initial **3.376617e-02** → best **2.542075e-08** (**~1.33e6×** reduction) |
| Mass gradient at start guess | `dLoss/dmass = -1.389748e-01` (genuinely nonzero — the basin is non-flat) |
| Determinism (D1) | **PASS** — two runs give a **bit-identical** recovered-mass sequence |

A condensed trace from the run:

```
iter    0  mass=0.900000  loss=3.376617e-02  rel_err=3.571e-01
iter    1  mass=0.930000  loss=2.973150e-02  rel_err=3.357e-01
iter    4  mass=1.019374  loss=1.929061e-02  rel_err=2.719e-01
iter   20  mass=1.405878  loss=4.370030e-06  rel_err=4.199e-03
iter   60  mass=1.379662  loss=5.251106e-05  rel_err=1.453e-02
iter   79  mass=1.405160  loss=3.367940e-06  rel_err=3.686e-03

RESULT: recovered_mass = 1.400448  true_mass = 1.4  (best-loss iterate @ iter 74)
        relative error = 3.2015e-04  (tol 0.01)
        loss: initial 3.376617e-02 -> best 2.542075e-08 (1328291.4x reduction)
        determinism (D1): PASS
```

## Output artifacts

The demo writes to `examples/demos/go2_system_id_results/`:

- **`run_log.txt`** — the full console log (the source of record).
- **`convergence.csv`** — `iter,mass,loss` for every iteration.
- **`loss_curve.png` / `mass_curve.png`** — written **only if matplotlib is
  installed**; the demo skips them gracefully otherwise (the CSV is the source of
  truth).

## The regression test

A cheaper, fixed-seed version runs in CI as
`python/tests/test_go2_system_id_convergence.py`. It asserts, empirically:

1. the recovered (best-loss) mass is within 1% of ground truth;
2. the loss dropped substantially (best `< 0.1 ×` initial);
3. the mass-gradient channel is non-flat at the start guess; and
4. **D1**: two identical-seed runs give a bit-identical recovered-mass sequence.

> Spec note: the v0.5 plan named this test under `tests/regression/`, but that
> directory holds C++ ctests the pytest gate does not discover, so the runnable
> Python regression lives under `python/tests/` where the suite finds it.

## See also

- The model behind it: [differentiable simulation](../concepts/diff-sim.md).
- The tape internals: [diff-sim tape + checkpointing](../architecture/diffsim-tape.md).
