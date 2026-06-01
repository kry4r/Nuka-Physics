# Diff-sim tape + gradient checkpointing

This is the storage and replay strategy behind Nuka's differentiable rollout. The
goal: a **bounded-memory, D1 byte-exact** reverse pass through a multi-step
contact-free PD rollout, where the backward is the engine's own analytical
adjoint over a recorded tape.

Source:
[`src/diffsim/tape.hpp`](../../src/diffsim/tape.hpp),
[`src/diffsim/checkpoint.hpp`](../../src/diffsim/checkpoint.hpp),
[`src/diffsim/recompute_orchestrator.hpp`](../../src/diffsim/recompute_orchestrator.hpp).
C ABI: [`src/include/nuka/nuka_diffsim.h`](../../src/include/nuka/nuka_diffsim.h).

## The forward this is built on

The differentiable rollout composes one specific contact-free PD step, repeated:

```
tau     = clamp(Kp·(target − q) − Kd·qdot)          (explicit damping)
qddot, a_base = ABA(state, tau, mass, g)
qdot'   = qdot + qddot·dt
v_root' = v_root + (a_base − a_grav)·dt
q'      = q + qdot'·dt
base_pose' advanced
```

This is deliberately **not** `BatchedArticulatedWorld::Step()` — that path folds
deferred drive damping into an implicit `(M + dt·C)⁻¹` solve and runs the full
contact/CRBA pipeline, which the step adjoint does not model. The diff-sim
forward composes the same contact-free kernel sequence the single-step
finite-difference adjoint test validates against, so it is FD-exact on the joint
channel and the floating-base **velocity** channel. (The base **orientation**
channel is deferred — held fixed, not differentiated — and is a v0.7 item.)

## What the tape stores (and what it doesn't)

The tape uses **gradient checkpointing**: it stores cheap per-step inputs, not
expensive intermediates.

- **Per step:** the per-link **action slice** (the drive targets the policy
  applied that step) — `total_link_count` floats, env-major, laid out contiguous
  in one device buffer so step `k`'s slice is a plain pointer offset (zero-copy
  into the backward's inputs). Plus a small host-side `TapeEntry` record
  (step index, whether a checkpoint was captured, the checkpoint slot).
- **Rollout-constant, stored once:** the drive gains (stiffness / damping /
  force-limits), `dt`, and `gravity_z` live on the tape descriptor / orchestrator,
  not per step.
- **The expensive ABA intermediates are NOT stored.** They are regenerated on
  demand by re-rolling the forward from the nearest checkpoint.

All device storage is `phi::Buffer` (sized once at construction); there is no
hot-path allocation.

## Checkpoint cadence and recompute

The tape descriptor (`nuka_tape_desc_t` / `TapeDesc`) controls the strategy:

| Field | Meaning |
|-------|---------|
| `checkpoint_interval` | capture a full authoritative-state snapshot every K steps (step 0 always gets one). Smaller K → less recompute, more memory. |
| `max_tape_entries` | hard cap on recorded steps (device storage sized once). |
| `max_checkpoints` | hard cap on snapshots. |
| `recompute_on_backward` | `1` (default): only every-K-th step gets a checkpoint; the reverse re-rolls each window to regenerate intermediates. `0` (debug): **every** step gets a checkpoint, so the reverse is a single pass with no recompute — this full-tape mode is the CI oracle the checkpoint path is validated against. |

The Python `Tape.create` exposes all four:

```python
tape = nuka.Tape.create(
    world,
    checkpoint_interval=3,
    max_tape_entries=4096,
    max_checkpoints=512,
    recompute_on_backward=1,
)
```

## The recompute orchestrator

The `RecomputeOrchestrator` is the single forward function used in all three
roles, which is what guarantees record/replay bit-exactness:

1. **Record** — `StepOnce` advances the live state one contact-free PD step in
   place, using the action at a device pointer. This is the live forward.
2. **Replay** — to reconstruct a window's intermediates, the backward restores
   the nearest checkpoint `≤` the target step and re-runs `StepOnce` forward
   through the window. Because the kernels have no float atomics and a fixed
   reduction order, identical inputs produce identical bits — replay is byte-exact
   with the original record.
3. **FD oracle** — the same function backs the finite-difference validation, so
   the adjoint is paired with exactly the forward it was validated against.

The orchestrator also owns the uploaded drive descriptors and the
representation-consistent `dI/dmass` slope used to assemble the mass gradient.

## The backward pass

`tape.backward(seed)` runs the engine's deterministic reverse over the recorded
tape and returns `(grad_actions, grad_parameters)`:

- The **seed** is a host `float32[8·n]` adjoint vector (`n = link_count`) laid out
  as `[dL/dq' (n) | dL/dqd' (n) | dL/dv_root' (6n)]`. The frontends place the
  upstream cotangent into the third matching the chosen observation channel
  (`q` / `qdot` / `v_root`).
- **`grad_actions`** is `(K, n)`; the frontend takes the actuated slots `[1:1+ad]`
  (mirroring the root-slot skip used when applying actions).
- **`grad_parameters`** is `(n,)`; the frontend gathers the entries at the
  parameter link indices. This host→device copy is itself bit-deterministic, so
  the parameter gradient preserves D1.

The C contract is HOST pointers for the seed and the returned gradients (the
single-env tape lives on one device); the nanobind `Tape.backward` accepts a
host-contiguous seed and returns numpy arrays, which the frontends move to the
torch/jax device.

## Forward-compat for contacts (reserved)

Each `TapeEntry` already carries present-but-unused slots for the per-row contact
impulses (lambda) and contact event flags, plus a **λ warm-start** reservation.
The contact-free path never writes them. When the contact solve lands (a v0.7
item), replay-from-a-checkpoint is only bit-exact if the persistent warm-start λ
is restored, and the adjoint of the solve needs the converged λ — both ride here.
Reserving the layout now keeps the tape ABI stable across that landing.

## Memory and the IFT alternative

The tape's memory grows with the *checkpointed* step count, not the full step
count — that is the point of checkpointing. For the converged-contact gradient,
the IFT path (see [diff-sim](../concepts/diff-sim.md)) takes the gradient through
the implicit function theorem at the fixed point and uses **constant memory** (no
tape). The honest trade-off: measured IFT-vs-tape on the rigid + Featherstone
scope is **0.65–0.69×** — slower, launch-bound at small active-row counts — so the
IFT path is chosen for its constant memory, D1 determinism, and correctness at
the fixed point, not for speed.
