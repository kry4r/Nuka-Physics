# Sim-to-real noise (N1 + N2)

Nuka ships two sim-to-real noise mechanisms for v0.5:

- **N1 — per-field sensor noise** (Gaussian / Poisson) applied to a sensor
  field's live device buffer.
- **N2 — per-episode domain randomization** of physical parameters (mass,
  friction, restitution, armature, gravity).

Both are driven by a **counter-based Philox4x32-10** RNG, which is what makes them
deterministic, replayable, and oracle-safe.

C ABI: [`src/include/nuka/nuka_noise.h`](../../src/include/nuka/nuka_noise.h).
Python wrapper: [`python/nuka/noise.py`](../../python/nuka/noise.py).

## Why counter-based RNG

A Philox4x32-10 sample is a **pure function of `(seed, element_index, sequence)`** —
there is no mutable RNG state to advance or checkpoint. This gives three
properties that matter for a deterministic, differentiable engine:

1. **D1 two-run bit-exact.** The same `(seed, index, sequence)` always yields the
   same sample, so a noised run reproduces byte-for-byte across two runs.
2. **Replay-stable on the backward pass.** The diff-sim reverse / replay pass
   re-derives the identical noise from the same indices — there is no RNG state
   to record on the tape, and the backward stays byte-exact.
3. **Off-by-default is exactly identity.** A field with no registered noise (kind
   `NONE`), or DR with `enabled=false`, makes `apply` a **byte no-op**. No
   existing call site invokes `apply`, so V1 oracle scenes stay byte-for-byte
   identical unless noise is explicitly enabled.

## N1 — sensor noise

Noise kinds (mirroring `nuka_noise_kind_t`, also exposed as the module ints
`NOISE_NONE` / `NOISE_GAUSSIAN` / `NOISE_POISSON`):

| Kind | Int | Parameters |
|------|-----|------------|
| `NONE` | 0 | clears the field's noise (apply becomes a byte no-op) |
| `GAUSSIAN` | 1 | `param1 = mean`, `param2 = stddev` |
| `POISSON` | 2 | `param1 = lambda` |

Registering a descriptor on a field resets that field's per-field sequence
counter to 0; each `apply` advances the counter, so successive applies produce
independent noise across steps. Only float-stride fields are supported (the
primitive operates on float32 elements); a struct field (e.g. the 7-float pose or
6-float velocity fields) returns `NOT_SUPPORTED`.

```python
import nuka

with nuka.Device.create(0) as dev:
    world = nuka.World.create_from_scene(dev, scene, env_count=64)

    noise = nuka.GaussianNoise(mean=0.0, stddev=0.02, seed=123)
    world.step()
    noise.apply_to(world, nuka.JOINT_VELOCITY)   # register + apply in one call
    nuka.sync()                                  # the apply kernel is async
    # JOINT_VELOCITY now carries Gaussian sensor noise.
```

`PoissonNoise(lam=..., seed=...)` works the same way. `configure(world, field)`
registers without applying; `apply_to(world, field)` does both.

## N2 — domain randomization

`DomainRandomization` randomizes physical parameters once per env per
episode-reset. Each range is sampled to a per-env value that is a pure function of
`(seed, env_idx, param)` via the same counter-based Philox, so the diff-sim
backward stays D1 two-run bit-exact.

| Parameter | Range semantics | Engine mapping |
|-----------|-----------------|----------------|
| `mass_range` | **multiplier** (`nominal × mult`) | per-link spatial inertia (same rebuild path as `set_link_mass`). **Tape-visible** — affects the contact-free ABA forward and its gradient. |
| `gravity_range` | **offset** (`nominal + off`) | world `gravity.z` (read by `Tape.create` into the rollout params). **Tape-visible.** One world scalar → applies to all envs. |
| `friction_range` | multiplier | batched contact path's friction coefficient. Inert in the single-env contact-free tape (no contact solve). |
| `armature_range` | offset | per-DOF joint armature. Present on the articulation state but inert in the contact-free tape forward. |
| `restitution_range` | offset | no engine buffer in the contact-free path — sampled for RL completeness but inert there. |

```python
import nuka

with nuka.Device.create(0) as dev:
    world = nuka.World.create_from_scene(dev, scene, env_count=64)

    dr = nuka.DomainRandomization(seed=7)   # default mass mult [0.8, 1.2], etc.
    dr.configure(world)                     # register the descriptor
    dr.apply(world)                         # sample + apply for ALL envs
```

Important ordering for the diff-sim tape: **call `apply` before
`nuka.Tape.create`.** The tape captures gravity at create time and mass must be
in place before the first `step_with_tape`.

The first enabled `apply` snapshots a **nominal baseline** (per-link mass,
gravity, per-DOF armature). Repeated applies re-randomize *around* nominal
(idempotent across resets) rather than compounding a random walk. `enabled=false`
makes `apply` a byte no-op.

## Interaction with the diff-sim backward

Because both N1 and N2 are stateless pure functions of their indices, they are
fully compatible with the differentiable backward:

- The reverse/replay pass re-derives identical noise from the same `(seed, index,
  sequence)` — nothing about the RNG needs to be stored on the tape.
- The tape-visible N2 parameters (mass, gravity) participate correctly in the
  contact-free ABA forward and its gradient: a DR'd mass simply shifts the
  forward operating point that the analytical adjoint linearizes around.
- The overall backward remains **byte-exact** two-run, preserving the D1 contract
  end to end.

See [diff-sim](../concepts/diff-sim.md) and the
[diff-sim tape](diffsim-tape.md) for the backward model.
