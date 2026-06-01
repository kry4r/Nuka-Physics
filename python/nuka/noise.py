"""nuka.noise -- ergonomic sim-to-real noise + domain-randomization config.

v0.5 p04 Task 5.4.9 (the Python user-facing wrapper for the N1 sensor noise +
N2 per-episode domain randomization the engine + C-ABI already implement -- see
``src/include/nuka/nuka_noise.h``).

PURE PYTHON: this module imports NO torch / NO jax, so it is safe to import
eagerly from ``nuka/__init__.py`` (``import nuka`` must never hard-require a DL
framework). The dataclasses are thin config holders; the ``apply_to`` /
``configure`` helpers call straight through to the ``World`` nanobind methods
(``set_sensor_noise`` / ``apply_sensor_noise`` / ``set_domain_randomization`` /
``apply_domain_randomization``).

Noise kinds (mirror ``nuka_noise_kind_t``; also exposed as module ints on the
extension as ``NOISE_NONE`` / ``NOISE_GAUSSIAN`` / ``NOISE_POISSON``):
    0 = NONE      clears the field's noise (apply becomes a byte no-op)
    1 = GAUSSIAN  param1 = mean, param2 = stddev
    2 = POISSON   param1 = lambda

Typical use::

    import nuka

    with nuka.Device.create(0) as dev:
        world = nuka.World.create_from_scene(dev, scene, env_count=64)

        # Sensor noise on the observed joint velocities.
        noise = nuka.GaussianNoise(mean=0.0, stddev=0.02, seed=123)
        world.step()
        noise.apply_to(world, nuka.JOINT_VELOCITY)   # set + apply in one call
        nuka.sync()                                  # apply kernel is async

        # Per-episode domain randomization (call at episode reset).
        dr = nuka.DomainRandomization(seed=7)
        dr.configure(world)            # register the descriptor
        dr.apply(world)                # sample + apply for all envs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Tuple

# Noise-kind ints -- mirror nuka_noise_kind_t. Hardcoded (NOT imported from the
# extension) so this module stays import-light and dependency-free.
NOISE_NONE = 0
NOISE_GAUSSIAN = 1
NOISE_POISSON = 2

__all__ = [
    "NOISE_NONE",
    "NOISE_GAUSSIAN",
    "NOISE_POISSON",
    "GaussianNoise",
    "PoissonNoise",
    "DomainRandomization",
]


@dataclass
class GaussianNoise:
    """Additive zero-mean (configurable) Gaussian sensor noise.

    Maps to ``nuka_sensor_noise_desc_t{NUKA_NOISE_GAUSSIAN, mean, stddev, seed}``.
    Counter-based (Philox) -> D1 two-run bit-exact and replay-stable.
    """

    mean: float = 0.0
    stddev: float = 0.01
    seed: int = 0

    #: The ``nuka_noise_kind_t`` int this dataclass registers. ClassVar so it is
    #: NOT an __init__ parameter / repr field (the spec constructor is
    #: GaussianNoise(mean, stddev, seed)).
    kind: ClassVar[int] = NOISE_GAUSSIAN

    def configure(self, world, field) -> None:
        """Register this noise on ``field`` (does NOT apply yet).

        ``world`` is a ``nuka.World``; ``field`` is a ``nuka.Field`` enum value.
        Recording resets the field's per-field sequence counter to 0.
        """
        world.set_sensor_noise(field, NOISE_GAUSSIAN, float(self.mean),
                               float(self.stddev), int(self.seed))

    def apply_to(self, world, field) -> None:
        """Register + apply the noise to ``field``'s live device buffer once.

        The apply kernel is async -- call ``nuka.sync()`` before reading the
        buffer back. Applying advances the field's sequence counter, so a
        subsequent apply produces independent noise.
        """
        self.configure(world, field)
        world.apply_sensor_noise(field)


@dataclass
class PoissonNoise:
    """Poisson-distributed sensor noise with rate ``lam``.

    Maps to ``nuka_sensor_noise_desc_t{NUKA_NOISE_POISSON, lam, 0, seed}``
    (param2 is unused for Poisson). Counter-based -> D1 two-run bit-exact.
    """

    lam: float = 1.0
    seed: int = 0

    #: The ``nuka_noise_kind_t`` int this dataclass registers. ClassVar so it is
    #: NOT an __init__ parameter / repr field (the spec constructor is
    #: PoissonNoise(lam, seed)).
    kind: ClassVar[int] = NOISE_POISSON

    def configure(self, world, field) -> None:
        """Register this noise on ``field`` (does NOT apply yet)."""
        world.set_sensor_noise(field, NOISE_POISSON, float(self.lam), 0.0,
                               int(self.seed))

    def apply_to(self, world, field) -> None:
        """Register + apply the noise to ``field``'s live device buffer once.

        The apply kernel is async -- call ``nuka.sync()`` before reading back.
        """
        self.configure(world, field)
        world.apply_sensor_noise(field)


@dataclass
class DomainRandomization:
    """Per-episode domain randomization (Task 5.4.7).

    Maps to ``nuka_domain_randomization_desc_t``. ``mass_range`` / ``friction_range``
    are MULTIPLIER ranges ``[lo, hi]`` applied as ``nominal * mult``;
    ``restitution_range`` / ``armature_range`` / ``gravity_range`` are OFFSET ranges
    applied as ``nominal + offset``. Each is sampled ONCE per env per episode-reset
    as a pure function of ``(seed, env_idx, param)`` via counter-based Philox, so
    the diff-sim backward stays D1 two-run bit-exact.

    ``enabled=False`` (or never configuring) makes ``apply`` a byte no-op (V1
    oracle scenes stay byte-identical unless DR is explicitly enabled).

    NOTE: ``apply`` must be called BEFORE ``nuka.Tape.create`` -- the tape
    captures gravity at create time and mass must be in place before the first
    ``step_with_tape``.
    """

    mass_range: Tuple[float, float] = (0.8, 1.2)
    friction_range: Tuple[float, float] = (0.5, 1.5)
    restitution_range: Tuple[float, float] = (-0.1, 0.1)
    armature_range: Tuple[float, float] = (0.0, 0.05)
    gravity_range: Tuple[float, float] = (-0.5, 0.5)
    seed: int = 0
    enabled: bool = True

    def configure(self, world) -> None:
        """Record this DR descriptor on ``world`` (does NOT sample/apply yet)."""
        world.set_domain_randomization(
            float(self.mass_range[0]), float(self.mass_range[1]),
            float(self.friction_range[0]), float(self.friction_range[1]),
            float(self.restitution_range[0]), float(self.restitution_range[1]),
            float(self.armature_range[0]), float(self.armature_range[1]),
            float(self.gravity_range[0]), float(self.gravity_range[1]),
            int(self.seed), int(1 if self.enabled else 0),
        )

    def apply(self, world) -> None:
        """Record + sample + apply the randomization for ALL envs.

        Call at episode reset, BEFORE creating a ``nuka.Tape``. ``enabled=False``
        -> a byte no-op. The first enabled apply snapshots a nominal baseline so
        repeated applies re-randomize AROUND nominal (idempotent), not compound.
        """
        self.configure(world)
        world.apply_domain_randomization()
