"""Physical observation errors and per-episode domain randomization.

Configure a field once, then call ``world.sample_observation`` or
``world.apply_sensor_noise`` for each acquisition. Read ``get_observation_view``
or ``download_observation``; physics remains available through ``buffer_view``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import ClassVar, Tuple

NOISE_NONE = 0
NOISE_GAUSSIAN = 1
NOISE_POISSON = 2

__all__ = ["NOISE_NONE", "NOISE_GAUSSIAN", "NOISE_POISSON", "GaussianNoise",
           "PoissonNoise", "MeasurementError", "DomainRandomization"]


@dataclass
class GaussianNoise:
    """Per-acquisition additive Gaussian error in measurement units."""

    mean: float = 0.0
    stddev: float = 0.01
    seed: int = 0
    kind: ClassVar[int] = NOISE_GAUSSIAN

    def configure(self, world, field) -> None:
        """Configure observations and reset their history, leaving physics intact."""
        world.set_sensor_noise(field, self.kind, float(self.mean), float(self.stddev), int(self.seed))

    def apply_to(self, world, field) -> None:
        """Configure, then acquire once; further apply_sensor_noise calls advance the sequence."""
        self.configure(world, field)
        world.apply_sensor_noise(field)


@dataclass
class PoissonNoise:
    """Add an independent Poisson count; this is not a camera photon model."""

    lam: float = 1.0
    seed: int = 0
    kind: ClassVar[int] = NOISE_POISSON

    def configure(self, world, field) -> None:
        """Configure a nonnegative rate up to 1e8 counts per acquisition."""
        world.set_sensor_noise(field, self.kind, float(self.lam), 0.0, int(self.seed))

    def apply_to(self, world, field) -> None:
        """Configure, then acquire into the independent observation buffer."""
        self.configure(world, field)
        world.apply_sensor_noise(field)


@dataclass
class MeasurementError:
    """Calibration and stochastic error in the observed field's physical units.

    ``noise_density`` is the square root of two-sided PSD, in units * sqrt(s).
    ``bias_random_walk`` is in units / sqrt(s); correlation and response times
    are seconds. Temperatures are Celsius. Both limits are needed to saturate.
    """

    bias: float = 0.0
    scale_error: float = 0.0
    noise_density: float = 0.0
    initial_bias_stddev: float = 0.0
    bias_random_walk: float = 0.0
    correlated_bias_stddev: float = 0.0
    correlation_time: float = 0.0
    quantization: float = 0.0
    minimum: float | None = None
    maximum: float | None = None
    response_time: float = 0.0
    temperature_coefficient: float = 0.0
    reference_temperature: float = 25.0
    seed: int = 0

    def configure(self, world, field) -> None:
        """Replace the observation model and reset its per-environment history."""
        world.set_sensor_error(field, **self._parameters())

    def configure_sensor(self, world, sensor, component) -> None:
        """Configure a mounted sensor component and restart that sensor's acquisition history."""
        world.set_state_sensor_error(sensor, component, **self._parameters())

    def _parameters(self) -> dict:
        parameters = asdict(self)
        minimum = parameters.pop("minimum")
        maximum = parameters.pop("maximum")
        if (minimum is None) != (maximum is None):
            raise ValueError("minimum and maximum must both be supplied")
        if minimum is not None:
            parameters.update(minimum=minimum, maximum=maximum, saturation_enabled=True)
        return parameters


@dataclass
class DomainRandomization:
    """Per-episode domain randomization.

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
