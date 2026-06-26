"""The CONTROL layer for the authoring facade.

Per-step choreography that drives a built ``World`` through the PUBLIC field I/O
(``upload_field`` / ``download_field`` / ``stand_targets``). Holding a robot in a
crouch and parking/releasing a cloth are control INPUTS, not solver forks, and
they live HERE -- never baked onto a morph or entity. The morphs stay geometry.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as _np

import nuka as _nuka


class StandHold:
    """Hold an articulated robot in a fixed crouch by NAME.

    Seeds the joint positions + PD targets from ``stand_targets(**targets)``
    (matched against the cooked DOF names, correct under cook reordering), pins
    the floating base at ``base_z``, and each ``hold()`` re-stamps the PD target,
    the base pose, and zeros the root spatial velocity.
    """

    def __init__(self, world, targets: Mapping[str, float], base_z: float,
                 stiffness: float = 60.0) -> None:
        self.world = world
        field = _nuka.Field
        blc = int(world.base_link_count)
        act = _np.asarray(world.stand_targets(**dict(targets)), dtype=_np.float32)
        # The per-link buffers are blc wide (slot 0 = inert root, slots [1:] the
        # actuated joints); the stand_targets vector is action_dim == blc-1.
        self.q = _np.concatenate([_np.zeros(1, _np.float32), act]).astype(_np.float32)
        stiff = _np.zeros(blc, _np.float32)
        damp = _np.zeros(blc, _np.float32)
        stiff[1:] = float(stiffness)
        damp[1:] = 2.0 * math.sqrt(float(stiffness))
        world.upload_field(field.JOINT_POSITION, self.q)
        world.upload_field(field.DRIVE_STIFFNESS, stiff)
        world.upload_field(field.DRIVE_DAMPING, damp)
        self.base = _np.asarray(
            world.download_field(field.BASE_POSE, 0, 7), dtype=_np.float32).copy()
        self.base[2] = float(base_z)
        self._root_zero = _np.zeros(6, _np.float32)

    def hold(self) -> None:
        """Re-stamp the PD target + the base pose + zero the root velocity (one
        per-step control input). Call before each ``world.step()``."""
        field = _nuka.Field
        self.world.upload_field(field.DRIVE_TARGET, self.q)
        self.world.upload_field(field.BASE_POSE, self.base)
        self.world.upload_field(field.LINK_VELOCITY, self._root_zero, 0)


class ClothDrape:
    """Park a free cloth sheet flat above the scene, then release it to drape.

    ``park()`` pins the sheet ``park_lift`` m above its rest lay (so it clears the
    robot during settle); ``release()`` drops it to the rest lay with a slight
    tilt + a faint ripple (seeds the aero flutter); ``damp()`` applies a uniform
    per-step fabric drag. All over the public PARTICLE_POSITION/VELOCITY fields.
    """

    def __init__(self, world, grid, park_lift: float, damp: float = 0.0,
                 tilt_deg: float = 4.0, ripple: float = 0.018) -> None:
        self.world = world
        self.damp_rate = float(damp)
        particles = int(world.particle_count)
        rest = _np.ascontiguousarray(grid.rest_positions(), dtype=_np.float32)
        if rest.shape[0] != particles:
            raise ValueError(
                f"ClothDrape: grid has {rest.shape[0]} verts but the world has "
                f"{particles} particles")
        self._zero_vel = _np.zeros((particles, 3), _np.float32)
        self._park = rest.copy()
        self._park[:, 2] += float(park_lift)
        rel = rest.copy()
        tilt = math.tan(math.radians(float(tilt_deg)))
        cx, cy = float(grid.origin[0]), float(grid.origin[1])
        dx = rel[:, 0] - cx
        dy = rel[:, 1] - cy
        rel[:, 2] += (tilt * dx
                      + float(ripple) * _np.sin(3.0 * math.pi * dx)
                      * _np.cos(2.0 * math.pi * dy))
        self._release = rel

    def park(self) -> None:
        """Hold the sheet parked high (a kinematic re-seat each step)."""
        field = _nuka.Field
        self.world.upload_field(field.PARTICLE_POSITION, self._park.reshape(-1))
        self.world.upload_field(field.PARTICLE_VELOCITY, self._zero_vel.reshape(-1))

    def release(self) -> None:
        """Drop the parked sheet to the rest lay (tilt + ripple, zero velocity)."""
        field = _nuka.Field
        self.world.upload_field(field.PARTICLE_POSITION, self._release.reshape(-1))
        self.world.upload_field(field.PARTICLE_VELOCITY, self._zero_vel.reshape(-1))

    def damp(self) -> None:
        """Scale the particle velocity by ``1 - damp`` (uniform fabric drag)."""
        if self.damp_rate <= 0.0:
            return
        field = _nuka.Field
        v = _np.asarray(
            self.world.download_field(field.PARTICLE_VELOCITY), dtype=_np.float32)
        v *= (1.0 - self.damp_rate)
        self.world.upload_field(field.PARTICLE_VELOCITY, v)


class SoftDrop:
    """Drop a particle soft body onto the scene from its authored rest.

    Captures the cooked particle rest (the lattice the morph placed), ``hold()``
    pins the body kinematically there with zero velocity (a clear airborne beat),
    then ``release()`` stops pinning so gravity drops it -- the drop is a control
    INPUT, never baked onto the morph. ``centroid()`` reads the live centre for a
    tracking hero camera. All over the public PARTICLE_POSITION/VELOCITY fields.
    """

    def __init__(self, world) -> None:
        self.world = world
        particles = int(world.particle_count)
        self._rest = _np.ascontiguousarray(
            _np.asarray(world.download_field(_nuka.Field.PARTICLE_POSITION),
                        dtype=_np.float32).reshape(-1, 3))
        if self._rest.shape[0] != particles:
            raise ValueError(
                f"SoftDrop: world has {particles} particles but downloaded "
                f"{self._rest.shape[0]} positions")
        self._zero_vel = _np.zeros((particles, 3), _np.float32)

    def hold(self) -> None:
        """Pin the body at its cooked rest with zero velocity (a kinematic re-seat
        each step). Call before each ``world.step()`` during the airborne hold."""
        field = _nuka.Field
        self.world.upload_field(field.PARTICLE_POSITION, self._rest.reshape(-1))
        self.world.upload_field(field.PARTICLE_VELOCITY, self._zero_vel.reshape(-1))

    def release(self) -> None:
        """Zero the velocity once and stop holding so gravity drops the body."""
        self.world.upload_field(_nuka.Field.PARTICLE_VELOCITY,
                                self._zero_vel.reshape(-1))

    def centroid(self) -> "tuple":
        """The live particle centroid (x, y, z) for a tracking hero camera."""
        p = _np.asarray(self.world.download_field(_nuka.Field.PARTICLE_POSITION),
                        dtype=_np.float32).reshape(-1, 3)
        c = p.mean(axis=0)
        return (float(c[0]), float(c[1]), float(c[2]))

    def damp(self, rate: float) -> None:
        """Scale the particle velocity by ``1 - rate`` (uniform settle drag)."""
        if rate <= 0.0:
            return
        field = _nuka.Field
        v = _np.asarray(self.world.download_field(field.PARTICLE_VELOCITY),
                        dtype=_np.float32)
        v *= (1.0 - rate)
        self.world.upload_field(field.PARTICLE_VELOCITY, v)


class PinBody:
    """Stamp a static rigid collidable's world transform each step (a control pin).

    A static body cooked from the built-scene path carries NO seeded device pose
    (the cook populates body_init only for a MOVABLE FREE body), so the broadphase
    cannot place it until its transform is stamped. ``pin()`` re-seats the chosen
    body's RIGID_BODY_TRANSFORM (position + quaternion) so a particle medium rests
    on it -- a control INPUT, not a solver fork. Call before each ``world.step()``.
    """

    def __init__(self, world, pos, quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                 body_index: int = 0) -> None:
        self.world = world
        t = _np.asarray(world.download_field(_nuka.Field.RIGID_BODY_TRANSFORM),
                        dtype=_np.float32)
        if t.size % 7 != 0 or t.size == 0:
            raise ValueError(
                f"PinBody: RIGID_BODY_TRANSFORM size {t.size} is not a body*7 array")
        self._t = _np.ascontiguousarray(t.reshape(-1, 7))
        if body_index >= self._t.shape[0]:
            raise ValueError(
                f"PinBody: body_index {body_index} >= rigid body count "
                f"{self._t.shape[0]}")
        self._t[body_index] = [pos[0], pos[1], pos[2],
                               quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3]]

    def pin(self) -> None:
        """Re-stamp the pinned body transform (one per-step control input)."""
        self.world.upload_field(_nuka.Field.RIGID_BODY_TRANSFORM,
                                self._t.reshape(-1))


__all__ = ["StandHold", "ClothDrape", "SoftDrop", "PinBody"]
