"""``SimulationContext`` -- a thin wrapper over ``nuka.World``.

Mirrors the slice of Isaac Lab's ``SimulationContext`` an RL training loop
touches: holding the device + world, stepping, resetting, and exposing the
underlying ``nuka.World`` for the managers to read zero-copy buffers from. NO
rendering / UI / physx-fabric anything.
"""

from __future__ import annotations

import nuka


class SimulationContext:
    """Owns the ``nuka.Device`` + ``nuka.World`` for a manager-based env.

    Parameters
    ----------
    scene_path : str
        USDA scene path.
    num_envs : int
        Parallel env count.
    dt : float
        Physics timestep.
    device : nuka.Device | None
        Reuse an open device, or create+own one (ordinal 0).
    """

    def __init__(self, scene_path, num_envs, dt, device=None):
        self._owns_device = device is None
        self.device = device if device is not None else nuka.Device.create(0)
        self.world = nuka.World.create_from_scene(
            self.device, scene_path, int(num_envs), float(dt)
        )
        self.num_envs = int(num_envs)
        self.dt = float(dt)

    def step(self, n: int = 1) -> None:
        """Advance the physics ``n`` steps."""
        if n == 1:
            self.world.step()
        else:
            self.world.step_n(int(n))

    def reset(self) -> None:
        """Reset all envs to the creation pose."""
        self.world.reset()

    def reset_envs(self, env_ids) -> None:
        """Masked reset of the listed envs."""
        self.world.reset_envs(env_ids)

    def sync(self) -> None:
        nuka.sync()

    def close(self) -> None:
        if getattr(self, "world", None) is not None:
            self.world.destroy()
            self.world = None
        if self._owns_device and getattr(self, "device", None) is not None:
            self.device.close()
            self.device = None
