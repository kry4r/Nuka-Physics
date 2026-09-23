"""Measured first-surface range projected into a gravity-aligned local elevation grid."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch

import nuka
from nuka.kinematics import EncoderKinematics, quaternion_product, rotate_vector


@dataclass
class TerrainConfig:
    x_bounds: tuple[float, float] = (0.15, 1.75)
    y_bounds: tuple[float, float] = (-0.6, 0.6)
    cell_size: float = 0.1
    height_bounds: tuple[float, float] = (-1.8, -0.35)
    minimum_returns: int = 3

    def shape(self):
        values = (*self.x_bounds, *self.y_bounds, *self.height_bounds, self.cell_size)
        if not all(math.isfinite(value) for value in values) or self.cell_size <= 0:
            raise ValueError("Terrain bounds and cell size must be finite with positive spacing")
        if any(bounds[0] >= bounds[1] for bounds in (self.x_bounds, self.y_bounds, self.height_bounds)):
            raise ValueError("Terrain bounds must be increasing")
        shape = tuple(round((bounds[1] - bounds[0]) / self.cell_size)
                      for bounds in (self.x_bounds, self.y_bounds))
        if min(shape) < 2 or self.minimum_returns < 1:
            raise ValueError("Terrain grids need at least two cells per axis and one return per cell")
        if any(abs(count * self.cell_size - (bounds[1] - bounds[0])) > 1e-6
               for count, bounds in zip(shape, (self.x_bounds, self.y_bounds))):
            raise ValueError("Terrain bounds must contain a whole number of cells")
        return shape


class DepthTerrainProjector:
    """Project each sample with simultaneous encoder and IMU measurements, without world-state input."""

    channels = ("minimum_height", "maximum_height", "forward_difference", "lateral_difference", "valid")

    def __init__(self, controller, camera, optics, config):
        if controller.proprioception is None:
            raise ValueError("Terrain projection requires measured encoders and IMU gravity")
        if camera["mount"] not in (nuka.SensorMount.LINK.value, nuka.SensorMount.BASE.value):
            raise ValueError("Terrain cameras must be mounted on an encoder-observed articulation")
        self.config = config
        self.shape = config.shape()
        self.controller = controller
        self.tree = controller.world.kinematic_tree()
        self.kinematics = EncoderKinematics(self.tree, controller.contract.joint_names, device=controller.device)
        self.link = camera["mount_index"]
        if camera["mount"] == nuka.SensorMount.BASE.value:
            roots = [index for index, row in enumerate(self.tree) if row["joint_type"] == "root"]
            self.link = roots[self.link]
        if self.tree[self.link]["articulation_index"] != 0:
            raise ValueError("Terrain projection requires the same articulation as its IMU")
        self.offset = torch.tensor(camera["local_offset"], device=controller.device)
        half_height = math.tan(math.radians(optics.vfov) * 0.5)
        u = ((torch.arange(optics.width, device=controller.device) + 0.5) * 2 / optics.width - 1)
        v = 1 - (torch.arange(optics.height, device=controller.device) + 0.5) * 2 / optics.height
        dy, dx = torch.meshgrid(v * half_height, u * half_height * optics.width / optics.height, indexing="ij")
        rays = torch.stack((dx, dy, -torch.ones_like(dx)), dim=-1).reshape(-1, 3)
        self.rays = rays / rays.norm(dim=-1, keepdim=True)

    def camera_pose(self):
        measured = self.controller.proprioception.joint_state[:, :, 0]
        mount = self.kinematics.pose(measured, self.link)
        position = mount[:, :3] + rotate_vector(mount[:, 3:], self.offset[:3])
        rotation = quaternion_product(mount[:, 3:], self.offset[3:].expand_as(mount[:, 3:]))
        return torch.cat((position, rotation), dim=-1)

    @torch.no_grad()
    def project(self, depth, valid):
        pose = self.camera_pose()
        points = pose[:, None, :3] + rotate_vector(pose[:, None, 3:], self.rays) * depth.flatten(1)[..., None]
        up = -self.controller.proprioception.gravity
        forward = up.new_tensor([1, 0, 0]) - up[:, :1] * up
        forward = forward / forward.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        left = torch.cross(up, forward, dim=-1)
        basis = torch.stack((forward, left, up), dim=-1)
        local = torch.bmm(points, basis)
        grid = self.rasterize(local, valid.flatten(1))
        return grid, pose.clone(), up.clone()

    def rasterize(self, points, valid):
        cfg = self.config
        nx, ny = self.shape
        x, y, z = points.unbind(-1)
        accepted = (valid & torch.isfinite(points).all(-1)
            & (x >= cfg.x_bounds[0]) & (x < cfg.x_bounds[1])
            & (y >= cfg.y_bounds[0]) & (y < cfg.y_bounds[1])
            & (z >= cfg.height_bounds[0]) & (z <= cfg.height_bounds[1]))
        ix = torch.nan_to_num((x - cfg.x_bounds[0]) / cfg.cell_size).floor().long().clamp(0, nx - 1)
        iy = torch.nan_to_num((y - cfg.y_bounds[0]) / cfg.cell_size).floor().long().clamp(0, ny - 1)
        index = ix * ny + iy
        low = points.new_full((len(points), nx * ny), float("inf"))
        high = points.new_full((len(points), nx * ny), -float("inf"))
        count = torch.zeros_like(low, dtype=torch.int32)
        low.scatter_reduce_(1, index, torch.where(accepted, z, float("inf")), reduce="amin")
        high.scatter_reduce_(1, index, torch.where(accepted, z, -float("inf")), reduce="amax")
        count.scatter_add_(1, index, accepted.int())
        known = (count >= cfg.minimum_returns).reshape(-1, nx, ny)
        low = torch.where(known, low.reshape(-1, nx, ny), 0)
        high = torch.where(known, high.reshape(-1, nx, ny), 0)
        middle = 0.5 * (low + high)
        forward = torch.zeros_like(middle)
        lateral = torch.zeros_like(middle)
        forward[:, :-1] = torch.where(known[:, :-1] & known[:, 1:], middle[:, 1:] - middle[:, :-1], 0)
        lateral[:, :, :-1] = torch.where(known[:, :, :-1] & known[:, :, 1:], middle[:, :, 1:] - middle[:, :, :-1], 0)
        return torch.stack((low, high, forward, lateral, known.float()), dim=1)

    def metadata(self):
        return {**asdict(self.config), "shape": self.shape, "channels": self.channels,
            "source": "measured metric ray range, encoder kinematics and IMU gravity at acquisition",
            "coordinates": "sample-time base origin; x horizontal body-forward, y left, z against estimated gravity",
            "unknown": "zero values with validity zero; no terrain-truth fill or extrapolation",
            "surface_contract": "visible first surface, including liquid returns; not a guarantee of solid support",
            "temporal_contract": "sample-frame grids retain explicit age; old depth is never reprojected with current poses"}
