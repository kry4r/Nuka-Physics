"""Relative articulation poses from immutable calibration and measured joint coordinates."""

from __future__ import annotations

import torch


def quaternion_product(left, right):
    scalar = left[..., :1] * right[..., :1] - (left[..., 1:] * right[..., 1:]).sum(-1, keepdim=True)
    axis = (left[..., :1] * right[..., 1:] + right[..., :1] * left[..., 1:]
            + torch.cross(left[..., 1:], right[..., 1:], dim=-1))
    return torch.cat((scalar, axis), dim=-1)


def rotate_vector(quaternion, vector):
    axis = quaternion[..., 1:]
    axis, vector = torch.broadcast_tensors(axis, vector)
    cross = torch.cross(axis, vector, dim=-1)
    return vector + 2 * (quaternion[..., :1] * cross + torch.cross(axis, cross, dim=-1))


class EncoderKinematics:
    """Compute root-relative link transforms without reading simulated poses or joint state."""

    def __init__(self, tree, joint_names, *, device):
        self.tree = tree
        names = [name.rsplit("/", 1)[-1] for name in joint_names]
        if len(set(names)) != len(names):
            raise ValueError("Encoder names must be unambiguous")
        self.coordinates = {name: index for index, name in enumerate(names)}
        self.local = torch.tensor([row["local_pose"] for row in tree], device=device)
        axis = torch.tensor([row["axis"] for row in tree], device=device)
        self.axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        self.chains = {}
        for index in range(len(tree)):
            chain, visited, cursor = [], set(), index
            while tree[cursor]["joint_type"] != "root":
                if cursor in visited:
                    raise ValueError("A kinematic hierarchy cannot contain cycles")
                visited.add(cursor)
                chain.append(cursor)
                cursor = tree[cursor]["parent_index"]
                if not 0 <= cursor < len(tree):
                    raise ValueError("A kinematic parent must reference the same tree")
            self.chains[index] = tuple(reversed(chain))

    def pose(self, measured_position, link_index):
        position = measured_position.new_zeros((measured_position.shape[0], 3))
        rotation = measured_position.new_tensor([1, 0, 0, 0]).expand(measured_position.shape[0], 4)
        for index in self.chains[link_index]:
            row = self.tree[index]
            local_position = self.local[index, :3].expand_as(position)
            local_rotation = self.local[index, 3:].expand_as(rotation)
            kind = row["joint_type"]
            if kind in ("revolute", "prismatic"):
                coordinate = measured_position[:, self.coordinates[row["name"].rsplit("/", 1)[-1]], None]
                if kind == "revolute":
                    half = coordinate * 0.5
                    motion = torch.cat((half.cos(), half.sin() * self.axis[index]), dim=-1)
                    local_rotation = quaternion_product(local_rotation, motion)
                else:
                    local_position = local_position + rotate_vector(local_rotation, coordinate * self.axis[index])
            elif kind != "fixed":
                raise ValueError(f"Unsupported encoder joint: {kind}")
            position = position + rotate_vector(rotation, local_position)
            rotation = quaternion_product(rotation, local_rotation)
            rotation = rotation / rotation.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return torch.cat((position, rotation), dim=-1)
