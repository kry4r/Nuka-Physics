"""Named joint motion for independent kinematic rigid boundaries."""

from __future__ import annotations

import numpy as np


def _rotate(q, v):
    t = 2 * np.cross(q[..., 1:], v)
    return v + q[..., :1] * t + np.cross(q[..., 1:], t)


def _multiply(a, b):
    return np.concatenate((a[..., :1] * b[..., :1] - np.sum(a[..., 1:] * b[..., 1:], axis=-1, keepdims=True),
                           a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + np.cross(a[..., 1:], b[..., 1:])), -1)


def _compose(a, b):
    return np.concatenate((a[..., :3] + _rotate(a[..., 3:], b[..., :3]), _multiply(a[..., 3:], b[..., 3:])), -1)


def _pose(value):
    out = np.asarray(value, dtype=np.float64).copy()
    if out.shape[-1:] != (7,) or not np.isfinite(out).all():
        raise ValueError("A pose must contain finite xyz and wxyz")
    norm = np.linalg.norm(out[..., 3:], axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("A pose quaternion must be nonzero")
    out[..., 3:] /= norm
    return out


class JointMotion:
    """FK and world COM twists for named scalar joints, using xyz/wxyz frames.
    Body pose = parent pose * parent_frame * motion * inverse(child_frame)."""

    def __init__(self, joints, body_poses, body_com=None):
        self.body_poses = _pose(body_poses)
        if self.body_poses.ndim != 2:
            raise ValueError("body_poses must be a body by seven array")
        count = len(self.body_poses)
        self.body_com = np.zeros((count, 3)) if body_com is None else np.asarray(body_com, np.float64).copy()
        if self.body_com.shape != (count, 3) or not np.isfinite(self.body_com).all():
            raise ValueError("body_com must be a finite body by three array")
        self.joints, children, names = [], set(), set()
        for source in joints:
            row = dict(source)
            parent, child = int(row["parent"]), int(row["child"])
            if not -1 <= parent < count or not 0 <= child < count or parent == child or child in children:
                raise ValueError("Joint body indices must define a forest with one parent per child")
            kind, name = row["type"], row["name"]
            if kind not in ("fixed", "revolute", "prismatic") or name in names:
                raise ValueError("Joint types must be supported and names unique")
            pre, post = _pose(row["parent_frame"]), _pose(row["child_frame"])
            if pre.shape != (7,) or post.shape != (7,):
                raise ValueError("Joint frames must each contain seven scalars")
            inverse = post.copy()
            inverse[4:] *= -1
            inverse[:3] = _rotate(inverse[3:], -post[:3])
            axis = np.asarray(row.get("axis", (0, 0, 1)), np.float64)
            if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-12:
                raise ValueError("Joint axes must be finite and nonzero")
            self.joints.append((name, parent, child, kind, pre, inverse, axis / np.linalg.norm(axis)))
            children.add(child)
            names.add(name)
        pending, ordered = self.joints, []
        ready = set(range(count)) - children
        while pending:
            available = [row for row in pending if row[1] == -1 or row[1] in ready]
            if not available:
                raise ValueError("Joint hierarchy cannot contain cycles")
            ordered.extend(available)
            ready.update(row[2] for row in available)
            pending = [row for row in pending if row[2] not in ready]
        self.joints = tuple(ordered)
        self.coordinate_names = tuple(row[0] for row in ordered if row[3] != "fixed")

    def evaluate(self, position, velocity):
        """Return (body poses, COM linear velocities, angular velocities)."""
        expected = set(self.coordinate_names)
        if set(position) != expected or set(velocity) != expected:
            raise ValueError("Position and velocity must specify every coordinate name exactly once")
        values = [np.asarray(mapping[name], np.float64) for mapping in (position, velocity)
                  for name in self.coordinate_names]
        if any(not np.isfinite(value).all() for value in values):
            raise ValueError("Joint coordinates and velocities must be finite")
        shape = np.broadcast_shapes(*(value.shape for value in values)) if values else ()
        poses = np.broadcast_to(self.body_poses, shape + self.body_poses.shape).copy()
        origin_velocity = np.zeros(shape + (len(self.body_poses), 3))
        angular = np.zeros_like(origin_velocity)
        identity = np.array([0, 0, 0, 1, 0, 0, 0], np.float64)
        for name, parent, child, kind, pre, post, axis in self.joints:
            pose = poses[..., parent, :] if parent >= 0 else np.broadcast_to(identity, shape + (7,))
            spin = angular[..., parent, :] if parent >= 0 else np.zeros(3)
            speed = origin_velocity[..., parent, :] if parent >= 0 else np.zeros(3)
            joint = _compose(pose, pre)
            speed = speed + np.cross(spin, joint[..., :3] - pose[..., :3])
            world_axis = _rotate(joint[..., 3:], axis)
            if kind == "revolute":
                half = np.broadcast_to(np.asarray(position[name]) * 0.5, shape)[..., None]
                rotation = np.concatenate((np.cos(half), np.sin(half) * axis), axis=-1)
                joint[..., 3:] = _multiply(joint[..., 3:], rotation)
                spin = spin + world_axis * np.asarray(velocity[name])[..., None]
            elif kind == "prismatic":
                shift = world_axis * np.asarray(position[name])[..., None]
                joint[..., :3] += shift
                speed = speed + np.cross(spin, shift) + world_axis * np.asarray(velocity[name])[..., None]
            body = _compose(joint, post)
            poses[..., child, :] = body
            angular[..., child, :] = spin
            origin_velocity[..., child, :] = speed + np.cross(spin, body[..., :3] - joint[..., :3])
        linear = origin_velocity + np.cross(angular, _rotate(poses[..., 3:], self.body_com))
        return poses, linear, angular


class KinematicBoundary:
    """Apply joint motion to independent zero-mass rows through public field I/O.
    The external driver absorbs contact reactions; dynamic rows are rejected."""

    def __init__(self, world, motion, body_indices):
        import nuka
        self.world, self.motion = world, motion
        self.indices = np.asarray(body_indices, np.int64)
        if self.indices.shape != (len(motion.body_poses),) or len(set(self.indices.tolist())) != len(self.indices):
            raise ValueError("body_indices must map each motion body to a distinct rigid row")
        if np.any(self.indices < 0):
            raise ValueError("Body indices must be nonnegative")
        self.envs = int(world.env_count)
        self.fields = (nuka.Field.RIGID_BODY_TRANSFORM, nuka.Field.BODY_LINEAR_VELOCITY,
                       nuka.Field.BODY_ANGULAR_VELOCITY)
        self.widths = (7, 3, 3)
        for env in range(self.envs):
            for body, index in enumerate(self.indices):
                binding = world.body_binding(int(index), env)
                if (binding["inverse_mass"] != 0 or binding["link_index"] != 0xFFFFFFFF
                        or binding["owner_body_index"] != 0xFFFFFFFF):
                    raise ValueError("A kinematic boundary requires independent zero-mass rigid rows")
                if not np.allclose(binding["center_of_mass"], motion.body_com[body], rtol=1e-6, atol=1e-7):
                    raise ValueError("Motion COM must match the cooked body COM")

    def apply(self, position, velocity):
        """Upload the requested FK poses and analytic COM twists for all environments."""
        values = self.motion.evaluate(position, velocity)
        for field, width, value in zip(self.fields, self.widths, values):
            current = np.asarray(self.world.download_field(field), np.float32).reshape(self.envs, -1, width).copy()
            current[:, self.indices] = value
            self.world.upload_field(field, np.ascontiguousarray(current.ravel()))
        self.world.forward_kinematics()
