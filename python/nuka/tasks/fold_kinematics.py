"""Kinematics, world frames and hand calibration for replaying the G1 towel-folding recordings."""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation

ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
WAIST = ("yaw", "roll", "pitch")
LEGS = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
HAND = ("thumb_1", "thumb_2", "index_1", "middle_1", "ring_1", "little_1")
SIDES = ("left", "right")
FPS = 30.0
GRIPPER_OPEN = 4.5
# Pad centre of the Dex1 gripper in its base link; the jaws approach along the base x axis and close along y.
DEX1_PINCH = np.array([0.1333, 0.0, 0.0])
# Sole sphere centres lie 0.03 below the ankle roll frame with radius 0.005.
SOLE_OFFSET = 0.035


def arm_joints(side):
    return [f"{side}_{name}_joint" for name in ARM]


def waist_joints():
    return [f"waist_{name}_joint" for name in WAIST]


def axis_angle(axis, angle):
    """Rotation matrix of ``angle`` about the unit ``axis``."""
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    k = 1.0 - c
    return np.array([[c + x * x * k, x * y * k - z * s, x * z * k + y * s],
                     [y * x * k + z * s, c + y * y * k, y * z * k - x * s],
                     [z * x * k - y * s, z * y * k + x * s, c + z * z * k]])


def _vector(element, key, default="0 0 0"):
    text = element.get(key) if element is not None else None
    return np.array((text or default).split(), dtype=np.float64)


def binary_stl(path):
    """Triangles (n, 3, 3) of a binary STL file."""
    data = Path(path).read_bytes()
    count = int(np.frombuffer(data, np.uint32, 1, 80)[0]) if len(data) >= 84 else -1
    if len(data) != 84 + 50 * count:
        raise ValueError(f"{path} is not a binary STL")
    record = np.dtype([("normal", "<f4", (3,)), ("corners", "<f4", (3, 3)), ("attribute", "<u2")])
    return np.frombuffer(data, record, count, 84)["corners"].astype(np.float64)


class Kinematics:
    """Tree kinematics of a URDF: fixed, revolute, continuous and prismatic joints, mimic joints, link inertials
    and mesh collisions."""

    def __init__(self, path):
        self.source = Path(path)
        root = ET.parse(self.source).getroot()
        self.joints, self.inertials, self.collisions = {}, {}, {}
        for joint in root.findall("joint"):
            kind = joint.get("type")
            if kind not in ("fixed", "revolute", "continuous", "prismatic"):
                raise ValueError(f"unsupported joint type {kind}")
            origin, limit, mimic = joint.find("origin"), joint.find("limit"), joint.find("mimic")
            axis = _vector(joint.find("axis"), "xyz", "1 0 0")
            bounded = kind != "continuous" and limit is not None
            self.joints[joint.find("child").get("link")] = {
                "name": joint.get("name"), "type": kind, "parent": joint.find("parent").get("link"),
                "xyz": _vector(origin, "xyz"),
                "rotation": Rotation.from_euler("xyz", _vector(origin, "rpy")).as_matrix(),
                "axis": axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 0 else axis,
                "lower": float(limit.get("lower", "-inf")) if bounded else -math.inf,
                "upper": float(limit.get("upper", "inf")) if bounded else math.inf,
                "effort": float(limit.get("effort", "inf")) if limit is not None else math.inf,
                "mimic": None if mimic is None else (mimic.get("joint"), float(mimic.get("multiplier", "1")),
                                                    float(mimic.get("offset", "0")))}
        for link in root.findall("link"):
            name, inertial = link.get("name"), link.find("inertial")
            if inertial is not None:
                self.inertials[name] = (float(inertial.find("mass").get("value")),
                                        _vector(inertial.find("origin"), "xyz"))
            for collision in link.findall("collision"):
                mesh = collision.find("geometry/mesh")
                if mesh is None:
                    continue
                if np.any(_vector(mesh, "scale", "1 1 1") != 1.0):
                    raise ValueError(f"{name}: scaled collision meshes are not supported")
                origin = collision.find("origin")
                self.collisions.setdefault(name, []).append(
                    (Rotation.from_euler("xyz", _vector(origin, "rpy")).as_matrix(), _vector(origin, "xyz"),
                     mesh.get("filename")))
        self.link_of = {joint["name"]: child for child, joint in self.joints.items()}
        self.children = {}
        for child, joint in self.joints.items():
            self.children.setdefault(joint["parent"], []).append(child)
        self._subtrees, self._meshes = {}, {}

    def joint(self, name):
        return self.joints[self.link_of[name]]

    def path(self, tip, base):
        """Links from just below ``base`` down to ``tip``."""
        links = []
        while tip != base:
            if tip not in self.joints:
                raise ValueError(f"{base} is not an ancestor of the requested link")
            links.append(tip)
            tip = self.joints[tip]["parent"]
        return links[::-1]

    def subtree(self, link):
        if link not in self._subtrees:
            self._subtrees[link] = [link] + [name for child in self.children.get(link, [])
                                             for name in self.subtree(child)]
        return self._subtrees[link]

    def limits(self, names):
        return (np.array([self.joint(name)["lower"] for name in names]),
                np.array([self.joint(name)["upper"] for name in names]))

    def value(self, name, q):
        """Coordinate of joint ``name`` from ``q``; a mimic joint follows its source unless it is given."""
        joint = self.joint(name)
        if name in q or joint["mimic"] is None:
            return q.get(name, 0.0)
        source, multiplier, offset = joint["mimic"]
        return multiplier * self.value(source, q) + offset

    def poses(self, tip, base, q, frames):
        """Batched pose (R (F, 3, 3), p (F, 3)) of ``tip`` in ``base``; ``q`` maps joint names to (F,) arrays."""
        R = np.broadcast_to(np.eye(3), (frames, 3, 3)).copy()
        p = np.zeros((frames, 3))
        for link in self.path(tip, base):
            joint = self.joints[link]
            p = p + R @ joint["xyz"]
            R = R @ joint["rotation"]
            if joint["type"] in ("revolute", "continuous"):
                angle = np.broadcast_to(np.asarray(self.value(joint["name"], q), np.float64), (frames,))
                R = R @ Rotation.from_rotvec(angle[:, None] * joint["axis"]).as_matrix()
            elif joint["type"] == "prismatic":
                shift = np.broadcast_to(np.asarray(self.value(joint["name"], q), np.float64), (frames,))
                p = p + R @ joint["axis"] * shift[:, None]
        return R, p

    def frames(self, base, q, pose=None):
        """Every link frame below ``base`` for one configuration, with ``base`` at ``pose`` (R, p)."""
        out = {base: pose if pose is not None else (np.eye(3), np.zeros(3))}
        pending = [base]
        while pending:
            parent = pending.pop()
            R_parent, p_parent = out[parent]
            for child in self.children.get(parent, []):
                joint = self.joints[child]
                p = p_parent + R_parent @ joint["xyz"]
                R = R_parent @ joint["rotation"]
                if joint["type"] in ("revolute", "continuous"):
                    R = R @ axis_angle(joint["axis"], self.value(joint["name"], q))
                elif joint["type"] == "prismatic":
                    p = p + R @ joint["axis"] * self.value(joint["name"], q)
                out[child] = (R, p)
                pending.append(child)
        return out

    def chain(self, tip, base, q, tool=None):
        """Pose of ``tip`` (times ``tool`` (R, p)) in ``base`` and the geometric Jacobian (rows: linear, angular)
        of the revolute joints on the path, with their names."""
        R, p = np.eye(3), np.zeros(3)
        axes, origins, names = [], [], []
        for link in self.path(tip, base):
            joint = self.joints[link]
            p = p + R @ joint["xyz"]
            R = R @ joint["rotation"]
            if joint["type"] in ("revolute", "continuous"):
                axes.append(R @ joint["axis"])
                origins.append(p)
                names.append(joint["name"])
                R = R @ axis_angle(joint["axis"], self.value(joint["name"], q))
            elif joint["type"] == "prismatic":
                raise ValueError("prismatic joints on an IK chain are not supported")
        if tool is not None:
            p = p + R @ tool[1]
            R = R @ tool[0]
        J = np.zeros((6, len(axes)))
        for k, (axis, origin) in enumerate(zip(axes, origins)):
            J[:3, k] = np.cross(axis, p - origin)
            J[3:, k] = axis
        return R, p, J, names

    def mesh(self, link):
        """Collision triangles of ``link`` in its own frame (empty when it has no mesh collision)."""
        if link not in self._meshes:
            blocks = []
            for rotation, offset, filename in self.collisions.get(link, []):
                path = self.source.parent / filename
                if not path.is_file():
                    path = self.source.parent.parent / filename
                blocks.append(binary_stl(path) @ rotation.T + offset)
            self._meshes[link] = np.concatenate(blocks) if blocks else np.zeros((0, 3, 3))
        return self._meshes[link]

    def triangles(self, frames, links):
        """Triangles of ``links`` placed by ``frames`` and the link owning each triangle."""
        placed = [(self.mesh(link) @ frames[link][0].T + frames[link][1], link) for link in links
                  if len(self.mesh(link))]
        return (np.concatenate([block for block, _ in placed]),
                np.repeat(np.array([link for _, link in placed]), [len(block) for block, _ in placed]))

    def holding_torques(self, frames, names, gravity):
        """Torques that hold joints ``names`` against ``gravity`` acting on the subtree below each; ``frames``
        are link frames in the gravity frame (zero velocity and acceleration, as a recursive Newton-Euler
        evaluation reduces to)."""
        torques = np.zeros(len(names))
        for k, name in enumerate(names):
            link = self.link_of[name]
            R, origin = frames[link]
            moment = np.zeros(3)
            for body in self.subtree(link):
                if body in self.inertials and body in frames:
                    mass, centre = self.inertials[body]
                    moment += np.cross(frames[body][0] @ centre + frames[body][1] - origin, mass * gravity)
            torques[k] = -(R @ self.joints[link]["axis"]) @ moment
        return torques


def solve_ik(kin, tip, base, names, target, q0, tool=None, rotation_weight=0.1, damping=1e-3, max_step=0.2,
             iterations=200, position_tolerance=1e-5, rotation_tolerance=1e-4, q_fixed=None):
    """Damped least squares on the pose of ``tip`` (times ``tool``) in ``base``. Rotation errors weigh
    ``rotation_weight`` metres per radian, steps are clamped to ``max_step`` and joints clipped to their limits.
    Returns the joint values and the final position and rotation errors."""
    lower, upper = kin.limits(names)
    q = np.clip(np.asarray(q0, np.float64), lower, upper)
    fixed = dict(q_fixed or {})
    target_R, target_p = target
    for iteration in range(1, iterations + 1):
        R, p, J, chain_names = kin.chain(tip, base, {**fixed, **dict(zip(names, q))}, tool)
        if chain_names != list(names):
            raise ValueError(f"the IK chain joints {chain_names} differ from {list(names)}")
        e_p = target_p - p
        e_r = Rotation.from_matrix(target_R @ R.T).as_rotvec()
        if np.linalg.norm(e_p) < position_tolerance and np.linalg.norm(e_r) < rotation_tolerance:
            break
        A = np.vstack([J[:3], rotation_weight * J[3:]])
        dq = np.linalg.solve(A.T @ A + damping * np.eye(len(q)), A.T @ np.concatenate([e_p, rotation_weight * e_r]))
        scale = np.max(np.abs(dq)) / max_step
        q = np.clip(q + (dq / scale if scale > 1.0 else dq), lower, upper)
    R, p, _, _ = kin.chain(tip, base, {**fixed, **dict(zip(names, q))}, tool)
    return q, {"position_error_m": float(np.linalg.norm(target_p - p)),
               "rotation_error_rad": float(np.linalg.norm(Rotation.from_matrix(target_R @ R.T).as_rotvec())),
               "iterations": iteration, "at_limit": int(np.sum((q <= lower + 1e-9) | (q >= upper - 1e-9)))}


def episode_columns(path):
    """Float arrays (frames, n) of every recorded observation and action column of one episode."""
    table = pq.read_table(path)
    return {name: np.stack([np.asarray(value, np.float64).reshape(-1) for value in table[name].to_pylist()])
            for name in table.column_names if name.startswith(("observation.", "action.")) and "images" not in name}


def farthest_points(points, count):
    """``count`` of ``points`` picked one by one farthest from those already picked, from the first point."""
    chosen = [0]
    distance = np.linalg.norm(points - points[0], axis=1)
    for _ in range(min(count, len(points)) - 1):
        chosen.append(int(distance.argmax()))
        distance = np.minimum(distance, np.linalg.norm(points - points[chosen[-1]], axis=1))
    return points[chosen]


class PinchRetarget:
    """Arm and hand coordinates that put one hand's thumb and index contact pad points on the two jaw pads of a
    parallel gripper pose. Damped Gauss-Newton on both pad positions, the pinch approach axis against the gripper's,
    the step from a reference, the depth of the hand's proximal hull vertices below a support plane, the depth of
    each distal thumb hull vertex inside the distal index hulls and back, so closing stops where the fingers meet,
    and each finger's lowest distal point held at its jaw pad's height (no deeper than the target's press below
    that finger's support), so both tips stay loaded on their supports while the thumb arcs closed; that term fades out as
    the jaw pad rises one contact band above the support.
    The middle, ring and little fingers start tucked at full curl, clear of the support and of the closing thumb. ``contact`` "pad"
    takes the closed pinch's closest pad points, "tip" each distal link's hull vertex farthest from its finger root
    (a tip pinch, whose contact points are the fingers' lowest points on a flat support)."""

    def __init__(self, kin, side, hand, base="torso_link", contact="pad", approach_scale=0.05, reference_scale=0.02,
                 support_scale=10.0, clearance_scale=30.0, contact_scale=10.0, clearance_samples=48,
                 contact_band=0.01):
        self.kin, self.base, self.wrist = kin, base, f"{side}_wrist_yaw_link"
        self.fingers = [f"{side}_{name}_joint" for name in HAND]
        self.names = arm_joints(side) + self.fingers
        self.lower, self.upper = kin.limits(self.names)
        self.hand_names = list(hand["targets_open_closed"])
        opened = {name: values[0] for name, values in hand["targets_open_closed"].items()}
        self.squeeze = hand["targets_open_closed"][self.fingers[1]][1] - hand["thumb_bend"]
        self.open_fingers = np.array([opened[name] for name in self.fingers[:3]] + list(self.upper[-3:]))
        closed = {**opened, **dict(zip(self.fingers, (hand["thumb_rotation"], hand["thumb_bend"],
                                                      hand["index_bend"])))}
        frames = kin.frames(self.wrist, closed)
        side_links = (kin.subtree(f"{side}_thumb_3"), kin.subtree(f"{side}_index_2"))
        (first, owner), (second, other) = (kin.triangles(frames, links) for links in side_links)
        p, i, q, j = closest_pair(first, second)
        self.pads = [(str(link), frames[str(link)][0].T @ (point - frames[str(link)][1]))
                     for link, point in ((owner[i], p), (other[j], q))]
        self.gap = float(np.linalg.norm(q - p))
        if contact == "tip":
            roots = (frames[f"{side}_thumb_1"][1], frames[f"{side}_index_1"][1])
            self.pads = []
            for link, root in zip((str(owner[i]), str(other[j])), roots):
                points = np.unique(kin.mesh(link).reshape(-1, 3), axis=0)
                placed = points @ frames[link][0].T + frames[link][1]
                self.pads.append((link, points[np.argmax(np.linalg.norm(placed - root, axis=1))]))
        elif contact != "pad":
            raise ValueError(f"unknown pinch contact {contact}")
        rotation = np.array(hand["pinch_rotation"])
        self.sign = float(np.sign(rotation[:, 1] @ (q - p)))
        self.approach = rotation[:, 0]
        self.hull, self.coarse, self.planes, self.sphere = {}, {}, {}, {}
        for link in kin.subtree(self.wrist):
            if len(kin.mesh(link)):
                points = np.unique(kin.mesh(link).reshape(-1, 3), axis=0)
                hull = ConvexHull(points)
                self.hull[link] = points[hull.vertices]
        self.facing = [[link for link in kin.subtree(name) if link in self.hull]
                       for name in (f"{side}_thumb_3", f"{side}_index_2")]
        self.supported = [link for link in self.hull if link not in self.facing[0] + self.facing[1]]
        for link in self.facing[0] + self.facing[1]:
            coarse = farthest_points(self.hull[link], clearance_samples)
            self.coarse[link], self.planes[link] = coarse, ConvexHull(coarse).equations
            centre = coarse.mean(axis=0)
            self.sphere[link] = (centre, float(np.linalg.norm(coarse - centre, axis=1).max()))
        self.scales = (approach_scale, reference_scale, support_scale, clearance_scale, contact_scale)
        self.band = contact_band

    def target(self, R, p, width, support, reference, press=0.0, floors=None):
        """Gripper pinch frame (R, p) in ``base``, jaw pad distance, support plane (normal, offset) with
        normal . x >= offset, the reference coordinates, the thumb and index support heights (offsets along the
        normal, the plane's by default) and how far the fingertips may be driven below them."""
        floors = (support[1], support[1]) if floors is None else tuple(floors)
        return R, p, max(width, self.gap), support, np.asarray(reference, np.float64), press, floors

    def points(self, x):
        q = dict(zip(self.names, x))
        R, p = self.kin.poses(self.wrist, self.base, q, 1)
        frames = self.kin.frames(self.wrist, q, (R[0], p[0]))
        pads = [frames[link][0] @ local + frames[link][1] for link, local in self.pads]
        hull = np.concatenate([self.hull[link] @ frames[link][0].T + frames[link][1] for link in self.supported])
        distal = [np.concatenate([self.coarse[link] @ frames[link][0].T + frames[link][1] for link in own])
                  for own in self.facing]
        return pads, frames[self.wrist][0] @ self.approach, hull, distal, self.overlap(frames, distal)

    def overlap(self, frames, distal):
        """Depth of every coarse distal thumb hull vertex inside each coarse distal index hull and back, by the
        hull's least violated face plane (zero outside)."""
        depths = []
        for points, other in ((distal[0], self.facing[1]), (distal[1], self.facing[0])):
            for link in other:
                R, p = frames[link]
                local = (points - p) @ R
                centre, radius = self.sphere[link]
                near = np.flatnonzero(np.einsum("ij,ij->i", local - centre, local - centre) < radius * radius)
                depth = np.zeros(len(points))
                if len(near):
                    planes = self.planes[link]
                    depth[near] = np.maximum(-(local[near] @ planes[:, :3].T + planes[:, 3]).max(axis=1), 0.0)
                depths.append(depth)
        return np.concatenate(depths)

    def contact(self, distal, pads, normal, floors, press):
        """Height of each finger's lowest distal point over its jaw pad's height, held no deeper than ``press``
        below that finger's support, so both tips keep loading their supports while the closing thumb arcs.
        Each term fades out as its jaw pad rises one contact band above that support."""
        return np.array([np.clip(1.0 - (pad @ normal - floor) / self.band, 0.0, 1.0) *
                         ((points @ normal).min() - max(pad @ normal, floor - press))
                         for points, pad, floor in zip(distal, pads, floors)])

    def residual(self, x, target):
        R, p, width, (normal, offset), reference, press, floors = target
        (thumb, index), approach, hull, distal, overlap = self.points(x)
        half = 0.5 * width * self.sign * R[:, 1]
        return np.concatenate([thumb - (p - half), index - (p + half), self.scales[0] * (approach - R[:, 0]),
                               self.scales[1] * (x - reference),
                               self.scales[2] * np.maximum(offset - hull @ normal, 0.0), self.scales[3] * overlap,
                               self.scales[4] * self.contact(distal, (p - half, p + half), normal, floors, press)])

    def solve(self, x0, target, iterations=40, damping=1e-4, max_step=0.2):
        """Coordinates and their pad errors, approach angle and deepest hull vertex below the support plane."""
        x = np.clip(np.asarray(x0, np.float64), self.lower, self.upper)
        r = self.residual(x, target)
        for _ in range(iterations):
            J = np.column_stack([(self.residual(x + e, target) - r) / 1e-7 for e in np.eye(len(x)) * 1e-7])
            step = np.linalg.solve(J.T @ J + damping * np.eye(len(x)), -J.T @ r)
            step *= min(1.0, max_step / max(np.abs(step).max(), 1e-12))
            trial = np.clip(x + step, self.lower, self.upper)
            r_trial = self.residual(trial, target)
            if r_trial @ r_trial < r @ r:
                converged = np.abs(trial - x).max() < 1e-7
                x, r, damping = trial, r_trial, max(0.3 * damping, 1e-9)
                if converged:
                    break
            else:
                damping *= 10.0
                if damping > 1e4:
                    break
        R, p, width, (normal, offset), _, press, floors = target
        (thumb, index), approach, hull, distal, overlap = self.points(x)
        half = 0.5 * width * self.sign * R[:, 1]
        return x, {"thumb_pad_error_m": float(np.linalg.norm(thumb - (p - half))),
                   "finger_overlap_m": float(overlap.max()),
                   "contact_height_error_m": float(np.abs(self.contact(distal, (p - half, p + half), normal,
                                                                       floors, press)).max()),
                   "index_pad_error_m": float(np.linalg.norm(index - (p + half))),
                   "approach_error_rad": float(np.arccos(np.clip(approach @ R[:, 0], -1.0, 1.0))),
                   "support_depth_m": float(max(0.0, (offset - hull @ normal).max())),
                   "at_limit": int(np.sum((x <= self.lower + 1e-9) | (x >= self.upper - 1e-9)))}

    def hand(self, x, width):
        """Six hand drive targets; the thumb bend passes contact by the pinch squeeze as the jaw pad distance
        ``width`` closes below the fingers' contact gap, so the closing thumb follows its retargeted arc."""
        values = dict(zip(self.names, x))
        thumb = self.names.index(self.fingers[1])
        engaged = float(np.clip(1.0 - width / self.gap, 0.0, 1.0))
        values[self.fingers[1]] = min(values[self.fingers[1]] + engaged * self.squeeze, self.upper[thumb])
        return np.array([values[name] for name in self.hand_names])


def closure(columns, side, source="action"):
    """Normalized gripper closure in [0, 1] from the recorded Dex1 motor command (open at GRIPPER_OPEN)."""
    return np.clip(1.0 - columns[f"{source}.{side}_gripper"][:, 0] / GRIPPER_OPEN, 0.0, 1.0)


def grasp_events(values, threshold=0.5):
    """(grasp, release) frame pairs where ``values`` rises through ``threshold`` and next falls back through it;
    a grasp still held at the end releases at the last frame."""
    above = values >= threshold
    rises = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    falls = np.flatnonzero(~above[1:] & above[:-1]) + 1
    events = []
    for rise in rises:
        later = falls[falls > rise]
        events.append((int(rise), int(later[0]) if len(later) else len(values) - 1))
    return events


def leg_world(kin, body, slope_y=0.0):
    """Pelvis poses (R (F, 3, 3), p (F, 3)) in a gravity-aligned world, with the left foot fixed flat on the
    floor: z along its sole normal turned about x so that a table measured there as z = a + slope_y * y is
    level, x along the initial pelvis heading, and the initial pelvis at (0, 0, h) above the floor."""
    frames = len(body)
    q = {f"left_{name}_joint": body[:, i] for i, name in enumerate(LEGS)}
    R_pf, p_pf = kin.poses("left_ankle_roll_link", "pelvis", q, frames)
    z = R_pf[0][:, 2]
    x = np.array([1.0, 0.0, 0.0]) - z[0] * z
    x /= np.linalg.norm(x)
    R_wp0 = axis_angle((1.0, 0.0, 0.0), math.atan(-slope_y)) @ np.stack([x, np.cross(z, x), z])
    height = float(-p_pf[0] @ z + SOLE_OFFSET)
    R_p0p = R_pf[0] @ np.transpose(R_pf, (0, 2, 1))
    p_p0p = p_pf[0] - np.einsum("fij,fj->fi", R_p0p, p_pf)
    return R_wp0 @ R_p0p, np.einsum("ij,fj->fi", R_wp0, p_p0p) + np.array([0.0, 0.0, height])


def compose(R_a, p_a, R_b, p_b):
    """Batched pose product (R_a, p_a) * (R_b, p_b)."""
    return R_a @ R_b, p_a + np.einsum("...ij,...j->...i", R_a, p_b)


def torso_world(kin, columns, R_wp, p_wp):
    """World pose of torso_link per frame from the recorded waist joints."""
    body = columns["observation.body"]
    q = {name: body[:, 12 + i] for i, name in enumerate(waist_joints())}
    return compose(R_wp, p_wp, *kin.poses("torso_link", "pelvis", q, len(body)))


def dex1_pinch(kin, columns, R_wt, p_wt, side, source="action"):
    """World pose of the Dex1 pinch frame per frame: the base link's axes at the pad centre."""
    arm = columns[f"{source}.{side}_arm"]
    q = {name: arm[:, i] for i, name in enumerate(arm_joints(side))}
    R, p = compose(R_wt, p_wt, *kin.poses(f"{side}_dex1_base_link", "torso_link", q, len(arm)))
    return R, p + R @ DEX1_PINCH


class HeadCamera:
    """Calibrated head camera fixed to torso_link (X_cam = R X_torso + t), pinhole with radial k1, k2."""

    def __init__(self, fit, width=640, height=480):
        camera = fit["camera"]
        self.R, self.t = np.array(camera["R_torso_to_cam"]), np.array(camera["t_torso_to_cam"])
        (self.fx, self.fy), (self.cx, self.cy) = camera["fx_fy"], camera["cx_cy"]
        self.k1, self.k2 = camera["k1_k2"]
        self.width, self.height = width, height

    def world_pose(self, R_wt, p_wt):
        """Camera axes as world columns and the camera centre."""
        return R_wt @ self.R.T, p_wt + np.einsum("...ij,j->...i", R_wt, -self.R.T @ self.t)

    def distort(self, u, v):
        r2 = u * u + v * v
        d = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
        return self.fx * u * d + self.cx, self.fy * v * d + self.cy

    def undistort(self, px, py, iterations=20):
        x, y = (px - self.cx) / self.fx, (py - self.cy) / self.fy
        u, v = x.copy(), y.copy()
        for _ in range(iterations):
            r2 = u * u + v * v
            d = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
            u, v = x / d, y / d
        return u, v

    def project(self, points, R_wc, c_w):
        """Pixel coordinates and depth of world points."""
        local = (points - c_w) @ R_wc
        px, py = self.distort(local[..., 0] / local[..., 2], local[..., 1] / local[..., 2])
        return np.stack([px, py], -1), local[..., 2]

    def rays(self, pixels, R_wc):
        """World directions (unnormalized, unit depth) through distorted pixel coordinates."""
        u, v = self.undistort(pixels[..., 0], pixels[..., 1])
        return np.stack([u, v, np.ones_like(u)], -1) @ R_wc.T


def closest_on_triangles(p, a, b, c):
    """Row-wise closest points on triangles (closest-point regions of Ericson, RTCD 5.1.5)."""
    ab, ac = b - a, c - a
    d1, d2 = np.einsum("ij,ij->i", ab, p - a), np.einsum("ij,ij->i", ac, p - a)
    d3, d4 = np.einsum("ij,ij->i", ab, p - b), np.einsum("ij,ij->i", ac, p - b)
    d5, d6 = np.einsum("ij,ij->i", ab, p - c), np.einsum("ij,ij->i", ac, p - c)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        t_ab, t_ac, t_bc = d1 / (d1 - d3), d2 / (d2 - d6), (d4 - d3) / ((d4 - d3) + (d5 - d6))
        v, w = vb / (va + vb + vc), vc / (va + vb + vc)
    regions = [(d1 <= 0) & (d2 <= 0), (d3 >= 0) & (d4 <= d3), (vc <= 0) & (d1 >= 0) & (d3 <= 0),
               (d6 >= 0) & (d5 <= d6), (vb <= 0) & (d2 >= 0) & (d6 <= 0), (va <= 0) & (d4 >= d3) & (d5 >= d6)]
    closest = [a, b, a + ab * t_ab[:, None], c, a + ac * t_ac[:, None], b + (c - b) * t_bc[:, None]]
    return np.select([r[:, None] for r in regions], closest, a + ab * v[:, None] + ac * w[:, None])


def project_on_triangles(point, triangles):
    corners = triangles.transpose(1, 0, 2)
    closest = closest_on_triangles(np.broadcast_to(point, corners[0].shape), *corners)
    index = int(np.argmin(np.linalg.norm(closest - point, axis=1)))
    return closest[index], index


def closest_pair(first, second, iterations=20):
    """Closest points of two triangle sets: the closest vertex pair refined by alternating exact projections,
    which never increase the distance. Returns (p, triangle index in first, q, triangle index in second)."""
    vertices = np.unique(first.reshape(-1, 3), axis=0)
    distance, _ = cKDTree(np.unique(second.reshape(-1, 3), axis=0)).query(vertices)
    p = vertices[int(np.argmin(distance))]
    i = j = 0
    for _ in range(iterations):
        q, j = project_on_triangles(p, second)
        p, i = project_on_triangles(q, first)
    return p, i, q, j


def inspire_pinch(kin, side, thickness, kp, pinch_force, thumb_rotation, thumb_bend, squeeze=5.0):
    """Thumb-index pinch of one Inspire hand in its wrist yaw frame. The thumb holds ``thumb_rotation`` and bends
    to ``thumb_bend``; the index bends until the pads are one cloth thickness apart; the middle, ring and little
    fingers stay half closed. The pinch frame has x from the hand base towards the pad midpoint and y from the
    thumb pad to the index pad. The closed thumb target passes contact by ``squeeze`` times its torque limit over
    kp, so the drive saturates and the pads press with ``pinch_force``."""
    wrist = f"{side}_wrist_yaw_link"
    joints = {name: kin.joint(f"{side}_{name}_joint") for name in HAND}
    upper = {name: joint["upper"] for name, joint in joints.items()}
    for name, value in (("thumb_1", thumb_rotation), ("thumb_2", thumb_bend)):
        if not joints[name]["lower"] <= value <= upper[name]:
            raise ValueError(f"the {name} posture lies outside its joint range")

    def pose(thumb, index):
        return {f"{side}_thumb_1_joint": thumb_rotation, f"{side}_thumb_2_joint": thumb,
                f"{side}_index_1_joint": index,
                **{f"{side}_{name}_joint": 0.5 * upper[name] for name in ("middle_1", "ring_1", "little_1")}}

    thumb_links, index_links = kin.subtree(f"{side}_thumb_3"), kin.subtree(f"{side}_index_2")

    def pads(thumb, index):
        frames = kin.frames(wrist, pose(thumb, index))
        (first, owner), (second, _) = kin.triangles(frames, thumb_links), kin.triangles(frames, index_links)
        p, i, q, _ = closest_pair(first, second)
        return float(np.linalg.norm(q - p)), p, str(owner[i]), q, frames

    step = 0.01 * upper["index_1"]
    closed = next((b for b in np.arange(1, 101) * step if pads(thumb_bend, b)[0] < thickness), None)
    if closed is None:
        raise ValueError("thumb and index never close to one cloth thickness")
    low, high = closed - step, closed
    for _ in range(30):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if pads(thumb_bend, middle)[0] > thickness else (low, middle)
    index_bend = 0.5 * (low + high)
    closing = [pads(g * thumb_bend, g * index_bend)[0] for g in np.linspace(0.0, 0.95, 20)]
    if not min(closing) > thickness:
        raise ValueError("the pads must stay more than one cloth thickness apart until the hand closes")
    gap, p, thumb_link, q, frames = pads(thumb_bend, index_bend)
    normal, centre = (q - p) / np.linalg.norm(q - p), 0.5 * (p + q)
    approach = centre - frames[f"{side}_base_link"][1]
    approach -= (approach @ normal) * normal
    approach /= np.linalg.norm(approach)
    rotation = np.column_stack([approach, normal, np.cross(approach, normal)])
    local = frames[thumb_link][0].T @ (p - frames[thumb_link][1])

    def contact(value):
        moved = kin.frames(wrist, pose(value, index_bend))
        return moved[thumb_link][0] @ local + moved[thumb_link][1]

    lever = float(normal @ (contact(thumb_bend + 1e-6) - contact(thumb_bend - 1e-6)) / 2e-6)
    if not lever > 0.0:
        raise ValueError("closing the thumb must move its pad towards the index")
    torque = pinch_force * lever
    thumb_closed = thumb_bend + squeeze * torque / kp
    if thumb_closed > upper["thumb_2"]:
        raise ValueError("the closed thumb target exceeds its joint limit")
    targets = {"thumb_1": (thumb_rotation, thumb_rotation), "thumb_2": (0.0, thumb_closed),
               "index_1": (0.0, index_bend),
               **{name: (0.5 * upper[name], 0.5 * upper[name]) for name in ("middle_1", "ring_1", "little_1")}}
    return {"thumb_rotation": float(thumb_rotation), "thumb_bend": float(thumb_bend),
            "index_bend": float(index_bend), "pad_gap_m": gap, "open_pad_gap_m": closing[0],
            "thumb_pad_link": thumb_link, "pad_links": thumb_links + index_links,
            "pinch_rotation": rotation.tolist(), "pinch_position_m": centre.tolist(), "thumb_lever_m": lever,
            "thumb_torque_limit_nm": float(torque),
            "targets_open_closed": {f"{side}_{name}_joint": list(value) for name, value in targets.items()}}


def dex1_pinch_in_wrist(kin, side):
    """Dex1 pinch frame (base link axes at the pad centre) in the wrist yaw frame."""
    R, p = kin.poses(f"{side}_dex1_base_link", f"{side}_wrist_yaw_link", {}, 1)
    return R[0], p[0] + R[0] @ DEX1_PINCH


def match_closing_sign(dex1_rotation, pinch_rotation):
    """The pinch frame, or the same frame turned half a turn about its approach axis, whichever is the smaller
    rotation from the Dex1 pinch frame (a parallel pinch closes the same either way)."""
    flipped = pinch_rotation @ np.diag([1.0, -1.0, -1.0])
    angle = lambda candidate: np.linalg.norm(Rotation.from_matrix(dex1_rotation @ candidate.T).as_rotvec())
    return min((pinch_rotation, flipped), key=angle)
