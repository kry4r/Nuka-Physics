"""Replay a G1_Dex1_Fold_Towel episode on the simulated G1 upper body with Inspire hands: ``plan`` retargets the
recorded Dex1 pinch poses onto the arm and the thumb-index pads, ``run`` drives them through the diagnostic session
with side-by-side renders."""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from nuka.tasks import fold_kinematics as fk

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / ".nuka-assets/datasets/g1_fold_towel_105"
GRAVITY = np.array([0.0, 0.0, -9.81])


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def catmull_rom(values, k, tau):
    """Catmull-Rom value between rows k and k + 1 at fraction ``tau``, with clamped ends."""
    last = len(values) - 1
    p0, p1, p2, p3 = (values[min(max(index, 0), last)] for index in (k - 1, k, k + 1, k + 2))
    return 0.5 * (2.0 * p1 + (p2 - p0) * tau + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * tau ** 2
                  + (3.0 * p1 - p0 - 3.0 * p2 + p3) * tau ** 3)


def catmull_rom_rates(values, k, tau, period):
    """Rate and acceleration of the Catmull-Rom curve between rows k and k + 1 spaced ``period`` apart."""
    last = len(values) - 1
    p0, p1, p2, p3 = (values[min(max(index, 0), last)] for index in (k - 1, k, k + 1, k + 2))
    c, d = 2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3, 3.0 * p1 - p0 - 3.0 * p2 + p3
    return (0.5 * ((p2 - p0) + 2.0 * c * tau + 3.0 * d * tau ** 2) / period,
            (c + 3.0 * d * tau) / period ** 2)


def quat_matrix(wxyz):
    w, x, y, z = wxyz
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def load_scene(path):
    scene = json.loads(Path(path).read_text())
    scene["directory"] = Path(path).parent
    return scene


def pinch_tool(scene, side):
    return np.array(scene["hand"][side]["pinch_rotation"]), np.array(scene["hand"][side]["pinch_position_m"])


def solve_frame(sim, side, target, previous, recorded, tool, args):
    """Arm IK that keeps the previous solution's branch while it meets 1 mm and 0.01 rad, otherwise the best of it
    and the recorded arm with the wrist roll turned by 0 and +-90 degrees. Returns (q, info, switched)."""
    names, tip = fk.arm_joints(side), f"{side}_wrist_yaw_link"
    candidates = []
    if previous is not None:
        q, info = fk.solve_ik(sim, tip, "torso_link", names, target, previous, tool,
                              rotation_weight=args.rotation_weight, damping=args.damping, iterations=300)
        if info["position_error_m"] < 1e-3 and info["rotation_error_rad"] < 1e-2:
            return q, info, False
        candidates.append((q, info, False))
    for turn in (0.0, 0.5 * math.pi, -0.5 * math.pi):
        start = np.array(recorded, np.float64)
        start[4] += turn
        q, info = fk.solve_ik(sim, tip, "torso_link", names, target, start, tool, rotation_weight=args.rotation_weight,
                              damping=args.damping, iterations=300 if previous is not None else 1000)
        candidates.append((q, info, previous is not None))
    return min(candidates, key=lambda c: (c[1]["position_error_m"] + args.rotation_weight * c[1]["rotation_error_rad"],
                                          c[2]))


def support_plane(scene, R_s, p_s):
    """The table top as (normal, offset) in the torso frame (R_s, p_s): normal . x >= offset above it."""
    return R_s[2].copy(), scene["table"]["top_m"] - p_s[2]


def finger_floors(scene, positions, R_s, p_s, R, p, open_width, sign):
    """Thumb and index support heights in the torso frame (R_s, p_s): the cloth top or the table under each open jaw
    pad of the pinch frame (R, p), kept while the fingers close so each tip presses where it landed. A point lies
    over the cloth within 0.75 lattice spacings of a particle, just above the largest gap of the square lattice."""
    cloth = scene["cloth"]
    floors = []
    for side in (-1.0, 1.0):
        point = R_s @ (p + side * 0.5 * open_width * sign * R[:, 1]) + p_s
        under = np.linalg.norm(positions[:, :2] - point[:2], axis=1) < 0.75 * cloth["spacing"]
        top = positions[under, 2].max() + 0.5 * cloth["thickness"] if under.any() else -np.inf
        floors.append(max(scene["table"]["top_m"], top) - p_s[2])
    return floors


def jaw_width(args, closure):
    return (1.0 - closure) * args.open_width


def hover_lift(events, frames, height, approach):
    """Height added to a pinch command per frame: ``height`` away from grasps, easing to zero over ``approach``
    frames before each grasp and after its release, and zero while it holds."""
    k = np.arange(frames, dtype=np.float64)
    distance = np.full(frames, np.inf)
    for grasp, release in events:
        distance = np.minimum(distance, np.maximum(np.maximum(grasp - k, k - release), 0.0))
    s = np.clip(distance / max(approach, 1e-9), 0.0, 1.0)
    return height * s * s * (3.0 - 2.0 * s)


def make_plan(scene, camera_fit, args):
    """World pinch commands, retargeted arm and hand targets, waist targets and closures per recorded frame.
    The first frame's arm seeds from the pinch-tool IK branch search at the recorded, unlifted command."""
    episode = scene["episode"]["index"]
    columns = fk.episode_columns(DATA / f"data/chunk-000/episode_{episode:06d}.parquet")
    body = columns["observation.body"]
    frames = len(body)
    dex1 = fk.Kinematics(scene["world"]["dex1_urdf"]["path"])
    sim = fk.Kinematics(scene["directory"] / "upper_body.urdf")
    R_wp, p_wp = fk.leg_world(dex1, body, scene["world"]["slope_y"])
    pelvis_R = quat_matrix(scene["world"]["pelvis_quat_wxyz"])
    pelvis_p = np.array(scene["world"]["pelvis_position_m"])
    if not (np.allclose(pelvis_R, R_wp[0], atol=1e-6) and np.allclose(pelvis_p, p_wp[0], atol=1e-6)):
        raise ValueError("the scene's pelvis pose differs from the episode's leg kinematics")
    R_wt, p_wt = fk.torso_world(dex1, columns, R_wp, p_wp)
    waist = body[:, 12:15]
    q_waist = {name: waist[:, i] for i, name in enumerate(fk.waist_joints())}
    R_ws, p_ws = fk.compose(pelvis_R, pelvis_p, *sim.poses("torso_link", "pelvis", q_waist, frames))
    R_wc, c_w = fk.HeadCamera(camera_fit).world_pose(R_wt, p_wt)
    closure = np.stack([fk.closure(columns, side) for side in fk.SIDES], axis=1)
    plan = {"frames": frames, "times": np.arange(frames) / fk.FPS, "waist": waist, "closure": closure,
            "pelvis_R": R_wp, "pelvis_p": p_wp, "torso_R": R_wt, "torso_p": p_wt, "sim_torso_R": R_ws,
            "sim_torso_p": p_ws, "camera_R": R_wc, "camera_c": c_w}
    report = {"episode": episode, "frames": frames, "sides": {}}
    for side in fk.SIDES:
        R_cmd, p_cmd = fk.dex1_pinch(dex1, columns, R_wt, p_wt, side, "action")
        _, p_obs = fk.dex1_pinch(dex1, columns, R_wt, p_wt, side, "observation")
        events = fk.grasp_events(closure[:, fk.SIDES.index(side)])
        p_rec = p_cmd
        p_cmd = p_cmd + hover_lift(events, frames, args.hover, args.hover_approach * fk.FPS)[:, None] * np.array(
            [0.0, 0.0, 1.0])
        tool = pinch_tool(scene, side)
        names = fk.arm_joints(side)
        retarget = fk.PinchRetarget(sim, side, scene["hand"][side], contact=args.pinch_contact)
        column = fk.SIDES.index(side)
        x = np.zeros((frames, len(retarget.names)))
        hand = np.zeros((frames, len(retarget.hand_names)))
        errors = np.zeros((frames, 3))
        overlap = np.zeros(frames)
        limits = np.zeros(frames, np.int32)
        seed, _, _ = solve_frame(sim, side, (R_ws[0].T @ R_cmd[0], R_ws[0].T @ (p_rec[0] - p_ws[0])), None,
                                 columns[f"action.{side}_arm"][0], tool, args)
        previous = np.concatenate([seed, retarget.open_fingers])
        for k in range(frames):
            target = retarget.target(R_ws[k].T @ R_cmd[k], R_ws[k].T @ (p_cmd[k] - p_ws[k]),
                                     jaw_width(args, closure[k, column]), support_plane(scene, R_ws[k], p_ws[k]),
                                     previous)
            x[k], info = retarget.solve(previous, target, iterations=200 if k == 0 else 40)
            previous = x[k]
            hand[k] = retarget.hand(x[k], jaw_width(args, closure[k, column]))
            errors[k] = (max(info["thumb_pad_error_m"], info["index_pad_error_m"]), info["approach_error_rad"],
                         info["support_depth_m"])
            limits[k] = info["at_limit"]
            overlap[k] = info["finger_overlap_m"]
        q = x[:, :len(names)]
        switches = []
        steps = np.abs(np.diff(q, axis=0)).max(axis=1)
        plan[f"{side}_command_R"], plan[f"{side}_command_p"] = R_cmd, p_cmd
        plan[f"{side}_observed_p"] = p_obs
        plan[f"{side}_arm"], plan[f"{side}_ik_error"], plan[f"{side}_at_limit"] = q, errors, limits
        plan[f"{side}_retarget"], plan[f"{side}_hand"] = x, hand
        plan[f"{side}_events"] = np.array(events, np.int64).reshape(-1, 2)
        lag = np.linalg.norm(p_obs[3:] - p_cmd[:-3], axis=1)
        lower, upper = sim.limits(names)
        margin = np.minimum(q - lower, upper - q)
        report["sides"][side] = {
            "position_error_m": {"max": float(errors[:, 0].max()), "p99": float(np.percentile(errors[:, 0], 99)),
                                 "frames_over_1mm": int((errors[:, 0] > 1e-3).sum()),
                                 "measure": "larger of the thumb and index pad distances to their jaw pads"},
            "rotation_error_rad": {"max": float(errors[:, 1].max()), "p99": float(np.percentile(errors[:, 1], 99)),
                                   "frames_over_0p01": int((errors[:, 1] > 1e-2).sum()),
                                   "measure": "pinch approach axis against the gripper's"},
            "support_depth_m": {"max": float(errors[:, 2].max()),
                                "frames_over_1mm": int((errors[:, 2] > 1e-3).sum())},
            "finger_overlap_m": {"max": float(overlap.max()), "frames_over_1mm": int((overlap > 1e-3).sum()),
                                 "measure": "deepest distal thumb or index hull vertex inside the other's hulls"},
            "frames_at_joint_limit": int((limits > 0).sum()),
            "joint_limit_margin_min_rad": {name: float(margin[:, i].min()) for i, name in enumerate(names)},
            "joint_range_rad": {name: [float(q[:, i].min()), float(q[:, i].max())] for i, name in enumerate(names)},
            "recorded_arm_range_rad": {name: [float(columns[f"action.{side}_arm"][:, i].min()),
                                              float(columns[f"action.{side}_arm"][:, i].max())]
                                       for i, name in enumerate(names)},
            "max_joint_step_rad": float(steps.max()), "joint_step_p99_rad": float(np.percentile(steps, 99)),
            "branch_switch_frames": switches,
            "observed_vs_commanded_3_frames_later_m": {"median": float(np.median(lag)),
                                                        "p95": float(np.percentile(lag, 95))},
            "grasps": [{"grasp_frame": g, "release_frame": r, "grasp_s": g / fk.FPS, "release_s": r / fk.FPS,
                        "command_pinch_m": p_cmd[g].tolist()} for g, r in events]}
    report["pelvis_travel_m"] = np.ptp(p_wp, axis=0).tolist()
    report["hover"] = {"height_m": args.hover, "approach_s": args.hover_approach}
    return plan, report


def cloth_faces(nx, ny):
    i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1))
    a = (j * nx + i).ravel()
    return np.concatenate([np.stack([a, a + 1, a + nx + 1], 1), np.stack([a, a + nx + 1, a + nx], 1)])


def grasp_candidates(positions, faces, nx, ny, threshold=0.25):
    """Corner vertices and fold-edge vertices, the ones touching both an upward and a downward facing triangle."""
    normals = np.cross(positions[faces[:, 1]] - positions[faces[:, 0]], positions[faces[:, 2]] - positions[faces[:, 0]])
    nz = normals[:, 2] / np.maximum(np.linalg.norm(normals, axis=1), 1e-30)
    up, down = np.zeros(len(positions), bool), np.zeros(len(positions), bool)
    for column in range(3):
        np.logical_or.at(up, faces[:, column], nz > threshold)
        np.logical_or.at(down, faces[:, column], nz < -threshold)
    corners = np.array([0, nx - 1, (ny - 1) * nx, nx * ny - 1])
    return np.union1d(corners, np.flatnonzero(up & down))


class Alignment:
    """Per-hand grasp offsets from the commanded pinch point to the nearest corner or fold-edge vertex, rising over
    ``window`` frames before each grasp while following that vertex, held until release and falling after it.
    The pinch centre sits ``inset`` along the closing axis into the cloth and ``press`` below the vertex's
    underside, so each open fingertip loads its support through the arm drives as the fingers close."""

    def __init__(self, events, command_p, closing_axis, window_frames, inset, press, radius=0.05):
        self.events, self.command_p, self.window = [tuple(e) for e in events], command_p, window_frames
        self.closing_axis, self.inset, self.press, self.radius = closing_axis, inset, press, radius
        self.vertex, self.offset, self.log = {}, {}, []

    def inward(self, positions, vertex, grasp):
        """Horizontal closing axis signed toward the cloth around ``vertex``."""
        axis = self.closing_axis[grasp] * np.array([1.0, 1.0, 0.0])
        axis /= max(np.linalg.norm(axis), 1e-12)
        near = positions[np.linalg.norm(positions - positions[vertex], axis=1) < self.radius]
        return axis if axis @ (near.mean(axis=0) - positions[vertex]) >= 0.0 else -axis

    def weight(self, index, frame):
        grasp, release = self.events[index]
        rise = smoothstep((frame - (grasp - self.window)) / self.window)
        fall = 1.0 - smoothstep((frame - release) / self.window)
        return float(min(rise, fall))

    def update(self, frame, positions, faces, nx, ny):
        for index, (grasp, release) in enumerate(self.events):
            if grasp - self.window <= frame <= grasp:
                point = self.command_p[grasp]
                if index not in self.vertex:
                    candidates = grasp_candidates(positions, faces, nx, ny)
                    chosen = int(candidates[np.argmin(np.linalg.norm(positions[candidates] - point, axis=1))])
                    self.vertex[index] = chosen
                    self.log.append({"grasp": index, "chosen_at_frame": frame, "vertex": chosen,
                                     "candidates": int(len(candidates)),
                                     "distance_m": float(np.linalg.norm(positions[chosen] - point))})
                vertex = self.vertex[index]
                self.offset[index] = (positions[vertex] + self.inset * self.inward(positions, vertex, grasp)
                                      - np.array([0.0, 0.0, self.press]) - point)

    def delta(self, frame):
        total, weights = np.zeros(3), 0.0
        for index in self.offset:
            w = self.weight(index, frame)
            total += w * self.offset[index]
            weights += w
        return total / max(1.0, weights)


def render_maps(camera):
    """Size of a centred pinhole render with the recorded focal length that covers every recorded pixel, and the
    remap from the recorded (distorted) image into it; pixel centres sit at integer indices in both images."""
    py, px = np.mgrid[0:camera.height, 0:camera.width].astype(np.float64)
    u, v = camera.undistort(px, py)
    half_w = math.ceil(camera.fx * np.abs(u).max() + 1.0)
    half_h = math.ceil(camera.fy * np.abs(v).max() + 1.0)
    map_x = (half_w - 0.5 + camera.fx * u).astype(np.float32)
    map_y = (half_h - 0.5 + camera.fy * v).astype(np.float32)
    return map_x, map_y, 2 * half_w, 2 * half_h


def run(scene, camera_fit, plan, args):
    import av
    import cv2
    import nuka
    from nuka.diagnostics import DiagnosticSession, DiagnosticThresholds
    from nuka.diagnostics.capture import sha256, write_json
    from nuka.tasks.fold_scene import build_scene, folded_cloth

    tolerance = os.environ.get("NUKA_SOLVER_VEL_TOLERANCE")
    if tolerance is None:
        raise ValueError("set the production NUKA_SOLVER_VEL_TOLERANCE explicitly for reproducibility")
    substeps = round(1.0 / (fk.FPS * args.dt))
    if not math.isclose(substeps * args.dt * fk.FPS, 1.0, rel_tol=1e-9):
        raise ValueError("the recording period must be a whole number of time steps")
    frames = plan["frames"] if args.frames is None else min(args.frames, plan["frames"])
    settle = round(args.settle / args.dt)
    window = round(args.align_window * fk.FPS)
    start = 0 if args.resume_checkpoint is None else args.resume_frame
    if start and start >= min([int(e[0]) for side in fk.SIDES for e in plan[f"{side}_events"]], default=frames) - window:
        raise ValueError("resume before the first grasp alignment window, where the run state is the world alone")
    output = Path(args.output)
    device = nuka.Device.create(0)
    world, config = build_scene(device, scene["directory"] / "scene.json", dt=args.dt, sweeps=args.sweeps,
                                contact_capacity=args.contact_capacity)
    session = None
    try:
        names = list(world.dof_names())
        slots = config["active_indices"]
        arm = {side: [slots[name] for name in fk.arm_joints(side)] for side in fk.SIDES}
        waist_slots = [slots[name] for name in fk.waist_joints()]
        ff_names = fk.waist_joints() + fk.arm_joints("left") + fk.arm_joints("right")
        ff_slots = [slots[name] for name in ff_names]
        hand_names = {side: list(scene["hand"][side]["targets_open_closed"]) for side in fk.SIDES}
        hand_slots = {side: [slots[name] for name in hand_names[side]] for side in fk.SIDES}
        pelvis = (quat_matrix(scene["world"]["pelvis_quat_wxyz"]), np.array(scene["world"]["pelvis_position_m"]))
        targets = np.asarray(world.download_field(nuka.Field.DRIVE_TARGET), np.float32).reshape(-1).copy()
        feed = np.asarray(world.download_field(nuka.Field.JOINT_FEEDFORWARD), np.float32).reshape(-1).copy()
        velocity = np.asarray(world.download_field(nuka.Field.VELOCITY_TARGET), np.float32).reshape(-1).copy()
        initial = np.asarray(world.download_field(nuka.Field.JOINT_POSITION), np.float32).reshape(-1).copy()
        cloth = config["cloth"]
        nx, ny = cloth["nx"], cloth["ny"]
        rest = np.asarray(world.download_field(nuka.Field.PARTICLE_POSITION)).reshape(-1, 3).copy()
        if rest.shape[0] != nx * ny:
            raise ValueError("the world's particles differ from the authored towel lattice")
        placed = folded_cloth(config).astype(np.float32)
        world.upload_field(nuka.Field.PARTICLE_POSITION, np.ascontiguousarray(placed.reshape(-1)))
        positions, velocities = placed.astype(np.float64), np.zeros(placed.shape)
        faces = cloth_faces(nx, ny)
        metadata = {"fixture": "towel fold replay", "scene": str(scene["directory"] / "scene.json"),
                    "scene_sha256": sha256(scene["directory"] / "scene.json"), "episode": scene["episode"],
                    "dt": args.dt, "sweeps": args.sweeps, "substeps_per_frame": substeps, "settle_steps": settle,
                    "align_window_frames": window, "replayed_frames": frames, "pinch_contact": args.pinch_contact,
                    "open_width_m": args.open_width, "edge_margin_m": args.edge_margin, "press_depth_m": args.press_depth,
                    "drive_inputs": "spline targets, their rates as velocity targets and the inverse dynamics of the "
                                    "commanded motion as waist and arm feedforward",
                    "first_step": settle + start * substeps + 1 if start else 1,
                    "resume_checkpoint": None if start == 0 else str(args.resume_checkpoint),
                    "solver_velocity_tolerance_mps": float(tolerance), "ogc_contact_capacity": args.contact_capacity,
                    "build_record": str(args.build_record),
                    "binary_hashes": (Path(args.build_record) / "binaries.sha256").read_text(),
                    "replay_sha256": sha256(__file__), "kinematics_sha256": sha256(fk.__file__),
                    "owner_names": {"LINK": names}, "render_acceptance": "unmeasured",
                    "claims_full_physics_acceptance": False}
        session = DiagnosticSession(world, output / "session", metadata, chunk_steps=args.chunk_steps,
                                    state_fields=(nuka.Field.ARTICULATION_LINK_POSE, nuka.Field.JOINT_POSITION,
                                                  nuka.Field.JOINT_VELOCITY, nuka.Field.JOINT_LIMIT_IMPULSE,
                                                  nuka.Field.LINK_CONTACT_WRENCH, nuka.Field.JOINT_FEEDFORWARD,
                                                  nuka.Field.VELOCITY_TARGET),
                                    thresholds=DiagnosticThresholds(velocity_tolerance_mps=float(tolerance)))
        np.savez_compressed(output / "initial.npz", folded=placed, flat_rest=rest, faces=faces,
                            joint_position=initial)
        inset = 0.5 * args.open_width + args.edge_margin
        alignment = {side: Alignment(plan[f"{side}_events"], plan[f"{side}_command_p"],
                                     plan[f"{side}_command_R"][:, :, 1], window, inset,
                                     max(args.press_depth) + 0.5 * scene["cloth"]["thickness"])
                     for side in fk.SIDES}
        x_run = {side: plan[f"{side}_retarget"].copy() for side in fk.SIDES}
        h_run = {side: plan[f"{side}_hand"].copy() for side in fk.SIDES}
        delta = {side: np.zeros((plan["frames"], 3)) for side in fk.SIDES}
        held = {side: None for side in fk.SIDES}
        sim = fk.Kinematics(scene["directory"] / "upper_body.urdf")
        retarget = {side: fk.PinchRetarget(sim, side, scene["hand"][side], contact=args.pinch_contact)
                    for side in fk.SIDES}
        camera = fk.HeadCamera(camera_fit)
        map_x, map_y, render_w, render_h = render_maps(camera)
        fov = math.degrees(2.0 * math.atan(0.5 * render_h / camera.fy))
        video = DATA / f"videos/chunk-000/observation.images.cam_left_high/episode_{scene['episode']['index']:06d}.mp4"
        decoder = enumerate(av.open(str(video)).decode(video=0))
        (output / "render").mkdir()
        buffers, particle_chunks, frame_log = [], [], []
        reason, wall = None, time.perf_counter()

        def set_controls(q, qd, qdd):
            """Drive targets ``q`` with velocity targets ``qd``, and the inverse dynamics of the commanded motion
            as the waist and arm feedforward."""
            for name, value in q.items():
                targets[slots[name]] = value
                velocity[slots[name]] = qd[name]
            feed[ff_slots] = sim.inverse_dynamics("pelvis", pelvis, q, qd, qdd, ff_names, GRAVITY)
            world.set_drive_targets(np.ascontiguousarray(targets))
            world.upload_field(nuka.Field.VELOCITY_TARGET, np.ascontiguousarray(velocity))
            world.upload_field(nuka.Field.JOINT_FEEDFORWARD, np.ascontiguousarray(feed))

        def command(k, tau):
            """Commanded coordinates, rates and accelerations at fraction ``tau`` of recorded frame k."""
            period = 1.0 / fk.FPS
            q, qd, qdd = {}, {}, {}
            splines = [(fk.arm_joints(side), x_run[side][:, :7]) for side in fk.SIDES]
            splines.append((fk.waist_joints(), plan["waist"]))
            for names, values in splines:
                rate, acceleration = catmull_rom_rates(values, k, tau, period)
                q.update(zip(names, catmull_rom(values, k, tau)))
                qd.update(zip(names, rate))
                qdd.update(zip(names, acceleration))
            for side in fk.SIDES:
                q.update(zip(hand_names[side], (1 - tau) * h_run[side][k] + tau * h_run[side][k + 1]))
                qd.update(zip(hand_names[side], (h_run[side][k + 1] - h_run[side][k]) / period))
            return q, qd, qdd

        def advance():
            sample = session.step(controls=targets.copy())
            return "physics_failure" if np.any(sample["env_status"]) else None

        def flush_particles():
            if not buffers:
                return
            path = output / f"particles_{len(particle_chunks):04d}.npz"
            with path.open("xb") as target:
                np.savez_compressed(target, frame=np.array([b[0] for b in buffers]),
                                    position=np.stack([b[1] for b in buffers]),
                                    velocity=np.stack([b[2] for b in buffers]))
            particle_chunks.append({"file": path.name, "sha256": sha256(path), "frames": len(buffers)})
            buffers.clear()

        def keep_particles(frame):
            x = np.asarray(world.download_field(nuka.Field.PARTICLE_POSITION)).reshape(-1, 3).astype(np.float64)
            v = np.asarray(world.download_field(nuka.Field.PARTICLE_VELOCITY)).reshape(-1, 3).astype(np.float64)
            buffers.append((frame, x.astype(np.float32), v.astype(np.float32)))
            if len(buffers) == round(fk.FPS):
                flush_particles()
            return x, v

        def render(frame):
            nonlocal decoder
            index, recorded = next(decoder)
            while index < frame:
                index, recorded = next(decoder)
            R, c = plan["camera_R"][frame], plan["camera_c"][frame]
            pixels = np.asarray(world.render_beauty(eye=c.tolist(), look=(c + R[:, 2]).tolist(), up=(-R[:, 1]).tolist(),
                                                    fov_deg=fov, width=render_w, height=render_h, spp=args.spp))
            simulated = cv2.remap(cv2.cvtColor(pixels[..., :3], cv2.COLOR_RGB2BGR), map_x, map_y, cv2.INTER_LINEAR)
            pair = np.hstack([recorded.to_ndarray(format="bgr24"), simulated])
            cv2.putText(pair, f"ep{scene['episode']['index']} t={frame / fk.FPS:5.2f}s recorded | simulated",
                        (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(output / "render" / f"pair_{frame:05d}.png"), pair)

        first = {side: x_run[side][0, :7] for side in fk.SIDES}
        hold = max(settle // 2, 1)
        if start:
            checkpoint = world.read_checkpoint(str(args.resume_checkpoint))
            world.restore_checkpoint(checkpoint)
            checkpoint.close()
        settle_from = dict(zip(fk.waist_joints(), initial[waist_slots]))
        settle_to = dict(zip(fk.waist_joints(), plan["waist"][0]))
        for side in fk.SIDES:
            settle_from.update(zip(fk.arm_joints(side) + hand_names[side],
                                   np.concatenate([initial[arm[side]], initial[hand_slots[side]]])))
            settle_to.update(zip(fk.arm_joints(side) + hand_names[side], np.concatenate([first[side], h_run[side][0]])))
        for step in range(0 if start else settle):
            x = min((step + 1) / hold, 1.0)
            blend = smoothstep(x)
            # The smoothstep's rate and acceleration, zero once the blend has finished.
            rate = 6.0 * x * (1.0 - x) / (hold * args.dt) if x < 1.0 else 0.0
            acceleration = 6.0 * (1.0 - 2.0 * x) / (hold * args.dt) ** 2 if x < 1.0 else 0.0
            set_controls({n: (1 - blend) * settle_from[n] + blend * settle_to[n] for n in settle_from},
                         {n: rate * (settle_to[n] - settle_from[n]) for n in settle_from},
                         {n: acceleration * (settle_to[n] - settle_from[n]) for n in settle_from})
            reason = advance()
            if reason:
                break
        if reason is None:
            positions, velocities = keep_particles(start)
            if args.render_every:
                render(start)
            for k in range(start, frames - 1):
                for side in fk.SIDES:
                    alignment[side].update(k, positions, faces, nx, ny)
                    for j in (k + 1, k + 2):
                        if j >= plan["frames"]:
                            continue
                        delta[side][j] = alignment[side].delta(j)
                        if not np.any(delta[side][j]):
                            x_run[side][j] = plan[f"{side}_retarget"][j]
                            h_run[side][j] = plan[f"{side}_hand"][j]
                            continue
                        R_s, p_s = plan["sim_torso_R"][j], plan["sim_torso_p"][j]
                        g = plan["closure"][j, fk.SIDES.index(side)]
                        R_t = R_s.T @ plan[f"{side}_command_R"][j]
                        p_t = R_s.T @ (plan[f"{side}_command_p"][j] + delta[side][j] - p_s)
                        support = support_plane(scene, R_s, p_s)
                        floors = held[side] if held[side] is not None else finger_floors(
                            scene, positions, R_s, p_s, R_t, p_t, args.open_width, retarget[side].sign)
                        # Supports are held while the pinch is within a contact band of them, so cloth gathered
                        # under a landing point cannot shift one tip's goal against the other's.
                        held[side] = floors if support[0] @ p_t - max(floors) < retarget[side].band else None
                        target = retarget[side].target(R_t, p_t, jaw_width(args, g), support,
                                                       x_run[side][j - 1], args.press_depth, floors)
                        x_run[side][j], info = retarget[side].solve(x_run[side][j - 1], target)
                        h_run[side][j] = retarget[side].hand(x_run[side][j], jaw_width(args, g))
                        frame_log.append({"frame": j, "side": side, "delta_m": delta[side][j].tolist(),
                                          "ik_position_error_m": max(info["thumb_pad_error_m"],
                                                                     info["index_pad_error_m"]),
                                          "ik_rotation_error_rad": info["approach_error_rad"],
                                          "support_depth_m": info["support_depth_m"],
                                          "finger_floors_m": [f + float(p_s[2]) for f in floors],
                                          "finger_overlap_m": info["finger_overlap_m"]})
                for m in range(1, substeps + 1):
                    tau = m / substeps
                    set_controls(*command(k, tau))
                    reason = advance()
                    if reason:
                        break
                if reason:
                    break
                positions, velocities = keep_particles(k + 1)
                if not (np.isfinite(positions).all() and np.isfinite(velocities).all()):
                    reason = "nonfinite_state"
                    break
                if k + 1 == args.checkpoint_frame:
                    checkpoint = world.capture_checkpoint()
                    checkpoint.write(str(output / f"checkpoint_{k + 1:05d}.bin"))
                    checkpoint.close()
                if args.render_every and ((k + 1) % args.render_every == 0 or k + 1 == frames - 1):
                    render(k + 1)
        reason = reason or "completed"
        flush_particles()
        session.close(reason)
        speeds = np.linalg.norm(velocities, axis=1)
        summary = {"stop_reason": reason, "policy_steps": session.steps, "replayed_frames": frames,
                   "host_wall_seconds": time.perf_counter() - wall,
                   "wall_scope": "host wall time of the whole run including capture, IK and rendering",
                   "particle_chunks": particle_chunks, "alignment": {side: alignment[side].log for side in fk.SIDES},
                   "aligned_frames": len(frame_log),
                   "aligned_ik_position_error_max_m": max([e["ik_position_error_m"] for e in frame_log], default=0.0),
                   "final_speed_max_mps": float(speeds.max()), "cloth_lowest_z_m": float(positions[:, 2].min()),
                   "table_top_m": scene["table"]["top_m"], "render_size": [render_w, render_h], "render_fov_deg": fov,
                   "claims_full_physics_acceptance": False}
        np.savez_compressed(output / "replay_targets.npz", **{f"{side}_retarget": x_run[side] for side in fk.SIDES},
                            **{f"{side}_hand": h_run[side] for side in fk.SIDES},
                            **{f"{side}_delta": delta[side] for side in fk.SIDES})
        write_json(output / "alignment_log.json", frame_log)
        write_json(output / "summary.json", summary)
        print(json.dumps({key: value for key, value in summary.items() if key != "particle_chunks"}), flush=True)
    except BaseException as error:
        if session is not None:
            session.close("exception", error=repr(error))
        raise
    finally:
        world.destroy()
        device.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run"))
    parser.add_argument("--scene", type=Path, required=True, help="scene.json from build_g1_inspire_fold.py")
    parser.add_argument("--camera-fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rotation-weight", type=float, default=0.1, help="IK metres per radian of rotation error")
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--build-record", type=Path)
    parser.add_argument("--dt", type=float, default=1.0 / 600.0)
    parser.add_argument("--sweeps", type=int, default=512)
    parser.add_argument("--contact-capacity", type=int, default=524288)
    parser.add_argument("--chunk-steps", type=int, default=600)
    parser.add_argument("--settle", type=float, default=1.0, help="seconds before the first recorded frame")
    parser.add_argument("--align-window", type=float, default=1.0, help="grasp alignment ramp (s)")
    parser.add_argument("--open-width", type=float, default=0.04, help="jaw pad distance of the open gripper (m)")
    parser.add_argument("--hover", type=float, default=0.04, help="pinch command lift away from grasps (m)")
    parser.add_argument("--hover-approach", type=float, default=0.5,
                        help="seconds over which the lift eases out before a grasp and back in after its release")
    parser.add_argument("--edge-margin", type=float, default=0.005,
                        help="distance of the outer open fingertip inside the cloth edge (m)")
    parser.add_argument("--press-depth", type=float, nargs="+", default=[0.008],
                        help="target depth of the thumb and index tips below the support each lands on (m), one "
                             "value for both or a thumb and index pair")
    parser.add_argument("--pinch-contact", choices=("pad", "tip"), default="tip",
                        help="thumb and index contact points the retarget places on the gripper's jaw pads")
    parser.add_argument("--frames", type=int, help="replay only this many recorded frames")
    parser.add_argument("--render-every", type=int, default=15, help="recorded frames between renders, 0 for none")
    parser.add_argument("--spp", type=int, default=16)
    parser.add_argument("--plan", type=Path, help="reuse the plan.npz and plan.json saved in this directory")
    parser.add_argument("--checkpoint-frame", type=int, help="write the world checkpoint at the end of this frame")
    parser.add_argument("--resume-checkpoint", type=Path, help="world checkpoint a run wrote at --resume-frame")
    parser.add_argument("--resume-frame", type=int, default=0, help="frame whose end --resume-checkpoint holds")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.mode == "run" and args.build_record is None:
        raise ValueError("run needs --build-record")
    scene = load_scene(args.scene)
    camera_fit = json.loads(args.camera_fit.read_text())
    if args.plan is not None:
        plan = dict(np.load(args.plan / "plan.npz"))
        report = json.loads((args.plan / "plan.json").read_text())
        report["plan_source"] = str(args.plan)
    else:
        started = time.perf_counter()
        plan, report = make_plan(scene, camera_fit, args)
        report["plan_seconds"] = time.perf_counter() - started
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "plan.npz", **{key: np.asarray(value) for key, value in plan.items()})
    report.update(scene=str(args.scene), camera_fit=str(args.camera_fit), rotation_weight=args.rotation_weight,
                  damping=args.damping, open_width_m=args.open_width, pinch_contact=args.pinch_contact)
    (args.output / "plan.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({side: {key: report["sides"][side][key] for key in
                             ("position_error_m", "rotation_error_rad", "frames_at_joint_limit")}
                      for side in fk.SIDES}), flush=True)
    if args.mode == "run":
        run(scene, camera_fit, plan, args)


if __name__ == "__main__":
    main()
