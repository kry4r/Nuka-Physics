"""GENERATOR (i-A'): emit the EXACT reduced-DOF H1 training MJCF + verify it.

This is the SOURCE OF TRUTH for the reduced H1 training asset
``.nuka-assets/newton_assets/unitree_h1/mjcf/h1_reduced_train.xml`` (regenerable).
The prior spikes (``h1_standing_pd_gate.py`` / ``h1_balance_gate.py``) lumped a
GUESSED upper-body mass (z=0.18, isotropic inertia=0.8) into the pelvis. This
generator computes the lump EXACTLY: it forward-kinematics EVERY upper-body body
(torso + both arms + both hands + all fingers + imu site host) to the pelvis frame
at the upper-body REST pose (MJCF defaults, upper joints q=0), then accumulates
each body's real ``<inertial>`` (mass + CoM + diaginertia + inertial-frame quat)
into a single combined rigid body via the PARALLEL-AXIS THEOREM, and adds the
pelvis's own inertial. The result replaces the pelvis ``<inertial>`` in the 16-DOF
(6 floating base + 10 leg joints) reduced robot.

WHY this matters: the articulation cooker builds links by a BFS over JOINTS, so a
welded (jointless) body's mass is silently DROPPED. The reduced robot loses ~48%
of the full-H1 mass unless the upper body is lumped back. Getting the lump EXACT
(not the prior guess) means the RL campaign trains on dynamics that MATCH the
deploy H1 (the same h1_with_hand robot used for grasping).

ENGINE FACTS verified by reading src/import/mjcf_importer.cpp (drive the emit):
  * The importer reads inertial ``mass``, ``pos``, ``quat`` (w x y z, consumed at
    line 468-469), and ``diaginertia`` (Vec3). It does NOT parse ``fullinertia``.
    The combined upper-body inertia is a FULL symmetric 3x3 (arms break diagonality
    -> nonzero off-diagonals), so we EIGENDECOMPOSE it into principal-axis
    diaginertia + a principal-axis quat (det(R)=+1) and emit BOTH -- NO precision
    loss vs the true tensor, because the importer honors the inertial quat.
  * The importer derives mass ONLY from ``<inertial mass=...>``; it does NOT
    compute mass from geom density. So the added foot spheres carry ZERO mass and
    mass closure is exact.
  * The importer reads ``forcerange`` for actuator limits, NOT ``ctrlrange``. So
    nuka does NOT enforce the MJCF ctrlrange; the controller clamps torque in
    python (as the balance gate does). We still keep the 10 leg motors with their
    real ctrlrange/gear in the emitted asset (clean, MuJoCo-loadable, the task's
    requirement) and report that nuka itself ignores ctrlrange.

MASS CLOSURE BY CONSTRUCTION: lumped_pelvis_mass = pelvis_own + (full_total -
pelvis_own - Sigma 10 leg masses). So Sigma(emitted reduced inertial masses) ==
full_total EXACTLY (gate b cannot miss if the FK enumerates every body, which we
do by walking the XML tree -- NOT a hand list). A cross-check asserts the FK'd
upper mass equals (full_total - pelvis_own - Sigma legs).

Run (from repo root):
  export PATH=/root/.nuka-toolchain-gcc14/bin:$PATH
  export CUDA_VISIBLE_DEVICES=0
  /root/miniconda3/envs/nuka-v03/bin/python python/spikes/gen_h1_reduced_train.py
"""

from __future__ import annotations

import math
import pathlib
import xml.etree.ElementTree as ET

import numpy as np
import torch

import nuka

REPO = pathlib.Path(__file__).resolve().parents[2]
H1_MJCF = REPO / ".nuka-assets/newton_assets/unitree_h1/mjcf/h1_with_hand.xml"
OUT_MJCF = REPO / ".nuka-assets/newton_assets/unitree_h1/mjcf/h1_reduced_train.xml"
DT = 0.005
G = 9.81

# The 10 leg-chain link names KEPT in the reduced robot (besides the pelvis root).
LEG_LINKS = [
    "left_hip_yaw_link", "left_hip_roll_link", "left_hip_pitch_link",
    "left_knee_link", "left_ankle_link",
    "right_hip_yaw_link", "right_hip_roll_link", "right_hip_pitch_link",
    "right_knee_link", "right_ankle_link",
]
# Leg joints in cooked q[1:] order (slots 1..10).
LEG_JOINTS = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
# Real H1 actuator ctrlrange (N*m, gear=1) per leg joint, from h1_with_hand <actuator>.
CTRLRANGE = {
    "hip_yaw": 200.0, "hip_roll": 200.0, "hip_pitch": 200.0,
    "knee": 300.0, "ankle": 40.0,
}

# Foot spheres: REUSE the balance gate's symmetric foot (toe +0.10 / heel -0.10)
# so this is the SAME robot that PASSED the balance gate. radius 0.025, centre z
# -0.055 below the ankle joint (ankle-link frame). MASSLESS (importer ignores
# geom density), so mass closure is exact.
FOOT_TOE_X = 0.10
FOOT_HEEL_X = -0.10
FOOT_SPHERE_R = 0.025
FOOT_BOTTOM_Z = -0.055

PELVIS_Z0 = 1.1
# Nominal standing pose (legs); upper body at rest (q=0).
NOMINAL = {"hip_pitch": -0.40, "knee": 0.70, "ankle": -0.30}


# --------------------------------------------------------------------------- #
# Quaternion / rotation helpers (w,x,y,z; same convention as the MJCF importer).
def quat_to_mat(q):
    """q=[w,x,y,z] -> 3x3 rotation matrix (numpy)."""
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_quat(R):
    """3x3 rotation (det +1) -> [w,x,y,z], w>=0."""
    t = np.trace(R)
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def parse_vec(text, default):
    if text is None:
        return np.array(default, dtype=float)
    return np.array([float(v) for v in text.split()], dtype=float)


# --------------------------------------------------------------------------- #
# FK + inertia accumulation.
def fk_and_lump():
    """Walk the H1 tree; FK every upper-body body to the pelvis frame at the rest
    pose (upper joints q=0); accumulate inertials via the parallel-axis theorem.

    Returns a dict with the combined (pelvis_own + lumped upper) inertial in the
    pelvis frame, plus accounting numbers for the verification gate.
    """
    tree = ET.parse(H1_MJCF)
    worldbody = tree.getroot().find("worldbody")
    pelvis = worldbody.find("body")
    assert pelvis.get("name") == "pelvis", pelvis.get("name")

    leg_set = set(LEG_LINKS)

    # full-H1 mass accounting + pelvis own inertial.
    all_masses = []

    def inertial_of(body):
        ine = body.find("inertial")
        m = float(ine.get("mass"))
        com = parse_vec(ine.get("pos"), [0, 0, 0])
        diag = parse_vec(ine.get("diaginertia"), [0, 0, 0])
        q = parse_vec(ine.get("quat"), [1, 0, 0, 0])
        R = quat_to_mat(q)
        I_body = R @ np.diag(diag) @ R.T  # inertia about body CoM, body-frame axes.
        return m, com, I_body

    pelvis_m, pelvis_com, pelvis_I = inertial_of(pelvis)

    # Accumulators for the combined body about the PELVIS frame origin first
    # (combine masses+first moments+inertia-about-origin), convert to CoM at the end.
    # We separately track the UPPER-body-only lump (for cross-checks) and the
    # full combined (pelvis_own + upper).
    upper = {"m": 0.0, "mr": np.zeros(3), "I_origin": np.zeros((3, 3)),
             "n": 0}

    def add_body(m, com_pelvis, I_about_com_pelvisaxes, acc):
        """Add a body whose CoM is com_pelvis (pelvis frame) and inertia tensor
        I_about_com_pelvisaxes is about ITS OWN CoM expressed in pelvis-frame axes."""
        acc["m"] += m
        acc["mr"] += m * com_pelvis
        # shift this body's inertia from its CoM to the pelvis ORIGIN (parallel axis):
        d = com_pelvis  # vector from origin to body CoM
        shift = m * ((d @ d) * np.eye(3) - np.outer(d, d))
        acc["I_origin"] += I_about_com_pelvisaxes + shift
        acc["n"] += 1

    def recurse(body, T_parent_pos, T_parent_rot):
        """body transform = parent * (body pos/quat). Lump if upper-body."""
        name = body.get("name")
        bpos = parse_vec(body.get("pos"), [0, 0, 0])
        bquat = parse_vec(body.get("quat"), [1, 0, 0, 0])
        Rb = quat_to_mat(bquat)
        # world(=pelvis-frame) transform of this body:
        T_pos = T_parent_pos + T_parent_rot @ bpos
        T_rot = T_parent_rot @ Rb

        is_pelvis = (name == "pelvis")
        is_leg = name in leg_set
        is_upper = (not is_pelvis) and (not is_leg)

        if is_upper:
            m, com_local, I_body = inertial_of(body)
            all_masses.append(m)
            # CoM in pelvis frame:
            com_pelvis = T_pos + T_rot @ com_local
            # inertia about body CoM, rotated body-frame -> pelvis-frame axes:
            I_pelvisaxes = T_rot @ I_body @ T_rot.T
            add_body(m, com_pelvis, I_pelvisaxes, upper)
        elif is_leg:
            all_masses.append(inertial_of(body)[0])
        # else pelvis: counted separately below.

        for child in body.findall("body"):
            recurse(child, T_pos, T_rot)

    # Walk upper-body children of pelvis (and legs, for mass accounting). Pelvis is
    # the root: its frame is identity in "pelvis frame".
    for child in pelvis.findall("body"):
        recurse(child, np.zeros(3), np.eye(3))
    all_masses.append(pelvis_m)  # pelvis own

    full_total = sum(all_masses)
    leg_total = sum(inertial_of(b)[0] for b in pelvis.iter("body")
                    if b.get("name") in leg_set)

    # --- combined = pelvis_own + lumped upper (about pelvis frame). ---
    comb = {"m": 0.0, "mr": np.zeros(3), "I_origin": np.zeros((3, 3))}
    # pelvis own:
    comb["m"] += pelvis_m
    comb["mr"] += pelvis_m * pelvis_com
    d = pelvis_com
    comb["I_origin"] += pelvis_I + pelvis_m * ((d @ d) * np.eye(3) - np.outer(d, d))
    # upper:
    comb["m"] += upper["m"]
    comb["mr"] += upper["mr"]
    comb["I_origin"] += upper["I_origin"]

    comb_com = comb["mr"] / comb["m"]
    # shift combined inertia from pelvis origin to combined CoM (parallel axis back):
    dd = comb_com
    comb_I_com = comb["I_origin"] - comb["m"] * ((dd @ dd) * np.eye(3)
                                                 - np.outer(dd, dd))
    comb_I_com = 0.5 * (comb_I_com + comb_I_com.T)  # symmetrize numerically.

    # eigendecompose -> principal diaginertia + quat (det +1).
    evals, evecs = np.linalg.eigh(comb_I_com)
    if np.linalg.det(evecs) < 0:
        evecs[:, 0] = -evecs[:, 0]  # force right-handed.
    quat = mat_to_quat(evecs)

    # upper-only CoM (for the report / FaithfulCoM lump).
    upper_com = upper["mr"] / upper["m"]

    return {
        "full_total": full_total,
        "leg_total": leg_total,
        "pelvis_own_mass": pelvis_m,
        "upper_mass": upper["m"],
        "upper_n": upper["n"],
        "upper_com_pelvis": upper_com,
        "comb_mass": comb["m"],
        "comb_com_pelvis": comb_com,
        "comb_I_com": comb_I_com,
        "diaginertia": evals,
        "quat": quat,
    }


# --------------------------------------------------------------------------- #
# MJCF emit.
def emit_mjcf(lump):
    """Build the reduced 16-DOF training MJCF from the full H1 + the exact lump."""
    src = H1_MJCF.read_text()
    tree = ET.parse(H1_MJCF)
    root = tree.getroot()
    wb = root.find("worldbody")
    pelvis = wb.find("body")

    # 1) prune everything that is NOT pelvis or a leg-chain body. The leg chain is
    #    the two hip_yaw subtrees; the upper body is the torso subtree.
    leg_roots = {"left_hip_yaw_link", "right_hip_yaw_link"}
    for child in list(pelvis.findall("body")):
        if child.get("name") not in leg_roots:
            pelvis.remove(child)

    # 2) strip ALL <geom> everywhere (mesh visuals/colliders) and the <asset> block.
    asset = root.find("asset")
    if asset is not None:
        root.remove(asset)
    for body in root.iter("body"):
        for g in list(body.findall("geom")):
            body.remove(g)
        for s in list(body.findall("site")):
            body.remove(s)

    # 3) replace the pelvis <inertial> with the EXACT lumped (pelvis_own + upper).
    ine = pelvis.find("inertial")
    com = lump["comb_com_pelvis"]
    diag = lump["diaginertia"]
    q = lump["quat"]
    ine.set("pos", f"{com[0]:.9g} {com[1]:.9g} {com[2]:.9g}")
    ine.set("quat", f"{q[0]:.9g} {q[1]:.9g} {q[2]:.9g} {q[3]:.9g}")
    ine.set("mass", f"{lump['comb_mass']:.9g}")
    ine.set("diaginertia", f"{diag[0]:.9g} {diag[1]:.9g} {diag[2]:.9g}")

    # 4) add 2 sphere foot colliders (toe+heel) per ankle (MASSLESS).
    for side in ("left", "right"):
        ankle = None
        for b in pelvis.iter("body"):
            if b.get("name") == f"{side}_ankle_link":
                ankle = b
                break
        assert ankle is not None, f"{side} ankle not found"
        for tag, xoff in (("toe", FOOT_TOE_X), ("heel", FOOT_HEEL_X)):
            g = ET.SubElement(ankle, "geom")
            g.set("name", f"{side}_{tag}")
            g.set("type", "sphere")
            g.set("size", f"{FOOT_SPHERE_R}")
            g.set("pos", f"{xoff} 0 {FOOT_BOTTOM_Z}")
            g.set("mass", "0")

    # 5) prune the <actuator> block to the 10 leg motors with real ctrlrange/gear.
    act = root.find("actuator")
    keep_joints = {f"{s}_{j}_joint" for s in ("left", "right")
                   for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")}
    for a in list(act):
        if a.get("joint") not in keep_joints:
            act.remove(a)

    # 6) drop the <sensor> block: its imu site lived on torso_link (now deleted),
    #    so the sensor refs are dangling. nuka ignores it, but removing it keeps
    #    the asset clean + loadable by other tools (e.g. MuJoCo).
    sensor = root.find("sensor")
    if sensor is not None:
        root.remove(sensor)

    # serialize.
    body_xml = ET.tostring(root, encoding="unicode")
    header = ('<?xml version="1.0"?>\n'
              "<!-- GENERATED by python/spikes/gen_h1_reduced_train.py. DO NOT "
              "EDIT BY HAND.\n"
              "     Reduced 16-DOF H1 (6 floating base + 10 leg joints) for the "
              "batched nuka.World\n"
              "     RL training world. Pelvis inertial = EXACT (pelvis_own + "
              "FK-lumped upper body\n"
              "     via parallel-axis theorem). Mass + CoM match the full "
              "h1_with_hand H1. -->\n")
    OUT_MJCF.write_text(header + body_xml + "\n")
    return OUT_MJCF


# --------------------------------------------------------------------------- #
# Whole-body CoM helpers for the verification gate.
def _quat_rotate_np(q, v):
    return quat_to_mat(q) @ v


def full_h1_com_at_pose(nominal):
    """Analytic whole-body CoM of the FULL H1 at the nominal leg pose + upper rest,
    expressed in the PELVIS frame (pelvis body origin, identity orientation).

    Independent of the lump: FK every body from the pelvis, place its <inertial>
    CoM, weight by mass. Leg joints rotate about their axis by the nominal angle;
    upper joints at 0.
    """
    tree = ET.parse(H1_MJCF)
    pelvis = tree.getroot().find("worldbody").find("body")

    joint_angle = {}
    for s in ("left", "right"):
        joint_angle[f"{s}_hip_yaw_joint"] = 0.0
        joint_angle[f"{s}_hip_roll_joint"] = 0.0
        joint_angle[f"{s}_hip_pitch_joint"] = nominal["hip_pitch"]
        joint_angle[f"{s}_knee_joint"] = nominal["knee"]
        joint_angle[f"{s}_ankle_joint"] = nominal["ankle"]

    def axis_angle_to_mat(axis, ang):
        a = np.array(axis, dtype=float)
        a = a / np.linalg.norm(a)
        c, s = math.cos(ang), math.sin(ang)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        return np.eye(3) + s * K + (1 - c) * (K @ K)

    msum = [0.0]
    mr = [np.zeros(3)]

    def recurse(body, T_pos, T_rot):
        bpos = parse_vec(body.get("pos"), [0, 0, 0])
        bquat = parse_vec(body.get("quat"), [1, 0, 0, 0])
        Tp = T_pos + T_rot @ bpos
        Tr = T_rot @ quat_to_mat(bquat)
        # apply this body's joint rotation (about its joint axis at the body origin).
        jt = body.find("joint")
        if jt is not None and jt.get("name") in joint_angle:
            axis = parse_vec(jt.get("axis"), [0, 0, 1])
            Tr = Tr @ axis_angle_to_mat(axis, joint_angle[jt.get("name")])
        ine = body.find("inertial")
        m = float(ine.get("mass"))
        com_local = parse_vec(ine.get("pos"), [0, 0, 0])
        com_pelvis = Tp + Tr @ com_local
        msum[0] += m
        mr[0] += m * com_pelvis
        for child in body.findall("body"):
            recurse(child, Tp, Tr)

    # pelvis own inertial (root frame = identity).
    ine = pelvis.find("inertial")
    m0 = float(ine.get("mass"))
    com0 = parse_vec(ine.get("pos"), [0, 0, 0])
    msum[0] += m0
    mr[0] += m0 * com0
    for child in pelvis.findall("body"):
        recurse(child, np.zeros(3), np.eye(3))
    return mr[0] / msum[0], msum[0]


# --------------------------------------------------------------------------- #
# Engine-side verification (reuse the balance gate's machinery).
def _read(w, field, ec, n):
    return torch.from_dlpack(w.buffer_view(field)).view(ec, n)


def _tilt_deg(qw, qx, qy, qz):
    gz = max(-1.0, min(1.0, -(1.0 - 2.0 * (qx * qx + qy * qy))))
    return math.degrees(math.acos(-gz))


def _leg_inertials():
    """Per-leg link-local inertial CoM offset (m) + mass (kg) in cooked link order
    (link0=pelvis filled by caller; 1-5 left leg, 6-10 right leg), read directly
    from h1_with_hand.xml so the RIGHT leg gets its true MIRROR-signed offsets
    (the reused h1_balance_gate.LEG_INERTIAL table uses LEFT values for both legs,
    which adds a ~4 mm y error -- we avoid that here)."""
    tree = ET.parse(H1_MJCF)
    pelvis = tree.getroot().find("worldbody").find("body")
    by_name = {b.get("name"): b for b in pelvis.iter("body")}
    out = []
    for link in LEG_LINKS:
        ine = by_name[link].find("inertial")
        out.append((parse_vec(ine.get("pos"), [0, 0, 0]), float(ine.get("mass"))))
    return out


def _reduced_com_from_engine(lp0, pelvis_world, lump):
    """Whole-body CoM (pelvis frame) of the reduced robot from ENGINE per-link
    world poses lp0 (blc,7) + each link's link-local inertial CoM offset. Uses the
    FULL lump CoM (incl. y) for the pelvis link and the true per-leg offsets."""
    offs = [lump["comb_com_pelvis"]]
    ms = [lump["comb_mass"]]
    for o, m in _leg_inertials():
        offs.append(np.asarray(o))
        ms.append(m)
    offs = np.asarray(offs)
    ms = np.asarray(ms)
    acc = np.zeros(3)
    for i in range(len(ms)):
        acc += ms[i] * (lp0[i, 0:3] + quat_to_mat(lp0[i, 3:7]) @ offs[i])
    return acc / ms.sum() - pelvis_world


def _reduced_com_analytic(lump, nominal):
    """Whole-body CoM (pelvis frame) of the reduced robot computed ANALYTICALLY
    from the lump + leg inertials, FK'd with the SAME convention as
    full_h1_com_at_pose. This is the exact (lump-correctness) gate -- it shares no
    code with the upper-body lump path, so it is an independent cross-check that
    the combined lump reproduces the full-H1 CoM."""
    tree = ET.parse(H1_MJCF)
    pelvis = tree.getroot().find("worldbody").find("body")
    leg_set = set(LEG_LINKS)
    ja = {}
    for s in ("left", "right"):
        ja[f"{s}_hip_yaw_joint"] = 0.0
        ja[f"{s}_hip_roll_joint"] = 0.0
        ja[f"{s}_hip_pitch_joint"] = nominal["hip_pitch"]
        ja[f"{s}_knee_joint"] = nominal["knee"]
        ja[f"{s}_ankle_joint"] = nominal["ankle"]

    def aa(axis, ang):
        a = np.asarray(axis, float)
        a /= np.linalg.norm(a)
        c, s = math.cos(ang), math.sin(ang)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        return np.eye(3) + s * K + (1 - c) * (K @ K)

    msum = [lump["comb_mass"]]
    mr = [lump["comb_mass"] * lump["comb_com_pelvis"]]

    def recurse(body, Tp, Tr):
        Tp2 = Tp + Tr @ parse_vec(body.get("pos"), [0, 0, 0])
        Tr2 = Tr @ quat_to_mat(parse_vec(body.get("quat"), [1, 0, 0, 0]))
        jt = body.find("joint")
        if jt is not None and jt.get("name") in ja:
            Tr2 = Tr2 @ aa(parse_vec(jt.get("axis"), [0, 0, 1]), ja[jt.get("name")])
        if body.get("name") in leg_set:
            ine = body.find("inertial")
            m = float(ine.get("mass"))
            cp = Tp2 + Tr2 @ parse_vec(ine.get("pos"), [0, 0, 0])
            msum[0] += m
            mr[0] += m * cp
        for ch in body.findall("body"):
            recurse(ch, Tp2, Tr2)

    for ch in pelvis.findall("body"):
        recurse(ch, np.zeros(3), np.eye(3))
    return mr[0] / msum[0]


def verify(lump):
    import h1_standing_pd_gate as pd  # reuse leg-chain FK + seat helpers.

    print("\n================ VERIFICATION ================", flush=True)
    ec = 64
    full_total = lump["full_total"]

    # full-H1 analytic CoM at the nominal pose, in the pelvis frame (the reference).
    com_full_pelvis, full_total_fk = full_h1_com_at_pose(NOMINAL)

    # --- gate (b): total mass closure (sum of emitted inertials). ---
    # The importer reads mass ONLY from <inertial mass=...> (mjcf_importer.cpp:464;
    # geom density is NOT used), so the COOKED reduced mass == the emitted-inertial
    # sum exactly. Mass closes by construction: lumped pelvis = pelvis_own +
    # (full_total - pelvis_own - Sigma legs).
    emitted_masses = [float(x.get("mass"))
                      for x in ET.parse(OUT_MJCF).getroot().iter("inertial")]
    reduced_total = sum(emitted_masses)
    mass_err = abs(reduced_total - full_total)
    print(f"(b) MASS: full H1 = {full_total:.6f} kg ; reduced emitted = "
          f"{reduced_total:.6f} kg ; err = {mass_err:.2e} kg "
          f"({len(emitted_masses)} reduced inertials)", flush=True)
    mass_pass = mass_err < 1e-3

    # --- gate (c) PRIMARY (exact): analytic reduced CoM vs full-H1 CoM. ---
    # This is the lump-correctness gate. It shares NO code with the upper-body lump
    # accumulation (a separate FK that re-uses only the SCALAR lump result), so a
    # bug in the lump cannot cancel here.
    com_red_analytic = _reduced_com_analytic(lump, NOMINAL)
    err_analytic = np.abs(com_red_analytic - com_full_pelvis)

    # --- gate (a): cook + step the reduced MJCF in the batched world. ---
    com_red_engine = None
    with nuka.Device.create(0) as dev:
        w = nuka.World.create_from_scene(dev, str(OUT_MJCF), ec, DT, 0, 1)
        try:
            blc = w.base_link_count
            cook_ok = blc <= 18
            print(f"(a) COOK: base_link_count = {blc} (<=18 DOF cap: "
                  f"{'OK' if cook_ok else 'FAIL'})", flush=True)
            qnom = pd._build_qnom(NOMINAL["hip_pitch"], NOMINAL["knee"],
                                  NOMINAL["ankle"])
            ground = PELVIS_Z0 + pd._foot_z_under_pose(0.0, 0.0, 0.0)
            base_z = ground - pd._foot_z_under_pose(
                NOMINAL["hip_pitch"], NOMINAL["knee"], NOMINAL["ankle"],
                FOOT_TOE_X, FOOT_HEEL_X)
            pd._seat_pose(w, qnom, base_z, ec, blc)
            tq = _read(w, nuka.TORQUE_INPUT, ec, blc)
            tq[:] = 0.0
            nuka.sync()
            # ONE step refreshes ARTICULATION_LINK_POSE to the seated nominal pose
            # (at the seat, link poses still hold the q=0 cook FK; FK updates on
            # step). Then read link poses + a couple more steps for the NaN check.
            w.step()
            nuka.sync()
            lp = torch.from_dlpack(
                w.buffer_view(nuka.ARTICULATION_LINK_POSE)).view(ec, blc, 7)
            bp = _read(w, nuka.BASE_POSE, ec, 7)
            com_red_engine = _reduced_com_from_engine(
                lp[0].cpu().numpy(), bp[0, 0:3].cpu().numpy(), lump)
            for _ in range(5):
                w.step()
            nuka.sync()
            bp = _read(w, nuka.BASE_POSE, ec, 7)
            steps_ok = not bool(torch.isnan(bp).any())
            print(f"(a) STEP: 6 steps, base_pose finite = {steps_ok}", flush=True)
        finally:
            w.destroy()

    err_engine = np.abs(com_red_engine - com_full_pelvis)

    # --- report gate (c). ---
    print(f"(c) CoM @ nominal pose "
          f"(legs hip_pitch{NOMINAL['hip_pitch']:+.2f} knee{NOMINAL['knee']:+.2f} "
          f"ankle{NOMINAL['ankle']:+.2f}; upper rest), pelvis frame:", flush=True)
    print(f"      full H1 (analytic FK)        = "
          f"[{com_full_pelvis[0]:+.6f} {com_full_pelvis[1]:+.6f} "
          f"{com_full_pelvis[2]:+.6f}] m", flush=True)
    print(f"      reduced (analytic lump+legs) = "
          f"[{com_red_analytic[0]:+.6f} {com_red_analytic[1]:+.6f} "
          f"{com_red_analytic[2]:+.6f}] m   |err| max "
          f"{err_analytic.max() * 1e6:.3f} um  <-- EXACT lump gate", flush=True)
    print(f"      reduced (ENGINE link-pose FK)= "
          f"[{com_red_engine[0]:+.6f} {com_red_engine[1]:+.6f} "
          f"{com_red_engine[2]:+.6f}] m   |err| "
          f"[{err_engine[0] * 1000:.3f} {err_engine[1] * 1000:.3f} "
          f"{err_engine[2] * 1000:.3f}] mm  <-- cook-wiring gate", flush=True)
    # The exact gate proves the lump; the engine gate proves the cook wired it.
    # Engine x/z are the load-bearing balance axes; allow a few mm for the 1-step
    # FK-refresh dynamics drift. (y carries a tiny residual from finite settle.)
    com_pass = err_analytic.max() < 5e-5 and err_engine.max() < 0.006

    print(f"\n  full-H1 total (analytic FK recount) = {full_total_fk:.6f} kg "
          f"(matches accounting: {abs(full_total_fk - full_total) < 1e-6})",
          flush=True)
    print(f"\n  GATE (b) mass match: {'PASS' if mass_pass else 'FAIL'}", flush=True)
    print(f"  GATE (c) CoM match : {'PASS' if com_pass else 'FAIL'}", flush=True)
    return mass_pass and com_pass


# --------------------------------------------------------------------------- #
def main():
    print("nuka:", nuka.__file__, flush=True)
    assert H1_MJCF.exists(), f"source MJCF missing: {H1_MJCF}"

    lump = fk_and_lump()

    # --- report the exact lump + cross-checks. ---
    print("\n================ EXACT LUMP ================", flush=True)
    print(f"  full H1 total mass        = {lump['full_total']:.6f} kg", flush=True)
    print(f"  pelvis own mass           = {lump['pelvis_own_mass']:.6f} kg",
          flush=True)
    print(f"  Sigma 10 leg masses       = {lump['leg_total']:.6f} kg", flush=True)
    print(f"  FK'd upper-body mass      = {lump['upper_mass']:.6f} kg "
          f"({lump['upper_n']} bodies)", flush=True)
    expect_upper = lump["full_total"] - lump["pelvis_own_mass"] - lump["leg_total"]
    print(f"    cross-check upper mass  = full - pelvis_own - legs = "
          f"{expect_upper:.6f} kg  (FK err {abs(expect_upper - lump['upper_mass']):.2e})",
          flush=True)
    print(f"  upper-only CoM (pelvis fr)= "
          f"[{lump['upper_com_pelvis'][0]:+.5f} {lump['upper_com_pelvis'][1]:+.5f} "
          f"{lump['upper_com_pelvis'][2]:+.5f}] m", flush=True)
    print(f"\n  COMBINED lumped pelvis link (pelvis_own + upper):", flush=True)
    print(f"    mass     = {lump['comb_mass']:.6f} kg", flush=True)
    c = lump["comb_com_pelvis"]
    print(f"    CoM      = [{c[0]:+.6f} {c[1]:+.6f} {c[2]:+.6f}] m  (pelvis frame)",
          flush=True)
    print(f"    inertia about combined CoM (pelvis-frame axes, kg*m^2):", flush=True)
    I = lump["comb_I_com"]
    for r in range(3):
        print(f"      [{I[r,0]:+.6f} {I[r,1]:+.6f} {I[r,2]:+.6f}]", flush=True)
    d = lump["diaginertia"]
    q = lump["quat"]
    print(f"    -> principal diaginertia = [{d[0]:.6f} {d[1]:.6f} {d[2]:.6f}] "
          f"kg*m^2", flush=True)
    print(f"    -> principal-axis quat   = [{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} "
          f"{q[3]:.6f}] (w x y z)", flush=True)

    emit_mjcf(lump)
    print(f"\n  emitted reduced MJCF -> {OUT_MJCF}", flush=True)

    ok = verify(lump)
    print(f"\n================ {'ALL GATES PASS' if ok else 'GATE FAILURE'} "
          "================", flush=True)
    return ok


if __name__ == "__main__":
    main()
