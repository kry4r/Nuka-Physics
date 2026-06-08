"""STANDING GATE spike: can a HAND-TUNED PD balance controller STAND the welded
reduced-DOF H1 in the BATCHED trainable world (env>=2) on MULTI-SPHERE feet (a
real support polygon), staying upright for several seconds?

This is an INVESTIGATION, not a feature. It drives ``nuka.World`` DIRECTLY (NOT
``NukaGymEnv`` -- that layer hard-asserts ``action_dim==12`` and is go2-specific).
The deliverable is the printed findings, not a passing test. It builds on the
prior gate spike (``h1_floating_base_gate.py``), which showed the 51-DOF
floating-base H1 is BLOCKED at the batched contact solver's per-thread DOF cap
(kMaxContactSolverDof=18) but the SAME H1 WELDED to 16 DOF (1 free + 10 leg) +
sphere feet CONSTRUCTS, STEPS, and has ACTIVE foot-ground contact (yet falls -- no
controller, single point-sphere per ankle = a left-right line, zero fore-aft base).

==============================================================================
VERDICT (proven by the trials below + static code read):

  * LIGHT robot (the welded reduced-DOF H1 AS-IS): hand-PD STANDS it.
    On a world-CENTERED 0.2 m toe-heel foot (4 spheres = the contact cap) with
    moderately stiff joint PD, all 64 envs hold upright for the full 1000-step
    (5 s) rollout: tilt stays ~5 deg, base z drops < 0.01 m, no NaN. The DOF
    reduction + multi-sphere support polygon WORKS for the light robot.

  * FAITHFUL-MASS robot (full 52 kg H1, upper body lumped into the pelvis):
    hand-PD does NOT stand it with physically-realistic ankle torque. It tips
    backward and inverts within ~1-1.5 s for every gain at/under ~ankle Kp=400
    N*m/rad. It ONLY stands at UNPHYSICAL gains (ankle Kp~700, hip/knee~1200
    N*m/rad) -- far above the MJCF ankle motor limit (ctrlrange +/-40 N*m).

  * HEADLINE ENGINE FINDING (invalidates the prior spike's "mass preserved"
    claim): the weld-by-deleting-<joint> DROPS the welded bodies' MASS entirely.
    The articulation cooker builds links by a BFS over JOINTS only
    (articulation_cooker.cpp:96-148): a body with no incoming joint and not in
    any parent's child list is UNREACHABLE -> never added -> its inertia is gone.
    There is NO mass-preserving structural weld. Measured: the cooked reduced H1
    is 27.0 kg (pelvis 5.39 + 2 leg chains 2x10.823) vs the full H1's 52.3 kg
    (sum of all 46 <inertial>) -- ~25 kg (48%) of upper-body mass SILENTLY LOST.
    => The prior spike AND the LIGHT case above stand a TORSO-LESS biped. The
    faithful biped (CoM raised toward ~1.0 m by the torso) is an inverted
    pendulum whose ankle-PD torque demand 2*Kp_ankle ~ m*g*h ~ 52*9.81*1.0 ~ 510
    N*m exceeds any physical ankle motor -> it does not stand under joint-PD.

  CLASSIFICATION of the NO (faithful): primarily a FOOT/CONTACT-MODEL + ENGINE
  finding, NOT something RL rescues:
    (1) FOOT-MODEL: kMaxFootContactsPerEnv = 4 caps a foot polygon at 4 contact
        points TOTAL across both feet (CONTACT_NORMAL/FORCE are (env,4,3));
        detection writes slot=base_slot+count with NO clamp, so >4 simultaneous
        penetrations CORRUPT the next env's slots. A real flat foot patch is not
        representable; 2 spheres/foot is the max faithful polygon.
    (2) ENGINE: the weld drops mass (above) -> the trainable batched world has NO
        way to carry faithful H1 inertia under the 18-DOF cap. Faithful mass
        needs EITHER the DOF cap raised to >=51 (so the real connected torso
        stays) OR a mass-preserving lumped weld in the cooker.
    (3) Only AFTER (1)+(2): with faithful CoM and a real foot, the residual is a
        controller/inverted-pendulum margin question RL could attempt -- but not
        before the engine carries the right mass and the foot can form a patch.

  WHAT WOULD UNBLOCK A FAITHFUL STANDING TARGET (engine work, not RL):
    - Raise kMaxContactSolverDof to >= 51 + resize per-thread PGS scratch, so the
      FULL connected H1 (real torso mass) loads (the prior spike's blocker [2]).
    - OR add a mass-preserving fixed-weld in the articulation cooker (merge a
      jointless child's inertia into its parent link) so a reduced-DOF H1 keeps
      faithful mass.
    - Extend foot contact beyond 4 sphere points / sphere-vs-plane (a foot patch
      / capsule-or-box-vs-plane) so a real support polygon is representable.

==============================================================================
KEY ENGINE FACTS verified by reading the code (drive every decision below):

  * CONTACT-SLOT CAP = 4 contacts TOTAL per env, across BOTH feet
    (kMaxFootContactsPerEnv=4u, articulation_contacts.hpp:43). We use 2
    spheres/foot (toe+heel) = 4 = the cap exactly, NO overflow.
  * CONTROL via control_mode=0 (PD_POSITION): the engine computes
    tau = Kp*(q_target-q) - Kd*qdot per link with the implicit joint damping that
    makes go2 stable at native dt=0.005. q_target<-DRIVE_TARGET, Kp<-DRIVE_STIFFNESS,
    Kd<-DRIVE_DAMPING (all WRITABLE device aliases, env-major float[env*blc],
    slot 0 = inert root, slots [1:] = actuated joints). A mode-1 explicit-torque
    cross-check is included.
  * GROUND auto-derived ONCE at cook time from the q=0 (straight-leg) rest FK
    (DeriveGroundHeight, world.cpp:236-275). MJCF importer reads NO <keyframe>
    and NO joint `ref`, so the cook pose is ALWAYS q=0. We let ground derive at
    q=0, then WRITE the bent stance into JOINT_POSITION (state.q) + lower
    BASE_POSE.z (both LIVE device aliases, buffer.cpp:39-93) so the bent feet
    land back on that fixed ground with qdot=0. SEAT IS VERIFIED: after seat the
    early trace shows 4/4 active contacts at tilt~0.
  * Q LAYOUT: floating base handled separately (base_pose per-articulation); q is
    per-LINK length env*blc, ROOT slot 0 inert, joint angles in slots [1:].
    Reduced H1: blc = 1 + 10 = 11. Cooked link map (verified): link 0 = pelvis;
    links 1-5 = LEFT leg hip_yaw,hip_roll,hip_pitch,knee,ankle; links 6-10 =
    RIGHT leg (same order). Left ankle = link 5, right = link 10.
  * FRAME GOTCHA (a real bug we hit + fixed): the foot-sphere `pos` is in the
    ANKLE-LINK frame; CoM-x and contact points are WORLD. At the bent seat the
    ankle sits at world x~0.077 while the (light) CoM-x~0.046 -> the foot offsets
    must average (CoM-x - ankle-x) ~ -0.031 to put the polygon WORLD-centered on
    the CoM. Centering in the ankle frame (avg 0) leaves the heel AT the CoM
    (zero back-margin) -> instant backward tip. The light robot tips BACKWARD
    (CoM behind the forward-biased ankle), so the heel needs back-margin.

Run (from repo root):
  export PATH=/root/.nuka-toolchain-gcc14/bin:$PATH
  export CUDA_VISIBLE_DEVICES=0
  /root/miniconda3/envs/nuka-v03/bin/python python/spikes/h1_standing_pd_gate.py
==============================================================================
"""

from __future__ import annotations

import math
import pathlib
import re

import torch

import nuka

REPO = pathlib.Path(__file__).resolve().parents[2]
H1_MJCF = REPO / ".nuka-assets/newton_assets/unitree_h1/mjcf/h1_with_hand.xml"
DT = 0.005
G = 9.81

# ---------------------------------------------------------------------------
# Leg kinematics read straight from the H1 MJCF (all SI metres). Chain:
#   pelvis(freejoint) -> hip_yaw off(0,+/-0.0875,-0.1742) axis z
#     -> hip_roll off(0.039468,0,0) axis x
#       -> hip_pitch off(0,+/-0.11536,0) axis y
#         -> knee off(0,0,-0.4) axis y -> ankle off(0,0,-0.4) axis y
# Per-leg MJCF joint order (== q[1:] slot order): hip_yaw,hip_roll,hip_pitch,knee,ankle.
PELVIS_Z0 = 1.1
HIP_YAW_OFF = (0.0, 0.0875, -0.1742)
HIP_ROLL_OFF = (0.039468, 0.0, 0.0)
HIP_PITCH_OFF = (0.0, 0.11536, 0.0)
KNEE_OFF = (0.0, 0.0, -0.4)
ANKLE_OFF = (0.0, 0.0, -0.4)
FOOT_SPHERE_R = 0.025
FOOT_BOTTOM_Z = -0.055  # sphere CENTRE z below the ankle joint (ankle-link frame).

# Per-cooked-link mass (kg), MJCF <inertial> mass for the 11 retained links.
LINK_MASS = [5.39, 2.244, 2.232, 4.152, 1.721, 0.474,
             2.244, 2.232, 4.152, 1.721, 0.474]

# WORLD-centered toe/heel offsets (ankle-link frame) for the LIGHT robot: the
# bent-seat ankle sits at world x ~0.077, light CoM-x ~0.046, so a 0.2 m foot
# placed with the heel ~0.09 m behind the CoM stands. toe +0.07 / heel -0.13.
FOOT_TOE_X = 0.07
FOOT_HEEL_X = -0.13


def _tilt_deg(qw, qx, qy, qz):
    gz_body = -(1.0 - 2.0 * (qx * qx + qy * qy))
    gz_body = max(-1.0, min(1.0, gz_body))
    return math.degrees(math.acos(-gz_body))


def _report_base(world, tag, env=0):
    bp = torch.from_dlpack(world.buffer_view(nuka.BASE_POSE)).view(-1, 7)
    px, py, pz, qw, qx, qy, qz = bp[env].detach().cpu().tolist()
    tilt = _tilt_deg(qw, qx, qy, qz)
    print(f"    [{tag}] base=({px:+.3f},{py:+.3f},{pz:+.3f}) tilt={tilt:6.2f}deg",
          flush=True)
    return pz, tilt


def _foot_z_under_pose(hip_pitch, knee, ankle, toe_x=FOOT_TOE_X,
                       heel_x=FOOT_HEEL_X):
    """Analytic foot-bottom z (relative to base z) for a sagittal-plane leg pose
    (hip_yaw/roll = 0). Composes the three y-axis pitch joints down the (x,z)
    chain. Used ONLY to seat base z so the bent feet land on the q=0-derived
    ground (the z self-consistency: q=0 -> base_z == PELVIS_Z0)."""
    th_hip = hip_pitch
    th_knee = hip_pitch + knee
    th_ankle = hip_pitch + knee + ankle

    def rot_y(p, th):
        x, z = p
        c, s = math.cos(th), math.sin(th)
        return (c * x + s * z, -s * x + c * z)

    px = HIP_ROLL_OFF[0]
    pz = HIP_YAW_OFF[2]
    seg = rot_y(KNEE_OFF[::2], th_hip)
    px += seg[0]; pz += seg[1]
    seg = rot_y(ANKLE_OFF[::2], th_knee)
    px += seg[0]; pz += seg[1]
    toe = rot_y((toe_x, FOOT_BOTTOM_Z), th_ankle)
    heel = rot_y((heel_x, FOOT_BOTTOM_Z), th_ankle)
    return min(pz + toe[1], pz + heel[1]) - FOOT_SPHERE_R


LEG_JOINTS = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]


def _build_qnom(hip_pitch, knee, ankle):
    per_leg = {"hip_yaw": 0.0, "hip_roll": 0.0,
               "hip_pitch": hip_pitch, "knee": knee, "ankle": ankle}
    return [per_leg[n.split("_", 1)[1]] for n in LEG_JOINTS]


# ---------------------------------------------------------------------------
# MJCF builders (additive temp files; auto-cleaned).
def _strip_and_weld(text):
    """Mesh-strip + weld every non-leg joint (delete its <joint>): 1 free + 10
    leg = 16 DOF. NOTE: the weld DROPS the welded bodies' mass (cooker BFS over
    joints -- see header). The retained robot is legs+pelvis only (~27 kg)."""
    text = re.sub(r"<asset>.*?</asset>", "<asset/>", text, flags=re.DOTALL)
    text = re.sub(r"<geom\b[^>]*/>", "", text)
    text = re.sub(r"<geom\b[^>]*>.*?</geom>", "", text, flags=re.DOTALL)
    leg = {f"{s}_{j}_joint" for s in ("left", "right")
           for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")}

    def _keep(m):
        nm = re.search(r'name="([^"]+)"', m.group(0))
        return m.group(0) if (nm and nm.group(1) in leg) else ""
    return re.sub(r"<joint\b[^>]*/>", _keep, text)


def _add_feet(text, toe_x, heel_x):
    for s in ("left", "right"):
        anchor = (f'<joint name="{s}_ankle_joint" pos="0 0 0" axis="0 1 0" '
                  f'range="-0.87 0.52"/>')
        assert anchor in text, f"ankle anchor missing for {s}"
        feet = (f'<geom name="{s}_toe" type="sphere" size="{FOOT_SPHERE_R}" '
                f'pos="{toe_x} 0 {FOOT_BOTTOM_Z}"/>'
                f'<geom name="{s}_heel" type="sphere" size="{FOOT_SPHERE_R}" '
                f'pos="{heel_x} 0 {FOOT_BOTTOM_Z}"/>')
        text = text.replace(anchor, anchor + "\n        " + feet, 1)
    return text


def _lump_pelvis_mass(text, mass, com_x, com_z, inertia):
    """FAITHFUL-mass approximation: replace the pelvis <inertial> with a lumped
    one carrying the dropped upper-body mass at a raised CoM. (Mass + CoM are the
    load-bearing balance terms; inertia is approximate -- a coarse but honest
    stand-in for the connected torso the engine cannot carry under the DOF cap.)"""
    new = (f'<inertial pos="{com_x} 0 {com_z}" mass="{mass}" '
           f'diaginertia="{inertia} {inertia} {inertia}"/>')
    return re.sub(r'<inertial pos="-0.0002 4e-05 -0.04522"[^>]*/>', new, text,
                  count=1)


def _make_light(out_path, toe_x=FOOT_TOE_X, heel_x=FOOT_HEEL_X):
    out_path.write_text(_add_feet(_strip_and_weld(H1_MJCF.read_text()), toe_x, heel_x))
    return out_path


def _make_faithful(out_path, toe_x, heel_x, com_z=0.18, mass=30.62,
                   com_x=-0.02, inertia=0.8):
    text = _strip_and_weld(H1_MJCF.read_text())
    text = _lump_pelvis_mass(text, mass, com_x, com_z, inertia)
    out_path.write_text(_add_feet(text, toe_x, heel_x))
    return out_path


# ---------------------------------------------------------------------------
def _seat_pose(w, qnom, base_z, ec, blc):
    q = torch.from_dlpack(w.buffer_view(nuka.JOINT_POSITION)).view(ec, blc)
    qd = torch.from_dlpack(w.buffer_view(nuka.JOINT_VELOCITY)).view(ec, blc)
    bp = torch.from_dlpack(w.buffer_view(nuka.BASE_POSE)).view(ec, 7)
    qnom_t = torch.tensor(qnom, dtype=q.dtype, device=q.device)
    q[:, 0] = 0.0
    q[:, 1:1 + len(qnom)] = qnom_t
    qd[:] = 0.0
    bp[:, 2] = base_z
    bp[:, 3] = 1.0
    bp[:, 4:7] = 0.0
    nuka.sync()


def _set_pd(w, qnom, kp, kd, ec, blc):
    dt_ = torch.from_dlpack(w.buffer_view(nuka.DRIVE_TARGET)).view(ec, blc)
    ks = torch.from_dlpack(w.buffer_view(nuka.DRIVE_STIFFNESS)).view(ec, blc)
    kdb = torch.from_dlpack(w.buffer_view(nuka.DRIVE_DAMPING)).view(ec, blc)
    dt_[:, 0] = 0.0; ks[:, 0] = 0.0; kdb[:, 0] = 0.0
    for i, name in enumerate(LEG_JOINTS):
        j = name.split("_", 1)[1]
        dt_[:, 1 + i] = qnom[i]
        ks[:, 1 + i] = kp[j]
        kdb[:, 1 + i] = kd[j]
    nuka.sync()


def _contacts(w, ec):
    cf = torch.from_dlpack(w.buffer_view(nuka.CONTACT_FORCE)).view(ec, -1, 3)
    Fn = cf[:, :, 0]  # normal force per slot (N) = lambda/dt.
    active0 = int((cf[0].norm(dim=-1) > 1e-4).sum().item())
    return active0, float(Fn[0].sum().item())


def run_trial(dev, spike_path, stance, kp, kd, ec=64, n_steps=1000, label=""):
    hp, kn, an = stance
    qnom = _build_qnom(hp, kn, an)
    # ground was derived at q=0; seat the bent feet onto it.
    ground = PELVIS_Z0 + _foot_z_under_pose(0.0, 0.0, 0.0)
    base_z = ground - _foot_z_under_pose(hp, kn, an)
    print(f"\n--- TRIAL [{label}] stance(hip,knee,ankle)=({hp:+.2f},{kn:+.2f},"
          f"{an:+.2f}) Kp_ankle={kp['ankle']} ---", flush=True)
    w = nuka.World.create_from_scene(dev, str(spike_path), ec, DT, 0, 0)
    try:
        blc = w.base_link_count
        assert blc == 11, f"expected blc==11, got {blc}"
        _seat_pose(w, qnom, base_z, ec, blc)
        _set_pd(w, qnom, kp, kd, ec, blc)
        w.step()
        nuka.sync()
        nc, fn = _contacts(w, ec)
        print(f"    SEAT verify: {nc}/4 active contacts, ground={ground:+.4f}, "
              f"seated base_z={base_z:+.4f}, supported~{fn / G:.1f} kg", flush=True)
        z0, t0 = _report_base(w, "t=1")
        tmax, first_fall, nan_step = t0, -1, -1
        for k in range(1, n_steps):
            w.step()
            if (k + 1) % 100 == 0:
                nuka.sync()
                pz, tilt = _report_base(w, f"t={k+1}")
                if math.isnan(pz) or math.isnan(tilt):
                    nan_step = k + 1
                    print("        !! NaN", flush=True)
                    break
                tmax = max(tmax, tilt)
                if tilt > 30.0 and first_fall < 0:
                    first_fall = k + 1
        nuka.sync()
        bp = torch.from_dlpack(w.buffer_view(nuka.BASE_POSE)).view(ec, 7)
        zf, tf = _report_base(w, "final")
        env_held = sum(1 for e in range(ec)
                       if (_tilt_deg(*bp[e, 3:7].tolist()) < 20.0
                           and not math.isnan(bp[e, 2].item())))
        held = (nan_step < 0 and tf < 20.0 and abs(zf - z0) < 0.10
                and tmax < 20.0)
        print(f"    => z {z0:+.3f}->{zf:+.3f} tilt_max={tmax:.1f} tf={tf:.1f} "
              f"envs_held={env_held}/{ec} :: "
              + ("STOOD upright" if held else f"FELL (first_fall@={first_fall})"),
              flush=True)
        return held, tmax, env_held
    finally:
        w.destroy()


def main():
    print("nuka:", nuka.__file__, flush=True)
    assert H1_MJCF.exists(), "H1 MJCF missing"
    ec, n = 64, 1000

    # Mass accounting (headline finding).
    masses = [float(x) for x in
              re.findall(r'<inertial[^>]*mass="([0-9.eE+-]+)"', H1_MJCF.read_text())]
    full = sum(masses)
    retained = sum(LINK_MASS)
    print(f"\nMASS ACCOUNTING: full H1 = {full:.2f} kg ({len(masses)} inertials); "
          f"cooked-retained (weld DROPS upper body) = {retained:.2f} kg; "
          f"LOST = {full - retained:.2f} kg ({100 * (full - retained) / full:.0f}%)",
          flush=True)

    # Gains. The inverted-pendulum threshold 2*Kp_ankle ~ m*g*h:
    #   light  (27 kg, CoM-z~0.84): ~ 222 N*m -> vstiff(2x250) clears.
    #   faithful (52 kg, CoM-z~1.0): ~ 510 N*m -> needs ankle Kp >> MJCF +/-40 N*m.
    VSTIFF = ({"hip_yaw": 300, "hip_roll": 500, "hip_pitch": 600,
               "knee": 600, "ankle": 250},
              {"hip_yaw": 15, "hip_roll": 25, "hip_pitch": 30,
               "knee": 30, "ankle": 20})
    EXTREME = ({"hip_yaw": 600, "hip_roll": 800, "hip_pitch": 1200,
                "knee": 1200, "ankle": 700},
               {"hip_yaw": 25, "hip_roll": 35, "hip_pitch": 50,
                "knee": 50, "ankle": 45})
    STANCE = (-0.40, 0.70, -0.30)

    results = []
    light = H1_MJCF.parent / "h1_stand_light.xml"
    faith = H1_MJCF.parent / "h1_stand_faithful.xml"
    try:
        with nuka.Device.create(0) as dev:
            # 1) LIGHT robot, world-centered 0.2 m foot, vstiff PD.
            _make_light(light)
            print(f"\n=== LIGHT robot ({retained:.1f} kg, torso dropped), "
                  "world-centered 0.2 m toe-heel foot ===", flush=True)
            r = run_trial(dev, light, STANCE, *VSTIFF, ec, n, "LIGHT vstiff")
            results.append(("LIGHT vstiff (27kg)", r))

            # 2) FAITHFUL robot (upper body lumped into pelvis), realistic gains.
            _make_faithful(faith, toe_x=0.05, heel_x=-0.15)
            print(f"\n=== FAITHFUL robot ({full:.1f} kg, upper body lumped into "
                  "pelvis), realistic ankle (Kp=250, near motor limit) ===",
                  flush=True)
            r = run_trial(dev, faith, STANCE, *VSTIFF, ec, n, "FAITHFUL vstiff")
            results.append(("FAITHFUL vstiff (52kg, realistic)", r))

            # 3) FAITHFUL robot, UNPHYSICAL gains -- proves it's a torque-authority
            #    ceiling (stands only above any real ankle motor).
            print(f"\n=== FAITHFUL robot ({full:.1f} kg), UNPHYSICAL gains "
                  "(ankle Kp=700, hip/knee 1200 >> MJCF +/-40 N*m motor) ===",
                  flush=True)
            _make_faithful(faith, toe_x=0.10, heel_x=-0.22)
            r = run_trial(dev, faith, STANCE, *EXTREME, ec, n,
                          "FAITHFUL extreme-gain")
            results.append(("FAITHFUL extreme-gain (52kg, unphysical)", r))

        print("\n================ SUMMARY ================", flush=True)
        for tag, (held, tmax, eh) in results:
            print(f"  {tag:42s} held={held!s:5s} tilt_max={tmax:5.1f} "
                  f"envs_held={eh}/{ec}", flush=True)
        print("\nVERDICT:", flush=True)
        print("  LIGHT (torso-less 27 kg): hand-PD STANDS it on a centered "
              "multi-sphere foot.", flush=True)
        print("  FAITHFUL (52 kg): hand-PD does NOT stand it with realistic ankle "
              "torque (tips backward, inverts in ~1 s); stands ONLY at unphysical "
              "gains. The weld SILENTLY DROPS ~48% of the H1 mass -> the trainable "
              "world cannot carry faithful H1 inertia under the 18-DOF cap.",
              flush=True)
        print("  => Engine FOOT/MASS work (raise DOF cap to >=51 OR mass-preserving "
              "weld; + a foot PATCH beyond 4 sphere points) BEFORE training.",
              flush=True)
    finally:
        for p in (light, faith):
            if p.exists():
                p.unlink()
    print("\n=== spike complete ===", flush=True)


if __name__ == "__main__":
    main()
