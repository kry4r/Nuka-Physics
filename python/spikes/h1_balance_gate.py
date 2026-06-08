"""BALANCE GATE spike: can the FAITHFUL-mass (52 kg) reduced-DOF H1 STAND in the
batched trainable world (env>=2) under a COMPETENT humanoid balance controller,
using ONLY physical leg torque (clamped to the H1 MJCF actuator limits), on the
4-corner toe/heel foot support polygon?

This is the PIVOTAL DECISION GATE. The prior spike (``h1_standing_pd_gate.py``)
showed pure joint-PD-to-a-fixed-pose does NOT stand the faithful robot at physical
ankle torque -- but that is NOT a balance controller. THIS spike replaces the
ankle's fixed-setpoint PD with a real textbook ANKLE STRATEGY (CoM/CoP feedback)
while keeping knee/hip on posture PD (where steady-state PD error legitimately
supplies the holding torque, INCLUDING the ground-reaction term), and re-asks the
gate honestly. It REUSES the prior spike's machinery (faithful-mass lump, weld,
foot spheres, seat); see that file for the verified engine facts.

==============================================================================
VERDICT (PASS) -- overturns the prior spike's "faithful does NOT stand":

  The FAITHFUL 52.27 kg reduced-DOF H1 STANDS for the full 1000-step (5 s)
  rollout in all 64 envs under the competent balance controller, with ALL
  commanded torques WITHIN the MJCF physical limits. Best run (Kp_cop=320,
  Kd_cop=50, settle=15 steps) -- ALL metrics tracked at EVERY step (not sampled):
    * envs_up = 64/64, tilt_max = 11.5 deg over the FULL rollout (final 2.5 deg,
      bounded < 15 deg the whole 1000 steps).
    * CoM converges to the polygon CENTRE: com_ex -0.065 -> -0.004 m by t=800,
      min CoM-to-sagittal-edge margin +0.025 m POSITIVE the whole rollout (CoM
      never leaves the +/-0.10 m polygon).
    * min_contacts drops to as low as 2/4 during the CoP-engagement transient
      (~t=15-20), recovering to STEADY 4/4 at 52.3 kg (== whole-body mass, the
      seat is faithful). The dip is transient (6/1000 steps with <4 contacts) and
      the CoM never leaves the polygon (edge margin stays positive throughout) --
      a recoverable correction, not a sustained collapse.
    * PEAK |tau| (unclamped, vs MJCF limit): ankle 30.7/40 (23% headroom, ZERO
      clamp events), knee 90/300, hip_pitch 55/200, hip_roll 2.4/200, hip_yaw
      5.9/200. Nothing saturates -> stands strictly WITHIN physical torque.

  CAVEATS (honest scope):
    - The 64 envs are DETERMINISTIC REPLICAS of one seat (no randomization), so
      envs_up=64/64 is env0 x64, NOT 64 independent trials. The breadth claim is
      single-condition. (env>=2 is satisfied only because the ground exists only
      for env_count>1.)
    - The within-torque GAIN WINDOW is NARROW: of the swept CoP gains, only
      Kp_cop=320 is a clean within-torque stand. Kp_cop=200 over-torques (peak
      ankle 182/40, FELL); Kp_cop=255 NEARLY FALLS -- every-step tracking shows
      its CoM crosses the polygon edge (min_edge -0.005 m, min_contacts 1/4, 194
      sphere-unload steps) before recovering into a squat (NOT a clean second
      pass). So the PASS rests on ONE clean gain, not a wide margin.
    - What IS robust is the AUTHORITY ARITHMETIC, not the gain margin: steady
      balance needs only ~1-3 N*m ankle and CoP-to-edge costs ~25 N*m, both
      < 40 -> a within-torque stand EXISTS on this foot. Finding it needs a
      tuned gain; that it exists is guaranteed by the margins.

  WHY the prior spike got a FALSE NO: it used fixed-setpoint joint-PD on the
  ANKLE, which makes restoring torque only at large joint error -> needs ~700
  N*m ankle gains (>> +/-40). That is not a balance controller. The textbook
  ankle strategy (regulate the CoP under the CoM) holds the SAME robot at < 35
  N*m peak ankle.

  MOTOR-WALL vs GEOMETRY-WALL (the gate's discriminator): NEITHER binds at the
  passing gain. The ankle motor is NOT the wall (peak 30.7 < 40, no clamp); the
  foot geometry is NOT the wall (CoM settles at the polygon centre, 4/4 contacts
  held, never pinned to an edge). The authority arithmetic predicted this: CoP-
  to-edge costs ~25 N*m < 40, and steady balance needs only ~1-3 N*m. The
  4-contact / 0.2 m foot HAS enough CoP authority for a 52 kg, ~1 m-CoM stand.
  (Over-driving Kp_cop to 500-570 DOES oscillate the ankle to clamp+NaN, but
  that is controller over-drive, not a foot/motor ceiling -- the natural gain
  Kp_cop ~ m*g/2 ~ 255-320 is stable and within torque.)

  ENGINE DECISION: faithful standing is REACHABLE in the batched world today
  with only a competent controller. The engine work needed is the CHEAP mass-
  preserving lumped weld in the articulation cooker (merge a jointless welded
  child's inertia into its parent link) so the reduced-DOF H1 carries the real
  52 kg mass under the 18-DOF cap -- NOT a foot-contact-model overhaul. The
  4-contact toe/heel foot polygon is sufficient for standing.

  STAGED EVIDENCE (each a cheap run; see run output):
    STAGE1 baseline (posture PD, ankle=0, no CoP): seat is HEALTHY -- 4/4
      contacts from t=1, supported 52 kg; with zero ankle authority it tips
      slowly BACKWARD (com_ex -0.028 -> -0.76, tilt 60 deg by t=200). Proves the
      support polygon + seat are real and the residual is a balance (ankle) job.
    STAGE2 sign calibration (+/-5 N*m fixed ankle): -5 N*m SLOWS the fall and
      holds 64/64 upright (within torque); +5 N*m makes it worse -> the
      stabilizing ankle sign is NEGATIVE (resist the backward tip). The CoP law
      is therefore tau = +Kp_cop*ex + Kd_cop*evx (both terms share the sign).
    STAGE3 CoP engaged (settle 15, then +Kp_cop*ex+Kd_cop*evx): the within-torque
      window is NARROW and Kcop320 is the sweet spot. Kcop200 under-damps the
      engagement transient -> over-torques (ankle 182/40), FELL. Kcop255 nearly
      falls (CoM crosses the polygon edge, min_ct 1/4) then recovers to a squat,
      within torque but not a clean stand. Kcop320 STANDS cleanly WITHIN torque
      (ankle peak 30.7/40, no clamp) -- the PASS. Kcop350 holds upright but its
      ankle peaks 45/40 (EXCEEDS the limit, 2 clamp events) -> a stand but not a
      within-torque one. So 320 is bracketed: <320 over-torques on the transient,
      >~350 over-torques at the ankle. Engaging CoP at step 0 (no settle) FAILS --
      the seat-impact com_v transient kicks +40 N*m into both ankles and lifts a
      sphere per foot; a short settle past the t=2-3 impact fixes it.
==============================================================================
CONTROLLER (torque mode = CONTROL_MODE_TORQUE; tau computed in python, clamped):
  * KNEE + HIP (yaw/roll/pitch): POSTURE PD at PHYSICAL gains
      tau = Kp*(q_nom - q) - Kd*qdot,  clamped to MJCF ctrlrange (hip +/-200,
      knee +/-300). The steady-state PD error automatically supplies the holding
      torque incl. the contact (ground-reaction) term -- the prior spike already
      proved posture PD holds these joints. (Analytic subtree gravity-comp would
      UNDER-supply the knee ~20x because it ignores the ~255 N/foot reaction
      transmitted up the bent leg, so we do NOT use it.)
  * ANKLE pitch: textbook ANKLE STRATEGY -- CoP-under-CoM feedback, NOT fixed PD:
      tau_ankle = +Kp_cop*(com_x - poly_cx) + Kd_cop*com_vx  (sign EMPIRICALLY
      calibrated in STAGE2: the body tips backward, ex<0, needs negative ankle
      torque -> both terms share the +sign; Kp_cop ~ m*g/2 ~ 255-320 is natural)
      (feedforward ~0 at balance; this is the ONE dof that needs unphysical gains
      under fixed-setpoint PD). Lateral handled by hip_roll posture PD + wide
      stance. Clamped to ankle +/-40.
  * EXPLICIT DAMPING: torque mode drops the implicit damping that keeps dt=0.005
      stable, so all joints carry -Kd*qdot (ankle Kd kept small to preserve +/-40).

WHOLE-BODY CoM: computed at runtime from ARTICULATION_LINK_POSE (per-link world
pose) + each link's MJCF <inertial> offset rotated into world; the lumped pelvis
carries the faithful upper-body mass at its lumped offset. CoM velocity by finite
difference across steps.

AUTHORITY ARITHMETIC (the verdict discriminator):
  Per-foot vertical load ~ m*g/2 ~ 255 N. CoP to the foot edge (+/-0.1 m on a
  0.2 m foot) costs ~25 N*m per ankle, UNDER the +/-40 N*m ankle limit. So the
  ankle MOTOR is NOT the expected wall; the foot GEOMETRY is (recoverable tilt ~
  atan(0.1/0.85) ~ 6.7 deg). We log per-ankle |tau| vs 40 AND whether a foot
  sphere unloads (per-slot CONTACT_FORCE->0, CoP pinned to an edge) and report
  WHICH SATURATES FIRST = the motor-wall-vs-geometry-wall answer.

Run (from repo root):
  export PATH=/root/.nuka-toolchain-gcc14/bin:$PATH
  export CUDA_VISIBLE_DEVICES=0
  /root/miniconda3/envs/nuka-v03/bin/python python/spikes/h1_balance_gate.py
==============================================================================
"""

from __future__ import annotations

import math
import pathlib

import torch

import nuka

# Reuse the verified builders/helpers from the prior spike (same directory).
import h1_standing_pd_gate as pd  # noqa: E402

H1_MJCF = pd.H1_MJCF
DT = 0.005
G = 9.81

# H1 MJCF <actuator> ctrlrange (N*m, gear=1 -> ctrlrange == torque). Leg joints:
TORQUE_LIMIT = {
    "hip_yaw": 200.0, "hip_roll": 200.0, "hip_pitch": 200.0,
    "knee": 300.0, "ankle": 40.0,
}
LEG_JOINTS = pd.LEG_JOINTS  # 10 names, q[1:] order (slots 1..10).

# Per-cooked-link <inertial> offset (link-local, metres) + mass (kg), in cooked
# link order: 0=pelvis(LUMPED below), 1-5 left leg, 6-10 right leg.
# (pelvis row overwritten with the lumped values at build time.)
LEG_INERTIAL = {
    "hip_yaw":   ((-0.04923, 0.0001, 0.0072), 2.244),
    "hip_roll":  ((-0.0058, -0.00319, -9e-05), 2.232),
    "hip_pitch": ((0.00746, -0.02346, -0.08193), 4.152),
    "knee":      ((-0.00136, -0.00512, -0.1384), 1.721),
    "ankle":     ((0.042575, -1e-06, -0.044672), 0.474),
}


def _quat_rotate(q, v):
    """Rotate vec3 v by quaternion q=[w,x,y,z] (batched: q (...,4), v (...,3))."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    # t = 2 * cross(qvec, v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return torch.stack([rx, ry, rz], dim=-1)


class FaithfulCoM:
    """Whole-body CoM (env-batched) from world link poses + inertial offsets."""

    def __init__(self, blc, com_x_lump, com_z_lump, mass_lump, device):
        # link-local inertial offset + mass per cooked link.
        offs = [(com_x_lump, 0.0, com_z_lump)]  # link0 = lumped pelvis
        ms = [mass_lump]
        for side in ("left", "right"):
            for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"):
                o, m = LEG_INERTIAL[j]
                offs.append(o)
                ms.append(m)
        assert len(offs) == blc, (len(offs), blc)
        self.offs = torch.tensor(offs, dtype=torch.float32, device=device)  # (blc,3)
        self.ms = torch.tensor(ms, dtype=torch.float32, device=device)      # (blc,)
        self.total = float(self.ms.sum())

    def com(self, lp):
        """lp: (ec, blc, 7) world poses [px,py,pz,qw,qx,qy,qz]. -> (ec,3) world CoM."""
        ec, blc, _ = lp.shape
        pos = lp[:, :, 0:3]                                   # (ec,blc,3)
        quat = lp[:, :, 3:7]                                  # (ec,blc,4)
        off = self.offs.unsqueeze(0).expand(ec, blc, 3)       # (ec,blc,3)
        com_world = pos + _quat_rotate(quat, off)             # (ec,blc,3)
        m = self.ms.view(1, blc, 1)
        return (com_world * m).sum(dim=1) / self.total        # (ec,3)


def _read(w, field, ec, n):
    return torch.from_dlpack(w.buffer_view(field)).view(ec, n)


def run_balance(dev, mjcf_path, stance, gains, foot, com_lump,
                ec=64, n_steps=1000, label="", cop_on=True, settle_steps=40,
                ankle_ff=0.0, verbose_early=0):
    """Run the balance controller. Returns a metrics dict + held verdict.

    Staging knobs:
      cop_on       : engage the CoP ankle-strategy feedback (else ankle = ankle_ff).
      settle_steps : let the robot rest (CoP off, ankle=ankle_ff) this many steps
                     before CoP turns on, so the seat/impact transient decays first.
      ankle_ff     : fixed ankle feedforward torque (N*m) -- used for sign
                     calibration (cop_on=False) and as a small bias when on.
      verbose_early: print EVERY step for the first N steps (blowup is in <30 steps).
    """
    hp, kn, an = stance
    Kp, Kd, Kp_cop, Kd_cop, ankle_kd = (
        gains["Kp"], gains["Kd"], gains["Kp_cop"], gains["Kd_cop"],
        gains["ankle_kd"])
    toe_x, heel_x = foot
    com_x_lump, com_z_lump, mass_lump = com_lump
    qnom = pd._build_qnom(hp, kn, an)
    ground = pd.PELVIS_Z0 + pd._foot_z_under_pose(0.0, 0.0, 0.0)
    base_z = ground - pd._foot_z_under_pose(hp, kn, an, toe_x, heel_x)

    print(f"\n--- BALANCE [{label}] stance(hp,kn,an)=({hp:+.2f},{kn:+.2f},{an:+.2f}) "
          f"foot(toe,heel)=({toe_x:+.2f},{heel_x:+.2f}) cop_on={cop_on} "
          f"settle={settle_steps} ankle_ff={ankle_ff} Kp_cop={Kp_cop} "
          f"Kd_cop={Kd_cop} ---", flush=True)

    w = nuka.World.create_from_scene(dev, str(mjcf_path), ec, DT, 0, 1)  # torque mode
    try:
        blc = w.base_link_count
        assert blc == 11, f"expected blc==11, got {blc}"
        comc = FaithfulCoM(blc, com_x_lump, com_z_lump, mass_lump, "cuda")
        print(f"    whole-body mass (CoM model) = {comc.total:.2f} kg", flush=True)

        pd._seat_pose(w, qnom, base_z, ec, blc)
        tq = _read(w, nuka.TORQUE_INPUT, ec, blc)
        tq[:] = 0.0
        nuka.sync()

        qnom_t = torch.tensor(qnom, dtype=torch.float32, device="cuda")  # (10,)
        # physical torque limit per slot (slots 1..10).
        tlim = torch.tensor([TORQUE_LIMIT[n.split("_", 1)[1]] for n in LEG_JOINTS],
                            dtype=torch.float32, device="cuda")
        # per-slot Kp/Kd (posture). ankle slots get ~0 Kp (CoP overrides), small Kd.
        kp_slot, kd_slot, is_ankle = [], [], []
        for n in LEG_JOINTS:
            j = n.split("_", 1)[1]
            if j == "ankle":
                kp_slot.append(0.0); kd_slot.append(ankle_kd); is_ankle.append(1.0)
            else:
                kp_slot.append(Kp[j]); kd_slot.append(Kd[j]); is_ankle.append(0.0)
        kp_slot = torch.tensor(kp_slot, device="cuda")
        kd_slot = torch.tensor(kd_slot, device="cuda")
        ankle_mask = torch.tensor(is_ankle, device="cuda").bool()  # (10,)

        # foot polygon centre (world x) = midpoint of toe/heel under the ankle.
        # ankle world-x measured ~0.077; toe/heel are offsets in ankle frame, so
        # polygon centre world-x ~ ankle_x + (toe_x+heel_x)/2.
        # We compute poly_cx live from the ankle link world x each step (robust).

        def step_control(prev_com, k):
            lp = torch.from_dlpack(
                w.buffer_view(nuka.ARTICULATION_LINK_POSE)).view(ec, blc, 7)
            com = comc.com(lp)                                # (ec,3)
            com_v = (com - prev_com) / DT if prev_com is not None else \
                torch.zeros_like(com)
            # ankle link world x (links 5 and 10) -> polygon centre per side.
            ankle_lx = lp[:, 5, 0]                            # (ec,) left ankle x
            ankle_rx = lp[:, 10, 0]
            poly_cx = 0.5 * (ankle_lx + ankle_rx) + 0.5 * (toe_x + heel_x)  # (ec,)
            # state
            q = _read(w, nuka.JOINT_POSITION, ec, blc)
            qd = _read(w, nuka.JOINT_VELOCITY, ec, blc)
            qj = q[:, 1:1 + 10]                               # (ec,10) joint angles
            qdj = qd[:, 1:1 + 10]
            # posture PD (knee/hip); ankle slots have Kp=0 here.
            tau = kp_slot.view(1, 10) * (qnom_t.view(1, 10) - qj) \
                - kd_slot.view(1, 10) * qdj                   # (ec,10)
            # ANKLE STRATEGY: CoP-under-CoM, engaged only after the settle window.
            # SIGN (empirically calibrated, STAGE2): the body tips BACKWARD (com_ex
            # < 0), and a NEGATIVE ankle torque arrests it. So the stabilizing law
            # is tau = +Kp_cop*ex + Kd_cop*evx (ex<0 & evx<0 -> negative torque).
            # Both terms share the sign (flip P only -> the D term anti-damps).
            ex = com[:, 0] - poly_cx                          # (ec,)
            evx = com_v[:, 0]
            if cop_on and k >= settle_steps:
                tau_ankle = Kp_cop * ex + Kd_cop * evx + ankle_ff   # (ec,)
            else:
                tau_ankle = torch.full((ec,), float(ankle_ff), device="cuda")
            # apply ankle torque to both ankle slots (ankle keeps its small -Kd*qdot)
            tau_full = tau.clone()
            tau_full[:, ankle_mask] = tau_ankle.unsqueeze(1) \
                + tau_full[:, ankle_mask]
            # CLAMP to physical limits
            tau_clamped = torch.clamp(tau_full, -tlim.view(1, 10), tlim.view(1, 10))
            # write to torque buffer (slot0 root inert)
            tq = _read(w, nuka.TORQUE_INPUT, ec, blc)
            tq[:, 0] = 0.0
            tq[:, 1:1 + 10] = tau_clamped
            return com, tau_full, tau_clamped, ex, poly_cx

        # warm CoM (k=-1 -> CoP off regardless)
        com_prev, _, _, _, _ = step_control(None, -1)
        nuka.sync()

        half_len = 0.5 * (toe_x - heel_x)  # foot half-length (sagittal)
        peak_tau = torch.zeros(10, device="cuda")          # peak |clamped| per slot
        peak_tau_uncl = torch.zeros(10, device="cuda")     # peak |unclamped| per slot
        n_clamp_ankle = 0
        n_sphere_unload = 0
        max_tilt = 0.0
        min_edge = 1e9
        max_ex = 0.0
        min_contacts = 4
        z0 = None
        traj = []

        for k in range(n_steps):
            com_prev, tau_uncl, tau_cl, ex, poly_cx = step_control(com_prev, k)
            # NaN-guard peak tracking (NaN poisons torch.maximum -> garbage peaks).
            cl_max = tau_cl.abs().max(dim=0).values
            uncl_max = tau_uncl.abs().max(dim=0).values
            if not bool(torch.isnan(cl_max).any()):
                peak_tau = torch.maximum(peak_tau, cl_max)
            if not bool(torch.isnan(uncl_max).any()):
                peak_tau_uncl = torch.maximum(peak_tau_uncl, uncl_max)
            nuka.sync()
            w.step()
            # ---- EVERY-STEP metrics (env0): tilt, com_ex, contacts, clamp/unload.
            # The PASS criterion names tilt/CoM/contacts over the WHOLE rollout, so
            # these must be measured at every step (not just print steps), exactly
            # as peak torque is. We already sync + read poses each step, so cheap.
            nuka.sync()
            lp = torch.from_dlpack(
                w.buffer_view(nuka.ARTICULATION_LINK_POSE)).view(ec, blc, 7)
            bp = _read(w, nuka.BASE_POSE, ec, 7)
            cf = torch.from_dlpack(
                w.buffer_view(nuka.CONTACT_FORCE)).view(ec, -1, 3)
            qw, qx, qy, qz = bp[0, 3:7].tolist()
            tilt = pd._tilt_deg(qw, qx, qy, qz)
            pz = bp[0, 2].item()
            if z0 is None:
                z0 = pz
            com0 = comc.com(lp)[0]
            exv = (com0[0] - poly_cx[0]).item()
            edge = half_len - abs(exv)   # CoM distance to nearer sagittal edge
            fmag = cf[0].norm(dim=-1)
            nactive = int((fmag > 1e-4).sum().item())
            supported = float(cf[0, :, 0].sum().item()) / G
            ankle_tau0 = tau_cl[0, ankle_mask].abs().max().item()
            ankle_uncl0 = tau_uncl[0, ankle_mask].abs().max().item()
            clamp_hit = ankle_uncl0 > TORQUE_LIMIT["ankle"] + 1e-3
            if clamp_hit:
                n_clamp_ankle += 1
            if 0 < nactive < 4:
                n_sphere_unload += 1
            nan_now = math.isnan(tilt) or math.isnan(pz)
            if not nan_now:
                max_tilt = max(max_tilt, tilt)
                min_edge = min(min_edge, edge)
                max_ex = max(max_ex, abs(exv))
                min_contacts = min(min_contacts, nactive)
            do_print = ((k + 1) % 100 == 0 or k == 0
                        or (k + 1) <= verbose_early or nan_now)
            if do_print:
                traj.append((k + 1, tilt, pz, exv, nactive, supported,
                             ankle_tau0, ankle_uncl0))
                print(f"    t={k+1:4d} tilt={tilt:5.2f}deg z={pz:+.3f} "
                      f"com_ex={exv:+.4f}m edge={edge:+.4f}m contacts={nactive}/4 "
                      f"sup={supported:4.1f}kg |tau_an|={ankle_tau0:5.1f}"
                      f"(uncl {ankle_uncl0:5.1f}){'!CLAMP' if clamp_hit else ''}",
                      flush=True)
            if nan_now:
                print("        !! NaN -- abort", flush=True)
                break

        nuka.sync()
        bp = _read(w, nuka.BASE_POSE, ec, 7)
        lp = torch.from_dlpack(
            w.buffer_view(nuka.ARTICULATION_LINK_POSE)).view(ec, blc, 7)
        tf = pd._tilt_deg(*bp[0, 3:7].tolist())
        zf = bp[0, 2].item()
        envs_up = sum(1 for e in range(ec)
                      if (pd._tilt_deg(*bp[e, 3:7].tolist()) < 15.0
                          and not math.isnan(bp[e, 2].item())))

        # peak torque report (per joint name, max across legs+envs)
        peak_by_joint = {}
        peak_uncl_by_joint = {}
        for i, n in enumerate(LEG_JOINTS):
            j = n.split("_", 1)[1]
            peak_by_joint[j] = max(peak_by_joint.get(j, 0.0),
                                   peak_tau[i].item())
            peak_uncl_by_joint[j] = max(peak_uncl_by_joint.get(j, 0.0),
                                        peak_tau_uncl[i].item())

        held = (tf < 15.0 and max_tilt < 15.0 and not math.isnan(zf)
                and abs(zf - (z0 or zf)) < 0.12 and envs_up >= ec // 2)
        within_torque = all(
            peak_uncl_by_joint[j] <= TORQUE_LIMIT[j] + 1e-3
            for j in TORQUE_LIMIT)

        print(f"    => tilt_max={max_tilt:.2f}deg tf={tf:.2f} z {z0:+.3f}->{zf:+.3f} "
              f"com_ex_max={max_ex:.4f}m min_edge={min_edge:+.4f}m "
              f"min_contacts={min_contacts}/4 envs_up={envs_up}/{ec} "
              f"(metrics tracked EVERY step)", flush=True)
        print(f"       PEAK |tau| (clamped) :", flush=True)
        for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"):
            print(f"         {j:9s} {peak_by_joint[j]:6.1f} / {TORQUE_LIMIT[j]:5.0f} "
                  f"N*m  (uncl peak {peak_uncl_by_joint[j]:6.1f})", flush=True)
        print(f"       within_physical_torque={within_torque}  "
              f"ankle_clamp_events={n_clamp_ankle}  "
              f"sphere_unload_events={n_sphere_unload}", flush=True)
        verdict = "STOOD" if (held and within_torque) else (
            "STOOD-but-EXCEEDS-TORQUE" if held else "FELL")
        print(f"    VERDICT[{label}] = {verdict}", flush=True)
        return {
            "label": label, "held": held, "within_torque": within_torque,
            "tilt_max": max_tilt, "tf": tf, "z0": z0, "zf": zf,
            "com_ex_max": max_ex, "min_edge": min_edge, "envs_up": envs_up, "ec": ec,
            "min_contacts": min_contacts,
            "peak": peak_by_joint, "peak_uncl": peak_uncl_by_joint,
            "ankle_clamp_events": n_clamp_ankle,
            "sphere_unload_events": n_sphere_unload, "half_len": half_len,
        }
    finally:
        w.destroy()


def main():
    print("nuka:", nuka.__file__, flush=True)
    ec, n = 64, 1000

    # Faithful lumped pelvis: 30.62 kg lumped at (-0.02, 0, 0.18) in pelvis frame
    # -> total ~52.3 kg (matches full H1). com_lump = (com_x, com_z, mass).
    COM_LUMP = (-0.02, 0.18, 30.62)
    STANCE = (-0.40, 0.70, -0.30)  # hip_pitch, knee, ankle (CoM ~ above ankle)

    # posture PD at physical gains (knee/hip). torque mode -> explicit Kd.
    KP = {"hip_yaw": 150, "hip_roll": 200, "hip_pitch": 200, "knee": 300, "ankle": 0}
    KD = {"hip_yaw": 8, "hip_roll": 12, "hip_pitch": 14, "knee": 18, "ankle": 0}

    # Foot symmetric about the ankle (polygon centred on the ankle so static
    # ankle torque ~0, preserving the +/-40 budget for feedback).
    FOOT = (0.10, -0.10)

    faith = H1_MJCF.parent / "h1_balance.xml"
    results = []
    try:
        with nuka.Device.create(0) as dev:
            pd._make_faithful(faith, toe_x=FOOT[0], heel_x=FOOT[1],
                              com_z=COM_LUMP[1], mass=COM_LUMP[2],
                              com_x=COM_LUMP[0], inertia=0.8)

            # === STAGE 1: BASELINE -- posture PD on knee/hip, ankle=0, NO CoP. ===
            # The real gate signal: does the seat settle to 4/4 at ~52 kg, and how
            # fast does it tip with zero ankle authority? Print EVERY early step.
            print("\n############ STAGE 1: baseline (posture PD, ankle=0, no CoP) "
                  "############", flush=True)
            g0 = {"Kp": KP, "Kd": KD, "Kp_cop": 0.0, "Kd_cop": 0.0, "ankle_kd": 1.5}
            r1 = run_balance(dev, faith, STANCE, g0, FOOT, COM_LUMP, ec,
                             n_steps=200, label="STAGE1 baseline",
                             cop_on=False, settle_steps=0, ankle_ff=0.0,
                             verbose_early=30)
            results.append(r1)

            # === STAGE 2: ankle SIGN calibration -- fixed +5 vs -5 N*m ankle. ===
            # From the settled posture, push a constant ankle torque and watch which
            # sign drives com_x toward the polygon centre (stabilizing sign).
            print("\n############ STAGE 2: ankle sign calibration (+/-5 N*m) "
                  "############", flush=True)
            for ff in (+5.0, -5.0):
                r = run_balance(dev, faith, STANCE, g0, FOOT, COM_LUMP, ec,
                                n_steps=120, label=f"STAGE2 ankle_ff={ff:+.0f}",
                                cop_on=False, settle_steps=0, ankle_ff=ff,
                                verbose_early=25)
                results.append(r)

            # === STAGE 3: CoP engaged with the VERIFIED sign + short settle. ===
            # Sign FLIPPED to +Kp_cop*ex (+Kd_cop*evx) per STAGE2 (negative ankle
            # torque arrests the backward tip). Natural gain: each ankle carries
            # ~m*g/2 ~ 255 N, tau = F*(cop-ankle) -> Kp_cop ~ 255 to make the CoP
            # track the CoM. Sweep ~200-350 (>=500 over-drives -> overshoot/oscill).
            # settle ~15: past the t=2-3 impact transient, before the slow divergence
            # (com_ex flat ~-0.065 with com_v~0 through t~30) -> clean engage ~17 N*m.
            print("\n############ STAGE 3: CoP ankle strategy (settle then engage) "
                  "############", flush=True)
            for kpc, kdc in [(200.0, 30.0), (255.0, 40.0), (320.0, 50.0),
                             (350.0, 55.0)]:
                g3 = {"Kp": KP, "Kd": KD, "Kp_cop": kpc, "Kd_cop": kdc,
                      "ankle_kd": 1.5}
                r = run_balance(dev, faith, STANCE, g3, FOOT, COM_LUMP, ec,
                                n_steps=n, label=f"STAGE3 Kcop{int(kpc)}",
                                cop_on=True, settle_steps=15, ankle_ff=0.0,
                                verbose_early=18)
                results.append(r)

        print("\n================ SUMMARY ================", flush=True)
        for r in results:
            print(f"  {r['label']:24s} held={r['held']!s:5s} "
                  f"within_torque={r['within_torque']!s:5s} "
                  f"tilt_max={r['tilt_max']:6.1f} min_ct={r['min_contacts']}/4 "
                  f"envs_up={r['envs_up']}/{r['ec']} "
                  f"peak_ankle_uncl={r['peak_uncl']['ankle']:6.1f}/40 "
                  f"clamp_ev={r['ankle_clamp_events']} "
                  f"unload_ev={r['sphere_unload_events']}", flush=True)
    finally:
        if faith.exists():
            faith.unlink()
    print("\n=== spike complete ===", flush=True)


if __name__ == "__main__":
    main()
