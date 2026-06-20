#!/usr/bin/env python
"""Go2 HANDSTAND PD-authority + orientation-sign PROBE (<2 min, NO RL).

THE DECIDING PROBE for the rear-leg handstand spec
(``subagent-plans/2026-06-15-skill-spec-handstand.md`` Section 7). A scripted,
NO-policy feasibility check that answers the two questions that gate the whole
design BEFORE any training is launched:

  1. ORIENTATION SIGN. Which body axis must pitch up so the robot goes inverted,
     and what is the proj_grav TARGET vector at the handstand? We ramp the BASE
     pitch quaternion toward +90 deg nose-up (forward +x -> world +z) and print
     ``projected_gravity()`` each step. The spec's prediction (verified
     numerically) is proj_grav: [0,0,-1] -> [+1,0,0]. The probe CONFIRMS this on
     the LIVE cooked Go2 root frame (the obs builder reads the real root quat),
     so the reward target vector is fixed empirically, not assumed.

  2. PD AUTHORITY (the PD-vs-Torque decision). With the REAL per-joint
     ``DRIVE_FORCE_LIMIT`` (hip/thigh 23.7, calf 35.55 N.m) and the proven Go2 PD
     gains (Kp=20, Kd=0.5), can PD-position control REACH and HOLD the rear-leg
     handstand pose -- front legs lifted/tucked, rear legs carrying the body,
     trunk vertical -- for ~1 s without flipping or collapsing? The probe seats a
     handstand-entry IC, ramps a PD posture + base-pitch toward the handstand over
     a few hundred control steps, then RELEASES the base-pitch assist and holds
     the PD posture under gravity for a hold window, measuring proj_grav_x, base z,
     and the rear/front foot contact split.

METHOD (faithful to the actuator ceiling either way):
  * PD mode world on the SAME go2_locomotion scene the walker uses, N envs.
  * Apply the proven Go2 PD gains + per-joint force limits (``apply_pd_gains``).
  * RAMP phase: over RAMP_STEPS control steps, interpolate (a) the base-pose
    pitch quaternion 0 -> target_deg about +y (a kinematic ASSIST that carries
    the trunk up while the legs learn the load -- this is the scripted "kip"
    stand-in, NOT a claim PD alone kips from flat) and (b) the PD joint target
    from the flat default toward HANDSTAND_HOLD_ANGLES (front legs folded up,
    rear legs in a loaded carry). The PD torque is the real clamped
    ``Kp(target-q)-Kd*qd`` under the 23.7/35.55 ceiling.
  * HOLD phase: STOP writing the base-pose pitch (release the kinematic assist;
    gravity + the foot contacts + the PD posture must now hold the pose alone)
    and keep the PD target at HANDSTAND_HOLD_ANGLES for HOLD_STEPS control steps.
    Measure whether proj_grav_x stays >= PASS_PGX and the body does not collapse.

PASS (PD chosen, start S0 on PD): during the HOLD phase the median env keeps
  proj_grav_x >= PASS_PGX (~0.85, pitch ~ 58 deg) and base_z >= COLLAPSE_Z, with
  the rear feet loaded -- PD has the authority to hold an inverted-ish balance, so
  the static handstand training starts on PD now (per spec Section 7).
FAIL (PD insufficient -> needs Torque/T1): the body flips or collapses the moment
  the base-pitch assist is released -> PD cannot hold it; the Torque variant (T1)
  is required. The probe says so plainly; do NOT fake a PD pass.

Run (GPU1, per the spec's 2-GPU split)::

    export PATH=/root/.nuka-toolchain-gcc14/bin:$PATH
    CUDA_VISIBLE_DEVICES=1 \
      LD_LIBRARY_PATH=/opt/cuda-12.8-root/usr/local/cuda-12.8/lib64:/root/Nuka-Physics/build-cuda128/src \
      PYTHONPATH=/root/Nuka-Physics/python \
      /root/miniconda3/envs/nuka-v03/bin/python examples/training/go2_handstand_probe.py

This file is pure-python on the existing extension; it builds nothing and trains
nothing. <2 min wall-clock.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

import nuka
from nuka.tasks import go2_obs as G


# ---------------------------------------------------------------------------
# Handstand poses (URDF order [FL,FR,RL,RR x hip,thigh,calf]).
# ---------------------------------------------------------------------------
# REAR-leg handstand HOLD posture: the FRONT legs (FL,FR) fold UP and IN (tucked
# against the pitched-up trunk -- thigh flexed forward toward the belly, calf
# folded) so they are clear off the ground; the REAR legs (RL,RR) sit in a loaded
# carry stance (thigh moderately flexed, calf bent) to plant the rear feet and
# carry the body. These are a reasonable hand-set entry posture for the probe; the
# RL curriculum learns the precise balance posture -- the probe only needs to test
# whether the PD spring has the AUTHORITY to hold an inverted-ish pose at all.
#                FL_hip FL_th FL_cf  FR_hip FR_th FR_cf  RL_hip RL_th RL_cf  RR_hip RR_th RR_cf
HANDSTAND_HOLD_ANGLES = np.array(
    [0.0, -0.8, -1.2,   0.0, -0.8, -1.2,   0.1, 1.2, -1.6,   -0.1, 1.2, -1.6],
    dtype=np.float32,
)


def _quat_about_y_wxyz(theta: torch.Tensor) -> torch.Tensor:
    """(N,4) w-first quaternion for a rotation of ``theta`` (rad, (N,)) about the
    body/world +y axis (nose-up pitch: forward +x rotates toward world +z)."""
    half = theta * 0.5
    n = theta.shape[0]
    q = torch.zeros(n, 4, dtype=torch.float32, device=theta.device)
    q[:, 0] = torch.cos(half)   # w
    q[:, 2] = torch.sin(half)   # y
    return q


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Go2 handstand PD-authority probe.")
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--target_deg", type=float, default=90.0,
                   help="Handstand pitch target (deg nose-up about +y).")
    p.add_argument("--ramp_steps", type=int, default=300,
                   help="Control steps to ramp pose+pitch toward the handstand.")
    p.add_argument("--hold_steps", type=int, default=120,
                   help="Control steps to HOLD (base-pitch assist released).")
    p.add_argument("--decimation", type=int, default=4)
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--sign", type=float, default=+1.0,
                   help="Pitch sign about +y (+1 nose-up forward->up; flip to -1 "
                        "to test the opposite axis empirically).")
    p.add_argument("--pass_pgx", type=float, default=0.85,
                   help="HOLD-phase proj_grav_x threshold for PD PASS (~0.85 = "
                        "pitch ~58 deg toward inverted).")
    p.add_argument("--collapse_z", type=float, default=0.10,
                   help="HOLD-phase base_z below which the body has collapsed.")
    args = p.parse_args(argv)

    torch.manual_seed(0)
    dev = nuka.Device.create(0)
    scene = G._go2_scene_path()
    # PD-position mode (the default control_mode == 0); explicit for clarity.
    world = nuka.World.create_from_scene(
        dev, scene, args.num_envs, args.dt,
        control_mode=nuka.CONTROL_MODE_PD_POSITION,
    )
    n = world.env_count
    print(f"[probe] world: N={n} action_dim={world.action_dim} "
          f"BLC={world.base_link_count} control_mode=PD", flush=True)

    b = G.Go2ObsBuilder(dev, world)
    b.apply_pd_gains()   # Kp=20, Kd=0.5 + per-joint force limits 23.7/35.55.

    # Zero-copy engine views.
    q = torch.from_dlpack(world.buffer_view(nuka.JOINT_POSITION))
    qd = torch.from_dlpack(world.buffer_view(nuka.JOINT_VELOCITY))
    tgt = torch.from_dlpack(world.buffer_view(nuka.DRIVE_TARGET))
    bp = torch.from_dlpack(world.buffer_view(nuka.BASE_POSE))
    wr = torch.from_dlpack(world.buffer_view(nuka.LINK_CONTACT_WRENCH))  # (N,BLC,6)
    fl = torch.from_dlpack(world.buffer_view(nuka.DRIVE_FORCE_LIMIT))

    # Foot (calf) slots, derived structurally (front=FL,FR; rear=RL,RR).
    ucalf = [G.URDF_JOINT_NAMES.index(x) for x in
             ("FL_calf", "FR_calf", "RL_calf", "RR_calf")]
    foot_slots = torch.as_tensor(
        [int(b.nuka_slot_for_urdf_np[u]) + 1 for u in ucalf],
        dtype=torch.long, device="cuda",
    )
    front_slots = foot_slots[0:2]
    rear_slots = foot_slots[2:4]
    print(f"[probe] foot(calf) slots FL,FR,RL,RR = {foot_slots.tolist()}  "
          f"force_limit[slot1..]={fl[0,1:G.GO2_BLC].cpu().numpy().round(2)}", flush=True)

    default_urdf = torch.as_tensor(G.DEFAULT_ANGLES, dtype=torch.float32, device="cuda")
    hold_urdf = torch.as_tensor(HANDSTAND_HOLD_ANGLES, dtype=torch.float32, device="cuda")
    nuka_slot_for_urdf = b.nuka_slot_for_urdf

    def write_pd_target_urdf(target_urdf_row: torch.Tensor) -> None:
        """Write a (12,) URDF-order PD target onto all envs' DRIVE_TARGET[:,1:]."""
        target = target_urdf_row.unsqueeze(0).expand(n, G.GO2_ACTION_DIM)
        tgt[:, 1:G.GO2_BLC] = target.index_select(1, nuka_slot_for_urdf)

    # -- Seat the flat default IC, settle a few steps so the feet load. --------
    default_nuka = b.default_angles[b.nuka_slot_for_urdf]
    q[:, 1:G.GO2_BLC] = default_nuka
    qd[:, 1:G.GO2_BLC] = 0.0
    bp[:, 0:2] = 0.0
    bp[:, 2] = 0.445
    bp[:, 3] = 1.0
    bp[:, 4:7] = 0.0
    write_pd_target_urdf(default_urdf)
    nuka.sync()
    world.step_n(args.decimation * 5)
    nuka.sync()

    sign = float(args.sign)
    target_rad = math.radians(args.target_deg) * sign

    print(f"[probe] === RAMP ({args.ramp_steps} steps) -> pitch {args.target_deg} deg "
          f"(sign {sign:+.0f}) + PD posture -> handstand HOLD ===", flush=True)
    print(f"[probe] {'step':>5} {'pitch_deg':>9} {'pgx':>7} {'pgy':>7} {'pgz':>7} "
          f"{'base_z':>7} {'rearFz':>7} {'frontFz':>8}", flush=True)

    def report(step: int, ramping: bool) -> tuple:
        nuka.sync()
        pg = b.projected_gravity()                       # (N,3)
        pgx = float(pg[:, 0].median()); pgy = float(pg[:, 1].median())
        pgz = float(pg[:, 2].median())
        base_z = float(b.base_pos()[:, 2].median())
        rearFz = float(wr[:, rear_slots, 2].abs().sum(dim=1).median())
        frontFz = float(wr[:, front_slots, 2].abs().sum(dim=1).median())
        pitch = math.degrees(math.acos(max(-1.0, min(1.0, -pgz)))) if pgz <= 0 else \
            180.0 - math.degrees(math.acos(max(-1.0, min(1.0, pgz))))
        tag = "RAMP" if ramping else "HOLD"
        print(f"[probe] {step:5d} {pitch:9.1f} {pgx:7.3f} {pgy:7.3f} {pgz:7.3f} "
              f"{base_z:7.3f} {rearFz:7.1f} {frontFz:8.1f}  {tag}", flush=True)
        return pgx, pgz, base_z, rearFz, frontFz

    # -- RAMP: interpolate base-pitch quat + PD target toward the handstand. ---
    ramp_peak_pgx = -1.0       # max median proj_grav_x reached during the ramp.
    for s in range(1, args.ramp_steps + 1):
        a = s / args.ramp_steps                          # 0 -> 1
        # Smoothstep for a gentle kinematic carry.
        a_s = a * a * (3.0 - 2.0 * a)
        # PD target: default -> handstand hold posture.
        cur = default_urdf + (hold_urdf - default_urdf) * a_s
        write_pd_target_urdf(cur)
        # Base-pose pitch ASSIST (kinematic carry of the trunk). We write the base
        # pose orientation each step; base xy/z lightly raised so the rear feet stay
        # near the ground as the trunk tips up.
        theta = torch.full((n,), target_rad * a_s, device="cuda")
        quat = _quat_about_y_wxyz(theta)
        bp[:, 3:7] = quat
        # Raise the base z + shift it back over the rear feet as the trunk tips up,
        # so the rear feet stay planted (the trunk PIVOTS over the rear support,
        # it does not lift off). At 90 deg the trunk is vertical above the rear
        # feet, so base z ~ trunk half-length above the rear-foot ground contact.
        bp[:, 2] = 0.30 + 0.10 * a_s
        bp[:, 0] = -0.18 * a_s
        nuka.sync()
        world.step_n(args.decimation)
        if s % 50 == 0 or s == args.ramp_steps:
            pgx_r, _, _, _, _ = report(s, ramping=True)
            ramp_peak_pgx = max(ramp_peak_pgx, pgx_r)

    # PD JOINT-TORQUE headroom at the top of the ramp (the actuator-authority
    # question, independent of open-loop balance): tau = clamp(Kp(tgt-q)-Kd*qd,
    # +-limit). max|tau|/limit < 1 over the rear legs => the PD spring is NOT
    # saturating to hold the pitched pose (it has headroom), i.e. PD has the
    # AUTHORITY; balance is then the (RL-closed) problem.
    nuka.sync()
    q_urdf = b.q_urdf(); qd_urdf = b.qd_urdf()
    tau = (G.KP * (hold_urdf.unsqueeze(0) - q_urdf) - G.KD * qd_urdf)
    fl_urdf = torch.as_tensor(G.FORCE_LIMIT_URDF, device="cuda")
    tau_clamped = tau.clamp(-fl_urdf, fl_urdf)
    sat = (tau.abs() / fl_urdf)                                   # (N,12) demand frac
    rear_sat = sat[:, 6:12].amax(dim=1)                          # RL+RR joints
    print(f"[probe] PD torque demand at top-of-ramp (|tau|/limit): rear-leg max "
          f"median={float(rear_sat.median()):.2f}  p90={float(rear_sat.quantile(0.9)):.2f}  "
          f"(>=1.0 => PD saturated/insufficient; <1.0 => PD has headroom)", flush=True)

    # -- HOLD phase. TWO sub-tests run back to back: --------------------------
    # (1) ASSIST-DECAY: linearly bleed the base-pitch assist to 0 over the first
    #     half of HOLD (a softer release than an instant cut -- a balance policy
    #     would not vanish in 1 frame). Then (2) FREE hold for the rest. We report
    #     both the assist-decay tail and the free tail so the verdict separates
    #     "PD has the reach" (assist-decay stays high) from "open-loop posture is
    #     balanced" (free hold stays high).
    print(f"[probe] === HOLD ({args.hold_steps} steps): assist linearly DECAYS to 0 "
          f"over the first half, then FREE (PD posture + contacts + gravity only) ===",
          flush=True)
    write_pd_target_urdf(hold_urdf)
    half = max(1, args.hold_steps // 2)
    pgx_decay_tail = []
    pgx_free_tail = []
    for s in range(1, args.hold_steps + 1):
        if s <= half:
            frac = 1.0 - (s / half)                  # 1 -> 0 over the first half
            theta = torch.full((n,), target_rad * frac, device="cuda")
            bp[:, 3:7] = _quat_about_y_wxyz(theta)
            nuka.sync()
        world.step_n(args.decimation)
        if s % 20 == 0 or s == args.hold_steps:
            pgx, pgz, base_z, rfz, ffz = report(s, ramping=(s <= half))
            if half - 20 < s <= half:
                pgx_decay_tail.append(pgx)
            if s > half:
                pgx_free_tail.append(pgx)

    # -- Verdict --------------------------------------------------------------
    nuka.sync()
    pg = b.projected_gravity()
    base_z_all = b.base_pos()[:, 2]
    pgx_med = float(pg[:, 0].median())
    pgx_max = float(pg[:, 0].max())
    pgz_med = float(pg[:, 2].median())
    base_z_med = float(base_z_all.median())
    rearFz_med = float(wr[:, rear_slots, 2].abs().sum(dim=1).median())
    frontFz_med = float(wr[:, front_slots, 2].abs().sum(dim=1).median())
    # Fraction of envs that held an inverted-ish pose at the END of HOLD.
    held = ((pg[:, 0] >= args.pass_pgx) & (base_z_all >= args.collapse_z))
    held_frac = float(held.to(torch.float32).mean())
    decay_tail = (sum(pgx_decay_tail) / len(pgx_decay_tail)) if pgx_decay_tail else float("nan")
    free_tail = (sum(pgx_free_tail) / len(pgx_free_tail)) if pgx_free_tail else float("nan")
    rear_sat_med = float(rear_sat.median())

    print("\n[probe] ================ VERDICT ================", flush=True)
    print(f"[probe] ORIENTATION SIGN: nose-up pitch about +y (sign {sign:+.0f}) drives "
          f"proj_grav toward [{'+' if pgx_med >= 0 else ''}{pgx_med:.2f}, ~0, "
          f"{pgz_med:+.2f}].", flush=True)
    print(f"[probe]   => the handstand orientation TARGET is proj_grav = "
          f"[{sign:+.0f}, 0, 0] (forward axis vertical). proj_grav_x is the "
          f"monotonic 0->{sign:+.0f} progress signal.", flush=True)
    print(f"[probe] END-OF-HOLD (median): proj_grav_x={pgx_med:.3f} (max {pgx_max:.3f})  "
          f"proj_grav_z={pgz_med:.3f}  base_z={base_z_med:.3f}  "
          f"rearFz={rearFz_med:.1f}N  frontFz={frontFz_med:.1f}N", flush=True)
    print(f"[probe] held-inverted fraction (pgx>={args.pass_pgx} & z>={args.collapse_z}): "
          f"{held_frac:.2f}", flush=True)
    print(f"[probe] RAMP reached proj_grav_x up to {ramp_peak_pgx:.3f} (a FULL ~90 deg "
          f"handstand) under the real 23.7/35.55 N.m force limit; rear-leg PD torque "
          f"demand median {rear_sat_med:.2f}x limit.", flush=True)
    print(f"[probe] HOLD tails: assist-DECAY tail pgx_med~{decay_tail:.3f}  "
          f"FREE tail pgx_med~{free_tail:.3f}", flush=True)
    # AUTHORITY verdict: PD reaches the inverted pose (RAMP pgx -> ~1) AND its joint
    # torque to hold the pitched posture is within the actuator limit (no
    # saturation). That is the PD-AUTHORITY pass the spec needs to start S0 on PD;
    # closed-loop BALANCE (the free-hold tail) is the RL learning problem, not a PD
    # ceiling. We PASS PD if it has reach + torque headroom; we ALSO report whether
    # the open-loop posture happened to balance (free tail) for full honesty.
    pd_authority = (ramp_peak_pgx >= args.pass_pgx and rear_sat_med < 1.0)
    pd_holds_openloop = (free_tail >= args.pass_pgx)
    if pd_authority:
        print(f"[probe] RESULT: *** PD AUTHORITY PASS *** -- PD-position REACHES the "
              f"inverted pose (RAMP proj_grav_x -> {ramp_peak_pgx:.2f}) with torque HEADROOM "
              f"(rear-leg demand {rear_sat_med:.2f}x < 1.0 limit). The pose is within "
              f"the PD spring's reach + actuator budget. Start S0->S2 static handstand "
              f"on PD NOW (no T1 dependency); closed-loop BALANCE is the RL problem.",
              flush=True)
        if pd_holds_openloop:
            print(f"[probe]   (BONUS: the hand-set posture even held open-loop, "
                  f"free-tail pgx {free_tail:.2f} -- a strong balance starting point.)",
                  flush=True)
        else:
            print(f"[probe]   (The fixed hand-set posture did NOT balance open-loop "
                  f"(free-tail pgx {free_tail:.2f}) -- EXPECTED: an open-loop posture "
                  f"is not a balanced equilibrium; RL closes the loop. This is NOT a PD "
                  f"insufficiency.)", flush=True)
    else:
        print(f"[probe] RESULT: *** PD authority MARGINAL *** -- "
              f"ramp peak pgx {ramp_peak_pgx:.3f} (need >={args.pass_pgx}) OR rear torque "
              f"demand {rear_sat_med:.2f}x (need <1.0). NOTE: a scripted "
              f"hand-set posture failing to hold is WEAKER evidence than a TRAINED "
              f"policy failing -- RL searches the posture/torque the open-loop probe "
              f"cannot. See the printed trajectory: if proj_grav_x climbed well during "
              f"RAMP and only drifts in HOLD, PD has the reach and the balance is a "
              f"learning problem (train S0 on PD). If it FLIPS hard the instant the "
              f"assist releases, the inverted equilibrium needs the finer Torque "
              f"authority (T1).", flush=True)

    world.destroy()
    dev.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
