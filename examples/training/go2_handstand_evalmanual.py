#!/usr/bin/env python
"""Manual (NO-autoreset) rollout of the trained Go2 handstand policy.

Bypasses env.step()'s internal autoreset: we drive the world directly (write the
policy action -> DRIVE_TARGET, step_n(decimation)) and evaluate the env's OWN
compute_terminated() each step on the LIVE pre-reset state. This shows the TRUE
termination cause + the real held trajectory of a SINGLE episode (no fresh-IC
injection). Builds the obs exactly as the env does (49-dim, theta appended).
"""

from __future__ import annotations
import argparse
import numpy as np
import torch
import nuka  # noqa
from nuka.tasks.go2_handstand import make_env, COLLAPSE_FLOOR_Z
from nuka.tasks import go2_obs as G
from go2_handstand_eval import load_actor


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--stage", default="S2")
    p.add_argument("--theta-cmd", type=float, default=90.0)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--settle", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--zero-action", action="store_true")
    args = p.parse_args(argv)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    env = make_env(args.num_envs, stage=args.stage, theta_cmd_obs=True, seed=args.seed)
    dev = env._torch_device; n = env.num_envs
    forward, ep, rew = load_actor(args.ckpt, env._obs_dim, G.GO2_ACTION_DIM, dev)
    print(f"[man] ckpt epoch={ep} rew={rew} stage={args.stage} n={n} "
          f"zero_action={args.zero_action} steps={args.steps}", flush=True)

    # Reset seats the IC + samples theta; force theta=cmd and re-pin each step.
    env.reset(seed=args.seed)
    env._theta_cmd_deg[:] = float(args.theta_cmd)
    nuka.sync()
    # One settle step so FK-lagged poses reflect the seated IC before we read/obs.
    env._world.step_n(env.decimation); nuka.sync()

    b = env._obs
    front_slot = env._front_foot_slot; rear_slot = env._rear_foot_slot
    last_action = torch.zeros(n, G.GO2_ACTION_DIM, device=dev)

    alive = torch.ones(n, dtype=torch.bool, device=dev)
    steps_to_term = torch.full((n,), float(args.steps), device=dev)
    # accumulators
    acc = {k: torch.zeros(n, device=dev) for k in
           ("pgx", "pgyabs", "pgz", "bz", "rear", "frontc", "frontz", "cnt")}
    term_cause = torch.zeros(4, device=dev)  # flip, coll, nonf, runa (first-death)

    for k in range(args.steps):
        # Build obs the env way (48-dim) + append theta.
        obs48 = b.compute_obs(env.command, last_action)
        obs = env._append_theta(obs48)
        act = (torch.zeros(n, G.GO2_ACTION_DIM, device=dev) if args.zero_action
               else forward(obs))
        # Write action -> DRIVE_TARGET (clipped), step the world (no autoreset).
        clipped = b.write_action(act)
        last_action = clipped
        nuka.sync()
        env._world.step_n(env.decimation)
        nuka.sync()

        pg = b.projected_gravity(); base_z = b.base_pos()[:, 2]
        q = b.q_urdf(); qd = b.qd_urdf()
        rear_loaded = (env._wrench_view[:, rear_slot, 2].abs() > 1.0).to(torch.float32).sum(1)/2.0
        front_cnt = (env._wrench_view[:, front_slot, 2].abs() > 1.0).to(torch.float32).sum(1)
        front_z = env._pose_view[:, front_slot, 2].mean(1)

        # NOSE-DOWN convention (corrected): flip-over = overshot vertical onto the
        # back => pgz>+0.7 AND pgx>0 (mirrors Go2HandstandEnv.compute_terminated).
        flipped = (pg[:, 2] > 0.7) & (pg[:, 0] > 0.0)
        collapsed = base_z < COLLAPSE_FLOOR_Z
        nonfinite = (~torch.isfinite(b.base_pos()).all(1) | ~torch.isfinite(q).all(1)
                     | ~torch.isfinite(qd).all(1))
        runaway = qd.abs().amax(1) > 50.0
        term = flipped | collapsed | nonfinite | runaway

        new = term & alive
        if bool(new.any()):
            steps_to_term[new] = float(k + 1)
            term_cause[0] += float((flipped & new).sum())
            term_cause[1] += float((collapsed & new).sum())
            term_cause[2] += float((nonfinite & new).sum())
            term_cause[3] += float((runaway & new).sum())
        alive = alive & (~term)

        if k >= args.settle:
            m = alive.to(torch.float32)
            acc["pgx"] += (-pg[:, 0])*m; acc["pgyabs"] += pg[:, 1].abs()*m   # INV=-pgx
            acc["pgz"] += pg[:, 2]*m; acc["bz"] += base_z*m
            acc["rear"] += rear_loaded*m; acc["frontc"] += front_cnt*m
            acc["frontz"] += front_z*m; acc["cnt"] += m

        if k % 20 == 0 or k == args.steps-1:
            a = alive
            mp = float((-pg[a, 0]).median()) if bool(a.any()) else float("nan")
            mz = float(base_z[a].median()) if bool(a.any()) else float("nan")
            print(f"[man] step {k:4d} alive={int(a.sum()):4d}/{n} | "
                  f"inv(-pgx, alive med)={mp:+.3f} bz={mz:.3f} | "
                  f"qdmax(med)={float(qd.abs().amax(1).median()):.1f}", flush=True)

    contributed = acc["cnt"] > 0
    nc = int(contributed.sum())
    def pem(key): return acc[key][contributed] / acc["cnt"][contributed]
    surv = (steps_to_term >= args.steps)
    surv_frac = float(surv.to(torch.float32).mean())
    def fmt(t):
        if t.numel() == 0: return "EMPTY"
        return (f"med={float(t.median()):+.3f} mean={float(t.mean()):+.3f} "
                f"p10={float(t.quantile(0.1)):+.3f} p90={float(t.quantile(0.9)):+.3f}")

    print("\n[man] ============ SINGLE-EPISODE (no autoreset) VERDICT ============", flush=True)
    print(f"[man] stage={args.stage} theta={args.theta_cmd}deg NOSE-DOWN "
          f"window=[{args.settle},{args.steps}) contributed={nc}/{n}", flush=True)
    inv_med = rear_med = frontc_med = float("nan")
    if nc > 0:
        inve = pem("pgx")     # INV = -pgx accumulator
        inv_med = float(inve.median())
        reare = pem("rear"); frontce = pem("frontc")
        rear_med = float(reare.median()); frontc_med = float(frontce.median())
        pitch = np.degrees(np.arcsin(np.clip(inv_med, -1, 1)))
        print(f"[man] inversion -proj_grav_x: {fmt(inve)}  -> rear-up pitch~{pitch:.1f}deg", flush=True)
        print(f"[man] |pgy|: {fmt(pem('pgyabs'))}", flush=True)
        print(f"[man] proj_grav_z: {fmt(pem('pgz'))}", flush=True)
        print(f"[man] base_z: {fmt(pem('bz'))}", flush=True)
        print(f"[man] rear contact frac: {fmt(reare)}", flush=True)
        print(f"[man] front contact cnt: {fmt(frontce)}", flush=True)
        print(f"[man] front foot z: {fmt(pem('frontz'))}", flush=True)
    print(f"[man] survival(no-term over {args.steps})={surv_frac:.3f}  "
          f"mean steps-to-term={float(steps_to_term.mean()):.1f}/{args.steps}", flush=True)
    tc = term_cause.cpu().numpy()
    print(f"[man] first-death cause counts: flip={int(tc[0])} collapse={int(tc[1])} "
          f"nonfinite={int(tc[2])} runaway={int(tc[3])} (total dead="
          f"{int((~surv).sum())})", flush=True)

    # THE DETERMINISTIC GATE: survival>=0.8, window-mean inversion(-pgx)>=0.8,
    # rear-foot contact>=0.5, front-foot contact<0.5 (the task's handstand gate).
    g_surv = surv_frac >= 0.8
    g_inv = (nc > 0) and (inv_med >= 0.8)
    g_rear = (nc > 0) and (rear_med >= 0.5)
    g_front = (nc > 0) and (frontc_med < 0.5)
    passed = g_surv and g_inv and g_rear and g_front
    print(f"\n[man] GATE: survival>=0.8 [{surv_frac:.3f} {'OK' if g_surv else 'FAIL'}]  "
          f"inv>=0.8 [{inv_med:+.3f} {'OK' if g_inv else 'FAIL'}]  "
          f"rear>=0.5 [{rear_med:.3f} {'OK' if g_rear else 'FAIL'}]  "
          f"front<0.5 [{frontc_med:.3f} {'OK' if g_front else 'FAIL'}]", flush=True)
    print(f"[man] RESULT: {'*** GATE PASS -- REAL SUSTAINED HANDSTAND ***' if passed else '*** GATE FAIL ***'}",
          flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
