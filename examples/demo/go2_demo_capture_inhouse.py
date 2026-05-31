#!/usr/bin/env python3
"""[v0.3 demo] Capture a Go2 locomotion rollout driven by the IN-HOUSE,
from-scratch-PPO-trained, CONVERGED policy -- the artifact that UNIFIES exit
criteria #3 and #6.

HONESTY FRAMING (read before trusting the video)
------------------------------------------------
This produces the v0.3 demo video from the **IN-HOUSE, from-scratch-trained Go2
policy** (`out/go2_policy/go2_walk_randomcmd_ep250.pth`, an rl_games PPO
checkpoint, random-command curriculum, exit-#3 CONVERGED). UNLIKE the original
`go2_demo_capture.py` -- which REPLAYS the EXTERNAL unitree_rl_gym PR#62
`motion.pt` and therefore satisfies only exit-#6 ("a locomotion video") while
explicitly NOT satisfying exit-#3 ("from-scratch PPO convergence") -- this script
drives the self-trained converged policy. So this single video UNIFIES:
  * exit #3 -- the from-scratch PPO policy that converged to a command-tracking gait
  * exit #6 -- a Go2 locomotion video
The rendered motion is the in-house policy, and the per-env command spread below
SHOWCASES command-conditioning, the in-house policy's distinguishing achievement
(the external PR#62 replay could not condition: same-cmd envs were clones).

WHAT IS REUSED (provably identical pipeline)
--------------------------------------------
The obs/world/permutation/drive plumbing is IMPORTED unchanged from
`examples/sim_val/go2_policy_drive.py` (as `G`): `make_world`, `warm_start`,
`build_obs`, `write_targets_urdf`, `DEFAULT_ANGLES`, `ACTION_SCALE`, `Ladder`,
`d1_permutation`, `projected_gravity_body`, `DT`, `DECIMATION`, `GO2_BLC`. The
in-house policy's TRAINING obs (`python/nuka/tasks/go2_obs.py`, 48-dim) is
bit-exact to `G.build_obs(...)` (same legged_gym order/scales), so the ONLY thing
swapped vs the original capture is the policy FORWARD: instead of the external
TorchScript `motion.pt`, we run the in-house checkpoint's deterministic-mu MLP
(rl_games RunningMeanStd obs-norm -> 3x elu MLP -> mu head), clamped to the
trained action space (+-1, NOT +-100).

GATE
----
A per-env command-TRACKING gate runs BEFORE any render:
  * envs with cmd_vx >= 0.2 must advance forward (net world dx > 0);
  * standing / low-cmd envs (cmd_vx < 0.2) must NOT run away (|dx| bounded);
  * ALL envs stay upright (final tilt < 35 deg) with no NaN;
  * command-conditioning monotonicity: vx_world broadly RISES with cmd_vx and the
    fastest-cmd envs clearly outpace the standing env.
If the capture does not track-and-stay-upright, we ABORT and do NOT render.

Run:
  export CUDA_VISIBLE_DEVICES=0
  /root/miniconda3/envs/nuka-v03/bin/python \
      examples/demo/go2_demo_capture_inhouse.py \
      --envs 16 --seconds 6 --out out/go2_demo/go2_rollout.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

# Import the PROVEN policy-driving plumbing. Capturing via import (not a copy)
# makes the rendered motion provably the same validated obs/world/drive pipeline;
# only the policy forward differs (in-house checkpoint instead of motion.pt).
_SIM_VAL = os.path.join(os.path.dirname(__file__), "..", "sim_val")
sys.path.insert(0, os.path.abspath(_SIM_VAL))

import nuka  # noqa: E402
import go2_policy_drive as G  # noqa: E402

INHOUSE_CKPT = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "out", "go2_policy", "go2_walk_randomcmd_ep250.pth",
)
# rl_games trained the policy with the action space declared as +-1 (the
# ACTION_SPACE_LIMIT=1.0 fix). The deterministic-mu output is clamped to that band
# -- NOT the loose +-100 SAFETY clip the external replay used.
ACTION_SPACE_LIMIT = 1.0


# ---------------------------------------------------------------------------
# In-house policy: rl_games RunningMeanStd obs-norm + deterministic-mu MLP.
# ---------------------------------------------------------------------------
class InHousePolicy:
    """Deterministic-mu forward of the in-house rl_games PPO checkpoint.

    Recipe (the controller's re-verified recipe; produces a command-tracking walk):
        norm = clamp((obs - running_mean) / sqrt(running_var + 1e-5), -5, 5)
        h = elu(L0(norm)); h = elu(L2(h)); h = elu(L4(h))
        mu = mu_head(h)
        action = clamp(mu, -1, 1)
    `sigma` (log-std) is NOT used -- we drive the deterministic mean.
    """

    def __init__(self, ckpt_path: str, device: str = "cpu"):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        m = ck["model"]
        # strip the torch.compile `_orig_mod.` prefix on every key.
        sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
              for k, v in m.items()}

        def need(name, shape):
            t = sd[name]
            assert tuple(t.shape) == shape, (
                f"{name}: got {tuple(t.shape)}, expected {shape}")
            return t.float()

        # --- obs normalizer (rl_games RunningMeanStd.forward) ---
        self.run_mean = need("running_mean_std.running_mean", (48,)).to(device)
        self.run_var = need("running_mean_std.running_var", (48,)).to(device)
        # count is a scalar bookkeeping value; not needed for the forward.
        _ = sd["running_mean_std.count"]

        # --- MLP: 3 hidden layers (48->512->256->128) + mu head (128->12) ---
        w0 = need("a2c_network.actor_mlp.0.weight", (512, 48))
        b0 = need("a2c_network.actor_mlp.0.bias", (512,))
        w2 = need("a2c_network.actor_mlp.2.weight", (256, 512))
        b2 = need("a2c_network.actor_mlp.2.bias", (256,))
        w4 = need("a2c_network.actor_mlp.4.weight", (128, 256))
        b4 = need("a2c_network.actor_mlp.4.bias", (128,))
        wmu = need("a2c_network.mu.weight", (12, 128))
        bmu = need("a2c_network.mu.bias", (12,))
        _ = need("a2c_network.sigma", (12,))  # log-std: deliberately unused

        self.L0 = nn.Linear(48, 512); self._load(self.L0, w0, b0)
        self.L2 = nn.Linear(512, 256); self._load(self.L2, w2, b2)
        self.L4 = nn.Linear(256, 128); self._load(self.L4, w4, b4)
        self.mu = nn.Linear(128, 12); self._load(self.mu, wmu, bmu)
        for lyr in (self.L0, self.L2, self.L4, self.mu):
            lyr.to(device).eval()
        self.act = nn.ELU()
        self.device = device
        self.epoch = int(ck.get("epoch", -1))
        self.frame = int(ck.get("frame", -1))
        self.reward = float(ck.get("last_mean_rewards", float("nan")))

    @staticmethod
    def _load(layer: nn.Linear, w: torch.Tensor, b: torch.Tensor) -> None:
        # nn.Linear stores weight as (out,in) -- the checkpoint already is (out,in).
        assert layer.weight.shape == w.shape, (layer.weight.shape, w.shape)
        assert layer.bias.shape == b.shape, (layer.bias.shape, b.shape)
        with torch.no_grad():
            layer.weight.copy_(w)
            layer.bias.copy_(b)

    @torch.no_grad()
    def forward(self, obs_np: np.ndarray) -> np.ndarray:
        """(N,48) float32 obs -> (N,12) float32 action = clamp(mu, -1, 1)."""
        obs = torch.from_numpy(np.ascontiguousarray(obs_np, np.float32)).to(self.device)
        # rl_games RunningMeanStd.forward (eval / norm_only):
        norm = (obs - self.run_mean) / torch.sqrt(self.run_var + 1e-5)
        norm = norm.clamp(-5.0, 5.0)
        h = self.act(self.L0(norm))
        h = self.act(self.L2(h))
        h = self.act(self.L4(h))
        mu = self.mu(h)                              # (N,12) action mean
        action = mu.clamp(-ACTION_SPACE_LIMIT, ACTION_SPACE_LIMIT)
        return action.cpu().numpy().astype(np.float32)


def read_all_envs_urdf(w, urdf_from_nuka_slot, env_count):
    """Vectorized read of ALL envs' state in URDF order. Returns
    (q_urdf(N,12), qd_urdf(N,12), base_lin_vel(N,3), base_ang_vel(N,3),
     proj_grav(N,3), base_xyz(N,3), base_quat_wxyz(N,4), tilt_deg(N)).
    Copied verbatim from go2_demo_capture.py (the proven batched read)."""
    q = torch.from_dlpack(w.buffer_view(nuka.JOINT_POSITION))
    qd = torch.from_dlpack(w.buffer_view(nuka.JOINT_VELOCITY))
    pose = torch.from_dlpack(w.buffer_view(nuka.ARTICULATION_LINK_POSE))
    vel = torch.from_dlpack(w.buffer_view(nuka.LINK_VELOCITY))

    q_slots = q[:env_count, 1:G.GO2_BLC].detach().cpu().numpy()      # (N,12) nuka order
    qd_slots = qd[:env_count, 1:G.GO2_BLC].detach().cpu().numpy()
    q_urdf = q_slots[:, urdf_from_nuka_slot]                          # -> URDF order
    qd_urdf = qd_slots[:, urdf_from_nuka_slot]

    base = pose[:env_count, 0, :].detach().cpu().numpy()             # (N,7)
    base_xyz = base[:, 0:3]
    base_quat = base[:, 3:7]                                          # wxyz
    pg = np.stack([G.projected_gravity_body(base_quat[i]) for i in range(env_count)])
    tilt = np.degrees(np.arccos(np.clip(-pg[:, 2], -1.0, 1.0)))

    root_vel = vel[:env_count, 0, :].detach().cpu().numpy()          # (N,6) omega-first
    base_ang_vel = root_vel[:, 0:3]
    base_lin_vel = root_vel[:, 3:6]
    return q_urdf, qd_urdf, base_lin_vel, base_ang_vel, pg, base_xyz, base_quat, tilt


def capture(dev, policy, urdf_from_nuka_slot, env_count, policy_steps, cmds):
    """Drive the in-house policy BATCHED across env_count envs (per-env command
    `cmds`, shape (N,3)), capturing per-frame link poses. Returns a dict for
    np.savez. SAME npz schema the renderer expects."""
    print("\n" + "-" * 78)
    print(f"CAPTURE: batched {env_count}-env IN-HOUSE policy rollout, {policy_steps} "
          f"steps @ 50 Hz ({policy_steps * G.DT * G.DECIMATION:.1f} s sim)")
    print(f"  per-env cmd vx range [{cmds[:,0].min():.2f},{cmds[:,0].max():.2f}] "
          f"(forward-speed ramp; env0 stands, fastest env walks)")
    print("-" * 78, flush=True)

    frames = []          # list of (N,13,7) link-pose snapshots
    base_track = []       # list of (N,3) base xyz per frame
    cmds = np.asarray(cmds, np.float32)

    with G.make_world(dev, env_count) as w:
        G.warm_start(w, urdf_from_nuka_slot)
        pose_buf = torch.from_dlpack(w.buffer_view(nuka.ARTICULATION_LINK_POSE))

        (_, _, _, _, _, base_xyz0, _, tilt0) = read_all_envs_urdf(
            w, urdf_from_nuka_slot, env_count)
        base_xy0 = base_xyz0[:, 0:2].copy()
        last_action = np.zeros((env_count, 12), np.float32)

        nan_seen = False
        for k in range(policy_steps):
            (q_u, qd_u, blv, bav, pg, base_xyz, base_quat, tilt) = read_all_envs_urdf(
                w, urdf_from_nuka_slot, env_count)
            obs = G.build_obs(blv, bav, pg, cmds, q_u, qd_u, last_action)
            if not np.isfinite(obs).all():
                nan_seen = True
                print(f"  [abort] non-finite obs at step {k}", flush=True)
                break
            action = policy.forward(obs)                              # (N,12) batched
            # action space is +-1 (trained ACTION_SPACE_LIMIT); the policy already
            # clamps to +-1. Feed the CLAMPED action back as obs[36:48] last_action
            # (the band the policy trained in).
            action = np.clip(action, -ACTION_SPACE_LIMIT, ACTION_SPACE_LIMIT)
            last_action = action.copy()
            target_urdf = G.DEFAULT_ANGLES[None, :] + G.ACTION_SCALE * action  # (N,12)
            G.write_targets_urdf(w, urdf_from_nuka_slot, target_urdf)
            w.step_n(G.DECIMATION)

            nuka.sync()
            frames.append(pose_buf[:env_count].detach().cpu().numpy().copy())
            base_track.append(
                pose_buf[:env_count, 0, 0:3].detach().cpu().numpy().copy())

            if (k % 25) == 0 or k == policy_steps - 1:
                dx = float((base_track[-1][:, 0] - base_xy0[:, 0]).mean())
                print(f"  t={k*G.DT*G.DECIMATION:5.2f}s step{k:4d}  "
                      f"mean base_z={base_xyz[:,2].mean():.3f}  "
                      f"mean tilt={tilt.mean():5.1f}deg  "
                      f"mean net dx={dx:+.3f}m", flush=True)

        (_, _, _, _, _, base_xyzf, _, tiltf) = read_all_envs_urdf(
            w, urdf_from_nuka_slot, env_count)

    frames = np.stack(frames) if frames else np.zeros((1, env_count, 13, 7), np.float32)
    base_track = np.stack(base_track) if base_track else np.zeros((1, env_count, 3))
    sim_time = max(len(frames), 1) * G.DT * G.DECIMATION
    dx_world = base_xyzf[:, 0] - base_xy0[:, 0]      # (N,) per-env net world-x advance
    dy_world = base_xyzf[:, 1] - base_xy0[:, 1]
    vx_world = dx_world / sim_time

    return dict(
        frames=frames.astype(np.float32),       # (T,N,13,7) world link pose [xyz,wxyz]
        base_track=base_track.astype(np.float32),
        cmds=cmds, env_count=env_count, sim_time=sim_time,
        dt=G.DT, decimation=G.DECIMATION, policy_steps=len(frames),
        dx_world=dx_world.astype(np.float32), dy_world=dy_world.astype(np.float32),
        vx_world=vx_world.astype(np.float32),
        tilt0=tilt0.astype(np.float32), tilt_final=tiltf.astype(np.float32),
        nan=nan_seen,
    )


def make_cmds(env_count):
    """Per-env forward-speed ramp: vx = linspace(0.0, 1.0, env_count); vy=0, wyaw=0.
    env0 stands still (cmd 0.0), the last env fast-walks (cmd 1.0). This is visually
    clean (all forward or standing -- no backward) AND demonstrates command-
    conditioning, the in-house policy's distinguishing achievement. The controller's
    eval confirmed this regime tracks (cmd 0.0 -> vx_body ~0.00 stand-still, 0.5 ->
    ~0.46, 1.0 -> ~1.03), 0 terminations."""
    vx = np.linspace(0.0, 1.0, env_count, dtype=np.float32)
    cmds = np.zeros((env_count, 3), np.float32)
    cmds[:, 0] = vx
    return cmds


# command threshold above which an env is expected to WALK forward; below it the
# env is expected to (roughly) stand. linspace(0,1,16): envs 0..2 are < 0.2.
CMD_WALK_THRESHOLD = 0.2
# upper bound on |net dx| for a standing/low-cmd env (catches a runaway/bolt or a
# fall-and-slide); generous enough to accept the legitimate slow creep of a small
# but nonzero command over the rollout.
STAND_DX_ABS_MAX = 1.2


def track_gate(cap):
    """Per-env command-TRACKING gate. Returns (passed: bool, env_ok: (N,) bool,
    reasons: dict). Prints the per-env table cmd_vx | vx_world | tilt_final."""
    cmds = cap["cmds"]
    dx = cap["dx_world"]
    vx = cap["vx_world"]
    tilt0 = cap["tilt0"]
    tiltf = cap["tilt_final"]
    N = cap["env_count"]
    cmd_vx = cmds[:, 0]
    sim_time = cap["sim_time"]

    is_walk_cmd = cmd_vx >= CMD_WALK_THRESHOLD
    is_stand_cmd = ~is_walk_cmd

    upright = tiltf < 35.0
    nan_ok = not cap["nan"]
    finite = np.isfinite(dx).all() and np.isfinite(vx).all() and np.isfinite(tiltf).all()

    # per-env pass: walking envs advance forward; standing envs don't run away;
    # all envs upright.
    fwd_ok = np.where(is_walk_cmd, dx > 0.0, np.abs(dx) < STAND_DX_ABS_MAX)
    env_ok = fwd_ok & upright

    # command-conditioning monotonicity (the discriminating proof): the fastest-cmd
    # envs must clearly OUTPACE the standing env, and vx_world must broadly rise with
    # cmd_vx (Spearman-like: positive correlation; fast group >> stand group).
    stand_vx_mean = float(vx[is_stand_cmd].mean()) if is_stand_cmd.any() else float(vx[0])
    top_k = max(1, N // 4)
    order = np.argsort(cmd_vx)
    fast_idx = order[-top_k:]
    fast_vx_mean = float(vx[fast_idx].mean())
    # Pearson correlation of cmd_vx vs vx_world across envs.
    if np.std(cmd_vx) > 1e-9 and np.std(vx) > 1e-9:
        corr = float(np.corrcoef(cmd_vx, vx)[0, 1])
    else:
        corr = float("nan")
    conditioning_ok = (corr > 0.7) and (fast_vx_mean - stand_vx_mean > 0.3)

    passed = bool(nan_ok and finite and upright.all()
                  and env_ok.all() and conditioning_ok)

    print("\n" + "=" * 78)
    print("PER-ENV COMMAND-TRACKING TABLE (in-house policy)")
    print("=" * 78)
    print(f"  {'env':>3} {'cmd_vx':>7} {'vx_world':>9} {'net_dx(m)':>10} "
          f"{'tilt0':>6} {'tiltf':>6}  {'class':<6} {'ok':>3}")
    for i in range(N):
        cls = "WALK" if is_walk_cmd[i] else "STAND"
        print(f"  {i:>3} {cmd_vx[i]:>7.3f} {vx[i]:>9.4f} {dx[i]:>10.3f} "
              f"{tilt0[i]:>6.1f} {tiltf[i]:>6.1f}  {cls:<6} "
              f"{'OK' if env_ok[i] else 'XX':>3}")
    print("-" * 78)
    print(f"  sim_time={sim_time:.2f}s   walk_cmd envs (cmd_vx>={CMD_WALK_THRESHOLD}): "
          f"{int(is_walk_cmd.sum())}   stand_cmd envs: {int(is_stand_cmd.sum())}")
    print(f"  CONDITIONING: corr(cmd_vx, vx_world)={corr:+.3f} (need >0.70); "
          f"stand_vx_mean={stand_vx_mean:+.4f}  fast_vx_mean={fast_vx_mean:+.4f} "
          f"(fast-stand={fast_vx_mean-stand_vx_mean:+.4f}, need >0.30)")
    print(f"  upright all (<35deg): {bool(upright.all())} "
          f"(max tiltf={float(tiltf.max()):.1f}deg)   NaN: {cap['nan']}")
    walk_envs = np.where(is_walk_cmd)[0]
    bad_walk = walk_envs[dx[is_walk_cmd] <= 0.0]
    stand_envs = np.where(is_stand_cmd)[0]
    bad_stand = stand_envs[np.abs(dx[is_stand_cmd]) >= STAND_DX_ABS_MAX]
    if bad_walk.size:
        print(f"  [FAIL] walk-cmd envs that did NOT advance forward: {bad_walk.tolist()}")
    if bad_stand.size:
        print(f"  [FAIL] stand-cmd envs that ran away (|dx|>={STAND_DX_ABS_MAX}): "
              f"{bad_stand.tolist()}")
    if not conditioning_ok:
        print(f"  [FAIL] command-conditioning not demonstrated "
              f"(corr or fast-vs-stand spread too small)")
    flagged = np.where(~env_ok)[0]
    print(f"  TRACK GATE: {'PASS' if passed else 'FAIL'}  "
          f"(flagged envs: {flagged.tolist()})")
    print("=" * 78, flush=True)

    return passed, env_ok, dict(corr=corr, stand_vx_mean=stand_vx_mean,
                                fast_vx_mean=fast_vx_mean)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--out", default="out/go2_demo/go2_rollout.npz")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    print("=" * 78)
    print("[v0.3 demo] Capturing the IN-HOUSE from-scratch-PPO converged Go2 policy")
    print("UNIFIES exit #3 (PPO convergence) + exit #6 (locomotion video).")
    print(f"  ckpt: {os.path.abspath(INHOUSE_CKPT)}")
    print("  UNLIKE the original capture, which replays the EXTERNAL PR#62 motion.pt.")
    print("=" * 78, flush=True)

    policy = InHousePolicy(os.path.abspath(INHOUSE_CKPT), device="cpu")
    print(f"  loaded in-house ckpt: epoch={policy.epoch} frame={policy.frame} "
          f"last_mean_rewards={policy.reward:.3f}", flush=True)

    dev = nuka.Device.create(0)
    try:
        # D1: discover the structural joint permutation (the proven path).
        ladder = G.Ladder()
        perm, perm_valid = G.d1_permutation(ladder, dev)
        if not perm_valid:
            print("[ABORT] joint permutation invalid -> refusing to capture a "
                  "scrambled-joint rollout.", flush=True)
            return 1

        policy_steps = int(round(args.seconds / (G.DT * G.DECIMATION)))
        cmds = make_cmds(args.envs)
        cap = capture(dev, policy, perm, args.envs, policy_steps, cmds)
    finally:
        dev.close()

    # ---- TRACK GATE: refuse to ship a non-tracking / fallen rollout. ----
    passed, env_ok, _ = track_gate(cap)
    if not passed:
        print("[ABORT] capture did not produce a clean command-tracking walk; not "
              "writing an npz for rendering. Investigate before rendering.", flush=True)
        return 1

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **cap)
    sz = os.path.getsize(out_path)
    print(f"\nWROTE {out_path}  ({sz/1e6:.2f} MB)")
    print("Next: render with examples/demo/go2_demo_render.py, encode with ffmpeg "
          "(see render_video.sh tail).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
