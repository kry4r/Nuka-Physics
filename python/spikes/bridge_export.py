"""POLICY->CO-RESIDENT INFERENCE BRIDGE SPIKE (GATE 1, Python side).

This is the GATE before a multi-week PPO standing campaign. PPO will train a
PyTorch NN in the BATCHED reduced-16-DOF world (Python, ``nuka.World``); the
demo runs that policy in the CO-RESIDENT full-51-DOF world (C++
``UnifiedCoResidentStepper``, NOT bound to Python). The deploy bridge is NOT
python-bind / NOT ONNX / NOT torch-at-deploy: it is flat f32 weights + a ~50-line
C++ forward pass. This script is the TRAIN side of that bridge:

  1. Roll out the hand-coded CoM/CoP + posture-PD balance law (reused from
     ``h1_balance_gate.py``) in the BATCHED reduced world, collecting a few
     thousand (obs, action) samples spanning the standing transient + steady
     state, under the SHARED obs/action contract (see below).
  2. Fit a tiny MLP (32 -> 64 -> 64 -> 10; tanh hidden, linear out) by plain
     supervised MSE regression obs->action (distillation, NOT RL).
  3. Export (a) the flat f32 weights (documented layout) + (b) K golden triples
     (raw shared-state 32, obs_py 32, action_py 10 == the MLP output) that the
     C++ side (test_h1_bridge_spike.cpp) re-derives obs from and re-runs the MLP
     on, asserting EXACT parity (1e-5) -- the bridge verdict.

The SCRIPT is the source of truth (committed); the emitted binaries under
``python/spikes/out/`` are regenerable + gitignored.

=============================================================================
SHARED OBS/ACTION CONTRACT (pinned EXACTLY; identical in Python & C++).

Both worlds index leg state by joint, in THIS exact order (10 legs):
  [left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle,
   right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle].
In the BATCHED reduced cook these are links 1..10 (cook order; h1_balance_gate
asserts blc==11 and uses q[1:11]); in the FULL co-resident cook they are
resolved BY NAME (the full cook is 46 links, DFS order -- NOT contiguous 1..10),
which is exactly why the C++ side resolves leg links by name (Piece A proves the
reduced 1..10 == the full by-name set are structurally identical).

OBS vector (dim = 32), in THIS component order:
  [0:4]   base orientation quaternion (qw,qx,qy,qz)        <- BASE_POSE[3:7]
  [4:10]  base spatial velocity (6), ROOT-LINK BODY frame, omega-first
          [wx,wy,wz, vx,vy,vz]                             <- LINK_VELOCITY[0]
  [10:12] base horizontal position RELATIVE to the foot-polygon center (x,y)
          = BASE_POSE[0:2] - (poly_cx, poly_cy)  (poly_cy = 0, symmetric foot)
  [12:22] leg joint positions (10), the contract leg order  <- JOINT_POSITION[1:11]
  [22:32] leg joint velocities (10), the contract leg order <- JOINT_VELOCITY[1:11]

ACTION vector (dim = 10): leg joint torques in the SAME leg order [1..10].
  Python writes TORQUE_INPUT[1:11]; C++ sets per-link torque on the by-name legs.

raw == obs here (obs IS these raw quantities; NO obs normalization is applied),
so the goldens dump raw and obs identically. The Gate-1a obs-builder teeth
therefore live in the C++ BITE (a deliberate leg-index permutation): the C++
obs-builder indexes leg slots through a leg-index array, and the BITE swaps two
entries in it.

CONTRACT CORRECTIONS (documented; both verified against the engine):
  * LINK_VELOCITY layout is omega-first [wx,wy,wz, vx,vy,vz] in BOTH worlds
    (python/nuka/__init__.py:93 + StepStand mapping live.link_velocity[root].v),
    so no permutation -- contract is correct.
  * The C++ host scalar-q is a PER-LINK array (articulation_state.cpp:321 pushes
    one q slot per link), so the deploy side reads host.q[leg_link] DIRECTLY by
    link index -- NOT host.q[DofIndexOf(link)] (DofIndexOf is the flat prefix-sum
    column space for the chain-J, a different index). The costand reference test
    uses st.q[l] by link; we match that.
  * poly_center is a SINGLE fixed scalar pinned identically on both sides:
    poly_cx = 0.0770 (the full-cook seat-time value, printed by the costand SEAT
    test), poly_cy = 0.0. In the BATCHED training rollout we compute the same
    horizontal-position-rel-center obs slot from BASE_POSE - (poly_cx, poly_cy).
=============================================================================

Run (from repo root):
  export PATH=/root/.nuka-toolchain-gcc14/bin:$PATH
  export CUDA_VISIBLE_DEVICES=0
  /root/miniconda3/envs/nuka-v03/bin/python python/spikes/bridge_export.py
"""

from __future__ import annotations

import pathlib
import struct

import numpy as np
import torch

import nuka

# Reuse the verified balance-law machinery + builders from the prior spikes.
import h1_balance_gate as bg  # noqa: E402
import h1_standing_pd_gate as pd  # noqa: E402

DT = bg.DT
G = bg.G

# The reduced training asset (regenerable via gen_h1_reduced_train.py). We prefer
# the EXACT-lump generated asset; if absent we fall back to building the guessed
# faithful asset in-place (same robot the balance gate stood) so the script is
# self-contained.
REPO = pathlib.Path(__file__).resolve().parents[2]
REDUCED_MJCF = (REPO /
                ".nuka-assets/newton_assets/unitree_h1/mjcf/h1_reduced_train.xml")

OUT_DIR = pathlib.Path(__file__).resolve().parent / "out"
WEIGHTS_BIN = OUT_DIR / "h1_bridge_mlp.bin"
GOLDENS_TXT = OUT_DIR / "h1_bridge_goldens.txt"

# --- the SHARED contract dimensions + the pinned poly center (see module docstr) --
OBS_DIM = 32
ACT_DIM = 10
H = 64                                   # hidden width.
POLY_CX = 0.0770                         # full-cook seat-time foot-polygon center x.
POLY_CY = 0.0                            # symmetric foot rectangle.

# The stance + gains the balance gate STOOD at (Kp_cop=320 clean within-torque).
COM_LUMP = (-0.02, 0.18, 30.62)
STANCE = (-0.40, 0.70, -0.30)            # hip_pitch, knee, ankle.
KP = {"hip_yaw": 150, "hip_roll": 200, "hip_pitch": 200, "knee": 300, "ankle": 0}
KD = {"hip_yaw": 8, "hip_roll": 12, "hip_pitch": 14, "knee": 18, "ankle": 0}
FOOT = (0.10, -0.10)
KP_COP, KD_COP, ANKLE_KD = 320.0, 50.0, 1.5
SETTLE = 15


def _read(w, field, ec, n):
    return torch.from_dlpack(w.buffer_view(field)).view(ec, n)


def collect_samples(dev, mjcf_path, ec=64, n_steps=1000):
    """Roll out the hand-coded balance law in the BATCHED reduced world and gather
    (raw32, obs32, action10) per env per step, under the shared contract.

    raw == obs (no normalization), so we return ONE 32-wide obs tensor + the
    10-wide action; the goldens dump raw==obs from the SAME array.
    """
    hp, kn, an = STANCE
    com_x_lump, com_z_lump, mass_lump = COM_LUMP
    toe_x, heel_x = FOOT
    qnom = pd._build_qnom(hp, kn, an)
    ground = pd.PELVIS_Z0 + pd._foot_z_under_pose(0.0, 0.0, 0.0)
    base_z = ground - pd._foot_z_under_pose(hp, kn, an, toe_x, heel_x)

    w = nuka.World.create_from_scene(dev, str(mjcf_path), ec, DT, 0, 1)  # torque.
    obs_steps, act_steps = [], []
    try:
        blc = w.base_link_count
        assert blc == 11, f"expected reduced blc==11, got {blc}"
        comc = bg.FaithfulCoM(blc, com_x_lump, com_z_lump, mass_lump, "cuda")

        pd._seat_pose(w, qnom, base_z, ec, blc)
        tq = _read(w, nuka.TORQUE_INPUT, ec, blc)
        tq[:] = 0.0
        nuka.sync()

        qnom_t = torch.tensor(qnom, dtype=torch.float32, device="cuda")
        tlim = torch.tensor(
            [bg.TORQUE_LIMIT[n.split("_", 1)[1]] for n in bg.LEG_JOINTS],
            dtype=torch.float32, device="cuda")
        kp_slot, kd_slot, is_ankle = [], [], []
        for n in bg.LEG_JOINTS:
            j = n.split("_", 1)[1]
            if j == "ankle":
                kp_slot.append(0.0); kd_slot.append(ANKLE_KD); is_ankle.append(1.0)
            else:
                kp_slot.append(KP[j]); kd_slot.append(KD[j]); is_ankle.append(0.0)
        kp_slot = torch.tensor(kp_slot, device="cuda")
        kd_slot = torch.tensor(kd_slot, device="cuda")
        ankle_mask = torch.tensor(is_ankle, device="cuda").bool()

        def control(prev_com, k):
            """Compute the hand-coded action AND assemble the contract obs."""
            lp = torch.from_dlpack(
                w.buffer_view(nuka.ARTICULATION_LINK_POSE)).view(ec, blc, 7)
            com = comc.com(lp)
            com_v = (com - prev_com) / DT if prev_com is not None else \
                torch.zeros_like(com)
            # poly center: PINNED constant (shared with C++), NOT the live ankle x.
            q = _read(w, nuka.JOINT_POSITION, ec, blc)
            qd = _read(w, nuka.JOINT_VELOCITY, ec, blc)
            qj = q[:, 1:1 + 10]
            qdj = qd[:, 1:1 + 10]
            tau = kp_slot.view(1, 10) * (qnom_t.view(1, 10) - qj) \
                - kd_slot.view(1, 10) * qdj
            # ANKLE CoP strategy uses world CoM x rel the LIVE poly center (the law
            # the balance gate stood); the OBS uses the PINNED poly center for the
            # base-horiz slot. (The MLP learns the proxy; this is the train-deploy
            # boundary the SOFT Gate 2 measures, not a bridge concern.)
            ankle_lx = lp[:, 5, 0]
            ankle_rx = lp[:, 10, 0]
            poly_cx_live = 0.5 * (ankle_lx + ankle_rx) + 0.5 * (toe_x + heel_x)
            ex = com[:, 0] - poly_cx_live
            evx = com_v[:, 0]
            if k >= SETTLE:
                tau_ankle = KP_COP * ex + KD_COP * evx
            else:
                tau_ankle = torch.zeros(ec, device="cuda")
            tau_full = tau.clone()
            tau_full[:, ankle_mask] = tau_ankle.unsqueeze(1) + tau_full[:, ankle_mask]
            tau_clamped = torch.clamp(tau_full, -tlim.view(1, 10), tlim.view(1, 10))
            return com, tau_clamped

        def assemble_obs():
            """Build the 32-wide contract obs for all envs from the live world."""
            bp = _read(w, nuka.BASE_POSE, ec, 7)                       # (ec,7)
            lv = torch.from_dlpack(
                w.buffer_view(nuka.LINK_VELOCITY)).view(ec, blc, 6)   # omega-first.
            q = _read(w, nuka.JOINT_POSITION, ec, blc)
            qd = _read(w, nuka.JOINT_VELOCITY, ec, blc)
            obs = torch.zeros(ec, OBS_DIM, device="cuda", dtype=torch.float32)
            obs[:, 0:4] = bp[:, 3:7]                                  # quat (w x y z).
            obs[:, 4:10] = lv[:, 0, :]                                # base spatial v.
            obs[:, 10] = bp[:, 0] - POLY_CX                           # base x rel center.
            obs[:, 11] = bp[:, 1] - POLY_CY                           # base y rel center.
            obs[:, 12:22] = q[:, 1:1 + 10]                            # leg qpos.
            obs[:, 22:32] = qd[:, 1:1 + 10]                          # leg qvel.
            return obs

        com_prev, _ = control(None, -1)
        nuka.sync()

        for k in range(n_steps):
            # OBS is assembled from the PRE-step state (the policy sees the current
            # state, then acts) -- the standard (s_t, a_t) pairing.
            obs = assemble_obs()
            com_prev, tau_cl = control(com_prev, k)
            tq = _read(w, nuka.TORQUE_INPUT, ec, blc)
            tq[:, 0] = 0.0
            tq[:, 1:1 + 10] = tau_cl
            # record (skip NaN envs/steps defensively).
            ok = torch.isfinite(obs).all(dim=1) & torch.isfinite(tau_cl).all(dim=1)
            if bool(ok.any()):
                obs_steps.append(obs[ok].detach().cpu().clone())
                act_steps.append(tau_cl[ok].detach().cpu().clone())
            nuka.sync()
            w.step()
            nuka.sync()
    finally:
        w.destroy()

    obs_all = torch.cat(obs_steps, dim=0).float()
    act_all = torch.cat(act_steps, dim=0).float()
    return obs_all, act_all


class TinyMLP(torch.nn.Module):
    """32 -> 64 -> 64 -> 10; tanh on the two hidden layers, linear output. This is
    the EXACT architecture the C++ tiny_mlp.hpp forward reimplements."""

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(OBS_DIM, H)
        self.fc2 = torch.nn.Linear(H, H)
        self.fc3 = torch.nn.Linear(H, ACT_DIM)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)


def fit_mlp(obs, act, epochs=1200, lr=1e-3, batch=4096, seed=0):
    """Plain supervised MSE regression obs->action (distillation).

    NO input standardization: the exported net consumes RAW obs directly. We
    deliberately do NOT standardize-then-fold-into-fc1, because that fold is f32-
    UNSTABLE -- several obs columns are near-constant at standing (quat w~1, base
    spatial vel~0, base-horiz-pos~0) with tiny std, so W1/sd becomes O(1e2-1e3)
    paired with a huge cancelling bias; the f32 forward then computes a small
    result as the difference of two large numbers, and torch-f32 vs naive-C++-f32
    round that cancellation differently (~5e-4 divergence) -- which would TRIP the
    1e-5 Gate-1b bar that exists precisely to catch f32/f64-slip broken exports.
    Training on RAW obs keeps all weights O(1), so torch-f32 and C++-f32 agree to
    ~1e-6. The obs is already O(1)-scale (angles ~0.4-0.7, quat ~1, small vels),
    so Adam converges fine; we bump epochs to compensate for the lost conditioning.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyMLP().to(dev)
    obs = obs.to(dev).float()
    act = act.to(dev).float()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = obs.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            pred = model(obs[idx])
            loss = torch.nn.functional.mse_loss(pred, act[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * idx.numel()
        if (ep + 1) % 200 == 0 or ep == 0:
            print(f"  epoch {ep + 1:4d}/{epochs}  mse={tot / n:.6f}", flush=True)

    # weight magnitudes (the fold-conditioning diagnostic): with NO fold these stay
    # O(1), which is why the C++ f32 forward matches to ~1e-6.
    with torch.no_grad():
        print(f"  max|W1|={model.fc1.weight.abs().max().item():.3f} "
              f"max|W2|={model.fc2.weight.abs().max().item():.3f} "
              f"max|W3|={model.fc3.weight.abs().max().item():.3f}", flush=True)

    # final regression RMS per joint (RAW-obs forward, the exported net).
    with torch.no_grad():
        pred = model(obs.float())
        rms = torch.sqrt(((pred - act) ** 2).mean(dim=0)).cpu().numpy()
    return model.cpu().float(), rms


def export_weights(model, path):
    """Flat f32 weights. LAYOUT (all row-major, written in THIS order):
        W1 [H, OBS_DIM]   (H*OBS_DIM floats, row-major: row r = output unit r)
        b1 [H]
        W2 [H, H]
        b2 [H]
        W3 [ACT_DIM, H]
        b3 [ACT_DIM]
    The forward is: h1 = tanh(W1 x + b1); h2 = tanh(W2 h1 + b2); y = W3 h2 + b3.
    torch Linear.weight is already [out, in] row-major, so we dump it verbatim."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blobs = []
    for layer in (model.fc1, model.fc2, model.fc3):
        blobs.append(layer.weight.detach().numpy().astype(np.float32).ravel())
        blobs.append(layer.bias.detach().numpy().astype(np.float32).ravel())
    flat = np.concatenate(blobs).astype(np.float32)
    expect = (H * OBS_DIM + H) + (H * H + H) + (ACT_DIM * H + ACT_DIM)
    assert flat.size == expect, (flat.size, expect)
    with open(path, "wb") as f:
        f.write(flat.tobytes())
    print(f"  wrote {flat.size} f32 weights -> {path}", flush=True)
    return flat


def export_goldens(model, obs, path, k=16):
    """K golden triples spread across the rollout (transient + steady), each:
        raw[32] obs[32] action_py[10]   (raw == obs; action_py == MLP forward)
    All cast to float32 BEFORE dumping so the C++ f32 forward is judged against an
    f32 reference (the 1e-5 bar is not fought by f64-vs-f32 slip)."""
    n = obs.shape[0]
    # spread: a few in the transient (first ~2% of samples), rest across steady.
    idx_transient = np.linspace(0, max(1, n // 50 - 1), 4, dtype=int)
    idx_steady = np.linspace(n // 50, n - 1, k - 4, dtype=int)
    idx = np.unique(np.concatenate([idx_transient, idx_steady]))[:k]
    sel = obs[idx].float()
    model = model.float()
    with torch.no_grad():
        act = model(sel).float()
    sel_np = sel.numpy().astype(np.float32)
    act_np = act.numpy().astype(np.float32)
    lines = [f"{len(idx)} {OBS_DIM} {ACT_DIM}"]
    for r in range(len(idx)):
        raw = " ".join(f"{v:.9e}" for v in sel_np[r])      # raw == obs.
        ob = " ".join(f"{v:.9e}" for v in sel_np[r])
        ac = " ".join(f"{v:.9e}" for v in act_np[r])
        lines.append(f"{raw} | {ob} | {ac}")
    path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {len(idx)} golden triples -> {path}", flush=True)
    return idx


def main():
    print("nuka:", nuka.__file__, flush=True)
    torch.manual_seed(0)

    # Pick the training asset: the EXACT-lump generated reduced asset if present.
    if REDUCED_MJCF.exists():
        mjcf_path = REDUCED_MJCF
        cleanup = False
        print(f"using generated reduced asset: {mjcf_path}", flush=True)
    else:
        mjcf_path = bg.H1_MJCF.parent / "h1_balance.xml"
        pd._make_faithful(mjcf_path, toe_x=FOOT[0], heel_x=FOOT[1],
                          com_z=COM_LUMP[1], mass=COM_LUMP[2],
                          com_x=COM_LUMP[0], inertia=0.8)
        cleanup = True
        print(f"reduced asset missing; built faithful in-place: {mjcf_path}",
              flush=True)

    try:
        with nuka.Device.create(0) as dev:
            print("\n=== collect (batched reduced world, hand-coded balance law) ===",
                  flush=True)
            obs, act = collect_samples(dev, mjcf_path, ec=64, n_steps=1000)
            print(f"  collected {obs.shape[0]} samples "
                  f"(obs {tuple(obs.shape)}, act {tuple(act.shape)})", flush=True)

            print("\n=== fit MLP (32->64->64->10, tanh, MSE distillation) ===",
                  flush=True)
            model, rms = fit_mlp(obs, act)
            print("  final per-joint regression RMS (N*m):", flush=True)
            for i, n in enumerate(bg.LEG_JOINTS):
                print(f"    {n:18s} {rms[i]:8.4f}", flush=True)
            print(f"  overall RMS = {float(np.sqrt((rms ** 2).mean())):.4f} N*m",
                  flush=True)
    finally:
        if cleanup and mjcf_path.exists():
            mjcf_path.unlink()

    print("\n=== export ===", flush=True)
    export_weights(model, WEIGHTS_BIN)
    export_goldens(model, obs, GOLDENS_TXT)
    print("\n=== bridge_export complete ===", flush=True)


if __name__ == "__main__":
    main()
