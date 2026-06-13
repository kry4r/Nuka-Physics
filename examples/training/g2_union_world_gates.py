#!/usr/bin/env python
"""G2 gate harness -- nanobind exposure of the H1 WHOLE-BODY UNION world.

Spec: docs/specs/2026-06-10-h1-whole-body-rl-grasp-spec.md §G2. The world under
test is ``nuka.UnionWorld`` -- the C-ABI/nanobind exposure of the G1d
BatchedUnifiedWorld UNION scene (floating-base 51-DOF whole-body H1 + 10.9 cm
cup settled in the curled right hand + static ground under the feet + static
table under the cup-proxy), whose BatchedSceneTemplate is now COOKED from the
AUTHORED examples/scenes/h1_cup_table.nks (scene::nks::Load ->
scene::cook::CookSceneToUnionTemplate, which READS the baked settled IC + grasp
config -- the M7 replacement for the deleted 1026-line h1_union_scene_factory's
cook -> stance -> seat -> curl -> placement-search-at-seat-FK -> 200-step
oracle-settle authoring).

GATES (hard asserts, runnable headless from the repo root):
  gate1 : python N=4 rollout matches the C++ reference trajectory BYTE-EXACTLY.
          The C++ recorder (tests target ``nuka_h1_union_pyref_test``) drives
          the SAME C-ABI world through a closed-loop lift choreography and
          records the per-step action bytes + the per-step C-ABI obs export to
          .g2_logs/g2_pyref_n4.bin; python replays the recorded action bytes
          through nuka.UnionWorld on the identical template and compares EVERY
          exported field byte-for-byte, every step. Both sides construct through
          the same libnuka C-ABI cook of the same .nks (no cross-compile seam).
  bite  : the BITE smoke rides gate1's recording: replay the SAME bytes but
          with the 12 grip columns ZEROED from the CLOSE phase on -> the cup
          must fall (vs the byte-exact replay where it is held). Proves the
          python action seam drives the real physics.
  gate2 : seeded reset determinism: same seed -> byte-same post-reset obs
          (q/qdot/base/cup; and the FULL obs after 5 follow-up steps);
          different seeds -> different cup pose.
  d1    : two fresh worlds, the same seeded 300-step random-action sequence ->
          byte-identical exports at checkpoints (full-field).
  gate3 : 10k random-action steps at N=4 (torques uniform within the per-joint
          limits, the 6 dead base columns masked to 0 by convention) -> ZERO
          NaN/inf in any exported field. Plus a reset-mixed finite smoke.

OBS LAYOUT (the documented contract; see UnionWorld.export_obs docstring):
  q                    (n, 51) f32   flat-DOF joint positions, prefix-sum order;
                                     cols 0..5 = the floating-base columns (the
                                     root's scalar-q slot -- constant filler).
  qdot                 (n, 51) f32   cols 0..5 = base spatial velocity, 6..50 =
                                     joint velocities.
  base_pose            (n, 7)  f32   [px,py,pz, qw,qx,qy,qz]
  base_vel             (n, 6)  f32   root-link spatial velocity (== qdot[:, :6])
  cup_pose             (n, 7)  f32   [px,py,pz, qw,qx,qy,qz]
  cup_vel              (n, 6)  f32   [vx,vy,vz, wx,wy,wz]
  fingertip_world_pos  (n, 30, 3) f32
  finger_normal_impulse(n, 30) f32   per-fingertip normal-row impulse (>= 0)
  last_action          (n, 51) f32   the last set_actions bytes (zeros before)
  foot_rows            (n,) u32      # foot<->ground NORMAL rows last step
  foot_impulse         (n,) f32      Σλ over foot normal rows (cast from double)
  table_rows           (n,) u32      # cup<->table rows last step
  table_impulse        (n,) f32      Σλ*j_z over table rows (cast from double)
  finger_contacts      (n,) u32      # fingertip<->cup manifold points

Run (from the repo root, nuka-v03 env, GPU lock held by the caller):
  python examples/training/g2_union_world_gates.py --gates all
"""
from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
os.chdir(_REPO_ROOT)  # the cook's union_scene_constants.hpp defaults are repo-relative.

import nuka  # noqa: E402

H1_MJCF = os.path.join(_REPO_ROOT, ".nuka-assets", "newton_assets", "unitree_h1",
                       "mjcf", "h1_with_hand.xml")
CUP_USDA = os.path.join(_REPO_ROOT, ".nuka-assets", "newton_assets",
                        "manipulation_objects", "cup", "model.usda")
REF_BIN = os.path.join(_REPO_ROOT, ".g2_logs", "g2_pyref_n4.bin")
RECORDER = os.path.join(_REPO_ROOT, "build-cuda128", "tests",
                        "nuka_h1_union_pyref_test")

MAGIC = 0x47325546  # 'G2UF'

# The exported-field names in the RECORDED order (floats then uints).
FLOAT_FIELDS = ("q", "qdot", "base_pose", "base_vel", "cup_pose", "cup_vel",
                "fingertip_world_pos", "finger_normal_impulse",
                "foot_impulse", "table_impulse")
UINT_FIELDS = ("foot_rows", "table_rows", "finger_contacts")


def assets_present() -> bool:
    return os.path.exists(H1_MJCF) and os.path.exists(CUP_USDA)


def make_world(dev, n: int):
    # Defaults: empty paths -> the cook's repo-relative union_scene_constants.hpp
    # defaults (the .nks + the assets it imports); 0 dt / 0 gravity -> the engine
    # constants (1/240, -9.81) -- the SAME sentinel convention the C++ recorder
    # uses, so the cooked templates are byte-identical.
    return nuka.UnionWorld.create(dev, env_count=n)


def flat_fields(w, obs: dict) -> dict:
    """Flatten the obs dict to 1-D arrays in the recorded field order."""
    out = {}
    for k in FLOAT_FIELDS + UINT_FIELDS:
        out[k] = np.ascontiguousarray(obs[k]).reshape(-1)
    return out


# ---------------------------------------------------------------------------
# The reference-recording file (.g2_logs/g2_pyref_n4.bin) reader.
# ---------------------------------------------------------------------------
class RefRecording:
    def __init__(self, path: str):
        with open(path, "rb") as f:
            raw = f.read()
        off = 0
        (magic, version, self.n, self.steps, self.dof, self.nf,
         n_events, _reserved) = struct.unpack_from("<8I", raw, off)
        off += 32
        (self.dt, self.gravity_z) = struct.unpack_from("<2f", raw, off)
        off += 8
        assert magic == MAGIC, f"bad magic 0x{magic:08x}"
        assert version == 1, f"unknown version {version}"
        self.events = []
        for _ in range(n_events):
            step, env, enabled = struct.unpack_from("<3I", raw, off)
            off += 12
            self.events.append((step, env, enabled))
        n, dof, nf = self.n, self.dof, self.nf
        self.float_lens = {
            "q": n * dof, "qdot": n * dof, "base_pose": n * 7, "base_vel": n * 6,
            "cup_pose": n * 7, "cup_vel": n * 6, "fingertip_world_pos": n * nf * 3,
            "finger_normal_impulse": n * nf, "foot_impulse": n, "table_impulse": n,
        }
        self.uint_lens = {"foot_rows": n, "table_rows": n, "finger_contacts": n}
        fl = sum(self.float_lens.values())
        ul = sum(self.uint_lens.values())
        per_step = 4 * (n * dof + fl + ul)
        avail = len(raw) - off
        assert avail == self.steps * per_step, (
            f"file size mismatch: {avail} != {self.steps}*{per_step}")
        self.actions = []
        self.obs = []
        for _ in range(self.steps):
            a = np.frombuffer(raw, dtype=np.float32, count=n * dof, offset=off)
            off += 4 * n * dof
            rec = {}
            for k in FLOAT_FIELDS:
                cnt = self.float_lens[k]
                rec[k] = np.frombuffer(raw, dtype=np.float32, count=cnt, offset=off)
                off += 4 * cnt
            for k in UINT_FIELDS:
                cnt = self.uint_lens[k]
                rec[k] = np.frombuffer(raw, dtype=np.uint32, count=cnt, offset=off)
                off += 4 * cnt
            self.actions.append(a)
            self.obs.append(rec)


def run_recorder():
    os.makedirs(os.path.dirname(REF_BIN), exist_ok=True)
    assert os.path.exists(RECORDER), f"recorder binary missing: {RECORDER}"
    print(f"[gate1] running the C++ recorder: {RECORDER}")
    r = subprocess.run(
        [RECORDER, "--gtest_filter=H1UnionPyRef.RecordReferenceTrajectory"],
        cwd=_REPO_ROOT, capture_output=True, text=True, timeout=1800)
    sys.stdout.write(r.stdout[-4000:])
    assert r.returncode == 0, f"recorder failed (rc={r.returncode}):\n{r.stderr[-4000:]}"
    assert os.path.exists(REF_BIN), "recorder did not produce the reference file"


def apply_events(w, ref: RefRecording, step: int):
    for (s, env, enabled) in ref.events:
        if s == step:
            w.set_table_enabled(int(env) if env != 0xFFFFFFFF else -1,
                                bool(enabled))


# ---------------------------------------------------------------------------
# gate 1: python N=4 replay == the C++ reference trajectory, byte-exact.
# ---------------------------------------------------------------------------
def gate1(dev, ref: RefRecording):
    n = ref.n
    w = make_world(dev, n)
    assert w.action_dim == ref.dof, (w.action_dim, ref.dof)
    assert w.num_fingertips == ref.nf
    mismatches = 0
    t0 = time.perf_counter()
    for s in range(ref.steps):
        apply_events(w, ref, s)
        w.step_with_actions(ref.actions[s])
        raw = w.export_obs()
        obs = flat_fields(w, raw)
        for k in FLOAT_FIELDS + UINT_FIELDS:
            got, want = obs[k], ref.obs[s][k]
            if not np.array_equal(got.view(np.uint8), want.view(np.uint8)):
                bad = int(np.sum(got.view(np.uint32) != want.view(np.uint32)))
                md = float(np.max(np.abs(got.astype(np.float64)
                                         - want.astype(np.float64))))
                print(f"[gate1] MISMATCH step {s} field {k}: {bad} words, "
                      f"max|d|={md:.3e}")
                mismatches += 1
        # last_action must echo the injected bytes exactly.
        la = np.ascontiguousarray(raw["last_action"]).reshape(-1)
        assert np.array_equal(la, ref.actions[s]), f"last_action echo broke @ {s}"
        if mismatches > 10:
            break
    wall = time.perf_counter() - t0
    final = ref.obs[ref.steps - 1]
    print(f"[gate1] replayed {ref.steps} steps N={n} in {wall:.1f}s; "
          f"mismatching field-blocks={mismatches}")
    print(f"[gate1] final cup_z per env: "
          f"{[round(float(final['cup_pose'][e*7+2]), 4) for e in range(n)]}")
    assert mismatches == 0, "python rollout is NOT byte-exact vs the C++ reference"
    w.destroy()
    print("[gate1] PASS: byte-exact, every field, every step")


# ---------------------------------------------------------------------------
# BITE: same recorded bytes, grip columns zeroed from the CLOSE phase -> falls.
# ---------------------------------------------------------------------------
def bite(dev, ref: RefRecording, close_start: int = 100):
    n = ref.n
    w = make_world(dev, n)
    grip = np.asarray(w.grip_dofs(), dtype=np.int64)
    assert grip.size == 12, f"expected 12 grip columns, got {grip.size}"
    for s in range(ref.steps):
        apply_events(w, ref, s)
        a = np.array(ref.actions[s], copy=True).reshape(n, ref.dof)
        if s >= close_start:
            a[:, grip] = 0.0
        w.step_with_actions(a.reshape(-1))
    obs = w.export_obs()
    cup_z = np.ascontiguousarray(obs["cup_pose"])[:, 2]
    ref_z = np.array([ref.obs[ref.steps - 1]["cup_pose"][e * 7 + 2]
                      for e in range(n)])
    drop0 = float(ref_z[0] - cup_z[0])
    print(f"[bite] grip cols zeroed from step {close_start}: "
          f"env0 cup_z {float(cup_z[0]):.3f} vs held {float(ref_z[0]):.3f} "
          f"(fell {drop0:.3f} m); all: {np.round(cup_z, 3).tolist()}")
    assert drop0 > 0.5, ("BITE FAILED: zeroing the grip columns did not drop the "
                         "cup -- the python action seam is not driving the grip")
    w.destroy()
    print("[bite] PASS: the python action seam drives the real grip physics")


# ---------------------------------------------------------------------------
# gate 2: seeded reset determinism.
# ---------------------------------------------------------------------------
RESET_FIELDS = ("q", "qdot", "base_pose", "base_vel", "cup_pose", "cup_vel")


def gate2(dev, n: int = 4):
    w = make_world(dev, n)
    ids = np.arange(n, dtype=np.uint32)

    def post_reset_obs(seed):
        w.reset_envs(ids, seed)
        o = w.export_obs()
        snap = {k: np.array(o[k], copy=True) for k in RESET_FIELDS}
        # 5 zero-action follow-up steps -> FULL-field snapshot (physics from the
        # reset state must be deterministic too).
        zero = np.zeros(n * w.action_dim, dtype=np.float32)
        for _ in range(5):
            w.step_with_actions(zero)
        full = {k: np.array(v, copy=True) for k, v in flat_fields(w, w.export_obs()).items()}
        return snap, full

    snap_a, full_a = post_reset_obs(7)
    # scramble: 20 random steps so the pre-reset state genuinely differs.
    rng = np.random.default_rng(123)
    lim = np.asarray(w.dof_limits(), dtype=np.float32)
    for _ in range(20):
        a = (rng.uniform(-1, 1, size=(n, w.action_dim)).astype(np.float32) * lim)
        a[:, :w.base_dof] = 0.0
        w.step_with_actions(a.reshape(-1))
    snap_b, full_b = post_reset_obs(7)
    for k in RESET_FIELDS:
        assert np.array_equal(snap_a[k], snap_b[k]), f"post-reset {k} differs (seed 7)"
    for k in full_a:
        assert np.array_equal(full_a[k], full_b[k]), \
            f"5-step post-reset rollout field {k} differs (seed 7)"
    snap_c, _ = post_reset_obs(8)
    diff = not np.array_equal(snap_a["cup_pose"], snap_c["cup_pose"])
    print(f"[gate2] seed 7 == seed 7 byte-exact (reset fields + 5-step rollout); "
          f"seed 8 cup_pose differs = {diff}")
    assert diff, "different seeds produced identical cup poses"
    w.destroy()
    print("[gate2] PASS: seeded reset deterministic")


# ---------------------------------------------------------------------------
# D1: two fresh worlds, one seeded random action stream -> byte-identical.
# ---------------------------------------------------------------------------
def d1(dev, n: int = 4, steps: int = 300):
    checkpoints = {1, 50, 150, steps}
    snaps = []
    for run in range(2):
        w = make_world(dev, n)
        lim = np.asarray(w.dof_limits(), dtype=np.float32)
        rng = np.random.default_rng(20260611)
        snap = {}
        for s in range(1, steps + 1):
            a = (rng.uniform(-1, 1, size=(n, w.action_dim)).astype(np.float32) * lim)
            a[:, :w.base_dof] = 0.0
            w.step_with_actions(a.reshape(-1))
            if s in checkpoints:
                snap[s] = {k: np.array(v, copy=True)
                           for k, v in flat_fields(w, w.export_obs()).items()}
        w.destroy()
        snaps.append(snap)
    for s in sorted(checkpoints):
        for k in snaps[0][s]:
            assert np.array_equal(snaps[0][s][k], snaps[1][s][k]), \
                f"D1 broke: step {s} field {k} differs between runs"
    print(f"[d1] PASS: two-run byte-identity over {steps} random steps "
          f"(checkpoints {sorted(checkpoints)}, all fields)")


# ---------------------------------------------------------------------------
# gate 3: 10k random-action steps, zero NaN/inf anywhere.
# ---------------------------------------------------------------------------
def gate3(dev, n: int = 4, steps: int = 10000, check_every: int = 50):
    w = make_world(dev, n)
    lim = np.asarray(w.dof_limits(), dtype=np.float32)
    rng = np.random.default_rng(0xC0FFEE)
    t0 = time.perf_counter()
    nonfinite = 0
    for s in range(1, steps + 1):
        a = (rng.uniform(-1, 1, size=(n, w.action_dim)).astype(np.float32) * lim)
        a[:, :w.base_dof] = 0.0  # the 6 dead base columns, masked by convention.
        w.step_with_actions(a.reshape(-1))
        if s % check_every == 0 or s == steps:
            obs = w.export_obs()
            for k in FLOAT_FIELDS + ("last_action",):
                arr = np.asarray(obs[k])
                if not np.all(np.isfinite(arr)):
                    nonfinite += 1
                    print(f"[gate3] NON-FINITE at step {s} field {k}")
            if s % 2000 == 0:
                cz = np.ascontiguousarray(obs["cup_pose"])[:, 2]
                bz = np.ascontiguousarray(obs["base_pose"])[:, 2]
                print(f"[gate3] step {s}: cup_z={np.round(cz, 2).tolist()} "
                      f"base_z={np.round(bz, 2).tolist()} "
                      f"({s / (time.perf_counter() - t0):.1f} steps/s)")
        if nonfinite:
            break
    wall = time.perf_counter() - t0
    assert nonfinite == 0, "gate3 FAILED: NaN/inf appeared under random torques"
    print(f"[gate3] PASS: {steps} random-action steps N={n}, ZERO NaN/inf "
          f"({wall:.0f}s, {steps / wall:.1f} world-steps/s)")
    # reset-mixed finite smoke (resets under randomness stay finite too).
    for s in range(1, 1501):
        a = (rng.uniform(-1, 1, size=(n, w.action_dim)).astype(np.float32) * lim)
        a[:, :w.base_dof] = 0.0
        w.step_with_actions(a.reshape(-1))
        if s % 250 == 0:
            w.reset_envs(np.array([s // 250 % n], dtype=np.uint32), seed=s)
            obs = w.export_obs()
            for k in FLOAT_FIELDS:
                assert np.all(np.isfinite(np.asarray(obs[k]))), \
                    f"reset-mixed smoke: non-finite {k} at step {s}"
    print("[gate3] PASS: reset-mixed 1500-step finite smoke")
    w.destroy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", default="all",
                    help="comma list: gate1,bite,gate2,d1,gate3 or 'all'")
    ap.add_argument("--gate3-steps", type=int, default=10000)
    ap.add_argument("--skip-record", action="store_true",
                    help="reuse an existing .g2_logs/g2_pyref_n4.bin")
    args = ap.parse_args()
    if not assets_present():
        print("SKIP: h1_with_hand / cup assets not present")
        return 0
    want = (["gate1", "bite", "gate2", "d1", "gate3"] if args.gates == "all"
            else args.gates.split(","))

    ref = None
    if "gate1" in want or "bite" in want:
        if not args.skip_record or not os.path.exists(REF_BIN):
            run_recorder()
        ref = RefRecording(REF_BIN)
        print(f"[ref] n={ref.n} steps={ref.steps} dof={ref.dof} nf={ref.nf} "
              f"events={ref.events}")

    with nuka.Device.create(0) as dev:
        if "gate1" in want:
            gate1(dev, ref)
        if "bite" in want:
            bite(dev, ref)
        if "gate2" in want:
            gate2(dev)
        if "d1" in want:
            d1(dev)
        if "gate3" in want:
            gate3(dev, steps=args.gate3_steps)
    print("\nALL REQUESTED G2 GATES PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
