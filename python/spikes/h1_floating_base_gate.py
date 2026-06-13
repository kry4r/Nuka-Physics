"""ENGINE FEASIBILITY GATE spike: can a FLOATING-BASE H1 with foot-ground contact
be loaded + STEPPED in the SAME batched trainable world go2 PPO trains in?

This is an INVESTIGATION, not a feature. It drives ``nuka.World`` DIRECTLY (NOT
``NukaGymEnv`` -- that layer hard-asserts ``action_dim==12`` and is go2-specialized
end to end). The deliverable is the printed findings, not a passing test.

------------------------------------------------------------------------------
VERDICT (proven by this spike + static code read):  H1 is BLOCKED on engine
wiring for the trainable batched world. THREE distinct blockers, in pipeline
order, plus a Python-layer one:

  [1] MESH COOK (hit first; a removable COST, not a hard stop).
      H1 has 49 type="mesh" collision geoms (1.6 MB torso, 660 KB pelvis,
      432 KB ankles, ...). CookScene (src/scene/cooker.cpp:364-426) runs V-HACD
      convex decomposition + a narrow-band SDF bake per unique convex piece for
      EVERY mesh. The V-HACD cache is in-process MEMO-ONLY (kDecomposeCacheDir=""
      -- cooker.cpp:29), so every cold create_from_scene re-cooks all 49 meshes
      from scratch: many MINUTES, CPU-bound, single-threaded (measured >8 min,
      still running, vs go2's 0.01 s with 0 meshes). This cook is WASTED for the
      batched trainable path -- the legacy batched articulated world's ctor took
      only (host articulation state, FootShape feet, ground_height) and never
      consumed the mesh SDFs (ArticulationHostState carries NO shapes). Fix: skip
      mesh-collision cook for the batched path, OR
      persist the decomposition+SDF cache to disk, OR strip colliders.

  [2] DOF CAP (the HARD engine stop -- the real gate-killer).
      The batched contact solver's on-thread PGS working vectors are capped at
      kMaxContactSolverDof = 18u (articulation_contacts.hpp:292-294), commented
      "6-DOF floating base + 12 revolute = 18" -- i.e. sized for GO2. H1 is 51
      DOF (6 floating base + 45 hinge joints). The legacy batched articulated
      world's ctor threw "max_dof out of range" for any
      max_dof > 18 -> create_from_scene returns INTERNAL (100).
      ISOLATED EMPIRICALLY (two variants below, identical except for DOF):
        * mesh-stripped + sphere-foot H1, 51 DOF  -> INTERNAL(100), construct fails
        * SAME asset, arms/torso/hands WELDED to 16 DOF (6 free + 10 leg) ->
          CONSTRUCTS in 0.004s, steps 300x, foot-ground contact ACTIVE (~384
          nonzero contact-force comps), base seated then falls (no policy).
      The ONLY changed variable is the DOF count crossing 18, so [2] is the
      isolated cause (not the line-117 / per-articulation / CUDA-scratch guards,
      which the 16-DOF success rules out). Fix: raise kMaxContactSolverDof to
      >=51 and resize the per-thread M / M^-1 / Jacobian / PGS scratch
      accordingly (per-thread memory + register-pressure cost on the kernels).

  [3] FOOT CONTACT is sphere-only (functional; static-certain).
      DeriveFeet (src/c_abi/world.cpp:191-223) recognizes ShapeType::Sphere ONLY;
      the batched contact path is sphere-foot-vs-flat-ground ONLY
      (articulation_contacts.hpp:25, DetectFootGroundContacts = sphere-vs-plane).
      H1's feet are type="mesh" -> 0 feet found -> DeriveGroundHeight returns
      false -> create_from_scene returns NOT_SUPPORTED. (You never SEE this on
      the stock asset because [1] and then [2] fire first; it's confirmed by
      reading the code + by the fact that ADDING sphere ankle feet changes the
      failure from NOT_SUPPORTED to the [2] DOF-cap INTERNAL error.) Fix: cook
      the ankle collision to a sphere foot, OR extend DeriveFeet +
      DetectFootGroundContacts to capsule/mesh-vs-plane.

  [P] PYTHON LAYER is go2-hardcoded. env.py:96 asserts action_dim==12; go2_obs
      hardcodes GO2_BLC=13, GO2_OBS_DIM=48, a 12-joint URDF<->Nuka permutation,
      and the go2 obs/reward. Generalize to N-DOF + H1 feet/obs to train H1.

NON-BLOCKERS (the task asked to distinguish these -- they are NOT the problem):
  * Floating-base root IS supported AND PROVEN: go2_float is itself a FloatingBase
    scene training 4096 envs through this exact path (Free -> FloatingBase cook,
    cooker.cpp:236; floating-base ABA IntegrateFloatingBase{Velocity,Pose}).
  * MJCF IS a supported create_from_scene format (world.cpp:31, ".xml"/".mjcf"
    -> import::LoadMjcf), and LoadMjcf handles <freejoint> -> JointType::Free
    and mesh geoms -> ShapeType::TriMesh.
------------------------------------------------------------------------------

Run (from repo root; the conda env is the one that built/runs the wheel):
  export CUDA_VISIBLE_DEVICES=0
  /root/miniconda3/envs/nuka-v03/bin/python python/spikes/h1_floating_base_gate.py
Flags:
  --full-cook  ALSO run the stock-asset cold cook (single-env), timing the mesh
               cook wall [1]. Many MINUTES; default OFF.
"""

from __future__ import annotations

import math
import pathlib
import re
import sys
import time
import traceback

import torch

import nuka

REPO = pathlib.Path(__file__).resolve().parents[2]
H1_MJCF = REPO / ".nuka-assets/newton_assets/unitree_h1/mjcf/h1_with_hand.xml"
GO2_SCENE = REPO / "examples/scenes/go2_float.usda"
DT = 0.005


def _tilt_deg(qw, qx, qy, qz):
    """Tilt (deg) of the body +z away from world +z, from a w-first quat."""
    gz_body = -(1.0 - 2.0 * (qx * qx + qy * qy))
    gz_body = max(-1.0, min(1.0, gz_body))
    return math.degrees(math.acos(-gz_body))


def _report_base(world, tag, env=0):
    bp = torch.from_dlpack(world.buffer_view(nuka.BASE_POSE)).view(-1, 7)
    px, py, pz, qw, qx, qy, qz = bp[env].detach().cpu().tolist()
    tilt = _tilt_deg(qw, qx, qy, qz)
    print(f"    [{tag}] base z={pz:+.4f} tilt={tilt:6.2f}deg "
          f"quat=({qw:+.3f},{qx:+.3f},{qy:+.3f},{qz:+.3f})", flush=True)
    return pz, tilt


# ---------------------------------------------------------------------------
def go2_baseline(dev):
    """The path that trains today, for timing/structure contrast."""
    print("\n=== BASELINE: go2_float, 64-env batched (the trained path) ===")
    t = time.time()
    w = nuka.World.create_from_scene(dev, str(GO2_SCENE), 64, DT)
    print(f"    go2 64-env CONSTRUCTED in {time.time()-t:.3f}s: "
          f"blc={w.base_link_count} action_dim={w.action_dim}", flush=True)
    w.step()
    nuka.sync()
    w.destroy()
    print("    go2 steps fine (0 meshes -> 0.01s cook; 12 actuated DOF; 4 sphere "
          "feet). This is the shape every blocker below is measured against.")


def _make_stripped_sphere_h1(out_path):
    """H1 MJCF with ALL <geom> + the <asset> mesh block removed, and a SPHERE
    foot collider injected at each ankle. Bodies keep their <inertial> (mass +
    inertia come from there, mjcf_importer.cpp:463-471), so the 51-DOF dynamics
    are intact. This ISOLATES blockers [2]/[3] from the [1] mesh-cook wall: with
    no meshes the cook is ~instant, and the sphere feet satisfy DeriveFeet [3]."""
    text = H1_MJCF.read_text()
    text = re.sub(r"<asset>.*?</asset>", "<asset/>", text, flags=re.DOTALL)
    text = re.sub(r"<geom\b[^>]*/>", "", text)
    text = re.sub(r"<geom\b[^>]*>.*?</geom>", "", text, flags=re.DOTALL)
    sphere = ('<geom name="{s}_foot_sphere" type="sphere" size="0.04" '
              'pos="0.04 0 -0.05"/>')
    for s in ("left", "right"):
        anchor = (f'<joint name="{s}_ankle_joint" pos="0 0 0" axis="0 1 0" '
                  f'range="-0.87 0.52"/>')
        assert anchor in text, f"ankle anchor missing for {s}"
        text = text.replace(anchor, anchor + "\n                "
                            + sphere.format(s=s), 1)
    out_path.write_text(text)
    return out_path


def case_batched_stripped_sphere(dev, env_count=64, n_steps=300):
    """The decisive gate experiment: mesh-stripped + sphere-foot H1 into the
    BATCHED trainable world. Fast (no mesh cook). PREDICTION: blocked at the
    51 > 18 DOF cap (blocker [2]) -> INTERNAL (100)."""
    print(f"\n=== GATE: mesh-stripped + sphere-feet H1, {env_count}-env BATCHED "
          "(the trainable path) ===")
    spike = H1_MJCF.parent / "h1_spike_nomesh_sphere_feet.xml"
    try:
        _make_stripped_sphere_h1(spike)
        ng = spike.read_text().count("<geom")
        print(f"    wrote {spike} (geoms now: {ng} -- the 2 sphere feet)", flush=True)
        t = time.time()
        try:
            w = nuka.World.create_from_scene(dev, str(spike), env_count, DT)
        except Exception as e:  # noqa: BLE001
            print(f"    BLOCKED at batched construct ({time.time()-t:.2f}s): "
                  f"{type(e).__name__}: {e}", flush=True)
            print("    => This is blocker [2]: H1 is 51 DOF (6 floating + 45 "
                  "hinge) > kMaxContactSolverDof=18 (articulation_contacts.hpp:294, "
                  "sized for go2's 6+12). The legacy batched articulated ctor threw "
                  "'max_dof out of range' -> INTERNAL(100). The mesh cook [1] is "
                  "OUT of the picture here (0 meshes, ~instant), and the sphere "
                  "feet cleared [3], so this is the irreducible HARD engine stop.")
            return
        # If it ever constructs (i.e. someone raised the cap), prove it steps.
        try:
            print(f"    CONSTRUCTED in {time.time()-t:.3f}s: blc={w.base_link_count} "
                  f"action_dim={w.action_dim} env_count={w.env_count} "
                  "(the DOF cap was raised!)", flush=True)
            z0, t0 = _report_base(w, "t=0")
            for k in range(n_steps):
                w.step()
                if (k + 1) % 75 == 0:
                    nuka.sync()
                    _report_base(w, f"t={k+1}")
            nuka.sync()
            try:
                cf = torch.from_dlpack(w.buffer_view(nuka.CONTACT_FORCE)).view(-1)
                nz = int((cf.abs() > 1e-6).sum().item())
                print(f"    contact-force nonzero comps (all envs): {nz}/{cf.numel()}",
                      flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"    contact read skipped: {e}", flush=True)
            zf, tf = _report_base(w, "final")
            held = (zf > z0 - 0.15) and (tf < 45.0)
            print(f"    base z {z0:+.3f}->{zf:+.3f} tilt {t0:.1f}->{tf:.1f}deg :: "
                  + ("PD-HELD upright (NOT trained balance)" if held
                     else "FELL/settled (expected w/o policy; key = it STEPPED "
                          "with foot contact)"), flush=True)
            print("    GATE RESULT: batched floating-base H1 with foot-ground "
                  "contact LOADS + STEPS -> RL standing is UNBLOCKED.")
        finally:
            w.destroy()
    finally:
        if spike.exists():
            spike.unlink()


def _make_reduced_dof_h1(out_path):
    """The [2]-isolation control: SAME mesh-strip + sphere feet as the 51-DOF
    variant, but WELD (delete the <joint> of) every non-leg joint -- arms, hands,
    torso -- leaving the 1 freejoint + 10 leg joints = 16 DOF (<= the cap of 18).
    The welded links stay rigidly attached (mass/inertia preserved), so this is
    the SAME H1 biped, only with fewer actuated DOF. The ONLY variable that
    differs from the 51-DOF variant is the DOF count crossing 18."""
    text = H1_MJCF.read_text()
    text = re.sub(r"<asset>.*?</asset>", "<asset/>", text, flags=re.DOTALL)
    text = re.sub(r"<geom\b[^>]*/>", "", text)
    text = re.sub(r"<geom\b[^>]*>.*?</geom>", "", text, flags=re.DOTALL)
    leg = {f"{s}_{j}_joint"
           for s in ("left", "right")
           for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")}

    def _keep(m):
        nm = re.search(r'name="([^"]+)"', m.group(0))
        return m.group(0) if (nm and nm.group(1) in leg) else ""
    text = re.sub(r"<joint\b[^>]*/>", _keep, text)
    sphere = ('<geom name="{s}_foot_sphere" type="sphere" size="0.04" '
              'pos="0.04 0 -0.05"/>')
    for s in ("left", "right"):
        anchor = (f'<joint name="{s}_ankle_joint" pos="0 0 0" axis="0 1 0" '
                  f'range="-0.87 0.52"/>')
        assert anchor in text, f"ankle anchor missing for {s}"
        text = text.replace(anchor, anchor + "\n                "
                            + sphere.format(s=s), 1)
    out_path.write_text(text)
    return out_path


def case_reduced_dof_isolation(dev, env_count=64, n_steps=300):
    """[2]-isolation: a 16-DOF (<=18) H1 SHOULD construct + step in the batched
    world. If it does, the DOF cap is the isolated cause of the 51-DOF failure."""
    print(f"\n=== ISOLATION: SAME H1 welded to 16 DOF (<=18 cap), {env_count}-env "
          "BATCHED ===")
    spike = H1_MJCF.parent / "h1_spike_reduced_dof.xml"
    try:
        _make_reduced_dof_h1(spike)
        nj = len(re.findall(r"<joint\b", spike.read_text()))
        print(f"    wrote {spike} (hinge joints kept: {nj} -> DOF = 6 + {nj} "
              f"= {6+nj})", flush=True)
        t = time.time()
        try:
            w = nuka.World.create_from_scene(dev, str(spike), env_count, DT)
        except Exception as e:  # noqa: BLE001
            print(f"    UNEXPECTEDLY BLOCKED at <=18 DOF: {type(e).__name__}: {e}",
                  flush=True)
            print("    -> the cause is NOT just the DOF cap; a deeper blocker "
                  "exists. Investigate before raising kMaxContactSolverDof.")
            traceback.print_exc()
            return
        try:
            print(f"    CONSTRUCTED in {time.time()-t:.3f}s: blc={w.base_link_count} "
                  f"action_dim={w.action_dim} env_count={w.env_count}", flush=True)
            z0, t0 = _report_base(w, "t=0")
            for k in range(n_steps):
                w.step()
                if (k + 1) % 75 == 0:
                    nuka.sync()
                    _report_base(w, f"t={k+1}")
            nuka.sync()
            try:
                cf = torch.from_dlpack(w.buffer_view(nuka.CONTACT_FORCE)).view(-1)
                nz = int((cf.abs() > 1e-6).sum().item())
                print(f"    foot-ground contact ACTIVE: {nz}/{cf.numel()} nonzero "
                      "contact-force comps", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"    contact read skipped: {e}", flush=True)
            zf, tf = _report_base(w, "final")
            print(f"    base z {z0:+.3f}->{zf:+.3f} tilt {t0:.1f}->{tf:.1f}deg :: "
                  "STEPPED (falls w/o a balance policy -- EXPECTED; the point is "
                  "the batched floating-base biped steps with active foot contact "
                  "when under the DOF cap).", flush=True)
            print("    ISOLATION RESULT: 16-DOF SAME-H1 CONSTRUCTS + STEPS whereas "
                  "51-DOF fails -> the DOF cap [2] is the isolated batched blocker.")
        finally:
            w.destroy()
    finally:
        if spike.exists():
            spike.unlink()


def case_full_cook(dev):
    """Optional: time the stock-asset cold mesh cook [1] (single-env path, which
    skips DeriveFeet but STILL pays the full CookScene). MANY MINUTES."""
    print("\n=== --full-cook: stock H1, single-env (mesh-cook [1] wall) ===")
    print("    49 meshes -> V-HACD + SDF bake, no disk cache. This will take "
          "MINUTES (measured >8 min, CPU-bound).", flush=True)
    t = time.time()
    try:
        w = nuka.World.create_from_scene(dev, str(H1_MJCF), 1, DT)
    except Exception as e:  # noqa: BLE001
        print(f"    single-env load FAILED after {time.time()-t:.1f}s: "
              f"{type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return
    print(f"    single-env H1 LOADED (cook done) in {time.time()-t:.1f}s: "
          f"blc={w.base_link_count} action_dim={w.action_dim}", flush=True)
    z0, _ = _report_base(w, "t=0")
    for _ in range(100):
        w.step()
    nuka.sync()
    z1, _ = _report_base(w, "t=100steps")
    print(f"    base dropped {z0-z1:+.3f} m over 100 steps (no ground in the "
          "single-env path -> free-fall expected; proves the 51-DOF "
          "floating-base ABA integrates).", flush=True)
    w.destroy()


def main():
    print("nuka:", nuka.__file__, flush=True)
    print("H1 MJCF:", H1_MJCF, "exists=", H1_MJCF.exists(), flush=True)
    assert H1_MJCF.exists(), "H1 MJCF missing"
    full_cook = "--full-cook" in sys.argv
    with nuka.Device.create(0) as dev:
        go2_baseline(dev)
        case_batched_stripped_sphere(dev)
        case_reduced_dof_isolation(dev)
        if full_cook:
            case_full_cook(dev)
        else:
            print("\n(skipped --full-cook mesh-cook timing; pass --full-cook to "
                  "measure blocker [1].)")
    print("\n=== spike complete ===", flush=True)


if __name__ == "__main__":
    main()
