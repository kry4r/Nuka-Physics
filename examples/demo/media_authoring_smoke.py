#!/usr/bin/env python3
"""Smoke gate for the general media authoring facade (soft-tet / fluid / rigid).

All PURE PYTHON over the prebuilt nuka module (no recompile). Authors each scene
through nuka.author (morphs + materials + surfaces -> Scene.build) and steps it on
device 0, proving the facade reaches the SAME CookSceneToModel a file scene does:

  (a) a robot-free free-space SOFT-TET (no ground) builds, has particles, no NaN;
  (b) a Ground plane + a FLUID box builds: a rigid body AND fluid particles, no NaN;
  (c) a Ground plane + a SOFT-TET builds and steps without NaN;
  (d) two illegal authorings raise LOUDLY -- a cloth Grid with a Soft material, and
      an MLS-MPM fluid co-resident with an XPBD soft in one scene;
  (e) the cloth fast path is UNCHANGED -- the go2 cloth scene still routes through
      create_coupled_from_scene (not the SceneBuilder), at the expected particle count.

Run from the repo root:
    python examples/demo/media_authoring_smoke.py
"""

import os

import numpy as np

import nuka
from nuka.author import Scene, SimOptions, materials, morphs, surfaces

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GO2 = os.path.join(REPO, "examples", "scenes", "go2.nks")

_FAILS = []


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        _FAILS.append(name)


def field(world, f):
    return np.asarray(world.download_field(f), dtype=np.float32)


def has_nan(world):
    bad = False
    for f in (nuka.Field.PARTICLE_POSITION, nuka.Field.PARTICLE_VELOCITY):
        a = field(world, f)
        if a.size and not np.isfinite(a).all():
            bad = True
    return bad


def rigid_body_count(world):
    a = field(world, nuka.Field.RIGID_BODY_TRANSFORM)
    return a.size // 7


def case_free_soft_tet(dev):
    print("== (a) free-space soft-tet (no ground), Soft.XPBD ==")
    s = Scene(SimOptions(dt=1.0 / 240.0, env_count=1))
    s.add_entity(morphs.TetSphere((0.0, 0.0, 0.5), 0.10, 8),
                 materials.Soft.XPBD(mass=0.01, volume_alpha=1.5e-6, iters=40),
                 surfaces.Soft(smooth_iters=1))
    w = s.build(dev)
    p0 = w.particle_count
    for _ in range(30):
        w.step()
    nan = has_nan(w)
    gate("free soft-tet has particles", p0 > 0, f"particle_count={p0}")
    gate("free soft-tet steps without NaN", not nan)
    w.destroy()


def case_ground_fluid(dev):
    print("== (b) Ground + FluidBox, Fluid.PBF ==")
    s = Scene(SimOptions(dt=1.0 / 240.0, env_count=1))
    s.add_entity(morphs.Ground(), materials.Rigid(friction=0.6))
    s.add_entity(morphs.FluidBox((-0.08, -0.08, 0.06), (0.08, 0.08, 0.22), 0.025),
                 materials.Fluid.PBF(rest_density=1000.0, iters=4))
    w = s.build(dev)
    rb, p0 = rigid_body_count(w), w.particle_count
    for _ in range(30):
        w.step()
    nan = has_nan(w)
    gate("ground+fluid has a rigid body", rb >= 1, f"rigid_bodies={rb}")
    gate("ground+fluid has fluid particles", p0 > 0, f"particle_count={p0}")
    gate("ground+fluid steps without NaN", not nan)
    w.destroy()


def case_ground_soft_tet(dev):
    print("== (c) Ground + soft-tet, Soft.XPBD ==")
    s = Scene(SimOptions(dt=1.0 / 240.0, env_count=1))
    s.add_entity(morphs.Ground(), materials.Rigid(friction=0.8))
    s.add_entity(morphs.TetSphere((0.0, 0.0, 0.28), 0.10, 8),
                 materials.Soft.XPBD(mass=0.01, volume_alpha=1.5e-6, iters=40))
    w = s.build(dev)
    p0 = w.particle_count
    for _ in range(30):
        w.step()
    nan = has_nan(w)
    gate("ground+soft-tet has particles", p0 > 0, f"particle_count={p0}")
    gate("ground+soft-tet steps without NaN", not nan)
    w.destroy()


def case_lone_mpm(dev):
    print("== (f) lone MLS-MPM fluid medium builds + steps ==")
    s = Scene(SimOptions(dt=1.0 / 240.0, env_count=1))
    s.add_entity(morphs.FluidBox((-0.06, -0.06, 0.05), (0.06, 0.06, 0.17), 0.022),
                 materials.Fluid.MPM(dx=0.011, substeps=40, floor_d=0.0))
    w = s.build(dev)
    p0 = w.particle_count
    for _ in range(30):
        w.step()
    nan = has_nan(w)
    gate("lone MPM fluid has particles", p0 > 0, f"particle_count={p0}")
    gate("lone MPM fluid steps without NaN", not nan)
    w.destroy()


def case_illegal(dev):
    print("== (d) illegal authorings raise loudly ==")
    # (d1) a cloth Grid with a Soft material (a morph x material mismatch).
    s = Scene(SimOptions(dt=1.0 / 240.0, env_count=1, contact_family=1))
    s.add_entity(morphs.NKS(GO2))
    s.add_entity(morphs.Grid(8, 8, 0.05, origin=(0, 0, 0.6)), materials.Soft.XPBD())
    try:
        s.build(dev)
        gate("Grid + Soft material raises", False, "no exception")
    except Exception as exc:  # noqa: BLE001 -- assert the loud rejection
        gate("Grid + Soft material raises", True,
             f"{type(exc).__name__}: {str(exc)[:70]}")

    # (d2) an MLS-MPM fluid co-resident with an XPBD soft in one scene.
    s = Scene(SimOptions(dt=1.0 / 240.0, env_count=1))
    s.add_entity(morphs.TetSphere((0.0, 0.0, 0.3), 0.08, 6), materials.Soft.XPBD())
    s.add_entity(morphs.FluidBox((-0.1, -0.1, 0.5), (0.1, 0.1, 0.7), 0.03),
                 materials.Fluid.MPM())
    try:
        s.build(dev)
        gate("MLS-MPM fluid + XPBD soft mix raises", False, "no exception")
    except Exception as exc:  # noqa: BLE001 -- assert the loud rejection
        gate("MLS-MPM fluid + XPBD soft mix raises", True,
             f"{type(exc).__name__}: {str(exc)[:70]}")


def case_cloth_fast_path_unchanged(dev):
    print("== (e) cloth fast path unchanged (go2 cloth routes via create_coupled) ==")
    if not os.path.exists(GO2):
        gate("go2.nks available", False, GO2)
        return
    # Spy on the routing: the cloth scene must take the coupled fast path and never
    # the general SceneBuilder path. Patch the pure-Python Scene methods.
    import nuka.author.scene as scn
    calls = {"coupled": 0, "general": 0}
    real_coupled = scn.Scene._build_coupled
    real_general = scn.Scene._build_general

    def spy_coupled(self, *a, **kw):
        calls["coupled"] += 1
        return real_coupled(self, *a, **kw)

    def spy_general(self, *a, **kw):
        calls["general"] += 1
        return real_general(self, *a, **kw)

    scn.Scene._build_coupled = spy_coupled
    scn.Scene._build_general = spy_general
    try:
        nx, ny, spacing = 55, 51, 0.024
        s = Scene(SimOptions(
            dt=1.0 / 240.0, env_count=1, contact_family=1,
            solver_vel_iters=80, solver_pos_iters=28, solver_contact_margin=0.044,
            solver_max_pairs=64, baumgarte_max_velocity=0.45))
        s.add_entity(morphs.NKS(GO2))
        s.add_entity(
            morphs.Grid(nx, ny, spacing, origin=(0.0, 0.0, 0.90)),
            materials.Cloth.XPBD(mass=0.012, friction=1.8, bend_alpha=0.09, iters=80),
            surfaces.Cloth(free=True, aero=(30.0, 0.12, 0.16)),
            contact_radius=0.022)
        w = s.build(dev)
        p = w.particle_count
        w.destroy()
    finally:
        scn.Scene._build_coupled = real_coupled
        scn.Scene._build_general = real_general
    gate("cloth scene used the coupled fast path (not SceneBuilder)",
         calls["coupled"] == 1 and calls["general"] == 0,
         f"coupled={calls['coupled']} general={calls['general']}")
    gate("cloth particle_count == nx*ny", p == nx * ny, f"particles={p} expected={nx*ny}")


def main():
    print(f"nuka from:   {nuka.__file__}")
    dev = nuka.Device.create(0)
    case_free_soft_tet(dev)
    case_ground_fluid(dev)
    case_ground_soft_tet(dev)
    case_lone_mpm(dev)
    case_illegal(dev)
    case_cloth_fast_path_unchanged(dev)
    dev.close()
    print("=" * 70)
    print(f"MEDIA AUTHORING SMOKE = {'ALL PASS' if not _FAILS else 'FAIL: ' + ', '.join(_FAILS)}")
    raise SystemExit(1 if _FAILS else 0)


if __name__ == "__main__":
    main()
