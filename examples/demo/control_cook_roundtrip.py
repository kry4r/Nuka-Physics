#!/usr/bin/env python3
"""Verify the Python control + cook-flag surface over the coupled world:
  (1) upload_field/download_field round-trip per field (BASE_POSE re-seat, the
      LINK_VELOCITY / BODY_*_VELOCITY park, a PARTICLE_POSITION pin);
  (2) dof_index / dof_names / stand_targets resolve the Go2 by joint NAME;
  (3) a cloth_free + aero cook produces an UNPINNED drape (perimeter z drops on
      step) vs the pinned membrane.
Host-only (numpy); no torch needed -- field I/O goes through the general byte path.
"""
import os
import numpy as np
import nuka

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCENE = os.path.join(REPO, "examples", "scenes", "go2.nks")

# Modest cloth patch high above the dog so a free corner free-falls cleanly.
GRID_NX, GRID_NY, SPACING = 12, 12, 0.05
ORIGIN = (0.0, 0.0, 1.20)

_fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        _fails.append(name)


def make_world(dev, cloth_free=False, aero=(0.0, 0.0, 0.0)):
    return nuka.World.create_coupled_from_scene(
        device=dev, scene_path=SCENE, env_count=1, dt=1.0 / 240.0,
        cloth_nx=GRID_NX, cloth_ny=GRID_NY, cloth_spacing=SPACING,
        cloth_origin_x=ORIGIN[0], cloth_origin_y=ORIGIN[1], cloth_origin_z=ORIGIN[2],
        cloth_particle_mass=0.01, cloth_friction=0.8, cloth_bend_alpha=0.09,
        cloth_iters=20, cloth_free=cloth_free,
        aero_normal=aero[0], aero_tangent=aero[1], aero_max_dv=aero[2])


def roundtrip(world):
    print("== (1) upload_field / download_field round-trip ==")
    blc = world.base_link_count
    # -- BASE_POSE (7 floats/env): re-seat z, prove the byte write AND that step()
    #    consumes it (the cloth hold_dog re-seat). --
    bp = world.download_field(nuka.Field.BASE_POSE)
    check("BASE_POSE view non-empty", bp.size == 7, f"size={bp.size}")
    bp2 = bp.copy(); bp2[2] = 0.55  # set root z
    world.upload_field(nuka.Field.BASE_POSE, bp2)
    got = world.download_field(nuka.Field.BASE_POSE)
    check("BASE_POSE byte round-trip (no step)", np.allclose(got, bp2),
          f"z={got[2]:.4f} (want 0.5500)")
    world.step()
    after = world.download_field(nuka.Field.BASE_POSE)
    check("BASE_POSE re-seat consumed by step()", abs(after[2] - 0.55) < 0.05,
          f"post-step z={after[2]:.4f} (re-seated near 0.55, not the cooked ~0.32)")

    # -- LINK_VELOCITY (6 floats/link, env-major): write a known pattern, read back. --
    lv = world.download_field(nuka.Field.LINK_VELOCITY)
    check("LINK_VELOCITY view", lv.size == blc * 6, f"size={lv.size} (blc*6={blc*6})")
    patt = (np.arange(lv.size, dtype=np.float32) * 0.01).astype(np.float32)
    world.upload_field(nuka.Field.LINK_VELOCITY, patt)
    check("LINK_VELOCITY byte round-trip",
          np.allclose(world.download_field(nuka.Field.LINK_VELOCITY), patt))
    # zero the root spatial velocity (the per-step park), read it back.
    world.upload_field(nuka.Field.LINK_VELOCITY, np.zeros(6, np.float32))
    check("LINK_VELOCITY root zero", np.allclose(
        world.download_field(nuka.Field.LINK_VELOCITY, offset=0, count=6), 0.0))

    # -- BODY_LINEAR / ANGULAR velocity (new fields; per-body Vec3). --
    for fld in (nuka.Field.BODY_LINEAR_VELOCITY, nuka.Field.BODY_ANGULAR_VELOCITY):
        v = world.download_field(fld)
        ok_view = v.size > 0 and v.size % 3 == 0
        check(f"{fld} view (per-body Vec3)", ok_view, f"size={v.size}")
        if ok_view:
            p = (np.arange(v.size, dtype=np.float32) * 0.001 + 0.5).astype(np.float32)
            world.upload_field(fld, p)
            check(f"{fld} byte round-trip", np.allclose(world.download_field(fld), p))

    # -- PARTICLE_POSITION (the cloth/fluid pin verb). --
    pp = world.download_field(nuka.Field.PARTICLE_POSITION)
    check("PARTICLE_POSITION view", pp.size > 0 and pp.size % 3 == 0,
          f"size={pp.size} (particles={world.particle_count})")
    pin = pp.copy(); pin[0:3] = (0.111, 0.222, 0.333)  # pin particle 0
    world.upload_field(nuka.Field.PARTICLE_POSITION, pin)
    check("PARTICLE_POSITION pin round-trip",
          np.allclose(world.download_field(nuka.Field.PARTICLE_POSITION)[0:3],
                      (0.111, 0.222, 0.333)))

    # -- LOUD failure on an over-range write (never a silent clamp). --
    loud = False
    try:
        world.upload_field(nuka.Field.BASE_POSE, np.zeros(7, np.float32), offset=4)
    except Exception:
        loud = True
    check("over-range upload raises (loud, not silent)", loud)


def introspect(world):
    print("== (2) name -> DOF introspection ==")
    names = world.dof_names()
    print("  cooked DOF names:", names)
    thigh = world.dof_index("go2/.*_thigh")
    calf = world.dof_index(".*_calf")
    hip = world.dof_index(".*_hip")
    print(f"  dof_index('go2/.*_thigh') = {thigh.tolist()}")
    print(f"  dof_index('.*_calf')      = {calf.tolist()}")
    print(f"  dof_index('.*_hip')       = {hip.tolist()}")
    check("4 thigh DOFs resolved (FL/FR/RL/RR)", thigh.size == 4, f"n={thigh.size}")
    check("4 calf DOFs resolved", calf.size == 4, f"n={calf.size}")
    check("4 hip DOFs resolved", hip.size == 4, f"n={hip.size}")
    check("thigh names all match _thigh",
          all("_thigh" in names[i] for i in thigh.tolist()))
    stand = world.stand_targets(thigh=0.80, calf=-1.50, hip=0.10)
    print(f"  stand_targets(thigh=.8,calf=-1.5,hip=.1) = {np.round(stand, 3).tolist()}")
    check("stand_targets width == action_dim", stand.size == world.action_dim)
    # every actuated slot got a crouch value (no zeros left for go2's 12 joints).
    check("stand_targets crouch resolved by name", np.count_nonzero(stand) == stand.size,
          f"nonzero={np.count_nonzero(stand)}/{stand.size}")
    # the thigh columns carry +0.80 exactly.
    th_cols = (thigh - 1).tolist()  # action vector is slots [1:], so col == link-1.
    check("thigh targets == 0.80", all(abs(stand[c] - 0.80) < 1e-6 for c in th_cols))


def cloth_free_cook(dev):
    print("== (3) cloth_free + aero cook -> unpinned drape ==")
    corner_z0 = ORIGIN[2]
    results = {}
    for tag, free, aero in (("pinned", False, (0.0, 0.0, 0.0)),
                            ("free+aero", True, (30.0, 0.12, 0.16))):
        w = make_world(dev, cloth_free=free, aero=aero)
        for _ in range(60):
            w.step()
        # corner particle 0 == cloth grid (i=0,j=0), a perimeter corner.
        z = float(w.download_field(nuka.Field.PARTICLE_POSITION, offset=2, count=1)[0])
        results[tag] = z
        print(f"  {tag:10s}: perimeter-corner z after 60 steps = {z:.4f} (start {corner_z0:.2f})")
        w.destroy()
    check("pinned perimeter HELD (corner z ~ origin)",
          abs(results["pinned"] - corner_z0) < 1e-3,
          f"z={results['pinned']:.4f}")
    check("free perimeter DROPS (corner fell on step)",
          results["free+aero"] < corner_z0 - 0.02,
          f"z={results['free+aero']:.4f} < {corner_z0 - 0.02:.4f}")


def main():
    print(f"nuka from: {nuka.__file__}")
    print(f"scene: {SCENE}")
    dev = nuka.Device.create(0)
    w = make_world(dev, cloth_free=False)
    print(f"world: env_count={w.env_count} base_link_count={w.base_link_count} "
          f"particles={w.particle_count} action_dim={w.action_dim}")
    roundtrip(w)
    introspect(w)
    w.destroy()
    cloth_free_cook(dev)
    dev.close()
    print("=" * 60)
    print(f"VERDICT = {'ALL PASS' if not _fails else 'FAIL: ' + ', '.join(_fails)}")
    raise SystemExit(1 if _fails else 0)


if __name__ == "__main__":
    main()
