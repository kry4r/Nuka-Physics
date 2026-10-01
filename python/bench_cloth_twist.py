"""Drive a reproducible cloth twist through the common production diagnostic session."""

import argparse
import hashlib
import json
import math
import os
import signal
from pathlib import Path

import numpy as np

import nuka
from nuka.author import Scene, SimOptions, materials, morphs, surfaces
from nuka.diagnostics import DiagnosticSession, DiagnosticThresholds
from nuka.diagnostics.capture import sha256, write_json


def topology(nx, ny):
    faces = []
    for y in range(ny - 1):
        for x in range(nx - 1):
            a = y * nx + x
            faces.extend(((a, a + 1, a + nx + 1), (a, a + nx + 1, a + nx)))
    edges = sorted({tuple(sorted((face[k], face[(k + 1) % 3]))) for face in faces for k in range(3)})
    return np.asarray(faces, dtype=np.int32), np.asarray(edges, dtype=np.int32)


def targets(rest, nx, ny, angle):
    result = rest.copy()
    center_y, center_z = float(rest[:, 1].mean()), float(rest[:, 2].mean())
    for side, theta in ((0, angle), (nx - 1, -angle)):
        ids = np.arange(ny) * nx + side
        y, z = rest[ids, 1] - center_y, rest[ids, 2] - center_z
        c, s = math.cos(theta), math.sin(theta)
        result[ids, 1], result[ids, 2] = center_y + c * y - s * z, center_z + s * y + c * z
    return result


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    steps = round(args.duration / args.dt)
    if not math.isclose(steps * args.dt, args.duration, rel_tol=1e-9):
        raise ValueError("duration must be an integer multiple of dt")
    tolerance = os.environ.get("NUKA_SOLVER_VEL_TOLERANCE")
    if tolerance is None:
        raise ValueError("set the production NUKA_SOLVER_VEL_TOLERANCE explicitly for reproducibility")
    physical = {key: getattr(args, key) for key in ("nx", "ny", "spacing", "height", "duration", "turns",
        "density", "stretch", "poisson", "bend", "thickness", "gravity_z")}
    metadata = {**physical, "dt": args.dt, "integrator": args.integrator, "sweeps": args.sweeps,
        "expected_policy_steps": steps, "ogc_contact_capacity": args.contact_capacity,
        "solver_velocity_tolerance_mps": float(tolerance), "position_iterations": 4,
        "momentum_scope": "vbd_subsystem", "aerodynamic_forces": False,
        "state_system": "dynamic_particles", "gravity": [0, 0, args.gravity_z],
        "vbd_particle_begin": 0, "vbd_vertices": args.nx * args.ny,
        "physical_input_sha256": hashlib.sha256(json.dumps(physical, sort_keys=True).encode()).hexdigest(),
        "boundary": "Opposite x edges rotate by +/- pi*turns*t/duration about their initial y/z center",
        "render_acceptance": "unmeasured", "build_record": str(args.build_record),
        "binary_hashes": (args.build_record / "binaries.sha256").read_text(),
        "fixture_sha256": sha256(__file__)}
    scene = Scene(SimOptions(dt=args.dt, gravity=(0, 0, args.gravity_z),
        cloth_integrator=0 if args.integrator == "bdf2" else 1,
        solver_vel_iters=args.sweeps, solver_pos_iters=4,
        ogc_contact_capacity=args.contact_capacity, baumgarte_max_velocity=0))
    scene.add_entity(morphs.Grid(args.nx, args.ny, args.spacing, origin=(0, 0, args.height)),
        materials.Cloth.VBD(areal_density=args.density, stretch_stiffness=args.stretch,
            poisson=args.poisson, bend_stiffness=args.bend, thickness=args.thickness), surfaces.Cloth(free=True))
    device = nuka.Device.create(0)
    world = scene.build(device)
    session = None
    interrupted = False
    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True
    previous_handler = signal.signal(signal.SIGINT, stop)
    try:
        world.set_gravity_z(args.gravity_z)
        rest = np.asarray(world.download_field(nuka.Field.PARTICLE_POSITION)).reshape(-1, 3).copy()
        inv_mass = np.asarray(world.download_field(nuka.Field.PARTICLE_INV_MASS)).copy()
        original_inv_mass = inv_mass.copy()
        pinned = np.zeros(len(rest), dtype=bool)
        pinned[0::args.nx] = True
        pinned[args.nx - 1::args.nx] = True
        inv_mass[pinned] = 0
        world.upload_field(nuka.Field.PARTICLE_INV_MASS, inv_mass)
        initial_velocity = np.asarray(world.download_field(nuka.Field.PARTICLE_VELOCITY)).reshape(-1, 3).copy()
        elements = np.asarray(world.download_field(nuka.Field.VBD_ELEMENTS), dtype=np.uint32).reshape(-1, 16).copy()
        fields = (nuka.Field.PARTICLE_POSITION, nuka.Field.PARTICLE_VELOCITY,
                  nuka.Field.PARTICLE_INV_MASS, nuka.Field.VBD_EFFECTIVE_DT)
        limits = DiagnosticThresholds(velocity_tolerance_mps=float(tolerance))
        session = DiagnosticSession(world, args.output, metadata, chunk_steps=args.chunk_steps,
                                    state_fields=fields, thresholds=limits)
        faces, edges = topology(args.nx, args.ny)
        np.savez_compressed(args.output / "initial.npz", rest=rest, velocity=initial_velocity,
                            vbd_elements=elements, faces=faces, edges=edges,
                            inv_mass=inv_mass, original_inv_mass=original_inv_mass, pinned=pinned)
        session.manifest["initial_geometry_sha256"] = sha256(args.output / "initial.npz")
        write_json(args.output / "inputs.json", {"parameters": vars(args) | {"output": str(args.output),
            "build_record": str(args.build_record)}, "metadata": metadata})
        reason = "completed"
        count = steps if args.stop_after <= 0 else min(steps, args.stop_after)
        angle_per_step = math.pi * args.turns / steps
        for step in range(count):
            target = targets(rest, args.nx, args.ny, (step + 1) * angle_per_step)
            world.upload_field(nuka.Field.PARTICLE_KINEMATIC_TARGET, np.ascontiguousarray(target))
            sample = session.step(controls=target[pinned])
            if np.any(sample["env_status"]):
                reason = "physics_failure"
                break
            if not np.isfinite(sample["state_PARTICLE_POSITION"]).all() or not np.isfinite(sample["state_PARTICLE_VELOCITY"]).all():
                reason = "nonfinite_state"
                break
            if np.any(sample["dat_truncations"]):
                reason = "first_truncation"
                break
            if interrupted:
                reason = "interrupted"
                break
            if (step + 1) % 300 == 0:
                print(json.dumps({"policy_step": step + 1, "ogc_contacts": sample["ogc_contacts"].tolist()}), flush=True)
        else:
            if count < steps:
                reason = "diagnostic_prefix"
        session.close(reason)
        print(json.dumps({"output": str(args.output), "steps": session.steps, "stop_reason": reason}), flush=True)
    except BaseException as error:
        if session is not None:
            session.close("exception", error=repr(error))
        raise
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        world.destroy()
        device.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-record", type=Path, required=True)
    parser.add_argument("--nx", type=int, default=101)
    parser.add_argument("--ny", type=int, default=61)
    parser.add_argument("--spacing", type=float, default=.005)
    parser.add_argument("--height", type=float, default=.15)
    parser.add_argument("--dt", type=float, default=1 / 600)
    parser.add_argument("--duration", type=float, default=12.8)
    parser.add_argument("--turns", type=float, default=8)
    parser.add_argument("--density", type=float, default=.4)
    parser.add_argument("--stretch", type=float, default=5000)
    parser.add_argument("--poisson", type=float, default=.3)
    parser.add_argument("--bend", type=float, default=5e-5)
    parser.add_argument("--thickness", type=float, default=.003)
    parser.add_argument("--gravity-z", type=float, default=-9.81)
    parser.add_argument("--integrator", choices=("bdf2", "be"), default="bdf2")
    parser.add_argument("--sweeps", type=int, default=512)
    parser.add_argument("--contact-capacity", type=int, default=524288)
    parser.add_argument("--stop-after", type=int, default=0)
    parser.add_argument("--chunk-steps", type=int, default=128)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
