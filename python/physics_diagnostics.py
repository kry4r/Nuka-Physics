"""Capture, analyze, audit and compare production physics evidence."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import nuka
from nuka.diagnostics import DiagnosticSession, DiagnosticThresholds
from nuka.diagnostics.analysis import compare_runs, verify_particle_trace
from nuka.diagnostics.capture import sha256, write_json
from nuka.diagnostics.constitutive import verify_elastic_trace
from nuka.diagnostics.geometry import audit_mesh_trace
from nuka.diagnostics.report import make_report


def capture_scene(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.steps < 1 or args.env_count < 1 or not np.isfinite(args.dt) or args.dt <= 0:
        raise ValueError("capture requires positive steps, environment count and finite timestep")
    tolerance = os.environ.get("NUKA_SOLVER_VEL_TOLERANCE")
    if tolerance is None:
        raise ValueError("set the production NUKA_SOLVER_VEL_TOLERANCE explicitly for reproducibility")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("NUKA_SOLVER_VEL_TOLERANCE must be finite and positive")
    scene_hash = sha256(args.scene)
    physical_input = {"scene_sha256": scene_hash, "controls": "Authored scene defaults; no external inputs"}
    physical_input_hash = hashlib.sha256(json.dumps(physical_input, sort_keys=True).encode()).hexdigest()
    fields = tuple(getattr(nuka.Field, name) for name in args.state_field)
    device = nuka.Device.create(0)
    world = nuka.World.create_from_scene(device, str(args.scene), env_count=args.env_count, dt=args.dt,
                                         solver_vel_iters=args.sweeps, ogc_contact_capacity=args.contact_capacity)
    session = None
    try:
        execution_info = json.loads(json.dumps(world.execution_info, default=str))
        integrator = execution_info.get("integrator", execution_info.get("cloth_integrator"))
        metadata = {"scene": str(args.scene), "scene_sha256": scene_hash, "dt": args.dt,
            "sweeps": args.sweeps, "expected_policy_steps": args.steps,
            "duration_s": args.steps * args.dt, "physical_input": physical_input,
            "physical_input_sha256": physical_input_hash,
            "ogc_contact_capacity": args.contact_capacity or None,
            "solver_velocity_tolerance_mps": float(tolerance),
            "controls": physical_input["controls"],
            "owner_names": {"LINK": list(world.dof_names())},
            "kinematic_tree": [{key: link[key] for key in ("parent_index", "articulation_index", "joint_type")}
                               for link in world.kinematic_tree()],
            "execution_info": execution_info, "integrator": integrator,
            "integrator_source": "world.execution_info" if integrator is not None else "unavailable_in_public_execution_info",
            "build_record": str(args.build_record),
            "binary_hashes": (args.build_record / "binaries.sha256").read_text(),
            "fixture_sha256": sha256(__file__)}
        session = DiagnosticSession(world, args.output, metadata, env_count=args.env_count, state_fields=fields,
                                    thresholds=DiagnosticThresholds(velocity_tolerance_mps=float(tolerance)))
        positions = np.asarray(world.download_field(nuka.Field.PARTICLE_POSITION)).copy()
        if positions.size:
            if positions.size % (args.env_count * 3):
                raise ValueError("initial particle positions do not match the environment extent")
            count = positions.size // (args.env_count * 3)
            positions = positions.reshape(args.env_count, count, 3)
            velocity = np.asarray(world.download_field(nuka.Field.PARTICLE_VELOCITY)).reshape(positions.shape).copy()
            inverse_mass = np.asarray(world.download_field(nuka.Field.PARTICLE_INV_MASS)).reshape(args.env_count, count).copy()
            initial_path = args.output / "initial.npz"
            with initial_path.open("xb") as target:
                np.savez_compressed(target, positions=positions, rest=positions, velocity=velocity, inverse_mass=inverse_mass)
            session.manifest["initial_geometry_sha256"] = sha256(initial_path)
            metadata["initial_positions_equal_rest"] = True
            metadata["initial_state_capture"] = {
                "status": "recorded", "env_count": args.env_count, "particles_per_env": count,
                "scope": "Particles only; no topology or rigid trajectories; rest aliases initial positions for the host state shape contract",
                "is_complete_geometry_evidence": False}
        else:
            metadata["initial_state_capture"] = {
                "status": "unmeasured", "reason": "Particle-free world; no initial particle input is fabricated",
                "is_complete_geometry_evidence": False}
        write_json(args.output / "manifest.json", session.manifest)
        reason = "completed"
        for _ in range(args.steps):
            if np.any(session.step()["env_status"]):
                reason = "physics_failure"
                break
        session.close(reason)
        return {"output": str(args.output), "steps": session.steps, "stop_reason": reason}
    except BaseException as error:
        if session is not None:
            session.close("exception", error=repr(error))
        raise
    finally:
        world.destroy()
        device.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("scene", type=Path)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--build-record", type=Path, required=True)
    capture.add_argument("--steps", type=int, required=True)
    capture.add_argument("--dt", type=float, required=True)
    capture.add_argument("--sweeps", type=int, default=0)
    capture.add_argument("--contact-capacity", type=int, default=0)
    capture.add_argument("--env-count", type=int, default=1)
    capture.add_argument("--state-field", action="append", default=[])
    report = commands.add_parser("report")
    report.add_argument("source", type=Path)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--plot", action="store_true")
    audit = commands.add_parser("audit")
    audit.add_argument("source", type=Path)
    audit.add_argument("--geometry", type=Path, required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("sources", type=Path, nargs="+")
    compare.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify-particles")
    verify.add_argument("source", type=Path)
    verify.add_argument("--geometry", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    elastic = commands.add_parser("verify-elastic")
    elastic.add_argument("source", type=Path)
    elastic.add_argument("--geometry", type=Path, required=True)
    elastic.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "capture":
        print(json.dumps(capture_scene(args)))
    elif args.command == "report":
        result = make_report(args.source, args.output, plot=args.plot)
        print(json.dumps({"output": str(args.output), "passes_full_physics_acceptance": result["passes_full_physics_acceptance"]}))
    elif args.command == "audit":
        print(json.dumps(audit_mesh_trace(args.source, args.geometry)))
    elif args.command in ("verify-particles", "verify-elastic"):
        if args.output.exists():
            raise FileExistsError(args.output)
        result = (verify_particle_trace if args.command == "verify-particles" else verify_elastic_trace)(args.source, args.geometry)
        write_json(args.output, result)
        print(json.dumps(result))
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        result = compare_runs(args.sources)
        write_json(args.output, result)
        print(json.dumps({"output": str(args.output), "comparisons": len(result["comparisons"]),
            "timestep_convergence": result["timestep_convergence"]["status"], "claims_physical_acceptance": False}))


if __name__ == "__main__":
    main()
