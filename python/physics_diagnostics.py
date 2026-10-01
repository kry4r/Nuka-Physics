"""Capture, analyze, audit and compare production physics evidence."""

import argparse
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
    tolerance = os.environ.get("NUKA_SOLVER_VEL_TOLERANCE")
    if tolerance is None:
        raise ValueError("set the production NUKA_SOLVER_VEL_TOLERANCE explicitly for reproducibility")
    fields = tuple(getattr(nuka.Field, name) for name in args.state_field)
    device = nuka.Device.create(0)
    world = nuka.World.create_from_scene(device, str(args.scene), env_count=args.env_count, dt=args.dt,
                                         solver_vel_iters=args.sweeps, ogc_contact_capacity=args.contact_capacity)
    session = None
    try:
        metadata = {"scene": str(args.scene), "scene_sha256": sha256(args.scene), "dt": args.dt,
            "sweeps": args.sweeps, "expected_policy_steps": args.steps,
            "ogc_contact_capacity": args.contact_capacity or None,
            "solver_velocity_tolerance_mps": float(tolerance),
            "controls": "Authored scene defaults; no external inputs",
            "owner_names": {"LINK": list(world.dof_names())},
            "kinematic_tree": [{key: link[key] for key in ("parent_index", "articulation_index", "joint_type")}
                               for link in world.kinematic_tree()],
            "execution_info": json.loads(json.dumps(world.execution_info, default=str)),
            "build_record": str(args.build_record),
            "binary_hashes": (args.build_record / "binaries.sha256").read_text(),
            "fixture_sha256": sha256(__file__)}
        session = DiagnosticSession(world, args.output, metadata, env_count=args.env_count, state_fields=fields,
                                    thresholds=DiagnosticThresholds(velocity_tolerance_mps=float(tolerance)))
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
        print(json.dumps({"output": str(args.output), "comparisons": len(result["comparisons"])}))


if __name__ == "__main__":
    main()
