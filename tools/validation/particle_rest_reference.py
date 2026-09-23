"""Compare a zero-load structural equilibrium with independent simulators."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np


def quality(positions, velocities, mass):
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    rest = positions[0]
    edges = np.asarray(((0, 1), (1, 2), (2, 0)))
    lengths = np.linalg.norm(positions[:, edges[:, 0]] - positions[:, edges[:, 1]], axis=-1)
    strain = np.abs(lengths / lengths[0] - 1.0)
    displacement = np.linalg.norm(positions - rest, axis=-1)
    momentum = np.linalg.norm(mass * velocities.sum(axis=1), axis=-1)
    finite = bool(np.isfinite(positions).all() and np.isfinite(velocities).all())
    return {"finite": finite, "max_displacement_m": float(displacement.max()),
            "max_distance_strain": float(strain.max()), "max_momentum_kg_m_s": float(momentum.max()),
            "valid": finite and bool(displacement.max() <= 1.0e-7 and strain.max() <= 1.0e-5)}


def run_mujoco(case, output):
    import mujoco

    initial = np.asarray(case["positions"][0], dtype=np.float64)
    bodies = []
    for index, point in enumerate(initial):
        position = " ".join(format(value, ".17g") for value in point)
        bodies.append(f'''<body name="p{index}" pos="{position}">
          <joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 1 0"/>
          <joint type="slide" axis="0 0 1"/>
          <inertial pos="0 0 0" mass="{case['particle_mass_kg']}" diaginertia="1e-6 1e-6 1e-6"/>
        </body>''')
    radius = case["contact_separation_m"] / 2.0
    xml = f'''<mujoco model="structural_rest_equilibrium">
      <option timestep="{case['dt']}" gravity="0 0 0" solver="Newton" iterations="100" tolerance="1e-12"/>
      <worldbody>{''.join(bodies)}</worldbody>
      <deformable><flex name="sheet" dim="2" radius="{radius}" body="p0 p1 p2"
        vertex="0 0 0 0 0 0 0 0 0" element="0 1 2">
        <contact selfcollide="narrow" internal="true"/>
      </flex></deformable>
      <equality><flex flex="sheet"/></equality>
    </mujoco>'''
    (output / "model.xml").write_text(xml + "\n")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    positions, velocities = [data.flexvert_xpos.copy()], [data.qvel.reshape(3, 3).copy()]
    contacts = []
    for _ in range(case["steps"]):
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        positions.append(data.flexvert_xpos.copy())
        velocities.append(data.qvel.reshape(3, 3).copy())
        contacts.append(int(data.ncon))
    return positions, velocities, {
        "version": mujoco.__version__, "solver": "Newton; flex edge equality",
        "device": "CPU", "precision": "float64", "max_contacts": max(contacts),
        "model_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "scope": "zero-load equilibrium only; flex contact and compliance differ from particle XPBD",
    }


def run_newton(case, device):
    import newton
    import warp as wp

    wp.init()
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for point in case["positions"][0]:
        builder.add_particle(wp.vec3(*point), wp.vec3(0.0), case["particle_mass_kg"],
                             radius=case["contact_separation_m"] / 2.0)
    builder.add_triangle(0, 1, 2, tri_ke=1.0e5, tri_ka=1.0e5, tri_kd=0.0)
    builder.color()
    model = builder.finalize(device=device)
    collision = newton.CollisionPipeline(model)
    solver = newton.solvers.SolverVBD(
        model, iterations=24, particle_enable_self_contact=True,
        particle_self_contact_margin=case["contact_separation_m"], particle_self_contact_gap=0.01,
        particle_topological_contact_filter_threshold=1, particle_rest_shape_contact_exclusion_radius=0.0,
        rigid_compliant_alm=True, collision_pipeline=collision)
    state, next_state = model.state(), model.state()
    control = model.control()
    positions, velocities = [state.particle_q.numpy()], [state.particle_qd.numpy()]
    for _ in range(case["steps"]):
        state.clear_forces()
        solver.step(state, next_state, control, None, case["dt"])
        state, next_state = next_state, state
        positions.append(state.particle_q.numpy())
        velocities.append(state.particle_qd.numpy())
    return positions, velocities, {
        "version": importlib.metadata.version("newton"), "warp_version": wp.__version__,
        "solver": "VBD with self-contact and topological filtering", "device": str(model.device),
        "precision": "float32", "iterations": 24, "triangle_stiffness": [1.0e5, 1.0e5],
        "self_contact_margin_m": case["contact_separation_m"], "self_contact_gap_m": 0.01,
        "topological_filter_threshold": 1, "rest_shape_exclusion_radius_m": 0.0,
        "scope": "zero-load equilibrium only; finite triangle elasticity is not hard XPBD distance compliance",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nuka-trace", type=Path, required=True)
    parser.add_argument("--engine", choices=("newton", "mujoco"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-setup", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    case = json.loads(args.nuka_trace.read_text())
    if np.asarray(case["positions"]).shape != (case["steps"] + 1, 3, 3):
        raise ValueError("expected one triangle and the complete Nuka trace")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.engine == "mujoco":
        positions, velocities, engine = run_mujoco(case, args.output)
    else:
        positions, velocities, engine = run_newton(case, args.device)
    setup = json.loads(args.reference_setup.read_text())
    if args.engine == "newton":
        engine["revision"] = setup["newton_revision"]
        engine["source_archive_sha256"] = setup["archive_sha256"]
    trace = args.output / "trace.npz"
    np.savez(trace, positions=np.asarray(positions), velocities=np.asarray(velocities))
    nuka_quality = quality(case["positions"], case["velocities"], case["particle_mass_kg"])
    reference_quality = quality(positions, velocities, case["particle_mass_kg"])
    report = {"engine": args.engine, "reference": engine, "dt": case["dt"], "steps": case["steps"],
              "input_sha256": hashlib.sha256(args.nuka_trace.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "reference_setup": setup, "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
              "nuka": nuka_quality, "reference_quality": reference_quality,
              "oracle": "a rest triangle with zero forces and velocity must remain at rest despite overlapping collision thickness",
              "limits": {"max_displacement_m": 1.0e-7, "max_distance_strain": 1.0e-5},
              "valid": nuka_quality["valid"] and reference_quality["valid"] and case["env_status_union"] == 0}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
