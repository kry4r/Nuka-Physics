"""Measure complete production pipelines in independent processes."""

import argparse
from contextlib import ExitStack
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import time


def elapsed_time():
    if hasattr(time, "CLOCK_MONOTONIC_RAW"):
        return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
    return time.perf_counter()


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def relevant_environment(environment):
    prefixes = ("NUKA_", "PHI_", "CUDA_", "NVIDIA_", "NV_", "CUBLAS_", "CUDNN_",
                "PTXAS_", "LD_", "OMP_", "TBB_", "OPENBLAS_", "GL_", "EGL_", "VK_")
    names = ("PATH", "PYTHONPATH", "DISPLAY", "XDG_RUNTIME_DIR", "CUDA_VISIBLE_DEVICES",
             "NUKA_BLOCK_DESCENT", "NUKA_BLOCK_DESCENT_ITERATIONS",
             "NUKA_BLOCK_DESCENT_SPECTRAL_RADIUS", "NUKA_SOLVER_VEL_TOLERANCE")
    return {key: environment.get(key) for key in sorted(set(names) | {
        key for key in environment if key.startswith(prefixes)})}


def dependency_evidence(binary, environment):
    evidence = {}
    for name, command in (("elf_dynamic_section", ["readelf", "-d", str(binary)]),
                          ("loader_resolution", ["ldd", str(binary)])):
        try:
            result = subprocess.run(command, env=environment, text=True, capture_output=True)
            evidence[name] = {"command": command, "exit_code": result.returncode,
                              "stdout": result.stdout, "stderr": result.stderr}
        except OSError as error:
            evidence[name] = {"command": command, "unavailable": str(error)}
    dynamic = evidence["elf_dynamic_section"].get("stdout", "")
    evidence["dt_needed"] = re.findall(r"\(NEEDED\).*\[([^\]]+)\]", dynamic)
    evidence["runpath_rpath"] = re.findall(r"\((?:RUNPATH|RPATH)\).*\[([^\]]*)\]", dynamic)
    return evidence


def mapped_identity_matches(metadata, record):
    device = tuple(int(part, 16) for part in record["device"].split(":"))
    return (metadata.st_ino == int(record["inode"]) and
            (os.major(metadata.st_dev), os.minor(metadata.st_dev)) == device)


def observe_loaded_images(process, images, errors, sources, mapped_files):
    try:
        lines = Path(f"/proc/{process.pid}/maps").read_text().splitlines()
    except OSError as error:
        if process.poll() is None:
            errors.add(str(error))
        return False
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or "x" not in fields[1] or not fields[5].startswith("/"):
            continue
        path = Path(fields[5])
        identity = (fields[3], fields[4], str(path))
        if identity in images:
            continue
        record = {"mapped_path": str(path), "device": fields[3], "inode": fields[4],
                  "identity_source": "unverified", "identity_status": "unverified", "identity_errors": []}
        map_file = Path(f"/proc/{process.pid}/map_files/{fields[0]}")
        for source_kind, source_path in (("proc_map_files", map_file), ("path_identity_fallback", path)):
            source = None
            try:
                source = mapped_files.enter_context(source_path.open("rb"))
                metadata = os.fstat(source.fileno())
                if not mapped_identity_matches(metadata, record):
                    raise OSError("opened file identity does not match maps device/inode")
                record.update({"resolved_path": str(path.resolve()), "bytes": metadata.st_size,
                               "mtime_ns": metadata.st_mtime_ns, "ctime_ns": metadata.st_ctime_ns,
                               "identity_source": source_kind, "identity_status": "verified",
                               "identity_file": str(source_path)})
                sources[identity] = source
                break
            except OSError as error:
                if source is not None:
                    source.close()
                record["identity_errors"].append({"source": source_kind, "path": str(source_path), "error": str(error)})
        if record["identity_status"] != "verified":
            record["unavailable"] = "no readable file descriptor matched the mapped device/inode"
        images[identity] = record
    return True


def hash_loaded_images(images, sources):
    loaded = sorted(images.values(), key=lambda item: item["mapped_path"])
    for record in loaded:
        if "unavailable" in record:
            continue
        identity = (record["device"], record["inode"], record["mapped_path"])
        source = sources[identity]
        try:
            metadata = os.fstat(source.fileno())
            if (not mapped_identity_matches(metadata, record) or
                    metadata.st_size != record["bytes"] or metadata.st_mtime_ns != record["mtime_ns"] or
                    metadata.st_ctime_ns != record["ctime_ns"]):
                raise OSError("retained mapped file changed before hashing")
            record["sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
            after = os.fstat(source.fileno())
            if (not mapped_identity_matches(after, record) or
                    (after.st_size, after.st_mtime_ns, after.st_ctime_ns) !=
                    (metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns)):
                raise OSError("retained mapped file changed while hashing")
        except OSError as error:
            record.pop("sha256", None)
            record["identity_status"] = "unverified"
            record["unavailable"] = str(error)
    return loaded


def run_process(command, environment, log_path):
    images, errors, observed = {}, set(), False
    sources = {}
    start = elapsed_time()
    sampling = {"maximum_seconds": 2.0, "maximum_samples": 20, "interval_seconds": 0.1,
                "samples_taken": 0, "window_seconds": 0.0, "stopped_when": "child_not_started",
                "executable_observed": False, "cuda_runtime_observed": False, "cuda_driver_observed": False}
    executable = str(Path(command[0]).resolve())
    with log_path.open("w") as log, ExitStack() as mapped_files:
        try:
            with subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT) as process:
                sampling_start = elapsed_time()
                sampling["stopped_when"] = "initialization_sampling_limit"
                while (process.poll() is None and sampling["samples_taken"] < sampling["maximum_samples"] and
                       elapsed_time() - sampling_start < sampling["maximum_seconds"]):
                    observed |= observe_loaded_images(process, images, errors, sources, mapped_files)
                    sampling["samples_taken"] += 1
                    paths = {item.get("resolved_path") for item in images.values() if "resolved_path" in item}
                    names = {Path(path).name for path in paths}
                    sampling["executable_observed"] = executable in paths
                    sampling["cuda_runtime_observed"] = any(re.fullmatch(r"libcudart\.so(?:\..*)?", name) for name in names)
                    sampling["cuda_driver_observed"] = any(re.fullmatch(r"libcuda\.so(?:\..*)?", name) for name in names)
                    if all(sampling[key] for key in ("executable_observed", "cuda_runtime_observed", "cuda_driver_observed")):
                        sampling["stopped_when"] = "executable_cuda_runtime_driver_observed"
                        break
                    remaining = sampling["maximum_seconds"] - (elapsed_time() - sampling_start)
                    if remaining <= 0.0:
                        break
                    try:
                        process.wait(timeout=min(sampling["interval_seconds"], remaining))
                    except subprocess.TimeoutExpired:
                        pass
                sampling["window_seconds"] = elapsed_time() - sampling_start
                if process.poll() is not None:
                    sampling["stopped_when"] = "child_exited"
                exit_code = process.wait()
        except OSError as error:
            errors.add(str(error))
            exit_code = None
        wall_seconds = elapsed_time() - start
        loaded = hash_loaded_images(images, sources)
    core = [item for item in loaded if re.fullmatch(r"libnuka(?:_core)?\.so(?:\..*)?",
                                                  Path(item["mapped_path"]).name)]
    identity_verified = observed and bool(loaded) and all("sha256" in record for record in loaded)
    evidence = {"status": "measured" if identity_verified else "unmeasured",
                "mappings_observation_status": "measured" if observed else "unmeasured",
                "scope": "actual child executable mappings observed only during the bounded initialization window",
                "sampling": sampling,
                "hash_boundary": "retained read-only descriptors after child completion; maps identity and size/timestamps must remain unchanged",
                "images": loaded, "core_shared_libraries": core, "errors": sorted(errors),
                "core_loading_status": "observed" if core else
                                       "no_core_shared_library_observed" if observed else "unmeasured"}
    return exit_code, wall_seconds, evidence


def file_evidence(path):
    try:
        path = path.resolve(strict=True)
        return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}
    except OSError as error:
        return {"path": str(path), "unavailable": str(error)}


def budget_evidence(configuration, requested, environment):
    solver = configuration.get("solver", {})
    actual = solver.get("actual_velocity_iterations")
    actual_valid = type(actual) is int and actual > 0
    matches = (solver.get("requested_velocity_iterations") == requested and
               solver.get("configured_velocity_iterations") == requested and
               configuration.get("vel_iters") == requested and solver.get("request_explicit") is True)
    recorded = solver.get("environment", {})
    matches &= all(recorded.get(key) == environment.get(key) for key in
                   ("NUKA_BLOCK_DESCENT", "NUKA_BLOCK_DESCENT_ITERATIONS",
                    "NUKA_BLOCK_DESCENT_SPECTRAL_RADIUS", "NUKA_SOLVER_VEL_TOLERANCE"))
    expected = requested if solver.get("mode") == "row_island" or solver.get("velocity_iterations_override") is True else None
    if solver.get("mode") == "block_descent" and expected is None:
        value = environment.get("NUKA_BLOCK_DESCENT_ITERATIONS")
        if value is not None:
            expected = int(value) if value.isascii() and value.isdigit() and int(value) > 0 else 0
    matches &= actual_valid and (expected is None or actual == expected)
    calls = solver.get("solve_calls", [])
    if solver.get("mode") == "block_descent":
        matches &= bool(calls) and all(call.get("iterations") == actual for call in calls)
    return {"status": "verified" if matches else "unmeasured",
            "requested_matches_actual": actual == requested if actual_valid else None,
            "expected_from_explicit_priority": expected,
            "actual_budget": actual, "scope": "scheduled budget and recorded initialization priority; executed sweeps unmeasured"}


def acceptance(result, exit_code, binding_valid):
    status = result.get("status", {})
    basic = status.get("valid") is True and exit_code == 0
    physical = status.get("physical_acceptance_status", "unmeasured")
    full = status.get("passes_full_physics_acceptance") is True and physical == "passed"
    outcome = "failed" if not basic or physical == "failed" else "unmeasured"
    return {"valid": basic, "passes_full_physics_acceptance": full,
            "physical_acceptance_status": physical, "evidence_binding_valid": binding_valid,
            "full_physics_evidence_binding_status": "unmeasured",
            "performance_acceptance_reason": "independent full physics evidence and asset/source closure are not integrated",
            "performance_acceptance_status": outcome,
            "accepted_performance": False,
            "measurement_role": "failing_baseline" if outcome == "failed" else "unaccepted_baseline"}


def gpu_state():
    query = ["nvidia-smi", "--query-gpu=name,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used,utilization.gpu,utilization.memory",
             "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(query, text=True, capture_output=True, check=True)
        return {"query": query, "values": result.stdout.strip()}
    except (OSError, subprocess.SubprocessError) as error:
        return {"unavailable": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--scene", default="robot-cloth-fluid")
    parser.add_argument("--dt", type=float)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--velocity-iterations", type=int, default=48)
    parser.add_argument("--envs", type=int, nargs="+", default=[1, 16, 256])
    parser.add_argument("--executions", nargs="+", choices=["eager", "graph"], default=["eager", "graph"])
    parser.add_argument("--processes", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=250)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--capacity-scale", type=int, default=1)
    parser.add_argument("--cloth-grid", type=int)
    parser.add_argument("--mpm-implicit-stress", type=int, choices=[0, 1])
    parser.add_argument("--state-sensors", type=int, choices=[0, 1, 2])
    parser.add_argument("--tactile-grid", type=int)
    parser.add_argument("--input-file", type=Path, action="append", default=[])
    parser.add_argument("--allow-unaccepted-baseline", action="store_true",
                        help="allow an unaccepted baseline to finish successfully; acceptance remains unmeasured")
    parser.add_argument("--cuda-launch-queues", choices=["default", "0.25x", "0.5x", "2x", "4x"])
    parser.add_argument("--save-state", action="store_true")
    parser.add_argument("--save-wrench", action="store_true")
    parser.add_argument("--render-sensors", type=int, default=0)
    parser.add_argument("--render-width", type=int, default=256)
    parser.add_argument("--render-height", type=int, default=256)
    parser.add_argument("--render-samples", type=int, default=4)
    parser.add_argument("--render-warmup", type=int, default=32)
    parser.add_argument("--render-shadows", type=int)
    parser.add_argument("--render-ao", type=int)
    parser.add_argument("--imaging-models", type=int, choices=[0, 1])
    parser.add_argument("--render-output", type=Path,
                        help="directory for one render output per independent process")
    args = parser.parse_args()
    if (args.processes < 1 or any(count < 1 for count in args.envs) or args.steps < 1 or
            args.warmup < 0 or args.substeps < 1 or not 0 < args.velocity_iterations <= 65535 or
            args.capacity_scale < 1 or (args.tactile_grid is not None and args.tactile_grid < 0)):
        parser.error("invalid process, environment, step or iteration counts")
    if args.dt is not None and (not math.isfinite(args.dt) or args.dt <= 0):
        parser.error("positive finite timestep required")
    if len(set(args.envs)) != len(args.envs) or len(set(args.executions)) != len(args.executions):
        parser.error("duplicate configurations would overwrite process evidence")
    if args.render_output is not None and not args.render_sensors:
        parser.error("--render-output requires --render-sensors")
    binary = args.binary.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.render_output is not None:
        if args.render_output.resolve() != args.output.resolve():
            args.render_output.mkdir(parents=True, exist_ok=False)
    binary_hash = digest(binary)
    results = []
    environment = dict(os.environ)
    if args.cuda_launch_queues == "default":
        environment.pop("CUDA_SCALE_LAUNCH_QUEUES", None)
    elif args.cuda_launch_queues:
        environment["CUDA_SCALE_LAUNCH_QUEUES"] = args.cuda_launch_queues
    environment["LD_LIBRARY_PATH"] = str(binary.parent) + ":" + environment.get("LD_LIBRARY_PATH", "")
    runner_hash = digest(Path(__file__))
    recorded_environment = relevant_environment(environment)
    dependencies = dependency_evidence(binary, environment)
    input_files = [file_evidence(path) for path in args.input_file]
    requested_names = ("scene", "dt", "substeps", "velocity_iterations", "warmup", "steps", "capacity_scale",
                       "cloth_grid", "mpm_implicit_stress", "state_sensors", "tactile_grid", "render_sensors",
                       "render_width", "render_height", "render_samples", "render_warmup", "render_shadows",
                       "render_ao", "imaging_models")
    requested = {name: getattr(args, name) for name in requested_names}
    requested["seed"] = 20260908
    for envs, execution, process in itertools.product(args.envs, args.executions, range(args.processes)):
        label = f"{execution}_e{envs}_p{process}"
        result_path = (args.output / (label + ".json")).resolve()
        command = [str(binary), "--scene", args.scene, "--envs", str(envs),
                   "--execution", execution, "--warmup", str(args.warmup), "--steps", str(args.steps),
                   "--substeps", str(args.substeps), "--velocity-iterations", str(args.velocity_iterations),
                   "--seed", "20260908", "--capacity-scale", str(args.capacity_scale), "--perf-json", str(result_path)]
        for name in ("dt", "cloth_grid", "mpm_implicit_stress", "state_sensors", "tactile_grid",
                     "render_shadows", "render_ao", "imaging_models"):
            value = getattr(args, name)
            if value is not None:
                command.extend(["--" + name.replace("_", "-"), str(value)])
        if args.render_sensors:
            for name in ("render_sensors", "render_width", "render_height", "render_samples", "render_warmup"):
                command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        state_path = args.output / (label + ".state") if args.save_state and process == 0 else None
        wrench_path = args.output / (label + ".wrench") if args.save_wrench and process == 0 else None
        render_path = args.render_output / (label + ".render") if args.render_output is not None else None
        for name, path in (("state", state_path), ("wrench", wrench_path), ("render", render_path)):
            if path is not None:
                command.extend(["--" + name + "-output", str(path.resolve())])
        before = gpu_state()
        exit_code, wall_seconds, loaded = run_process(command, environment, args.output / (label + ".log"))
        receipt = {"label": label, "command": command, "cwd": str(Path.cwd()), "source_id": args.source_id,
                   "binary_sha256": binary_hash, "binary_after": file_evidence(binary), "exit_code": exit_code,
                   "runner_sha256": runner_hash, "environment": recorded_environment,
                   "dynamic_dependencies": dependencies, "loaded_images": loaded,
                   "cuda_environment": {key: environment.get(key) for key in
                       ("CUDA_SCALE_LAUNCH_QUEUES", "CUDA_DEVICE_MAX_CONNECTIONS", "CUDA_LAUNCH_BLOCKING")},
                   "wall_seconds": wall_seconds,
                   "host_clock": "CLOCK_MONOTONIC_RAW" if hasattr(time, "CLOCK_MONOTONIC_RAW") else "perf_counter",
                   "gpu_before": before, "gpu_after": gpu_state()}
        result = {}
        try:
            result = json.loads(result_path.read_text())
            if not isinstance(result, dict):
                raise ValueError("benchmark JSON must be an object")
            for key in ("status", "config", "timing", "quality", "workload"):
                if key in result and not isinstance(result[key], dict):
                    raise ValueError("benchmark JSON section must be an object: " + key)
            receipt["result_sha256"] = digest(result_path)
        except (OSError, ValueError) as error:
            receipt["result_error"] = str(error)
            result = {}
        assets = input_files.copy()
        scene_source = result.get("config", {}).get("scene_source_path")
        if scene_source:
            assets.append(file_evidence(Path(scene_source)))
        payload = {"source_id": args.source_id, "binary_sha256": binary_hash, "runner_sha256": runner_hash,
                   "requested": dict(requested, envs=envs, execution=execution), "actual_config": result.get("config"),
                   "environment": recorded_environment, "input_files": assets,
                   "loaded_image_hashes": [{key: item.get(key) for key in ("resolved_path", "sha256")}
                                           for item in loaded["images"]]}
        receipt["input_configuration"] = payload
        receipt["input_configuration_sha256"] = signature(payload)
        receipt["input_asset_closure_status"] = "unmeasured"
        receipt["source_manifest_status"] = "unmeasured"
        receipt["evidence_binding_scope"] = "observed binary mappings, requested/actual configuration and listed input files"
        receipt["budget_evidence"] = budget_evidence(result.get("config", {}), args.velocity_iterations, environment)
        mapped_binary = [item for item in loaded["images"] if item.get("resolved_path") == str(binary)]
        binding_valid = (receipt["binary_after"].get("sha256") == binary_hash and
                         loaded["status"] == "measured" and not loaded["errors"] and
                         bool(mapped_binary) and all(item.get("sha256") == binary_hash for item in mapped_binary) and
                         all("sha256" in item for item in loaded["images"]) and
                         all(item.get("exit_code") == 0 for key, item in dependencies.items()
                             if key in ("elf_dynamic_section", "loader_resolution")) and
                         all("sha256" in item for item in assets) and receipt["budget_evidence"]["status"] == "verified")
        receipt["acceptance"] = acceptance(result, exit_code, binding_valid)
        for name, path in (("state", state_path), ("wrench", wrench_path), ("render", render_path)):
            if path is not None and path.exists():
                receipt[name + "_sha256"] = digest(path)
                receipt[name + "_bytes"] = path.stat().st_size
        (args.output / (label + "_command.json")).write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt), flush=True)
        results.append({"envs": envs, "execution": execution, "process": process, "result": result,
                        "receipt": receipt})
        if not receipt["acceptance"]["valid"] or receipt["acceptance"]["physical_acceptance_status"] == "failed":
            break
    summary = []
    for envs in args.envs:
        for execution in args.executions:
            entries = [entry for entry in results if entry["envs"] == envs and entry["execution"] == execution]
            samples = [entry["result"] for entry in entries]
            gpu = [sample["timing"]["batch_step_ms"] for sample in samples
                   if "batch_step_ms" in sample.get("timing", {})]
            wall = [sample["timing"]["synchronized_wall_ms"] / args.steps for sample in samples
                    if "synchronized_wall_ms" in sample.get("timing", {})]
            state_samples = [sample.get("quality", {}).get("state_fnv1a64") for sample in samples]
            states = sorted({value for value in state_samples if value is not None})
            wrench_samples = [sample.get("quality", {}).get("link_wrench_trace_fnv1a64") for sample in samples]
            wrenches = sorted({value for value in wrench_samples if value is not None})
            wrench_consistent = len(wrenches) <= 1 and (not wrenches or all(wrench_samples))
            schedule_samples = [sample.get("workload", {}).get("samples") for sample in samples]
            schedules = sorted({hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
                                for value in schedule_samples if value is not None})
            schedule_consistent = len(schedules) <= 1 and (not schedules or all(schedule_samples))
            identities = sorted({entry["receipt"]["input_configuration_sha256"] for entry in entries})
            complete = len(samples) == args.processes
            cross_process = (complete and all(state_samples) and len(states) == 1 and
                             wrench_consistent and schedule_consistent and len(identities) == 1)
            basic_valid = complete and all(entry["receipt"]["acceptance"]["valid"] for entry in entries)
            full = complete and all(entry["receipt"]["acceptance"]["passes_full_physics_acceptance"] for entry in entries)
            accepted = (basic_valid and full and cross_process and args.processes == 5 and
                        all(entry["receipt"]["acceptance"]["accepted_performance"] for entry in entries))
            any_failed = any(entry["receipt"]["acceptance"]["performance_acceptance_status"] == "failed"
                             for entry in entries)
            outcome = "passed" if accepted else "failed" if any_failed or (complete and not cross_process) else "unmeasured"
            summary.append({"envs": envs, "execution": execution, "processes": len(samples),
                            "requested_processes": args.processes, "valid": basic_valid,
                            "execution_status": "complete" if complete else "incomplete" if entries else "unmeasured",
                            "passes_full_physics_acceptance": full, "accepted_performance": accepted,
                            "full_physics_evidence_binding_status": "unmeasured",
                            "performance_acceptance_status": outcome,
                            "measurement_role": "accepted_measurement" if accepted else
                                                "failing_baseline" if outcome == "failed" else "unaccepted_baseline",
                            "gpu_batch_step_ms_median": statistics.median(gpu) if gpu else None,
                            "gpu_range_ms": [min(gpu), max(gpu)] if gpu else None,
                            "wall_batch_step_ms_median": statistics.median(wall) if wall else None,
                            "wall_range_ms": [min(wall), max(wall)] if wall else None,
                            "final_state_digests": states, "wrench_trace_digests": wrenches,
                            "wrench_trace_available": bool(wrenches),
                            "island_schedule_digests": schedules, "island_schedule_available": bool(schedules),
                            "input_configuration_digests": identities,
                            "cross_process_d1": cross_process})
    report = {"schema_version": 2, "source_id": args.source_id, "binary_sha256": binary_hash,
              "summary": summary, "process_results": results,
              "accepted_performance": all(entry["accepted_performance"] for entry in summary),
              "acceptance_scope": "full physics acceptance, bound inputs and binaries, and five independent processes",
              "input_asset_closure_status": "unmeasured", "source_manifest_status": "unmeasured"}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not all(entry["valid"] and entry["cross_process_d1"] for entry in summary):
        raise SystemExit("benchmark or cross-process identity failed; measurements retained in summary.json")
    if any(entry["performance_acceptance_status"] == "failed" for entry in summary):
        raise SystemExit("physics acceptance failed; measurements retained in summary.json")
    if not report["accepted_performance"] and not args.allow_unaccepted_baseline:
        raise SystemExit("performance acceptance is unmeasured; measurements retained as unaccepted baselines")


if __name__ == "__main__":
    main()
