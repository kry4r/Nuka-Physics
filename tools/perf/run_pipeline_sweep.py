"""Measure complete production pipelines in independent processes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


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
    parser.add_argument("--envs", type=int, nargs="+", default=[1, 16, 256])
    parser.add_argument("--executions", nargs="+", choices=["eager", "graph"], default=["eager", "graph"])
    parser.add_argument("--processes", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=250)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--capacity-scale", type=int, default=1)
    parser.add_argument("--cloth-grid", type=int)
    parser.add_argument("--cuda-launch-queues", choices=["default", "0.25x", "0.5x", "2x", "4x"])
    parser.add_argument("--save-state", action="store_true")
    parser.add_argument("--save-wrench", action="store_true")
    args = parser.parse_args()
    if args.processes < 1 or any(count < 1 for count in args.envs):
        parser.error("positive process and environment counts required")
    binary = args.binary.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    binary_hash = digest(binary)
    results = []
    environment = dict(os.environ)
    if args.cuda_launch_queues == "default":
        environment.pop("CUDA_SCALE_LAUNCH_QUEUES", None)
    elif args.cuda_launch_queues:
        environment["CUDA_SCALE_LAUNCH_QUEUES"] = args.cuda_launch_queues
    environment["LD_LIBRARY_PATH"] = str(binary.parent) + ":" + environment.get("LD_LIBRARY_PATH", "")
    runner_hash = digest(Path(__file__))
    for envs in args.envs:
        for execution in args.executions:
            for process in range(args.processes):
                label = f"{execution}_e{envs}_p{process}"
                result_path = (args.output / (label + ".json")).resolve()
                command = [str(binary), "--scene", "robot-cloth-fluid", "--envs", str(envs),
                           "--execution", execution, "--warmup", str(args.warmup), "--steps", str(args.steps),
                           "--seed", "20260908", "--capacity-scale", str(args.capacity_scale),
                           "--perf-json", str(result_path)]
                if args.cloth_grid is not None:
                    command.extend(["--cloth-grid", str(args.cloth_grid)])
                state_path = args.output / (label + ".state") if args.save_state and process == 0 else None
                wrench_path = args.output / (label + ".wrench") if args.save_wrench and process == 0 else None
                if state_path is not None:
                    command.extend(["--state-output", str(state_path.resolve())])
                if wrench_path is not None:
                    command.extend(["--wrench-output", str(wrench_path.resolve())])
                before = gpu_state()
                start = time.perf_counter()
                with (args.output / (label + ".log")).open("w") as log:
                    completed = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
                receipt = {"label": label, "command": command, "source_id": args.source_id,
                           "binary_sha256": binary_hash, "exit_code": completed.returncode,
                           "runner_sha256": runner_hash,
                           "cuda_environment": {key: environment.get(key) for key in
                               ("CUDA_SCALE_LAUNCH_QUEUES", "CUDA_DEVICE_MAX_CONNECTIONS", "CUDA_LAUNCH_BLOCKING")},
                           "wall_seconds": time.perf_counter() - start,
                           "gpu_before": before, "gpu_after": gpu_state()}
                if state_path is not None and state_path.exists():
                    receipt["state_sha256"] = digest(state_path)
                    receipt["state_bytes"] = state_path.stat().st_size
                if wrench_path is not None and wrench_path.exists():
                    receipt["wrench_sha256"] = digest(wrench_path)
                    receipt["wrench_bytes"] = wrench_path.stat().st_size
                (args.output / (label + "_command.json")).write_text(json.dumps(receipt, indent=2) + "\n")
                print(json.dumps(receipt), flush=True)
                if completed.returncode or not result_path.exists():
                    raise SystemExit(f"benchmark failed: {result_path}")
                result = json.loads(result_path.read_text())
                if not result["status"]["valid"]:
                    raise SystemExit(f"invalid quality: {result_path}")
                results.append({"envs": envs, "execution": execution, "process": process, "result": result})
    summary = []
    for envs in args.envs:
        for execution in args.executions:
            samples = [entry["result"] for entry in results if entry["envs"] == envs and entry["execution"] == execution]
            gpu = [sample["timing"]["batch_step_ms"] for sample in samples]
            wall = [sample["timing"]["synchronized_wall_ms"] / args.steps for sample in samples]
            states = sorted({sample["quality"]["state_fnv1a64"] for sample in samples})
            wrench_samples = [sample["quality"].get("link_wrench_trace_fnv1a64") for sample in samples]
            wrenches = sorted({value for value in wrench_samples if value is not None})
            wrench_consistent = len(wrenches) <= 1 and (not wrenches or all(wrench_samples))
            schedule_samples = [sample.get("workload", {}).get("samples") for sample in samples]
            schedules = sorted({hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
                                for value in schedule_samples if value is not None})
            schedule_consistent = len(schedules) <= 1 and (not schedules or all(schedule_samples))
            summary.append({"envs": envs, "execution": execution, "processes": len(samples),
                            "gpu_batch_step_ms_median": statistics.median(gpu), "gpu_range_ms": [min(gpu), max(gpu)],
                            "wall_batch_step_ms_median": statistics.median(wall), "wall_range_ms": [min(wall), max(wall)],
                            "final_state_digests": states, "wrench_trace_digests": wrenches,
                            "wrench_trace_available": bool(wrenches),
                            "island_schedule_digests": schedules, "island_schedule_available": bool(schedules),
                            "cross_process_d1": len(states) == 1 and wrench_consistent and schedule_consistent})
    report = {"source_id": args.source_id, "binary_sha256": binary_hash, "summary": summary}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not all(entry["cross_process_d1"] for entry in summary):
        raise SystemExit("cross-process state identity failed")


if __name__ == "__main__":
    main()
