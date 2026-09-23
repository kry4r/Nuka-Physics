"""Evaluate one measured G1 episode per environment without stopping at the first fall."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from nuka.provenance import record_run_identity
from nuka.tasks.g1_locomotion import projected_gravity
from nuka.tasks.g1_policy import G1FusionPolicy
from nuka.tasks.g1_wading import G1WadingTask, WadingConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--finish-x", type=float)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--require-success-rate", type=float, default=1.0)
    args = parser.parse_args()
    if args.envs < 1 or not 0 <= args.require_success_rate <= 1:
        parser.error("Environment count must be positive and success rate must be between zero and one")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    policy, training = G1FusionPolicy.load(args.policy)
    policy.eval()
    configuration = {**training["task"], "num_envs": args.envs, "seed": args.seed}
    for name, value in (("scene", args.scene), ("manifest", args.manifest),
                        ("finish_x", args.finish_x), ("episode_seconds", args.seconds)):
        if value is not None:
            configuration[name] = str(value) if isinstance(value, Path) else value
    task = G1WadingTask(WadingConfig(**configuration))
    results = [None] * args.envs
    started = time.perf_counter()
    try:
        task.validate_policy(policy)
        record_run_identity(Path(__file__).resolve().parents[2], args.out,
            [args.policy, task.config.scene, task.config.manifest, task.config.deployment],
            ["src", "python/src", "python/nuka", "examples/training"])
        (args.out / "config.json").write_text(json.dumps(asdict(task.config), indent=2))
        controller = task.controller
        finished = torch.zeros(args.envs, device=task.device, dtype=torch.bool)
        minimum_up = torch.ones(args.envs, device=task.device)
        maximum_speed = torch.zeros(args.envs, device=task.device)
        squared_speed_error = torch.zeros(args.envs, device=task.device)
        maximum_attitude_error = torch.zeros(args.envs, device=task.device)
        saturated_steps = torch.zeros(args.envs, device=task.device)
        observation = task.observation()
        with torch.no_grad():
            for step in range(task.episode_limit):
                observation, _, terminated, truncated, info = task.step(policy(observation))
                active = ~finished
                gravity = projected_gravity(controller.base[:, 3:])
                up = -gravity[:, 2]
                velocity = controller.velocity[:, 0, 3:5]
                error = (velocity - controller.commands[:, :2]).square().sum(-1)
                cosine = torch.nn.functional.cosine_similarity(controller.proprioception.gravity, gravity, dim=-1)
                attitude_error = torch.rad2deg(cosine.clamp(-1, 1).acos())
                minimum_up = torch.where(active, torch.minimum(minimum_up, up), minimum_up)
                maximum_speed = torch.where(active, torch.maximum(maximum_speed, velocity.norm(dim=-1)), maximum_speed)
                squared_speed_error += active * error
                maximum_attitude_error = torch.where(active,
                    torch.maximum(maximum_attitude_error, attitude_error), maximum_attitude_error)
                saturated_steps += active * controller.saturated[:, controller.joint_slots].bool().any(-1)
                done = terminated | truncated
                for env in (done & active).nonzero().flatten().cpu().tolist():
                    count = int(info["steps"][env])
                    result = {name: float(info[name][env]) for name in
                        ("return", "progress", "fallen", "off_course", "reached", "finish_hold_seconds")}
                    result.update(env=env, steps=count, seconds=count * task.dt,
                        timed_out=bool(truncated[env]), final_base=controller.base[env].cpu().tolist(),
                        minimum_upright=float(minimum_up[env]), maximum_speed=float(maximum_speed[env]),
                        rms_speed_error=float((squared_speed_error[env] / count).sqrt()),
                        maximum_attitude_error_degrees=float(maximum_attitude_error[env]),
                        saturated_step_fraction=float(saturated_steps[env] / count))
                    results[env] = result
                    print(json.dumps(result), flush=True)
                finished |= done
                if bool(finished.all()):
                    break
                if bool(done.any()):
                    observation = task.reset(done.nonzero().flatten())
                if (step + 1) % 100 == 0:
                    print(json.dumps({"step": step + 1, "finished": int(finished.sum())}), flush=True)
        if any(result is None for result in results):
            raise RuntimeError("Evaluation ended before every environment produced an outcome")
        passed = sum(int(result["reached"]) for result in results)
        report = {"policy": str(args.policy), "training_iteration": training.get("iteration"),
            "seed": args.seed, "environments": args.envs, "successes": passed,
            "success_rate": passed / args.envs,
            "falls": sum(int(result["fallen"]) for result in results),
            "off_course": sum(int(result["off_course"]) for result in results),
            "time_outs": sum(int(result["timed_out"]) for result in results),
            "camera_enabled": task.camera is not None, "terrain_enabled": task.config.terrain is not None,
            "load_source": task.controller.proprioception.config.load_source, "finish_x": task.finish_x,
            "required_finish_hold_seconds": task.config.finish_hold_seconds,
            "wall_seconds": time.perf_counter() - started, "episodes": results}
        (args.out / "metrics.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({key: value for key, value in report.items() if key != "episodes"}), flush=True)
        if report["success_rate"] < args.require_success_rate:
            raise RuntimeError("The policy did not meet the required success rate; see metrics.json")
    finally:
        task.close()


if __name__ == "__main__":
    main()
