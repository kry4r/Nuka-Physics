"""Train and export a measured G1 locomotion residual with optional RGB-D and clipped PPO."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import yaml

from nuka.provenance import record_run_identity
from nuka.tasks.g1_locomotion import G1VelocityActor
from nuka.tasks.g1_policy import G1FusionConfig, G1FusionPolicy
from nuka.tasks.g1_wading import G1WadingTask, WadingConfig


def copy_observation(observation):
    return {name: value.detach().clone() for name, value in observation.items()}


def generalized_advantage(reward, value, next_value, terminated, truncated, gamma, gae_lambda):
    advantages = torch.zeros_like(reward)
    accumulated = torch.zeros_like(reward[0])
    for t in reversed(range(len(reward))):
        delta = reward[t] + gamma * next_value[t] * ~terminated[t] - value[t]
        accumulated = delta + gamma * gae_lambda * ~(terminated[t] | truncated[t]) * accumulated
        advantages[t] = accumulated
    return advantages, advantages + value


def train(task, policy, optimizer, configuration, output, start_iteration=0, provenance=None,
          start_transitions=0):
    horizon = int(configuration.get("horizon", 128))
    epochs = int(configuration.get("update_epochs", 4))
    batch_size = int(configuration.get("minibatch", 128))
    iterations = int(configuration.get("iterations", 200))
    gamma = float(configuration.get("gamma", 0.99))
    gae_lambda = float(configuration.get("gae_lambda", 0.95))
    clip = float(configuration.get("clip", 0.2))
    target_kl = float(configuration.get("target_kl", 0.02))
    minimum_lr = float(configuration.get("minimum_learning_rate", 1e-6))
    maximum_lr = float(configuration.get("learning_rate", 3e-5))
    if min(horizon, epochs, batch_size, iterations) <= 0:
        raise ValueError("PPO rollout, minibatch, epoch and iteration counts must be positive")
    if not 0 < minimum_lr <= maximum_lr or not 0 < target_kl:
        raise ValueError("PPO learning rates and target KL must be ordered and positive")
    parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    observation = task.observation()
    transitions = int(start_transitions)
    for iteration in range(start_iteration, start_iteration + iterations):
        started = time.perf_counter()
        storage = {name: [] for name in ("privileged", "latent", "mean", "std", "log_probability", "value", "next_value",
                                        "reward", "terminated", "truncated")}
        observations = {name: [] for name in observation}
        episodes = []
        reward_terms = {}
        policy.eval()
        with torch.no_grad():
            for _ in range(horizon):
                for name, value in copy_observation(observation).items():
                    observations[name].append(value)
                privileged = task.critic_observation()
                distribution = policy.distribution(observation)
                latent = distribution.sample()
                log_probability = distribution.log_prob(latent).sum(-1)
                action = policy.action(observation, latent)
                value = policy.critic(observation, privileged)
                next_observation, reward, terminated, truncated, info = task.step(action)
                next_value = policy.critic(next_observation, task.critic_observation())
                record = {"privileged": privileged, "latent": latent,
                          "mean": distribution.mean, "std": distribution.stddev,
                          "log_probability": log_probability,
                          "value": value, "next_value": next_value, "reward": reward,
                          "terminated": terminated, "truncated": truncated}
                for name, value in record.items():
                    storage[name].append(value.clone())
                for name, value in info["reward_terms"].items():
                    reward_terms[name] = reward_terms.get(name, 0.0) + float(value.mean()) / horizon
                done = terminated | truncated
                if bool(done.any()):
                    ids = done.nonzero().flatten()
                    for env in ids.cpu().tolist():
                        episodes.append({name: float(info[name][env]) for name in
                                         ("return", "steps", "progress", "fallen", "off_course", "reached")})
                    observation = task.reset(ids)
                else:
                    observation = next_observation
        rollout_seconds = time.perf_counter() - started
        storage = {name: torch.stack(values) for name, values in storage.items()}
        advantages, returns = generalized_advantage(storage["reward"], storage["value"],
            storage["next_value"], storage["terminated"], storage["truncated"], gamma, gae_lambda)
        advantages = advantages.flatten()
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
        returns = returns.flatten()
        observations = {name: torch.stack(values).flatten(0, 1) for name, values in observations.items()}
        storage = {name: value.flatten(0, 1) for name, value in storage.items()}
        count = len(advantages)
        transitions += count
        losses = []
        policy.train()
        stopped_for_kl = False
        maximum_kl = 0.0
        for _ in range(epochs):
            order = torch.randperm(count, device=task.device)
            for begin in range(0, count, batch_size):
                ids = order[begin:begin + batch_size]
                batch = {name: value[ids] for name, value in observations.items()}
                distribution = policy.distribution(batch)
                with torch.no_grad():
                    old_std = storage["std"][ids]
                    kl = float((torch.log(distribution.stddev / old_std)
                        + (old_std.square() + (storage["mean"][ids] - distribution.mean).square())
                        / (2 * distribution.variance) - 0.5).sum(-1).mean())
                if not np.isfinite(kl):
                    raise RuntimeError("PPO produced a nonfinite policy divergence")
                maximum_kl = max(maximum_kl, kl)
                if kl > 1.5 * target_kl:
                    stopped_for_kl = True
                    break
                log_probability = distribution.log_prob(storage["latent"][ids]).sum(-1)
                log_ratio = log_probability - storage["log_probability"][ids]
                ratio = log_ratio.exp()
                surrogate = torch.minimum(ratio * advantages[ids],
                    ratio.clamp(1 - clip, 1 + clip) * advantages[ids])
                predicted = policy.critic(batch, storage["privileged"][ids])
                value_loss = (predicted - returns[ids]).square().mean()
                entropy = distribution.entropy().sum(-1).mean()
                loss = (-surrogate.mean() + 0.5 * value_loss
                        - float(configuration.get("entropy", 0.0)) * entropy)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                losses.append({"policy": float(-surrogate.mean().detach()), "value": float(value_loss.detach()),
                               "entropy": float(entropy.detach()), "kl": kl, "gradient_norm": float(norm)})
            if stopped_for_kl:
                break
        with torch.no_grad():
            divergences = []
            for begin in range(0, count, batch_size):
                ids = slice(begin, begin + batch_size)
                distribution = policy.distribution({name: value[ids] for name, value in observations.items()})
                old_std = storage["std"][ids]
                divergences.append((torch.log(distribution.stddev / old_std)
                    + (old_std.square() + (storage["mean"][ids] - distribution.mean).square())
                    / (2 * distribution.variance) - 0.5).sum(-1))
            final_kl = float(torch.cat(divergences).mean())
        if not np.isfinite(final_kl):
            raise RuntimeError("PPO produced a nonfinite final policy divergence")
        for group in optimizer.param_groups:
            if stopped_for_kl or final_kl > 1.5 * target_kl:
                group["lr"] = max(minimum_lr, group["lr"] / 1.5)
            elif final_kl < 0.5 * target_kl:
                group["lr"] = min(maximum_lr, group["lr"] * 1.5)
        report = {"iteration": iteration + 1, "transitions": transitions,
                  "rollout_seconds": rollout_seconds, "update_seconds": time.perf_counter() - started - rollout_seconds,
                  "mean_reward": float(storage["reward"].mean()), "reward_terms": reward_terms,
                  "episodes": episodes,
                  "losses": {name: float(np.mean([row[name] for row in losses])) for name in losses[0]} if losses else {},
                  "policy_kl": final_kl, "maximum_minibatch_kl": maximum_kl,
                  "stopped_for_kl": stopped_for_kl, "updates": len(losses),
                  "learning_rate": optimizer.param_groups[0]["lr"],
                  "residual_std": policy.log_std.detach().exp().cpu().tolist(),
                  "maximum_progress": float((task.controller.base[:, 0] - task.start[:, 0]).max())}
        print(json.dumps(report), flush=True)
        with (output / "training.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(report) + "\n")
        if (iteration + 1) % int(configuration.get("save_every", 10)) == 0 or iteration + 1 == start_iteration + iterations:
            policy.save(output / f"policy_{iteration + 1:05d}.pt", iteration=iteration + 1,
                transitions=transitions,
                optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(), task=asdict(task.config), ppo=configuration,
                provenance=provenance,
                completed_episodes=episodes, resume_contract="Resume optimizer and policy; reset physical episodes")
    return observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initialize", type=Path, help="Initialize from a proprioceptive checkpoint with a fresh optimizer")
    parser.add_argument("--iterations", type=int)
    args = parser.parse_args()
    if args.resume and args.initialize:
        parser.error("Resume and initialization are mutually exclusive")
    configuration = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    task_config = WadingConfig(**configuration["task"])
    ppo = configuration.get("ppo", {})
    if args.iterations is not None:
        ppo["iterations"] = args.iterations
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "config.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(task_config.seed)
    random.seed(task_config.seed)
    np.random.seed(task_config.seed)
    task = G1WadingTask(task_config)
    try:
        identity = record_run_identity(Path(__file__).resolve().parents[2], args.out,
            [args.config, args.resume, args.initialize, task_config.scene, task_config.deployment,
             task_config.manifest, configuration["prior"]],
            ["src", "python/src", "python/nuka", "tools/assets", "examples/training"])
        if args.resume:
            policy, training = G1FusionPolicy.load(args.resume, device=task.device)
            source_task = training["task"]
            source_load = source_task.get("proprioception", {}).get("load_source", "foot_wrench")
            if source_load != task.controller.proprioception.config.load_source:
                raise ValueError("Resume requires the same load measurements; use --initialize when changing sensors")
            if source_task.get("terrain") != task_config.terrain:
                raise ValueError("Resume requires the same terrain projection contract")
        else:
            prior = G1VelocityActor.from_onnx(configuration["prior"], device=task.device)
            fusion = G1FusionConfig(**{**task.policy_observation_contract(), **configuration.get("fusion", {})})
            policy = G1FusionPolicy(prior, fusion).to(task.device)
            training = {}
        if args.initialize:
            source, source_training = G1FusionPolicy.load(args.initialize, device=task.device)
            old_load = source_training["task"].get("proprioception", {}).get("load_source", "foot_wrench")
            new_load = task.controller.proprioception.config.load_source
            policy.initialize_from(source, reset_load_inputs=old_load != new_load)
            del source
        task.validate_policy(policy)
        optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad],
                                     lr=float(ppo.get("learning_rate", 3e-5)))
        if args.resume:
            optimizer.load_state_dict(training["optimizer"])
            torch.set_rng_state(training["torch_rng"].cpu())
            torch.cuda.set_rng_state_all([value.cpu() for value in training["cuda_rng"]])
        metadata = task.metadata()
        metadata["runtime"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                               "device": torch.cuda.get_device_name(task.device)}
        metadata["prior_sha256"] = hashlib.sha256(Path(configuration["prior"]).read_bytes()).hexdigest()
        (args.out / "observations.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        provenance = {"head": identity["head"], "libraries": identity["libraries"],
                      "identity_sha256": hashlib.sha256((args.out / "identity.json").read_bytes()).hexdigest()}
        if args.initialize:
            provenance["initialization"] = {"checkpoint": str(args.initialize),
                "iteration": source_training.get("iteration"), "load_inputs_reset": old_load != new_load}
        start_iteration = training.get("iteration", 0)
        start_transitions = training.get("transitions", start_iteration *
            training.get("task", {}).get("num_envs", task.num_envs) *
            training.get("ppo", {}).get("horizon", ppo.get("horizon", 128)))
        observation = train(task, policy, optimizer, ppo, args.out, start_iteration, provenance,
                            start_transitions=start_transitions)
        policy.eval()
        with torch.no_grad():
            traced = torch.jit.trace(policy, (observation,), strict=False)
            torch.testing.assert_close(traced(observation), policy(observation), rtol=2e-5, atol=2e-5)
            traced.save(str(args.out / "actor.ts"))
    finally:
        task.close()


if __name__ == "__main__":
    main()
