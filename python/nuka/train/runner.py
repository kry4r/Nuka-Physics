"""Runs one training job from its folder and reports through files the editor reads.

The folder holds `job.json`: {task, module, root, config, scene, name}, plus `checkpoint` to start
from one. The run writes `status.json` (rewritten whole), `metrics.jsonl` (a row per epoch),
`checkpoints.jsonl` (a row per save), `log.txt`, `nn/*.pth` and `policies/*.nkml`. A `stop` file
ends the run after the current epoch, with a final checkpoint. While it trains, status.json's
`observe` is the port viewers watch it through (see observe.py).
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

from .config import resolve
from .tasks import TASKS


def _write_json(path: Path, value) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1), encoding="utf-8")
    # Windows refuses to replace a file someone has open; the editor reads it for moments.
    for _ in range(200):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.005)
    os.replace(tmp, path)


def _redirect(path: Path) -> None:
    """Sends this process's output, Python's and the native libraries', to the run's log."""
    log = open(path, "a", encoding="utf-8", buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    sys.stdout = sys.stderr = log


class Run:
    """The run folder's report files."""

    def __init__(self, folder: Path, job: dict, config: dict):
        self.folder = folder
        self.status = {"state": "starting", "pid": os.getpid(), "task": job["task"], "name": job["name"],
                       "envs": config["run"]["num_envs"], "epoch": 0, "max_epochs": config["run"]["max_epochs"],
                       "frames": 0, "fps": None, "reward": None, "length": None, "best_reward": None,
                       "checkpoints": 0, "last_policy": None, "observe": None, "started": time.time(),
                       "updated": None, "ended": None, "error": None}

    def update(self, **fields) -> None:
        self.status.update(fields, updated=time.time())
        _write_json(self.folder / "status.json", self.status)

    def _append(self, name: str, row: dict) -> None:
        with open(self.folder / name, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def epoch(self, row: dict) -> None:
        self._append("metrics.jsonl", row)
        best = self.status["best_reward"]
        if row["reward"] is not None and (best is None or row["reward"] > best):
            best = row["reward"]
        self.update(epoch=row["epoch"], frames=row["frames"], fps=row["fps"], reward=row["reward"],
                    length=row["length"], best_reward=best)

    def saved(self, pth: str, algo) -> None:
        from .export import export

        pth = Path(pth)
        policies = self.folder / "policies"
        policies.mkdir(exist_ok=True)
        nkml = policies / (pth.stem + ".nkml")
        obs_dim, act_dim = export(pth, nkml)
        kind = "best" if pth.stem == self.status["name"] else (
            "final" if algo.epoch_num >= algo.max_epochs else "epoch")
        done = algo.game_rewards.current_size > 0
        self._append("checkpoints.jsonl", {
            "checkpoint": str(pth), "policy": str(nkml), "kind": kind, "epoch": int(algo.epoch_num),
            "frames": self.status["frames"], "reward": float(algo.game_rewards.get_mean()[0]) if done else None,
            "obs": int(obs_dim), "actions": int(act_dim), "time": time.time()})
        self.update(checkpoints=self.status["checkpoints"] + 1, last_policy=str(nkml))

    def stop_requested(self) -> bool:
        return (self.folder / "stop").exists()


def run(job_path: str) -> int:
    path = Path(job_path).resolve()
    folder = path.parent
    job = json.loads(path.read_text(encoding="utf-8"))
    _redirect(folder / "log.txt")
    # A task an extension registers is found by importing the module that registers it.
    sys.path.insert(0, job["root"])
    importlib.import_module(job["module"])
    task = TASKS[job["task"]]
    config, errors = resolve(task, job["config"])
    report = Run(folder, job, config)
    if errors:
        report.update(state="failed", ended=time.time(),
                      error="; ".join(f"{e['path']}: {e['message']}" for e in errors))
        return 1
    report.update()
    try:
        params = task.build(config, job["scene"], job["name"])
        cfg = params["config"]
        cfg["train_dir"], cfg["full_experiment_name"] = str(folder.parent), folder.name
        # torch.compile needs triton, which Windows lacks.
        cfg["torch_compile"] = importlib.util.find_spec("triton") is not None
        import nuka.rl_games  # noqa: F401  (registers the Nuka envs with rl_games)
        from rl_games.torch_runner import Runner

        from .observer import TrainObserver

        observer = TrainObserver(report, job["scene"], task.spacing)
        runner = Runner(algo_observer=observer)
        runner.load_config(params)
        args = {"train": True, "play": False}
        if job.get("checkpoint"):
            args["checkpoint"] = job["checkpoint"]
        runner.run(args)
        # rl_games leaves its env open; the world and device go before the interpreter does.
        observer.close()
    except BaseException as exc:
        traceback.print_exc()
        report.update(state="failed", error=f"{type(exc).__name__}: {exc}", observe=None, ended=time.time())
        return 1
    report.update(state="stopped" if report.stop_requested() else "finished", observe=None, ended=time.time())
    return 0
