"""What the editor asks of the training package, as JSON files; nothing here imports torch.

`call(request, response)` reads `{"op": ...}` from `request` and writes the answer to `response`:
- `tasks`: {} -> {"tasks": [schema, ...]}
- `resolve`: {task, config, set, layout} -> {task, config, changed, errors}; see config.resolve.
  Without `task` it is the first registered one whose robot the scene holds, else the first registered one.
- `trainer`: {python} -> {python, argv, ready, error}: the interpreter a run starts in, `argv` +
  [job.json] being its command.
A request the package cannot answer, such as one for an unknown task, is answered `{"failed": why}`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .config import changed, resolve, scene_errors
from .tasks import TASKS

# The run imports this editor's nuka, whatever the interpreter has installed.
_BOOT = ("import sys; sys.path.insert(0, sys.argv[1]); from nuka.train.runner import run; "
         "raise SystemExit(run(sys.argv[2]))")
_PROBE = ("import importlib.util as u, sys; "
          "print(*sys.version_info[:2], *(int(u.find_spec(m) is not None) for m in ('torch', 'rl_games')))")


def _tasks(req: dict) -> dict:
    return {"tasks": [t.schema() for t in TASKS.values()]}


def _resolve(req: dict) -> dict:
    layout = req.get("layout")
    fits = [n for n, t in TASKS.items() if layout and not scene_errors(t, layout)]
    name = req.get("task") or (fits + list(TASKS))[0]
    if name not in TASKS:
        raise ValueError(f"no training task {name!r}; there are {', '.join(TASKS)}")
    task = TASKS[name]
    config, errors = resolve(task, req.get("config"), req.get("set"), req.get("layout"))
    return {"task": task.name, "config": config, "changed": changed(task, config), "errors": errors}


def _trainer(req: dict) -> dict:
    """`python` when set, else this interpreter unless it is the app's own."""
    python = req.get("python") or ("" if getattr(sys, "frozen", False) else sys.executable)
    out = {"python": python, "argv": [], "ready": False, "error": ""}
    version = "%d.%d" % sys.version_info[:2]
    if not python:
        out["error"] = f"the app's Python cannot train; choose a Python {version} with torch and rl_games"
        return out
    try:
        probe = subprocess.run([python, "-c", _PROBE], capture_output=True, text=True, timeout=60,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except OSError as exc:
        out["error"] = f"cannot run {python}: {exc.strerror or exc}"
        return out
    if probe.returncode != 0:
        out["error"] = f"{python} failed: " + (probe.stderr.strip().splitlines() or ["no output"])[-1]
        return out
    major, minor, torch, rl_games = map(int, probe.stdout.split())
    if f"{major}.{minor}" != version:
        out["error"] = f"{python} is Python {major}.{minor}; this editor's nuka needs {version}"
        return out
    missing = [name for name, found in (("torch", torch), ("rl_games", rl_games)) if not found]
    if missing:
        out["error"] = f"{python} lacks " + " and ".join(missing)
        return out
    root = str(Path(__file__).resolve().parents[2])
    out.update(argv=[python, "-c", _BOOT, root], ready=True)
    return out


OPS = {"tasks": _tasks, "resolve": _resolve, "trainer": _trainer}


def call(request: str, response: str) -> None:
    with open(request, encoding="utf-8") as f:
        req = json.load(f)
    try:
        out = OPS[req["op"]](req)
    except ValueError as exc:
        out = {"failed": str(exc)}
    with open(response, "w", encoding="utf-8") as f:
        json.dump(out, f)
