"""New training tasks of the user's own, as editor extensions that register them.

`write(folder, base)` makes `<folder>/<name>/` holding `extension.toml` and `<name>.py`: a task built on
the registered task `base` with a Goal group of its own. The editor reloads the extension when the file
is saved, and the Training panel shows the task's groups as the file declares them.
"""

from __future__ import annotations

from pathlib import Path

from .tasks import TASKS

_MANIFEST = '''[package]
id = "{name}"
version = "1.0.0"
title = "{title}"
description = "A training task of my own, built on {base_title}."

[python]
module = "{name}"
'''

_MODULE = '''"""{title}: a training task of my own, built on {base_title}.

Each Group is a section of the Training panel and each Param a field in it; save this file and the
editor reloads it. `build` turns the panel's values into the rl_games parameters of a run.
"""

import copy

from nuka.editor import Extension
from nuka.train import TASKS, Group, Param, TaskSpec, register, unregister

BASE = TASKS[{base!r}]


def build(config: dict, scene: str, name: str) -> dict:
    goal = config["goal"]
    params = BASE.build(config, scene, name)
    env = params["config"]["env_config"]
    env["fixed_command"] = True
    env["command"] = [goal["speed"], 0.0, goal["turn"]]
    return params


TASK = TaskSpec(name={name!r}, title={title!r}, env=BASE.env, robot=dict(BASE.robot), dt=BASE.dt,
                spacing=BASE.spacing, build=build, check=BASE.check, groups=copy.deepcopy(BASE.groups) + [
    Group("goal", "Goal", [
        Param("speed", "float", 0.8, 0.0, 2.0, unit="m/s", help="The forward speed every robot is asked for"),
        Param("turn", "float", 0.0, -1.0, 1.0, unit="rad/s", help="The yaw rate every robot is asked for"),
    ]),
])
register(TASK)


class Task(Extension):
    """Keeps the task registered while the extension is on."""

    def on_shutdown(self):
        unregister(TASK.name)
'''


def write(folder: str, base: str) -> dict:
    """{name, title, file}: the first of my_task, my_task_2, ... free both as a folder and as a task."""
    root = Path(folder)
    n = 1
    while (root / _name(n)).exists() or _name(n) in TASKS:
        n += 1
    name, title = _name(n), "My task" if n == 1 else f"My task {n}"
    fields = {"name": name, "title": title, "base": base, "base_title": TASKS[base].title}
    (root / name).mkdir(parents=True)
    (root / name / "extension.toml").write_text(_MANIFEST.format(**fields), encoding="utf-8")
    file = root / name / f"{name}.py"
    file.write_text(_MODULE.format(**fields), encoding="utf-8")
    return {"name": name, "title": title, "file": str(file)}


def _name(n: int) -> str:
    return "my_task" if n == 1 else f"my_task_{n}"
