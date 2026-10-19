"""Training configurations: a task's defaults, edits by `group.key` path, and validation.

A configuration is `{group: {key: value}}` over every parameter of the task's schema. Errors are
`{"path": "group.key", "message": ...}`, with `"rejected": true` when the value was not taken;
`scene` is the path of errors about the scene itself.
"""

from __future__ import annotations

import copy

from .tasks import Param, TaskSpec

_LENGTHS = {"range": 2, "vec3": 3}


def defaults(task: TaskSpec) -> dict:
    return {g.key: {p.key: copy.deepcopy(p.default) for p in g.params} for g in task.groups}


def _bounded(p: Param, v):
    if p.min is not None and v < p.min:
        raise ValueError(f"at least {p.min:g}")
    if p.max is not None and v > p.max:
        raise ValueError(f"at most {p.max:g}")
    return v


def _number(v, integer: bool):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError("expects a number")
    if integer and float(v) != int(v):
        raise ValueError("expects a whole number")
    return int(v) if integer else float(v)


def coerce(p: Param, value):
    """The value in the form the parameter's kind holds; ValueError says why it does not fit."""
    if p.readonly:
        raise ValueError("is fixed: " + p.help if p.help else "is fixed")
    if p.kind == "bool":
        if not isinstance(value, bool):
            raise ValueError("expects true or false")
        return value
    if p.kind in ("int", "float"):
        return _bounded(p, _number(value, p.kind == "int"))
    if p.kind == "enum":
        if value not in p.choices:
            raise ValueError("expects one of " + ", ".join(map(str, p.choices)))
        return value
    if p.kind in ("range", "vec3", "ints"):
        n = _LENGTHS.get(p.kind)
        if not isinstance(value, (list, tuple)) or not value or (n is not None and len(value) != n):
            raise ValueError(f"expects {n} numbers" if n else "expects a list of whole numbers")
        values = [_bounded(p, _number(v, p.kind == "ints")) for v in value]
        if p.kind == "range" and values[0] > values[1]:
            raise ValueError("the low end is above the high end")
        return values
    raise ValueError(f"has an unknown kind {p.kind!r}")


def scene_errors(task: TaskSpec, layout: dict) -> list:
    """What keeps a scene of `{articulations, links, joints}` from training the task.

    The run steps physics at the task's `dt`, whatever step the editor shows the scene at.
    """
    robots = layout.get("articulations", 0)
    if robots != 1:
        return [{"path": "scene", "message": f"{task.title} trains one robot, copied per env; "
                                             f"the scene has {robots}"}]
    return [{"path": "scene", "message": f"{task.title} needs a robot with {n} {key}; "
                                         f"the scene's has {layout.get(key, 0)}"}
            for key, n in task.robot.items() if layout.get(key, 0) != n]


def resolve(task: TaskSpec, stored: dict | None = None, edits: dict | None = None,
            layout: dict | None = None) -> tuple[dict, list]:
    """The full configuration from the defaults, a stored one and `{path: value}` edits, with errors.

    A stored or edited value that does not fit keeps the value before it and reports why. Without
    a `layout` the scene is not checked.
    """
    config, errors = defaults(task), []
    for group, values in (stored or {}).items():
        for key, value in values.items():
            p = task.param(group, key)
            if p is not None and not p.readonly:
                try:
                    config[group][key] = coerce(p, value)
                except ValueError as exc:
                    errors.append({"path": f"{group}.{key}", "message": str(exc), "rejected": True})
    for path, value in (edits or {}).items():
        group, _, key = path.partition(".")
        p = task.param(group, key)
        if p is None:
            errors.append({"path": path, "message": f"{task.name} has no parameter {path}", "rejected": True})
            continue
        try:
            config[group][key] = coerce(p, value)
        except ValueError as exc:
            errors.append({"path": path, "message": str(exc), "rejected": True})
    if layout:
        errors.extend(scene_errors(task, layout))
    errors.extend(task.check(config, layout or {}))
    return config, errors


def changed(task: TaskSpec, config: dict) -> dict:
    """The values that differ from the defaults, `{group: {key: value}}`, as the editor keeps them."""
    base, out = defaults(task), {}
    for group, values in config.items():
        for key, value in values.items():
            if value != base[group][key]:
                out.setdefault(group, {})[key] = value
    return out
