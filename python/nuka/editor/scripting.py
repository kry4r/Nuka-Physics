"""Helpers bound into the editor console and its script tabs, next to ``app``, ``world`` and ``ui``.

Each one is a thin call of an editor action, so a script does what the menus and panels do::

    for node in nodes(kind="body"):
        print(node["path"])
    select(find("cube"))
    action("view.frame_selected")
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence


def helpers(app) -> Dict[str, Callable]:
    """The helper functions over ``app``, by the names scripts call them."""

    def action(action_id: str, /, **args):
        """Run an editor action now and return its value; raises ActionError when it fails."""
        return app.actions.run(action_id, **args)

    def nodes(under: str = "", kind: str = "") -> List[dict]:
        """The scene's nodes as {path, name, kind, pose}: all, those under ``under``, or of one ``kind``
        ("artic", "body", "static", "joint", "geom", "visual", "camera", "light", "script", "group"); a pose
        is the world {pos, quat} the node stands at now."""
        return action("scene.nodes", under=under, kind=kind)

    def find(name: str) -> Optional[str]:
        """The path of the first node whose path or name is ``name``; None when none is."""
        for node in nodes():
            if name in (node["path"], node["name"]):
                return node["path"]
        return None

    def select(*paths: str) -> dict:
        """Make ``paths`` the selection, the first one leading; no path clears it."""
        if not paths:
            return action("scene.select", mode="replace")
        reply = action("scene.select", path=paths[0], mode="replace")
        for path in paths[1:]:
            reply = action("scene.select", path=path, mode="add")
        return reply

    def move(path: str, pos: Sequence[float], quat: Sequence[float] = (1.0, 0.0, 0.0, 0.0)) -> dict:
        """Place ``path`` at ``pos`` with ``quat`` (w, x, y, z) as one undoable edit."""
        pose = {"pos": [float(v) for v in pos], "quat": [float(v) for v in quat]}
        return action("edit.apply", label="Move", ops=[{"target": path, "field": "transform", "after": pose}])

    def log(*parts, level: str = "info") -> None:
        """Print ``parts`` to the console at ``level`` ("info", "warn" or "error")."""
        print(f"[{level}]", *parts)

    return {"action": action, "nodes": nodes, "find": find, "select": select, "move": move, "log": log}
