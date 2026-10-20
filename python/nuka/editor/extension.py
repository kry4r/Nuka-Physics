"""Editor extensions written in Python.

An extension is a folder holding ``extension.toml`` and its module::

    [package]
    id = "hello_py"
    version = "1.0.0"
    title = "Hello Python"

    [python]
    module = "hello_py"

The module defines one ``Extension`` subclass. Enabling the extension imports the module
fresh and calls ``on_startup(ctx)``; disabling or reloading it calls ``on_shutdown()`` and
drops the module, so an edited file takes effect on the next load. What ``ctx`` builds or
registers belongs to the extension and goes away with it::

    class Hello(Extension):
        def on_startup(self, ctx):
            panel = ctx.ui.panel("hello_py", "Hello Python")
            panel.button("Greet", on_click=lambda: ctx.log("hello"))
            ctx.actions.register("hello_py.greet", "Greet", lambda args: {"text": "hi"})
            ctx.overlays.register("hello_py.axes", self.draw)

        def draw(self, frame):
            frame.line((0, 0, 0), (1, 0, 0), 0xFF5050FF)

``ctx.teleop`` adds input devices (a SpaceMouse, an XR controller, ...) and teleop modes that
turn a device's reading into joint targets; the Teleop panel and the ``teleop.*`` actions pick them.

Callbacks run on the thread that runs ``update()``. An action or overlay that raises logs its
traceback to the extension's log; three failures in a row disable the extension.
"""

from __future__ import annotations

import importlib
import json
import sys
import traceback
from pathlib import Path
from typing import Callable, Optional

from .ui import _LEVELS, UI, _adapt, _capture


class Extension:
    """Subclass once per extension; each enable makes a new instance."""

    def on_startup(self, ctx: "Context") -> None:
        pass

    def on_shutdown(self) -> None:
        pass


class _Actions:
    def __init__(self, ctx: "Context"):
        self._ctx = ctx

    def register(self, action_id: str, label: str, fn: Callable, *, category: Optional[str] = None,
                 shortcut: Optional[str] = None, icon: Optional[str] = None, hidden: bool = False,
                 doc: Optional[dict] = None) -> None:
        """An editor action running fn(args) or fn(); its return value is the action's value.

        ``doc`` tells agents what it does, as assets/editor/actions.toml does for the editor's own:
        {"summary", "kind", "args": {name: "type, meaning, default"}, "returns", "example"}. kind is
        read, view, edit, sim, file, process or app; an action without one counts as process.
        """
        fn = _adapt(fn)
        ctx = self._ctx

        def run(args_json: str):
            try:
                return True, json.dumps(fn(json.loads(args_json)))
            except Exception as error:
                ctx.log(traceback.format_exc(), "error")
                return False, f"{type(error).__name__}: {error}"

        ctx._host.register_action(action_id, label, category or "", shortcut or "", icon or "", hidden, run,
                                  json.dumps(doc) if doc else "")

    def run(self, action_id: str, /, **args):
        """Run any editor action now; see ``EditorApp.actions``."""
        return self._ctx.app.actions.run(action_id, **args)


class _Overlays:
    def __init__(self, ctx: "Context"):
        self._ctx = ctx

    def register(self, overlay_id: str, fn: Callable) -> None:
        """Call fn(frame) each frame to draw over the viewport; see ``_editor_ext.OverlayFrame``."""
        ctx = self._ctx

        def draw(frame) -> bool:
            try:
                return fn(frame) is not False
            except Exception:
                ctx.log(traceback.format_exc(), "error")
                return False

        ctx._host.register_overlay(overlay_id, draw)


class _Teleop:
    def __init__(self, ctx: "Context"):
        self._ctx = ctx

    def register_device(self, device_id: str, label: str, read: Callable, *, axes=(), buttons=(),
                        kind: str = "extension") -> None:
        """A teleop device: read() returns (axes, buttons) in the order named here, axes in
        [-1, 1] and buttons true or false. It is read once a frame while it is the active device."""
        ctx = self._ctx

        def poll():
            try:
                a, b = read()
                return [float(x) for x in a], [1.0 if x else 0.0 for x in b]
            except Exception:
                ctx.log(traceback.format_exc(), "error")
                raise

        ctx._host.register_teleop_device(device_id, label, kind, list(axes), list(buttons), poll)

    def register_retarget(self, retarget_id: str, label: str, fn: Callable) -> None:
        """A teleop mode: fn(state) gets {"device": {"id", "axes", "buttons"}, "quantities",
        "joints": [{"name", "q", "target"}], "dt"} each frame and returns {joint name: target}."""
        ctx = self._ctx

        def run(state_json: str):
            try:
                return True, json.dumps(fn(json.loads(state_json)))
            except Exception as error:
                ctx.log(traceback.format_exc(), "error")
                return False, f"{type(error).__name__}: {error}"

        ctx._host.register_teleop_retarget(retarget_id, label, run)


class Context:
    """What an enabled extension reaches the editor through."""

    def __init__(self, app, host):
        self._host = host
        self.id: str = host.id
        self.path = Path(host.dir)
        self.settings: dict = json.loads(host.settings_json)
        self.app = app
        self.ui = UI(app, native=host)
        self.actions = _Actions(self)
        self.overlays = _Overlays(self)
        self.teleop = _Teleop(self)

    def log(self, text, level: str = "info") -> None:
        """To the extension's log and the editor console; level: info, ok, warn or error."""
        self._host.log(_LEVELS[level], str(text))


def _purge(module: str) -> None:
    for name in [n for n in sys.modules if n == module or n.startswith(module + ".")]:
        del sys.modules[name]


def _extension_class(mod) -> type:
    found = [value for value in vars(mod).values()
             if isinstance(value, type) and issubclass(value, Extension) and value is not Extension
             and (value.__module__ == mod.__name__ or value.__module__.startswith(mod.__name__ + "."))]
    if len(found) != 1:
        raise TypeError(f"{mod.__name__} defines {len(found)} Extension subclasses; an extension needs one")
    return found[0]


class ExtensionLoader:
    """Imports and drops Python extensions when the editor enables and disables them."""

    def __init__(self, app):
        self._app = app
        self._loaded = {}

    def load(self, extension_id: str, module: str, path: str):
        """(ok, text): what loading printed, with the traceback when it failed."""
        state = {}

        def start():
            if path not in sys.path:
                sys.path.insert(0, path)
            _purge(module)
            importlib.invalidate_caches()
            cls = _extension_class(importlib.import_module(module))
            state["ctx"] = Context(self._app, self._app._native.extension_host(extension_id))
            state["ext"] = cls()
            state["ext"].on_startup(state["ctx"])

        ok, text = _capture(start)
        if ok:
            self._loaded[extension_id] = (state["ext"], state["ctx"], module, path)
        else:
            self._drop(state.get("ctx"), module, path)
        return ok, text

    def unload(self, extension_id: str) -> Optional[str]:
        ext, ctx, module, path = self._loaded.pop(extension_id)
        text = _capture(ext.on_shutdown)[1]
        self._drop(ctx, module, path)
        return text or None

    @staticmethod
    def _drop(ctx: Optional[Context], module: str, path: str) -> None:
        if ctx is not None:
            ctx._host.close()
        _purge(module)
        if path in sys.path:
            sys.path.remove(path)
