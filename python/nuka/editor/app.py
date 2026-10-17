"""The editor and headless apps behind ``nuka.editor.launch``, and the script runner."""

from __future__ import annotations

import atexit
import builtins
import json
import os
import sys
import weakref
from typing import Callable, List, Optional, Sequence

import nuka

from .extension import ExtensionLoader
from .ui import UI, _capture

_DEFAULT_DT = 1.0 / 240.0
_CALLBACKS = ("on_ready", "on_step", "on_reset")
_TICK = 3
# A lifecycle callback that fails this many times in a row is disabled.
_MAX_FAILURES = 3


class _Script:
    def __init__(self, ns: dict):
        self.ns = ns
        self.ready_fired = False
        self.failures = dict.fromkeys(_CALLBACKS, 0)
        self.disabled = set()


class ScriptRunner:
    """Runs the editor console and the scene's /script nodes in this interpreter.

    Each script gets its own namespace holding ``app``, ``world`` and ``nuka`` and
    may define ``on_ready``, ``on_step`` and ``on_reset``. Callbacks run on the
    editor's physics thread while one runs.
    """

    def __init__(self, app):
        self._app = weakref.proxy(app)
        self._console = self._namespace("__console__")
        self._scripts = {}
        self._ticks: List[list] = []

    def _namespace(self, name: str) -> dict:
        return {"__builtins__": builtins, "__name__": name, "app": self._app, "nuka": nuka}

    def _bind(self, ns: dict) -> None:
        ns["world"] = self._app.world

    def run(self, code: str) -> Optional[str]:
        self._bind(self._console)
        return _capture(lambda: exec(compile(code, "<console>", "exec"), self._console))[1] or None

    def register(self, script_id: int, source: str) -> Optional[str]:
        ns = self._namespace("<nuka_script>")
        self._scripts[script_id] = _Script(ns)
        self._bind(ns)
        return _capture(lambda: exec(compile(source, f"<script {script_id}>", "exec"), ns))[1] or None

    def unregister(self, script_id: int) -> None:
        self._scripts.pop(script_id, None)

    def clear(self) -> None:
        self._scripts.clear()

    def add_tick(self, fn: Callable) -> None:
        self._ticks.append([fn, 0])

    def _dispatch_ticks(self) -> Optional[str]:
        out = []
        for entry in list(self._ticks):
            fn = entry[0]
            ok, text = _capture(lambda: fn(self._app))
            out.append(text)
            entry[1] = 0 if ok else entry[1] + 1
            if entry[1] >= _MAX_FAILURES:
                self._ticks.remove(entry)
                out.append(f"[nuka_editor] tick callback '{getattr(fn, '__name__', fn)}' disabled after repeated failures\n")
        return "".join(out) or None

    def dispatch(self, callback: int) -> Optional[str]:
        if callback == _TICK:
            return self._dispatch_ticks()
        name = _CALLBACKS[callback]
        out = []
        for script in list(self._scripts.values()):
            if name == "on_ready":
                if script.ready_fired:
                    continue
                script.ready_fired = True
            fn = script.ns.get(name)
            if not callable(fn) or name in script.disabled:
                continue
            self._bind(script.ns)
            ok, text = _capture(fn)
            if ok:
                script.failures[name] = 0
                out.append(text)
                continue
            script.failures[name] += 1
            if script.failures[name] == 1:
                out.append(text)
            if script.failures[name] >= _MAX_FAILURES:
                script.disabled.add(name)
                out.append(f"[nuka_editor] script callback '{name}' disabled after repeated failures\n")
        return "".join(out) or None


class ActionError(RuntimeError):
    """An editor action failed; the message carries the editor's reason."""


class EditorStartupError(RuntimeError):
    """The editor could not start; the message names the reason and the session log."""


def _ext():
    try:
        from nuka import _editor_ext
    except ImportError as error:
        raise ImportError("this nuka build has no editor (NK_BUILD_EDITOR=OFF); "
                          "use --headless or launch(..., headless=True)") from error
    return _editor_ext


def relaunch_command() -> List[str]:
    """The command that starts another editor like this one: the app, or this interpreter."""
    if getattr(sys, "frozen", False):
        return [sys.executable]
    return [sys.executable, "-m", "nuka.editor"]


def forward(args: Sequence[str]) -> bool:
    """Hand the scene in args to the editor already open on the same profile; False when none is."""
    return _ext().forward(list(args))


class Actions:
    """The editor's commands, the same ones its menus, buttons and shortcuts run."""

    def __init__(self, app: "EditorApp"):
        self._app = weakref.proxy(app)

    def run(self, action_id: str, /, **args):
        """Run one action now and return its value; raises ActionError when it fails."""
        reply = self._app._run(action_id, args, "api")
        if not reply["ok"]:
            raise ActionError(f"{action_id}: {reply['error']}")
        return reply["value"]

    def list(self) -> list:
        """Each action's id, label, category, shortcut, enabled and (for toggles) checked."""
        return json.loads(self._app._native.actions())

    def log(self, since: int = 0) -> list:
        """The logged runs after sequence number ``since``: seq, frame, id, source, args, ok, error."""
        return json.loads(self._app._native.action_log(since))


class EditorApp:
    """The editor over a world: a window, or a display-less surface with ``--offscreen``.

    A given world stays the caller's; without one the editor creates its own at the
    first scene load, and ``world`` borrows it until ``close()``.
    """

    def __init__(self, args: Sequence[str] = (), world=None, scene: Optional[str] = None,
                 dt: Optional[float] = None):
        _editor_ext = _ext()
        argv = ([scene] if scene else []) + list(args) + ["--relaunch", json.dumps(relaunch_command())]
        # A script's world: New and Open start editors of their own, the title names the script.
        if world is not None:
            argv += ["--script-name", os.path.basename(sys.argv[0]) or "python"]
        if dt is None:
            dt = world.dt if world is not None else 0.0
        self._world = world
        self._borrowed = None
        self._runner = ScriptRunner(self)
        self._extensions = ExtensionLoader(weakref.proxy(self))
        try:
            self._native = _editor_ext.App(argv, world.handle if world is not None else 0, dt, self._runner,
                                           self._extensions)
        except RuntimeError as error:
            raise EditorStartupError(str(error)) from None
        self.actions = Actions(self)
        self.ui = UI(weakref.proxy(self))
        atexit.register(self.close)

    @property
    def running(self) -> bool:
        return self._native is not None and self._native.running

    @property
    def playing(self) -> bool:
        return self._native.playing

    @property
    def dt(self) -> float:
        return self._native.dt

    @property
    def world(self):
        """The world on screen; None until the first scene load creates one."""
        if self._world is not None:
            return self._world
        handle = self._native.world_handle
        if handle == 0:
            return None
        if self._borrowed is None or self._borrowed.handle != handle:
            self._borrowed = nuka.World.borrow(handle, self._native.dt)
        return self._borrowed

    def on_tick(self, fn: Callable) -> Callable:
        """Call fn(app) at every tick boundary: each transport step, or each update when
        the caller steps the world. Usable as a decorator."""
        self._runner.add_tick(fn)
        return fn

    def update(self, step: bool = False) -> None:
        """One frame. With step=True the editor transport advances physics; otherwise
        no physics thread runs and the caller steps the world between updates."""
        self._native.update(step)

    def stop_physics(self) -> None:
        """Join the physics thread; the world is the caller's until the next update(True)."""
        self._native.stop_physics()

    def run(self, steps: Optional[int] = None) -> None:
        """Update with the transport until the window closes, or for ``steps`` frames."""
        count = 0
        try:
            while self.running and (steps is None or count < steps):
                self.update(True)
                count += 1
        finally:
            if self._native is not None:
                self._native.stop_physics()

    def console(self, text: str) -> None:
        self._native.console_append(text)

    def _ui_call(self, fn: Callable[[], object]) -> None:
        text = _capture(fn)[1]
        if text:
            self._native.console_append(text)

    def _run(self, action_id: str, args: dict, source: str) -> dict:
        return json.loads(self._native.run_action(action_id, json.dumps(args), source))

    def edit(self, *ops: dict, label: str = "Script edit") -> bool:
        """Apply one undoable transaction of ``{"target", "field", "after"}`` ops (an optional
        ``before`` overrides the recorded prior value). False when an op failed; see
        ``document()["last_error"]``. Not from tick callbacks: an edit parks the physics thread."""
        return self._run("edit.apply", {"label": label, "ops": list(ops)}, "api")["ok"]

    def undo(self) -> bool:
        return self._run("edit.undo", {}, "api")["ok"]

    def redo(self) -> bool:
        return self._run("edit.redo", {}, "api")["ok"]

    def document(self) -> dict:
        """Revision, dirty, hash, undo/redo depths, tick, generation, timeline branches, the
        last parameter write's pushed and applied ticks, last_error, selection and playing."""
        return json.loads(self._native.document())

    def stop(self) -> None:
        """End ``run()`` after the current frame; safe inside scripts."""
        self._native.close()

    def close(self) -> None:
        """Destroy the editor and any world it created."""
        if self._native is None:
            return
        atexit.unregister(self.close)
        if self._borrowed is not None:
            self._borrowed.destroy()  # detaches only: the editor owns that world
            self._borrowed = None
        self._native.destroy()
        self._native = None
        self._runner.clear()

    def __enter__(self) -> "EditorApp":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class HeadlessApp:
    """A world stepped without the editor: no window, no Vulkan and no editor library."""

    def __init__(self, world=None, scene: Optional[str] = None, dt: Optional[float] = None):
        self._device = None
        self._owned = None
        if world is None:
            if not scene:
                raise ValueError("headless mode needs a world or a scene")
            self._device = nuka.Device.create(0)
            builder = nuka.SceneBuilder.create(scene)
            world = builder.build(self._device, env_count=1, dt=dt or _DEFAULT_DT)
            builder.destroy()
            self._owned = world
        self._world = world
        self._running = True
        self._ticks: List[Callable] = []

    @property
    def running(self) -> bool:
        return self._running

    @property
    def playing(self) -> bool:
        return self._running

    @property
    def dt(self) -> float:
        return self._world.dt

    @property
    def world(self):
        return self._world

    def on_tick(self, fn: Callable) -> Callable:
        """Call fn(app) after every update; usable as a decorator."""
        self._ticks.append(fn)
        return fn

    def update(self, step: bool = False) -> None:
        """With step=True advance the world one step, then run the tick callbacks."""
        if step:
            self._world.step()
        for fn in self._ticks:
            fn(self)

    def run(self, steps: Optional[int] = None) -> None:
        """Step until stopped, or for ``steps`` updates."""
        count = 0
        while self._running and (steps is None or count < steps):
            self.update(True)
            count += 1

    def console(self, text: str) -> None:
        print(text, end="")

    def stop(self) -> None:
        self._running = False

    def close(self) -> None:
        """Stop, and destroy the world and device this app created."""
        self._running = False
        if self._owned is not None:
            self._owned.destroy()
            self._owned = None
        if self._device is not None:
            self._device.close()
            self._device = None

    def __enter__(self) -> "HeadlessApp":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
