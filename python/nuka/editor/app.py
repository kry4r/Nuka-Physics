"""The editor and headless apps behind ``nuka.editor.launch``, and the script runner."""

from __future__ import annotations

import ast
import atexit
import builtins
import ctypes
import json
import linecache
import os
import sys
import textwrap
import threading
import weakref
from typing import Callable, List, Optional, Sequence

import numpy

import nuka

from . import scripting
from .extension import ExtensionLoader
from .rpc import RpcServer
from .ui import UI, ScriptInterrupted, _capture

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
    """Runs the editor console, its script tabs and the scene's /script nodes in this interpreter.

    The console and the tabs share one namespace; every namespace holds ``app``, ``world``,
    ``ui``, ``nuka``, ``np`` and the helpers of ``nuka.editor.scripting``. A /script node gets
    its own and may define ``on_ready``, ``on_step`` and ``on_reset``; console runs also see
    ``sel``, the selected path when they start. Code runs on the editor's physics thread while
    one runs, and ``interrupt`` stops it with ScriptInterrupted.
    """

    def __init__(self, app):
        self._app = weakref.proxy(app)
        self._console = self._namespace("__main__")
        self._scripts = {}
        self._ticks: List[list] = []
        self._active = {}  # thread id -> hook calls in progress there
        self._lock = threading.Lock()

    def _namespace(self, name: str) -> dict:
        ns = {"__builtins__": builtins, "__name__": name, "app": self._app, "nuka": nuka, "np": numpy}
        ns.update(scripting.helpers(self._app))
        return ns

    def _bind(self, ns: dict) -> None:
        ns["world"] = self._app.world
        ns["ui"] = self._app.ui

    def _guarded(self, fn: Callable[[], object], stream: Optional[Callable[[str], None]] = None):
        """_capture(fn, stream), interruptible while it runs."""
        tid = threading.get_ident()
        with self._lock:
            self._active[tid] = self._active.get(tid, 0) + 1
        try:
            return _capture(fn, stream)
        finally:
            with self._lock:
                self._active[tid] -= 1
                last = self._active[tid] == 0
                if last:
                    del self._active[tid]
            # An interrupt that came as the code ended must not land in the editor's frames.
            if last:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), None)

    def run(self, code: str, filename: str = "", echo: bool = False):
        """Run code in the console namespace, compiled as ``filename`` ("" for a console line);
        with ``echo`` the value of a final expression prints. What it prints reaches the
        console line by line as it runs; returns (failed, text)."""
        name = filename or "<console>"
        ns = self._console
        if echo:
            code = textwrap.dedent(code)
        # Tracebacks quote the text that ran, saved or not.
        linecache.cache[name] = (len(code), None, code.splitlines(True), name)

        # The document waits for a UI frame, so it is read where the time budget can interrupt it.
        def body():
            self._bind(ns)
            ns["sel"] = self._app.document()["selection"]
            tree = ast.parse(code, name, "exec")
            last = tree.body.pop() if echo and tree.body and isinstance(tree.body[-1], ast.Expr) else None
            if filename and not filename.startswith("<"):
                ns["__file__"] = filename
            exec(compile(tree, name, "exec"), ns)
            if last is not None:
                value = eval(compile(ast.Expression(last.value), name, "eval"), ns)
                if value is not None:
                    ns["_"] = value
                    print(repr(value))

        ok, text = self._guarded(body, self._app._native.console_append)
        return not ok, text

    def interrupt(self) -> None:
        """Raise ScriptInterrupted in every script running now."""
        with self._lock:
            ids = list(self._active)
        for tid in ids:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), ctypes.py_object(ScriptInterrupted))

    def complete(self, code: str, line: int, column: int) -> str:
        """Up to 50 completions at 1-based ``line``, 0-based ``column`` of ``code``, against the
        console namespace, as JSON [{name, type, signature}]."""
        import jedi

        self._bind(self._console)
        found = jedi.Interpreter(code, [self._console]).complete(line, column)
        out = []
        for c in found[:50]:
            signatures = c.get_signatures() if c.type == "function" and len(out) < 12 else []
            out.append({"name": c.name, "type": c.type,
                        "signature": signatures[0].to_string() if signatures else ""})
        return json.dumps(out)

    def register(self, script_id: int, source: str) -> Optional[str]:
        ns = self._namespace("<nuka_script>")
        self._scripts[script_id] = _Script(ns)
        self._bind(ns)
        return self._guarded(lambda: exec(compile(source, f"<script {script_id}>", "exec"), ns))[1] or None

    def unregister(self, script_id: int) -> None:
        self._scripts.pop(script_id, None)

    def clear(self) -> None:
        self._scripts.clear()

    def add_tick(self, fn: Callable) -> None:
        self._ticks.append([fn, 0])

    def remove_tick(self, fn: Callable) -> None:
        self._ticks[:] = [entry for entry in self._ticks if entry[0] is not fn]

    def _dispatch_ticks(self) -> Optional[str]:
        out = []
        for entry in list(self._ticks):
            fn = entry[0]
            ok, text = self._guarded(lambda: fn(self._app))
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
            ok, text = self._guarded(fn)
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


def profile_dir(args: Sequence[str]) -> str:
    """The editor's user folder, as the editor library picks it."""
    if "--profile" in args[:-1]:
        return args[list(args).index("--profile") + 1]
    if os.name == "nt":
        return os.path.join(os.environ["APPDATA"], "Nuka", "editor")
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.environ["HOME"], ".config")
    return os.path.join(base, "nuka", "editor")


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
        """Each action's id, label, category, shortcut, enabled, (for toggles) checked, owner, summary
        and kind; the action ``action.help`` gives one action's arguments, return value and example."""
        return json.loads(self._app._native.actions())

    def log(self, since: int = 0) -> list:
        """The logged runs after sequence number ``since``: seq, frame, id, source, args, ok, error."""
        return json.loads(self._app._native.action_log(since))


class Teleop:
    """Teleoperation through the ``teleop.*`` actions. Extensions add devices and modes with
    ``ctx.teleop``; ``input`` drives the built-in virtual device and ``pad`` a virtual SDL gamepad."""

    def __init__(self, app: "EditorApp"):
        self._app = weakref.proxy(app)

    def devices(self) -> dict:
        """The devices, the modes and the active ones."""
        return self._app.actions.run("teleop.devices")

    def use(self, **args) -> dict:
        """Switch teleop on or off and pick the device, mode, joint, IK link and PD hold."""
        return self._app.actions.run("teleop.use", **args)

    def input(self, axes: Optional[dict] = None, buttons: Optional[dict] = None) -> dict:
        """Set the virtual device's axes and buttons by name; the rest keep their values."""
        return self._app.actions.run("teleop.input", axes=axes or {}, buttons=buttons or {})

    def map(self, **args) -> dict:
        """A mode's bindings [{input, quantity, gain, invert}] and the deadzone."""
        return self._app.actions.run("teleop.map", **args)

    def pad(self, **args) -> dict:
        """Attach, set or detach a virtual gamepad on SDL's joystick path."""
        return self._app.actions.run("teleop.pad", **args)

    def status(self) -> dict:
        """The active device's reading, the mode's quantities and what they drove."""
        return self._app.actions.run("teleop.status")


class Policy:
    """A locomotion policy through the ``policy.*`` actions; teleop's command mode steers it."""

    def __init__(self, app: "EditorApp"):
        self._app = weakref.proxy(app)

    def load(self, path: str, host: bool = False) -> dict:
        """Attach the NKML policy at ``path`` to the loaded robots; ``host`` runs the host MLP."""
        return self._app.actions.run("policy.load", path=path, host=host)

    def unload(self) -> dict:
        return self._app.actions.run("policy.unload")

    def enable(self, on: Optional[bool] = None) -> dict:
        """Run or stop the loaded policy; without ``on`` it toggles."""
        return self._app.actions.run("policy.enable", **({} if on is None else {"on": on}))

    def status(self) -> dict:
        """The loaded policy, its contract, the joints it drives and the command it reads."""
        return self._app.actions.run("policy.status")


class Demos:
    """Teleoperated demonstrations through the ``demo.*`` actions; files are robomimic-style HDF5,
    which ``nuka.editor.demos.read`` opens outside the editor."""

    def __init__(self, app: "EditorApp"):
        self._app = weakref.proxy(app)

    def record(self, on: Optional[bool] = None, all: Optional[bool] = None) -> dict:
        """Start or stop recording; ``all`` records every displayed environment. Bare, it toggles."""
        args = {k: v for k, v in (("on", on), ("all", all)) if v is not None}
        return self._app.actions.run("demo.record", **args)

    def episodes(self) -> dict:
        """The recording, the replay in progress and the episodes."""
        return self._app.actions.run("demo.episodes")

    def play(self, episode: int, play: bool = True) -> dict:
        """Replay an episode's drive targets from the state now; ``play`` starts the transport."""
        return self._app.actions.run("demo.play", episode=episode, play=play)

    def stop(self) -> dict:
        return self._app.actions.run("demo.stop")

    def save(self, path: str) -> dict:
        return self._app.actions.run("demo.save", path=path)

    def load(self, path: str) -> dict:
        """Replace the episodes with the file's; its joints must exist in the scene."""
        return self._app.actions.run("demo.load", path=path)

    def discard(self, episode: Optional[int] = None) -> dict:
        """Drop one episode, without ``episode`` all of them."""
        return self._app.actions.run("demo.discard", **({} if episode is None else {"episode": episode}))


class Clips:
    """Joint animation clips through the ``clip.*`` actions. A clip keys drive targets per joint with
    step, linear or Bezier interpolation; ``clip`` names one, else the edited clip is meant."""

    def __init__(self, app: "EditorApp"):
        self._app = weakref.proxy(app)

    def _run(self, action: str, **args) -> dict:
        return self._app.actions.run(action, **{k: v for k, v in args.items() if v is not None})

    def list(self) -> dict:
        """The clips, the edited one with its playhead and selection, and the clip playing."""
        return self._run("clip.list")

    def get(self, clip: Optional[str] = None) -> dict:
        """A clip's {name, duration, loop, tracks: [{joint, keys: [{time, value, interp, in, out}]}]}."""
        return self._run("clip.get", clip=clip)

    def new(self, name: Optional[str] = None, duration: float = 1.0, loop: bool = False) -> dict:
        return self._run("clip.new", name=name, duration=duration, loop=loop)

    def set(self, clip: Optional[str] = None, duration: Optional[float] = None, loop: Optional[bool] = None) -> dict:
        return self._run("clip.set", clip=clip, duration=duration, loop=loop)

    def rename(self, to: str, clip: Optional[str] = None) -> dict:
        return self._run("clip.rename", clip=clip, to=to)

    def remove(self, clip: Optional[str] = None) -> dict:
        return self._run("clip.remove", clip=clip)

    def key(self, joint: Optional[str] = None, time: Optional[float] = None, value: Optional[float] = None,
            interp: Optional[str] = None, slope_in: Optional[float] = None, slope_out: Optional[float] = None,
            keys: Optional[list] = None, clip: Optional[str] = None) -> dict:
        """Set one key, or each of ``keys`` [{joint, time, value, interp, in, out}]; time defaults to the
        playhead and value to the joint's drive target."""
        return self._run("clip.key", clip=clip, joint=joint, time=time, value=value, interp=interp,
                         keys=keys, **{"in": slope_in, "out": slope_out})

    def key_pose(self, time: Optional[float] = None, pose: Optional[str] = None, source: Optional[str] = None,
                 clip: Optional[str] = None) -> dict:
        """Key every joint from library pose ``pose``, else the live positions (``source`` "q") or the
        drive targets ("target")."""
        return self._run("clip.key_pose", clip=clip, time=time, pose=pose, **{"from": source})

    def remove_keys(self, keys: Optional[list] = None, joint: Optional[str] = None, time: Optional[float] = None,
                    clip: Optional[str] = None) -> dict:
        """Remove ``keys`` [{joint, time}], else those of ``joint`` and/or at ``time``, else the selection."""
        return self._run("clip.remove_keys", clip=clip, keys=keys, joint=joint, time=time)

    def move_keys(self, dt: float = 0.0, dv: float = 0.0, keys: Optional[list] = None,
                  clip: Optional[str] = None) -> dict:
        return self._run("clip.move_keys", clip=clip, keys=keys, dt=dt, dv=dv)

    def sample(self, time: Optional[float] = None, apply: bool = False, clip: Optional[str] = None) -> dict:
        """Each keyed joint's value at ``time``; ``apply`` makes them the drive targets at once."""
        return self._run("clip.sample", clip=clip, time=time, apply=apply)

    def play(self, clip: Optional[str] = None, loop: Optional[bool] = None, play: bool = True) -> dict:
        """Drive the shown environment through the clip a step a row; ``play`` starts the transport."""
        return self._run("clip.play", clip=clip, loop=loop, play=play)

    def stop(self) -> dict:
        return self._run("clip.stop")

    def save(self, path: str, clip: Optional[str] = None) -> dict:
        """Write the clip as portable ``.nkclip.json``."""
        return self._run("clip.save", clip=clip, path=path)

    def load(self, path: str) -> dict:
        """Add a ``.nkclip.json`` clip, replacing one of its name."""
        return self._run("clip.load", path=path)

    def editor(self, on: Optional[bool] = None, **args) -> dict:
        """Open or close the Clip Editor and set its clip, playhead, auto_key, interp, curve and selected."""
        return self._run("clip.editor", on=on, **args)

    def undo(self) -> dict:
        return self._run("clip.undo")

    def redo(self) -> dict:
        return self._run("clip.redo")


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
        self.teleop = Teleop(self)
        self.policy = Policy(self)
        self.demos = Demos(self)
        self.clips = Clips(self)
        self.ui = UI(weakref.proxy(self))
        # Other programs on this machine (nuka.editor.connect, the MCP server) reach the editor here.
        # It opens and closes as agent access leaves and enters "off".
        self.rpc = RpcServer(weakref.proxy(self), os.path.join(profile_dir(argv), "rpc"))
        self._native.set_rpc_hooks(self.rpc)
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

    def off_tick(self, fn: Callable) -> None:
        """Stop calling fn at the tick boundary."""
        self._runner.remove_tick(fn)

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
        ``document()["last_error"]``. From a script on the physics thread it waits for the
        editor's next frame."""
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
        self.rpc.close()
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

    def off_tick(self, fn: Callable) -> None:
        """Stop calling fn after updates."""
        self._ticks.remove(fn)

    def update(self, step: bool = False) -> None:
        """With step=True advance the world one step, then run the tick callbacks."""
        if step:
            self._world.step()
        for fn in list(self._ticks):
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
