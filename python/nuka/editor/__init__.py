"""nuka.editor -- run a world in the Nuka editor, or without it.

Command line (also the ``nuka-editor`` app)::

    python -m nuka.editor                               # start page
    python -m nuka.editor scene.nks                     # editor window
    python -m nuka.editor scene.nks --offscreen --frames 60
    python -m nuka.editor scene.nks --headless --steps 600

A scene given while an editor already runs on the same ``--profile`` opens in that window;
``--new-window`` starts another editor instead.

From a script, over a world the script owns::

    parser = argparse.ArgumentParser()
    nuka.editor.add_launcher_args(parser)
    opts, rest = parser.parse_known_args()
    app = nuka.editor.launch(rest, world=world, headless=opts.headless)
    app.run(opts.steps)
    app.close()

``launch`` returns an ``EditorApp`` or, headless, a ``HeadlessApp`` with the same
running/update/run/on_tick/stop/close surface that never loads the editor library,
SDL3 or Vulkan. ``EditorApp.ui`` builds panels in the editor shell (see ``nuka.editor.ui``).
Editor extensions in Python subclass ``Extension`` (see ``nuka.editor.extension``).
"""

from __future__ import annotations

import argparse
import os
import platform
import site
import sys
from typing import Optional, Sequence

from .app import ActionError, Actions, EditorApp, EditorStartupError, HeadlessApp, ScriptRunner, forward
from .extension import Context, Extension
from .ui import UI, Node, ScriptInterrupted

__all__ = ["ActionError", "Actions", "Context", "EditorApp", "EditorStartupError", "Extension", "HeadlessApp", "Node",
           "ScriptInterrupted", "ScriptRunner", "UI", "add_launcher_args", "launch", "main"]


def add_launcher_args(parser: argparse.ArgumentParser) -> None:
    """Add --headless and --steps to an argparse parser."""
    group = parser.add_argument_group("nuka.editor")
    group.add_argument("--headless", action="store_true",
                       help="run the world without the editor (no window, no Vulkan)")
    group.add_argument("--steps", type=int, default=None, help="stop after this many updates")


def launch(args: Optional[Sequence[str]] = None, *, world=None, scene: Optional[str] = None,
           dt: Optional[float] = None, headless: bool = False):
    """Open the editor with the given editor flags, or a headless app when headless."""
    if headless:
        return HeadlessApp(world=world, scene=scene, dt=dt)
    return EditorApp(args or (), world=world, scene=scene, dt=dt)


def _profile_dir(args: Sequence[str]) -> str:
    """The editor's user folder, as the editor library picks it."""
    if "--profile" in args[:-1]:
        return args[list(args).index("--profile") + 1]
    if os.name == "nt":
        return os.path.join(os.environ["APPDATA"], "Nuka", "editor")
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.environ["HOME"], ".config")
    return os.path.join(base, "nuka", "editor")


def _site_dir(env: str) -> str:
    if os.name == "nt":
        return os.path.join(env, "Lib", "site-packages")
    return os.path.join(env, "lib", f"python{sys.version_info[0]}.{sys.version_info[1]}", "site-packages")


def _same_minor(env: str) -> bool:
    cfg = os.path.join(env, "pyvenv.cfg")
    if os.path.exists(cfg):
        with open(cfg, encoding="utf-8") as f:
            for line in f:
                key, _, value = line.partition("=")
                if key.strip() in ("version", "version_info"):
                    return value.strip().split(".")[:2] == [str(v) for v in sys.version_info[:2]]
    if os.name == "nt":
        return os.path.exists(os.path.join(env, f"python{sys.version_info[0]}{sys.version_info[1]}.dll"))
    return os.path.isdir(_site_dir(env))


def _use_python_env(args: Sequence[str]) -> tuple:
    """The app's packages come with it; settings.toml's python.env, else an activated venv or
    conda env, adds the user's. It must match this interpreter's minor version."""
    version = platform.python_version()
    if not getattr(sys, "frozen", False):
        return f"{version}  {sys.executable}", True
    env = None
    settings = os.path.join(_profile_dir(args), "settings.toml")
    if os.path.exists(settings):
        import tomllib
        with open(settings, "rb") as f:
            env = tomllib.load(f).get("python", {}).get("env")
    env = env or os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX")
    if not env:
        return f"{version}  bundled", True
    if not _same_minor(env):
        return f"{version}  bundled; {env} ignored, not Python {sys.version_info[0]}.{sys.version_info[1]}", False
    site.addsitedir(_site_dir(env))
    return f"{version}  + {env}", True


def _alert(text: str) -> None:
    """The app has no console to print to, so startup failures open a message box."""
    if os.name == "nt" and getattr(sys, "frozen", False):
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, text, "Nuka Editor", 0x10)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="nuka-editor", add_help=False, allow_abbrev=False)
    add_launcher_args(parser)
    parser.add_argument("--new-window", action="store_true")
    parser.add_argument("--console", action="store_true")
    parser.add_argument("--render", default=None, help=argparse.SUPPRESS)
    opts, rest = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if opts.render:
        from .render_job import run
        return run(opts.render)
    if opts.headless:
        headless = argparse.ArgumentParser(prog="nuka-editor --headless", allow_abbrev=False)
        headless.add_argument("scene")
        headless.add_argument("--dt", type=float, default=None)
        scene_opts = headless.parse_args(rest)
        app = launch(scene=scene_opts.scene, dt=scene_opts.dt, headless=True)
    else:
        if "--help" in rest:
            print("app: [--headless] [--steps N] [--new-window] [--console]")
        elif not opts.new_window and forward(rest):
            return 0
        python, python_ok = _use_python_env(rest)
        try:
            app = launch(rest + ["--primary"])
        except EditorStartupError as error:
            _alert(f"Nuka Editor could not start.\n\n{error}")
            raise
        if app.running:
            app.actions.run("app.check", name="Python", value=python, ok=python_ok)
    try:
        app.run(opts.steps)
    finally:
        app.close()
    return 0
