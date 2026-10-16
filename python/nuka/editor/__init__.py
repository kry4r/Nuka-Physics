"""nuka.editor -- run a world in the Nuka editor, or without it.

Command line (also the ``nuka-editor`` launcher)::

    python -m nuka.editor scene.nks                     # editor window
    python -m nuka.editor scene.nks --offscreen --frames 60
    python -m nuka.editor scene.nks --headless --steps 600

From a script, over a world the script owns::

    parser = argparse.ArgumentParser()
    nuka.editor.add_launcher_args(parser)
    opts, rest = parser.parse_known_args()
    app = nuka.editor.launch(rest, world=world, headless=opts.headless)
    app.run(opts.steps)
    app.close()

``launch`` returns an ``EditorApp`` or, headless, a ``HeadlessApp`` with the same
running/update/run/on_tick/stop/close surface that never loads the editor library,
SDL3 or Vulkan.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from .app import EditorApp, HeadlessApp, ScriptRunner

__all__ = ["EditorApp", "HeadlessApp", "ScriptRunner", "add_launcher_args", "launch", "main"]


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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="nuka-editor", add_help=False, allow_abbrev=False)
    add_launcher_args(parser)
    opts, rest = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if opts.headless:
        headless = argparse.ArgumentParser(prog="nuka-editor --headless", allow_abbrev=False)
        headless.add_argument("scene")
        headless.add_argument("--dt", type=float, default=None)
        scene_opts = headless.parse_args(rest)
        app = launch(scene=scene_opts.scene, dt=scene_opts.dt, headless=True)
    else:
        if "--help" in rest:
            print("launcher: [--headless] [--steps N]")
        app = launch(rest)
    try:
        app.run(opts.steps)
    finally:
        app.close()
    return 0
