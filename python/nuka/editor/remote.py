"""nuka.editor.remote -- drive an editor that is already open, from another process.

``editors()`` lists the editors running on this machine for a profile, ``connect()`` opens one
and ``spawn()`` starts a new one and connects to it. A ``RemoteEditor`` offers what an
``EditorApp`` offers a script through actions and document state: ``actions.run/list/log``,
``document()``, ``console()``, ``screenshot()`` and ``wait()``.
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import threading
import time
from multiprocessing.connection import Client
from typing import List, Optional, Sequence

from .app import ActionError, profile_dir, relaunch_command
from .rpc import family


class RemoteError(RuntimeError):
    """The editor could not be reached, or refused a request; the message says why."""


def _alive(pid: int) -> bool:
    if sys.platform == "win32":
        kernel = ctypes.windll.kernel32
        handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        kernel.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def default_profile() -> str:
    """The profile folder an editor started without ``--profile`` uses."""
    return profile_dir([])


def editors(profile: Optional[str] = None) -> List[dict]:
    """The editors running on ``profile``, the last started first: pid, address and start time."""
    folder = os.path.join(profile or default_profile(), "rpc")
    found = []
    for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        if not name.endswith(".json"):
            continue
        with open(os.path.join(folder, name), encoding="utf-8") as f:
            info = json.load(f)
        if _alive(info["pid"]):
            found.append(info)
    return sorted(found, key=lambda e: e["started"], reverse=True)


def connect(pid: Optional[int] = None, client: str = "python", profile: Optional[str] = None) -> "RemoteEditor":
    """The editor with process id ``pid``, else the one started last on ``profile``."""
    found = [e for e in editors(profile) if pid is None or e["pid"] == pid]
    if not found:
        which = f"with pid {pid} " if pid is not None else ""
        raise RemoteError(f"no editor {which}is running on profile {profile or default_profile()}")
    return RemoteEditor(found[0], client)


def spawn(scene: Optional[str] = None, args: Sequence[str] = (), profile: Optional[str] = None,
          client: str = "python", timeout_s: float = 60.0) -> "RemoteEditor":
    """Start an editor in a window of its own (on ``scene`` when given) and connect to it."""
    argv = relaunch_command() + ([scene] if scene else []) + ["--new-window"]
    argv += (["--profile", profile] if profile else []) + list(args)
    process = subprocess.Popen(argv)
    deadline = time.monotonic() + timeout_s
    while True:
        for info in editors(profile):
            if info["pid"] == process.pid:
                return RemoteEditor(info, client)
        if process.poll() is not None:
            raise RemoteError(f"the editor exited with code {process.returncode} before it could be reached")
        if time.monotonic() > deadline:
            raise RemoteError(f"the editor did not open its endpoint within {timeout_s:g} s")
        time.sleep(0.2)


class RemoteActions:
    """The editor's actions, as ``EditorApp.actions`` offers them."""

    def __init__(self, editor: "RemoteEditor"):
        self._editor = editor

    def run(self, action_id: str, /, **args):
        """Run one action and return its value; raises ActionError when it fails."""
        reply = self._editor.call("run", id=action_id, args=args)
        if not reply["ok"]:
            raise ActionError(f"{action_id}: {reply['error']}")
        return reply["value"]

    def list(self) -> list:
        return self._editor.call("actions")

    def log(self, since: int = 0) -> list:
        return self._editor.call("log", since=since)


class RemoteEditor:
    """One connection to an open editor; requests from several threads take turns."""

    def __init__(self, info: dict, client: str = "python"):
        self.pid = info["pid"]
        try:
            self._conn = Client(info["address"], family(), authkey=bytes.fromhex(info["key"]))
        except OSError as error:
            raise RemoteError(f"cannot reach the editor with pid {self.pid}: {error}") from None
        self._lock = threading.Lock()
        self._next = 0
        self.actions = RemoteActions(self)
        self.hello = self.call("hello", client=client)

    def call(self, op: str, **args):
        """One request; returns its value or raises RemoteError with the editor's reason."""
        with self._lock:
            self._next += 1
            self._conn.send_bytes(json.dumps({"id": self._next, "op": op, "args": args}).encode("utf-8"))
            try:
                reply = json.loads(self._conn.recv_bytes())
            except (EOFError, OSError):
                raise RemoteError("the editor closed the connection") from None
        if not reply["ok"]:
            raise RemoteError(f"{op}: {reply['error']}")
        return reply["value"]

    def document(self) -> dict:
        return self.call("document")

    def console(self, since: int = 0) -> dict:
        """The console text after offset ``since``: text, next and dropped."""
        return self.call("console", since=since)

    def screenshot(self, path: Optional[str] = None, timeout_s: float = 10.0) -> dict:
        """The next frame as a PPM: its path and the viewport's rectangle in it."""
        return self.call("screenshot", path=path, timeout_s=timeout_s)

    def wait(self, action_id: str, until: dict, timeout_s: float = 30.0, **args) -> dict:
        """Run an action until its value holds ``until``: met, value and polls."""
        return self.call("wait", id=action_id, args=args, until=until, timeout_s=timeout_s)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "RemoteEditor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
