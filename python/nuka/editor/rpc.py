"""nuka.editor.rpc -- the editor's endpoint for other programs on this machine.

Every editor listens on a pipe of its own (a unix socket on Linux) that only clients holding
its key may use; ``<profile>/rpc/<pid>.json`` names the address and the key, and the profile
folder is the user's own. Messages are JSON: a request ``{"id", "op", "args"}`` is answered by
``{"id", "ok", "value", "error"}``. Calls run on the editor's UI thread, like a script's.
"""

from __future__ import annotations

import json
import os
import secrets
import sys
import threading
import time
from multiprocessing import AuthenticationError
from multiprocessing.connection import Client, Listener, answer_challenge, deliver_challenge
from typing import Callable, Dict, List

PROTOCOL = "nuka.rpc.v1"


def address_for(pid: int) -> str:
    """The pipe or socket the editor with process id ``pid`` listens on."""
    if sys.platform == "win32":
        return rf"\\.\pipe\nuka-rpc-{pid}"
    return os.path.join(os.environ.get("XDG_RUNTIME_DIR") or "/tmp", f"nuka-rpc-{pid}.sock")


def family() -> str:
    return "AF_PIPE" if sys.platform == "win32" else "AF_UNIX"


class RpcError(RuntimeError):
    """A request the editor refuses; the message goes back to the client."""


class _Session:
    def __init__(self, number: int):
        self.number = number
        self.client = f"client-{number}"
        self.since = time.time()
        self.calls = 0


class RpcServer:
    """Serves one editor until ``close()``, which also removes its file from ``folder``."""

    def __init__(self, app, folder: str):
        self._app = app
        self._key = secrets.token_bytes(32)
        self._address = address_for(os.getpid())
        self._listener = Listener(self._address, family())
        self._closing = False
        self._lock = threading.Lock()
        self._sessions: Dict[int, _Session] = {}
        self._count = 0
        self._shots = os.path.join(folder, "shots")
        os.makedirs(self._shots, exist_ok=True)
        self.file = os.path.join(folder, f"{os.getpid()}.json")
        info = {"protocol": PROTOCOL, "pid": os.getpid(), "address": self._address, "key": self._key.hex(),
                "started": time.time()}
        with os.fdopen(os.open(self.file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
            json.dump(info, f)
        self._ops: Dict[str, Callable[[_Session, dict], object]] = {
            "hello": self._hello, "actions": self._actions, "run": self._run, "document": self._document,
            "log": self._log, "console": self._console, "screenshot": self._screenshot, "wait": self._wait,
        }
        self._thread = threading.Thread(target=self._accept, name="nuka-rpc", daemon=True)
        self._thread.start()

    def sessions(self) -> List[dict]:
        """The connected clients: name, connected since and calls made."""
        with self._lock:
            return [{"client": s.client, "since": s.since, "calls": s.calls} for s in self._sessions.values()]

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        # A connection of our own wakes the accept loop so it can see the flag.
        Client(self._address, family()).close()
        self._thread.join()
        self._listener.close()
        os.remove(self.file)

    # ---- connections ------------------------------------------------------------------------

    def _accept(self) -> None:
        while True:
            conn = self._listener.accept()
            if self._closing:
                conn.close()
                return
            threading.Thread(target=self._serve, args=(conn,), name="nuka-rpc-client", daemon=True).start()

    def _serve(self, conn) -> None:
        # The key is checked here rather than in accept, so a silent client holds up no one else.
        try:
            deliver_challenge(conn, self._key)
            answer_challenge(conn, self._key)
        except (AuthenticationError, EOFError, OSError):
            conn.close()
            return
        with self._lock:
            self._count += 1
            session = _Session(self._count)
            self._sessions[session.number] = session
        try:
            while not self._closing:
                try:
                    data = conn.recv_bytes()
                except (EOFError, OSError):
                    return
                conn.send_bytes(json.dumps(self._reply(session, data)).encode("utf-8"))
        finally:
            with self._lock:
                del self._sessions[session.number]
            conn.close()

    def _reply(self, session: _Session, data: bytes) -> dict:
        try:
            request = json.loads(data)
        except ValueError as error:
            return {"id": None, "ok": False, "value": None, "error": f"the request is not JSON: {error}"}
        rid = request.get("id")
        op = self._ops.get(request.get("op"))
        if op is None:
            return {"id": rid, "ok": False, "value": None,
                    "error": f"unknown op {request.get('op')!r}; ops: {', '.join(self._ops)}"}
        session.calls += 1
        try:
            return {"id": rid, "ok": True, "value": op(session, request.get("args") or {}), "error": ""}
        except KeyError as error:
            return {"id": rid, "ok": False, "value": None, "error": f"missing argument {error}"}
        except (RpcError, RuntimeError, TypeError, ValueError) as error:
            return {"id": rid, "ok": False, "value": None, "error": str(error)}

    # ---- ops --------------------------------------------------------------------------------

    def _hello(self, session: _Session, args: dict) -> dict:
        session.client = str(args.get("client") or session.client)
        return {"protocol": PROTOCOL, "pid": os.getpid(), "client": session.client,
                "document": self._app.document()}

    def _actions(self, session: _Session, args: dict) -> list:
        return self._app.actions.list()

    def _run(self, session: _Session, args: dict) -> dict:
        return self._app._run(args["id"], args.get("args") or {}, f"agent:{session.client}")

    def _document(self, session: _Session, args: dict) -> dict:
        return self._app.document()

    def _log(self, session: _Session, args: dict) -> list:
        return self._app.actions.log(int(args.get("since", 0)))

    def _console(self, session: _Session, args: dict) -> dict:
        return self._app.actions.run("script.console", since=int(args.get("since", 0)))

    def _screenshot(self, session: _Session, args: dict) -> dict:
        """The next drawn frame as a PPM at ``path`` (a file of the editor's own when empty)."""
        path = args.get("path") or os.path.join(self._shots, f"{session.number}-{session.calls}.ppm")
        if os.path.exists(path):
            os.remove(path)
        reply = self._app._run("view.screenshot", {"path": path}, f"agent:{session.client}")
        if not reply["ok"]:
            raise RpcError(reply["error"])
        deadline = time.monotonic() + float(args.get("timeout_s", 10.0))
        while not os.path.exists(path):
            if time.monotonic() > deadline:
                raise RpcError("no frame was drawn in time; is the editor window open?")
            time.sleep(0.02)
        viewport = self._app.actions.run("view.stats")["viewport"]
        return {"path": path, "viewport": viewport}

    def _wait(self, session: _Session, args: dict) -> dict:
        """Runs action ``id`` until its value holds every key of ``until``, or ``timeout_s`` passes."""
        until = args.get("until") or {}
        deadline = time.monotonic() + float(args.get("timeout_s", 30.0))
        polls = 0
        while True:
            reply = self._app._run(args["id"], args.get("args") or {}, f"agent:{session.client}")
            polls += 1
            if not reply["ok"]:
                raise RpcError(reply["error"])
            value = reply["value"]
            met = isinstance(value, dict) and all(value.get(k) == v for k, v in until.items())
            if met or time.monotonic() > deadline:
                return {"met": met, "value": value, "polls": polls}
            time.sleep(0.05)
