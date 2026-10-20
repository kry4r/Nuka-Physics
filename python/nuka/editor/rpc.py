"""nuka.editor.rpc -- the editor's endpoint for other programs on this machine.

Every editor listens on a pipe of its own (a unix socket on Linux) that only clients holding
its key may use; ``<profile>/rpc/<pid>.json`` names the address and the key, and the profile
folder is the user's own. Messages are JSON: a request ``{"id", "op", "args"}`` is answered by
``{"id", "ok", "value", "error"}``. Calls run on the editor's UI thread, like a script's.

The endpoint is open while the editor's agent access is not "off", and the actions clients
run pass that access check (see the Agents panel).
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


def address_for(pid: int, opening: int) -> str:
    """The pipe or socket of the editor with process id ``pid`` at its ``opening``-th listen."""
    if sys.platform == "win32":
        return rf"\\.\pipe\nuka-rpc-{pid}-{opening}"
    return os.path.join(os.environ.get("XDG_RUNTIME_DIR") or "/tmp", f"nuka-rpc-{pid}-{opening}.sock")


def family() -> str:
    return "AF_PIPE" if sys.platform == "win32" else "AF_UNIX"


class RpcError(RuntimeError):
    """A request the editor refuses; the message goes back to the client."""


class _Session:
    def __init__(self, number: int):
        self.number = number
        self.client = f"client-{number}"
        self.pid = 0
        self.since = time.time()
        self.calls = 0
        self.dropped = False  # the editor let it go; its connection closes before the next call


class RpcServer:
    """One editor's endpoint. ``listen(True)`` opens it with a new key and writes its file to
    ``folder``; ``listen(False)`` or ``close()`` closes it, removes the file and drops the clients.
    The editor calls listen as agent access leaves or enters "off", and disconnect to drop them."""

    def __init__(self, app, folder: str):
        self._app = app
        self._listener = None
        self._closing = True
        self._lock = threading.Lock()
        self._sessions: Dict[int, _Session] = {}
        self._count = 0
        self._openings = 0
        self._shots = os.path.join(folder, "shots")
        self.file = os.path.join(folder, f"{os.getpid()}.json")
        self._ops: Dict[str, Callable[[_Session, dict], object]] = {
            "hello": self._hello, "actions": self._actions, "run": self._run, "document": self._document,
            "log": self._log, "console": self._console, "screenshot": self._screenshot, "wait": self._wait,
        }

    @property
    def listening(self) -> bool:
        return self._listener is not None

    def listen(self, on: bool) -> None:
        if on and self._listener is None:
            self._open()
        elif not on:
            self.close()

    def _open(self) -> None:
        os.makedirs(self._shots, exist_ok=True)
        self._key = secrets.token_bytes(32)
        # A fresh address, as clients dropped by the last close may still hold the old one.
        self._openings += 1
        self._address = address_for(os.getpid(), self._openings)
        self._listener = Listener(self._address, family())
        self._closing = False
        info = {"protocol": PROTOCOL, "pid": os.getpid(), "address": self._address, "key": self._key.hex(),
                "started": time.time()}
        with os.fdopen(os.open(self.file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
            json.dump(info, f)
        self._thread = threading.Thread(target=self._accept, name="nuka-rpc", daemon=True)
        self._thread.start()

    def sessions(self) -> List[dict]:
        """The connected clients: name, process id, connected since and calls made."""
        with self._lock:
            return self._clients()

    def _clients(self) -> List[dict]:
        return [{"client": s.client, "pid": s.pid, "since": s.since, "calls": s.calls}
                for s in self._sessions.values() if not s.dropped]

    def _report(self) -> None:
        # Under the lock, so no report reaches the editor after close() returned.
        with self._lock:
            if not self._closing:
                self._app._native.rpc_clients(json.dumps(self._clients()))

    def disconnect(self) -> None:
        """Drops every client; each finds its connection closed at its next call."""
        with self._lock:
            for session in self._sessions.values():
                session.dropped = True
        self._report()

    def close(self) -> None:
        if self._listener is None:
            return
        with self._lock:
            self._closing = True
            for session in self._sessions.values():
                session.dropped = True
        # A connection of our own wakes the accept loop so it can see the flag.
        Client(self._address, family()).close()
        self._thread.join()
        self._listener.close()
        self._listener = None
        os.remove(self.file)
        self._app._native.rpc_clients("[]")

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
        # Polling lets a dropped client go while it is silent.
        try:
            while not session.dropped:
                if conn.poll(0.2):
                    data = conn.recv_bytes()
                    conn.send_bytes(json.dumps(self._reply(session, data)).encode("utf-8"))
        except (EOFError, OSError):
            pass
        finally:
            with self._lock:
                del self._sessions[session.number]
            conn.close()
            self._report()

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
        session.pid = int(args.get("pid") or 0)
        self._report()
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
        """The next drawn frame as a PPM at ``path`` (a file of the editor's own when empty).
        A shot into the editor's own folder only observes, so any access mode but off allows it;
        writing to the client's ``path`` is a file action that the access mode rules on."""
        path = args.get("path") or os.path.join(self._shots, f"{session.number}-{session.calls}.ppm")
        # The shot is new once the file's time moves; nothing is deleted before access is checked.
        before = os.stat(path).st_mtime_ns if os.path.exists(path) else None
        source = f"agent:{session.client}" if args.get("path") else f"rpc:{session.client}"
        reply = self._app._run("view.screenshot", {"path": path}, source)
        if not reply["ok"]:
            raise RpcError(reply["error"])
        if isinstance(reply["value"], dict) and reply["value"].get("state") == "pending":
            return reply["value"]
        deadline = time.monotonic() + float(args.get("timeout_s", 10.0))
        while not os.path.exists(path) or os.stat(path).st_mtime_ns == before:
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
