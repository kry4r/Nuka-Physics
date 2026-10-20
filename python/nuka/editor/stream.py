"""nuka.editor.stream -- the Python half of the editor's ``rerun.stream`` action.

The editor gathers the shown environment's link and body poses every few ticks; a thread here
takes them and logs them to Rerun the way ``nuka.editor.nkrec`` converts recordings. The action
calls ``start``, ``rename`` and ``stop``; it reports what ``start`` raises.
"""

from __future__ import annotations

import json
import threading
from typing import Optional

from . import nkrec

_PERIOD = 0.05  # seconds between takes


class _Stream:
    def __init__(self, native, config: dict):
        rr = nkrec.rerun_sdk()
        self._native = native
        self.rec = rr.RecordingStream("nuka_editor")
        if config["file"]:
            self.rec.save(config["file"])
        elif config["addr"]:
            self.rec.connect_grpc(config["addr"])
        else:
            self.rec.spawn()
        self._poses = nkrec.RerunPoses(self.rec, config["links"], config["bodies"], float(config["dt"]))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="nuka-rerun", daemon=True)
        self._thread.start()

    def rename(self, config: dict) -> None:
        with self._lock:
            self._take()
            self._poses.rename(config["links"], config["bodies"])

    def close(self) -> None:
        self._stop.set()
        self._thread.join()
        self.rec.flush()
        self.rec.disconnect()

    def _run(self) -> None:
        while not self._stop.wait(_PERIOD):
            with self._lock:
                self._take()
        with self._lock:
            self._take()

    def _take(self) -> None:
        for sample in nkrec.track(self._native.stream_take()):
            self._poses.log(sample)


_current: Optional[_Stream] = None


def start(app, config: str) -> None:
    """Streams to the viewer or file ``config`` names, ending the stream before."""
    global _current
    stop()
    _current = _Stream(app._native, json.loads(config))


def rename(config: str) -> None:
    """The links and bodies are named and numbered as ``config`` says from the next sample on."""
    _current.rename(json.loads(config))


def stop() -> None:
    """Logs what the editor gathered, then ends the stream."""
    global _current
    if _current is not None:
        _current.close()
        _current = None
