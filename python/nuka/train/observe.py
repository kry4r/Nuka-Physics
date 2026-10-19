"""nuka.observe.v1: a training run shows its first environments to viewers on this machine.

A viewer connects to the run's loopback port (`observe` in status.json) and sends one JSON line,
{proto, scene, layout: {links, bodies, articulations}, envs}. The run answers one line, {ok, shm,
bytes, slots, envs, links, bodies, articulations, run_envs, step_dt, spacing, terrain} or {ok: false,
error}, and keeps the connection open while the viewer watches. `scene` is the hex FNV-1a 64 of the run's
scene file followed by its archive. `envs` is what the viewer wants; the answer's is what the ring
holds. `spacing` is 0 when each environment has its own place in the world, else they overlap and a
viewer spreads them that many metres apart. Links and articulations must match the viewer's. The
run's world may add bodies of its own after the scene's (a baked ground), so a viewer with as many
bodies or fewer is accepted and reads the first ones of each environment. `terrain` is null or places
the run's height grid, {nrow, ncol, cell_x, cell_y, origin, base_z, scale_z, offset}: nrow x ncol
row-major float32 heights in [0, 1] at byte `offset` of the shared memory, written once, where world
z = base_z + height * scale_z, column c lies at origin x + c * cell_x and row r at origin y + r * cell_y.

The shared memory starts with a 64-byte header: u32 magic "NKOB", version, slots, envs, links,
bodies, articulations and slot bytes, then the u64 sequence of the newest whole frame (0 before the
first). Frame n fills slot n % slots: u64 begin, steps and end, 8 bytes of padding, then float32
arrays for the first `envs` environments in turn: link poses (links x 7), body poses (bodies x 7),
articulation base poses (articulations x 7), joint positions (links), and the commanded and actual
base velocity (3 each: world vx, vy and yaw rate). A pose is a position, then a quaternion w, x, y,
z. A copy taken after reading end == n and before reading begin == n is whole.
"""

from __future__ import annotations

import json
import mmap
import os
import socket
import struct
import threading
import time
from pathlib import Path

import numpy as np
import torch

import nuka

PROTO = "nuka.observe.v1"
MAGIC = 0x424F4B4E
VERSION = 1
SLOTS = 3
HEADER = 64
SLOT_HEAD = 32
MAX_ENVS = 32
PERIOD = 1.0 / 30.0


def scene_hash(scene: str) -> str:
    """FNV-1a 64 of the scene file, then of the archive beside it when it has one."""
    h = 0xCBF29CE484222325
    nks = Path(scene)
    for part in (nks, nks.with_suffix(".nka")):
        if part.is_file():
            for byte in part.read_bytes():
                h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def _rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """v rotated by the unit quaternions q (w, x, y, z)."""
    w, u = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(u, v, dim=1)
    return v + w * t + torch.cross(u, t, dim=1)


def _yaw(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class Server:
    """Serves the first environments of `env`, a Nuka gym env training `scene`, to local viewers."""

    def __init__(self, env, scene: str, spacing: float):
        world = env.world
        count = env.num_envs
        self._env = env
        self._links = torch.from_dlpack(world.buffer_view(nuka.ARTICULATION_LINK_POSE))
        self._bodies = torch.from_dlpack(world.buffer_view(nuka.RIGID_BODY_TRANSFORM))
        self._bases = torch.from_dlpack(world.buffer_view(nuka.BASE_POSE)).reshape(count, -1)
        self._joints = torch.from_dlpack(world.buffer_view(nuka.JOINT_POSITION))
        self._velocity = torch.from_dlpack(world.buffer_view(nuka.LINK_VELOCITY))
        self.envs = min(count, MAX_ENVS)
        self.layout = {"links": self._links.shape[1], "bodies": self._bodies.shape[1],
                       "articulations": self._bases.shape[1] // 7}
        poses = self.layout["links"] + self.layout["bodies"] + self.layout["articulations"]
        floats = self.envs * (poses * 7 + self.layout["links"] + 6)
        self.slot_bytes = SLOT_HEAD + (floats * 4 + 7) // 8 * 8
        terrain = world.terrain()
        heights = HEADER + SLOTS * self.slot_bytes
        self.bytes = heights + (0 if terrain is None else terrain[1].nbytes)
        self.shm = f"Local\\nuka-observe-{os.getpid()}"
        self._map = mmap.mmap(-1, self.bytes, tagname=self.shm)
        struct.pack_into("<8I", self._map, 0, MAGIC, VERSION, SLOTS, self.envs, self.layout["links"],
                         self.layout["bodies"], self.layout["articulations"], self.slot_bytes)
        self._payload = [np.frombuffer(self._map, np.float32, floats, HEADER + k * self.slot_bytes + SLOT_HEAD)
                         for k in range(SLOTS)]
        if terrain is not None:
            placement, values = terrain
            np.frombuffer(self._map, np.float32, values.size, heights)[:] = values.reshape(-1)
            terrain = {**placement, "offset": heights}
        self._answer = {"ok": True, "shm": self.shm, "bytes": self.bytes, "slots": SLOTS, "envs": self.envs,
                        **self.layout, "run_envs": count, "step_dt": env.dt * env.decimation,
                        "spacing": spacing, "terrain": terrain}
        self._scene = scene_hash(scene)
        self._sequence = 0
        self._steps = 0
        self._due = 0.0
        self._watchers = 0
        self._lock = threading.Lock()
        self._socket = socket.create_server(("127.0.0.1", 0))
        self.port = self._socket.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while True:
            conn, _ = self._socket.accept()
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _answer_to(self, line: bytes) -> dict:
        try:
            hello = json.loads(line)
        except ValueError:
            return {"ok": False, "error": "the handshake is not a JSON line"}
        if hello.get("proto") != PROTO:
            return {"ok": False, "error": f"this run speaks {PROTO}, not {hello.get('proto')}"}
        if hello.get("scene") != self._scene:
            return {"ok": False, "error": "the scene differs from the one this run trains"}
        theirs = hello.get("layout") or {}
        robot = all(theirs.get(k) == self.layout[k] for k in ("links", "articulations"))
        if not robot or not isinstance(theirs.get("bodies"), int) or theirs["bodies"] > self.layout["bodies"]:
            return {"ok": False, "error": f"the layout {theirs} does not fit this run's {self.layout}"}
        return self._answer

    def _serve(self, conn: socket.socket) -> None:
        with conn, conn.makefile("rwb") as stream:
            answer = self._answer_to(stream.readline())
            stream.write(json.dumps(answer).encode() + b"\n")
            stream.flush()
            if not answer["ok"]:
                return
            with self._lock:
                self._watchers += 1
            # The viewer sends nothing more; the connection ends when it stops watching.
            try:
                while conn.recv(4096):
                    pass
            except OSError:
                pass
            with self._lock:
                self._watchers -= 1

    def step(self) -> None:
        """Counts an environment step; while someone watches, publishes at most 30 frames a second."""
        self._steps += 1
        now = time.monotonic()
        if self._watchers == 0 or now < self._due:
            return
        self._due = now + PERIOD
        k = self.envs
        q = self._links[:k, 0, 3:7]
        twist = self._velocity[:k, 0]
        linear = _rotate(q, twist[:, 3:6])
        yaw_rate = _rotate(q, twist[:, 0:3])[:, 2:3]
        heading = _yaw(q)
        c, s = torch.cos(heading), torch.sin(heading)
        cmd = self._env.command[:k]
        command = torch.stack((c * cmd[:, 0] - s * cmd[:, 1], s * cmd[:, 0] + c * cmd[:, 1], cmd[:, 2]), dim=1)
        velocity = torch.cat((linear[:, :2], yaw_rate), dim=1)
        parts = (self._links[:k], self._bodies[:k], self._bases[:k], self._joints[:k], command, velocity)
        payload = torch.cat([p.reshape(-1) for p in parts]).cpu().numpy()
        self._sequence += 1
        n = self._sequence
        at = HEADER + (n % SLOTS) * self.slot_bytes
        struct.pack_into("<QQ", self._map, at, n, self._steps)
        self._payload[n % SLOTS][:] = payload
        struct.pack_into("<Q", self._map, at + 16, n)
        struct.pack_into("<Q", self._map, 32, n)
