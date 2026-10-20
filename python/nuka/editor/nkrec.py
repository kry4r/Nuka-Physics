"""nuka.editor.nkrec -- read the editor's recordings and turn them into Rerun recordings.

A recording (``.nkrec``, written by the ``timeline.save`` action) holds one line of the editor's
history: a JSON manifest, the keyframes, the inputs between them, the shown environment's link
and body poses, and the solver records::

    python -m nuka.editor.nkrec info run.nkrec
    python -m nuka.editor.nkrec rrd run.nkrec [run.rrd]

``rrd`` needs the optional rerun-sdk (``pip install rerun-sdk``).
"""

from __future__ import annotations

import argparse
import json
import re
import struct
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np

MAGIC = b"NKREC\0\0\0"
FORMAT = 2

# The leading fields of a nuka_telemetry_record_t.
_RECORD = struct.Struct("<6I4fI3f")
_RECORD_FIELDS = ("sequence", "substep", "sweeps", "sweep_budget", "flags", "contacts", "vertex_residual",
                  "vertex_force", "contact_residual", "max_penetration", "penetration_row", "acceleration",
                  "energy", "energy_residual")


@dataclass
class Sample:
    """The shown environment's poses at one tick; each row is x, y, z and the quaternion w, x, y, z."""
    tick: int
    env: int
    links: np.ndarray
    bodies: np.ndarray


@dataclass
class Keyframe:
    tick: int
    sim_time: float
    pinned: bool  # holds writes the input log cannot replay
    loads: int    # the perturbations held then
    bytes: int    # its size in the file


@dataclass
class Recording:
    manifest: dict
    keyframes: List[Keyframe]
    inputs: int
    samples: List[Sample]
    records: List[dict]


def read(path: str) -> Recording:
    """Reads a recording; the keyframes' world state stays in the file."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != MAGIC:
        raise ValueError(f"{path} is no Nuka recording")
    (fmt,) = struct.unpack_from("<I", data, 8)
    if fmt != FORMAT:
        raise ValueError(f"{path} is recording format {fmt}; this reader takes {FORMAT}")
    manifest, keyframes, inputs, samples, records = {}, [], 0, [], []
    at = 12
    while at < len(data):
        tag = data[at:at + 4]
        (size,) = struct.unpack_from("<Q", data, at + 4)
        body = memoryview(data)[at + 12:at + 12 + size]
        at += 12 + size
        if tag == b"MANI":
            manifest = json.loads(bytes(body).decode("utf-8"))
        elif tag == b"KEYF":
            tick, sim_time, _, pinned, loads = struct.unpack_from("<QdQII", body)
            keyframes.append(Keyframe(tick, sim_time, bool(pinned), loads, size))
        elif tag == b"LOGS":
            (inputs,) = struct.unpack_from("<Q", body)
        elif tag == b"TRAK":
            samples = track(body)
        elif tag == b"TELE":
            count, stride = struct.unpack_from("<II", body)
            records = [dict(zip(_RECORD_FIELDS, _RECORD.unpack_from(body, 8 + i * stride))) for i in range(count)]
    return Recording(manifest, keyframes, inputs, samples, records)


def track(body) -> List[Sample]:
    """The samples of a track chunk: a count, then each sample's tick, environment, link and body
    counts and poses."""
    (count,) = struct.unpack_from("<Q", body)
    samples, off = [], 8
    for _ in range(count):
        tick, env, links, bodies = struct.unpack_from("<QIII", body, off)
        off += 20
        poses = np.frombuffer(body, dtype="<f4", count=(links + bodies) * 7, offset=off).reshape(-1, 7)
        off += (links + bodies) * 28
        samples.append(Sample(tick, env, poses[:links].copy(), poses[links:].copy()))
    return samples


def _entity(kind: str, names: Sequence[str], row: int) -> str:
    name = names[row] if row < len(names) and names[row] else str(row)
    return f"world/{kind}/" + re.sub(r"[^A-Za-z0-9_/-]", "_", name).strip("/")


def rerun_sdk():
    try:
        import rerun
    except ImportError as error:
        raise RuntimeError("Rerun needs rerun-sdk: pip install rerun-sdk") from error
    return rerun


class RerunPoses:
    """Logs samples to a Rerun stream: each link and body, named after its node, as a transform
    over the ``tick`` and ``sim_time`` timelines. Body rows named None belong to links and are left out."""

    def __init__(self, stream, links: Sequence[str], bodies: Sequence[str], dt: float):
        self.rr = rerun_sdk()
        self.stream = stream
        self.dt = dt
        self.rename(links, bodies)

    def rename(self, links: Sequence[str], bodies: Sequence[str]) -> None:
        self._names = {"links": list(links), "bodies": list(bodies)}
        self._paths = {"links": [], "bodies": []}

    def _rows(self, kind: str, count: int) -> List[Optional[str]]:
        paths, names = self._paths[kind], self._names[kind]
        for row in range(len(paths), count):
            if row < len(names) and names[row] is None:
                paths.append(None)
                continue
            paths.append(_entity(kind, names, row))
            self.stream.log(paths[-1], self.rr.TransformAxes3D(0.1), static=True)
        return paths

    def log(self, s: Sample) -> None:
        self.stream.set_time("tick", sequence=s.tick)
        self.stream.set_time("sim_time", duration=s.tick * self.dt)
        for kind, poses in (("links", s.links), ("bodies", s.bodies)):
            for path, p in zip(self._rows(kind, len(poses)), poses):
                if path is None:
                    continue
                rotation = self.rr.Quaternion(xyzw=[p[4], p[5], p[6], p[3]])
                self.stream.log(path, self.rr.Transform3D(translation=p[:3], quaternion=rotation))


def to_rerun(recording: Union[str, Recording], out: str) -> str:
    """Writes a Rerun recording: each link and body as a transform over the ``tick`` timeline, and
    the solver records as scalars over ``record``."""
    rr = rerun_sdk()
    rec = read(recording) if isinstance(recording, str) else recording
    m = rec.manifest
    stream = rr.RecordingStream("nuka_editor")
    stream.save(out)
    poses = RerunPoses(stream, m.get("links", []), m.get("bodies", []), float(m.get("dt", 0.0)))
    for s in rec.samples:
        poses.log(s)
    for r in rec.records:
        stream.set_time("record", sequence=r["sequence"])
        for key in ("vertex_residual", "contact_residual", "max_penetration", "energy"):
            stream.log(f"solver/{key}", rr.Scalars(r[key]))
    stream.flush()
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m nuka.editor.nkrec")
    sub = parser.add_subparsers(dest="command", required=True)
    info = sub.add_parser("info", help="print a recording's manifest and contents")
    info.add_argument("path")
    rrd = sub.add_parser("rrd", help="write a Rerun recording")
    rrd.add_argument("path")
    rrd.add_argument("out", nargs="?")
    args = parser.parse_args(argv)
    if args.command == "info":
        rec = read(args.path)
        print(json.dumps(rec.manifest, indent=2))
        print(f"keyframes {len(rec.keyframes)}  inputs {rec.inputs}  samples {len(rec.samples)}  records {len(rec.records)}")
        return 0
    out = args.out or re.sub(r"\.nkrec$", "", args.path) + ".rrd"
    print(to_rerun(args.path, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
