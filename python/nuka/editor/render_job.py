"""The editor's offline renders: ``nuka-editor --render job.json`` path-traces one view of a saved
scene and state to a PNG and answers in the job's result file."""

from __future__ import annotations

import json
import struct
import time
import zlib


def _write_png(path: str, rgb) -> None:
    height, width, _ = rgb.shape
    rows = b"".join(b"\x00" + rgb[y].tobytes() for y in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
                chunk(b"IDAT", zlib.compress(rows, 6)) + chunk(b"IEND", b""))


def run(job_path: str) -> int:
    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)
    start = time.perf_counter()
    result = {"ok": False, "path": job["output"], "width": job["width"], "height": job["height"], "spp": job["spp"]}
    try:
        import nuka
        world = nuka.World.create_from_scene(nuka.Device.create(0), job["scene"], env_count=job["envs"], dt=job["dt"])
        checkpoint = world.read_checkpoint(job["checkpoint"])
        world.restore_checkpoint(checkpoint)
        checkpoint.close()
        camera = job["camera"]
        rgb = world.render_beauty(eye=camera["eye"], look=camera["look"], up=camera["up"], fov_deg=camera["fov"],
                                  width=job["width"], height=job["height"], spp=job["spp"])
        _write_png(job["output"], rgb)
        result["ok"] = True
    except Exception as error:  # the editor shows what failed
        result["error"] = f"{type(error).__name__}: {error}"
    result["seconds"] = time.perf_counter() - start
    with open(job["result"], "w", encoding="utf-8") as f:
        json.dump(result, f)
    return 0 if result["ok"] else 1
