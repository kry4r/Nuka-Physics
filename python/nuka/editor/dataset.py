"""nuka.editor.dataset -- the editor's dataset export, built from the readings it captured.

``dataset.export`` steps the scene and appends each frame's sensor readings under
``<out>/.capture``; ``build`` then lays them out as::

    rgb/<camera>/000000.png        8-bit RGB, the color plane clipped to [0, 1]
    depth/<camera>/000000.npy      (H, W) float32, metres along each pixel's ray; inf where nothing is hit
    normal/<camera>/000000.npy     (H, W, 3) float32 surface normals
    albedo/<camera>/000000.npy     (H, W, 3) float32
    instance/<camera>/000000.png   16-bit, 1 + the index into meta ``instances`` (node paths); 0 for none
    semantic/<camera>/000000.png   16-bit, 1 + the index into meta ``classes``; 0 for no label or no hit
    lidar/<lidar>/000000.npy       (azimuth, elevation) float32 ranges
    state.npz                      each state sensor's (frames, values) float32, with ``tick`` and ``time``
    annotations.json               COCO: an image per frame and camera, a category per class, and an
                                   annotation per labeled node in view with its box, area and mask
    meta.json                      the scene, step, seed, randomization, labels, and every sensor's
                                   intrinsics and pose per frame

A camera looks down its -Z axis with +Y up; its poses are camera to world as x, y, z, qw, qx, qy,
qz, and K maps a ray to pixel (u, v) with rows running down from the top. ``read`` loads a dataset.
"""

from __future__ import annotations

import io
import json
import math
import pathlib
import shutil
import struct
import zipfile
import zlib

import numpy as np

_MISS = 0xFFFFFFFF
_INSTANCE_SHIFT = 20
_PLANES = ("rgb", "depth", "instance", "semantic", "normal", "albedo")
_FILES = ("state.npz", "annotations.json", "meta.json")


# The label of the node at `path` or its nearest labeled ancestor, and the node that carries it.
def _label_of(labels: dict, path: str) -> tuple[str, str]:
    at = path
    while True:
        if at in labels:
            return labels[at], at
        if "/" not in at:
            return "", ""
        at = at.rpartition("/")[0]


def _png(path: pathlib.Path, image: np.ndarray) -> None:
    """Writes (H, W, 3) uint8 as RGB or (H, W) uint16 as gray."""
    h, w = image.shape[:2]
    if image.dtype == np.uint16:
        rows, depth, color = image.astype(">u2").view(np.uint8).reshape(h, 2 * w), 16, 0
    else:
        rows, depth, color = image.reshape(h, 3 * w), 8, 2
    raw = np.concatenate([np.zeros((h, 1), np.uint8), rows], axis=1).tobytes()

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    header = struct.pack(">IIBBBBB", w, h, depth, color, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 6)) +
                     chunk(b"IEND", b""))


def _read_png(path: pathlib.Path) -> np.ndarray:
    """A PNG as ``_png`` writes it."""
    data = path.read_bytes()
    pos, idat = 8, b""
    while pos < len(data):
        size, tag = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + size]
        if tag == b"IHDR":
            w, h, depth, color = struct.unpack(">IIBB", body[:10])
        elif tag == b"IDAT":
            idat += body
        pos += 12 + size
    channels = 3 if color == 2 else 1
    raw = np.frombuffer(zlib.decompress(idat), np.uint8)
    rows = raw.reshape(h, 1 + w * channels * depth // 8)[:, 1:].copy()
    if depth == 16:
        return rows.view(">u2").reshape(h, w).astype(np.uint16)
    return rows.reshape(h, w, channels)


# Uncompressed COCO RLE: run lengths of the column-major mask, starting with a run of zeros.
def _rle(mask: np.ndarray) -> dict:
    flat = np.concatenate([[0], mask.flatten(order="F").astype(np.int8), [0]])
    bounds = np.concatenate([[0], np.flatnonzero(flat[1:] != flat[:-1]), [mask.size]])
    counts = np.diff(bounds).tolist()
    if counts[-1] == 0:
        counts.pop()
    return {"counts": counts, "size": [int(mask.shape[0]), int(mask.shape[1])]}


# An npz whose bytes depend only on the arrays.
def _npz(path: pathlib.Path, arrays: dict) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, array in arrays.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.ascontiguousarray(array))
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, buffer.getvalue())


def _intrinsics(c: dict) -> list:
    f = 0.5 * c["height"] / math.tan(math.radians(c["vfov"]) * 0.5)
    return [[f, 0.0, 0.5 * c["width"]], [0.0, f, 0.5 * c["height"]], [0.0, 0.0, 1.0]]


def build(out: str) -> None:
    """Lays out the capture under ``out`` as the dataset and drops the capture."""
    root = pathlib.Path(out)
    capture = root / ".capture"
    m = json.loads((capture / "manifest.json").read_text(encoding="utf-8"))
    for name in (*_PLANES, "lidar"):
        shutil.rmtree(root / name, ignore_errors=True)
    for name in _FILES:
        (root / name).unlink(missing_ok=True)
    frames, classes, labels = m["frames"], m["classes"], m["labels"]
    owned = [_label_of(labels, path) for path in m["instances"]]
    owners = sorted({node for _, node in owned if node})
    # By instance: its semantic class and the labeled node it belongs to, both 0 for none.
    class_of = np.zeros(1 << (32 - _INSTANCE_SHIFT), np.uint16)
    owner_of = np.zeros(1 << (32 - _INSTANCE_SHIFT), np.int64)
    for k, (label, node) in enumerate(owned):
        if label:
            class_of[k] = 1 + classes.index(label)
            owner_of[k] = 1 + owners.index(node)

    def stream(name: str) -> np.ndarray:
        return np.fromfile(capture / name, dtype=np.float32)

    images, annotations = [], []
    cameras = []
    for c in m["cameras"]:
        h, w = c["height"], c["width"]
        planes = {name: stream(file).reshape(frames, h, w, -1) for name, file in c["streams"].items()}
        for name in m["channels"]:
            (root / name / c["name"]).mkdir(parents=True)
        cameras.append((c, planes))
    for f in range(frames):
        for c, planes in cameras:
            h, w, name = c["height"], c["width"], c["name"]
            frame = f"{f:06d}"
            bits = planes["instance"][f, ..., 0].view(np.uint32)
            miss = bits == _MISS
            instance = np.where(miss, 0, bits >> _INSTANCE_SHIFT).astype(np.int64)
            if "rgb" in m["channels"]:
                rgb = np.rint(np.clip(planes["color"][f, ..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)
                _png(root / "rgb" / name / f"{frame}.png", rgb)
            if "depth" in m["channels"]:
                np.save(root / "depth" / name / f"{frame}.npy", planes["depth"][f, ..., 0])
            for plane in ("normal", "albedo"):
                if plane in m["channels"]:
                    np.save(root / plane / name / f"{frame}.npy", planes[plane][f, ..., :3])
            ids = {"instance": np.where(miss, 0, instance + 1), "semantic": np.where(miss, 0, class_of[instance])}
            for plane, values in ids.items():
                if plane in m["channels"]:
                    _png(root / plane / name / f"{frame}.png", values.astype(np.uint16))
            image = len(images) + 1
            images.append({"id": image, "file_name": f"rgb/{name}/{frame}.png", "width": w, "height": h,
                           "camera": name, "frame": f, "tick": m["ticks"][f], "time": m["times"][f]})
            owner = np.where(miss, 0, owner_of[instance])
            for o in np.unique(owner[owner > 0]).tolist():
                mask = owner == o
                rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
                x, y = int(cols[0]), int(rows[0])
                label = labels[owners[o - 1]]
                annotations.append({"id": len(annotations) + 1, "image_id": image,
                                    "category_id": 1 + classes.index(label), "label": label,
                                    "node": owners[o - 1], "instances": np.unique(instance[mask]).tolist(),
                                    "bbox": [x, y, int(cols[-1]) - x + 1, int(rows[-1]) - y + 1],
                                    "area": int(mask.sum()), "iscrowd": 0, "segmentation": _rle(mask)})
    for lidar in m["lidars"]:
        ranges = stream(lidar["stream"]).reshape(frames, lidar["az_count"], lidar["el_count"])
        (root / "lidar" / lidar["name"]).mkdir(parents=True)
        for f in range(frames):
            np.save(root / "lidar" / lidar["name"] / f"{f:06d}.npy", ranges[f])
    if m["states"]:
        state = {s["name"]: stream(s["stream"]).reshape(frames, -1) for s in m["states"]}
        _npz(root / "state.npz", {**state, "tick": np.array(m["ticks"], np.int64),
                                  "time": np.array(m["times"], np.float64)})
    coco = {"info": {"description": "Nuka editor dataset", "version": "1"}, "images": images,
            "categories": [{"id": k + 1, "name": c, "supercategory": ""} for k, c in enumerate(classes)],
            "annotations": annotations}
    (root / "annotations.json").write_text(json.dumps(coco), encoding="utf-8")
    meta = {key: m[key] for key in ("scene", "dt", "env", "every", "seed", "frames", "randomize", "channels",
                                    "labels", "classes", "instances", "ticks", "times")}
    meta["format"], meta["version"] = "nuka-dataset", 1
    meta["cameras"] = [{**{k: v for k, v in c.items() if k != "streams"}, "K": _intrinsics(c)} for c, _ in cameras]
    meta["lidars"] = [{k: v for k, v in lidar.items() if k != "stream"} for lidar in m["lidars"]]
    meta["states"] = [{k: v for k, v in s.items() if k != "stream"} for s in m["states"]]
    (root / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    shutil.rmtree(capture)


class Dataset:
    """A built dataset: ``meta`` and ``annotations`` as written, and each frame's arrays."""

    def __init__(self, path: str):
        self.path = pathlib.Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text(encoding="utf-8"))
        self.annotations = json.loads((self.path / "annotations.json").read_text(encoding="utf-8"))

    @property
    def frames(self) -> int:
        return self.meta["frames"]

    @property
    def cameras(self) -> list:
        return [c["name"] for c in self.meta["cameras"]]

    def camera(self, name: str) -> dict:
        return next(c for c in self.meta["cameras"] if c["name"] == name)

    def rgb(self, camera: str, frame: int) -> np.ndarray:
        return _read_png(self.path / "rgb" / camera / f"{frame:06d}.png")

    def depth(self, camera: str, frame: int) -> np.ndarray:
        return np.load(self.path / "depth" / camera / f"{frame:06d}.npy")

    def normal(self, camera: str, frame: int) -> np.ndarray:
        return np.load(self.path / "normal" / camera / f"{frame:06d}.npy")

    def albedo(self, camera: str, frame: int) -> np.ndarray:
        return np.load(self.path / "albedo" / camera / f"{frame:06d}.npy")

    def instance(self, camera: str, frame: int) -> np.ndarray:
        return _read_png(self.path / "instance" / camera / f"{frame:06d}.png")

    def semantic(self, camera: str, frame: int) -> np.ndarray:
        return _read_png(self.path / "semantic" / camera / f"{frame:06d}.png")

    def lidar(self, name: str, frame: int) -> np.ndarray:
        return np.load(self.path / "lidar" / name / f"{frame:06d}.npy")

    def state(self) -> dict:
        with np.load(self.path / "state.npz") as z:
            return {key: z[key] for key in z.files}


def read(path: str) -> Dataset:
    return Dataset(path)
