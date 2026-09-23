"""Build editable NKS laboratory layouts and their shared NKA mesh library."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def rectangles(mask):
    width, height = mask.size
    active, result = {}, []
    for y in range(height + 1):
        spans, start = [], None
        if y < height:
            for x in range(width + 1):
                opaque = x < width and mask.getpixel((x, y)) >= 128
                if opaque and start is None:
                    start = x
                elif not opaque and start is not None:
                    spans.append((start, x))
                    start = None
        for span in list(active):
            if span not in spans:
                result.append((*span, active.pop(span), y))
        for span in spans:
            active.setdefault(span, y)
    return [(x0 / width - 0.5, 0.5 - y1 / height, x1 / width - 0.5, 0.5 - y0 / height)
            for x0, x1, y0, y1 in result]


def lettering(text, font):
    bounds = font.getbbox(text)
    mask = Image.new("L", (bounds[2] - bounds[0], bounds[3] - bounds[1]))
    ImageDraw.Draw(mask).text((-bounds[0], -bounds[1]), text, fill=255, font=font)
    return mask


class Mesh:
    def __init__(self):
        self.positions, self.normals, self.indices = [], [], []

    def face(self, a, b, c, d):
        first = len(self.positions)
        normal = np.cross(np.array(b) - a, np.array(c) - a)
        normal /= np.linalg.norm(normal)
        self.positions.extend([a, b, c, d])
        self.normals.extend([normal] * 4)
        self.indices.extend([first, first + 1, first + 2])
        if np.linalg.norm(np.array(d) - a) > 0:
            self.indices.extend([first, first + 2, first + 3])

    def quad(self, x0, y0, x1, y1, z):
        self.face((x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z))

    def payload(self):
        return (struct.pack("<II", len(self.positions), len(self.indices))
                + np.array(self.positions, dtype="<f4").tobytes()
                + np.array(self.normals, dtype="<f4").tobytes()
                + np.zeros((len(self.positions), 2), dtype="<f4").tobytes()
                + np.array(self.indices, dtype="<u4").tobytes())


def slab(half, bevel):
    x, y, z = half
    edge, cut = min(bevel, x * 0.3, y * 0.3, z * 0.45), min(x, y) * 0.12

    def octagon(rx, ry, rz):
        return [(-rx + cut, -ry, rz), (rx - cut, -ry, rz), (rx, -ry + cut, rz),
                (rx, ry - cut, rz), (rx - cut, ry, rz), (-rx + cut, ry, rz),
                (-rx, ry - cut, rz), (-rx, -ry + cut, rz)]

    rings = [octagon(x - edge, y - edge, -z), octagon(x, y, -z + edge),
             octagon(x, y, z - edge), octagon(x - edge, y - edge, z)]
    mesh = Mesh()
    for level in range(3):
        for i in range(8):
            j = (i + 1) % 8
            mesh.face(rings[level][i], rings[level][j], rings[level + 1][j], rings[level + 1][i])
    for i in range(1, 7):
        mesh.face(rings[0][0], rings[0][i + 1], rings[0][i], rings[0][0])
        mesh.face(rings[3][0], rings[3][i], rings[3][i + 1], rings[3][0])
    return mesh


def sign(spans, width, height):
    mesh = Mesh()
    for x0, y0, x1, y1 in spans:
        mesh.face((x0 * width, 0, y0 * height), (x1 * width, 0, y0 * height),
                  (x1 * width, 0, y1 * height), (x0 * width, 0, y1 * height))
    return mesh


def look_rotation(eye, target, up=(0, 0, 1)):
    backward = np.array(eye, dtype=float) - target
    backward /= np.linalg.norm(backward)
    right = np.cross(up, backward)
    right /= np.linalg.norm(right)
    rotation = np.column_stack((right, np.cross(backward, right), backward))
    trace = np.trace(rotation)
    if trace > 0:
        s = math.sqrt(trace + 1) * 2
        q = [s / 4, (rotation[2, 1] - rotation[1, 2]) / s,
             (rotation[0, 2] - rotation[2, 0]) / s, (rotation[1, 0] - rotation[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(rotation)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1 + rotation[i, i] - rotation[j, j] - rotation[k, k]) * 2
        q = [(rotation[k, j] - rotation[j, k]) / s, 0, 0, 0]
        q[i + 1] = s / 4
        q[j + 1] = (rotation[j, i] + rotation[i, j]) / s
        q[k + 1] = (rotation[k, i] + rotation[i, k]) / s
    return [float(value) for value in np.array(q) / np.linalg.norm(q)]


class Library:
    def __init__(self):
        self.payloads, self.indices = [], {}

    def add(self, mesh):
        payload = mesh.payload()
        digest = hashlib.sha256(payload).digest()
        if digest not in self.indices:
            self.indices[digest] = len(self.payloads)
            self.payloads.append(payload)
        return f"lab_geometry.nka#MESH/{self.indices[digest]}"

    def write(self, path):
        tag = int.from_bytes(b"MESH", "big")
        offset = 12 + 28 * len(self.payloads)
        toc = []
        for payload in self.payloads:
            digest = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
            toc.append(struct.pack("<IQQQ", tag, offset, len(payload), digest))
            offset += len(payload)
        path.write_bytes(b"NKA1" + struct.pack("<II", 1, len(toc)) + b"".join(toc) + b"".join(self.payloads))


class Layout:
    def __init__(self, library, origin):
        self.library = library
        self.root = {"name": "laboratory", "transform": {"pos": list(origin)},
                     "rigid_body": {"is_static": True, "mass": 0}, "children": []}
        self.document = {"nks_version": 1, "imports": [{"file": "appearance.nks"}], "tree": [self.root]}

    def mesh(self, name, mesh, material, center=(0, 0, 0), quat=(1, 0, 0, 0)):
        self.root["children"].append({"name": name, "transform": {"pos": list(center), "quat": list(quat)},
                                      "visual_mesh": {"type": "trimesh", "mesh": self.library.add(mesh),
                                                      "material": material}})

    def slab(self, name, center, half, bevel, material, upright=False):
        quat = (math.sqrt(0.5), math.sqrt(0.5), 0, 0) if upright else (1, 0, 0, 0)
        self.mesh(name, slab(half, bevel), material, center, quat)

    def camera(self, name, eye, target, fov=40, near=0.01, far=100, up=(0, 0, 1), shadow_radius=0):
        self.document["tree"].append({"name": name, "camera": {
            "local": {"pos": list(eye), "quat": look_rotation(eye, target, up)},
            "vertical_fov_degrees": fov, "near_clip": near, "far_clip": far,
            "focus_distance": float(np.linalg.norm(np.array(eye) - target)), "shadow_radius": shadow_radius}})

    def lighting(self, center, radius):
        self.document["tree"].append({"name": "key_light", "light": {
            "type": "directional", "local": {"quat": look_rotation((-0.50, -0.70, 0.95), (0, 0, 0))},
            "color": [3.4, 3.2, 2.9], "intensity": 1}})
        self.document["environment"] = {"shadow": {"center": list(center), "radius": radius}}


def laboratory(library, marks, cell, cells, origin=(0, 0, 0), plinth=None):
    layout = Layout(library, origin)
    scale = cell * cells
    x = y = scale
    half, line, elevation = cell * 64, cell * 0.002, cell * 0.0006
    room = Mesh()
    room.quad(-half, -half, half, half, -scale * 0.15)
    for i in range(48):
        a, b = i * math.pi / 96, (i + 1) * math.pi / 96
        ya, yb = y * 1.8 + scale * math.sin(a), y * 1.8 + scale * math.sin(b)
        za, zb = scale * (0.85 - math.cos(a)), scale * (0.85 - math.cos(b))
        room.face((-half, ya, za), (half, ya, za), (half, yb, zb), (-half, yb, zb))
    room.face((-half, y * 1.8 + scale, scale * 0.85), (half, y * 1.8 + scale, scale * 0.85),
              (half, y * 1.8 + scale, scale * 8), (-half, y * 1.8 + scale, scale * 8))
    layout.mesh("cyclorama", room, "room")
    layout.slab("lower_plinth", (0, 0, -scale * 0.105), (x * 1.07, y * 1.07, scale * 0.045), scale * 0.012, "graphite")
    layout.slab("avocado_edge", (0, 0, -scale * 0.051), (x * 1.085, y * 1.085, scale * 0.009), scale * 0.004, "avocado")
    layout.slab("plinth_shell", (0, 0, -scale * 0.024), (x * 1.1, y * 1.1, scale * 0.022), scale * 0.009, "warm_shell")
    layout.slab("work_deck", (0, 0, -scale * 0.008), (x, y, scale * 0.008), scale * 0.003, "work_deck")
    for side in (-1, 1):
        layout.slab(f"edge_light_{side}", (side * x * 0.67, -y * 1.075, -scale * 0.05),
                    (x * 0.17, scale * 0.004, scale * 0.004), scale * 0.001, "edge_light")
    minor, major, corners, ticks, mark = Mesh(), Mesh(), Mesh(), Mesh(), Mesh()
    for i in range(-cells + 1, cells):
        coordinate = i * cell
        mesh = major if i % 4 == 0 else minor
        mesh.quad(coordinate - line, -y * 0.92, coordinate + line, y * 0.92, elevation * (2 if i % 4 == 0 else 1))
        if abs(coordinate) < y * 0.92:
            mesh.quad(-x * 0.92, coordinate - line, x * 0.92, coordinate + line, elevation * (3 if i % 4 == 0 else 1.5))
    for side in (-1, 1):
        for end in (-1, 1):
            cx, cy = side * x * 0.91, end * y * 0.91
            corners.quad(min(cx, cx - side * cell * 0.5), cy - line * 6,
                         max(cx, cx - side * cell * 0.5), cy + line * 6, elevation * 5)
            corners.quad(cx - line * 6, min(cy, cy - end * cell * 0.5),
                         cx + line * 6, max(cy, cy - end * cell * 0.5), elevation * 5)
        for i in range(-cells * 2 + 1, cells * 2):
            tick = i * cell * 0.5
            ticks.quad(tick - line * 2, side * y * 1.036 - cell * 0.02,
                       tick + line * 2, side * y * 1.036 + cell * (0.05 if i % 2 == 0 else 0.02), elevation)
    for x0, y0, x1, y1 in marks[0]:
        mark.quad(-x * 0.71 + x0 * scale * 0.17, -y * 0.70 + y0 * scale * 0.17,
                  -x * 0.71 + x1 * scale * 0.17, -y * 0.70 + y1 * scale * 0.17, elevation * 6)
    for name, mesh, material in (("grid_minor", minor, "grid_minor"), ("grid_major", major, "grid_major"),
                                 ("corner_marks", corners, "avocado"), ("ruler", ticks, "edge_metal"),
                                 ("deck_mark", mark, "avocado")):
        layout.mesh(name, mesh, material)
    rear = y * 1.04
    layout.slab("backboard", (0, rear + scale * 0.09, scale * 0.54),
                (x * 1.08, scale * 0.56, scale * 0.035), scale * 0.014, "graphite", True)
    for side in (-1, 1):
        layout.slab(f"back_panel_{side}", (side * x * 0.56, rear + scale * 0.042, scale * 0.56),
                    (x * 0.515, scale * 0.52, scale * 0.012), scale * 0.006, "warm_shell", True)
        layout.slab(f"back_frame_{side}", (side * x * 1.065, rear + scale * 0.012, scale * 0.56),
                    (scale * 0.018, scale * 0.53, scale * 0.020), scale * 0.005, "edge_metal", True)
    layout.slab("back_frame_top", (0, rear + scale * 0.012, scale * 1.09),
                (x * 1.07, scale * 0.018, scale * 0.020), scale * 0.005, "edge_metal", True)
    plaque = np.array((x * 0.53, rear - scale * 0.043, scale * 0.69))
    layout.slab("plaque", plaque, (scale * 0.35, scale * 0.28, scale * 0.018), scale * 0.01, "graphite", True)
    layout.mesh("plaque_mark", sign(marks[0], scale * 0.28, scale * 0.28), "avocado", plaque + (0, -scale * 0.019, scale * 0.07))
    layout.mesh("plaque_name", sign(marks[1], scale * 0.39, scale * 0.055), "lettering", plaque + (0, -scale * 0.019, -scale * 0.11))
    layout.mesh("plaque_label", sign(marks[2], scale * 0.41, scale * 0.027), "lettering", plaque + (0, -scale * 0.019, -scale * 0.19))
    for i in range(7):
        layout.slab(f"vertical_inset_{i}", (-scale * (0.24 + 0.105 * i), rear + scale * 0.022, scale * 0.53),
                    (scale * 0.009, scale * 0.41, scale * 0.018), scale * 0.004, "edge_metal", True)
    if plinth:
        center, p = np.array(plinth[0]) - origin, plinth[1]
        layout.slab("cabinet_core", center, (p[0] * 0.90, p[1] * 0.89, p[2]), scale * 0.018, "graphite")
        for side in (-1, 1):
            layout.slab(f"cabinet_shell_{side}", center + (side * p[0] * 0.89, 0, 0),
                        (p[0] * 0.075, p[1] * 0.94, p[2] * 0.94), scale * 0.015, "warm_shell")
        layout.slab("cabinet_cap", center + (0, 0, p[2] * 0.89), (p[0], p[1], p[2] * 0.075), scale * 0.008, "edge_metal")
        layout.slab("cabinet_edge", center + (0, -p[1] * 0.905, p[2] * 0.68),
                    (p[0] * 0.75, scale * 0.010, scale * 0.006), scale * 0.002, "avocado")
        for side in (-1, 1):
            for i in range(4):
                layout.slab(f"cabinet_vent_{side}_{i}", center + (side * p[0] * (0.48 + 0.085 * i), -p[1] * 0.9, -p[2] * 0.2),
                            (p[0] * 0.008, p[2] * 0.28, scale * 0.004), scale * 0.002, "edge_metal", True)
        layout.mesh("cabinet_name", sign(marks[1], p[0] * 0.63, p[2] * 0.16), "lettering", center + (0, -p[1] * 0.912, -p[2] * 0.20))
    return layout


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "examples/assets/nuka_lab")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    alpha = Image.open(root / "docs/media/nuka-logo.png").convert("RGBA").getchannel("A")
    font = ImageFont.truetype(str(root / "external/imgui/fonts/JetBrainsMono-Bold.ttf"), 96)
    marks = [rectangles(alpha), rectangles(lettering("N U K A", font)), rectangles(lettering("DYNAMICS LAB", font))]
    library = Library()
    gripper = laboratory(library, marks, 0.25, 5, (0.35, 0, 0), ((0.35, 0, 0.18), (0.62, 0.42, 0.18)))
    target, radius = np.array((0.35, 0, 0.45)), 1.6
    for name, direction, distance, up in (("three-quarter", (0.9, -1.8, 0.9), 3.2, (0, 0, 1)),
                                        ("front", (0, -2, 0.65), 3.6, (0, 0, 1)),
                                        ("top", (0, 0, 1), 3.2, (0, 1, 0))):
        eye = target + np.array(direction) * (distance * radius / np.linalg.norm(direction))
        gripper.camera(name, eye, target, near=radius * 0.05, far=radius * 100, up=up)
    gripper.camera("close", (0.67, -0.10, 0.62), (0.5033, -0.0435, 0.5140), 29, 0.001, 30, shadow_radius=0.21)
    gripper.lighting(target, radius * 2)
    bunny = laboratory(library, marks, 0.05, 4)
    bunny.camera("overview", (0.48, -0.78, 0.46), (0, 0, 0.16), 32, 0.005, 30)
    bunny.camera("close", (0.34, -0.51, 0.38), (0, 0, 0.08), 32, 0.005, 30)
    bunny.lighting((0, 0, 0.12), 0.4)
    compression = laboratory(library, marks, 0.025, 4)
    compression.camera("overview", (0.20, -0.48, 0.245), (0, 0, 0.09), 30, 0.003, 30)
    compression.lighting((0, 0, 0.06), 0.2)
    for name, layout in (("gripper", gripper), ("bunny", bunny), ("compression", compression)):
        (args.output / f"{name}.nks").write_text(json.dumps(layout.document, indent=2) + "\n", encoding="utf-8")
        print(f"{name}: {len(layout.root['children'])} editable visual nodes")
    library.write(args.output / "lab_geometry.nka")
    print(f"{len(library.payloads)} shared meshes, {(args.output / 'lab_geometry.nka').stat().st_size} bytes")


if __name__ == "__main__":
    main()
