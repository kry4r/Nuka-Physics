#!/usr/bin/env python3
"""Minimal host-only unit test for the new SceneBuilder authoring verbs
(add_collision_shape / compose / save). No device build -- exercises the
SceneIR authoring + .nks round-trip only, so it runs without a GPU.

    python examples/demo/bdx_oneshot_verbs_test.py
"""

from __future__ import annotations

import json
import os
import tempfile

import nuka
from bdx_oneshot_rope import write_rope_usda

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BDX = os.path.join(REPO, "examples", "scenes", "bdx_stand.nks")
HEAD = ("base/trunk_assembly/neck_pitch_assembly/head_pitch_to_yaw/"
        "neck_yaw_assembly/head_assembly")


def _count(doc, pred):
    n = [0]

    def walk(node):
        if isinstance(node, list):
            for c in node:
                walk(c)
            return
        if pred(node):
            n[0] += 1
        for c in node.get("children", []):
            walk(c)
    walk(doc["tree"])
    return n[0]


def test_add_collision_shape():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "a.nks")
        b = nuka.SceneBuilder.create(BDX)
        b.add_collision_shape(HEAD, nuka.PRIMITIVE_SPHERE, dims=[0.05],
                              pos=[0.0, 0.0, 0.0])
        b.save(out)
        b.destroy()
        doc = json.load(open(out))
        # a new sphere collision shape now exists (the duck ships none on the head)
        spheres = _count(doc, lambda n: (n.get("collision_shape") or {}).get("type")
                         == "sphere")
        assert spheres >= 1, f"expected an added sphere geom, got {spheres}"
        # a bad path is rejected loudly
        b2 = nuka.SceneBuilder.create(BDX)
        try:
            b2.add_collision_shape("no/such/node", nuka.PRIMITIVE_SPHERE, dims=[0.05])
            raise AssertionError("expected add_collision_shape to raise on a bad path")
        except RuntimeError:
            pass
        finally:
            b2.destroy()
    print("  add_collision_shape: OK")


def test_compose_and_save():
    with tempfile.TemporaryDirectory() as d:
        rope = os.path.join(d, "rope.usda")
        out = os.path.join(d, "c.nks")
        write_rope_usda(rope, n_links=4, seg=0.05, radius=0.01,
                        block=(0.05, 0.04, 0.02, 0.2))
        base = nuka.SceneBuilder.create(BDX)
        addon = nuka.SceneBuilder.create(rope)
        base.compose(addon, pos=[1.0, 0.0, 0.6], attach_at="rope")
        addon.destroy()
        base.save(out)
        base.destroy()
        doc = json.load(open(out))
        # the composed rope brought its anchor + 4 links + block as bodies with joints
        rope_bodies = _count(doc, lambda n: n.get("rigid_body") is not None
                             and n.get("name", "").startswith("rope"))
        joints = _count(doc, lambda n: n.get("joint") is not None)
        assert rope_bodies >= 6, f"expected >=6 rope bodies, got {rope_bodies}"
        assert joints >= 14 + 5, f"expected duck+rope joints, got {joints}"
    print(f"  compose + save: OK ({rope_bodies} rope bodies, {joints} joints)")


def test_save_roundtrip_loads():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "r.nks")
        b = nuka.SceneBuilder.create(BDX)
        b.add_material("m", base_color=[0.5, 0.5, 0.5])
        b.add_rigid_primitive(nuka.PRIMITIVE_BOX, dims=[0.2, 0.2, 0.1],
                              pos=[0.5, 0.0, 0.1], static=True, material="m")
        b.save(out)
        b.destroy()
        # the saved .nks re-imports as a fresh builder (round-trip is loadable)
        b2 = nuka.SceneBuilder.create(out)
        b2.destroy()
        assert os.path.exists(out.replace(".nks", ".nka")) or os.path.exists(
            os.path.join(d, "r.nka")), "expected a sibling .nka"
    print("  save round-trip loads: OK")


def main():
    print("[verbs_test] SceneBuilder add_collision_shape / compose / save")
    test_add_collision_shape()
    test_compose_and_save()
    test_save_roundtrip_loads()
    print("[verbs_test] ALL PASS")


if __name__ == "__main__":
    main()
