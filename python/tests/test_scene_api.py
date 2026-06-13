"""pytest: nuka.Scene -- the GENERIC scene-authoring C-ABI (M9 T4).

An end-to-end API gate (NOT a TDD unit test): exercise the public scene surface
on a real authored scene the way a caller would.

Host-only path (no CUDA needed):
  * load examples/scenes/h1_cup_table.nks (dispatch by extension; resolves imports);
  * find("cup/body") -> True, a bogus path -> False;
  * set_physics_material("cup.*", static_friction=0.8) touches the cup shape;
  * save to a tmp .nks (+ sibling .nka), reload, and assert the node still exists
    AND the material took (friction persisted through Save/Load);
  * compose() a copy onto itself stays consistent (find still resolves).

CUDA-gated path (skips cleanly with no device / no phi v2 backend):
  * settle determinism -- settle twice, save each, and assert the persisted
    initial_state is byte-identical (D1).

Asset-gated: skips if examples/scenes/h1_cup_table.nks is absent.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

nuka = pytest.importorskip("nuka")

REPO_ROOT = "/root/Nuka-Physics"
SCENE = os.path.join(REPO_ROOT, "examples", "scenes", "h1_cup_table.nks")

# The settle determinism check uses a SMALL generic articulation, NOT the H1
# union scene: the generic CookToModel single-articulation path (M7 named debt)
# is pathologically slow on the 51-DOF + ~367-loose-body union scene. The union
# scene's OWN cook is CookSceneToUnionTemplate (a different, fast bridge); the
# generic scene-authoring settle here is exercised on a scene the generic cook
# handles, which is the honest coverage for this API.
SETTLE_SCENE = os.path.join(REPO_ROOT, "examples", "scenes", "go2_stand.urdf")

pytestmark = pytest.mark.skipif(
    not os.path.exists(SCENE), reason="h1_cup_table.nks asset not present"
)

# The cup collision shape's DERIVED tree path in this scene (see the .nks tree).
CUP_SHAPE_PATH = "cup/body/cup/collision"


def _cup_friction_in_nks(nks_path: str):
    """Read the friction_mu of the cup collision shape from a saved .nks JSON
    (the on-disk persistence authority). Returns None if not found."""
    j = json.load(open(nks_path))
    found = []

    def walk(node, path=""):
        if isinstance(node, list):
            for n in node:
                walk(n, path)
            return
        if not isinstance(node, dict):
            return
        nm = node.get("name", "")
        p = (path + "/" + nm).strip("/") if nm else path
        cs = node.get("collision_shape")
        if cs is not None and p == CUP_SHAPE_PATH:
            found.append(cs.get("friction_mu"))
        walk(node.get("children", []), p)

    walk(j.get("tree", []))
    return found[0] if found else None


# ---------------------------------------------------------------------------
# host-only: load / find / edit / save / reload
# ---------------------------------------------------------------------------
def test_load_and_find():
    s = nuka.Scene.load(SCENE)
    try:
        assert s.find("cup/body") is True
        assert s.find("table/body") is True
        assert s.find("does/not/exist") is False
    finally:
        s.destroy()


def test_set_physics_material_persists():
    s = nuka.Scene.load(SCENE)
    try:
        n = s.set_physics_material("cup.*", static_friction=0.8)
        assert n >= 1, "regex 'cup.*' matched no collision shape"

        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "edited.nks")
            s.save(out)
            assert os.path.exists(out)
            assert os.path.exists(out[:-4] + ".nka"), "sibling .nka not written"

            # Reload and assert the node + material survived the round-trip.
            s2 = nuka.Scene.load(out)
            try:
                assert s2.find("cup/body") is True
            finally:
                s2.destroy()

            fmu = _cup_friction_in_nks(out)
            assert fmu is not None, "cup collision shape not found in saved .nks"
            assert abs(fmu - 0.8) < 1e-6, f"friction not persisted (got {fmu})"
    finally:
        s.destroy()


def test_set_physics_material_static_dynamic_mismatch_raises():
    s = nuka.Scene.load(SCENE)
    try:
        with pytest.raises(RuntimeError):
            # The record is isotropic; unequal static/dynamic must be rejected.
            s.set_physics_material("cup.*", static_friction=0.8, dynamic_friction=0.3)
    finally:
        s.destroy()


def test_set_local_unknown_path_raises():
    s = nuka.Scene.load(SCENE)
    try:
        with pytest.raises(RuntimeError):
            s.set_local("no/such/node", pos=(1.0, 2.0, 3.0))
        # A real node edits cleanly.
        s.set_local("cup/body", pos=(0.0, 0.0, 0.95))
    finally:
        s.destroy()


def test_compose_keeps_scene_consistent():
    base = nuka.Scene.load(SCENE)
    addon = nuka.Scene.load(SCENE)
    try:
        # Graft a copy at an offset with a name prefix; the base's own nodes must
        # still resolve afterwards (compose is in-place on base, addon untouched).
        base.compose(addon, pose=(0.0, 0.0, 0.0), attach_at="copy")
        assert base.find("cup/body") is True
        assert addon.find("cup/body") is True
    finally:
        base.destroy()
        addon.destroy()


# ---------------------------------------------------------------------------
# CUDA-gated: settle determinism (skip cleanly with no device / phi v2 backend)
# ---------------------------------------------------------------------------
def _try_device():
    try:
        return nuka.Device.create(0)
    except Exception:  # noqa: BLE001 -- no CUDA device available.
        return None


def test_settle_determinism():
    if not os.path.exists(SETTLE_SCENE):
        pytest.skip("settle scene asset not present")
    dev = _try_device()
    if dev is None:
        pytest.skip("no CUDA device")
    try:
        def settle_and_dump():
            s = nuka.Scene.load(SETTLE_SCENE)
            try:
                s.settle(dev, steps=8)
            except RuntimeError as e:
                pytest.skip(f"settle not supported on this build: {e}")
            with tempfile.TemporaryDirectory() as d:
                out = os.path.join(d, "settled.nks")
                s.save(out)
                blob = json.load(open(out))
            s.destroy()
            return json.dumps(blob.get("initial_state", {}), sort_keys=True)

        a = settle_and_dump()
        b = settle_and_dump()
        assert "qpos" in a, "settle wrote no articulation IC (initial_state)"
        assert a == b, "two settle runs produced different initial_state (D1)"
    finally:
        dev.close()
