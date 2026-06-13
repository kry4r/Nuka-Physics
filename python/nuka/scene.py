"""nuka.scene -- the GENERIC scene-authoring surface (M9 T4): a thin Python
wrapper over the ``_nuka_ext.Scene`` C-ABI surface.

ONE entry loads any supported format (mjcf / urdf / usd / nks); a small uniform
editing / settle / save surface lets you compose, transform, retexture physics
materials, settle to rest, and persist a ``.nks``. This is DELIBERATELY generic:
nothing here special-cases grasp / union / any demo. Behavior differences come
from the imported scene DATA plus a per-scene control SCRIPT (e.g. the
``h1_grasp_choreo`` choreography that drives the generic world), NEVER from a
special API (owner: unified world, no special grasp/union binding).

CONTRACT (plan L347-356)::

    s = nuka.Scene.load("examples/scenes/h1_cup_table.nks")
    s.find("cup/body")                              # -> True
    s.set_physics_material("cup.*", static_friction=0.8)
    s.set_local("cup/body", pos=(0.0, 0.0, 0.92))
    cup = nuka.Scene.load("assets/cup.usda")
    s.compose(cup, pose=(0.1, 0.0, 0.9), attach_at="extra")
    with nuka.Device.create(0) as dev:
        s.settle(dev, steps=200)                    # CUDA-only; cooks + settles
    s.save("out/edited.nks")                        # writes .nks + .nka

NODE ADDRESSING -- by STRING PATH. Every editing call takes a node's DERIVED
tree path ("h1/right_hand_link", "cup/body"). The C surface re-projects fresh
node/entity handles after every edit, so a path string (stable) is the contract,
not an opaque node handle.

QUATERNIONS are (w, x, y, z). Positions are (x, y, z) metres.

CUDA GATING. ``settle`` cooks the scene to an ``nk::World`` on the device's
backend; on a CUDA-less build (no phi v2 backend) it raises ``RuntimeError``
(NUKA_RESULT_NOT_SUPPORTED). Every other method is host-only.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ._nuka_ext import Scene as _ExtScene


def _as_floats(v: Optional[Sequence[float]]) -> list:
    """Coerce an optional (x,y,z) / (w,x,y,z) sequence to a float list; None/empty
    -> [] (the C side reads an empty list as 'leave this component unchanged')."""
    if v is None:
        return []
    return [float(x) for x in v]


class Scene:
    """A generic authored scene over ``_nuka_ext.Scene``."""

    def __init__(self, ext: "_ExtScene") -> None:
        # Prefer Scene.load(...); this ctor adopts an already-created ext handle.
        self._ext = ext

    @classmethod
    def load(cls, path: str) -> "Scene":
        """Load a scene from ``path`` (dispatch by extension: .nks / .xml / .mjcf
        / .urdf / .usd / .usda). ``.nks`` resolves its ``imports`` internally."""
        return cls(_ExtScene.load(str(path)))

    def compose(
        self,
        addon: "Scene",
        pose: Optional[Sequence[float]] = None,
        *,
        pos: Optional[Sequence[float]] = None,
        quat: Optional[Sequence[float]] = None,
        attach_at: str = "",
    ) -> "Scene":
        """Graft ``addon`` into this scene at a placement, returning ``self``.

        The placement is either ``pose`` (a 3-tuple position OR a 7-tuple
        [x,y,z, w,x,y,z]) or the explicit ``pos`` / ``quat`` keywords. ``quat``
        is (w,x,y,z). ``attach_at`` is an optional name prefix that disambiguates
        duplicate names in the merged scene. Mutates this scene in place.
        """
        p = pos
        q = quat
        if pose is not None:
            seq = [float(x) for x in pose]
            if len(seq) == 3:
                p = seq if p is None else p
            elif len(seq) == 7:
                if p is None:
                    p = seq[0:3]
                if q is None:
                    q = seq[3:7]
            else:
                raise ValueError(
                    "Scene.compose: pose must be 3 (xyz) or 7 (xyz+wxyz) floats"
                )
        self._ext.compose(addon._ext, _as_floats(p), _as_floats(q), str(attach_at))
        return self

    def find(self, path: str) -> bool:
        """True if a node at the derived tree path ``path`` exists."""
        return bool(self._ext.find(str(path)))

    def set_local(
        self,
        path: str,
        *,
        pos: Optional[Sequence[float]] = None,
        quat: Optional[Sequence[float]] = None,
    ) -> "Scene":
        """Set the LOCAL transform of the node at ``path`` (its body/shape record).

        ``pos`` is (x,y,z); ``quat`` is (w,x,y,z); a ``None`` leaves that
        component unchanged. Raises if no node matches ``path``. Returns ``self``.
        """
        self._ext.set_local(str(path), _as_floats(pos), _as_floats(quat))
        return self

    def set_physics_material(
        self,
        path_regex: str,
        *,
        static_friction: float = -1.0,
        dynamic_friction: float = -1.0,
        restitution: float = -1.0,
    ) -> int:
        """Set physics-material params on every collision-shape whose derived path
        matches ``path_regex`` (full-match). A param < 0 leaves it untouched; when
        both frictions are given they must be EQUAL (the record is isotropic).
        ``restitution`` is applied in-memory but does NOT persist to ``.nks``.
        Returns the number of shapes touched.
        """
        return int(
            self._ext.set_physics_material(
                str(path_regex),
                float(static_friction),
                float(dynamic_friction),
                float(restitution),
            )
        )

    def settle(self, device, steps: int = 200, dt: float = 0.0) -> "Scene":
        """Cook to an ``nk::World`` on ``device``'s backend, settle ``steps``
        deterministic steps (``dt`` <= 0 -> 1/240), and write the settled state
        back into the scene so ``save`` persists it. Raises ``RuntimeError`` on a
        CUDA-less build. Returns ``self``.
        """
        self._ext.settle(device, int(steps), float(dt))
        return self

    def save(self, nks_path: str) -> None:
        """Save the scene to ``nks_path`` (+ a sibling ``<base>.nka``)."""
        self._ext.save(str(nks_path))

    def destroy(self) -> None:
        self._ext.destroy()

    def __enter__(self) -> "Scene":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.destroy()


__all__ = ["Scene"]
