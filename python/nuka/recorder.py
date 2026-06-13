"""nuka.recorder -- the offscreen RECORDER (M8 T5): a thin Python wrapper over
the ``_nuka_ext.Recorder`` C-ABI surface.

The recorder drives the M8 host frame loop over an ``nk::World`` cooked from an
authored ``.nks`` scene, through the offscreen Vulkan forward raster renderer,
captures each frame to a binary PPM P6, then muxes an mp4 via ffmpeg.

CONTRACT (plan L353)::

    rec = Recorder(device, scene_path, camera=...)   # or Recorder(device, world=...)
    rec.capture("out/frames", n_frames=120)
    rec.to_video("out/run.mp4", fps=30)

The heavy state (the cooked ``nk::World`` + ``RenderWorld`` + ``Simulation`` +
the Vulkan renderer) lives entirely inside the C record; this class is a
RAII-friendly handle over it.

VULKAN GATING (Decision D5). When ``libnuka`` was built WITHOUT Vulkan the
underlying C entry points return ``NUKA_RESULT_NOT_SUPPORTED`` and construction
raises ``RuntimeError`` -- ``import nuka.recorder`` still works, only the runtime
use fails. Likewise ``to_video`` raises cleanly when ffmpeg is missing (Risk R6).
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

from ._nuka_ext import Recorder as _ExtRecorder

# The default H1 union scene the M8 deliverable renders end-to-end (Decision D2:
# the nk-backed union world). Repo-relative -> run from the repo root.
DEFAULT_SCENE = "examples/scenes/h1_cup_table.nks"


class Camera:
    """A pinned camera shot (eye / target / up + vertical FOV in degrees).

    Passing a Camera to ``Recorder`` is informational/forward-looking: the M8 C
    surface auto-frames the scene AABB (or uses the authored scene camera) so a
    frame is always visible. An explicit-camera C path is a follow-on; for now a
    Camera records the intended shot for callers/tooling.
    """

    def __init__(
        self,
        eye: Sequence[float] = (2.5, -2.5, 1.6),
        target: Sequence[float] = (0.0, 0.0, 0.9),
        up: Sequence[float] = (0.0, 0.0, 1.0),
        fov_degrees: float = 45.0,
    ) -> None:
        self.eye = tuple(float(v) for v in eye)
        self.target = tuple(float(v) for v in target)
        self.up = tuple(float(v) for v in up)
        self.fov_degrees = float(fov_degrees)


class Recorder:
    """Offscreen recorder over an nk::World cooked from a ``.nks`` scene."""

    def __init__(
        self,
        device,
        scene_path: str = DEFAULT_SCENE,
        *,
        camera: Optional[Camera] = None,
        width: int = 0,
        height: int = 0,
        env_index: int = 0,
        dt: float = 0.0,
    ) -> None:
        """Create a recorder.

        Args:
            device: a ``nuka.Device`` (the CUDA device the world cooks onto).
            scene_path: the authored ``.nks`` to cook + render (default H1 scene).
            camera: an optional :class:`Camera` (records the intended shot).
            width/height: framebuffer size; 0 -> 1920x1080 (Decision D6).
            env_index: which env to render (Decision D4; default 0).
            dt: fixed step dt; <= 0 -> 1/240.
        """
        self._camera = camera
        self._scene_path = scene_path
        self._ext = _ExtRecorder.create(
            device,
            scene_path,
            width=int(width),
            height=int(height),
            env_index=int(env_index),
            dt=float(dt),
        )

    def capture(self, out_dir: str, n_frames: int) -> int:
        """Drive ``n_frames`` frames, writing ``out_dir/frame_%06d.ppm`` each.

        The frame index continues across ``capture`` calls so a multi-segment
        rollout muxes contiguously. Returns the total frame count so far.
        """
        self._ext.capture(int(n_frames), out_dir)
        return self.frame_count

    def to_video(self, out_mp4: str, out_dir: str = "", fps: int = 30) -> None:
        """Mux the captured PPM sequence into ``out_mp4`` via ffmpeg at ``fps``.

        ``out_dir`` must be the directory the frames were captured into. Raises
        cleanly if ffmpeg is absent or the mux fails (Risk R6).
        """
        if not out_dir:
            raise ValueError(
                "Recorder.to_video: out_dir is required (the directory the "
                "frames were captured into)"
            )
        self._ext.to_video(out_dir, out_mp4, int(fps))

    @property
    def frame_count(self) -> int:
        return int(self._ext.frame_count)

    @property
    def camera(self) -> Optional[Camera]:
        return self._camera

    @property
    def scene_path(self) -> str:
        return self._scene_path

    def destroy(self) -> None:
        self._ext.destroy()

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.destroy()


__all__ = ["Recorder", "Camera", "DEFAULT_SCENE"]
