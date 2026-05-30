"""Minimal self-contained ctypes binding for the Nuka C ABI.

Mirrors the public surface declared in ``src/include/nuka/nuka.h``. The only
job of this module is to let a Python process:

  * dlopen ``libnuka.so`` and query the version,
  * create a CUDA device + a batched (multi-env) world from a USDA scene,
  * step the world,
  * obtain ``nuka_buffer_view_t`` views (CUDA *device* pointers) for the
    joint position / velocity / link-pose / drive-target fields,
  * move bytes host<->device via ``cudaMemcpy`` (we reuse the *same*
    ``libcudart.so.12`` that ``libnuka.so`` itself links against so there is a
    single CUDA runtime / primary context in play).

Nothing here is engine-specific cleverness; it is a faithful, dependency-light
(ctypes-only) mirror of the header. See ``g1h1_drive_harness.py`` for usage and
``README.md`` for the buffer-layout / quaternion-order gotchas.
"""

from __future__ import annotations

import ctypes
import os
from typing import Optional

import numpy as np

# --- Default install locations (override via env/args) ----------------------
DEFAULT_LIBNUKA = "/root/Nuka-Physics/build-cuda128/src/libnuka.so"
DEFAULT_LIBCUDART = (
    "/opt/cuda-12.8-root/usr/local/cuda-12.8/lib64/libcudart.so.12"
)

# --- Enums (mirror nuka.h) --------------------------------------------------
NUKA_RESULT_OK = 0

# nuka_state_field_t
NUKA_FIELD_RIGID_BODY_TRANSFORM = 0
NUKA_FIELD_ARTICULATION_LINK_POSE = 1
NUKA_FIELD_JOINT_POSITION = 2
NUKA_FIELD_JOINT_VELOCITY = 3
NUKA_FIELD_OBSERVATIONS = 4
NUKA_FIELD_CONTACT_POINTS = 5
NUKA_FIELD_DRIVE_TARGET = 6

# cudaMemcpyKind
CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2

# Go2 articulation: base_link_count == 13 (root + 12 actuated leg joints).
GO2_BASE_LINK_COUNT = 13
# ARTICULATION_LINK_POSE element = math::Transform = 7 floats
# [px,py,pz, qw,qx,qy,qz]  (quaternion is w-FIRST).
LINK_POSE_FLOATS = 7


# --- Structs (mirror nuka.h) ------------------------------------------------
class NukaVersion(ctypes.Structure):
    _fields_ = [
        ("major", ctypes.c_uint16),
        ("minor", ctypes.c_uint16),
        ("patch", ctypes.c_uint16),
    ]


class NukaDeviceDesc(ctypes.Structure):
    _fields_ = [
        ("gpu_index", ctypes.c_uint32),
        ("cuda_stream", ctypes.c_void_p),
        ("backend_selection_layer_enabled", ctypes.c_uint8),
    ]


class NukaWorldDesc(ctypes.Structure):
    _fields_ = [
        ("scene_path", ctypes.c_char_p),
        ("env_count", ctypes.c_uint32),
        ("fixed_dt", ctypes.c_float),
    ]


class NukaBufferView(ctypes.Structure):
    _fields_ = [
        ("device_ptr", ctypes.c_void_p),
        ("element_count", ctypes.c_size_t),
        ("element_stride_bytes", ctypes.c_uint32),
        ("dtype", ctypes.c_uint8),
    ]


class NukaError(RuntimeError):
    """Raised when a C ABI call returns a non-OK result."""


class NukaABI:
    """Thin ctypes wrapper over libnuka.so + the matching libcudart."""

    def __init__(
        self,
        libnuka_path: str = DEFAULT_LIBNUKA,
        libcudart_path: str = DEFAULT_LIBCUDART,
    ) -> None:
        if not os.path.exists(libnuka_path):
            raise FileNotFoundError(libnuka_path)
        # Reuse the engine's own CUDA runtime so a single primary context backs
        # both the engine allocations and our memcpy.
        self.cudart = ctypes.CDLL(libcudart_path, mode=ctypes.RTLD_GLOBAL)
        self.lib = ctypes.CDLL(libnuka_path)
        self._bind()

    def _bind(self) -> None:
        lib = self.lib

        lib.nuka_get_version.restype = NukaVersion
        lib.nuka_get_version.argtypes = []

        lib.nuka_result_message.restype = ctypes.c_char_p
        lib.nuka_result_message.argtypes = [ctypes.c_int]

        lib.nuka_device_create.restype = ctypes.c_int
        lib.nuka_device_create.argtypes = [
            ctypes.POINTER(NukaDeviceDesc),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.nuka_device_destroy.restype = None
        lib.nuka_device_destroy.argtypes = [ctypes.c_void_p]

        lib.nuka_world_create_from_scene.restype = ctypes.c_int
        lib.nuka_world_create_from_scene.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(NukaWorldDesc),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.nuka_world_destroy.restype = None
        lib.nuka_world_destroy.argtypes = [ctypes.c_void_p]

        lib.nuka_world_step.restype = ctypes.c_int
        lib.nuka_world_step.argtypes = [ctypes.c_void_p]
        lib.nuka_world_step_n.restype = ctypes.c_int
        lib.nuka_world_step_n.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

        lib.nuka_world_get_buffer_view.restype = ctypes.c_int
        lib.nuka_world_get_buffer_view.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(NukaBufferView),
        ]

        # cudaMemcpy(void* dst, const void* src, size_t count, int kind)
        self.cudart.cudaMemcpy.restype = ctypes.c_int
        self.cudart.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.cudart.cudaDeviceSynchronize.restype = ctypes.c_int
        self.cudart.cudaDeviceSynchronize.argtypes = []
        self.cudart.cudaGetErrorString.restype = ctypes.c_char_p
        self.cudart.cudaGetErrorString.argtypes = [ctypes.c_int]

    # --- result helper ------------------------------------------------------
    def _check(self, result: int, what: str) -> None:
        if result != NUKA_RESULT_OK:
            msg = self.lib.nuka_result_message(result)
            msg = msg.decode() if msg else "?"
            raise NukaError(f"{what} failed: {msg} ({result})")

    def _cuda_check(self, result: int, what: str) -> None:
        if result != 0:
            msg = self.cudart.cudaGetErrorString(result)
            msg = msg.decode() if msg else "?"
            raise NukaError(f"{what} failed: {msg} ({result})")

    # --- version ------------------------------------------------------------
    def version(self) -> NukaVersion:
        return self.lib.nuka_get_version()

    # --- device / world lifecycle ------------------------------------------
    def device_create(self, gpu_index: int = 0) -> ctypes.c_void_p:
        desc = NukaDeviceDesc(
            gpu_index=gpu_index,
            cuda_stream=None,
            backend_selection_layer_enabled=1,
        )
        handle = ctypes.c_void_p()
        self._check(
            self.lib.nuka_device_create(ctypes.byref(desc), ctypes.byref(handle)),
            "nuka_device_create",
        )
        return handle

    def device_destroy(self, device: ctypes.c_void_p) -> None:
        self.lib.nuka_device_destroy(device)

    def world_create(
        self,
        device: ctypes.c_void_p,
        scene_path: str,
        env_count: int,
        fixed_dt: float = 1.0 / 240.0,
    ) -> ctypes.c_void_p:
        desc = NukaWorldDesc(
            scene_path=scene_path.encode(),
            env_count=env_count,
            fixed_dt=fixed_dt,
        )
        handle = ctypes.c_void_p()
        self._check(
            self.lib.nuka_world_create_from_scene(
                device, ctypes.byref(desc), ctypes.byref(handle)
            ),
            "nuka_world_create_from_scene",
        )
        return handle

    def world_destroy(self, world: ctypes.c_void_p) -> None:
        self.lib.nuka_world_destroy(world)

    def step(self, world: ctypes.c_void_p, n: int = 1) -> None:
        if n == 1:
            self._check(self.lib.nuka_world_step(world), "nuka_world_step")
        else:
            self._check(
                self.lib.nuka_world_step_n(world, n), "nuka_world_step_n"
            )

    # --- buffer views -------------------------------------------------------
    def buffer_view(
        self, world: ctypes.c_void_p, field: int
    ) -> NukaBufferView:
        view = NukaBufferView()
        self._check(
            self.lib.nuka_world_get_buffer_view(world, field, ctypes.byref(view)),
            f"nuka_world_get_buffer_view(field={field})",
        )
        if not view.device_ptr:
            raise NukaError(f"buffer_view(field={field}) returned null device_ptr")
        return view

    # --- device <-> host transfer (float buffers) ---------------------------
    def read_floats(self, view: NukaBufferView, n_floats: int) -> np.ndarray:
        """Copy ``n_floats`` float32 from the device buffer to a host array."""
        out = np.empty(n_floats, dtype=np.float32)
        self._cuda_check(
            self.cudart.cudaMemcpy(
                out.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_void_p(view.device_ptr),
                n_floats * 4,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            ),
            "cudaMemcpy D2H",
        )
        return out

    def write_floats(self, view: NukaBufferView, host: np.ndarray) -> None:
        """Copy a host float32 array into the device buffer (in place)."""
        host = np.ascontiguousarray(host, dtype=np.float32)
        self._cuda_check(
            self.cudart.cudaMemcpy(
                ctypes.c_void_p(view.device_ptr),
                host.ctypes.data_as(ctypes.c_void_p),
                host.size * 4,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            ),
            "cudaMemcpy H2D",
        )

    def sync(self) -> None:
        self._cuda_check(self.cudart.cudaDeviceSynchronize(), "cudaDeviceSynchronize")
