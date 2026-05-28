"""Shared NUKAGOLD binary trajectory parsing helpers."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from pathlib import Path

import numpy as np


MAGIC = b"NUKAGOLD"
VERSION = 1
KIND_RANDOM_QACC = 1
KIND_JOINT_TRAJECTORY = 2


@dataclass(frozen=True)
class GoldenHeader:
    version: int
    kind: int
    samples: int
    qpos: int
    qvel: int
    qacc: int
    name: str

    @property
    def record_floats(self) -> int:
        if self.kind == KIND_RANDOM_QACC:
            return self.qpos + self.qvel + self.qvel + self.qacc
        if self.kind == KIND_JOINT_TRAJECTORY:
            return self.qpos + self.qvel + self.qacc
        return 0

    @property
    def kind_name(self) -> str:
        if self.kind == KIND_RANDOM_QACC:
            return "random-qacc"
        if self.kind == KIND_JOINT_TRAJECTORY:
            return "joint-trajectory"
        return f"unknown-{self.kind}"


def load_golden(path: Path) -> tuple[GoldenHeader, np.ndarray]:
    data = path.read_bytes()
    if len(data) < 36 or data[:8] != MAGIC:
        raise ValueError(f"{path}: invalid NUKAGOLD header")

    version, kind, samples, qpos, qvel, qacc, name_size = struct.unpack_from(
        "<IIIIIII",
        data,
        8,
    )
    if version != VERSION:
        raise ValueError(f"{path}: unsupported version {version}")
    if kind not in (KIND_RANDOM_QACC, KIND_JOINT_TRAJECTORY):
        raise ValueError(f"{path}: unsupported kind {kind}")
    if name_size > 4096:
        raise ValueError(f"{path}: model name is too large")

    name_start = 36
    payload_start = name_start + name_size
    if payload_start > len(data):
        raise ValueError(f"{path}: truncated model name")
    name = data[name_start:payload_start].decode("utf-8")

    header = GoldenHeader(
        version=version,
        kind=kind,
        samples=samples,
        qpos=qpos,
        qvel=qvel,
        qacc=qacc,
        name=name,
    )
    if samples == 0 or header.record_floats == 0:
        raise ValueError(f"{path}: empty golden shape")

    expected_bytes = payload_start + samples * header.record_floats * 4
    if expected_bytes != len(data):
        raise ValueError(
            f"{path}: size mismatch expected {expected_bytes} bytes, got {len(data)}"
        )

    payload = np.frombuffer(data[payload_start:], dtype="<f4").copy()
    return header, payload
