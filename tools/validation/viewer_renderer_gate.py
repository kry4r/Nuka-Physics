#!/usr/bin/env python3
"""Run renderer smoke tests with mandatory Vulkan synchronization validation."""

import argparse
import ctypes
import ctypes.util
import json
import os
from pathlib import Path
import subprocess
import sys


class LayerProperties(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 256),
        ("spec_version", ctypes.c_uint32),
        ("implementation_version", ctypes.c_uint32),
        ("description", ctypes.c_char * 256),
    ]


def validation_available():
    library = ctypes.util.find_library("vulkan")
    loader = ctypes.CDLL(library or ("vulkan-1.dll" if os.name == "nt" else "libvulkan.so.1"))
    enumerate_layers = loader.vkEnumerateInstanceLayerProperties
    enumerate_layers.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(LayerProperties)]
    enumerate_layers.restype = ctypes.c_int32
    count = ctypes.c_uint32()
    if enumerate_layers(ctypes.byref(count), None) != 0:
        return False
    layers = (LayerProperties * count.value)()
    if enumerate_layers(ctypes.byref(count), layers) != 0:
        return False
    return any(layer.name == b"VK_LAYER_KHRONOS_validation" for layer in layers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(".nuka-runs/viewer-validation"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        if not validation_available():
            raise RuntimeError("VK_LAYER_KHRONOS_validation is not available")
    except (OSError, RuntimeError) as error:
        print(f"Renderer validation unavailable: {error}", file=sys.stderr)
        return 2

    env = os.environ.copy()
    layers = env.get("VK_INSTANCE_LAYERS", "").split(os.pathsep)
    if "VK_LAYER_KHRONOS_validation" not in layers:
        layers.append("VK_LAYER_KHRONOS_validation")
    env["VK_INSTANCE_LAYERS"] = os.pathsep.join(layer for layer in layers if layer)
    env["VK_LAYER_ENABLES"] = "VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT"
    names = ["nuka_render_raster_smoke_test", "nuka_viewer_frame_smoke_test",
             "nuka_viewport_present_smoke_test"]
    results = []
    for name in names:
        binary = args.build_dir.resolve() / "tests" / (name + (".exe" if os.name == "nt" else ""))
        try:
            run = subprocess.run([str(binary)], cwd=root, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, timeout=120, check=False)
            text = run.stdout
            code = run.returncode
        except (OSError, subprocess.TimeoutExpired) as error:
            text, code = str(error), -1
        (output / f"{name}.log").write_text(text, encoding="utf-8")
        errors = text.count("Validation Error:")
        skipped = "[  SKIPPED ]" in text
        failed = (code != 0 or errors != 0 or skipped or "SYNC-HAZARD" in text
                  or "[  FAILED  ]" in text or "[  PASSED  ]" not in text)
        result = dict(test=name, exit_code=code, validation_errors=errors,
                      skipped=skipped, passed=not failed)
        results.append(result)
        print(json.dumps(result))
    (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
