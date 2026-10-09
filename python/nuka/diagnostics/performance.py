"""Resource inventory and GPU completion windows for a production World."""

import json
import os
import time
from pathlib import Path

import numpy as np


class PerformanceCapture:
    def __init__(self, device, world, output, metadata):
        self.device = device
        self.world = world
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.manifest = {
            "schema_version": 1,
            "metadata": metadata,
            "module_loading": os.environ.get("CUDA_MODULE_LOADING"),
            "allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "world": world.resource_info(),
            "fields": world.resource_fields(),
            "resources": [],
            "windows": [],
            "timing_scope": "Backend stream events around the recorded production window; includes GPU completion.",
            "node_scope": "Top-level nodes in the captured physics step; excludes caller operations.",
        }
        self.write()

    def write(self):
        (self.output / "performance.json").write_text(json.dumps(self.manifest, indent=2) + "\n")

    def resources(self, label, runtime):
        row = {"label": label, "device": self.device.resource_info(), "runtime": runtime,
               "execution": self.world.execution_info}
        self.manifest["resources"].append(row)
        self.write()
        return row

    def prepare_window(self, name, state_fields):
        state = {field.name: np.asarray(self.world.download_field(field)).copy() for field in state_fields}
        np.savez(self.output / f"{name}_start.npz", **state)
        self.world.synchronize()
        self.window_name = name

    def begin_window(self):
        self.window = self.world.begin_execution_window()
        self.wall_start = time.perf_counter()

    def end_window(self, steps):
        row = self.world.end_execution_window(self.window)
        row.update(name=self.window_name, steps=steps,
                   wall_seconds=time.perf_counter() - self.wall_start,
                   start_state=f"{self.window_name}_start.npz")
        self.manifest["windows"].append(row)
        self.write()
        return row

    def fields(self, name, fields):
        state = {field.name: np.asarray(self.world.download_field(field)).copy() for field in fields}
        np.savez(self.output / f"{name}.npz", **state)
        return state
