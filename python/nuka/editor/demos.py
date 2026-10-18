"""nuka.editor.demos -- the editor's demonstrations as robomimic-style HDF5.

The ``demo.save`` action writes, and ``demo.load`` reads, files laid out as::

    data                  attrs: total, env_args (JSON: scene, dt, joints, links, axes)
    data/demo_<k>         attrs: num_samples, env, start_tick
        actions           (T, J)     the drive targets each step ran with
        obs/q, obs/qd     (T, J)     joint positions and velocities after the step
        obs/root_pose     (T, R, 7)  each root link's x, y, z, qw, qx, qy, qz
        obs/root_vel      (T, R, 6)  its body-frame wx, wy, wz, vx, vy, vz
        obs/command       (T, 3)     the controller's vx, vy, wyaw
        obs/axes          (T, A)     the teleop device's axes
        obs/time          (T,)       the simulated time after the step

``read`` returns a file's demonstrations as numpy arrays. Both need h5py (``pip install h5py``).
"""

from __future__ import annotations

import json

import numpy as np

_HEADER = ("scene", "dt", "joints", "links", "axes")
_OBS = ("q", "qd", "root_pose", "root_vel", "command", "axes", "time")


def read(path: str) -> dict:
    """The file's env_args with ``demos``, a list of {name, env, start_tick, actions, obs}."""
    import h5py

    with h5py.File(path, "r") as h5:
        group = h5["data"]
        out = json.loads(group.attrs["env_args"])
        out["demos"] = []
        for name in sorted(group, key=lambda n: int(n.rpartition("_")[2])):
            demo = group[name]
            out["demos"].append({"name": name, "env": int(demo.attrs["env"]),
                                 "start_tick": int(demo.attrs["start_tick"]), "actions": demo["actions"][()],
                                 "obs": {key: demo["obs"][key][()] for key in _OBS}})
    return out


# The editor's side: an episode's arrays follow one another in the data file, time as float64
# and the rest float32, at the offset its manifest entry gives.
def _layout(e: dict, joints: int, axes: int) -> list:
    steps, roots = e["steps"], e["roots"]
    return [("time", "<f8", (steps,)), ("q", "<f4", (steps, joints)), ("qd", "<f4", (steps, joints)),
            ("actions", "<f4", (steps, joints)), ("root_pose", "<f4", (steps, roots, 7)),
            ("root_vel", "<f4", (steps, roots, 6)), ("command", "<f4", (steps, 3)), ("axes", "<f4", (steps, axes))]


def _save(path: str, manifest: str, data: str) -> None:
    import h5py

    with open(manifest, encoding="utf-8") as f:
        m = json.load(f)
    with open(data, "rb") as f:
        blob = f.read()
    with h5py.File(path, "w") as h5:
        group = h5.create_group("data")
        group.attrs["total"] = sum(e["steps"] for e in m["episodes"])
        group.attrs["env_args"] = json.dumps({key: m[key] for key in _HEADER})
        for k, e in enumerate(m["episodes"]):
            demo = group.create_group(f"demo_{k}")
            demo.attrs["num_samples"] = e["steps"]
            demo.attrs["env"] = e["env"]
            demo.attrs["start_tick"] = e["start"]
            obs = demo.create_group("obs")
            offset = e["offset"]
            for name, dtype, shape in _layout(e, len(m["joints"]), len(m["axes"])):
                count = int(np.prod(shape))
                array = np.frombuffer(blob, dtype=dtype, count=count, offset=offset).reshape(shape)
                offset += array.nbytes
                (demo if name == "actions" else obs).create_dataset(name, data=array)


def _load(path: str, manifest: str, data: str) -> None:
    demos = read(path)
    episodes = []
    with open(data, "wb") as f:
        for d in demos["demos"]:
            e = {"env": d["env"], "start": d["start_tick"], "steps": len(d["obs"]["time"]),
                 "roots": d["obs"]["root_pose"].shape[1], "offset": f.tell()}
            episodes.append(e)
            for name, dtype, shape in _layout(e, len(demos["joints"]), len(demos["axes"])):
                array = d["actions"] if name == "actions" else d["obs"][name]
                f.write(np.ascontiguousarray(array, dtype=dtype).reshape(shape).tobytes())
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump({**{key: demos[key] for key in _HEADER}, "episodes": episodes}, f)
