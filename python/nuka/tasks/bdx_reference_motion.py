"""Polynomial reference-motion adapter for the BDX biped (open_duck_mini_v2).

Loads the Open Duck Playground polynomial gait fit and turns it into an on-GPU
lookup: a per-command periodic joint/velocity/contact reference the imitation
reward tracks. The fit stores, per discrete velocity bin ``(vx, vy, wyaw)``, a set
of polynomials in the normalized period phase ``t in [0,1]``; evaluated on the
``nb_steps_in_period`` control-step grid it yields a 40-D reference frame:

    [ 0:16] joints_pos   (5 left leg | 6 head/antennas | 5 right leg)
    [16:32] joints_vel   (same order)
    [32:34] foot_contacts (left, right; continuous, thresholded at 0.5)
    [34:37] base_lin_vel  (body frame at the +x reference heading)
    [37:40] base_ang_vel

The 10 LEG joints are ``[:5] + [11:]`` -- the head/antenna block is dropped, which
lines up 1:1 with our policy leg order. All bins x steps are baked once into a
single CUDA tensor; per env we keep only a flat bin index and a phase counter, so
a step is a gather. The velocity grid is coarse, so a command is snapped to its
nearest bin (matching the Playground ``vel_to_index``).
"""

from __future__ import annotations

import pickle

import numpy as np
import torch

# Reference-frame slice layout (40-D polynomial output).
JOINT_POS = slice(0, 16)
JOINT_VEL = slice(16, 32)
FOOT_CONTACTS = slice(32, 34)
BASE_LIN_VEL = slice(34, 37)
BASE_ANG_VEL = slice(37, 40)
# The 10 leg joints inside the 16-joint block (drop the 6-joint head/antenna span).
LEG_IN_16 = list(range(0, 5)) + list(range(11, 16))


class BdxReferenceMotion:
    """On-GPU periodic gait reference, indexed by nearest velocity bin + phase."""

    def __init__(self, pkl_path: str, device, *, dtype=torch.float32):
        self._dev = device
        self.dtype = dtype
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        # Discrete velocity grid + period metadata (identical across bins).
        dxs, dys, dthetas = set(), set(), set()
        any_entry = next(iter(data.values()))
        self.nb_steps = int(any_entry["nb_steps_in_period"])
        self.period = float(any_entry["period"])
        self.fps = float(any_entry["fps"])
        for name in data:
            dx, dy, dth = (float(v) for v in name.split("_"))
            dxs.add(dx); dys.add(dy); dthetas.add(dth)
        sorted_dx, sorted_dy, sorted_dth = sorted(dxs), sorted(dys), sorted(dthetas)
        self.dxs = np.array(sorted_dx, np.float32)
        self.dys = np.array(sorted_dy, np.float32)
        self.dthetas = np.array(sorted_dth, np.float32)
        n_dx, n_dy, n_dth = len(self.dxs), len(self.dys), len(self.dthetas)
        self.dx_range = (float(self.dxs[0]), float(self.dxs[-1]))
        self.dy_range = (float(self.dys[0]), float(self.dys[-1]))
        self.dtheta_range = (float(self.dthetas[0]), float(self.dthetas[-1]))

        # Bake every bin x step into one table: (n_bins, nb_steps, D_ref).
        i2x = {v: i for i, v in enumerate(sorted_dx)}
        i2y = {v: i for i, v in enumerate(sorted_dy)}
        i2t = {v: i for i, v in enumerate(sorted_dth)}
        d_ref = None
        table = None
        ts = (np.arange(self.nb_steps, dtype=np.float64) / self.nb_steps)
        for name, entry in data.items():
            dx, dy, dth = (float(v) for v in name.split("_"))
            coeffs = list(entry["coefficients"].values())  # per-dim ascending coeffs
            if table is None:
                d_ref = len(coeffs)
                table = np.zeros((n_dx * n_dy * n_dth, self.nb_steps, d_ref), np.float32)
            frame = np.stack(
                [np.polyval(np.flip(np.asarray(c, np.float64)), ts) for c in coeffs],
                axis=1,
            )  # (nb_steps, d_ref)
            flat = (i2x[dx] * n_dy + i2y[dy]) * n_dth + i2t[dth]
            table[flat] = frame.astype(np.float32)

        self.d_ref = int(d_ref)
        self.table = torch.as_tensor(table, dtype=dtype, device=device)  # (B,T,D)
        self._dxs_t = torch.as_tensor(self.dxs, dtype=dtype, device=device)
        self._dys_t = torch.as_tensor(self.dys, dtype=dtype, device=device)
        self._dths_t = torch.as_tensor(self.dthetas, dtype=dtype, device=device)
        self._n_dy = n_dy
        self._n_dth = n_dth
        # Precomputed phase features for every step index (nb_steps, 2) = (cos, sin).
        ph = (torch.arange(self.nb_steps, device=device, dtype=dtype)
              / self.nb_steps) * (2.0 * np.pi)
        self._phase_lut = torch.stack((torch.cos(ph), torch.sin(ph)), dim=1)

    def command_to_bin(self, cmd: torch.Tensor) -> torch.Tensor:
        """(N,3) velocity command -> (N,) flat bin index of the nearest grid cell."""
        vx = cmd[:, 0].clamp(*self.dx_range)
        vy = cmd[:, 1].clamp(*self.dy_range)
        vt = cmd[:, 2].clamp(*self.dtheta_range)
        ix = (vx.unsqueeze(1) - self._dxs_t).abs().argmin(dim=1)
        iy = (vy.unsqueeze(1) - self._dys_t).abs().argmin(dim=1)
        it = (vt.unsqueeze(1) - self._dths_t).abs().argmin(dim=1)
        return (ix * self._n_dy + iy) * self._n_dth + it

    def phase_features(self, phase_i: torch.Tensor) -> torch.Tensor:
        """(N,) integer phase counter -> (N,2) [cos, sin] of the normalized phase."""
        return self._phase_lut[phase_i]

    def gather(self, bin_idx: torch.Tensor, phase_i: torch.Tensor) -> torch.Tensor:
        """(N,), (N,) -> (N,D_ref) reference frame at each env's bin and phase."""
        return self.table[bin_idx, phase_i]

    # -- named slices of a gathered (N,D_ref) frame ---------------------------
    def ref_leg_pos(self, frame: torch.Tensor) -> torch.Tensor:
        return frame[:, JOINT_POS][:, LEG_IN_16]

    def ref_leg_vel(self, frame: torch.Tensor) -> torch.Tensor:
        return frame[:, JOINT_VEL][:, LEG_IN_16]

    def ref_foot_contacts(self, frame: torch.Tensor) -> torch.Tensor:
        return (frame[:, FOOT_CONTACTS] > 0.5).to(self.dtype)

    def ref_base_lin_vel(self, frame: torch.Tensor) -> torch.Tensor:
        return frame[:, BASE_LIN_VEL]

    def ref_base_ang_vel(self, frame: torch.Tensor) -> torch.Tensor:
        return frame[:, BASE_ANG_VEL]
