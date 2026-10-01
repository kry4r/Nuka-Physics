"""Summarize and plot production interval energy records from saved traces."""

import argparse
import json
from pathlib import Path

import numpy as np

from nuka.energy import EnergyColumn as C, summarize


def load_trace(path, env_count):
    if path.suffix == ".npz":
        with np.load(path) as trace:
            values = trace["energy"].copy()
            flags = trace["energy_status"].copy()
        if values.ndim != 4:
            raise ValueError("NPZ energy must have policy/env/substep/column axes")
        return values, flags, {}
    payload = json.loads(path.read_text())
    rows = payload["rows"]
    values = np.asarray([row["ENERGY_LEDGER"] for row in rows], dtype=np.float32)
    flags = np.asarray([row["ENERGY_LEDGER_STATUS"] for row in rows], dtype=np.uint32)
    values = values.reshape(len(rows), env_count, -1, len(C))
    flags = flags.reshape(values.shape[:-1])
    return values, flags, payload.get("parameters", {})


def events(records, count=10):
    records = records.reshape(-1, len(C)).astype(np.float64)
    finite = np.flatnonzero(np.isfinite(records[:, C.RESIDUAL]))
    indices = finite[np.argsort(np.abs(records[finite, C.RESIDUAL]))[-count:][::-1]]
    time = np.cumsum(records[:, C.DT])
    return [{"interval": int(index + 1), "time_s": float(time[index]),
             "terms": {column.name: float(records[index, column]) for column in C}}
            for index in indices]


def plot_records(records, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = records.reshape(-1, len(C)).astype(np.float64)
    time = np.cumsum(records[:, C.DT])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    ax = axes[0, 0]
    for column, label in [(C.END_KINETIC, "Kinetic"), (C.END_GRAVITY, "Gravity"),
                          (C.END_ELASTIC, "Elastic")]:
        ax.plot(time, records[:, column], label=label)
    ax.set(title="Completed physical state", ylabel="J")
    ax = axes[0, 1]
    for column, label in [(C.RESIDUAL, "Unclassified residual"), (C.DAT_KINETIC_LOSS, "DAT kinetic loss"),
                          (C.POSITION_POTENTIAL, "Position potential change")]:
        ax.plot(time, records[:, column] * 1000, label=label)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set(title="Per-interval energy balance", ylabel="mJ")
    ax = axes[1, 0]
    boundary = records[:, C.KINEMATIC_ELASTIC_WORK] + records[:, C.KINEMATIC_CONTACT_WORK]
    ax.plot(time, np.cumsum(boundary), label="Kinematic boundary work")
    for column, label in [(C.DRIVE_WORK, "Drive work"), (C.NORMAL_LOSS, "Normal loss"),
                          (C.FRICTION_LOSS, "Friction loss"), (C.RAYLEIGH_LOSS, "Rayleigh loss")]:
        ax.plot(time, np.cumsum(records[:, column]), label=label)
    ax.set(title="Accumulated actual work and dissipation", ylabel="J")
    ax = axes[1, 1]
    ax.plot(time, records[:, C.ROW_RATE_WORK] * 1000, label="Row displacement-rate work")
    ax.plot(time, records[:, C.ROW_PHYSICAL_IMPULSE_WORK] * 1000, label="Physical applied impulse work")
    ax.set(title="Row variable and physical impulse", ylabel="mJ")
    for ax in axes.flat:
        ax.set_xlabel("Physical time (s)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-count", type=int, default=1)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    values, flags, parameters = load_trace(args.trace, args.env_count)
    args.output.mkdir(parents=True, exist_ok=False)
    report = summarize(values, flags)
    report.update({"source": str(args.trace), "parameters": parameters,
                   "axes": ["policy_step", "environment", "substep", "energy_column"],
                   "shape": list(values.shape),
                   "largest_residual_events": [events(values[:, env]) for env in range(values.shape[1])],
                   "scope": "Energy gates only; CCD, capacity, convergence and render acceptance are separate."})
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(args.output / "records.npz", energy=values, energy_status=flags,
                        columns=np.asarray([column.name for column in C]))
    if args.plot:
        for env in range(values.shape[1]):
            plot_records(values[:, env], args.output / f"energy_env{env}.png")
    print(json.dumps({"output": str(args.output), "passes_energy_gates": report["passes_energy_gates"],
                      "shape": report["shape"]}))


if __name__ == "__main__":
    main()
