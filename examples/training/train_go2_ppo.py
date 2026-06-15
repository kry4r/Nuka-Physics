#!/usr/bin/env python
"""Launch PPO training for the Nuka Go2 locomotion task via rl_games (1.6.5).

Loads ``go2_ppo_cfg.yaml``, registers the on-GPU Nuka env with rl_games (by
importing :mod:`nuka.rl_games`), applies CLI overrides, and runs
``rl_games.torch_runner.Runner().run({'train': True, ...})``.

The SAME script drives both modes:
  * default          -> the real config (num_actors 4096, max_epochs 1500) -- p04.
  * ``--smoke``      -> tiny launch proof: 3 epochs @ 256 actors, single minibatch,
                        short horizon. Used for the p03 EXIT gate ("training
                        launches and begins"), NOT convergence.

Single GPU only -- always run with ``CUDA_VISIBLE_DEVICES=0``::

    CUDA_VISIBLE_DEVICES=0 python examples/training/train_go2_ppo.py --smoke
    CUDA_VISIBLE_DEVICES=0 python examples/training/train_go2_ppo.py            # full (p04)

CLI overrides (all optional): --num_actors --max_epochs --horizon_length
--minibatch_size --mini_epochs --seed --command vx vy wyaw --experiment-name.
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

from rl_games.common.algo_observer import DefaultAlgoObserver


class _StdoutMetricsWriter:
    """Delegating shim around rl_games' SummaryWriter that ALSO prints the
    train metrics rl_games logs (losses/*, rewards/*, episode_lengths/*) to
    stdout, so a headless launch shows verbatim per-epoch reward/loss numbers
    (the p03 gate). Everything else passes straight through to the real writer.
    """

    _PRINT_PREFIXES = ("losses/", "rewards/", "episode_lengths/", "info/")

    def __init__(self, writer):
        self._writer = writer
        self._row: dict = {}

    def add_scalar(self, tag, value, *args, **kwargs):
        if self._writer is not None:
            self._writer.add_scalar(tag, value, *args, **kwargs)
        if tag.startswith(self._PRINT_PREFIXES):
            try:
                v = float(value)
            except (TypeError, ValueError):
                return
            self._row[tag] = v

    # Distinct labels so the verbatim tail is unambiguous (several rl_games tags
    # share a leaf name, e.g. rewards/step vs episode_lengths/step both -> step).
    _LABELS = {
        "rewards/step": "reward",
        "episode_lengths/step": "ep_len",
        "losses/a_loss": "a_loss",
        "losses/c_loss": "c_loss",
        "losses/entropy": "entropy",
        "losses/bounds_loss": "bounds_loss",
        "info/kl": "kl",
        "info/last_lr": "lr",
        "info/e_clip": "e_clip",
    }

    def flush_row(self, epoch_num, frame):
        if not self._row:
            return
        parts = []
        for tag, label in self._LABELS.items():
            if tag in self._row:
                parts.append(f"{label}={self._row[tag]:.4f}")
        print(f"[metrics] epoch {epoch_num} frames {frame}: " + "  ".join(parts),
              flush=True)
        self._row = {}

    def __getattr__(self, name):
        return getattr(self._writer, name)


# Always-on terrain-curriculum diagnostic cadence (epochs). Cheap per-call (a few
# reductions over (num_envs,) tensors) so a 50-epoch period is negligible.
_TERRAIN_DIAG_EVERY = int(os.environ.get("NUKA_GO2_TERRAIN_DIAG_EVERY", "50"))


class StdoutAlgoObserver(DefaultAlgoObserver):
    """DefaultAlgoObserver + per-epoch stdout dump of the metrics rl_games logs +
    an ALWAYS-ON periodic terrain-curriculum difficulty report.

    Wraps ``algo.writer`` with :class:`_StdoutMetricsWriter` so we capture
    EXACTLY what rl_games emits (version-stable; ``a_loss``/``c_loss`` are
    local to ``train_epoch`` and only ever reach the writer). The reward/length
    scalars are guarded by ``game_rewards.current_size>0`` inside rl_games, so
    they appear once episodes complete; losses appear every epoch.

    When terrain is ON it ALSO prints, every ``_TERRAIN_DIAG_EVERY`` epochs, the
    per-type difficulty histogram (min/mean/max) + the #promotions/#demotions since
    the last print + mean done-episode path length -- so the training log SHOWS the
    curriculum difficulty climbing flat -> 0.15 m. This is NOT gated behind
    NUKA_GO2_TERRAIN_DIAG (that env-var guards the much noisier per-25-step base-z
    histogram in the env); the difficulty report is the curriculum's headline
    health metric and is always emitted.
    """

    def after_init(self, algo):
        super().after_init(algo)
        self._shim = _StdoutMetricsWriter(algo.writer)
        algo.writer = self._shim
        self.writer = self._shim
        # Reach the live Go2 env's curriculum (algo.vec_env is the NukaVecEnv;
        # .env is the Go2LocomotionEnv). None when terrain is OFF or unavailable.
        self._curriculum = None
        try:
            env = getattr(getattr(algo, "vec_env", None), "env", None)
            self._curriculum = getattr(env, "_curriculum", None)
        except Exception:
            self._curriculum = None
        print(f"[metrics] observer attached (writer={'real' if self._shim._writer else 'none'}; "
              f"terrain_curriculum={'on' if self._curriculum is not None else 'off'})",
              flush=True)

    def after_print_stats(self, frame, epoch_num, total_time):
        super().after_print_stats(frame, epoch_num, total_time)
        if isinstance(getattr(self, "_shim", None), _StdoutMetricsWriter):
            self._shim.flush_row(epoch_num, frame)
        # Always-on terrain-curriculum difficulty report (every N epochs).
        if (self._curriculum is not None
                and _TERRAIN_DIAG_EVERY > 0
                and epoch_num % _TERRAIN_DIAG_EVERY == 0):
            try:
                print(f"[terrain] epoch {epoch_num}: "
                      f"{self._curriculum.difficulty_report()}", flush=True)
            except Exception as exc:  # never let a diagnostic kill training.
                print(f"[terrain] epoch {epoch_num}: report failed: {exc}",
                      flush=True)


_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CFG = os.path.join(_HERE, "go2_ppo_cfg.yaml")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Nuka Go2 PPO training (rl_games).")
    p.add_argument("--cfg", default=_DEFAULT_CFG, help="rl_games yaml config path.")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny launch-proof run (few epochs, small actors).")
    p.add_argument("--num_actors", type=int, default=None,
                   help="Override env_count / num_actors.")
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--horizon_length", type=int, default=None)
    p.add_argument("--minibatch_size", type=int, default=None)
    p.add_argument("--mini_epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--command", type=float, nargs=3, default=None,
                   metavar=("VX", "VY", "WYAW"),
                   help="Constant velocity command for all envs.")
    p.add_argument("--gait-mode", dest="gait_mode",
                   choices=("walk", "trot", "pronk", "bound"), default=None,
                   help="Go2 gait reward mode (overrides env_config.gait_mode). "
                        "walk=default flat walker; trot/pronk/bound re-weight the "
                        "reward for flight gaits. Default None => keep the yaml.")
    p.add_argument("--experiment-name", dest="experiment_name", default=None)
    p.add_argument("--train-dir", dest="train_dir", default=None,
                   help="rl_games output root (config.train_dir). The run writes "
                        "checkpoints to <train_dir>/<experiment_name>/nn/. Default "
                        "'runs'; set 'out' + --experiment-name go2_terrain_policy "
                        "for out/go2_terrain_policy/nn/.")
    # Go2-on-stairs Phase 2b: warm-start from a flat-policy checkpoint
    # (rl_games restores it into the agent BEFORE training -- so terrain training
    # resumes from a walking policy, not from scratch).
    p.add_argument("--checkpoint", default=None,
                   help="Path to a .pth checkpoint to warm-start from "
                        "(rl_games agent.restore). E.g. the flat baseline "
                        "out/go2_policy/go2_walk_randomcmd_ep250.pth.")
    # Toggle the terrain curriculum without editing the yaml (additive override).
    p.add_argument("--terrain", dest="terrain", action="store_true", default=None,
                   help="Force-ENABLE the procedural-terrain curriculum "
                        "(overrides config.env_config.terrain.enable).")
    p.add_argument("--no-terrain", dest="terrain", action="store_false",
                   help="Force-DISABLE the terrain curriculum (flat training).")
    # Toggle the legged_gym-style terrain HEIGHT-SCAN obs (needs --terrain). It
    # grows the obs to 48 + n_scan, so a run with it ON cannot warm-start from a
    # blind (48-dim) checkpoint -- it is a COLD-START.
    p.add_argument("--height-scan", dest="height_scan", action="store_true", default=None,
                   help="Force-ENABLE the terrain height-scan obs "
                        "(overrides config.env_config.terrain.height_scan.enable).")
    p.add_argument("--no-height-scan", dest="height_scan", action="store_false",
                   help="Force-DISABLE the height-scan obs (blind proprioception).")
    return p


def _apply_overrides(params: dict, args: argparse.Namespace) -> dict:
    cfg = params["config"]

    if args.smoke:
        # Tiny but VALID: minibatch must divide horizon_length * num_actors, so
        # set minibatch = horizon * actors (a single minibatch). Keep it fast.
        params["seed"] = args.seed if args.seed is not None else 0
        cfg["num_actors"] = args.num_actors or 256
        cfg["horizon_length"] = args.horizon_length or 16
        cfg["max_epochs"] = args.max_epochs or 3
        cfg["mini_epochs"] = args.mini_epochs or 2
        cfg["minibatch_size"] = (
            args.minibatch_size or cfg["num_actors"] * cfg["horizon_length"]
        )
        cfg["save_frequency"] = 10_000  # don't checkpoint during the smoke
        cfg["save_best_after"] = 10_000
        cfg["score_to_win"] = 10**9     # never "win" -> always runs all epochs
        cfg["name"] = cfg["full_experiment_name"] = "go2_smoke"
        # Short episodes so truncations fire WITHIN a rollout -> rl_games' reward
        # / episode-length scalars populate (guarded by current_size>0) and the
        # autoreset/done path is exercised inside the smoke. 1.0s @ 50Hz ~= 50
        # control steps.
        cfg.setdefault("env_config", {})["episode_length_s"] = 1.0

    # Explicit CLI overrides (apply on top of smoke/full defaults).
    if args.seed is not None:
        params["seed"] = args.seed
    if args.num_actors is not None:
        cfg["num_actors"] = args.num_actors
    if args.max_epochs is not None:
        cfg["max_epochs"] = args.max_epochs
    if args.horizon_length is not None:
        cfg["horizon_length"] = args.horizon_length
    if args.minibatch_size is not None:
        cfg["minibatch_size"] = args.minibatch_size
    if args.mini_epochs is not None:
        cfg["mini_epochs"] = args.mini_epochs
    if args.command is not None:
        cfg.setdefault("env_config", {})["command"] = list(args.command)
    if args.gait_mode is not None:
        cfg.setdefault("env_config", {})["gait_mode"] = args.gait_mode
    if args.experiment_name is not None:
        cfg["name"] = cfg["full_experiment_name"] = args.experiment_name
    if args.train_dir is not None:
        cfg["train_dir"] = args.train_dir
    # Terrain curriculum enable/disable override (CLI wins over yaml).
    if args.terrain is not None:
        ec = cfg.setdefault("env_config", {})
        ec.setdefault("terrain", {})["enable"] = bool(args.terrain)
    # Height-scan obs enable/disable override (CLI wins over yaml; needs terrain).
    if args.height_scan is not None:
        ec = cfg.setdefault("env_config", {})
        ec.setdefault("terrain", {}).setdefault("height_scan", {})["enable"] = bool(args.height_scan)

    # Guard the rl_games divisibility assertion early with a clear message.
    batch = cfg["horizon_length"] * cfg["num_actors"]
    if batch % cfg["minibatch_size"] != 0:
        raise SystemExit(
            f"minibatch_size ({cfg['minibatch_size']}) must divide "
            f"horizon_length*num_actors ({cfg['horizon_length']}*"
            f"{cfg['num_actors']}={batch})."
        )
    return params


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)

    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        print("[train_go2_ppo] WARNING: CUDA_VISIBLE_DEVICES unset -- this stack "
              "is single-GPU; export CUDA_VISIBLE_DEVICES=0.", file=sys.stderr)

    # Registers the 'nuka_go2' env_name + 'NUKA' vecenv type with rl_games.
    import nuka.rl_games  # noqa: F401

    from rl_games.torch_runner import Runner

    with open(args.cfg, "r") as fh:
        raw = yaml.safe_load(fh)
    # rl_games yaml nests everything under a top-level ``params:`` key;
    # Runner.load_config wants that inner dict.
    params = raw["params"] if "params" in raw else raw
    params = _apply_overrides(params, args)

    cfg = params["config"]
    print(f"[train_go2_ppo] env_name={cfg['env_name']} num_actors={cfg['num_actors']} "
          f"horizon={cfg['horizon_length']} minibatch={cfg['minibatch_size']} "
          f"mini_epochs={cfg['mini_epochs']} max_epochs={cfg['max_epochs']} "
          f"smoke={args.smoke}", flush=True)

    runner = Runner(algo_observer=StdoutAlgoObserver())
    runner.load_config(params)
    run_args = {"train": True, "play": False}
    if args.checkpoint:
        if not os.path.isfile(args.checkpoint):
            raise SystemExit(f"--checkpoint not found: {args.checkpoint}")
        # rl_games restores this into the agent BEFORE the first epoch (warm-start).
        run_args["checkpoint"] = args.checkpoint
        print(f"[train_go2_ppo] WARM-START from checkpoint {args.checkpoint}",
              flush=True)
    runner.run(run_args)
    print("[train_go2_ppo] run() returned cleanly.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
