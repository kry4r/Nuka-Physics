#!/usr/bin/env python
"""Launch PPO training for the Nuka H1 torque standing task via rl_games.

Smoke:
    CUDA_VISIBLE_DEVICES=0 python examples/training/train_h1_stand_ppo.py --smoke

Full phase-1 run:
    CUDA_VISIBLE_DEVICES=0 python examples/training/train_h1_stand_ppo.py
"""

from __future__ import annotations

import os
import sys

# Reuse the proven rl_games launcher/metrics shim from the Go2 trainer. The H1
# differences live in the YAML env_name/env_config and the default config path.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import train_go2_ppo as _base  # noqa: E402

_DEFAULT_CFG = os.path.join(_HERE, "h1_stand_ppo_cfg.yaml")


def _build_argparser():
    parser = _base._build_argparser()
    parser.description = "Nuka H1 torque standing PPO training (rl_games)."
    parser.set_defaults(cfg=_DEFAULT_CFG)
    parser.add_argument("--checkpoint", default=None,
                        help="Resume training from an rl_games checkpoint.")
    return parser


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)

    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        print("[train_h1_stand_ppo] WARNING: CUDA_VISIBLE_DEVICES unset -- this stack "
              "is single-GPU; export CUDA_VISIBLE_DEVICES=0.", file=sys.stderr)

    import nuka.rl_games  # noqa: F401  (registers nuka_h1_stand)
    from rl_games.torch_runner import Runner
    import yaml

    with open(args.cfg, "r") as fh:
        raw = yaml.safe_load(fh)
    params = raw["params"] if "params" in raw else raw
    params = _base._apply_overrides(params, args)

    cfg = params["config"]
    if args.smoke:
        cfg["name"] = cfg["full_experiment_name"] = "h1_stand_smoke"
    print(f"[train_h1_stand_ppo] env_name={cfg['env_name']} num_actors={cfg['num_actors']} "
          f"horizon={cfg['horizon_length']} minibatch={cfg['minibatch_size']} "
          f"mini_epochs={cfg['mini_epochs']} max_epochs={cfg['max_epochs']} "
          f"smoke={args.smoke}", flush=True)

    runner = Runner(algo_observer=_base.StdoutAlgoObserver())
    runner.load_config(params)
    run_args = {"train": True, "play": False}
    if args.checkpoint:
        run_args["checkpoint"] = args.checkpoint
    runner.run(run_args)
    print("[train_h1_stand_ppo] run() returned cleanly.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
