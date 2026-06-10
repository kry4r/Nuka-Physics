#!/usr/bin/env python
"""Launch PPO training for the Nuka H1 2-finger cup-grasp task via rl_games (A4).

Mirrors train_h1_stand_ppo.py: reuses the proven Go2 rl_games launcher
(``_apply_overrides`` + ``StdoutAlgoObserver``); the grasp differences live in the
YAML (``env_name: nuka_h1_grasp``, 27-obs / 2-action continuous, empty env_config)
and one env_config-hygiene fix (below).

Smoke (PLUMBING proof -- a few epochs, NOT convergence; run from the REPO ROOT so
the cup asset path ``.nuka-assets/...`` resolves)::

    CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/cuda-12.8-root/usr/local/cuda-12.8/lib64 \
        python examples/training/train_h1_grasp_ppo.py --smoke

Full run::

    CUDA_VISIBLE_DEVICES=0 python examples/training/train_h1_grasp_ppo.py

ENV_CONFIG HYGIENE (why this isn't just train_h1_stand with a new cfg):
  ``H1GraspEnv.__init__`` has NO ``**kwargs`` catch-all and ``make_env`` forwards
  ``**kw`` straight into it, so ANY env_config key it doesn't declare is a hard
  ``TypeError`` at construction. The shared ``_apply_overrides`` smoke path injects
  ``env_config['episode_length_s']`` (valid for go2/h1_stand, which convert it to a
  step count) -- but H1GraspEnv takes ``episode_length`` (raw steps, default 220),
  NOT ``episode_length_s``. So after the base overrides we POP that key. (We do NOT
  translate it: the grasp done/autoreset path is already exercised every rollout --
  under no-op the cup free-falls through termination_z in ~40 steps -- so the smoke
  does not need short episodes.)
"""

from __future__ import annotations

import os
import sys

# Reuse the proven rl_games launcher/metrics shim from the Go2 trainer.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import train_go2_ppo as _base  # noqa: E402
from grasp_catch_eval import CatchRateAlgoObserver  # noqa: E402

_DEFAULT_CFG = os.path.join(_HERE, "h1_grasp_ppo_cfg.yaml")


def _build_argparser():
    parser = _base._build_argparser()
    parser.description = "Nuka H1 2-finger cup-grasp PPO training (rl_games)."
    parser.set_defaults(cfg=_DEFAULT_CFG)
    parser.add_argument("--checkpoint", default=None,
                        help="Resume training from an rl_games checkpoint.")
    # A5b: periodic DETERMINISTIC catch-rate eval on a SEPARATE env (the metric
    # that answers 'does PPO learn to grasp', which return cannot -- cup transit
    # farms return). --catch-eval-every K (0 disables -> plain StdoutAlgoObserver).
    parser.add_argument("--catch-eval-every", dest="catch_eval_every",
                        type=int, default=5,
                        help="Run the deterministic catch-rate eval every K "
                             "epochs (0 disables).")
    parser.add_argument("--catch-eval-envs", dest="catch_eval_envs",
                        type=int, default=256,
                        help="Env count for the catch-rate eval env.")
    parser.add_argument("--catch-log", dest="catch_log", default=None,
                        help="CSV path for the catch-rate log "
                             "(default runs/grasp_catch_eval.csv).")
    # A5b: let the SHORT bootstrapping window checkpoint frequently so the trained
    # high-catch policy is captured as an artifact (the cfg default save_frequency=
    # 100 never fires in a <100-epoch run).
    parser.add_argument("--save-frequency", dest="save_frequency", type=int,
                        default=None, help="Override cfg save_frequency (epochs).")
    parser.add_argument("--save-best-after", dest="save_best_after", type=int,
                        default=None, help="Override cfg save_best_after (epochs).")
    return parser


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)

    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        print("[train_h1_grasp_ppo] WARNING: CUDA_VISIBLE_DEVICES unset -- this stack "
              "is single-GPU; export CUDA_VISIBLE_DEVICES=0.", file=sys.stderr)

    import nuka.rl_games  # noqa: F401  (registers nuka_h1_grasp)
    from rl_games.torch_runner import Runner
    import yaml

    with open(args.cfg, "r") as fh:
        raw = yaml.safe_load(fh)
    params = raw["params"] if "params" in raw else raw
    params = _base._apply_overrides(params, args)

    cfg = params["config"]
    # ENV_CONFIG HYGIENE: drop the go2/h1_stand-only smoke key the shared override
    # injects -- H1GraspEnv does not accept episode_length_s (see module docstring).
    cfg.get("env_config", {}).pop("episode_length_s", None)
    if args.save_frequency is not None:
        cfg["save_frequency"] = args.save_frequency
    if args.save_best_after is not None:
        cfg["save_best_after"] = args.save_best_after
    if args.smoke:
        cfg["name"] = cfg["full_experiment_name"] = "h1_grasp_smoke"
    print(f"[train_h1_grasp_ppo] env_name={cfg['env_name']} num_actors={cfg['num_actors']} "
          f"horizon={cfg['horizon_length']} minibatch={cfg['minibatch_size']} "
          f"mini_epochs={cfg['mini_epochs']} max_epochs={cfg['max_epochs']} "
          f"env_config={cfg.get('env_config', {})} smoke={args.smoke}", flush=True)

    if args.catch_eval_every and args.catch_eval_every > 0:
        observer = CatchRateAlgoObserver(
            eval_envs=args.catch_eval_envs,
            eval_every=args.catch_eval_every,
            log_path=args.catch_log,
        )
        print(f"[train_h1_grasp_ppo] catch-rate eval ON: every "
              f"{args.catch_eval_every} epochs, {args.catch_eval_envs} envs "
              f"(SEPARATE eval env; does not perturb training).", flush=True)
    else:
        observer = _base.StdoutAlgoObserver()
    runner = Runner(algo_observer=observer)
    runner.load_config(params)
    run_args = {"train": True, "play": False}
    if args.checkpoint:
        run_args["checkpoint"] = args.checkpoint
    runner.run(run_args)
    print("[train_h1_grasp_ppo] run() returned cleanly.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
