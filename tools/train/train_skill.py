"""rl_games training launcher for the Nuka Go2 skill tasks.

Usage:
    python tools/train/train_skill.py --config configs/rl_games/go2_front_handstand.yaml \
        [--max-iters 5000] [--checkpoint runs/.../nn/<name>.pth] [--play]
"""
from __future__ import annotations

import argparse

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="total env-frame budget (this rl_games fork stops on "
                         "config.max_frames)")
    ap.add_argument("--num-actors", type=int, default=0)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--sigma", type=float, default=0.0,
                    help="policy std override for --play")
    ap.add_argument("--play", action="store_true")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    conf = cfg["params"]["config"]
    if args.max_frames > 0:
        conf["max_frames"] = args.max_frames
        conf["max_agent_steps"] = args.max_frames
    if args.num_actors > 0:
        conf["num_actors"] = args.num_actors
        # Keep the batch/minibatch divisibility invariant when scaling actors.
        batch = args.num_actors * conf.get("horizon_length", 24)
        conf["minibatch_size"] = max(1, batch // 4)
    run_args = {"train": not args.play, "play": args.play}
    if args.play and args.sigma > 0:
        run_args["sigma"] = args.sigma
    if args.checkpoint:
        # This fork's _restore() reads the checkpoint from RUN ARGS, not config.
        run_args["checkpoint"] = args.checkpoint

    import nuka.rl_games  # noqa: F401
    from rl_games.common.algo_observer import DefaultAlgoObserver
    from rl_games.torch_runner import Runner

    class RewardLoggingObserver(DefaultAlgoObserver):
        """Print episode rewards on every rl_games epoch."""

        def after_print_stats(self, frame, epoch_num, total_time):
            super().after_print_stats(frame, epoch_num, total_time)
            rewards = self.algo.game_rewards
            shaped = self.algo.game_shaped_rewards
            lengths = self.algo.game_lengths
            if rewards.current_size > 0:
                mean_reward = float(rewards.get_mean()[0])
                mean_shaped = float(shaped.get_mean()[0])
                mean_length = float(lengths.get_mean())
                count = int(rewards.current_size)
                print(
                    f"[epoch-reward] epoch={epoch_num} frame={frame} "
                    f"mean_reward={mean_reward:.6f} "
                    f"mean_shaped_reward={mean_shaped:.6f} "
                    f"mean_episode_length={mean_length:.2f} episodes={count}",
                    flush=True,
                )
            else:
                print(
                    f"[epoch-reward] epoch={epoch_num} frame={frame} "
                    "mean_reward=NA mean_shaped_reward=NA "
                    "mean_episode_length=NA episodes=0",
                    flush=True,
                )

    runner = Runner(algo_observer=RewardLoggingObserver())
    runner.load(cfg)
    runner.run(run_args)


if __name__ == "__main__":
    main()
