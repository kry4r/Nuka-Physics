"""The rl_games observer through which a training run reports epochs, saves and stops."""

from __future__ import annotations

import sys

from rl_games.common.algo_observer import AlgoObserver

# The scalars rl_games writes before it reports an epoch, by the name a metrics row gives them.
_TAGS = {"fps": "performance/step_inference_rl_update_fps", "actor_loss": "losses/a_loss",
         "critic_loss": "losses/c_loss", "entropy": "losses/entropy", "kl": "info/kl", "lr": "info/last_lr"}


class _Writer:
    """Passes scalars on to rl_games' SummaryWriter and keeps the latest of each tag."""

    def __init__(self, writer):
        self._writer = writer
        self.values = {}

    def add_scalar(self, tag, value, *args, **kwargs):
        self._writer.add_scalar(tag, value, *args, **kwargs)
        self.values[tag] = float(value)

    def __getattr__(self, name):
        return getattr(self._writer, name)


class TrainObserver(AlgoObserver):
    """Reports to a runner.Run: `epoch(row)` after each epoch, `saved(path, algo)` after each save.

    When the run asks to stop, the epoch that just ended becomes the last; rl_games then saves a
    final checkpoint and returns. On Windows, viewers can watch the env's first environments
    through an observe.Server.
    """

    def __init__(self, run, scene: str, spacing: float):
        super().__init__()
        self.run = run
        self.scene = scene
        self.spacing = spacing

    def after_init(self, algo):
        self.algo = algo
        self.writer = algo.writer = _Writer(algo.writer)
        save = algo.save

        def saved(fn):
            save(fn)
            self.run.saved(fn + ".pth", algo)

        algo.save = saved
        fields = {"state": "running", "max_epochs": int(algo.max_epochs)}
        if sys.platform == "win32":
            from .observe import Server

            server = Server(algo.vec_env.env, self.scene, self.spacing)
            step = algo.vec_env.step

            def stepped(actions):
                out = step(actions)
                server.step()
                return out

            algo.vec_env.step = stepped
            fields["observe"] = server.port
        self.run.update(**fields)

    def after_clear_stats(self):
        pass

    def after_print_stats(self, frame, epoch_num, total_time):
        a = self.algo
        done = a.game_rewards.current_size > 0
        row = {"epoch": int(epoch_num), "frames": int(frame), "time": float(total_time),
               "reward": float(a.game_rewards.get_mean()[0]) if done else None,
               "length": float(a.game_lengths.get_mean()) if done else None}
        row.update({key: self.writer.values.get(tag) for key, tag in _TAGS.items()})
        self.run.epoch(row)
        if self.run.stop_requested():
            a.max_epochs = epoch_num
