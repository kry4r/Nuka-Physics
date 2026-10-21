"""Training tasks the editor configures and launches.

A task describes its configuration as manager-shaped groups (observation, action, command,
reward, termination, event, algorithm, run) of typed parameters, says what robot its scene
must hold, and turns a configuration into the rl_games parameters that train it. Nothing here
imports torch, so the editor can read schemas without the training stack.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List

# Parameter kinds: int, float, bool, enum, range (lo, hi), vec3, ints (a list), term (read-only
# observation or action term: `dims` values scaled by `default`).
KINDS = ("int", "float", "bool", "enum", "range", "vec3", "ints", "term")


@dataclass
class Param:
    key: str
    kind: str
    default: object
    min: float | None = None
    max: float | None = None
    choices: tuple = ()
    unit: str = ""
    help: str = ""
    readonly: bool = False
    dims: int = 0


@dataclass
class Group:
    key: str
    title: str
    params: List[Param]


@dataclass
class TaskSpec:
    name: str
    title: str
    env: str
    groups: List[Group]
    # The one robot the scene holds, copied per env: its links (root included) and driven joints.
    robot: Dict[str, int]
    # The physics timestep the task trains at and its policy runs at, in seconds.
    dt: float
    # The metres apart a viewer spreads the environments, which overlap; 0 when each has its own place.
    spacing: float = 0.0
    build: Callable[[dict, str, str], dict] = field(repr=False, default=None)
    check: Callable[[dict, dict], list] = field(repr=False, default=None)

    def schema(self) -> dict:
        # The training process imports `module` from `root` to find a task an extension registered.
        module = self.build.__module__
        file = Path(sys.modules[module].__file__).resolve()
        root = file.parents[module.count(".") + (file.name == "__init__.py")]
        return {"name": self.name, "title": self.title, "env": self.env, "robot": dict(self.robot), "dt": self.dt,
                "spacing": self.spacing, "module": module, "root": str(root),
                "groups": [{"key": g.key, "title": g.title, "params": [asdict(p) for p in g.params]}
                           for g in self.groups]}

    def param(self, group: str, key: str) -> Param | None:
        for g in self.groups:
            if g.key == group:
                for p in g.params:
                    if p.key == key:
                        return p
        return None


# rl_games PPO for legged locomotion, aligned to the unitree_rl_gym / rsl_rl recipe.
_PPO = {
    "seed": 0,
    "algo": {"name": "a2c_continuous"},
    "model": {"name": "continuous_a2c_logstd"},
    "network": {
        "name": "actor_critic",
        "separate": False,
        "space": {"continuous": {"mu_activation": "None", "sigma_activation": "None",
                                 "mu_init": {"name": "default"},
                                 "sigma_init": {"name": "const_initializer", "val": 0},
                                 "fixed_sigma": True}},
        "mlp": {"units": [512, 256, 128], "activation": "elu", "d2rl": False,
                "initializer": {"name": "default"}, "regularizer": {"name": "None"}},
    },
    "config": {
        "multi_gpu": False, "mixed_precision": False, "normalize_input": True, "normalize_value": True,
        "normalize_advantage": True, "value_bootstrap": True, "use_smooth_clamp": True,
        "reward_shaper": {"scale_value": 1.0}, "lr_schedule": "adaptive", "score_to_win": 10 ** 9,
        "save_best_after": 0, "grad_norm": 1.0, "truncate_grads": True, "critic_coef": 1.0,
        "clip_value": True, "bounds_loss_coef": 0.0001,
    },
}

# legged_gym Go2 reward scales (nuka.tasks.go2_rewards.SCALES, walk gait).
_GO2_REWARDS = [
    ("tracking_lin_vel", 1.0, "Track the commanded xy velocity"),
    ("tracking_ang_vel", 0.5, "Track the commanded yaw rate"),
    ("lin_vel_z", -2.0, "Penalize vertical base velocity"),
    ("ang_vel_xy", -0.05, "Penalize roll and pitch rates"),
    ("torques", -0.0002, "Penalize joint torques"),
    ("dof_acc", -2.5e-7, "Penalize joint accelerations"),
    ("action_rate", -0.01, "Penalize action changes"),
    ("dof_pos_limits", -10.0, "Penalize joints near their limits"),
    ("feet_air_time", 1.0, "Reward long swing phases"),
    ("collision", -1.0, "Penalize links clipping the ground"),
    ("feet_clearance", 0.0, "Penalize swing feet below 0.10 m"),
    ("base_height", 0.0, "Penalize the base below 0.34 m"),
    ("flight_phase", 0.0, "Reward all feet in the air"),
    ("vertical_oscillation", 0.0, "Reward vertical bouncing"),
    ("gait_symmetry", 0.0, "Reward synchronized feet"),
]

_CONTRACT = "fixed by the deployment contract the editor's policy runner uses"


def _go2_groups() -> List[Group]:
    term = lambda key, dims, scale, help: Param(key, "term", scale, dims=dims, help=help, readonly=True)
    return [
        Group("run", "Run", [
            Param("num_envs", "int", 1024, 1, 16384, help="Robots trained in parallel"),
            Param("max_epochs", "int", 200, 1, 100000, help="PPO iterations"),
            Param("save_every", "int", 25, 1, 100000, help="Epochs between checkpoints"),
            Param("seed", "int", 0, 0, 2 ** 31 - 1, help="Seed of the training run"),
        ]),
        Group("observation", "Observation", [
            term("base_lin_vel", 3, 2.0, "Base linear velocity, body frame"),
            term("base_ang_vel", 3, 0.25, "Base angular velocity, body frame"),
            term("projected_gravity", 3, 1.0, "Gravity direction, body frame"),
            term("velocity_command", 3, 2.0, "Commanded vx, vy (x2.0) and yaw rate (x0.25)"),
            term("joint_pos", 12, 1.0, "Joint positions minus the default stance"),
            term("joint_vel", 12, 0.05, "Joint velocities"),
            term("last_action", 12, 1.0, "The previous action"),
        ]),
        Group("action", "Action", [
            Param("joint_position_scale", "float", 0.25, readonly=True, unit="rad",
                  help="Target = default stance + scale x action; " + _CONTRACT),
            Param("decimation", "int", 4, readonly=True, help="Physics steps per policy step; " + _CONTRACT),
            Param("stiffness", "float", 20.0, readonly=True, unit="N m/rad", help="PD gain; " + _CONTRACT),
            Param("damping", "float", 0.5, readonly=True, unit="N m s/rad", help="PD gain; " + _CONTRACT),
        ]),
        Group("command", "Command", [
            Param("fixed", "bool", False, help="Hold one command instead of resampling every 10 s"),
            Param("fixed_value", "vec3", [0.5, 0.0, 0.0], -2.0, 2.0, unit="m/s, m/s, rad/s",
                  help="The command held when fixed"),
            Param("lin_vel_x", "range", [-1.0, 1.0], -2.0, 2.0, unit="m/s", help="Forward velocity range"),
            Param("lin_vel_y", "range", [-1.0, 1.0], -2.0, 2.0, unit="m/s", help="Lateral velocity range"),
            Param("ang_vel_yaw", "range", [-1.0, 1.0], -3.0, 3.0, unit="rad/s", help="Yaw rate range"),
        ]),
        Group("reward", "Reward", [Param(k, "float", w, -100.0, 100.0, help=h) for k, w, h in _GO2_REWARDS]),
        Group("termination", "Termination", [
            Param("base_height", "float", 0.18, 0.0, 1.0, unit="m", help="End an episode below this base height"),
            Param("tilt", "float", 120.0, 0.0, 180.0, unit="deg", help="End an episode past this tilt"),
            Param("episode_length", "float", 20.0, 1.0, 120.0, unit="s", help="Time out an episode"),
        ]),
        Group("event", "Event", [
            Param("randomize", "bool", False, help="Draw physics once per robot at startup"),
            Param("mass", "range", [0.8, 1.2], 0.1, 5.0, unit="x", help="Link mass multiplier"),
            Param("friction", "range", [1.0, 1.0], 0.0, 5.0, unit="x", help="Contact friction multiplier"),
            Param("armature", "range", [0.0, 0.05], 0.0, 1.0, unit="kg m^2", help="Joint armature offset"),
            Param("gravity", "range", [-0.5, 0.5], -5.0, 5.0, unit="m/s^2", help="Gravity z offset"),
        ]),
        Group("algorithm", "Algorithm", [
            Param("learning_rate", "float", 1e-3, 1e-6, 1.0, help="Adam step size, adapted to the KL"),
            Param("kl_threshold", "float", 0.01, 1e-4, 1.0, help="KL the adaptive step size aims for"),
            Param("gamma", "float", 0.99, 0.0, 1.0, help="Discount"),
            Param("tau", "float", 0.95, 0.0, 1.0, help="GAE lambda"),
            Param("entropy_coef", "float", 0.01, 0.0, 1.0, help="Exploration bonus"),
            Param("e_clip", "float", 0.2, 0.01, 1.0, help="PPO clip"),
            Param("horizon_length", "int", 24, 1, 1024, help="Steps per robot per epoch"),
            Param("minibatches", "int", 4, 1, 1024, help="Minibatches per epoch; divides envs x horizon"),
            Param("mini_epochs", "int", 5, 1, 100, help="Passes over each epoch's batch"),
            Param("units", "ints", [512, 256, 128], 8, 4096, help="Hidden layer widths of actor and critic"),
        ]),
    ]


def _go2_build(config: dict, scene: str, name: str) -> dict:
    run, cmd, rew = config["run"], config["command"], config["reward"]
    term, event, algo = config["termination"], config["event"], config["algorithm"]
    params = copy.deepcopy(_PPO)
    params["seed"] = run["seed"]
    params["network"]["mlp"]["units"] = list(algo["units"])
    env_config = {
        "scene": scene, "contact_family": 1, "gait_mode": "walk",
        "fixed_command": cmd["fixed"], "command": list(cmd["fixed_value"]),
        "command_ranges": {k: list(cmd[k]) for k in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")},
        "reward_scales": dict(rew),
        "termination_height": term["base_height"], "termination_tilt_deg": term["tilt"],
        "episode_length_s": term["episode_length"],
    }
    if event["randomize"]:
        env_config["domain_randomization"] = {
            "mass_range": list(event["mass"]), "armature_range": list(event["armature"]),
            "gravity_range": list(event["gravity"]), "friction_range": list(event["friction"]),
            "restitution_range": [0.0, 0.0], "seed": run["seed"], "enabled": True}
    batch = run["num_envs"] * algo["horizon_length"]
    params["config"].update({
        "name": name, "env_name": "nuka_go2", "num_actors": run["num_envs"], "max_epochs": run["max_epochs"],
        "save_frequency": run["save_every"], "learning_rate": algo["learning_rate"],
        "kl_threshold": algo["kl_threshold"], "gamma": algo["gamma"], "tau": algo["tau"],
        "entropy_coef": algo["entropy_coef"], "e_clip": algo["e_clip"],
        "horizon_length": algo["horizon_length"], "minibatch_size": batch // algo["minibatches"],
        "mini_epochs": algo["mini_epochs"], "env_config": env_config,
    })
    return params


def _batch_check(config: dict, layout: dict) -> list:
    errors = []
    run, algo = config["run"], config["algorithm"]
    if (run["num_envs"] * algo["horizon_length"]) % algo["minibatches"] != 0:
        errors.append({"path": "algorithm.minibatches",
                       "message": f"{algo['minibatches']} minibatches do not divide "
                                  f"{run['num_envs']} envs x {algo['horizon_length']} steps"})
    return errors


# unitree_rl_lab G1 velocity reward weights (nuka.tasks.g1_rough.REWARD_SCALES).
_G1_REWARDS = [
    ("track_lin_vel_xy", 1.0, "Track the commanded xy velocity"),
    ("track_ang_vel_z", 0.5, "Track the commanded yaw rate"),
    ("alive", 0.15, "Reward every step not fallen"),
    ("lin_vel_z", -2.0, "Penalize vertical base velocity"),
    ("ang_vel_xy", -0.05, "Penalize roll and pitch rates"),
    ("joint_vel", -0.001, "Penalize joint velocities"),
    ("joint_acc", -2.5e-7, "Penalize joint accelerations"),
    ("action_rate", -0.05, "Penalize action changes"),
    ("dof_pos_limits", -5.0, "Penalize joints past 90% of their range"),
    ("energy", -2e-5, "Penalize mechanical power"),
    ("joint_deviation_arms", -0.1, "Keep the arms near the default stance"),
    ("joint_deviation_waist", -1.0, "Keep the waist near the default stance"),
    ("joint_deviation_legs", -1.0, "Keep hip roll and yaw near the default stance"),
    ("flat_orientation", -5.0, "Keep the base level"),
    ("base_height", -10.0, "Hold the base 0.78 m above the ground"),
    ("gait", 0.5, "Alternate the feet on a 0.8 s cycle"),
    ("feet_slide", -0.2, "Penalize feet sliding on the ground"),
    ("feet_clearance", 1.0, "Lift swing feet 0.10 m"),
    ("undesired_contacts", -1.0, "Penalize knees, elbows and wrists near the ground"),
]


def _g1_groups() -> List[Group]:
    term = lambda key, dims, scale, help: Param(key, "term", scale, dims=dims, help=help, readonly=True)
    return [
        Group("run", "Run", [
            Param("num_envs", "int", 4096, 1, 16384, help="Robots trained in parallel"),
            Param("max_epochs", "int", 3000, 1, 100000, help="PPO iterations"),
            Param("save_every", "int", 100, 1, 100000, help="Epochs between checkpoints"),
            Param("seed", "int", 0, 0, 2 ** 31 - 1, help="Seed of the training run"),
        ]),
        Group("observation", "Observation", [
            term("base_ang_vel", 3, 0.2, "Base angular velocity, body frame"),
            term("projected_gravity", 3, 1.0, "Gravity direction, body frame"),
            term("velocity_commands", 3, 1.0, "Commanded vx, vy and yaw rate"),
            term("joint_pos_rel", 23, 1.0, "Joint positions minus the default stance"),
            term("joint_vel_rel", 23, 0.05, "Joint velocities"),
            term("last_action", 23, 1.0, "The previous action"),
            Param("history", "int", 5, readonly=True, help="Frames of each term, oldest first; " + _CONTRACT),
            Param("noise", "bool", True, help="Add uniform sensor noise to the observations"),
        ]),
        Group("action", "Action", [
            Param("joint_position_scale", "float", 0.25, readonly=True, unit="rad",
                  help="Target = default stance + scale x action; " + _CONTRACT),
            Param("decimation", "int", 10, readonly=True, help="Physics steps per policy step; " + _CONTRACT),
        ]),
        Group("command", "Command", [
            Param("fixed", "bool", False, help="Hold one command instead of resampling every 10 s"),
            Param("fixed_value", "vec3", [0.5, 0.0, 0.0], -2.0, 2.0, unit="m/s, m/s, rad/s",
                  help="The command held when fixed"),
            Param("curriculum", "bool", True, help="Start the ranges at 0.1 and widen them as tracking improves"),
            Param("lin_vel_x", "range", [-0.5, 1.0], -2.0, 2.0, unit="m/s", help="Forward velocity limits"),
            Param("lin_vel_y", "range", [-0.3, 0.3], -2.0, 2.0, unit="m/s", help="Lateral velocity limits"),
            Param("ang_vel_yaw", "range", [-0.5, 0.5], -3.0, 3.0, unit="rad/s", help="Yaw rate limits"),
        ]),
        Group("terrain", "Terrain", [
            Param("kind", "enum", "bumps", choices=("flat", "bumps", "mixed"),
                  help="Flat ground, random bumps, or tiles of stairs, pits and bumps"),
            Param("bump_height", "float", 0.05, 0.0, 0.3, unit="m", help="Highest bump"),
            Param("bump_cell", "float", 0.4, 0.05, 2.0, unit="m", help="Bump footprint edge"),
            Param("step_height", "float", 0.08, 0.0, 0.3, unit="m", help="Stair rise of the mixed tiles"),
        ]),
        Group("reward", "Reward", [Param(k, "float", w, -100.0, 100.0, help=h) for k, w, h in _G1_REWARDS]),
        Group("termination", "Termination", [
            Param("base_height", "float", 0.3, 0.0, 1.0, unit="m", help="End an episode below this base height"),
            Param("tilt", "float", 45.0, 0.0, 180.0, unit="deg", help="End an episode past this tilt"),
            Param("episode_length", "float", 20.0, 1.0, 120.0, unit="s", help="Time out an episode"),
        ]),
        Group("event", "Event", [
            Param("push", "bool", True, help="Kick each robot up to 0.5 m/s sideways every 5 s"),
        ]),
        Group("algorithm", "Algorithm", [
            Param("learning_rate", "float", 1e-3, 1e-6, 1.0, help="Adam step size, adapted to the KL"),
            Param("kl_threshold", "float", 0.01, 1e-4, 1.0, help="KL the adaptive step size aims for"),
            Param("gamma", "float", 0.99, 0.0, 1.0, help="Discount"),
            Param("tau", "float", 0.95, 0.0, 1.0, help="GAE lambda"),
            Param("entropy_coef", "float", 0.01, 0.0, 1.0, help="Exploration bonus"),
            Param("e_clip", "float", 0.2, 0.01, 1.0, help="PPO clip"),
            Param("horizon_length", "int", 24, 1, 1024, help="Steps per robot per epoch"),
            Param("minibatches", "int", 4, 1, 1024, help="Minibatches per epoch; divides envs x horizon"),
            Param("mini_epochs", "int", 5, 1, 100, help="Passes over each epoch's batch"),
            Param("units", "ints", [512, 256, 128], 8, 4096, help="Hidden layer widths of actor and critic"),
        ]),
    ]


def _g1_build(config: dict, scene: str, name: str) -> dict:
    run, obs, cmd, rew = config["run"], config["observation"], config["command"], config["reward"]
    terr, term, event, algo = config["terrain"], config["termination"], config["event"], config["algorithm"]
    params = copy.deepcopy(_PPO)
    params["seed"] = run["seed"]
    params["network"]["mlp"]["units"] = list(algo["units"])
    env_config = {
        "scene": scene, "seed": run["seed"], "noise": obs["noise"], "push": event["push"],
        "fixed_command": cmd["fixed"], "command": list(cmd["fixed_value"]), "curriculum": cmd["curriculum"],
        "command_ranges": {k: list(cmd[k]) for k in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")},
        "terrain": dict(terr), "reward_scales": dict(rew),
        "termination_height": term["base_height"], "termination_tilt_deg": term["tilt"],
        "episode_length_s": term["episode_length"],
    }
    batch = run["num_envs"] * algo["horizon_length"]
    params["config"].update({
        "name": name, "env_name": "nuka_g1", "num_actors": run["num_envs"], "max_epochs": run["max_epochs"],
        "save_frequency": run["save_every"], "learning_rate": algo["learning_rate"],
        "kl_threshold": algo["kl_threshold"], "gamma": algo["gamma"], "tau": algo["tau"],
        "entropy_coef": algo["entropy_coef"], "e_clip": algo["e_clip"],
        "horizon_length": algo["horizon_length"], "minibatch_size": batch // algo["minibatches"],
        "mini_epochs": algo["mini_epochs"], "env_config": env_config,
        # Actions go to the robot unscaled and unclipped, as in rsl_rl.
        "clip_actions": False, "bounds_loss_coef": 0.0,
        # The critic is a network of its own on the env's privileged states, as in unitree_rl_lab.
        "use_experimental_cv": False,
        "central_value_config": {
            "minibatch_size": batch // algo["minibatches"], "mini_epochs": algo["mini_epochs"],
            "learning_rate": algo["learning_rate"], "e_clip": algo["e_clip"], "clip_value": True,
            "normalize_input": True, "truncate_grads": True, "grad_norm": 1.0,
            "network": {"name": "actor_critic", "central_value": True, "mlp": dict(params["network"]["mlp"])},
        },
    })
    return params


TASKS: Dict[str, TaskSpec] = {}


def register(spec: TaskSpec) -> None:
    """Adds a task, or replaces the one of the same name."""
    TASKS[spec.name] = spec


def unregister(name: str) -> None:
    """Drops a task, as the extension that registered it stops."""
    TASKS.pop(name)


register(TaskSpec(name="go2_velocity", title="Go2 velocity tracking", env="nuka_go2", groups=_go2_groups(),
                  robot={"links": 13, "joints": 12}, dt=0.005, spacing=1.0, build=_go2_build, check=_batch_check))
register(TaskSpec(name="g1_rough", title="G1 rough-terrain walking", env="nuka_g1", groups=_g1_groups(),
                  robot={"links": 31, "joints": 30}, dt=0.002, spacing=0.0, build=_g1_build, check=_batch_check))
