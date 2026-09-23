"""Fetch the pinned Unitree G1 velocity policy and its deployment contract."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import urllib.request


REVISION = "4960b84732b0c2ec593dccbfe963fda1bcd7b1e3"
REPOSITORY = "unitreerobotics/unitree_rl_lab"
POLICY_ROOT = "deploy/robots/g1_29dof/config/policy/velocity/v0"
FILES = {
    "policy.onnx": POLICY_ROOT + "/exported/policy.onnx",
    "deploy.yaml": POLICY_ROOT + "/params/deploy.yaml",
    "LICENSE": "LICENCE",
    "unitree.py": "source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py",
    "velocity_env_cfg.py": "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py",
    "observation_manager.h": "deploy/include/isaaclab/manager/observation_manager.h",
    "manager_term_cfg.h": "deploy/include/isaaclab/manager/manager_term_cfg.h",
    "observations.h": "deploy/include/isaaclab/envs/mdp/observations/observations.h",
    "unitree_articulation.h": "deploy/include/unitree_articulation.h",
}
POLICY_BLOB = "c50bb5f6b11d7f464ff6799a78987a12ce68fb97"


def actuator_contract(source: str) -> dict:
    tree = ast.parse(source)
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "UNITREE_G1_29DOF_CFG"
                for target in node.targets))
    arguments = {entry.arg: entry.value for entry in assignment.value.keywords}
    names = ast.literal_eval(arguments["joint_sdk_names"])
    actuators = arguments["actuators"]
    result = {"joint_sdk_names": names, "source_revision": REVISION, "joints": {}}
    for name in names:
        matches = []
        for key, value in zip(actuators.keys, actuators.values):
            config = {entry.arg: ast.literal_eval(entry.value) for entry in value.keywords}
            if any(re.fullmatch(pattern, name) for pattern in config["joint_names_expr"]):
                record = {"actuator": ast.literal_eval(key)}
                for field in ("effort_limit_sim", "velocity_limit_sim", "stiffness", "damping", "armature"):
                    item = config[field]
                    if isinstance(item, dict):
                        values = [v for pattern, v in item.items() if re.fullmatch(pattern, name)]
                        if len(values) != 1:
                            raise ValueError(f"Ambiguous actuator parameter: {name}/{field}")
                        item = values[0]
                    record[field] = float(item)
                matches.append(record)
        if len(matches) != 1:
            raise ValueError(f"Ambiguous actuator: {name}")
        result["joints"][name] = matches[0]
    return result


def fetch(destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"repository": f"https://github.com/{REPOSITORY}",
                "revision": REVISION, "files": {}}
    for name, source in FILES.items():
        url = f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/{source}"
        path = destination / name
        if path.exists():
            data = path.read_bytes()
        else:
            request = urllib.request.Request(url, headers={"User-Agent": "Nuka-Physics asset fetcher"})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            if not data:
                raise ValueError(f"Empty asset: {source}")
            path.write_bytes(data)
        if name == "policy.onnx":
            blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            if blob != POLICY_BLOB:
                raise ValueError("The G1 policy does not match the pinned Git blob")
        manifest["files"][name] = {"source": source, "url": url,
                                    "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    actuators = actuator_contract((destination / "unitree.py").read_text(encoding="utf-8"))
    (destination / "actuators.json").write_text(json.dumps(actuators, indent=2) + "\n", encoding="utf-8")
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=root / ".nuka-assets/policies/g1_velocity_v0")
    args = parser.parse_args()
    print(json.dumps(fetch(args.out), indent=2))


if __name__ == "__main__":
    main()
