"""Writes the nuka-editor skill's action index, references/actions.md, from assets/editor/actions.toml.

    python tools/skill_actions.py [--check]

With --check it writes nothing and exits 1 when the index differs from what the catalog gives.
"""

import argparse
import pathlib
import sys
import tomllib

REPO = pathlib.Path(__file__).resolve().parents[1]
CATALOG = REPO / "assets/editor/actions.toml"
INDEX = REPO / "python/nuka/mcp/skill/nuka-editor/references/actions.md"
# Groups by id prefix, in the order a scene is built; prefixes not listed land in "Other".
GROUPS = [
    (("file",), "Documents: new, open, import, save, export"),
    (("edit",), "Edits and undo"),
    (("scene",), "Scene: add, collision, nodes"),
    (("asset", "assets"), "Assets"),
    (("material",), "Materials"),
    (("robot", "sysid"), "Robots: joints, poses, gains, identification, IK"),
    (("clip", "demo", "teleop", "policy"), "Motion: clips, demonstrations, teleoperation, policies"),
    (("sim", "timeline", "solver", "rerun"), "Simulation, timeline, solver and Rerun"),
    (("sensor", "label", "dataset", "dr"), "Sensors, labels, datasets and randomization"),
    (("train",), "Training"),
    (("view",), "View and camera"),
    (("window", "ui", "help", "tool", "tools"), "Windows, panels and tools"),
    (("script",), "Scripts and console"),
    (("agent", "action", "app", "extensions"), "The editor itself, agents and extensions"),
]
HEADER = """\
# Editor actions by group

Every editor feature is one of these actions. Run one with `action_run {id, args}`;
`action_help {id}` gives its whole entry: arguments with types and defaults, what it returns and an
example. `?` marks an optional argument. The kind in brackets is what the action touches, and so what
the user's access mode lets through (see safety.md).

This page is generated from assets/editor/actions.toml by tools/skill_actions.py.
"""


def first_sentence(text: str) -> str:
    end = text.find(". ")
    return text if end < 0 else text[:end + 1]


def entry(action_id: str, doc: dict) -> str:
    args = []
    for name, spec in doc.get("args", {}).items():
        optional = "default" in spec or "optional" in spec
        args.append(name + ("?" if optional else ""))
    shown = f" {{{', '.join(args)}}}" if args else ""
    return f"- `{action_id}`{shown} [{doc['kind']}]: {first_sentence(doc['summary'])}"


def render(catalog: dict) -> str:
    grouped = {title: [] for _, title in GROUPS}
    grouped["Other"] = []
    owner = {prefix: title for prefixes, title in GROUPS for prefix in prefixes}
    for action_id, doc in catalog.items():
        grouped[owner.get(action_id.split(".")[0], "Other")].append(entry(action_id, doc))
    parts = [HEADER]
    for title, lines in grouped.items():
        if lines:
            parts.append(f"## {title}\n\n" + "\n".join(lines) + "\n")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    opts = parser.parse_args()
    with open(CATALOG, "rb") as f:
        text = render(tomllib.load(f))
    if opts.check:
        current = INDEX.read_text(encoding="utf-8") if INDEX.exists() else ""
        print("actions.md is current" if current == text else "actions.md is stale; run tools/skill_actions.py")
        return 0 if current == text else 1
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {INDEX}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
