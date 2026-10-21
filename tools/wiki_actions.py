"""Writes the Wiki's action reference, docs/reference/actions/, from assets/editor/actions.toml.

    python tools/wiki_actions.py [--check]

One page per group of tools/skill_actions.py, each action with its arguments, what it returns and an
example, and an index of the groups. With --check it writes nothing and exits 1 when a page differs
from what the catalog gives.
"""

import argparse
import json
import re
import sys
import tomllib

from skill_actions import CATALOG, GROUPS, REPO

OUT = REPO / "docs/reference/actions"
# Each group's page, by the group's first prefix.
SLUGS = {"file": "documents", "edit": "edits", "scene": "scene", "asset": "assets", "material": "materials",
         "robot": "robots", "clip": "motion", "sim": "simulation", "sensor": "sensors", "train": "training",
         "view": "view", "window": "windows", "script": "scripts", "agent": "editor"}
KINDS = [("read", "Reads state and changes nothing."),
         ("view", "The camera, panels, overlays and other display state; never the document."),
         ("edit", "The document and what is saved beside it: poses, clips, randomization, labels."),
         ("sim", "Runs, steps, resets or drives the simulation, or records it."),
         ("file", "Reads or writes files."),
         ("process", "Runs code, scripts or other programs."),
         ("app", "The editor itself: settings, extensions, windows, exit.")]
GENERATED = "<!-- Generated from assets/editor/actions.toml by tools/wiki_actions.py; edit those instead. -->\n"


def text(s: str) -> str:
    """Catalog prose as Markdown: code spans kept, the characters Markdown would read as markup escaped."""
    parts = s.split("`")
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r"([\\*_\[\]])", r"\\\1", parts[i]).replace("<", "&lt;").replace(">", "&gt;")
    return "`".join(parts)


def entry(action_id: str, doc: dict) -> str:
    out = [f"## `{action_id}`\n", f"**Kind:** [{doc['kind']}](index.md#kinds)\n", text(doc["summary"]) + "\n"]
    args = doc.get("args", {})
    if args:
        out.append("**Arguments**\n")
        out += [f"- `{name}`: {text(spec)}" for name, spec in args.items()]
        out.append("")
    if "returns" in doc:
        out.append(f"**Returns:** {text(doc['returns'])}\n")
    if "example" in doc:
        call = json.dumps({"id": action_id, "args": json.loads(doc["example"])}, ensure_ascii=False)
        out.append(f"**Example**\n\n```json\n{call}\n```\n")
    return "\n".join(out)


def pages(catalog: dict) -> dict:
    """Every page's file name and text."""
    owner = {prefix: prefixes[0] for prefixes, _ in GROUPS for prefix in prefixes}
    titles = {prefixes[0]: title for prefixes, title in GROUPS}
    grouped = {prefixes[0]: [] for prefixes, _ in GROUPS}
    for action_id, doc in catalog.items():
        grouped[owner[action_id.split(".")[0]]].append((action_id, doc))
    out = {}
    rows = []
    for first, actions in grouped.items():
        page = f"{SLUGS[first]}.md"
        rows.append(f"| [{titles[first]}]({page}) | {len(actions)} |")
        body = [GENERATED, f"# {titles[first]}\n", f"{len(actions)} actions. Run one from the command palette "
                "(++ctrl+p++), from Python with `app.actions.run(id, **args)` or through MCP with `action_run`.\n"]
        body += [entry(action_id, doc) for action_id, doc in actions]
        out[page] = "\n".join(body)
    kinds = [f"| `{kind}` | {meaning} |" for kind, meaning in KINDS]
    out["index.md"] = "\n".join([
        GENERATED, "# Actions\n",
        "Every menu item, toolbar button, shortcut and panel command in the editor runs an *action*: an id such "
        "as `file.import` with JSON arguments. The same actions are what scripts, extensions, smoke scenarios "
        f"and agents run, so this reference covers everything the editor does. It lists {len(catalog)} actions; "
        "extensions add their own.\n",
        "| Group | Actions |", "| --- | --- |", *rows, "",
        "## Running an action\n",
        "- **In the editor**: menus, toolbar, shortcuts, or the command palette (++ctrl+p++), which finds any "
        "action by name.",
        "- **Python in the editor** (Script panel, Console): `app.actions.run(\"sim.step\", steps=10)`.",
        "- **Another process**: `nuka.editor.connect()` returns a client whose `actions.run` takes the same "
        "arguments; see [Agents and MCP](../../editor/agents.md).",
        "- **MCP**: `action_run {id, args}`; `action_help {id}` returns an entry of this reference.\n",
        "An argument without a default is required. A path argument left out opens a file dialog, which "
        "waits for the user. An action that changes the document is one undoable step.\n",
        "## Kinds\n",
        "Each action's kind says what it touches, and with it what an agent's access mode lets through.\n",
        "| Kind | Touches |", "| --- | --- |", *kinds, ""])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    opts = parser.parse_args()
    with open(CATALOG, "rb") as f:
        wanted = pages(tomllib.load(f))
    if opts.check:
        stale = [name for name, body in wanted.items()
                 if not (OUT / name).exists() or (OUT / name).read_text(encoding="utf-8") != body]
        print(f"stale: {', '.join(stale)}; run tools/wiki_actions.py" if stale else "the action pages are current")
        return 1 if stale else 0
    OUT.mkdir(parents=True, exist_ok=True)
    for name, body in wanted.items():
        (OUT / name).write_text(body, encoding="utf-8", newline="\n")
    print(f"wrote {len(wanted)} pages to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
