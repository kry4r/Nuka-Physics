"""Drafts the editor's action catalog, assets/editor/actions.toml, from its registrations.

    python tools/action_docs.py [--pid PID] [--out assets/editor/actions.toml]

Lists the actions of the editor with process id PID, else of one started for the purpose, and
drafts an entry for each id the catalog lacks: the comment above its registration gives
``summary``, the arguments its body reads give ``args``, the keys of the object it returns give
``returns`` and its category seeds ``kind``. Entries already in the catalog stay as edited and
ids no longer registered are dropped. Drafts want a read before they are kept; the review list
printed at the end names each new entry with what the source could not tell.
"""

import argparse
import json
import pathlib
import re
import sys
import tempfile
import tomllib

REPO = pathlib.Path(__file__).resolve().parents[1]
SOURCES = [REPO / "src/runtime/app/viewer" / name
           for name in ("editor_host.cpp", "imgui_layer.cpp", "editor_extensions.cpp")]
HEADER = """\
# The editor's actions: what each does, the arguments it reads and what it returns. The action
# list (app.actions.list(), the RPC `actions` call) carries each summary and kind, the action
# `action.help {id}` all of an entry; tools/action_docs.py drafts the entries of new actions.
#
# args maps a name to "type, meaning, default"; an argument without a default is required.
# kind says what an action touches, and so what an agent's access mode lets through:
#   read     reads state and changes nothing
#   view     the camera, panels, overlays and other display state; never the document
#   edit     the document and what is saved beside it (poses, clips, randomization, labels)
#   sim      runs, steps, resets or drives the simulation, or records it
#   file     reads or writes files
#   process  runs code, scripts or other programs
#   app      the editor itself: settings, extensions, windows, exit
"""
CATEGORY_KINDS = {"file": "file", "edit": "edit", "scene": "edit", "assets": "file", "material": "edit",
                  "robot": "edit", "sensor": "edit", "randomize": "edit", "sim": "sim", "timeline": "sim",
                  "view": "view", "window": "view", "help": "view", "tools": "view", "solver": "view",
                  "script": "process", "train": "process", "dataset": "file", "app": "app", "extensions": "app"}
# Action families registered in a loop; {label} is the member's label.
FAMILIES = {"scene.add_": ("Adds a {label} to the scene, where `scene.spawn` would put it.", "edit"),
            "window.layout_": ("Arranges the panels in the {label}.", "view"),
            "window.toggle_": ("Shows or hides the {label} panel.", "view")}
ARG_HELPERS = {"StrArg": "string", "NumArg": "number", "BoolArg": "bool", "OptBoolArg": "bool",
               "StrListArg": "list of strings", "Vec3Arg": "[x, y, z]"}
AS_TYPES = {"AsString": "string", "AsDouble": "number", "AsFloat": "number", "AsInt": "int", "AsBool": "bool",
            "Elements": "list", "Items": "object"}
VALUE_TYPES = {"Str": "a string", "Float": "a number", "Int": "an integer", "Bool": "a bool", "Array": "a list",
               "Object": "an object"}


def skip(text: str, i: int) -> int:
    """The index past the comment, string or character literal starting at i, else i."""
    if text.startswith("//", i):
        end = text.find("\n", i)
        return len(text) if end < 0 else end
    if text.startswith("/*", i):
        return text.index("*/", i) + 2
    if text[i] in "\"'":
        j = i + 1
        while text[j] != text[i]:
            j += 2 if text[j] == "\\" else 1
        return j + 1
    return i


def close(text: str, i: int) -> int:
    """The index of the bracket that closes the one at i."""
    depth, j = 0, i
    while True:
        k = skip(text, j)
        if k != j:
            j = k
            continue
        if text[j] in "([{":
            depth += 1
        elif text[j] in ")]}":
            depth -= 1
            if depth == 0:
                return j
        j += 1


def split_top(text: str) -> list:
    """The comma-separated items of a bracket's contents, nested brackets kept whole."""
    parts, depth, start, j = [], 0, 0, 0
    while j < len(text):
        k = skip(text, j)
        if k != j:
            j = k
            continue
        if text[j] in "([{":
            depth += 1
        elif text[j] in ")]}":
            depth -= 1
        elif text[j] == "," and depth == 0:
            parts.append(text[start:j].strip())
            start = j + 1
        j += 1
    parts.append(text[start:].strip())
    return parts


def calls(body: str, name: str):
    """The argument lists of each call to ``name`` in body."""
    for m in re.finditer(r"\b%s\(" % name, body):
        yield split_top(body[m.end():close(body, m.end() - 1)])


def comment_above(text: str, at: int) -> str:
    lines = text[:at].splitlines()[:-1]
    found = []
    for line in reversed(lines):
        if not line.strip().startswith("//"):
            break
        found.append(line.strip()[2:].strip())
    return " ".join(reversed(found))


def registrations() -> dict:
    """Each id registered as a literal, to the comment above its registration and its body."""
    found = {}
    pattern = re.compile(r"\b(?:actions_|registry|actions)\s*\.\s*(?:Add|push_back)\(\s*\{")
    for path in SOURCES:
        text = path.read_text(encoding="utf-8")
        for m in pattern.finditer(text):
            body = text[m.end():close(text, m.end() - 1)]
            if literal := re.fullmatch(r'"([\w.]+)"', split_top(body)[0]):
                found[literal[1]] = {"comment": comment_above(text, m.start()), "body": body}
    return found


def default_of(expr: str) -> str:
    expr = expr.strip()
    if re.fullmatch(r'"[^"]*"|true|false', expr):
        return "default " + expr
    if re.fullmatch(r"-?[\d.]+(e-?\d+)?f?", expr):
        return "default " + expr.rstrip("f")
    if expr.startswith("!"):
        return "default: toggles"
    return "default: the current value"


def read_args(body: str) -> tuple:
    """The arguments a registration body reads, and the functions it hands its args to."""
    found = {}
    for helper, kind in ARG_HELPERS.items():
        for a in calls(body, helper):
            if a[0] != "args" or not (key := re.fullmatch(r'"(\w+)"', a[1])):
                continue
            tail = default_of(a[2]) if helper in ("StrArg", "NumArg", "BoolArg") and len(a) > 2 else ""
            found.setdefault(key[1], ", ".join(t for t in (kind, "?", tail) if t))
    if re.search(r"\bClipArg\(", body):
        found.setdefault("clip", "string, the clip's name, default the clip being edited")
    if re.search(r"\bTabArg\(", body):
        found.setdefault("tab", "int, the script tab's id, default the active tab")
    for m in re.finditer(r'\bargs\.At\("(\w+)"\)(?:\.(\w+)\(\))?', body):
        found.setdefault(m[1], AS_TYPES.get(m[2], "value") + ", ?")
    for m in re.finditer(r'\bargs\.Find\("(\w+)"\)', body):
        found.setdefault(m[1], "value, ?, optional")
    known = set(ARG_HELPERS) | {"ClipArg", "TabArg"}
    handed = sorted({m[1] for m in re.finditer(r"\b(\w+)\(\s*args\s*[,)]", body)} - known)
    return found, handed


def returns_of(body: str) -> str:
    keys = []
    for m in re.finditer(r"\bOk\(Object\(\{", body):
        for item in split_top(body[m.end():close(body, m.end() - 1)]):
            if key := re.match(r'\{\s*"(\w+)"', item):
                keys.append(key[1])
    if keys:
        return "{" + ", ".join(dict.fromkeys(keys)) + "}"
    if m := re.search(r"(?:\bOk\(|ActionResult\{true,\s*)(?:std::move\()?json::Value::(\w+)\(", body):
        return VALUE_TYPES.get(m[1], "")
    if "SubmitTxn(" in body:
        return "{focus}: the node the edit selects"
    return "nothing"


def draft(action: dict, registered: dict) -> tuple:
    """A new entry for one listed action, and what its review should look at."""
    aid, label = action["id"], action["label"]
    if aid in registered:
        reg = registered[aid]
        summary = reg["comment"] or (f"Turns {label} on or off; without `on` it toggles." if "checked" in action
                                     else f"{label}.")
        args, handed = read_args(reg["body"])
        kind = CATEGORY_KINDS.get(action["category"], "process")
        if "SubmitTxn(" in reg["body"]:
            kind = "edit"
        elif summary.startswith(("The ", "Each ", "Every ")) and returns_of(reg["body"]) != "nothing":
            kind = "read"
        entry = {"summary": summary, "kind": kind, "args": args, "returns": returns_of(reg["body"])}
        notes = ([] if reg["comment"] else ["no comment"]) + [f"args via {h}()" for h in handed]
        return {k: v for k, v in entry.items() if v}, notes
    for prefix, (summary, kind) in FAMILIES.items():
        if aid.startswith(prefix):
            return {"summary": summary.format(label=label), "kind": kind, "returns": "nothing"}, []
    return {"summary": f"{label}.", "kind": "process"}, ["not in the editor's source"]


def toml_text(value: str) -> str:
    if "'" not in value and "\n" not in value and '"' in value:
        return f"'{value}'"
    return json.dumps(value, ensure_ascii=False)


def write_catalog(path: pathlib.Path, order: list, entries: dict) -> None:
    out = [HEADER]
    for aid in order:
        e = entries[aid]
        out.append(f'\n["{aid}"]\n')
        for key in ("summary", "kind"):
            out.append(f"{key} = {toml_text(e[key])}\n")
        for name, text in e.get("args", {}).items():
            out.append(f"args.{name} = {toml_text(text)}\n")
        for key in ("returns", "example"):
            if key in e:
                out.append(f"{key} = {toml_text(e[key])}\n")
    path.write_text("".join(out), encoding="utf-8")


def listed(pid) -> list:
    from nuka.editor import connect, spawn
    if pid is not None:
        with connect(pid, client="action_docs") as editor:
            return editor.actions.list()
    editor = spawn(profile=tempfile.mkdtemp(prefix="nuka-action-docs-"), client="action_docs")
    actions = editor.actions.list()
    editor.call("run", id="app.exit", args={})
    editor.close()
    return actions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pid", type=int)
    parser.add_argument("--out", type=pathlib.Path, default=REPO / "assets/editor/actions.toml")
    opts = parser.parse_args()
    actions = [a for a in listed(opts.pid) if not a.get("owner") or a["owner"].startswith("nuka.")]
    kept = tomllib.loads(opts.out.read_text(encoding="utf-8")) if opts.out.exists() else {}
    registered = registrations()
    entries, review = {}, []
    for action in actions:
        if action["id"] in kept:
            entries[action["id"]] = kept[action["id"]]
            continue
        entries[action["id"]], notes = draft(action, registered)
        review.append((action["id"], entries[action["id"]], notes))
    write_catalog(opts.out, [a["id"] for a in actions], entries)
    for aid, entry, notes in review:
        print(f"{aid} [{entry['kind']}] {entry['summary']}")
        for name, text in entry.get("args", {}).items():
            print(f"    {name}: {text}")
        if "returns" in entry:
            print(f"    -> {entry['returns']}")
        for note in notes:
            print(f"    ! {note}")
    dropped = sorted(set(kept) - set(entries))
    print(f"{len(review)} drafted, {len(entries) - len(review)} kept, {len(dropped)} dropped: {', '.join(dropped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
