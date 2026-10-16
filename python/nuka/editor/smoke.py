"""Editor scenarios ``{"name", "launch", "steps"}``: ``python -m nuka.editor.smoke s.json --out DIR``.

Strings take ``${out}``, ``${scenario_dir}`` and bound values; exit 0 only when every step passed."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
import traceback
from typing import Optional, Sequence

from .app import EditorApp

_REF = re.compile(r"\$\{([^}]+)\}")
_TIMEOUT_S = 120.0
_FRAMES_PER_POLL = 4


def _lookup(ref: str, variables: dict):
    head, *keys = ref.split(".")
    value = variables[head]
    for key in keys:
        value = value[key]
    return value


def _resolve(value, variables: dict):
    """Substitute ``${...}`` references; a string that is one reference takes the value's type."""
    if isinstance(value, str):
        whole = _REF.fullmatch(value)
        if whole:
            return _lookup(whole.group(1), variables)
        return _REF.sub(lambda m: str(_lookup(m.group(1), variables)), value)
    if isinstance(value, list):
        return [_resolve(v, variables) for v in value]
    if isinstance(value, dict):
        return {k: _resolve(v, variables) for k, v in value.items()}
    return value


def _to_png(ppm: pathlib.Path) -> pathlib.Path:
    try:
        from PIL import Image
    except ImportError:
        return ppm
    png = ppm.with_suffix(".png")
    with Image.open(ppm) as image:
        image.save(png)
    ppm.unlink()
    return png


class _Run:
    def __init__(self, app: EditorApp, out: pathlib.Path, variables: dict):
        self.app = app
        self.out = out
        self.vars = variables
        self.marks = {}
        self.screenshots = []
        self.handlers = {
            "action": self.action, "tick": self.tick, "frames": self.frames, "mark": self.mark,
            "expect": self.expect, "undo": self.undo, "save_reopen": self.save_reopen,
            "screenshot": self.screenshot, "python": self.python,
        }

    def run(self, action_id: str, args: Optional[dict] = None) -> dict:
        return self.app._run(action_id, args or {}, "smoke")

    def stats(self) -> dict:
        return self.run("view.stats")["value"]

    def pump_until(self, done, what: str) -> None:
        deadline = time.monotonic() + _TIMEOUT_S
        while not done():
            if time.monotonic() > deadline or not self.app.running:
                raise TimeoutError(f"timed out waiting for {what}")
            for _ in range(_FRAMES_PER_POLL):
                self.app.update(True)

    def hashes(self) -> dict:
        # A frame queues the edit's pose writes; the next boundary applies them.
        self.app.update(False)
        self.app.update(False)
        world = self.app.world
        return {"state": world.state_hash() if world is not None else None, "doc": self.app.document()["hash"]}

    def action(self, step: dict):
        """``{"action": id, "args", "bind": name, "expect_error"}``; bind keeps the reply value."""
        reply = self.run(step["action"], _resolve(step.get("args", {}), self.vars))
        if "bind" in step:
            self.vars[step["bind"]] = reply["value"]
        return reply["ok"] != step.get("expect_error", False), {"reply": reply}

    def tick(self, step: dict):
        """``{"tick": n}``: sim.step when paused, the transport when playing."""
        n = int(step["tick"])
        before = self.stats()
        if not before["playing"]:
            reply = self.run("sim.step", {"steps": n})
            if not reply["ok"]:
                return False, {"reply": reply}
        self.pump_until(lambda: self.stats()["tick"] >= before["tick"] + n, f"{n} ticks")
        return True, {"from": before["tick"], "to": self.stats()["tick"]}

    def frames(self, step: dict):
        """``{"frames": n}`` draws n frames."""
        for _ in range(int(step["frames"])):
            self.app.update(True)
        return True, {}

    def mark(self, step: dict):
        """``{"mark": name, "differs_from": mark}`` records both hashes; both must differ from mark's."""
        hashes = self.marks[step["mark"]] = self.hashes()
        if "differs_from" not in step:
            return True, {"hashes": hashes}
        other = self.marks[step["differs_from"]]
        return all(hashes[k] != other[k] for k in hashes), {"hashes": hashes, "differs_from": other}

    def expect(self, step: dict):
        """``{"expect": {key: value}, "same_as": mark}`` checks document() keys and the mark's hashes."""
        detail = {}
        if "same_as" in step:
            detail = {"hashes": self.hashes(), "same_as": self.marks[step["same_as"]]}
        doc = self.app.document()
        want = _resolve(step["expect"], self.vars)
        diffs = {k: {"want": v, "got": doc.get(k)} for k, v in want.items() if doc.get(k) != v}
        if detail and detail["hashes"] != detail["same_as"]:
            diffs["hashes"] = {"want": detail["same_as"], "got": detail["hashes"]}
        detail["diffs"] = diffs
        return not diffs, detail

    def undo(self, step: dict):
        """``{"undo": k}`` undoes k edits."""
        done = [self.app.undo() for _ in range(int(step["undo"]))]
        return all(done), {"undone": sum(done)}

    def save_reopen(self, step: dict):
        """``{"save_reopen": path}``: the document hash must survive a save and reopen."""
        path = _resolve(step["save_reopen"], self.vars)
        before = self.hashes()
        saved = self.run("file.save", {"path": path})
        opened = self.run("file.open", {"path": path})
        after = self.hashes()
        ok = saved["ok"] and opened["ok"] and before["doc"] == after["doc"]
        return ok, {"before": before, "after": after, "save": saved, "open": opened}

    def screenshot(self, step: dict):
        """``{"screenshot": name}`` writes ``<out>/<name>.png`` (PPM without Pillow)."""
        ppm = self.out / f"{_resolve(step['screenshot'], self.vars)}.ppm"
        ppm.unlink(missing_ok=True)
        reply = self.run("view.screenshot", {"path": str(ppm)})
        self.pump_until(ppm.exists, ppm.name)
        path = _to_png(ppm)
        self.screenshots.append(str(path))
        return reply["ok"], {"path": str(path)}

    def python(self, step: dict):
        """``{"python": code}`` runs code with ``app`` bound, such as panels built through app.ui."""
        exec(compile(_resolve(step["python"], self.vars), "<smoke>", "exec"), {"app": self.app})
        return True, {}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m nuka.editor.smoke")
    parser.add_argument("scenario")
    parser.add_argument("--out", required=True)
    parser.add_argument("--offscreen", action="store_true")
    opts = parser.parse_args(argv)
    scenario_path = pathlib.Path(opts.scenario).resolve()
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    out = pathlib.Path(opts.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    variables = {"out": out.as_posix(), "scenario_dir": scenario_path.parent.as_posix()}
    launch = ["--empty"] + (["--offscreen"] if opts.offscreen else []) + _resolve(scenario.get("launch", []), variables)

    app = EditorApp(launch)
    run = _Run(app, out, variables)
    steps, crash, stats = [], None, {}
    try:
        for index, step in enumerate(scenario["steps"]):
            kind = next(k for k in run.handlers if k in step)
            t0 = time.perf_counter()
            ok, detail = run.handlers[kind](step)
            ms = (time.perf_counter() - t0) * 1000.0
            steps.append({"index": index, "type": kind, "step": step, "ok": bool(ok), "ms": round(ms, 1), **detail})
            print(f"[smoke] {index:02d} {kind:<11} {'ok  ' if ok else 'FAIL'} {json.dumps(step)}", flush=True)
        stats = run.stats()
    except Exception:
        crash = traceback.format_exc()
        print(crash, file=sys.stderr, flush=True)
    log = app.actions.log()
    app.close()

    passed = crash is None and len(steps) == len(scenario["steps"]) and all(s["ok"] for s in steps)
    report = {
        "scenario": scenario["name"], "passed": passed, "crash": crash, "steps": steps, "marks": run.marks,
        "frame_ms_p50": stats.get("frame_ms_p50"), "frame_ms_p95": stats.get("frame_ms_p95"),
        "snapshot_age_ms_p95": stats.get("age_ms_p95"), "stats": stats, "extension_errors": [],
        "screenshots": run.screenshots, "action_log": [e for e in log if e["id"] != "view.stats"],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"SMOKE {scenario['name']} {'PASS' if passed else 'FAIL'} "
          f"(p50 {report['frame_ms_p50']} ms, p95 {report['frame_ms_p95']} ms) -> {out / 'report.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
