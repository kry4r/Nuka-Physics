"""The agents smoke's agent: one phase of calls against the editor named on the command line,
written to <out>/agents_<phase>.json. agents_user.py plays the user around it.

    python agents_client.py <profile> <pid> <phase> <out>

When a phase has asked the user something it creates <out>/agents_asked, then waits for the answer.
"""

import json
import os
import sys
import time

from nuka.editor import ActionError, RemoteError, connect
from nuka.mcp import png


def attempt(call):
    """[ok, value] of a call, the error's text in place of the value when it fails."""
    try:
        return [True, call()]
    except (ActionError, RemoteError) as error:
        return [False, str(error)]


def shot(editor, out: str, name: str) -> None:
    taken = editor.screenshot()
    _, _, data = png.ppm_to_png(taken["path"], None, 1600)
    with open(os.path.join(out, name + ".png"), "wb") as f:
        f.write(data)


def asked(out: str) -> None:
    open(os.path.join(out, "agents_asked"), "w").close()


def settled(editor, request: int, timeout_s: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while True:
        found = next(r for r in editor.actions.run("agent.proposals") if r["id"] == request)
        if found["state"] != "pending" or time.monotonic() > deadline:
            return found
        time.sleep(0.1)


def full(editor, plan: dict, out: str) -> dict:
    ops = [{"target": plan["collision"], "field": "shape.friction_mu", "after": 0.8}]
    seen = {"proposed": editor.actions.run("agent.propose", label="Lower ground friction", ops=ops,
                                           note="The robot should slide a little.")}
    seen["denied"] = attempt(lambda: editor.actions.run("scene.spawn", kind="box"))
    seen["mode"] = attempt(lambda: editor.actions.run("agent.mode", mode="off"))
    seen["accept"] = attempt(lambda: editor.actions.run("agent.accept", id=seen["proposed"]["request"]))
    shot(editor, out, "agents_full")
    asked(out)
    seen["settled"] = settled(editor, seen["proposed"]["request"])
    return seen


def read(editor, plan: dict, out: str) -> dict:
    return {"nodes": attempt(lambda: len(editor.actions.run("scene.nodes"))),
            "view": attempt(lambda: editor.actions.run("view.frame_all")),
            "shot": attempt(lambda: shot(editor, out, "agents_read")),
            "edit": attempt(lambda: editor.actions.run("edit.rename", path=plan["root"], name="floor")),
            "propose": attempt(lambda: editor.actions.run("agent.propose", label="Nothing", ops=[])),
            "mode": attempt(lambda: editor.actions.run("agent.mode", mode="full"))}


def propose(editor, plan: dict, out: str) -> dict:
    seen = {"sim": attempt(lambda: editor.actions.run("sim.step", steps=2)),
            "script": editor.actions.run("script.eval", code="print(6 * 7)"),
            "rename": editor.actions.run("edit.rename", path=plan["root"], name="floor")}
    shot(editor, out, "agents_propose")
    asked(out)
    seen["script_done"] = settled(editor, seen["script"]["request"])
    seen["rename_done"] = settled(editor, seen["rename"]["request"])
    seq = seen["script_done"]["value"]["seq"]
    seen["output"] = editor.wait("script.result", {"done": True}, 30.0, seq=seq)["value"]["output"]
    seen["again"] = editor.actions.run("script.eval", code="print(1)")
    return seen


def disconnect(editor, plan: dict, out: str) -> dict:
    asked(out)
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        ok, value = attempt(editor.document)
        if not ok:
            return {"dropped": value}
        time.sleep(0.1)
    return {"dropped": None}


PHASES = {"full": full, "read": read, "propose": propose, "disconnect": disconnect}


def main(profile: str, pid: int, phase: str, out: str) -> None:
    with open(os.path.join(out, "agents_plan.json"), encoding="utf-8") as f:
        plan = json.load(f)
    with connect(pid, client="smoke-agent", profile=profile) as editor:
        seen = PHASES[phase](editor, plan, out)
    with open(os.path.join(out, f"agents_{phase}.json"), "w", encoding="utf-8") as f:
        json.dump(seen, f)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4])
