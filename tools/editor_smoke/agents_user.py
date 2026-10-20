"""The agents smoke's user, run inside the editor: sets the agent access, runs agents_client.py
phase by phase and answers its requests the way the Agents panel does, through the editor's actions.

    runpy.run_path("agents_user.py", {"app": app, "scenario_dir": ..., "out": ...})
"""

import json
import os
import subprocess
import sys
import time

from nuka.editor import editors


def phase(name: str, answer=None) -> dict:
    """Runs one client phase; answer() plays the user once the client says it asked."""
    flag = os.path.join(out, "agents_asked")
    if os.path.exists(flag):
        os.remove(flag)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    client = subprocess.Popen([sys.executable, os.path.join(scenario_dir, "agents_client.py"), profile,
                               str(os.getpid()), name, out], env=env)
    deadline = time.monotonic() + 120.0
    while client.poll() is None:
        assert time.monotonic() < deadline, f"the {name} phase still runs"
        app.update(True)
        if answer is not None and os.path.exists(flag):
            # The panel and the status bar show the waiting requests before the user answers.
            for _ in range(5):
                app.update(True)
            seen[name + "_sessions"] = [s["pid"] for s in app.rpc.sessions()] == [client.pid]
            seen[name + "_clients"] = app.document()["agent"]["clients"]
            answer()
            answer = None
    assert client.returncode == 0, (name, client.returncode)
    with open(os.path.join(out, f"agents_{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def pending() -> dict:
    return {r["action"]: r for r in app.actions.run("agent.proposals") if r["state"] == "pending"}


profile = os.path.dirname(os.path.dirname(app.rpc.file))
seen = {}
# An agent finds the friction field and its target the way the Properties panel lists them.
root = next(n["path"] for n in app.actions.run("scene.nodes") if n["name"] == "ground")
friction = [(g["targets"][0], f) for g in app.actions.run("scene.fields", path=root) for f in g["fields"]
            if f["field"] == "shape.friction_mu"]
assert friction and abs(friction[0][1]["value"] - 1.0) < 1e-6, friction
collision = friction[0][0]
world = app.actions.run("scene.fields", path="")
assert [g["kind"] for g in world] == ["world"] and world[0]["targets"] == [""], world
plan = {"collision": collision, "root": root}
with open(os.path.join(out, "agents_plan.json"), "w", encoding="utf-8") as f:
    json.dump(plan, f)
app.actions.run("agent.mode", mode="full", allow=[], deny=["scene.*"])
assert app.rpc.listening and os.path.exists(app.rpc.file)


# full: edits run, but a proposal still waits for the user; the deny list and user-only actions hold.
def accept() -> None:
    proposal = pending()["edit.apply"]
    seen["changes"] = proposal["changes"]
    seen["hash_before"] = app.document()["hash"]
    seen["accepted"] = app.actions.run("agent.accept", id=proposal["id"])["state"]
    seen["hash_after"] = app.document()["hash"]


full = phase("full", accept)
print("AGENTS full", json.dumps(full)[:1500])
assert full["proposed"]["state"] == "pending" and full["proposed"]["kind"] == "proposal", full
assert not full["denied"][0] and "deny list" in full["denied"][1], full
assert not full["mode"][0] and "only the user runs agent.mode" in full["mode"][1], full
assert not full["accept"][0] and "only the user runs agent.accept" in full["accept"][1], full
assert seen["full_sessions"] and seen["full_clients"] == 1, seen
assert seen["changes"] == [{"target": collision, "field": "shape.friction_mu", "before": "1", "after": "0.8"}], seen
assert seen["accepted"] == "accepted" and full["settled"]["state"] == "accepted", (seen, full)
assert seen["hash_after"] != seen["hash_before"], seen
applied = [e for e in app.actions.log(0) if e["id"] == "edit.apply" and e["source"] == "agent:smoke-agent"]
assert applied[-1]["args"]["label"] == "Agent: Lower ground friction", applied[-1:]
app.actions.run("edit.undo")
assert app.document()["hash"] == seen["hash_before"], app.document()["hash"]

# read: reads and the view pass, edits and proposals are refused.
app.actions.run("agent.mode", mode="read", deny=[])
read = phase("read")
print("AGENTS read", read)
assert read["nodes"][0] and read["nodes"][1] > 0 and read["view"][0] and read["shot"][0], read
assert not read["edit"][0] and "agent access is read" in read["edit"][1], read
assert not read["propose"][0] and "proposals are off" in read["propose"][1], read
assert not read["mode"][0] and "only the user runs agent.mode" in read["mode"][1], read

# propose, in Chinese: the simulation runs, a script waits for approval, an edit for acceptance.
app.actions.run("agent.mode", mode="propose")
app.actions.run("view.language", locale="zh-CN")


def answer() -> None:
    asked = pending()
    seen["asked"] = sorted(asked)
    seen["approved"] = app.actions.run("agent.approve", request=asked["script.eval"]["id"], always=True)["state"]
    seen["rejected"] = app.actions.run("agent.reject", id=asked["edit.rename"]["id"], reason="keep the name")["state"]


propose = phase("propose", answer)
seen["missing"] = app.actions.run("view.missing_strings")
app.actions.run("view.language", locale="en")
print("AGENTS propose", json.dumps(propose)[:1500], seen)
assert propose["sim"][0], propose
assert propose["script"]["kind"] == "approval" and propose["rename"]["kind"] == "proposal", propose
assert seen["asked"] == ["edit.rename", "script.eval"] and seen["propose_clients"] == 1, seen
assert seen["approved"] == "approved" and propose["script_done"]["state"] == "approved", (seen, propose)
assert "42" in propose["output"] and "seq" in propose["again"], propose
assert propose["rename_done"]["state"] == "rejected" and propose["rename_done"]["error"] == "keep the name", propose
assert any(n["path"] == plan["root"] for n in app.actions.run("scene.nodes")), plan
assert app.actions.run("agent.mode")["allow"] == ["script.eval"], app.actions.run("agent.mode")
assert seen["missing"] == [], seen["missing"]

# disconnect drops the client; off closes the endpoint, and turning access back on reopens it.
app.actions.run("agent.mode", mode="full", allow=[])


def drop() -> None:
    seen["dropped"] = app.actions.run("agent.disconnect")


gone = phase("disconnect", drop)
assert seen["dropped"] == {"dropped": 1, "mode": "full"} and "closed the connection" in gone["dropped"], (seen, gone)
endpoint = app.rpc.file
assert app.actions.run("agent.disconnect", off=True)["mode"] == "off", app.actions.run("agent.mode")
assert not app.rpc.listening and not os.path.exists(endpoint), endpoint
assert all(e["pid"] != os.getpid() for e in editors(profile)), editors(profile)
app.actions.run("agent.mode", mode="full")
assert app.rpc.listening and [e["pid"] for e in editors(profile)].count(os.getpid()) == 1, editors(profile)

# Every agent call is in today's audit log.
with open(os.path.join(profile, "logs", "agent-" + time.strftime("%Y%m%d") + ".jsonl"), encoding="utf-8") as f:
    audit = [json.loads(line) for line in f if line.strip()]
outcomes = {e["outcome"].split()[0] for e in audit if e["client"] == "smoke-agent"}
print("AGENTS audit", len(audit), sorted(outcomes))
assert {"ran", "refused", "proposal", "approval", "accepted", "approved"} <= outcomes, outcomes
