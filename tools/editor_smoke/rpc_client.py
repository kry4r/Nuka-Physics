"""The rpc smoke's other program: drives the editor named on the command line through
nuka.editor.connect and writes what it saw as JSON.

    python rpc_client.py <profile> <pid> <out.json>
"""

import json
import sys
import threading
from multiprocessing import AuthenticationError
from multiprocessing.connection import Client

from nuka.editor import ActionError, RemoteError, connect, editors
from nuka.editor.rpc import family
from nuka.mcp import png


def main(profile: str, pid: int, out: str) -> None:
    seen = {"listed": [e["pid"] for e in editors(profile)]}
    with connect(pid, client="smoke", profile=profile) as editor:
        seen["hello"] = editor.hello
        actions = editor.actions.list()
        seen["actions"] = len(actions)
        seen["has_spawn"] = any(a["id"] == "scene.spawn" for a in actions)
        before = editor.document()["revision"]
        editor.actions.run("scene.spawn", kind="box", pose={"pos": [0.0, 0.0, 0.5], "quat": [1.0, 0.0, 0.0, 0.0]})
        seen["revised"] = editor.document()["revision"] > before
        spawns = [e for e in editor.actions.log(0) if e["id"] == "scene.spawn"]
        seen["source"] = spawns[-1]["source"]
        for name, call in (("bad_action", lambda: editor.actions.run("no.such_action")),
                           ("bad_op", lambda: editor.call("no_such_op")),
                           ("missing_arg", lambda: editor.call("run"))):
            try:
                call()
                seen[name] = None
            except (ActionError, RemoteError) as error:
                seen[name] = str(error)
        shot = editor.screenshot()
        width, height, data = png.ppm_to_png(shot["path"], shot["viewport"], 640)
        seen["shot"] = {"viewport": shot["viewport"], "width": width, "height": height, "png": len(data)}
        seen["wait"] = editor.wait("script.console", {"dropped": 0}, 10.0, since=0)["met"]
        seen["console"] = "next" in editor.console(0)

    # Two clients at once: each reply must be the one its own call asked for.
    wrong = []

    def hammer(kind: str) -> None:
        with connect(pid, client=f"smoke-{kind}", profile=profile) as editor:
            for _ in range(40):
                value = editor.document() if kind == "doc" else editor.actions.run("timeline.status")
                if ("revision" in value) != (kind == "doc"):
                    wrong.append(kind)

    threads = [threading.Thread(target=hammer, args=(kind,)) for kind in ("doc", "tick")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    seen["crossed"] = len(wrong)

    info = next(e for e in editors(profile) if e["pid"] == pid)
    try:
        Client(info["address"], family(), authkey=b"0" * 32).close()
        seen["bad_key"] = "accepted"
    except AuthenticationError:
        seen["bad_key"] = "refused"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(seen, f)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
