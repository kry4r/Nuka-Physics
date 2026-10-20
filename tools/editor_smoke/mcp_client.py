"""The mcp smoke's other program: runs ``python -m nuka.mcp`` against the editor named on the
command line, once statelessly (2026-07-28) and once in a 2025-11-25 session, and writes what it
saw as JSON.

    python mcp_client.py <profile> <pid> <asset to import> <out.json>
"""

import json
import os
import struct
import sys
import zlib

from nuka.mcp.client import McpClient, McpError

VERSION_KEY = "io.modelcontextprotocol/protocolVersion"


def png_info(data: bytes) -> dict:
    """Width, height and whether the pixel rows decode to 8-bit RGB."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n", data[:8]
    width, height = struct.unpack(">II", data[16:24])
    at, idat = 8, b""
    while at < len(data):
        length, kind = struct.unpack(">I4s", data[at:at + 8])
        if kind == b"IDAT":
            idat += data[at + 8:at + 8 + length]
        at += 12 + length
    return {"width": width, "height": height, "rows_ok": len(zlib.decompress(idat)) == height * (1 + width * 3)}


def code_of(call) -> int:
    try:
        call()
    except McpError as error:
        return error.code
    return 0


def main(profile: str, pid: int, asset: str, out: str) -> None:
    env = dict(os.environ, NUKA_EDITOR_PROFILE=profile, NUKA_EDITOR_PID=str(pid),
               PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    command = [sys.executable, "-m", "nuka.mcp"]
    seen = {}
    mcp = McpClient(command, env=env)
    seen["discover"] = mcp.server["supportedVersions"]
    listed = mcp.request("tools/list")
    seen["result_type"] = listed["resultType"]
    seen["tools"] = [t["name"] for t in listed["tools"]]
    seen["schemas"] = all(t["inputSchema"]["type"] == "object" for t in listed["tools"])
    seen["connected"] = mcp.call("editor_status")["connected"]
    seen["find"] = [a["id"] for a in mcp.call("actions_find", query="import", kind="edit")["actions"]]
    seen["help_args"] = sorted(mcp.call("action_help", id="file.import")["args"])
    mcp.call("action_run", id="file.new", args={"template": "ground", "discard": True})
    imported = mcp.call("action_run", id="file.import", args={"path": asset, "mode": "add", "name": "go2"})["value"]
    seen["import"] = imported
    seen["tree"] = mcp.call("scene_tree", under=imported["focus"], depth=1)["count"]
    seen["tree_all"] = mcp.call("scene_tree", under=imported["focus"])["count"]
    seen["dirty"] = mcp.call("document_get")["dirty"]
    bad = mcp.call_tool("action_run", {"id": "no.such_action"})
    seen["bad_action"] = {"isError": bad["isError"], "text": bad["content"][0]["text"]}
    seen["bad_tool"] = code_of(lambda: mcp.request("tools/call", {"name": "no_such_tool"}))
    seen["bad_method"] = code_of(lambda: mcp.request("no/such_method"))
    seen["bad_version"] = code_of(lambda: mcp.request("tools/list", {"_meta": {VERSION_KEY: "1999-01-01"}}))
    shot = mcp.call_tool("screenshot", {"viewport": True, "max_width": 400})
    seen["shot"] = png_info(mcp.images(shot)[0])
    seen["shot_reported"] = [shot["structuredContent"]["width"], shot["structuredContent"]["height"]]
    seen["eval"] = mcp.call("python_eval", code="print(6 * 7)")["output"]
    batch = mcp.call("actions_batch", steps=[{"id": "sim.step"}, {"id": "no.such_action"}, {"id": "sim.step"}])
    seen["batch"] = [r["ok"] for r in batch["results"]]
    good = mcp.call("scenario_run", steps=[{"action": "timeline.status"}, {"expect": {"dirty": True}}])
    failing = mcp.call("scenario_run", steps=[{"action": "no.such_action"}, {"expect": {"dirty": False}}])
    seen["scenario"] = [good["passed"], failing["passed"], [s["ok"] for s in failing["steps"]]]
    seen["resources"] = [r["uri"] for r in mcp.list_resources()]
    seen["catalog"] = len(json.loads(mcp.read_resource("nuka://catalog")))
    seen["exit"] = [mcp.close()]

    with McpClient(command, env=env, legacy="2025-11-25") as legacy:
        seen["legacy_version"] = legacy.server["protocolVersion"]
        seen["legacy_instructions"] = bool(legacy.server["instructions"])
        seen["legacy_tools"] = [t["name"] for t in legacy.list_tools()]
        seen["legacy_connected"] = legacy.call("editor_status")["connected"]
        seen["ping"] = isinstance(legacy.request("ping"), dict)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(seen, f)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4])
