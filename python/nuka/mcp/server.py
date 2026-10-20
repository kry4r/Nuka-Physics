"""nuka.mcp.server -- the editor's tools and resources over the Model Context Protocol, on stdio.

Messages are JSON-RPC 2.0, one per line. A request whose ``_meta`` names a protocol version is
served statelessly (2026-07-28); ``initialize`` starts a session of an earlier revision. Tools
reach the editor through nuka.editor.remote and connect at the first call that needs one.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ..editor.app import ActionError
from ..editor.remote import RemoteEditor, RemoteError, connect, editors, spawn
from . import png

MODERN = "2026-07-28"
LEGACY = ["2025-11-25", "2025-06-18", "2025-03-26"]
SERVER_INFO = {"name": "nuka-editor", "title": "Nuka Editor", "version": "1.0.0"}
UNSUPPORTED_VERSION = -32022
# The nuka-editor skill; its reference pages are served as nuka://guide/<name>.
SKILL = Path(__file__).resolve().parent / "skill" / "nuka-editor"
GUIDES = SKILL / "references"
INSTRUCTIONS = (
    "Drives the Nuka physics editor open on this machine. Begin with editor_status, or editor_launch when "
    "none runs. Every editor feature is an action: find one with actions_find, read its arguments with "
    "action_help, run it with action_run. An action's kind says what it touches: read, view, edit, sim, "
    "file, process or app. Check outcomes with document_get, scene_tree and screenshot; scenario_run runs a "
    "list of checks. Edits are undoable (action edit.undo). The nuka://guide/* resources hold workflows. "
    "The user sets what agents may do (document_get's agent.mode): full runs everything; propose turns edits "
    "into proposals and asks before file, process and app actions; read allows read and view actions only. "
    "A call that waits for the user returns {request, state: pending}; follow it with proposal_get. "
    "edit_propose offers an edit for the user to review in any mode but read."
)
KINDS = ["read", "view", "edit", "sim", "file", "process", "app"]


def _schema(properties: dict, required: Optional[List[str]] = None) -> dict:
    return {"type": "object", "properties": properties, "required": required or [], "additionalProperties": False}


_ARGS = {"type": "object", "description": "The action's arguments, as action_help lists them.", "default": {}}


class Server:
    """One stdio server; ``serve()`` runs until stdin closes."""

    def __init__(self, profile: Optional[str] = None, pid: Optional[int] = None):
        self.profile = profile
        self.pid = pid
        self.editor: Optional[RemoteEditor] = None
        self.tools: List[dict] = []
        self._calls: Dict[str, Callable[[dict], object]] = {}
        self._tool("editor_status", "List the editors running on this machine and show the connected one's "
                   "document: path, revision, dirty, selection, playing, and agent access (mode, pending "
                   "requests).", _schema({}), self._status, read=True)
        self._tool("editor_launch", "Start an editor window (optionally on a scene file) and connect to it.",
                   _schema({"scene": {"type": "string", "description": "A scene to open: .nks, .usd*, .urdf, "
                                                                        ".xml (MJCF)."},
                            "wait_s": {"type": "number", "default": 60}}), self._launch)
        self._tool("actions_find", "Search the editor's actions (every menu item, button and command is one) by "
                   "words in their id, label or summary; each comes with its kind and summary.",
                   _schema({"query": {"type": "string", "description": "Words that must all appear."},
                            "category": {"type": "string", "description": "file, edit, scene, sim, robot, view, "
                                                                          "train, ..."},
                            "kind": {"type": "string", "enum": KINDS},
                            "limit": {"type": "integer", "default": 40}}), self._find, read=True)
        self._tool("action_help", "Describe one action: what it does, its arguments, what it returns, an example "
                   "and its kind.",
                   _schema({"id": {"type": "string"}}, ["id"]), self._help, read=True)
        self._tool("action_run", "Run one editor action with arguments and return its value; a refusal comes "
                   "back as an error with the editor's reason, and one the user must allow first as "
                   "{request, state: pending}.",
                   _schema({"id": {"type": "string"}, "args": _ARGS}, ["id"]), self._run)
        self._tool("actions_batch", "Run several actions in order; stops at the first failure unless told not to.",
                   _schema({"steps": {"type": "array", "items": _schema({"id": {"type": "string"}, "args": _ARGS},
                                                                        ["id"])},
                            "stop_on_error": {"type": "boolean", "default": True}}, ["steps"]), self._batch)
        self._tool("wait_for", "Run a read action until its value has every key/value of `until` (for example "
                   "train.status until {alive: false}).",
                   _schema({"id": {"type": "string"}, "args": _ARGS,
                            "until": {"type": "object"}, "timeout_s": {"type": "number", "default": 60}},
                           ["id", "until"]), self._wait, read=True)
        self._tool("document_get", "The open document's state: path, revision, dirty, hash, undo depth, tick, "
                   "selection, playing, last error.", _schema({}), self._document, read=True)
        self._tool("scene_tree", "The scene's nodes as {path, name, kind, pose}: all, those under a path, those of "
                   "one kind (artic, body, static, joint, geom, visual, camera, light, script, group, ...), down to "
                   "`depth` levels. A pose is the world {pos, quat} the node stands at now.",
                   _schema({"under": {"type": "string"}, "kind": {"type": "string"},
                            "depth": {"type": "integer", "description": "Levels below `under`, 1 for its children."}}),
                   self._tree, read=True)
        self._tool("screenshot", "A picture of the editor window, or of its 3D viewport only.",
                   _schema({"viewport": {"type": "boolean", "default": False},
                            "max_width": {"type": "integer", "default": 1280}}), self._screenshot, read=True)
        self._tool("console_read", "The editor console's text after offset `since` (pass the last `next`).",
                   _schema({"since": {"type": "integer", "default": 0}}), self._console, read=True)
        self._tool("action_log", "The actions run so far, by anyone, after sequence number `since`.",
                   _schema({"since": {"type": "integer", "default": 0}}), self._log, read=True)
        self._tool("python_eval", "Run Python in the editor's console (`app`, `world`, `nuka` and `np` bound) and "
                   "return its output. When the user must allow it first it returns {pending: request}; once "
                   "approved, action_run script.result {seq} gives the output.",
                   _schema({"code": {"type": "string"}, "timeout_s": {"type": "number", "default": 60}},
                           ["code"]), self._eval)
        self._tool("edit_propose", "Offer the user an edit to review: they see each field's value before and "
                   "after, then accept (it applies as one undoable step) or reject it. Ops are edit.apply's.",
                   _schema({"label": {"type": "string", "description": "What the edit does, in a few words."},
                            "ops": {"type": "array", "items": _schema(
                                {"target": {"type": "string", "description": "A node path."},
                                 "field": {"type": "string"}, "before": {}, "after": {}},
                                ["target", "field", "after"])},
                            "note": {"type": "string", "description": "Why; shown to the user."}},
                           ["label", "ops"]), self._propose)
        self._tool("proposal_get", "The state of a request waiting for the user (pending, accepted, approved, "
                   "rejected or failed) with its value or error; waits up to wait_s while it is pending. "
                   "Without id, every request the editor holds.",
                   _schema({"id": {"type": "integer"}, "wait_s": {"type": "number", "default": 0}}),
                   self._proposal, read=True)
        self._tool("scenario_run", "Run a list of checks in the editor and report each: "
                   "{action, args, expect_error, expect_value}, {wait, args, until, timeout_s}, "
                   "{expect: {document key: value}}, {screenshot: name}.",
                   _schema({"steps": {"type": "array", "items": {"type": "object"}},
                            "stop_on_error": {"type": "boolean", "default": False}}, ["steps"]), self._scenario)

    def _tool(self, name: str, description: str, schema: dict, fn: Callable[[dict], object],
              read: bool = False) -> None:
        self.tools.append({"name": name, "description": description, "inputSchema": schema,
                           "annotations": {"readOnlyHint": read, "openWorldHint": False}})
        self._calls[name] = fn

    # ---- editor -----------------------------------------------------------------------------

    def connected(self) -> RemoteEditor:
        if self.editor is None or self.editor.closed:
            self.editor = connect(self.pid, client="mcp", profile=self.profile)
        return self.editor

    def _status(self, args: dict) -> dict:
        running = [{"pid": e["pid"], "started": e["started"]} for e in editors(self.profile)]
        # An editor that exited no longer lists itself; its connection is dropped before use.
        if self.editor is not None and self.editor.pid not in {e["pid"] for e in running}:
            self.editor.close()
        if (self.editor is None or self.editor.closed) and running:
            self.connected()
        live = self.editor is not None and not self.editor.closed
        return {"editors": running, "connected": self.editor.pid if live else None,
                "document": self.editor.document() if live else None}

    def _launch(self, args: dict) -> dict:
        self.editor = spawn(args.get("scene"), profile=self.profile, client="mcp",
                            timeout_s=float(args.get("wait_s", 60)))
        return {"connected": self.editor.pid, "document": self.editor.document()}

    def _find(self, args: dict) -> dict:
        words = str(args.get("query", "")).lower().split()
        category, kind = args.get("category"), args.get("kind")
        found = []
        for a in self.connected().actions.list():
            text = " ".join(str(a.get(k, "")) for k in ("id", "label", "summary")).lower()
            if all(w in text for w in words) and (not category or a["category"] == category) and \
                    (not kind or a["kind"] == kind):
                found.append({k: a[k] for k in ("id", "label", "category", "enabled", "summary", "kind") if k in a})
        limit = int(args.get("limit", 40))
        return {"count": len(found), "actions": found[:limit]}

    def _help(self, args: dict) -> dict:
        return self.connected().actions.run("action.help", id=args["id"])

    def _run(self, args: dict) -> dict:
        reply = self.connected().call("run", id=args["id"], args=args.get("args") or {})
        if not reply["ok"]:
            raise RemoteError(f"{args['id']}: {reply['error']}")
        return {"value": reply["value"]}

    def _batch(self, args: dict) -> dict:
        results = []
        for step in args["steps"]:
            reply = self.connected().call("run", id=step["id"], args=step.get("args") or {})
            results.append({"id": step["id"], "ok": reply["ok"], "value": reply["value"], "error": reply["error"]})
            if not reply["ok"] and args.get("stop_on_error", True):
                break
        return {"ok": all(r["ok"] for r in results), "results": results}

    def _wait(self, args: dict) -> dict:
        return self.connected().wait(args["id"], args["until"], float(args.get("timeout_s", 60)),
                                     **(args.get("args") or {}))

    def _document(self, args: dict) -> dict:
        return self.connected().document()

    def _tree(self, args: dict) -> dict:
        query = {k: args[k] for k in ("under", "kind") if args.get(k)}
        nodes = self.connected().actions.run("scene.nodes", **query)
        if args.get("depth") is not None:
            base = len([p for p in str(args.get("under", "")).split("/") if p])
            nodes = [n for n in nodes if len([p for p in n["path"].split("/") if p]) - base <= int(args["depth"])]
        return {"count": len(nodes), "nodes": nodes}

    def _screenshot(self, args: dict) -> dict:
        shot = self.connected().screenshot()
        rect = shot["viewport"] if args.get("viewport") else None
        width, height, data = png.ppm_to_png(shot["path"], rect, int(args.get("max_width", 1280)))
        return {"path": shot["path"], "width": width, "height": height, "_png": data}

    def _console(self, args: dict) -> dict:
        return self.connected().console(int(args.get("since", 0)))

    def _log(self, args: dict) -> dict:
        return {"entries": self.connected().actions.log(int(args.get("since", 0)))}

    def _eval(self, args: dict) -> dict:
        editor = self.connected()
        ran = editor.actions.run("script.eval", code=args["code"])
        if ran.get("state") == "pending":
            return {"pending": ran}
        seq = ran["seq"]
        done = editor.wait("script.result", {"done": True}, float(args.get("timeout_s", 60)), seq=seq)
        if not done["met"]:
            raise RemoteError(f"the code still runs after {args.get('timeout_s', 60)} s (script run {seq})")
        result = done["value"]
        if result["failed"]:
            raise RemoteError(result["output"])
        return {"output": result["output"]}

    def _propose(self, args: dict) -> dict:
        return self.connected().actions.run("agent.propose", label=args["label"], ops=args["ops"],
                                            note=args.get("note", ""))

    def _proposal(self, args: dict) -> dict:
        editor = self.connected()
        deadline = time.monotonic() + float(args.get("wait_s", 0))
        while True:
            requests = editor.actions.run("agent.proposals")
            if args.get("id") is None:
                return {"requests": requests}
            found = [r for r in requests if r["id"] == int(args["id"])]
            if not found:
                raise RemoteError(f"the editor holds no request {args['id']}")
            if found[0]["state"] != "pending" or time.monotonic() >= deadline:
                return found[0]
            time.sleep(0.25)

    def _scenario(self, args: dict) -> dict:
        editor, report = self.connected(), []
        for number, step in enumerate(args["steps"]):
            ok, detail = self._check(editor, step)
            report.append({"step": number, "ok": ok, "detail": detail})
            if not ok and args.get("stop_on_error"):
                break
        return {"passed": all(r["ok"] for r in report) and len(report) == len(args["steps"]), "steps": report}

    def _check(self, editor: RemoteEditor, step: dict):
        if "action" in step:
            reply = editor.call("run", id=step["action"], args=step.get("args") or {})
            if step.get("expect_error"):
                return not reply["ok"], reply
            want = step.get("expect_value") or {}
            value = reply["value"] if isinstance(reply["value"], dict) else {}
            return reply["ok"] and all(value.get(k) == v for k, v in want.items()), reply
        if "wait" in step:
            done = editor.wait(step["wait"], step.get("until") or {}, float(step.get("timeout_s", 60)),
                               **(step.get("args") or {}))
            return done["met"], done
        if "expect" in step:
            document = editor.document()
            wrong = {k: document.get(k) for k, v in step["expect"].items() if document.get(k) != v}
            return not wrong, {"differs": wrong}
        if "screenshot" in step:
            return True, editor.screenshot()
        return False, {"error": "a step has action, wait, expect or screenshot"}

    # ---- protocol ---------------------------------------------------------------------------

    def handle(self, message: dict) -> Optional[dict]:
        """The reply to one message, or None for a notification."""
        if "id" not in message:
            return None
        rid, method, params = message["id"], message.get("method"), message.get("params") or {}
        meta = params.get("_meta") or {}
        requested = meta.get("io.modelcontextprotocol/protocolVersion")
        if requested is not None and requested != MODERN and requested not in LEGACY:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": UNSUPPORTED_VERSION,
                    "message": "Unsupported protocol version",
                    "data": {"supported": [MODERN] + LEGACY, "requested": requested}}}
        methods = {"initialize": self._initialize, "server/discover": self._discover, "ping": lambda p: {},
                   "tools/list": self._tools_list, "tools/call": self._tools_call,
                   "resources/list": self._resources_list, "resources/read": self._resources_read,
                   "resources/templates/list": lambda p: self._cached({"resourceTemplates": []}),
                   "prompts/list": lambda p: self._cached({"prompts": []})}
        if method not in methods:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"no method {method!r}"}}
        try:
            result = methods[method](params)
        except (KeyError, ValueError) as error:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32602, "message": str(error)}}
        result["resultType"] = "complete"
        if requested is not None:
            result.setdefault("_meta", {})["io.modelcontextprotocol/serverInfo"] = SERVER_INFO
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    def _capabilities(self) -> dict:
        return {"tools": {"listChanged": False}, "resources": {"listChanged": False}, "prompts": {}}

    def _initialize(self, params: dict) -> dict:
        asked = params.get("protocolVersion")
        return {"protocolVersion": asked if asked in LEGACY else LEGACY[0], "capabilities": self._capabilities(),
                "serverInfo": SERVER_INFO, "instructions": INSTRUCTIONS}

    def _discover(self, params: dict) -> dict:
        return self._cached({"supportedVersions": [MODERN] + LEGACY, "capabilities": self._capabilities(),
                             "_meta": {"io.modelcontextprotocol/serverInfo": SERVER_INFO},
                             "instructions": INSTRUCTIONS})

    @staticmethod
    def _cached(result: dict, ttl_ms: int = 3600000) -> dict:
        result.update(ttlMs=ttl_ms, cacheScope="private")
        return result

    def _tools_list(self, params: dict) -> dict:
        return self._cached({"tools": self.tools})

    def _tools_call(self, params: dict) -> dict:
        name = params["name"]
        if name not in self._calls:
            raise KeyError(f"no tool {name!r}")
        try:
            value = self._calls[name](params.get("arguments") or {})
        except (RemoteError, ActionError) as error:
            return {"content": [{"type": "text", "text": str(error)}], "isError": True}
        except Exception as error:  # a tool's own failure is the agent's to read, not a dead server
            return {"content": [{"type": "text", "text": f"{type(error).__name__}: {error}"}], "isError": True}
        content = []
        image = value.pop("_png", None) if isinstance(value, dict) else None
        if image is not None:
            content.append({"type": "image", "data": base64.b64encode(image).decode("ascii"),
                            "mimeType": "image/png"})
        content.append({"type": "text", "text": json.dumps(value, ensure_ascii=False)})
        return {"content": content, "structuredContent": value, "isError": False}

    def _resources_list(self, params: dict) -> dict:
        guides = [{"uri": f"nuka://guide/{page.stem}", "name": f"guide-{page.stem}",
                   "title": page.read_text(encoding="utf-8").splitlines()[0].lstrip("# "),
                   "description": "How to work the editor through these tools.", "mimeType": "text/markdown"}
                  for page in sorted(GUIDES.glob("*.md"))]
        return self._cached({"resources": [
            {"uri": "nuka://catalog", "name": "catalog", "title": "Editor actions",
             "description": "Every action: id, label, category, enabled, kind and summary.",
             "mimeType": "application/json"},
            {"uri": "nuka://document", "name": "document", "title": "Open document",
             "description": "The connected editor's document state.", "mimeType": "application/json"},
        ] + guides}, ttl_ms=0)

    def _resources_read(self, params: dict) -> dict:
        uri = params["uri"]
        if uri.startswith("nuka://guide/"):
            page = GUIDES / (uri[len("nuka://guide/"):] + ".md")
            if page.parent != GUIDES or not page.is_file():
                raise ValueError(f"no resource {uri!r}")
            return self._cached({"contents": [{"uri": uri, "mimeType": "text/markdown",
                                               "text": page.read_text(encoding="utf-8")}]})
        sources = {"nuka://catalog": lambda: self.connected().actions.list(),
                   "nuka://document": lambda: self.connected().document()}
        if uri not in sources:
            raise ValueError(f"no resource {uri!r}")
        try:
            text = json.dumps(sources[uri](), ensure_ascii=False)
        except RemoteError as error:
            raise ValueError(str(error)) from None
        return self._cached({"contents": [{"uri": uri, "mimeType": "application/json", "text": text}]}, ttl_ms=0)

    def serve(self, stdin, stdout) -> None:
        for line in stdin:
            if not line.strip():
                continue
            try:
                message = json.loads(line)
            except ValueError as error:
                reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(error)}}
            else:
                reply = self.handle(message)
            if reply is not None:
                stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
                stdout.flush()


def install_skill(dest: str) -> Path:
    """Copies the nuka-editor skill into the skills folder ``dest``; returns the skill's folder there."""
    target = Path(dest).expanduser() / SKILL.name
    shutil.copytree(SKILL, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    return target


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["install-skill"]:
        install = argparse.ArgumentParser(prog="python -m nuka.mcp install-skill",
                                          description="Copy the nuka-editor skill into an agent's skills folder.")
        install.add_argument("--dest", default=os.path.join("~", ".claude", "skills"),
                             help="the skills folder (default ~/.claude/skills)")
        print(install_skill(install.parse_args(argv[1:]).dest))
        return 0
    parser = argparse.ArgumentParser(prog="python -m nuka.mcp", description="MCP server for the Nuka editor.")
    parser.add_argument("--profile", default=os.environ.get("NUKA_EDITOR_PROFILE"))
    parser.add_argument("--pid", type=int, default=int(os.environ.get("NUKA_EDITOR_PID", 0)) or None)
    opts = parser.parse_args(argv)
    # The protocol owns stdout; anything else printed goes to stderr.
    protocol = open(sys.stdout.fileno(), "w", encoding="utf-8", newline="\n", closefd=False)
    sys.stdout = sys.stderr
    stdin = open(sys.stdin.fileno(), "r", encoding="utf-8", closefd=False)
    Server(opts.profile, opts.pid).serve(stdin, protocol)
    return 0
