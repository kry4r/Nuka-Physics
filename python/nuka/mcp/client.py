"""nuka.mcp.client -- a small Model Context Protocol client over stdio, for scripts and smokes.

    with McpClient([sys.executable, "-m", "nuka.mcp"]) as mcp:
        names = [tool["name"] for tool in mcp.list_tools()]
        mcp.call("action_run", id="file.new", args={"template": "ground", "discard": True})

Each request carries protocol version 2026-07-28 in its ``_meta`` and no session is kept; with
``legacy="2025-11-25"`` (or another earlier revision) the client opens a session with
``initialize`` first, as clients of those revisions do.
"""

from __future__ import annotations

import base64
import json
import queue
import subprocess
import threading
from typing import List, Optional, Sequence

MODERN = "2026-07-28"
CLIENT_INFO = {"name": "nuka.mcp.client", "version": "1.0.0"}


class McpError(RuntimeError):
    """The server answered a request with a JSON-RPC error."""

    def __init__(self, code: int, message: str, data=None):
        super().__init__(f"{message} ({code})")
        self.code, self.data = code, data


class ToolError(RuntimeError):
    """A tool ran and reported failure; the message is its text."""


class McpClient:
    """One server process, spoken to one request at a time."""

    def __init__(self, command: Sequence[str], env: Optional[dict] = None, legacy: Optional[str] = None,
                 timeout_s: float = 120.0):
        self.legacy = legacy
        self.timeout_s = timeout_s
        self._process = subprocess.Popen(list(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        self._lines: "queue.Queue[Optional[dict]]" = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        self._next = 0
        if legacy:
            self.server = self.request("initialize", {"protocolVersion": legacy, "capabilities": {},
                                                      "clientInfo": CLIENT_INFO})
            self.notify("notifications/initialized")
        else:
            self.server = self.request("server/discover")

    def _read(self) -> None:
        for line in self._process.stdout:
            self._lines.put(json.loads(line))
        self._lines.put(None)

    def _send(self, message: dict) -> None:
        self._process.stdin.write(json.dumps(message, ensure_ascii=False).encode("utf-8") + b"\n")
        self._process.stdin.flush()

    def request(self, method: str, params: Optional[dict] = None):
        """The result of one request; raises McpError with the server's error."""
        self._next += 1
        params = dict(params or {})
        if not self.legacy:
            params.setdefault("_meta", {}).setdefault("io.modelcontextprotocol/protocolVersion", MODERN)
        self._send({"jsonrpc": "2.0", "id": self._next, "method": method, "params": params})
        try:
            reply = self._lines.get(timeout=self.timeout_s)
        except queue.Empty:
            raise McpError(-1, f"no reply to {method} within {self.timeout_s:g} s") from None
        if reply is None:
            raise McpError(-1, f"the server exited with code {self._process.wait()}")
        if "error" in reply:
            error = reply["error"]
            raise McpError(error["code"], error["message"], error.get("data"))
        return reply["result"]

    def notify(self, method: str, params: Optional[dict] = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def list_tools(self) -> List[dict]:
        return self.request("tools/list")["tools"]

    def call_tool(self, name: str, arguments: Optional[dict] = None) -> dict:
        """The tool's whole result: content, structuredContent and isError."""
        return self.request("tools/call", {"name": name, "arguments": arguments or {}})

    def call(self, name: str, **arguments):
        """The tool's structured value; raises ToolError when the tool reports failure."""
        result = self.call_tool(name, arguments)
        if result["isError"]:
            raise ToolError(" ".join(c["text"] for c in result["content"] if c["type"] == "text"))
        return result["structuredContent"]

    @staticmethod
    def images(result: dict) -> List[bytes]:
        """The PNG images in a tool result's content."""
        return [base64.b64decode(c["data"]) for c in result["content"] if c["type"] == "image"]

    def list_resources(self) -> List[dict]:
        return self.request("resources/list")["resources"]

    def read_resource(self, uri: str) -> str:
        return self.request("resources/read", {"uri": uri})["contents"][0]["text"]

    def close(self) -> int:
        """Ends the server by closing its input; returns its exit code."""
        self._process.stdin.close()
        try:
            return self._process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            self._process.kill()
            return self._process.wait()

    def __enter__(self) -> "McpClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
