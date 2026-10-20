"""nuka.mcp -- the Nuka editor as a Model Context Protocol server, for AI agents on this machine.

    python -m nuka.mcp [--profile DIR] [--pid PID]      # also nuka-mcp, or NukaEditor.exe --mcp

speaks MCP on stdio and drives the editor open on the profile (see ``nuka.editor.connect``);
``nuka.mcp.client.McpClient`` is a matching client for scripts.
"""

from .server import Server, main

__all__ = ["Server", "main"]
