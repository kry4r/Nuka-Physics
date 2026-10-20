"""nuka.mcp -- the Nuka editor as a Model Context Protocol server, for AI agents on this machine.

    python -m nuka.mcp [--profile DIR] [--pid PID]      # also nuka-mcp, or NukaEditor.exe --mcp
    python -m nuka.mcp install-skill [--dest DIR]       # the nuka-editor skill, into ~/.claude/skills

speaks MCP on stdio and drives the editor open on the profile (see ``nuka.editor.connect``);
``nuka.mcp.client.McpClient`` is a matching client for scripts. The skill (``skill/nuka-editor``)
teaches an agent to work the editor through these tools.
"""

from .server import Server, install_skill, main

__all__ = ["Server", "install_skill", "main"]
