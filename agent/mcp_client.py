"""Thin MCP client wrapper for connecting to stdio-based MCP servers."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class MCPClient:
    """Manages a connection to a single MCP server subprocess."""

    def __init__(self, name: str, session: ClientSession) -> None:
        self.name = name
        self._session = session
        self._tools: list[str] = []

    async def initialize(self) -> None:
        """Initialize the session and discover available tools."""
        await self._session.initialize()
        tools_response = await self._session.list_tools()
        self._tools = [t.name for t in tools_response.tools]
        logger.debug("MCP '%s': connected, tools=%s", self.name, self._tools)

    @property
    def tools(self) -> list[str]:
        return list(self._tools)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> Any:
        """Call a tool on the MCP server and return the parsed result.

        Raises RuntimeError if the tool call reports an error or times out.
        """
        logger.debug("MCP '%s': calling %s(%s)", self.name, name, arguments)
        try:
            result = await asyncio.wait_for(
                self._session.call_tool(name, arguments or {}),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            msg = f"MCP tool {name} timed out after {timeout:.0f}s"
            logger.error("MCP '%s': %s", self.name, msg)
            raise RuntimeError(msg) from None

        if getattr(result, "isError", False):
            error_text = str(result.content) if result.content else "Unknown MCP error"
            logger.error("MCP '%s': tool %s error: %s", self.name, name, error_text)
            raise RuntimeError(f"MCP tool {name} failed: {error_text}")

        text_parts = []
        for block in result.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)

        raw = "\n".join(text_parts) if text_parts else ""

        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return raw


@asynccontextmanager
async def connect_mcp(
    name: str,
    command: str,
    args: list[str],
) -> AsyncGenerator[MCPClient, None]:
    """Context manager that starts an MCP server subprocess and yields a connected client."""
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")

    server_params = StdioServerParameters(
        command=command,
        args=args,
        env=env,
    )
    logger.debug("Starting MCP server '%s': %s %s", name, command, " ".join(args))

    quiet = os.environ.get("MARK_LOG_LEVEL", "WARNING") != "DEBUG"
    errlog = open(os.devnull, "w") if quiet else sys.stderr

    try:
        async with stdio_client(server_params, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                client = MCPClient(name, session)
                await client.initialize()
                yield client
        logger.debug("MCP server '%s' disconnected", name)
    finally:
        if errlog is not sys.stderr:
            errlog.close()
