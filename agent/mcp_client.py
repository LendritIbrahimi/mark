"""MCP client wrapper for stdio-based MCP servers."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)),
)


class MCPClient:
    """Manages a connection to a single MCP server."""

    def __init__(
            self, name: str, session: ClientSession,
    ) -> None:
        self.name = name
        self._session = session
        self._tools: list[str] = []
        self._tool_schemas: dict[str, dict] = {}

    async def initialize(self) -> None:
        await self._session.initialize()
        tools_response = await self._session.list_tools()
        self._tools = [
            t.name for t in tools_response.tools
        ]
        self._tool_schemas = {
            t.name: {
                "description": t.description or "",
                "inputSchema": (
                        getattr(t, "inputSchema", {}) or {}
                ),
            }
            for t in tools_response.tools
        }

    @property
    def tools(self) -> list[str]:
        return list(self._tools)

    @property
    def tool_schemas(self) -> dict[str, dict]:
        return dict(self._tool_schemas)

    async def call_tool(
            self,
            name: str,
            arguments: dict[str, Any] | None = None,
            timeout: float = 30.0,
    ) -> Any:
        try:
            result = await asyncio.wait_for(
                self._session.call_tool(
                    name, arguments or {},
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"MCP tool {name} timed out "
                f"after {timeout:.0f}s",
            ) from None

        if getattr(result, "isError", False):
            error_text = (
                str(result.content)
                if result.content
                else "Unknown MCP error"
            )
            raise RuntimeError(
                f"MCP tool {name} failed: {error_text}",
            )

        text_parts = []
        for block in result.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)

        raw = "\n".join(text_parts) if text_parts else ""

        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return raw


_CLEANUP_EXC = (
    OSError, BrokenPipeError, ConnectionError,
    EOFError, asyncio.CancelledError,
)


def _is_cleanup_noise(exc: BaseException) -> bool:
    """True for exceptions that are just MCP subprocess teardown noise."""
    return isinstance(exc, _CLEANUP_EXC)


@asynccontextmanager
async def connect_mcp(
        name: str,
        command: str,
        args: list[str],
) -> AsyncGenerator[MCPClient, None]:
    env = os.environ.copy()
    env["PYTHONPATH"] = (
            PROJECT_ROOT
            + os.pathsep
            + env.get("PYTHONPATH", "")
    )

    server_params = StdioServerParameters(
        command=command, args=args, env=env,
    )
    log_path = os.path.join(
        PROJECT_ROOT, f".mcp_{name}.log",
    )
    errlog = open(log_path, "w")

    try:
        async with stdio_client(
                server_params, errlog=errlog,
        ) as (read, write):
            async with ClientSession(
                    read, write,
            ) as session:
                client = MCPClient(name, session)
                await client.initialize()
                yield client
    except BaseExceptionGroup as eg:
        real, cleanup = eg.split(
            lambda e: not _is_cleanup_noise(e),
        )
        if cleanup:
            logger.debug(
                "MCP '%s' cleanup errors suppressed: %s",
                name, cleanup,
            )
        if real:
            _log_server_stderr(name, log_path, errlog)
            raise real
    except BaseException:
        _log_server_stderr(name, log_path, errlog)
        raise
    finally:
        errlog.close()


def _log_server_stderr(
        name: str, log_path: str, errlog: Any,
) -> None:
    errlog.flush()
    try:
        with open(log_path, "r") as f:
            stderr_out = f.read().strip()
        if stderr_out:
            logger.error(
                "MCP server '%s' stderr:\n%s",
                name, stderr_out,
            )
    except OSError:
        pass
