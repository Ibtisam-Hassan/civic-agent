"""Open an MCP session to the tool server: over stdio (a real subprocess) or in memory (tests)."""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client
from mcp.shared.memory import create_connected_server_and_client_session


@asynccontextmanager
async def mcp_tools(transport: str = "stdio", server=None) -> AsyncIterator[list[BaseTool]]:
    if transport == "memory":
        from civic_agent.mcp_server import build_server

        async with create_connected_server_and_client_session(server or build_server()) as session:
            yield await load_mcp_tools(session)
        return
    # The SDK starts servers with a minimal environment on purpose; forward only this app's own settings.
    env = {**get_default_environment(), **{k: v for k, v in os.environ.items() if k.startswith("CIVIC_")}}
    params = StdioServerParameters(command=sys.executable, args=["-m", "civic_agent.mcp_server"], env=env)
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        yield await load_mcp_tools(session)
