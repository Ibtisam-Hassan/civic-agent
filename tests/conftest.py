from __future__ import annotations

import os

# Tests use the offline hashing embedder: no download, fully deterministic.
os.environ["CIVIC_EMBEDDER"] = "hash"

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from civic_agent.audit import Audit
from civic_agent.connect import mcp_tools
from civic_agent.graph import build_agent, run_turn
from civic_agent.policy import Session


class FixedModel(BaseChatModel):
    """Emits one fixed tool call until it sees a tool result, then echoes that result. Ignores tool binding,
    so it can attempt calls the policy layer must refuse."""

    call: dict

    @property
    def _llm_type(self) -> str:
        return "fixed"

    def bind_tools(self, tools: list[Any], **kw: Any) -> "FixedModel":
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw) -> ChatResult:
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if results and not self.call.get("repeat"):
            msg = AIMessage(content=f"RESULT: {results[-1].content}")
        else:
            msg = AIMessage(content="", tool_calls=[{"name": self.call["name"], "args": self.call["args"],
                                                     "id": f"c{len(results)}"}])
        return ChatResult(generations=[ChatGeneration(message=msg)])


@pytest.fixture
def ask():
    # Each call opens its own MCP session: anyio cancel scopes must close in the task that opened them.
    async def _ask(model, text, role="citizen", user="citizen-1", decision="deny", session=None, tool_list=None):
        audit = Audit()
        async with mcp_tools("memory") as tools:
            app = build_agent(model, tool_list or tools, session or Session(user, role), audit)
            out = await run_turn(app, "t", text, lambda payload: decision)
        return out, audit
    return _ask
