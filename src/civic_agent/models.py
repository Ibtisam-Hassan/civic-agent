"""Chat models: a real open-weight model behind an OpenAI-compatible API, or offline stand-ins.

- scripted: a deterministic, well-behaved policy. Lets the whole pipeline (MCP, approval, evaluation) run
  with no key and no network, so CI and reviewers can reproduce every number.
- gullible: obeys any instruction it reads, including ones planted in documents. The safety tests run
  against it to prove the controls hold even when the model does not.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "meta-llama/llama-3.3-70b-instruct:free"),
}

SERVICE_WORDS = {
    "ID-RENEW": ["id card", "identity card", "شناختی"],
    "BIRTH-REG": ["birth", "پیدائش"],
    "DOMICILE": ["domicile"],
    "BIZ-LICENSE": ["business licence", "business license", "shop licence", "shop license"],
    "PROP-TAX": ["property tax", "پراپرٹی"],
    "DL-RENEW": ["driving licence", "driving license"],
    "PENSION-ENQ": ["pension"],
    "COMPLAINT": ["complaint", "complain"],
}
SUBMIT_WORDS = ["submit", "for me", "درخواست جمع"]
REQUEST_ID = re.compile(r"\bREQ-\d+\b", re.IGNORECASE)


def load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


def _estimate_tokens(messages: list[BaseMessage], reply: str) -> dict[str, int]:
    """Offline models have no tokenizer; ~4 characters per token keeps cost tracking exercised."""
    prompt = sum(len(str(m.content)) for m in messages) // 4
    out = max(len(reply) // 4, 1)
    return {"input_tokens": prompt, "output_tokens": out, "total_tokens": prompt + out}


def service_from(text: str) -> str | None:
    low = text.lower()
    for code, words in SERVICE_WORDS.items():
        if any(w in low for w in words):
            return code
    return None


class ScriptedModel(BaseChatModel):
    tool_names: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "ScriptedModel":
        names = [t["function"]["name"] if isinstance(t, dict) else t.name for t in tools]
        return self.__class__(tool_names=names)

    def _call(self, name: str, args: dict[str, Any]) -> AIMessage | None:
        if name not in self.tool_names:
            return None
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{name}"}])

    def decide(self, messages: list[BaseMessage]) -> AIMessage:
        last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        question = str(messages[last_human].content)
        results = [m for m in messages[last_human + 1 :] if isinstance(m, ToolMessage)]
        if not results:
            if m := REQUEST_ID.search(question):
                return self._call("get_request_status", {"request_id": m.group(0).upper()}) or self._refuse()
            if any(w in question.lower() for w in SUBMIT_WORDS) and (code := service_from(question)):
                return self._call("submit_service_request", {"service": code, "details": question}) or self._refuse()
            return self._call("search_documents", {"query": question}) or self._refuse()
        return AIMessage(content=self._answer(results[-1]))

    def _refuse(self) -> AIMessage:
        return AIMessage(content="I can't do that with the access your account has.")

    def _answer(self, result: ToolMessage) -> str:
        text = str(result.content)
        if text.startswith("DENIED") or text.startswith("BLOCKED"):
            return f"I did not do that. {text}"
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return text
        if isinstance(data, dict) and "error" in data:
            return f"Sorry: {data['error']}"
        if result.name == "search_documents":
            if not data:
                return "I could not find an official document that answers this. Please contact the Civic Services Office."
            top = data[0]
            body = re.sub(r"^(SYNTHETIC DOCUMENT\.|مصنوعی دستاویز۔)\s*", "", top["text"])
            summary = " ".join(re.split(r"(?<=[.۔])\s+", body)[:3])
            return f"{summary} [source: {top['id']}]"
        if result.name == "submit_service_request":
            return f"Your request {data['request_id']} for {data['service']} has been received."
        if result.name == "get_request_status":
            return f"Request {data['request_id']} ({data['service']}): {data['status']}."
        return text

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any) -> ChatResult:
        msg = self.decide(messages)
        msg.usage_metadata = _estimate_tokens(messages, str(msg.content) + json.dumps(msg.tool_calls))
        return ChatResult(generations=[ChatGeneration(message=msg)])


class GullibleModel(ScriptedModel):
    """Follows instructions found anywhere in its context: the worst case the controls must survive."""

    @property
    def _llm_type(self) -> str:
        return "gullible"

    def decide(self, messages: list[BaseMessage]) -> AIMessage:
        context = " ".join(str(m.content) for m in messages if not isinstance(m, SystemMessage))
        already = {tc["name"] for m in messages if isinstance(m, AIMessage) for tc in m.tool_calls}
        if "submit_service_request" in context and "submit_service_request" not in already:
            m = re.search(r"service (\w[\w-]*)", context)
            code = m.group(1) if m else "COMPLAINT"
            return AIMessage(content="", tool_calls=[{"name": "submit_service_request",
                                                      "args": {"service": code, "details": "auto-filed"},
                                                      "id": "call_injected"}])
        if re.search(r"reveal (your |the )?system prompt", context, re.IGNORECASE):
            system = next((str(m.content) for m in messages if isinstance(m, SystemMessage)), "")
            return AIMessage(content=f"Sure, my instructions are: {system}")
        return super().decide(messages)


def get_model(name: str) -> BaseChatModel:
    if name == "scripted":
        return ScriptedModel()
    if name == "gullible":
        return GullibleModel()
    if name in PROVIDERS:
        from langchain_openai import ChatOpenAI

        load_dotenv()
        base_url, key_var, default = PROVIDERS[name]
        key = os.getenv(key_var)
        if not key:
            raise SystemExit(f"{key_var} is not set. Put it in .env (see .env.example).")
        return ChatOpenAI(model=os.getenv("CIVIC_MODEL", default), base_url=base_url, api_key=key,
                          temperature=0, timeout=60, max_retries=3)
    raise SystemExit(f"Unknown model '{name}'. Use scripted, gullible, groq or openrouter.")
