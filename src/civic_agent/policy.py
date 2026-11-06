"""Who may call which tool, which calls need a human, and the text checks on everything that flows through."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Least privilege: a role sees only the tools it needs. The model is never shown a tool outside this list.
ROLE_TOOLS = {
    "citizen": {"search_documents", "get_request_status", "submit_service_request"},
    "caseworker": {"search_documents", "get_request_status"},
}
WRITE_TOOLS = {"submit_service_request"}

# Identity comes from the authenticated session, never from the model. These arguments are hidden from the
# model's tool schemas and overwritten on every call.
INJECTED_ARGS = {"role", "requester_id"}

INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) instructions",
    r"disregard (the |your )?(system|previous) (prompt|instructions)",
    r"reveal (your |the )?system prompt",
    r"(system|developer) instruction\s*:",
    r"you are now\b",
    r"call (the )?\w+ (tool|function)?\s*with",
    r"immediately call \w+",
]
_INJECTION = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

# 13-digit ID numbers (with or without dashes), phone numbers, email addresses.
_PII = [
    (re.compile(r"\b\d{5}-?\d{7}-?\d\b"), "[ID-REDACTED]"),
    (re.compile(r"(?<!\w)(?:\+92|0)3\d{2}[- ]?\d{7}\b"), "[PHONE-REDACTED]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "[EMAIL-REDACTED]"),
]

# Planted in the system prompt; if it ever appears in an answer, the prompt has leaked.
CANARY = "CANARY-7f3a91"


@dataclass
class Limits:
    max_steps: int = 8
    tool_timeout_s: float = 10.0
    max_repeat_calls: int = 2


@dataclass
class Session:
    user_id: str
    role: str
    limits: Limits = field(default_factory=Limits)


def allowed_tools(role: str) -> set[str]:
    return ROLE_TOOLS.get(role, set())


def looks_like_injection(text: str) -> bool:
    return bool(_INJECTION.search(text))


def quarantine(text: str) -> str:
    """Neutralise instruction-like sentences inside retrieved content, keep the rest as plain data."""
    sentences = re.split(r"(?<=[.!?۔])\s+", text)
    kept = [s if not looks_like_injection(s) else "[REMOVED: instruction-like text in a document]" for s in sentences]
    return " ".join(kept)


def redact(text: str) -> str:
    for pattern, repl in _PII:
        text = pattern.sub(repl, text)
    return text


def leaks_system_prompt(text: str) -> bool:
    return CANARY in text
