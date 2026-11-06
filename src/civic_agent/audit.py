"""Append-only audit trail. Every event is redacted before it is stored, so the log itself never holds PII."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from civic_agent.policy import redact


class Audit:
    def __init__(self, path: str | Path | None = None):
        self.events: list[dict[str, Any]] = []
        self.path = Path(path) if path else None

    def log(self, kind: str, **fields: Any) -> None:
        event = json.loads(redact(json.dumps({"ts": round(time.time(), 3), "event": kind, **fields}, ensure_ascii=False)))
        self.events.append(event)
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["event"] == kind]
