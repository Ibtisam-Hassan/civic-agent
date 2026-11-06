"""Break each safety control on purpose and confirm the test suite notices. Usage: uv run python scripts/mutation_check.py

A test that still passes with its control removed proves nothing; every mutant below must make pytest fail.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "civic_agent"

MUTANTS = [
    ("approval gate skipped", "graph.py", 'elif name in policy.WRITE_TOOLS:', 'elif False:'),
    ("role allowlist ignored", "graph.py", "by_name = {t.name: t for t in tools if t.name in allowed}", "by_name = {t.name: t for t in tools}"),
    ("model may set identity", "graph.py", 'args = {**call["args"], "role": session.role, "requester_id": session.user_id}',
     'args = {"role": session.role, "requester_id": session.user_id, **call["args"]}'),
    ("loop guard removed", "graph.py", "elif history.count(key) > limits.max_repeat_calls:", "elif False:"),
    ("step limit removed", "graph.py", 'if state.get("steps", 0) >= limits.max_steps:', "if False:"),
    ("prompt-leak check removed", "graph.py", "if policy.leaks_system_prompt(text):", "if False:"),
    ("input guard removed", "graph.py", "if policy.looks_like_injection(question):", "if False:"),
    ("retrieved text not quarantined", "graph.py", "text, flagged = _quarantine_result(name, raw)", "text, flagged = raw, False"),
    ("tool timeout removed", "graph.py", "timeout=limits.tool_timeout_s", "timeout=None"),
    ("audit not redacted", "audit.py", "json.loads(redact(json.dumps(", "json.loads((json.dumps("),
    ("staff documents visible to all", "retrieval.py", "visible = [i for i, d in enumerate(self.docs) if d.access in allowed]",
     "visible = list(range(len(self.docs)))"),
    ("server trusts any role for status", "mcp_server.py", 'if req is None or (role != "caseworker" and req["owner"] != requester_id):',
     "if req is None:"),
]


def main() -> int:
    survived = []
    for label, fname, old, new in MUTANTS:
        path = SRC / fname
        original = path.read_text()
        if old not in original:
            print(f"STALE  {label}: anchor not found in {fname}")
            survived.append(label)
            continue
        try:
            path.write_text(original.replace(old, new, 1))
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"],
                               capture_output=True, text=True, timeout=300)
        finally:
            path.write_text(original)
        caught = r.returncode != 0
        print(f"{'caught ' if caught else 'SURVIVED'} {label}")
        if not caught:
            survived.append(label)
    print(f"\n{len(MUTANTS) - len(survived)}/{len(MUTANTS)} mutants caught")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
