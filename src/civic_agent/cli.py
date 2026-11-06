"""Chat with the agent in a terminal. Usage: civic-agent --model scripted --role citizen --user citizen-1"""

from __future__ import annotations

import argparse
import asyncio
import json

from civic_agent.audit import Audit
from civic_agent.connect import mcp_tools
from civic_agent.graph import build_agent, run_turn
from civic_agent.models import get_model, load_dotenv
from civic_agent.policy import Session


def ask_human(payload: dict) -> str:
    print(f"\n  APPROVAL NEEDED: {payload['action']} {json.dumps(payload['args'], ensure_ascii=False)}")
    return "approve" if input("  approve? [y/N] ").strip().lower() == "y" else "deny"


async def chat(model: str, role: str, user: str, audit_path: str) -> None:
    async with mcp_tools("stdio") as tools:
        app = build_agent(get_model(model), tools, Session(user, role), Audit(audit_path))
        print(f"civic-agent ({model}, {role} {user}). Synthetic data only. Ctrl-D to quit.")
        while True:
            try:
                text = input("\nyou> ").strip()
            except EOFError:
                return
            if text:
                out = await run_turn(app, "cli", text, ask_human)
                print(f"agent> {out['answer']}")


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="scripted", help="scripted | gullible | groq | openrouter")
    ap.add_argument("--role", default="citizen", choices=["citizen", "caseworker"])
    ap.add_argument("--user", default="citizen-1")
    ap.add_argument("--audit", default="audit.jsonl", help="where the redacted audit trail is appended")
    args = ap.parse_args()
    asyncio.run(chat(args.model, args.role, args.user, args.audit))


if __name__ == "__main__":
    main()
