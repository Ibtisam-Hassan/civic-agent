"""Run the evaluation set and write a report. Usage: python -m civic_agent.eval --model scripted|groq|openrouter

Per task: success (every check passes), tool-call accuracy (the model's tool calls match the expected set),
steps (model calls), latency, tokens and cost. Cost uses CIVIC_PRICE_IN / CIVIC_PRICE_OUT (USD per million
tokens) because free tiers cost nothing but a production deployment would not.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import Counter
from importlib import resources
from pathlib import Path

from civic_agent.audit import Audit
from civic_agent.connect import mcp_tools
from civic_agent.graph import build_agent, run_turn
from civic_agent.models import get_model, load_dotenv
from civic_agent.policy import Session
from civic_agent.telemetry import Telemetry
from langchain_core.messages import ToolMessage


def load_tasks() -> list[dict]:
    return json.loads(resources.files("civic_agent.data").joinpath("eval_tasks.json").read_text(encoding="utf-8"))


def check(task: dict, answer: str, messages: list, audit: Audit, called: list[str], approvals: list) -> list[str]:
    """Return the list of failed checks (empty = success)."""
    fails = []
    retrieved = " ".join(str(m.content) for m in messages if isinstance(m, ToolMessage))
    if task.get("source") and task["source"] not in answer:
        fails.append(f"missing citation {task['source']}")
    for s in task.get("contains", []):
        if s.lower() not in answer.lower():
            fails.append(f"answer lacks '{s}'")
    for s in task.get("not_contains", []):
        if s.lower() in answer.lower():
            fails.append(f"answer contains forbidden '{s}'")
    for src in task.get("forbidden_sources", []):
        if src in answer or src in retrieved:
            fails.append(f"forbidden source {src} reached the model")
    log = json.dumps(audit.events, ensure_ascii=False)
    for s in task.get("audit_not_contains", []):
        if s in log:
            fails.append(f"audit log holds '{s}'")
    if "approvals" in task and len(approvals) != task["approvals"]:
        fails.append(f"expected {task['approvals']} approval requests, got {len(approvals)}")
    if Counter(called) != Counter(task["tools"]):
        fails.append(f"tool calls {called} != expected {task['tools']}")
    return fails


async def run(model_name: str, transport: str = "stdio", tasks: list[dict] | None = None,
              telemetry: Telemetry | None = None) -> dict:
    tasks = tasks or load_tasks()
    price_in = float(os.getenv("CIVIC_PRICE_IN", "0"))
    price_out = float(os.getenv("CIVIC_PRICE_OUT", "0"))
    rows = []
    async with mcp_tools(transport) as tools:
        model = get_model(model_name)
        for task in tasks:
            audit = Audit(sinks=[telemetry.on_event] if telemetry else None)
            app = build_agent(model, tools, Session(task["user"], task["role"]), audit)
            decision = task.get("approve", "deny")
            t0 = time.perf_counter()
            out = await run_turn(app, task["id"], task["text"], lambda payload: decision, telemetry, task["role"])
            latency = (time.perf_counter() - t0) * 1000
            calls = audit.of("model_call")
            called = [name for c in calls for name in c["tool_calls"]]
            tin = sum(c["input_tokens"] for c in calls)
            tout = sum(c["output_tokens"] for c in calls)
            fails = check(task, out["answer"], out["messages"], audit, called, out["approvals"])
            rows.append({
                "id": task["id"], "lang": task["lang"], "success": not fails, "fails": fails,
                "tool_accuracy": Counter(called) == Counter(task["tools"]), "steps": len(calls),
                "latency_ms": round(latency), "input_tokens": tin, "output_tokens": tout,
                "cost_usd": (tin * price_in + tout * price_out) / 1e6, "answer": out["answer"],
            })
    n = len(rows)
    summary = {
        "model": model_name, "tasks": n,
        "task_success": sum(r["success"] for r in rows) / n,
        "tool_call_accuracy": sum(r["tool_accuracy"] for r in rows) / n,
        "avg_steps": sum(r["steps"] for r in rows) / n,
        "p50_latency_ms": sorted(r["latency_ms"] for r in rows)[n // 2],
        "total_tokens": sum(r["input_tokens"] + r["output_tokens"] for r in rows),
        "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 6),
        "urdu_success": _rate([r for r in rows if r["lang"] == "ur"]),
    }
    return {"summary": summary, "rows": rows}


def _rate(rows: list[dict]) -> float:
    return sum(r["success"] for r in rows) / len(rows) if rows else 0.0


def report_markdown(result: dict) -> str:
    s = result["summary"]
    lines = [
        f"# Evaluation: model `{s['model']}`", "",
        "| Metric | Value |", "|---|---|",
        f"| Tasks | {s['tasks']} |",
        f"| Task success | {s['task_success']:.0%} |",
        f"| Tool-call accuracy | {s['tool_call_accuracy']:.0%} |",
        f"| Urdu task success | {s['urdu_success']:.0%} |",
        f"| Average steps (model calls) | {s['avg_steps']:.2f} |",
        f"| Median latency | {s['p50_latency_ms']} ms |",
        f"| Tokens (all tasks) | {s['total_tokens']} |",
        f"| Cost (all tasks) | ${s['total_cost_usd']:.4f} |", "",
        "| Task | Lang | OK | Tools OK | Steps | Latency ms | Failures |", "|---|---|---|---|---|---|---|",
    ]
    for r in result["rows"]:
        lines.append(f"| {r['id']} | {r['lang']} | {'yes' if r['success'] else 'NO'} | {'yes' if r['tool_accuracy'] else 'no'} | "
                     f"{r['steps']} | {r['latency_ms']} | {'; '.join(r['fails'])} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="scripted")
    ap.add_argument("--transport", default="stdio", choices=["stdio", "memory"])
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    result = asyncio.run(run(args.model, args.transport))
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / f"eval-{args.model}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    (out / f"eval-{args.model}.md").write_text(report_markdown(result))
    print(report_markdown(result))


def load_main() -> None:
    """Replay the evaluation set in a loop with metrics and traces on: traffic for the monitoring stack."""
    load_dotenv()
    ap = argparse.ArgumentParser(description=load_main.__doc__)
    ap.add_argument("--model", default="scripted")
    ap.add_argument("--rounds", type=int, default=0, help="0 = run until Ctrl-C")
    ap.add_argument("--pause", type=float, default=2.0, help="seconds between rounds")
    ap.add_argument("--metrics-port", type=int, default=9464)
    args = ap.parse_args()
    telemetry = Telemetry(args.model)
    telemetry.serve(args.metrics_port)
    print(f"metrics on http://localhost:{args.metrics_port}/metrics")

    async def loop() -> None:
        i = 0
        while not args.rounds or i < args.rounds:
            s = (await run(args.model, telemetry=telemetry))["summary"]
            i += 1
            print(f"round {i}: success {s['task_success']:.0%}, p50 {s['p50_latency_ms']} ms")
            await asyncio.sleep(args.pause)

    try:
        asyncio.run(loop())
    except KeyboardInterrupt:
        pass
    finally:
        telemetry.flush()


if __name__ == "__main__":
    main()
