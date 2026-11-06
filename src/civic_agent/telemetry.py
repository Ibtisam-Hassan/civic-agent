"""Metrics (Prometheus) and traces (OpenTelemetry) built from the audit event stream.

The audit trail is the single source of truth: every event is redacted first, then fanned out to the JSONL
log, the metrics and the trace. Spans carry metadata only (tool, tokens, latency, decision), never user text,
so traces cannot leak personal data even if a redaction pattern misses something.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Iterator

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from prometheus_client import CollectorRegistry, Counter, Histogram, start_http_server

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)
BLOCK_EVENTS = {"tool_blocked", "input_blocked", "output_blocked", "step_limit"}


def tracer_provider(exporter: SpanExporter | None = None) -> TracerProvider:
    """OTLP over HTTP when OTEL_EXPORTER_OTLP_ENDPOINT is set (e.g. Jaeger on :4318); otherwise the given exporter."""
    provider = TracerProvider(resource=Resource.create({"service.name": "civic-agent"}))
    if exporter is None and os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter()
    if exporter is not None:
        provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider


class Telemetry:
    def __init__(self, model: str, registry: CollectorRegistry | None = None, provider: TracerProvider | None = None,
                 price_in: float | None = None, price_out: float | None = None):
        self.model = model
        self.registry = registry or CollectorRegistry()
        self.provider = provider or tracer_provider()
        self.tracer = self.provider.get_tracer("civic_agent")
        self.price_in = price_in if price_in is not None else float(os.getenv("CIVIC_PRICE_IN", "0"))
        self.price_out = price_out if price_out is not None else float(os.getenv("CIVIC_PRICE_OUT", "0"))
        r = self.registry
        self.turns = Counter("civic_turns", "Finished user turns by outcome", ["outcome"], registry=r)
        self.turn_latency = Histogram("civic_turn_latency_seconds", "End-to-end turn latency", buckets=LATENCY_BUCKETS, registry=r)
        self.model_calls = Counter("civic_model_calls", "LLM calls", ["model"], registry=r)
        self.model_latency = Histogram("civic_model_latency_seconds", "LLM call latency", ["model"], buckets=LATENCY_BUCKETS, registry=r)
        self.tokens = Counter("civic_tokens", "Tokens used", ["model", "direction"], registry=r)
        self.cost = Counter("civic_cost_usd", "Estimated spend in USD", ["model"], registry=r)
        self.tool_calls = Counter("civic_tool_calls", "Executed tool calls", ["tool"], registry=r)
        self.tool_latency = Histogram("civic_tool_latency_seconds", "Tool call latency", ["tool"], buckets=LATENCY_BUCKETS, registry=r)
        self.quarantined = Counter("civic_injection_quarantined", "Tool results with instruction-like text removed", ["tool"], registry=r)
        self.blocked = Counter("civic_blocked", "Actions stopped by a control", ["control", "reason"], registry=r)
        self.approvals = Counter("civic_approvals", "Human approval decisions", ["tool", "decision"], registry=r)
        self._root: trace.Span | None = None
        self._outcome = "answered"

    def serve(self, port: int) -> None:
        start_http_server(port, registry=self.registry)

    @contextmanager
    def turn(self, role: str) -> Iterator[None]:
        t0 = time.perf_counter()
        self._outcome = "answered"
        with self.tracer.start_as_current_span("agent.turn", attributes={"civic.role": role, "civic.model": self.model}) as span:
            self._root = span
            try:
                yield
            except Exception:
                self._outcome = "error"
                raise
            finally:
                span.set_attribute("civic.outcome", self._outcome)
                self.turns.labels(self._outcome).inc()
                self.turn_latency.observe(time.perf_counter() - t0)
                self._root = None

    def _child(self, name: str, ts: float, latency_ms: float, attributes: dict) -> None:
        end = int(ts * 1e9)
        start = end - int(latency_ms * 1e6)
        ctx = trace.set_span_in_context(self._root) if self._root else None
        span = self.tracer.start_span(name, context=ctx, start_time=start, attributes=attributes)
        span.end(end_time=end)

    def on_event(self, e: dict) -> None:
        """Audit sink. Receives events that are already redacted."""
        kind = e["event"]
        if kind == "model_call":
            tin, tout = e["input_tokens"], e["output_tokens"]
            self.model_calls.labels(self.model).inc()
            self.model_latency.labels(self.model).observe(e["latency_ms"] / 1000)
            self.tokens.labels(self.model, "input").inc(tin)
            self.tokens.labels(self.model, "output").inc(tout)
            self.cost.labels(self.model).inc((tin * self.price_in + tout * self.price_out) / 1e6)
            self._child("agent.model_call", e["ts"], e["latency_ms"], {
                "gen_ai.request.model": self.model, "gen_ai.usage.input_tokens": tin,
                "gen_ai.usage.output_tokens": tout, "civic.tool_calls": ",".join(e["tool_calls"])})
        elif kind == "tool_call":
            self.tool_calls.labels(e["tool"]).inc()
            self.tool_latency.labels(e["tool"]).observe(e["latency_ms"] / 1000)
            if e["injection_quarantined"]:
                self.quarantined.labels(e["tool"]).inc()
            self._child("agent.tool_call", e["ts"], e["latency_ms"], {
                "civic.tool": e["tool"], "civic.injection_quarantined": e["injection_quarantined"]})
        elif kind == "approval":
            self.approvals.labels(e["tool"], e["decision"]).inc()
            self._mark(kind, {"civic.tool": e["tool"], "civic.decision": e["decision"]})
            if e["decision"] == "deny":
                self._outcome = "denied"
        elif kind in BLOCK_EVENTS:
            reason = e.get("reason", kind)
            self.blocked.labels(kind, reason).inc()
            self._mark(kind, {"civic.reason": reason})
            self._outcome = "blocked" if kind != "step_limit" else "step_limit"

    def _mark(self, name: str, attributes: dict) -> None:
        if self._root:
            self._root.add_event(name, attributes)

    def flush(self) -> None:
        self.provider.force_flush()
