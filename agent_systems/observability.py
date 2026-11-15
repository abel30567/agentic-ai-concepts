"""Concept 10 — Observability & Tracing.

See every step your agent takes so you can debug fast and ship with
confidence instead of guessing.

The tracer records structured events (JSONL): prompts, tool calls,
model responses, errors, approvals, and timing. ``summary()`` gives the
numbers that matter per run: latency, success rate, cost, token usage.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field


@dataclass
class Tracer:
    """One trace per run. Events are kept in memory and optionally
    written to a JSONL file."""

    run_id: str = "run"
    log_path: str | None = None
    events: list[dict] = field(default_factory=list)
    _start: float = field(default_factory=time.time)
    _spans: dict = field(default_factory=dict)

    def event(self, kind: str, **data) -> dict:
        evt = {"run_id": self.run_id, "t": round(time.time() - self._start, 3),
               "kind": kind, **data}
        self.events.append(evt)
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(evt) + "\n")
        return evt

    def start_span(self, name: str) -> None:
        self._spans[name] = time.time()

    def end_span(self, name: str) -> float:
        started = self._spans.pop(name, None)
        elapsed = time.time() - started if started else 0.0
        self.event("span", name=name, seconds=round(elapsed, 3))
        return elapsed

    def summary(self) -> dict:
        tool_calls = [e for e in self.events if e["kind"] == "tool_call"]
        errors = [e for e in self.events if e["kind"] == "error"]
        tokens = sum(e.get("tokens", 0) for e in self.events
                     if e["kind"] == "llm_response")
        return {
            "run_id": self.run_id,
            "duration_seconds": round(time.time() - self._start, 2),
            "steps": len(self.events),
            "tool_calls": len(tool_calls),
            "errors": len(errors),
            "success_rate": round(
                1 - len(errors) / max(1, len(tool_calls)), 3),
            "tokens_used": tokens,
            "est_cost_usd": round(tokens / 1_000_000 * 15.0, 4),
        }
