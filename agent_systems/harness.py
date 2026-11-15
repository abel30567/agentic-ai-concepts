"""Concept 1 — Harness Engineering.

Give your agent a safe, observable stack so it doesn't spiral out of
control: retries with backoff, sandboxed execution, cost limits, and
graceful degradation when a tool fails.

The harness wraps every tool call. It is deliberately boring: the
difference between a demo and a production agent is mostly this file.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field


class BudgetExceeded(Exception):
    pass


class TransientError(Exception):
    """Retryable failure (network blip, rate limit, flaky tool)."""


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    backoff_seconds: float = 1.0  # doubles each attempt

    def run(self, fn, *args, **kwargs):
        last_exc: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return fn(*args, **kwargs)
            except TransientError as exc:
                last_exc = exc
                if attempt < self.max_attempts:
                    time.sleep(self.backoff_seconds * (2 ** (attempt - 1)))
        raise last_exc  # type: ignore[misc]


@dataclass
class CostBudget:
    """Hard limits. Token counts are estimates (see context.estimate_tokens);
    usd is computed at the configured rates and is a guardrail, not a bill."""

    max_tokens: int = 20_000
    max_tool_calls: int = 25
    max_usd: float = 1.00
    usd_per_million_input: float = 3.00
    usd_per_million_output: float = 15.00
    tokens_used: int = 0
    tool_calls: int = 0

    @property
    def usd_used(self) -> float:
        # conservative: price everything at the output rate
        return self.tokens_used / 1_000_000 * self.usd_per_million_output

    def charge_tokens(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.tokens_used += prompt_tokens + completion_tokens
        self._check()

    def charge_tool_call(self) -> None:
        self.tool_calls += 1
        self._check()

    def _check(self) -> None:
        if self.tokens_used > self.max_tokens:
            raise BudgetExceeded(f"token budget exceeded: "
                                 f"{self.tokens_used} > {self.max_tokens}")
        if self.tool_calls > self.max_tool_calls:
            raise BudgetExceeded(f"tool-call budget exceeded: "
                                 f"{self.tool_calls} > {self.max_tool_calls}")
        if self.usd_used > self.max_usd:
            raise BudgetExceeded(f"cost budget exceeded: "
                                 f"${self.usd_used:.2f} > ${self.max_usd:.2f}")


def run_sandboxed(code: str, timeout_seconds: int = 10) -> dict:
    """Run Python in a subprocess with a timeout, in a fresh temp dir.

    This is a *demo* sandbox: it isolates the filesystem and kills
    runaway code, but it is NOT a security boundary. In production use a
    real sandbox (gVisor, Firecracker, a locked-down container).
    """
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout_seconds, cwd=tmp,
        )
        return {"ok": proc.returncode == 0,
                "stdout": proc.stdout, "stderr": proc.stderr,
                "returncode": proc.returncode}


@dataclass
class Harness:
    """Wraps tool execution: retry transient failures, enforce the cost
    budget, and degrade gracefully — a failed tool returns an observation
    the agent can act on instead of crashing the run."""

    retry: RetryPolicy = field(default_factory=RetryPolicy)
    budget: CostBudget = field(default_factory=CostBudget)
    fallback: object = None  # optional callable(tool_name, args) -> result

    def execute(self, tool, args: dict) -> dict:
        self.budget.charge_tool_call()
        try:
            return self.retry.run(tool.run, args)
        except Exception as exc:  # noqa: BLE001 — degradation, not a crash
            if self.fallback is not None:
                try:
                    result = self.fallback(tool.name, args)  # type: ignore[operator]
                    return {"ok": True, "result": result,
                            "degraded": True,
                            "note": f"primary tool failed ({exc}); used fallback"}
                except Exception:
                    pass
            return {"ok": False,
                    "error": f"Tool '{tool.name}' failed after "
                             f"{self.retry.max_attempts} attempts: {exc}. "
                             "Try a different approach."}
