"""LLM interface (concept: none of the 10 stands alone — the model is the
engine, the harness around it is the product).

Bring your own model by implementing ``LLMClient``: one method,
``complete(messages, tools) -> LLMResponse``.

``MockLLM`` is a deterministic stand-in so the demo and evals run with no
API key. It follows a scripted plan of tool calls, then answers — it is a
harness test double, not a pretend intelligence.

``OpenAICompatibleLLM`` talks to any OpenAI-compatible
``/chat/completions`` endpoint when ``OPENAI_API_KEY`` is set.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    # each tool call: {"name": str, "arguments": dict}
    usage: dict = field(default_factory=dict)
    # {"prompt_tokens": int, "completion_tokens": int}
    confidence: float = 1.0


class LLMClient(Protocol):
    def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        """messages: [{"role": "system"|"user"|"assistant"|"tool", "content": str}]
        tools: JSON-schema-style tool definitions (see tools.Tool.schema()).
        """


class MockLLM:
    """Deterministic mock. ``plan`` is a list of steps; each step is either
    a tool call {"tool": name, "arguments": {...}} or a final answer
    {"answer": text}. The mock advances one step per agent turn, so the
    loop, harness, guardrails and tracing all get exercised for real.
    """

    def __init__(self, plan: list[dict] | None = None):
        self.plan = list(plan or [])
        self._step = 0

    def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        if self._step >= len(self.plan):
            return LLMResponse(
                text="Done. No further steps planned.",
                usage={"prompt_tokens": 120, "completion_tokens": 12},
                confidence=0.5,
            )
        step = self.plan[self._step]
        self._step += 1
        if "answer" in step:
            return LLMResponse(
                text=step["answer"],
                usage={"prompt_tokens": 120, "completion_tokens": 40},
                confidence=step.get("confidence", 0.95),
            )
        return LLMResponse(
            text=f"Calling {step['tool']}.",
            tool_calls=[{"name": step["tool"], "arguments": step.get("arguments", {})}],
            usage={"prompt_tokens": 120, "completion_tokens": 25},
            confidence=0.95,
        )


class OpenAICompatibleLLM:
    """Minimal OpenAI-compatible client (stdlib only).

    Env: OPENAI_API_KEY (required), OPENAI_BASE_URL
    (default https://api.openai.com/v1), OPENAI_MODEL (default gpt-4o-mini).
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL",
                                                    "https://api.openai.com/v1")).rstrip("/")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set")

    def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = [{"type": "function", "function": t} for t in tools]
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read())
        choice = body["choices"][0]["message"]
        calls = []
        for c in choice.get("tool_calls", []):
            fn = c["function"]
            calls.append({"name": fn["name"],
                          "arguments": json.loads(fn.get("arguments") or "{}")})
        usage = body.get("usage", {})
        return LLMResponse(
            text=choice.get("content") or "",
            tool_calls=calls,
            usage={"prompt_tokens": usage.get("prompt_tokens", 0),
                   "completion_tokens": usage.get("completion_tokens", 0)},
            confidence=0.9,
        )
