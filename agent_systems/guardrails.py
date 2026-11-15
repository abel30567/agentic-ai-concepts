"""Concept 7 — Guardrails & Permissions.

Keep the agent useful but safe: scope what tools it can touch, set
permission levels, filter inputs and outputs, and limit the blast radius
if something goes wrong.

Three checks, applied in order by ``Agent`` before/after every step:
  1. ``check_input`` — the task itself (deny-listed requests)
  2. ``check_tool_call`` — is this tool allowed for this task, at this
     permission level, within quota?
  3. ``check_output`` — does the final answer leak sensitive data?
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Deny-listed intents: destructive or exfiltration requests, phrased
# the way a user (or a prompt injection) would phrase them.
DENY_PATTERNS = [
    re.compile(r"\bdelete all\b", re.I),
    re.compile(r"\brm\s+-rf\b", re.I),
    re.compile(r"\bdrop\s+table\b", re.I),
    re.compile(r"\bformat\s+\w:", re.I),
    re.compile(r"\bexfiltrate\b", re.I),
    re.compile(r"\baccess personal data\b", re.I),
    re.compile(r"\bsend .* without approval\b", re.I),
]

# Output filtering: things that must never appear in an answer.
SENSITIVE_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "possible SSN"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
     "email address"),
    (re.compile(r"\bsk-[A-Za-z0-9]{8,}\b"), "possible API key"),
]


@dataclass
class Guardrails:
    """Per-task scope. ``allowed_tools`` is the blast-radius control:
    the agent only ever sees the tools listed here."""

    allowed_tools: list[str] | None = None  # None = all registered tools
    allowed_permissions: tuple[str, ...] = ("read", "write", "execute")
    max_writes: int = 5  # quota: how many write/execute calls per run
    _writes_used: int = 0

    def check_input(self, text: str) -> tuple[bool, str]:
        for pattern in DENY_PATTERNS:
            if pattern.search(text):
                return False, f"blocked: request matches deny pattern '{pattern.pattern}'"
        return True, ""

    def check_tool_call(self, tool_name: str, permission: str) -> tuple[bool, str]:
        if self.allowed_tools is not None and tool_name not in self.allowed_tools:
            return False, (f"blocked: tool '{tool_name}' is outside this "
                           "task's scope")
        if permission not in self.allowed_permissions:
            return False, (f"blocked: permission '{permission}' is not "
                           f"granted for this task (granted: "
                           f"{list(self.allowed_permissions)})")
        if permission in ("write", "execute"):
            if self._writes_used >= self.max_writes:
                return False, (f"blocked: write quota exhausted "
                               f"({self.max_writes} per run)")
            self._writes_used += 1
        return True, ""

    def check_output(self, text: str) -> tuple[bool, str]:
        for pattern, label in SENSITIVE_PATTERNS:
            if pattern.search(text):
                return False, f"blocked: output contains {label}"
        return True, ""
