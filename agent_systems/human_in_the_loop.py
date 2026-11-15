"""Concept 9 — Human-in-the-Loop Design.

Know exactly when a human should step in for approval, and when the
agent should just run.

Policy: a human is required for critical, irreversible, or high-impact
actions — writes to disk, code execution, anything external (send,
publish, pay), and any step the agent is unsure about. Everything else
runs autonomously, so the human never becomes the bottleneck.

The gate records every decision in the trace (see observability), so
approvals are auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# Tool permissions that always need a human (unless the policy says otherwise).
IRREVERSIBLE_PERMISSIONS = ("write", "execute")


@dataclass
class ApprovalPolicy:
    """When to keep a human in the loop."""

    require_approval_for: tuple[str, ...] = IRREVERSIBLE_PERMISSIONS
    min_confidence: float = 0.5  # below this, ask even for reads
    auto_approve_reads: bool = True

    def needs_approval(self, *, permission: str, confidence: float,
                       tool_name: str = "") -> tuple[bool, str]:
        if confidence < self.min_confidence:
            return True, (f"low confidence ({confidence:.2f}) — human review "
                          "required")
        if permission in self.require_approval_for:
            return True, (f"'{tool_name}' is a {permission} action — "
                          "irreversible, needs approval")
        return False, ""


@dataclass
class ApprovalGate:
    """Asks the human, records the answer. ``approver`` is a callable
    taking (description, context) and returning True/False. The console
    approver is the default; tests use an auto-approver."""

    policy: ApprovalPolicy = field(default_factory=ApprovalPolicy)
    approver: Callable[[str, str], bool] | None = None
    decisions: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.approver is None:
            self.approver = self._console_approver

    @staticmethod
    def _console_approver(description: str, context: str) -> bool:
        print(f"\n[approval needed] {description}")
        if context:
            print(f"  context: {context}")
        answer = input("Approve? [y/N] ").strip().lower()
        return answer in ("y", "yes")

    def request(self, *, description: str, context: str = "",
                permission: str = "read", confidence: float = 1.0,
                tool_name: str = "") -> bool:
        needed, reason = self.policy.needs_approval(
            permission=permission, confidence=confidence, tool_name=tool_name)
        if not needed:
            self.decisions.append({"action": description, "approved": True,
                                   "via": "auto (policy)"})
            return True
        approved = bool(self.approver(f"{description} — {reason}", context))
        self.decisions.append({"action": description, "approved": approved,
                               "via": "human", "reason": reason})
        return approved
