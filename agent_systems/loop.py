"""Concept 2 — Loop Engineering.

Design the repeat cycle for reliable agent behaviour: define when to
stop, budget the iterations, detect runaways, and give the agent clear
feedback signals (tool results as observations) so it can decide the
next step.

The loop itself lives in ``agent.Agent``; this module is the state
machine that decides when the loop ends and why.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class LoopController:
    """Termination conditions, checked after every step."""

    max_steps: int = 10
    max_consecutive_errors: int = 3
    max_repeats: int = 3  # same tool + arguments this many times = runaway
    min_confidence: float = 0.3

    steps: int = 0
    consecutive_errors: int = 0
    _recent_actions: deque = field(default_factory=lambda: deque(maxlen=3))

    def observe_action(self, action_key: str) -> str | None:
        """Record one agent action. Returns a stop reason, or None."""
        self.steps += 1
        self._recent_actions.append(action_key)
        if self.steps >= self.max_steps:
            return f"stop: max steps reached ({self.max_steps})"
        if (len(self._recent_actions) == self.max_repeats
                and len(set(self._recent_actions)) == 1):
            return (f"stop: runaway detected — '{action_key}' repeated "
                    f"{self.max_repeats}x")
        return None

    def observe_error(self) -> str | None:
        self.consecutive_errors += 1
        if self.consecutive_errors >= self.max_consecutive_errors:
            return (f"stop: {self.consecutive_errors} consecutive errors — "
                    "the plan is not working")
        return None

    def observe_success(self) -> None:
        self.consecutive_errors = 0

    def check_confidence(self, confidence: float) -> str | None:
        if confidence < self.min_confidence:
            return (f"stop: low confidence ({confidence:.2f} < "
                    f"{self.min_confidence}) — escalate to a human")
        return None
