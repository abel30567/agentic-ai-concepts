"""Concept 8 — Evals for Agents.

Test your agent before you trust it. Check outcomes, not just outputs,
and catch regressions before they ship.

- ``TestCase``: a task plus checks. Checks run against the *trajectory*
  (which tools were called, in what order) and the *outcome* (the final
  answer) — trajectory vs outcome evals.
- ``RuleJudge``: a deterministic stand-in for LLM-as-judge. Give it a
  rubric (a list of required strings); in production, swap it for a
  real judge model with the same ``grade(answer)`` interface.
- ``EvalRunner``: runs cases, reports pass/fail, and can save results
  for regression comparison (``compare`` fails when a previously
  passing case regresses).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class TestCase:
    name: str
    task: str
    # check(result) -> (passed, reason); result is agent.RunResult
    checks: list[Callable] = field(default_factory=list)
    # tools the agent is expected to have called, in order (subset match)
    expected_tools: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    name: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    stop_reason: str = ""
    tools_called: list[str] = field(default_factory=list)


class RuleJudge:
    """LLM-as-judge stand-in: grades an answer against a rubric of
    required phrases. Replace with a real judge model in production —
    same interface, richer grading."""

    def __init__(self, rubric: list[str]):
        self.rubric = rubric

    def grade(self, answer: str) -> tuple[bool, list[str]]:
        missing = [item for item in self.rubric
                   if item.lower() not in answer.lower()]
        return (not missing,
                [f"rubric item missing from answer: '{m}'" for m in missing])


@dataclass
class EvalRunner:
    """Runs test cases against an agent factory (a callable returning a
    fresh Agent per case, so state never leaks between cases)."""

    agent_factory: Callable
    results_path: str | None = None

    def run(self, cases: list[TestCase]) -> list[EvalResult]:
        results: list[EvalResult] = []
        for case in cases:
            agent = self.agent_factory()
            result = agent.run(case.task)
            failures: list[str] = []
            tools_called = [s["tool"] for s in result.trajectory
                            if s.get("tool")]
            for expected in case.expected_tools:
                if expected not in tools_called:
                    failures.append(f"expected tool '{expected}' was never called")
            for check in case.checks:
                passed, reason = check(result)
                if not passed:
                    failures.append(reason)
            results.append(EvalResult(
                name=case.name, passed=not failures, failures=failures,
                stop_reason=result.stop_reason, tools_called=tools_called))
        if self.results_path:
            with open(self.results_path, "w", encoding="utf-8") as f:
                json.dump([r.__dict__ for r in results], f, indent=2)
        return results

    @staticmethod
    def compare(previous_path: str, current: list[EvalResult]) -> list[str]:
        """Regression check: any case that passed before and fails now."""
        with open(previous_path, encoding="utf-8") as f:
            previous = {r["name"]: r["passed"] for r in json.load(f)}
        regressions = []
        for result in current:
            if previous.get(result.name) is True and not result.passed:
                regressions.append(f"REGRESSION: '{result.name}' used to "
                                   f"pass, now fails: {result.failures}")
        return regressions

    @staticmethod
    def report(results: list[EvalResult]) -> str:
        lines = []
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"[{status}] {r.name} (stop: {r.stop_reason})")
            for failure in r.failures:
                lines.append(f"       - {failure}")
        passed = sum(1 for r in results if r.passed)
        lines.append(f"{passed}/{len(results)} evals passed")
        return "\n".join(lines)
