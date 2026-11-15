"""Concept 6 — Orchestration Patterns.

Coordinate multiple agents (or tools) to solve bigger tasks: split the
work, hand off with context, run independent subtasks in parallel and
dependent ones in sequence, then evaluate and aggregate the results.

Use a single agent when the task needs one skill; use an orchestrator
when it needs several. The orchestrator below takes explicit subtasks —
decomposition itself is a job for the LLM in production; here the caller
(or a planning agent) supplies the split.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field


@dataclass
class Subtask:
    name: str
    task: str
    depends_on: list[str] = field(default_factory=list)  # names of subtasks
    agent: object = None  # an Agent (or anything with .run(task))


@dataclass
class Orchestrator:
    """Run subtasks respecting dependencies; aggregate at the end."""

    max_workers: int = 4

    def run(self, subtasks: list[Subtask]) -> dict:
        results: dict[str, dict] = {}
        failed: dict[str, str] = {}
        remaining = list(subtasks)
        while remaining:
            ready = [s for s in remaining
                     if all(d in results for d in s.depends_on)]
            if not ready:
                missing = [s.name for s in remaining]
                raise ValueError(f"unresolvable dependencies among: {missing}")
            # independent subtasks run in parallel; dependent ones wait
            # for the next round because their deps only just completed.
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {pool.submit(self._run_one, s, results): s
                           for s in ready}
                for future, subtask in futures.items():
                    try:
                        results[subtask.name] = future.result()
                    except Exception as exc:  # noqa: BLE001 — aggregate, don't crash
                        failed[subtask.name] = str(exc)
                        results[subtask.name] = {"ok": False, "error": str(exc)}
            remaining = [s for s in remaining if s not in ready]
        return {"results": results, "failed": failed,
                "ok": not failed}

    @staticmethod
    def _run_one(subtask: Subtask, prior_results: dict) -> dict:
        # Handoff: give the agent the prior results it depends on.
        context = ""
        for dep in subtask.depends_on:
            prior = prior_results.get(dep, {})
            context += f"\n[{dep} result] {prior.get('answer', prior)}"
        task = subtask.task + context if context else subtask.task
        if subtask.agent is None:
            return {"ok": True, "answer": f"(no agent assigned) {task}"}
        result = subtask.agent.run(task)
        return {"ok": result.ok, "answer": result.answer,
                "steps": result.steps, "stop_reason": result.stop_reason}
