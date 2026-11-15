"""The Agent: all 10 concepts wired together.

One run:
  guardrails.check_input -> loop -> context.build (with recalled
  memories) -> llm.complete -> guardrails.check_tool_call ->
  human-in-the-loop approval -> harness.execute (retry + budget) ->
  observability events -> loop termination check -> final answer
  (guardrails.check_output, memory.remember).

``auto_approve=True`` is for demos and tests. In production leave it
off: writes and code execution then ask a human on the console.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .context import ContextBuilder
from .guardrails import Guardrails
from .harness import BudgetExceeded, CostBudget, Harness, RetryPolicy
from .human_in_the_loop import ApprovalGate, ApprovalPolicy
from .llm import LLMClient
from .loop import LoopController
from .memory import Memory
from .observability import Tracer
from .tools import ToolRegistry


@dataclass
class AgentConfig:
    system_prompt: str = ("You are a careful agent. Use the tools provided, "
                          "one step at a time. When the task is complete, "
                          "answer with a concise summary.")
    max_steps: int = 10
    token_budget: int = 4000
    max_tokens: int = 20_000
    max_tool_calls: int = 25
    max_usd: float = 1.00
    auto_approve: bool = False  # demo/test convenience; production: False


@dataclass
class RunResult:
    ok: bool
    answer: str
    steps: int
    stop_reason: str
    trajectory: list[dict] = field(default_factory=list)
    trace_summary: dict = field(default_factory=dict)


class Agent:
    def __init__(self, llm: LLMClient, tools: ToolRegistry,
                 config: AgentConfig | None = None,
                 memory: Memory | None = None,
                 guardrails: Guardrails | None = None,
                 tracer: Tracer | None = None,
                 approval_gate: ApprovalGate | None = None):
        self.llm = llm
        self.tools = tools
        self.config = config or AgentConfig()
        self.memory = memory or Memory()
        self.guardrails = guardrails or Guardrails()
        self.tracer = tracer or Tracer(run_id="agent-run")
        self.budget = CostBudget(max_tokens=self.config.max_tokens,
                                 max_tool_calls=self.config.max_tool_calls,
                                 max_usd=self.config.max_usd)
        self.harness = Harness(retry=RetryPolicy(), budget=self.budget)
        if approval_gate is not None:
            self.approval_gate = approval_gate
        elif self.config.auto_approve:
            self.approval_gate = ApprovalGate(
                approver=lambda desc, ctx: True)
        else:
            self.approval_gate = ApprovalGate()
        self.loop = LoopController(max_steps=self.config.max_steps)
        self.context = ContextBuilder(
            system_prompt=self.config.system_prompt,
            token_budget=self.config.token_budget)

    def run(self, task: str) -> RunResult:
        self.tracer.event("run_start", task=task)
        allowed, reason = self.guardrails.check_input(task)
        if not allowed:
            self.tracer.event("error", error=reason)
            return RunResult(ok=False, answer=reason, steps=0,
                             stop_reason="blocked: input guardrail",
                             trace_summary=self.tracer.summary())

        history: list[dict] = []
        observations: list[dict] = []
        trajectory: list[dict] = []
        answer = ""
        stop_reason = "done"

        try:
            while True:
                memories = self.memory.recall(task)
                window = self.context.build(task, history,
                                            memories=memories,
                                            observations=observations)
                if window.rot_warning:
                    self.tracer.event("warning",
                                      warning="context rot risk — long context")
                resp = self.llm.complete(window.messages,
                                         self.tools.schemas())
                self.budget.charge_tokens(
                    resp.usage.get("prompt_tokens", 0),
                    resp.usage.get("completion_tokens", 0))
                self.tracer.event("llm_response",
                                  tokens=resp.usage.get("prompt_tokens", 0)
                                  + resp.usage.get("completion_tokens", 0),
                                  tool_calls=len(resp.tool_calls))

                if not resp.tool_calls:
                    # Final answer.
                    stop = self.loop.check_confidence(resp.confidence)
                    if stop:
                        stop_reason = stop
                        answer = resp.text
                        break
                    allowed, reason = self.guardrails.check_output(resp.text)
                    if not allowed:
                        answer = reason
                        stop_reason = "blocked: output guardrail"
                        break
                    answer = resp.text
                    self.memory.remember(f"Task: {task}\nAnswer: {answer}")
                    stop_reason = "done"
                    break

                observations = []
                for call in resp.tool_calls:
                    name, args = call["name"], call.get("arguments", {})
                    action_key = f"{name}:{json.dumps(args, sort_keys=True)}"
                    stop = self.loop.observe_action(action_key)
                    tool = self.tools.get(name)
                    if tool is None:
                        obs = f"Unknown tool '{name}'. Available tools: " \
                              f"{sorted(self.tools._tools)}."
                        self.loop.observe_error()
                        self.tracer.event("error", error=obs)
                    else:
                        allowed, reason = self.guardrails.check_tool_call(
                            name, tool.permission)
                        if not allowed:
                            obs = reason
                            self.loop.observe_error()
                            self.tracer.event("error", error=reason)
                        else:
                            approved = self.approval_gate.request(
                                description=f"Run tool '{name}' with {args}",
                                permission=tool.permission,
                                confidence=resp.confidence,
                                tool_name=name)
                            if not approved:
                                obs = (f"Tool '{name}' was not approved by "
                                       "the human. Do not retry it; finish "
                                       "with what you have or ask.")
                                self.loop.observe_error()
                                self.tracer.event("approval", tool=name,
                                                  approved=False)
                            else:
                                self.tracer.event("tool_call", tool=name,
                                                  args=args)
                                result = self.harness.execute(tool, args)
                                if result.get("ok"):
                                    self.loop.observe_success()
                                else:
                                    self.loop.observe_error()
                                obs = json.dumps(result)
                                self.tracer.event(
                                    "tool_result", tool=name,
                                    ok=bool(result.get("ok")))
                    trajectory.append({"tool": name, "args": args,
                                       "observation": obs})
                    history.append({"role": "assistant",
                                    "content": f"Called {name}({args})"})
                    observations.append({"role": "tool", "content": obs})
                    if stop:
                        stop_reason = stop
                        answer = ("Stopped: " + stop +
                                  ". Partial progress is in the trajectory.")
                        return RunResult(ok=False, answer=answer,
                                         steps=self.loop.steps,
                                         stop_reason=stop_reason,
                                         trajectory=trajectory,
                                         trace_summary=self.tracer.summary())
        except BudgetExceeded as exc:
            self.tracer.event("error", error=str(exc))
            return RunResult(ok=False, answer=str(exc), steps=self.loop.steps,
                             stop_reason=f"budget exceeded: {exc}",
                             trajectory=trajectory,
                             trace_summary=self.tracer.summary())

        self.tracer.event("run_end", stop_reason=stop_reason)
        return RunResult(ok=True, answer=answer, steps=self.loop.steps,
                         stop_reason=stop_reason, trajectory=trajectory,
                         trace_summary=self.tracer.summary())
