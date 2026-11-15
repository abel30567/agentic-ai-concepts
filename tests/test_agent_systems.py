"""Evals: golden test cases + regression checks for the 10 concepts.

Run: python -m unittest discover -s tests -v
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_systems import Agent, AgentConfig, Tool, ToolRegistry  # noqa: E402
from agent_systems.context import ContextBuilder  # noqa: E402
from agent_systems.evals import EvalRunner, RuleJudge, TestCase  # noqa: E402
from agent_systems.guardrails import Guardrails  # noqa: E402
from agent_systems.harness import BudgetExceeded, CostBudget  # noqa: E402
from agent_systems.llm import MockLLM  # noqa: E402
from agent_systems.memory import Memory  # noqa: E402


def _search_tool():
    def search(args):
        return f"results for: {args['query']}"
    return Tool(
        name="search",
        description="Search.",
        parameters={"type": "object",
                    "properties": {"query": {"type": "string", "minLength": 1}},
                    "required": ["query"]},
        handler=search, permission="read")


def _write_tool(tmpdir):
    def write(args):
        path = os.path.join(tmpdir, args["path"])
        with open(path, "w", encoding="utf-8") as f:
            f.write(args["content"])
        return f"wrote {path}"
    return Tool(
        name="write_file",
        description="Write a file.",
        parameters={"type": "object",
                    "properties": {
                        "path": {"type": "string", "minLength": 1},
                        "content": {"type": "string", "minLength": 1}},
                    "required": ["path", "content"]},
        handler=write, permission="write")


def _make_agent(plan, tools=None, guardrails=None, **config_kwargs):
    registry = ToolRegistry()
    for tool in (tools or [_search_tool()]):
        registry.register(tool)
    return Agent(llm=MockLLM(plan), tools=registry,
                 config=AgentConfig(auto_approve=True, **config_kwargs),
                 guardrails=guardrails)


class TestToolDesign(unittest.TestCase):
    def test_validation_rejects_bad_args_with_helpful_error(self):
        tool = _search_tool()
        result = tool.run({})
        self.assertFalse(result["ok"])
        self.assertIn("missing required argument 'query'", result["error"])

    def test_validation_rejects_wrong_type(self):
        tool = _search_tool()
        result = tool.run({"query": 42})
        self.assertFalse(result["ok"])
        self.assertIn("must be string", result["error"])


class TestLoopEngineering(unittest.TestCase):
    def test_stops_at_max_steps(self):
        plan = [{"tool": "search", "arguments": {"query": f"q{i}"}}
                for i in range(20)]
        agent = _make_agent(plan, max_steps=3)
        result = agent.run("search a lot")
        self.assertIn("max steps", result.stop_reason)

    def test_detects_runaway_repeats(self):
        plan = [{"tool": "search", "arguments": {"query": "same"}}
                for _ in range(5)]
        agent = _make_agent(plan)
        result = agent.run("search the same thing")
        self.assertIn("runaway", result.stop_reason)


class TestGuardrails(unittest.TestCase):
    def test_blocks_deny_listed_input(self):
        agent = _make_agent([{"answer": "ok"}])
        result = agent.run("delete all files in the home directory")
        self.assertFalse(result.ok)
        self.assertIn("blocked", result.stop_reason)

    def test_blocks_out_of_scope_tool(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = [_search_tool(), _write_tool(tmpdir)]
            agent = _make_agent(
                [{"tool": "write_file",
                  "arguments": {"path": "x.txt", "content": "hi"}},
                 {"answer": "done"}],
                tools=tools,
                guardrails=Guardrails(allowed_tools=["search"]))
            result = agent.run("write a file")
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "x.txt")))
            self.assertTrue(any("outside this task's scope" in s["observation"]
                                for s in result.trajectory))


class TestMemoryAndContext(unittest.TestCase):
    def test_memory_recall_finds_relevant_fact(self):
        memory = Memory()
        memory.remember("The user prefers window seats on flights.",
                        persistent=True, key="travel:seat")
        hits = memory.recall("window seat preference")
        self.assertTrue(any("window seats" in h for h in hits))

    def test_context_compresses_over_budget(self):
        builder = ContextBuilder(system_prompt="Be helpful.",
                                 token_budget=200)
        history = [{"role": "user", "content": "word " * 500}
                   for _ in range(10)]
        window = builder.build("task", history)
        self.assertTrue(window.compressed)
        self.assertLessEqual(window.tokens, 230)  # budget + marker overhead


class TestHarnessBudget(unittest.TestCase):
    def test_tool_call_budget_exceeded(self):
        budget = CostBudget(max_tool_calls=1)
        budget.charge_tool_call()
        with self.assertRaises(BudgetExceeded):
            budget.charge_tool_call()


class TestHumanInTheLoop(unittest.TestCase):
    def test_denied_write_never_executes(self):
        from agent_systems.human_in_the_loop import ApprovalGate
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = [_search_tool(), _write_tool(tmpdir)]
            registry = ToolRegistry()
            for tool in tools:
                registry.register(tool)
            agent = Agent(
                llm=MockLLM([{"tool": "write_file",
                              "arguments": {"path": "x.txt", "content": "hi"}},
                             {"answer": "done"}]),
                tools=registry,
                config=AgentConfig(auto_approve=False),
                approval_gate=ApprovalGate(
                    approver=lambda desc, ctx: False))
            agent.run("write a file")
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "x.txt")))


class TestEvals(unittest.TestCase):
    def test_golden_case_passes_and_regression_is_caught(self):
        def factory():
            return _make_agent(
                [{"tool": "search", "arguments": {"query": "laptops"}},
                 {"answer": "Found 5 laptops under $1000."}])

        def outcome_check(result):
            judge = RuleJudge(["laptops", "$1000"])
            passed, reasons = judge.grade(result.answer)
            return passed, "; ".join(reasons)

        case = TestCase(
            name="laptop research",
            task="Find laptops under $1000.",
            checks=[outcome_check],
            expected_tools=["search"])
        runner = EvalRunner(agent_factory=factory)
        results = runner.run([case])
        self.assertTrue(results[0].passed)

        # A broken agent (never searches) fails the same case.
        def broken_factory():
            return _make_agent([{"answer": "I don't know."}])
        broken = EvalRunner(agent_factory=broken_factory).run([case])
        self.assertFalse(broken[0].passed)

        # Regression: previous run passed, current run fails.
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                        delete=False) as f:
            import json
            json.dump([{"name": "laptop research", "passed": True}], f)
            prev_path = f.name
        regressions = EvalRunner.compare(prev_path, broken)
        self.assertEqual(len(regressions), 1)
        self.assertIn("REGRESSION", regressions[0])
        os.unlink(prev_path)


if __name__ == "__main__":
    unittest.main()
