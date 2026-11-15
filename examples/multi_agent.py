"""Demo: orchestrator — research, plan, review.

Task: plan a 3-day trip to Tokyo under $1500. Two independent research
subtasks run in parallel; planning depends on both; review depends on
the plan. Each subtask gets its own agent with scoped tools.

    python examples/multi_agent.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_systems import Agent, AgentConfig, Tool, ToolRegistry  # noqa: E402
from agent_systems.llm import MockLLM  # noqa: E402
from agent_systems.orchestration import Orchestrator, Subtask  # noqa: E402


def _search_tool():
    def search(args):
        q = args["query"].lower()
        if "flight" in q:
            return "Flights NYC–Tokyo: $620 round trip (JAL, Nov)."
        if "hotel" in q:
            return "Hotels: capsule $45/night, business hotel $90/night."
        return "No results."
    return Tool(
        name="search",
        description="Search the web.",
        parameters={"type": "object",
                    "properties": {"query": {"type": "string", "minLength": 1}},
                    "required": ["query"]},
        handler=search, permission="read")


def _agent(plan, tools):
    return Agent(llm=MockLLM(plan), tools=tools,
                 config=AgentConfig(auto_approve=True, max_steps=5))


def main():
    search_registry = ToolRegistry()
    search_registry.register(_search_tool())
    empty_registry = ToolRegistry()

    research_flights = _agent(
        [{"tool": "search", "arguments": {"query": "flights NYC Tokyo price"}},
         {"answer": "Flights: $620 round trip (JAL)."}],
        search_registry)
    research_hotels = _agent(
        [{"tool": "search", "arguments": {"query": "Tokyo hotels price"}},
         {"answer": "Hotels: $45–90/night; 3 nights = $135–270."}],
        search_registry)
    planner = _agent(
        [{"answer": "Plan: flights $620 + hotel $180 (3 nights, business) + "
                    "food/transit $300 = $1100 total, under the $1500 budget."}],
        empty_registry)
    reviewer = _agent(
        [{"answer": "Review: budget adds up ($1100 < $1500). Risk: flight "
                    "price is a single quote — re-check before booking."}],
        empty_registry)

    orchestrator = Orchestrator()
    outcome = orchestrator.run([
        Subtask(name="research_flights",
                task="Find flight prices NYC to Tokyo.",
                agent=research_flights),
        Subtask(name="research_hotels",
                task="Find Tokyo hotel prices for 3 nights.",
                agent=research_hotels),
        Subtask(name="plan",
                task="Build a 3-day Tokyo plan under $1500.",
                depends_on=["research_flights", "research_hotels"],
                agent=planner),
        Subtask(name="review",
                task="Review the plan for budget and risks.",
                depends_on=["plan"], agent=reviewer),
    ])
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    main()
