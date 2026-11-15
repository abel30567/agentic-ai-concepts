# Agentic AI Concepts — a working reference implementation

The real skill in 2027 is systems thinking. This repo turns the 10
agentic-AI concepts from
[@bhawna.io's reel](https://www.instagram.com/reel/DdgmEKfh_QF/)
("Agentic AI Concepts You Must Master By 2027") into running code:
a small, dependency-free Python agent harness where every concept is a
module you can read, run, and steal for your own agents.

> Master these 10 and you'll be building agents that actually hold up
> in production, not just demos that break on day two.

## The 10 concepts, and where each lives

| # | Concept | What it is | Module |
|---|---------|-----------|--------|
| 1 | **Harness Engineering** | A safe, observable stack: retries, sandboxing, cost limits, graceful degradation — so the agent doesn't spiral out of control | `agent_systems/harness.py` |
| 2 | **Loop Engineering** | The repeat cycle: termination conditions, max-iteration budgets, runaway detection, clear feedback signals | `agent_systems/loop.py` |
| 3 | **Context Engineering** | The right info at the right time: context builder, compression, recency vs relevance, context-rot warnings | `agent_systems/context.py` |
| 4 | **Tool Design** | Tools the agent can actually use: clear names, strict input schemas, helpful errors, fewer-better-tools | `agent_systems/tools.py` |
| 5 | **Memory Architecture** | Short-term, long-term, and semantic memory — not one giant context dump | `agent_systems/memory.py` |
| 6 | **Orchestration Patterns** | Multiple agents splitting tasks, handing off, combining results | `agent_systems/orchestration.py` |
| 7 | **Guardrails & Permissions** | Scoped tool access, read/write/execute levels, input/output filtering, blast radius | `agent_systems/guardrails.py` |
| 8 | **Evals for Agents** | Golden test sets, trajectory vs outcome checks, LLM-as-judge stub, regression runs | `agent_systems/evals.py` |
| 9 | **Human-in-the-Loop Design** | Approval gates for irreversible or high-impact actions, smooth handoffs | `agent_systems/human_in_the_loop.py` |
| 10 | **Observability & Tracing** | Every step logged: prompts, tool calls, responses, errors, latency, cost | `agent_systems/observability.py` |

`agent_systems/agent.py` wires all ten together into one `Agent`.

## Quickstart

```bash
python examples/demo.py
```

Runs the demo agent (mock LLM, no API key needed) on a real task:
find the top 5 laptops under $1000 and write a comparison table.
It prints the trace and a cost/latency summary.

```bash
python examples/multi_agent.py   # orchestrator: research + planning + review
python -m unittest discover -s tests -v   # evals / regression tests
```

## Bring your own model

The demo uses `MockLLM`, a deterministic stand-in so everything runs
offline. To use a real model, implement the `LLMClient` protocol
(`agent_systems/llm.py`) — one method, `complete(messages, tools)` —
or use the included `OpenAICompatibleLLM`, which talks to any
OpenAI-compatible `/chat/completions` endpoint when `OPENAI_API_KEY`
is set:

```python
from agent_systems.llm import OpenAICompatibleLLM
agent = Agent(llm=OpenAICompatibleLLM(), tools=[...])
```

## Design choices

- **Standard library only.** No dependencies, Python 3.10+. Clone it
  and run it.
- **Every concept is independently usable.** Import one module, ignore
  the rest.
- **Failures are first-class.** Budgets, runaway detection, guardrails
  and evals exist because production agents fail in boring ways.
- **The mock is honest.** `MockLLM` follows a scripted plan — it is a
  stand-in for testing the harness, not a pretend intelligence.

## Layout

```
agent_systems/      # the ten concept modules + Agent
examples/           # demo.py (single agent), multi_agent.py (orchestrator)
tests/              # unittest evals: golden cases + regression checks
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). PRs that add a concept,
a tool, or a golden eval case are welcome.

## License

MIT — see [LICENSE](LICENSE).
