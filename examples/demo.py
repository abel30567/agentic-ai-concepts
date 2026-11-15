"""Demo: single agent, mock LLM, real harness.

Task: find the top 5 laptops under $1000 and write a comparison table.
Runs offline — no API key. The mock LLM follows a scripted plan; the
loop, harness, guardrails, approval gate and tracer are all real.

    python examples/demo.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_systems import Agent, AgentConfig, Tool, ToolRegistry  # noqa: E402
from agent_systems.llm import MockLLM  # noqa: E402
from agent_systems.observability import Tracer  # noqa: E402

LAPTOPS = [
    {"name": "Acer Swift Go 14", "price": 649,
     "specs": '14" OLED, 16GB RAM, 512GB SSD'},
    {"name": "Lenovo IdeaPad Slim 5", "price": 749,
     "specs": '14" IPS, 16GB RAM, 512GB SSD'},
    {"name": "HP Pavilion 15", "price": 799,
     "specs": '15.6" IPS, 16GB RAM, 1TB SSD'},
    {"name": "ASUS Vivobook 16", "price": 899,
     "specs": '16" IPS, 16GB RAM, 512GB SSD'},
    {"name": "Dell Inspiron 14", "price": 999,
     "specs": '14" IPS, 16GB RAM, 1TB SSD'},
]


def search(args):
    query = args["query"].lower()
    if "laptop" not in query:
        return "No results."
    lines = [f"- {l['name']}: ${l['price']} — {l['specs']}" for l in LAPTOPS]
    return "Search results for laptops under $1000:\n" + "\n".join(lines)


def calculator(args):
    import ast
    import operator
    ops = {ast.Add: operator.add, ast.Sub: operator.sub,
           ast.Mult: operator.mul, ast.Div: operator.truediv}
    tree = ast.parse(args["expression"], mode="eval")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        raise ValueError("unsupported expression")
    return str(_eval(tree))


def write_file(args):
    # Sandboxed to output/ — the guardrail is in the handler, not just policy.
    root = os.path.abspath("output")
    target = os.path.abspath(os.path.join(root, args["path"]))
    if os.path.commonpath([root, target]) != root:
        raise ValueError("path escapes the output directory")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        f.write(args["content"])
    return f"Wrote {target}"


TABLE = """# Top 5 laptops under $1000

| Laptop | Price | Specs |
|---|---|---|
| Acer Swift Go 14 | $649 | 14" OLED, 16GB RAM, 512GB SSD |
| Lenovo IdeaPad Slim 5 | $749 | 14" IPS, 16GB RAM, 512GB SSD |
| HP Pavilion 15 | $799 | 15.6" IPS, 16GB RAM, 1TB SSD |
| ASUS Vivobook 16 | $899 | 16" IPS, 16GB RAM, 512GB SSD |
| Dell Inspiron 14 | $999 | 14" IPS, 16GB RAM, 1TB SSD |

Average price: $819. Best value: Acer Swift Go 14 at $649.
"""


def build_agent():
    registry = ToolRegistry()
    registry.register(Tool(
        name="search",
        description="Search the web for products.",
        parameters={"type": "object",
                    "properties": {"query": {"type": "string", "minLength": 1}},
                    "required": ["query"]},
        handler=search, permission="read"))
    registry.register(Tool(
        name="calculator",
        description="Evaluate an arithmetic expression.",
        parameters={"type": "object",
                    "properties": {"expression": {"type": "string", "minLength": 1}},
                    "required": ["expression"]},
        handler=calculator, permission="read"))
    registry.register(Tool(
        name="write_file",
        description="Write content to a file under output/.",
        parameters={"type": "object",
                    "properties": {
                        "path": {"type": "string", "minLength": 1},
                        "content": {"type": "string", "minLength": 1}},
                    "required": ["path", "content"]},
        handler=write_file, permission="write"))

    plan = [
        {"tool": "search", "arguments": {"query": "best laptops under $1000"}},
        {"tool": "calculator",
         "arguments": {"expression": "(649+749+799+899+999)/5"}},
        {"tool": "write_file",
         "arguments": {"path": "laptops.md", "content": TABLE}},
        {"answer": "Done — compared 5 laptops under $1000 (average $819). "
                   "Best value: Acer Swift Go 14 at $649. "
                   "Full table in output/laptops.md."},
    ]
    tracer = Tracer(run_id="demo", log_path="traces/demo.jsonl")
    return Agent(llm=MockLLM(plan), tools=registry,
                 config=AgentConfig(auto_approve=True), tracer=tracer)


def main():
    agent = build_agent()
    result = agent.run(
        "Find the top 5 laptops under $1000 and write a comparison table "
        "to output/laptops.md")
    print(f"\nAnswer: {result.answer}")
    print(f"Stop reason: {result.stop_reason} | Steps: {result.steps}")
    print("Trace summary:", json.dumps(result.trace_summary, indent=2))
    print("\nTrajectory:")
    for step in result.trajectory:
        print(f"  - {step['tool']}({step['args']})")


if __name__ == "__main__":
    main()
