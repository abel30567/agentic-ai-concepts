"""Concept 4 — Tool Design.

Tools your agent can actually use: clear names, strict input schemas,
helpful error messages it can recover from, and fewer, better tools.

Rules enforced here:
- every tool declares a JSON-schema-style ``parameters`` object
- arguments are validated *before* the handler runs, and a failure
  returns an error that says what went wrong and how to fix it
- every tool has a permission level (read | write | execute) so the
  guardrails module can scope access per task
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

PERMISSIONS = ("read", "write", "execute")

_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # {"type": "object", "properties": {...}, "required": [...]}
    handler: Callable[[dict], Any]
    permission: str = "read"

    def __post_init__(self):
        if self.permission not in PERMISSIONS:
            raise ValueError(f"permission must be one of {PERMISSIONS}")

    def schema(self) -> dict:
        return {"name": self.name, "description": self.description,
                "parameters": self.parameters}

    def validate(self, args: dict) -> list[str]:
        """Return a list of human-readable errors (empty = valid)."""
        errors: list[str] = []
        if not isinstance(args, dict):
            return ["arguments must be an object"]
        props = self.parameters.get("properties", {})
        for req in self.parameters.get("required", []):
            if req not in args:
                errors.append(f"missing required argument '{req}'")
        for key, value in args.items():
            spec = props.get(key)
            if spec is None:
                errors.append(f"unknown argument '{key}' — this tool accepts: "
                              f"{sorted(props) or 'none'}")
                continue
            expected = spec.get("type")
            if expected and not _TYPE_CHECKS.get(expected, lambda v: True)(value):
                errors.append(f"argument '{key}' must be {expected}, "
                              f"got {type(value).__name__}")
            if (expected == "string" and isinstance(value, str)
                    and spec.get("minLength")
                    and len(value) < spec["minLength"]):
                errors.append(f"argument '{key}' is too short "
                              f"(min {spec['minLength']} characters)")
            if "enum" in spec and value not in spec["enum"]:
                errors.append(f"argument '{key}' must be one of {spec['enum']}")
        return errors

    def run(self, args: dict) -> dict:
        """Run with validation. Never raises: failures come back as a dict
        the agent can read and recover from."""
        errors = self.validate(args)
        if errors:
            return {"ok": False,
                    "error": "Invalid arguments for tool "
                             f"'{self.name}': " + "; ".join(errors) +
                             ". Fix the arguments and try again."}
        try:
            return {"ok": True, "result": self.handler(args)}
        except Exception as exc:  # noqa: BLE001 — the agent must see it
            return {"ok": False,
                    "error": f"Tool '{self.name}' failed: {exc}. "
                             "Try a different approach or different arguments."}


@dataclass
class ToolRegistry:
    """Fewer, better tools: register only what the task needs, and hand
    the agent a scoped subset per task."""

    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self, names: list[str] | None = None) -> list[dict]:
        tools = [self._tools[n] for n in (names or sorted(self._tools))]
        return [t.schema() for t in tools]

    def scoped(self, names: list[str]) -> "ToolRegistry":
        """A new registry containing only the named tools — the unit the
        guardrails module scopes per task."""
        sub = ToolRegistry()
        for name in names:
            tool = self.get(name)
            if tool is None:
                raise ValueError(f"unknown tool '{name}'")
            sub.register(tool)
        return sub

    def __len__(self) -> int:
        return len(self._tools)
