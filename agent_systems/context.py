"""Concept 3 — Context Engineering.

Give the model the right information at the right time. Too much context
rots, too little breaks the response.

The builder assembles, in priority order:
  1. system instructions (always)
  2. recalled memories relevant to the task (memory architecture)
  3. recent conversation history (short-term memory)
  4. the latest tool observations

When the assembled context exceeds the token budget it compresses:
older turns are replaced by a short summary marker instead of being
silently dropped. ``rot_warning`` flags long contexts — long contexts
degrade reasoning, so the agent should compress early, not late.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CHARS_PER_TOKEN = 4  # rough estimate, good enough for budgeting


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass
class ContextWindow:
    messages: list[dict]
    tokens: int
    compressed: bool = False
    rot_warning: bool = False


@dataclass
class ContextBuilder:
    """Assemble the prompt for one agent turn."""

    system_prompt: str
    token_budget: int = 4000
    # Warn about context rot once history passes this many tokens.
    rot_threshold: int = 2500

    def build(self, task: str, history: list[dict],
              memories: list[str] | None = None,
              observations: list[dict] | None = None) -> ContextWindow:
        messages: list[dict] = [{"role": "system", "content": self.system_prompt}]
        if memories:
            joined = "\n".join(f"- {m}" for m in memories)
            messages.append({"role": "system",
                             "content": f"Relevant memories:\n{joined}"})
        messages.append({"role": "user", "content": task})
        for turn in history:
            messages.append({"role": turn["role"], "content": turn["content"]})
        for obs in observations or []:
            messages.append({"role": "tool", "content": obs["content"]})

        total = sum(estimate_tokens(m["content"]) for m in messages)
        compressed = False
        if total > self.token_budget:
            messages, total = self._compress(messages)
            compressed = True
        rot_warning = total > self.rot_threshold
        return ContextWindow(messages=messages, tokens=total,
                             compressed=compressed, rot_warning=rot_warning)

    def _compress(self, messages: list[dict]) -> tuple[list[dict], int]:
        """Keep system + task + as many of the newest turns as fit the
        budget; summarise the dropped middle as a marker. If even the
        newest turn alone doesn't fit, truncate it. Recency beats
        relevance for the tail, but the task and instructions are never
        dropped."""
        system = [m for m in messages if m["role"] == "system"]
        rest = [m for m in messages if m["role"] != "system"]
        if not rest:
            return messages, sum(estimate_tokens(m["content"]) for m in messages)
        task = rest[0]
        base = system + [task]
        base_tokens = sum(estimate_tokens(m["content"]) for m in base)
        marker_text = "[context compressed: earlier turns summarised away " \
                      "to stay within budget]"
        marker_tokens = estimate_tokens(marker_text)
        # Greedily keep the newest turns that fit.
        kept_newest_first: list[dict] = []
        used = base_tokens + marker_tokens
        for turn in reversed(rest[1:]):
            cost = estimate_tokens(turn["content"])
            if used + cost <= self.token_budget:
                kept_newest_first.append(turn)
                used += cost
            else:
                break
        kept = list(reversed(kept_newest_first))
        dropped = len(rest) - 1 - len(kept)
        marker = {"role": "system",
                  "content": f"[context compressed: {dropped} earlier turns "
                             "summarised away to stay within budget]"}
        out = system + [task, marker] + kept
        if not kept and len(rest) > 1:
            # Even one turn doesn't fit: truncate the newest to the room left.
            room = self.token_budget - base_tokens - marker_tokens
            if room > 0:
                last = dict(rest[-1])
                last["content"] = (last["content"][:room * CHARS_PER_TOKEN]
                                   + "… [truncated]")
                out = system + [task, marker, last]
        total = sum(estimate_tokens(m["content"]) for m in out)
        return out, total
