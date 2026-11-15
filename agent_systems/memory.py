"""Concept 5 — Memory Architecture.

Help the agent remember what matters: short-term, long-term, and
semantic memory — instead of dumping everything into one context window.

- ShortTermMemory: the current session, a bounded deque of turns.
- LongTermMemory: persistent facts (JSON file), with timestamps.
- SemanticMemory: retrieval over stored notes by word overlap
  (stdlib-only; swap in embeddings for production).
- Memory: the facade — write once, recall across all three.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter, deque
from dataclasses import dataclass, field

_STOPWORDS = set(
    "a an the is are was were be been to of in on for and or with at by from as it its this that".split()
)


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


@dataclass
class ShortTermMemory:
    """Session memory: the last N turns, in order. Bounded by design —
    this is what goes into the context window."""

    capacity: int = 20
    _turns: deque = field(default_factory=lambda: deque(maxlen=20))

    def __post_init__(self):
        self._turns = deque(maxlen=self.capacity)

    def add(self, role: str, content: str) -> None:
        self._turns.append({"role": role, "content": content})

    def recent(self, n: int | None = None) -> list[dict]:
        turns = list(self._turns)
        return turns[-n:] if n else turns

    def clear(self) -> None:
        self._turns.clear()


@dataclass
class LongTermMemory:
    """Persistent facts, JSON-backed. What to store (and not): keep useful
    facts, never store secrets or sensitive data — callers must filter."""

    path: str = "memory/long_term.json"
    _facts: dict = field(default_factory=dict)

    def __post_init__(self):
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                self._facts = json.load(f)

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._facts, f, indent=2)

    def store(self, key: str, value: str) -> None:
        self._facts[key] = {"value": value, "updated_at": time.time()}
        self._save()

    def get(self, key: str) -> str | None:
        fact = self._facts.get(key)
        return fact["value"] if fact else None

    def search(self, query: str) -> list[tuple[str, str]]:
        """Keyword search over keys and values."""
        query_tokens = set(_tokens(query))
        hits = []
        for key, fact in self._facts.items():
            haystack = set(_tokens(key + " " + fact["value"]))
            if query_tokens & haystack:
                hits.append((key, fact["value"]))
        return hits


@dataclass
class SemanticMemory:
    """Retrieval over notes by word overlap (TF cosine). A stand-in for a
    vector store: same interface (add / retrieve), swappable later."""

    _docs: list[dict] = field(default_factory=list)

    def add(self, text: str, metadata: dict | None = None) -> None:
        self._docs.append({"text": text, "tokens": Counter(_tokens(text)),
                           "metadata": metadata or {}})

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        q = Counter(_tokens(query))
        if not q or not self._docs:
            return []
        q_norm = math.sqrt(sum(v * v for v in q.values()))
        scored = []
        for doc in self._docs:
            dot = sum(q[t] * doc["tokens"][t] for t in q)
            d_norm = math.sqrt(sum(v * v for v in doc["tokens"].values()))
            score = dot / (q_norm * d_norm) if d_norm else 0.0
            if score > 0:
                scored.append((score, doc["text"]))
        scored.sort(key=lambda s: -s[0])
        return [text for _, text in scored[:top_k]]


@dataclass
class Memory:
    """One facade over all three stores. ``remember`` writes to the right
    store; ``recall`` searches long-term + semantic and returns the best
    hits for the context builder."""

    short_term: ShortTermMemory = field(default_factory=ShortTermMemory)
    long_term: LongTermMemory = field(default_factory=LongTermMemory)
    semantic: SemanticMemory = field(default_factory=SemanticMemory)

    def remember(self, text: str, *, persistent: bool = False,
                 key: str | None = None) -> None:
        self.short_term.add("memory", text)
        if persistent and key:
            self.long_term.store(key, text)
        self.semantic.add(text)

    def recall(self, query: str, top_k: int = 3) -> list[str]:
        seen: list[str] = []
        for hit in self.long_term.search(query):
            seen.append(f"{hit[0]}: {hit[1]}")
        seen.extend(self.semantic.retrieve(query, top_k=top_k))
        # de-duplicate, keep order
        out, seen_set = [], set()
        for item in seen:
            if item not in seen_set:
                seen_set.add(item)
                out.append(item)
        return out[:top_k]
