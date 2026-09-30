"""Tiny persistent memory: long-term facts + an episodic run log."""
from __future__ import annotations

import json
import os
import time


class Memory:
    def __init__(self, path: str = "~/.aura/memory.json"):
        self.path = os.path.expanduser(path)
        self.facts: dict[str, str] = {}
        self.episodes: list[dict] = []
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            self.facts = data.get("facts", {})
            self.episodes = data.get("episodes", [])
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"facts": self.facts, "episodes": self.episodes[-50:]}, f, indent=2)

    def remember(self, key: str, value: str) -> None:
        """Store a long-term fact, e.g. memory.remember("user", "Yashwanth")."""
        self.facts[key] = value
        self.save()

    def recall(self, key: str) -> str | None:
        return self.facts.get(key)

    def log_episode(self, task: str, result: str) -> None:
        self.episodes.append({
            "t": time.strftime("%Y-%m-%d %H:%M"),
            "task": task,
            "result": result[:500],
        })
        self.save()

    def context(self) -> str:
        if not self.facts:
            return "No long-term memories yet."
        return "\n".join(f"- {k}: {v}" for k, v in self.facts.items())
