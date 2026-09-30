"""LLM backends for AURA. Bring your own key, or run the zero-key demo."""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


class LLMClient:
    """Anything that can turn a message list into text can drive the agent."""

    def complete(self, messages: list[Message]) -> str:
        raise NotImplementedError


class OpenAICompatibleLLM(LLMClient):
    """Talks to any OpenAI-compatible chat-completions endpoint."""

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None):
        self.model = model or os.environ.get("AURA_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or os.environ.get("AURA_BASE_URL",
                         "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("AURA_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("Set AURA_API_KEY (or pass api_key=) to use OpenAICompatibleLLM.")

    def complete(self, messages: list[Message]) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": 0.2,
        }
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"]


class ScriptedLLM(LLMClient):
    """Deterministic stand-in so demos and tests run with zero API keys.

    Each call pops the next scripted reply. When the script runs out it
    closes the loop with an ANSWER instead of hanging.
    """

    def __init__(self, script: list[str]):
        self._script = list(script)

    def complete(self, messages: list[Message]) -> str:
        if self._script:
            return self._script.pop(0)
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        return f"ANSWER: Done. (script exhausted; last request was: {last_user[:80]})"
