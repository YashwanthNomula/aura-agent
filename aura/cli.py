"""Command line:  python -m aura.cli "your task"   (no args -> interactive shell)."""
from __future__ import annotations

import os
import sys

from .agent import Agent
from .llm import OpenAICompatibleLLM, ScriptedLLM
from .memory import Memory


def build_llm():
    backend = os.environ.get("AURA_LLM", "mock").lower()
    if backend == "mock":
        print("(demo mode: scripted model, no API key needed — "
              "set AURA_LLM=openai for a real model)")
        return ScriptedLLM([])
    return OpenAICompatibleLLM()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    agent = Agent(build_llm(), memory=Memory())
    if argv:
        print(agent.run(" ".join(argv)))
        return 0
    print("AURA shell — type a task, or 'quit'.")
    while True:
        try:
            task = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if task.lower() in {"quit", "exit"}:
            break
        if task:
            print(agent.run(task))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
