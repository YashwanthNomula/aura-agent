"""Command line:  python -m aura.cli [flags] "your task"   (no args -> interactive shell).

Flags (must come before the task):
  --quiet   one-shot mode, print only the final answer
  --json    implies --quiet, emit one JSON object with task/answer/backend/duration_ms
"""
from __future__ import annotations

import os
import sys

from .agent import Agent
from .llm import OpenAICompatibleLLM, ScriptedLLM
from .memory import Memory


def build_llm(quiet: bool = False):
    backend = os.environ.get("AURA_LLM", "mock").lower()
    if backend == "mock":
        if not quiet:
            print("(demo mode: scripted model, no API key needed — "
                  "set AURA_LLM=openai for a real model)")
        return ScriptedLLM([])
    return OpenAICompatibleLLM()


def _split_flags(argv: list[str]) -> tuple[bool, bool, list[str]]:
    """Split leading CLI flags from the task.

    Only exact ``--quiet``/``--json`` tokens *before* the task are flags;
    ``--`` ends flag parsing. A task token that merely starts with ``--``
    is treated as part of the task.
    """
    quiet = json_mode = False
    args = list(argv)
    while args and args[0] in ("--quiet", "--json", "--"):
        flag = args.pop(0)
        if flag == "--":
            break
        if flag == "--quiet":
            quiet = True
        else:  # --json
            json_mode = True
            quiet = True
    return quiet, json_mode, args


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    quiet, json_mode, args = _split_flags(argv)
    backend = os.environ.get("AURA_LLM", "mock").lower()

    if json_mode and args:
        import json
        import time

        agent = Agent(build_llm(quiet=True), memory=Memory(), verbose=False)
        task = " ".join(args)
        start = time.perf_counter()
        answer = agent.run(task)
        duration_ms = int((time.perf_counter() - start) * 1000)
        print(json.dumps({"task": task, "answer": answer,
                          "backend": backend, "duration_ms": duration_ms}))
        return 0

    if quiet and args:
        agent = Agent(build_llm(quiet=True), memory=Memory(), verbose=False)
        print(agent.run(" ".join(args)))
        return 0

    # Default behavior: banner, verbose agent chatter, interactive shell.
    agent = Agent(build_llm(), memory=Memory())
    if args:
        print(agent.run(" ".join(args)))
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
