"""Command line:  python -m aura.cli [flags] "your task"   (no args -> interactive shell).

Flags (must come before the task):
  --quiet   one-shot mode, print only the final answer
  --json    implies --quiet, emit one JSON object with task/answer/backend/duration_ms
  --recall <query>
            semantic recall over stored memories (facts + past episodes);
            prints top-5 matches ranked by cosine similarity and exits

Crew mode (planner + worker agents, needs AURA_LLM=openai for a real run,
scripted demo otherwise):
  crew "your task"     run the crew on the task; honors --quiet/--json
"""
from __future__ import annotations

import os
import sys

from .agent import Agent
from .crew import Crew, default_workers
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


def _split_flags(argv: list[str]) -> tuple[bool, bool, str | None, list[str]]:
    """Split leading CLI flags from the task.

    Only exact ``--quiet``/``--json``/``--recall <query>`` tokens *before*
    the task are flags; ``--`` ends flag parsing. A task token that merely
    starts with ``--`` is treated as part of the task.
    """
    quiet = json_mode = False
    recall_query: str | None = None
    args = list(argv)
    while args and args[0] in ("--quiet", "--json", "--recall", "--"):
        flag = args.pop(0)
        if flag == "--":
            break
        if flag == "--quiet":
            quiet = True
        elif flag == "--recall":
            recall_query = args.pop(0) if args else ""
        else:  # --json
            json_mode = True
            quiet = True
    return quiet, json_mode, recall_query, args


def _mock_crew_factory():
    """Canned crew scripts so `crew` demos with zero API key.

    The planner agent is reused for planning and the final synthesis, so
    its scripted backend gets two replies; each worker gets its own.
    """
    scripts = iter([
        ['ANSWER: [{"name": "research", "worker": "researcher", '
         '"brief": "Read README.md and list this project\'s main components."}, '
         '{"name": "notes", "worker": "writer", '
         '"brief": "Write a 3-bullet summary of the components to /tmp/aura-crew-notes.txt."}]',
         'ANSWER: AURA has five main components: the ReAct loop (agent.py), '
         'the tool registry (tools.py), LLM backends (llm.py), persistent '
         'memory (memory.py), and the CLI (cli.py). Summary saved to '
         '/tmp/aura-crew-notes.txt.'],
        ['THOUGHT: I will skim the README.\n'
         'ACTION: read_file {"path": "README.md"}',
         'ANSWER: main components: ReAct loop (agent.py), tool registry '
         '(tools.py), LLM backends (llm.py), memory (memory.py), CLI (cli.py)'],
        ['THOUGHT: writing the summary file.\n'
         'ACTION: write_file {"path": "/tmp/aura-crew-notes.txt", '
         '"content": "- ReAct loop in agent.py\\n- tool registry in tools.py\\n- memory in memory.py\\n"}',
         'ANSWER: wrote the 3-bullet summary to /tmp/aura-crew-notes.txt'],
    ])
    return lambda: ScriptedLLM(next(scripts))


def crew_main(argv: list[str], quiet: bool = False, json_mode: bool = False) -> int:
    task = " ".join(argv).strip()
    if not task:
        print('usage: python -m aura.cli crew "task"', file=sys.stderr)
        return 2
    backend = os.environ.get("AURA_LLM", "mock").lower()
    if backend == "mock":
        if not quiet:
            print("(demo mode: scripted crew, no API key needed — "
                  "set AURA_LLM=openai for a real model)")
        factory = _mock_crew_factory()
    else:
        factory = OpenAICompatibleLLM
    crew = Crew(factory, verbose=not quiet, memory=Memory())
    try:
        result = crew.run(task)
    except ValueError as exc:
        print(f"aura crew: error: {exc}", file=sys.stderr)
        return 1
    if json_mode:
        import json
        print(json.dumps({"task": result.task,
                          "plan": result.plan,
                          "assignments": result.assignments,
                          "answers": result.answers,
                          "final": result.final,
                          "backend": backend}))
    else:
        print(result.final)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    quiet, json_mode, recall_query, args = _split_flags(argv)
    backend = os.environ.get("AURA_LLM", "mock").lower()

    if recall_query is not None:
        memory = Memory()
        hits = memory.recall_semantic(recall_query, k=5)
        if not hits:
            print("No memories stored yet.")
        for doc_id, text, score in hits:
            print(f"[{score:.3f}] {doc_id}: {text}")
        return 0

    if args and args[0] == "crew":
        return crew_main(args[1:], quiet=quiet, json_mode=json_mode)

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
