"""Multi-agent crews: a planner decomposes the task, workers execute the parts.

A single Agent is good at a focused job. A crew is good at a *big* job: a
tool-less planner breaks the task into named subtasks, each subtask runs on
a worker Agent confined to a least-privilege tool allowlist (a read-only
researcher, a writer, ...), and the planner synthesizes the final answer
from the workers' results.

All agents share one Memory, so subtask episodes are logged in one place.
Each agent gets its own LLM backend from ``llm_factory`` — backends are
stateful (ScriptedLLM consumes its script), so they must never be shared.
A malformed planner plan fails closed with ValueError: silently dropping or
reordering subtasks would be worse than refusing to run.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from .agent import Agent
from .llm import LLMClient
from .memory import Memory
from .tools import REGISTRY

PLAN_PROMPT = """\
You are the PLANNER of an agent crew. Break the task below into 2-4 concrete
subtasks and assign each to one of these workers: {workers}.

Rules:
- Do NOT call any tools. You have none. Just plan.
- Reply with exactly one line:  ANSWER: <JSON array>
- Each subtask is an object:
  {{"name": "short-slug", "worker": "worker-name",
    "brief": "one-paragraph instructions for the worker"}}
- "worker" must be one of the worker names above (omit it to round-robin).
- Keep each brief self-contained: the worker sees only the brief, not your
  reasoning.

TASK: {task}"""

SYNTH_PROMPT = """\
You are the PLANNER of an agent crew. The workers finished their subtasks.
Write the final answer to the original task, using their results below.
Reply with exactly:  ANSWER: <final answer>

ORIGINAL TASK: {task}

WORKER RESULTS:
{results}"""


def default_workers() -> list["WorkerSpec"]:
    """A sensible default crew: a read-only researcher and a writer."""
    return [
        WorkerSpec("researcher", allowed_tools=[
            "read_file", "list_dir", "summarize_file",
            "web_fetch", "web_search", "recall_memory"]),
        WorkerSpec("writer", allowed_tools=[
            "write_file", "read_file", "todo_list",
            "python_eval", "recall_memory"]),
    ]


@dataclass
class WorkerSpec:
    """One crew member: a name plus its least-privilege tool allowlist."""
    name: str
    allowed_tools: list[str] | None = None
    max_steps: int = 10

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("WorkerSpec needs a non-empty name")
        if self.max_steps < 1:
            raise ValueError(
                f"WorkerSpec {self.name!r}: max_steps must be >= 1")


@dataclass
class CrewResult:
    """Everything a crew run produced."""
    task: str
    plan: list[dict]                      # the planner's subtask list
    answers: dict[str, str] = field(default_factory=dict)  # name -> answer
    final: str = ""                       # the planner's synthesis
    assignments: dict[str, str] = field(default_factory=dict)  # subtask -> worker actually used


class Crew:
    """A planner + a team of worker agents.

    ``llm_factory`` is called with no arguments and must return a *fresh*
    LLMClient each time (one per agent).
    """

    def __init__(self, llm_factory: Callable[[], LLMClient],
                 workers: list[WorkerSpec] | None = None,
                 planner_max_steps: int = 6,
                 memory: Memory | None = None,
                 verbose: bool = True):
        if not callable(llm_factory):
            raise ValueError("llm_factory must be callable")
        self._llm_factory = llm_factory
        specs = list(workers) if workers is not None else default_workers()
        if not specs:
            raise ValueError("a crew needs at least one worker")
        names = [s.name for s in specs]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate worker name(s): {', '.join(dupes)}")
        for s in specs:
            if s.allowed_tools is not None:
                unknown = sorted(set(s.allowed_tools) - set(REGISTRY))
                if unknown:
                    raise ValueError(
                        f"worker {s.name!r}: unknown tool(s): "
                        f"{', '.join(unknown)}")
        self._specs = {s.name: s for s in specs}
        self._order = names
        self._agents: dict[str, Agent] = {}
        self.memory = memory or Memory()
        self.verbose = verbose
        self._planner = Agent(self._llm_factory(), allowed_tools=[],
                              max_steps=max(planner_max_steps, 1),
                              memory=self.memory, verbose=verbose)

    def _say(self, text: str) -> None:
        if self.verbose:
            print(text, flush=True)

    def _worker(self, name: str) -> Agent:
        if name not in self._agents:
            spec = self._specs[name]
            self._agents[name] = Agent(
                self._llm_factory(), allowed_tools=spec.allowed_tools,
                max_steps=spec.max_steps, memory=self.memory,
                verbose=self.verbose)
        return self._agents[name]

    def plan(self, task: str) -> list[dict]:
        """Ask the planner to decompose *task*; return the subtask list.

        Raises ValueError when the planner's answer is not a usable plan.
        """
        prompt = PLAN_PROMPT.format(
            workers=", ".join(self._order), task=task)
        answer = self._planner.run(prompt)
        match = re.search(r"\[.*\]", answer, re.S)
        if not match:
            raise ValueError(
                f"planner did not return a JSON plan: {answer[:200]!r}")
        try:
            plan = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"planner returned invalid JSON: {exc}") from exc
        if not isinstance(plan, list) or not plan:
            raise ValueError("planner returned an empty or non-list plan")
        seen: set[str] = set()
        for i, item in enumerate(plan, 1):
            if not isinstance(item, dict):
                raise ValueError(
                    f"planner subtask #{i} is not an object: {item!r}")
            name = str(item.get("name") or f"subtask-{i}").strip()
            item["name"] = name
            if name in seen:
                raise ValueError(
                    f"planner returned duplicate subtask name {name!r}")
            seen.add(name)
            if not str(item.get("brief", "")).strip():
                raise ValueError(
                    f"planner subtask {name!r} is missing its 'brief'")
            worker = item.get("worker")
            if worker is not None and worker not in self._specs:
                raise ValueError(
                    f"planner assigned unknown worker {worker!r}; "
                    f"known: {', '.join(self._order)}")
        return plan

    def run(self, task: str) -> CrewResult:
        """Plan, dispatch to workers, synthesize. Returns a CrewResult."""
        self._say(f"\nCREW starting: {task}\n")
        plan = self.plan(task)
        self._say("plan: " + ", ".join(
            f"{s['name']}->{s.get('worker') or 'round-robin'}" for s in plan))

        answers: dict[str, str] = {}
        assignments: dict[str, str] = {}
        rr = 0
        for item in plan:
            worker_name = item.get("worker") or self._order[rr % len(self._order)]
            if item.get("worker") is None:
                rr += 1
            assignments[item["name"]] = worker_name
            self._say(f"\n[crew] dispatching '{item['name']}' "
                      f"to worker '{worker_name}'")
            worker = self._worker(worker_name)
            answer = worker.run(
                f"SUBTASK '{item['name']}' (part of a larger crew task).\n\n"
                f"{item['brief']}\n\n"
                f"Original task for context: {task}")
            answers[item["name"]] = answer

        results = "\n\n".join(
            f"### {name}\n{answers[name]}" for name in answers)
        self._say("\n[crew] synthesizing final answer")
        final = self._planner.run(SYNTH_PROMPT.format(task=task,
                                                      results=results))
        self.memory.log_episode(f"[crew] {task}", final)
        return CrewResult(task=task, plan=plan, answers=answers, final=final,
                          assignments=assignments)
