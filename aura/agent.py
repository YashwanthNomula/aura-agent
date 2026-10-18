"""The ReAct loop: reason -> act -> observe, until the task is done."""
from __future__ import annotations

import json
import re

from .llm import LLMClient, Message
from .memory import Memory
from .tools import REGISTRY, describe_tools, run_tool

SYSTEM_PROMPT = """You are AURA, an autonomous agent. You solve tasks by reasoning and using tools.

On every turn reply in EXACTLY one of these formats:

THOUGHT: <your private reasoning, one or two sentences>
ACTION: <tool_name> <JSON object of arguments>

...or, when the task is fully complete:

ANSWER: <final response to the user>

Rules:
- Never invent tool results. Only learn from OBSERVATION messages.
- Keep going until the task is done or you are stuck; then ANSWER with what you achieved.
- Prefer the smallest next step that makes progress.

Available tools:
{tools}

Long-term memory:
{memory}
"""


def _parse_action(text: str) -> tuple[str | None, dict]:
    m = re.search(r"ACTION:\s*(\w+)\s*(\{.*\})?", text, re.S)
    if not m:
        return None, {}
    raw = (m.group(2) or "{}").strip()
    try:
        args = json.loads(raw)
    except json.JSONDecodeError:
        args = {}
    return m.group(1), args if isinstance(args, dict) else {}


class Agent:
    """Runs the thought -> action -> observation loop against an LLM backend.

    ``allowed_tools`` confines the agent to a subset of the tool registry
    (least privilege — a read-only agent gets ``["read_file", "list_dir",
    "recall_memory"]`` and can never call ``write_file``/``run_shell``).
    Unknown names raise ValueError (fail-closed). ``None`` (default)
    allows every registered tool, and the system prompt lists only the
    tools the agent may actually use.
    """

    def __init__(self, llm: LLMClient, max_steps: int = 10,
                 memory: Memory | None = None, verbose: bool = True,
                 allowed_tools: list[str] | None = None):
        self.llm = llm
        self.max_steps = max_steps
        self.memory = memory or Memory()
        self.verbose = verbose
        if allowed_tools is not None:
            unknown = sorted(set(allowed_tools) - set(REGISTRY))
            if unknown:
                raise ValueError(
                    f"unknown tool(s) in allowed_tools: {', '.join(unknown)}")
        self.allowed_tools = list(allowed_tools) if allowed_tools is not None else None

    def _say(self, text: str) -> None:
        if self.verbose:
            print(text, flush=True)

    def run(self, task: str) -> str:
        system = SYSTEM_PROMPT.format(
            tools=describe_tools(only=self.allowed_tools),
            memory=self.memory.context())
        messages = [Message("system", system), Message("user", f"TASK: {task}")]
        self._say(f"\nAURA starting: {task}\n")

        for step in range(1, self.max_steps + 1):
            reply = self.llm.complete(messages)
            messages.append(Message("assistant", reply))

            if "ANSWER:" in reply:
                answer = reply.split("ANSWER:", 1)[1].strip()
                self._say(f"\nDone: {answer}")
                self.memory.log_episode(task, answer)
                return answer

            name, args = _parse_action(reply)
            thought = re.search(r"THOUGHT:\s*(.+?)(?=ACTION:|$)", reply, re.S)
            self._say(f"[step {step}] thinking: {(thought.group(1).strip()[:120] if thought else '')}")
            if not name:
                obs = "ERROR: no ACTION found. Reply with THOUGHT/ACTION or ANSWER."
            else:
                self._say(f"[step {step}] acting: {name}({json.dumps(args)[:100]})")
                obs = run_tool(name, args, memory=self.memory,
                               allowed=self.allowed_tools)
            self._say(f"[step {step}] observed: {obs[:200]}")
            messages.append(Message("user", f"OBSERVATION: {obs}"))

        answer = "I ran out of steps. Partial progress is in the transcript above."
        self.memory.log_episode(task, answer)
        return answer
