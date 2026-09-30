"""The ReAct loop: reason -> act -> observe, until the task is done."""
from __future__ import annotations

import json
import re

from .llm import LLMClient, Message
from .memory import Memory
from .tools import describe_tools, run_tool

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
    """Runs the thought -> action -> observation loop against an LLM backend."""

    def __init__(self, llm: LLMClient, max_steps: int = 10,
                 memory: Memory | None = None, verbose: bool = True):
        self.llm = llm
        self.max_steps = max_steps
        self.memory = memory or Memory()
        self.verbose = verbose

    def _say(self, text: str) -> None:
        if self.verbose:
            print(text, flush=True)

    def run(self, task: str) -> str:
        system = SYSTEM_PROMPT.format(tools=describe_tools(), memory=self.memory.context())
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
                obs = run_tool(name, args)
            self._say(f"[step {step}] observed: {obs[:200]}")
            messages.append(Message("user", f"OBSERVATION: {obs}"))

        answer = "I ran out of steps. Partial progress is in the transcript above."
        self.memory.log_episode(task, answer)
        return answer
