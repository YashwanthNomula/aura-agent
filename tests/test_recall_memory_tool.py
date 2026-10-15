import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aura import tools
from aura.agent import Agent
from aura.memory import Memory


def _seeded_memory(tmp_path):
    m = Memory(path=str(tmp_path / "m.json"))
    m.remember("pet", "has a cat named Milo")
    m.remember("work", "works on quarterly financial spreadsheets")
    return m


def test_recall_memory_returns_relevant_facts(tmp_path):
    m = _seeded_memory(tmp_path)
    out = tools.run_tool("recall_memory", {"query": "what animal does the user own"}, memory=m)
    assert "fact:pet" in out
    assert "cat named Milo" in out


def test_recall_memory_empty_memory(tmp_path):
    m = Memory(path=str(tmp_path / "empty.json"))
    out = tools.run_tool("recall_memory", {"query": "anything at all"}, memory=m)
    assert out == "(no memories matched that query)"
    assert "ERROR" not in out


def test_recall_memory_empty_query(tmp_path):
    m = _seeded_memory(tmp_path)
    assert "ERROR" in tools.run_tool("recall_memory", {"query": "   "}, memory=m)


def test_recall_memory_respects_k(tmp_path):
    m = Memory(path=str(tmp_path / "m.json"))
    for i in range(5):
        m.remember(f"note{i}", f"all about pizza toppings number {i}")
    out = tools.run_tool("recall_memory", {"query": "pizza", "k": 2}, memory=m)
    assert len(out.strip().splitlines()) == 2


def test_recall_memory_via_memory_scope(tmp_path):
    m = _seeded_memory(tmp_path)
    with tools.memory_scope(m):
        out = tools.run_tool("recall_memory", {"query": "what animal does the user own"})
    assert "cat named Milo" in out
    # scope restores: no bound memory leaks out of the block
    assert tools._BOUND_MEMORY is None


def test_recall_memory_binding_does_not_leak(tmp_path):
    m = _seeded_memory(tmp_path)
    tools.run_tool("recall_memory", {"query": "pet"}, memory=m)
    assert tools._BOUND_MEMORY is None


def test_recall_memory_registered():
    assert "recall_memory" in tools.REGISTRY
    assert "recall_memory" in tools.describe_tools()


class _RecallThenAnswerLLM:
    """First turn recalls memory; second turn answers from the observation."""

    def __init__(self):
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        if self.calls == 1:
            return ('THOUGHT: I should check memory first.\n'
                    'ACTION: recall_memory {"query": "what animal does the user own"}')
        last_obs = next(m.content for m in reversed(messages) if m.role == "user")
        assert "Milo" in last_obs, "agent never saw the recalled fact"
        return "ANSWER: The user owns a cat named Milo."


def test_agent_run_uses_recall_memory(tmp_path):
    m = _seeded_memory(tmp_path)
    agent = Agent(_RecallThenAnswerLLM(), memory=m, verbose=False)
    assert agent.run("What pet does the user own?") == "The user owns a cat named Milo."
