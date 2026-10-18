"""Tests for per-agent tool allowlists (least-privilege confinement)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aura import tools
from aura.agent import Agent
from aura.llm import ScriptedLLM
from aura.memory import Memory


def test_describe_tools_filtered():
    text = tools.describe_tools(only=["read_file"])
    assert "read_file" in text
    assert "write_file" not in text
    assert "run_shell" not in text


def test_describe_tools_default_lists_all():
    assert "write_file" in tools.describe_tools()
    assert "run_shell" in tools.describe_tools()


def test_run_tool_refuses_disallowed(tmp_path):
    p = str(tmp_path / "nope.txt")
    out = tools.run_tool("write_file", {"path": p, "content": "x"},
                         allowed=["read_file"])
    assert "not allowed" in out
    assert not os.path.exists(p)


def test_run_tool_allowed_still_works(tmp_path):
    p = str(tmp_path / "ok.txt")
    assert "Wrote" in tools.run_tool(
        "write_file", {"path": p, "content": "hi"}, allowed=["write_file"])
    assert open(p).read() == "hi"


def test_agent_unknown_allowed_tool_raises():
    with pytest.raises(ValueError, match="unknown tool"):
        Agent(ScriptedLLM([]), allowed_tools=["read_file", "teleport"])


def test_agent_allowlist_blocks_side_effect_tool(tmp_path):
    p = str(tmp_path / "blocked.txt")
    script = [
        f'THOUGHT: try writing\\nACTION: write_file {{"path": "{p}", "content": "evil"}}',
        "THOUGHT: refused, giving up\\nANSWER: blocked as expected",
    ]
    agent = Agent(ScriptedLLM(script),
                  memory=Memory(path=str(tmp_path / "m.json")),
                  verbose=False, allowed_tools=["read_file"])
    assert agent.run("write something") == "blocked as expected"
    assert not os.path.exists(p)


def test_agent_allowlist_allows_listed_tool(tmp_path):
    p = str(tmp_path / "ok.txt")
    script = [
        f'THOUGHT: write it\\nACTION: write_file {{"path": "{p}", "content": "wow"}}',
        "THOUGHT: done\\nANSWER: file written",
    ]
    agent = Agent(ScriptedLLM(script),
                  memory=Memory(path=str(tmp_path / "m.json")),
                  verbose=False, allowed_tools=["write_file"])
    assert agent.run("write wow") == "file written"
    assert open(p).read() == "wow"


def test_agent_default_allows_everything(tmp_path):
    # No allowlist: historical behavior, all tools available.
    p = str(tmp_path / "out.txt")
    script = [
        f'THOUGHT: write it\\nACTION: write_file {{"path": "{p}", "content": "wow"}}',
        "THOUGHT: done\\nANSWER: file written",
    ]
    agent = Agent(ScriptedLLM(script),
                  memory=Memory(path=str(tmp_path / "m.json")),
                  verbose=False)
    assert agent.run("write wow") == "file written"
    assert open(p).read() == "wow"
