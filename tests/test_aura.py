import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aura import tools
from aura.agent import Agent
from aura.llm import ScriptedLLM
from aura.memory import Memory


def test_write_read_roundtrip(tmp_path):
    p = str(tmp_path / "hello.txt")
    assert "Wrote" in tools.run_tool("write_file", {"path": p, "content": "hi"})
    assert tools.run_tool("read_file", {"path": p}) == "hi"


def test_unknown_tool():
    assert "unknown tool" in tools.run_tool("nope", {})


def test_python_eval():
    assert tools.run_tool("python_eval", {"code": "print(6 * 7)"}) == "42"


def test_shell_refuses_dangerous():
    assert "refused" in tools.run_tool("run_shell", {"command": "rm -rf /tmp/x"})


def test_agent_loop(tmp_path):
    p = str(tmp_path / "out.txt")
    script = [
        f'THOUGHT: write it\nACTION: write_file {{"path": "{p}", "content": "wow"}}',
        "THOUGHT: done\nANSWER: file written",
    ]
    agent = Agent(ScriptedLLM(script), memory=Memory(path=str(tmp_path / "m.json")),
                  verbose=False)
    assert agent.run("write wow") == "file written"
    assert open(p).read() == "wow"


def test_memory_roundtrip(tmp_path):
    mp = str(tmp_path / "mem.json")
    m = Memory(path=mp)
    m.remember("user", "Yashwanth")
    assert Memory(path=mp).recall("user") == "Yashwanth"
