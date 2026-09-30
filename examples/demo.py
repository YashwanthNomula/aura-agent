"""Zero-key demo: the agent writes a file, reads it back, and answers.

Run:  python examples/demo.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aura.agent import Agent
from aura.llm import ScriptedLLM
from aura.memory import Memory

SCRIPT = [
    'THOUGHT: I need a haiku about recursion saved to /tmp/aura_haiku.txt, '
    'then I should verify it.\n'
    'ACTION: write_file {"path": "/tmp/aura_haiku.txt", "content": '
    '"a function calls itself,\\ndreaming down the mirrored stairs —\\nbase case brings me home."}',
    'THOUGHT: Written. Now read it back to confirm the contents.\n'
    'ACTION: read_file {"path": "/tmp/aura_haiku.txt"}',
    "ANSWER: Done! Wrote a recursion haiku to /tmp/aura_haiku.txt "
    "and verified it reads back correctly.",
]

if __name__ == "__main__":
    agent = Agent(ScriptedLLM(SCRIPT), memory=Memory(path="/tmp/aura_demo_memory.json"))
    agent.run("Write a haiku about recursion to /tmp/aura_haiku.txt and verify it.")
