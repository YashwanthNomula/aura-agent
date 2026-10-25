# AURA — give your LLM hands 🛰️

**AURA** (*Autonomous Understanding, Reasoning & Action*) is a tiny, fully-working
**AI agent** in pure Python. Give it a task in plain English and it reasons,
picks tools, runs them, observes the results, and keeps going until the job is
done — the same ReAct loop that powers production agents, in ~300 lines you can
actually read.

No API key needed to try it. No dependencies. No framework magic.

## Watch it work

```
$ python examples/demo.py

AURA starting: Write a haiku about recursion to /tmp/aura_haiku.txt and verify it.

[step 1] thinking: I need a haiku about recursion saved to /tmp/aura_haiku.txt, then I should verify it.
[step 1] acting: write_file({"path": "/tmp/aura_haiku.txt", ...})
[step 1] observed: Wrote 93 chars to /tmp/aura_haiku.txt
[step 2] thinking: Written. Now read it back to confirm the contents.
[step 2] acting: read_file({"path": "/tmp/aura_haiku.txt"})
[step 2] observed: a function calls itself,
dreaming down the mirrored stairs —
base case brings me home.
Done: Done! Wrote a recursion haiku to /tmp/aura_haiku.txt and verified it reads back correctly.
```

It didn't just generate text — it **acted in the world** (wrote a file), then
**checked its own work** (read it back). That's the whole idea.

## Quickstart

```bash
git clone <this-repo> && cd aura

# zero-key demo (scripted model, runs offline)
python examples/demo.py

# interactive shell with a real model
export AURA_LLM=openai AURA_API_KEY=sk-...   # any OpenAI-compatible endpoint
export AURA_BASE_URL=https://api.openai.com/v1  # or Ollama, vLLM, Together...
export AURA_MODEL=gpt-4o-mini
python -m aura.cli "Find the largest file in ~/Downloads and summarize it"
```

## How it works

```
                    ┌──────────────────────────────┐
                    │            AGENT             │
  TASK ──▶         │  THOUGHT → ACTION → OBSERVE  │ ──▶ ANSWER
                    └──────────────┬───────────────┘
                                   │ tools
            ┌──────────────────────┼──────────────────────┐
            ▼                      ▼                      ▼
        🗂 read/write          💻 shell /             🌐 web
           list files             python eval           search
```

1. **Reason** — the model writes a `THOUGHT` and an `ACTION` (tool + JSON args).
2. **Act** — AURA executes the tool in a sandbox with timeouts and guardrails.
3. **Observe** — the result feeds back into the loop. Repeat until `ANSWER`.
4. **Remember** — facts and every run are persisted to `~/.aura/memory.json`.

## The toolbox

| Tool | What it does |
|---|---|
| `read_file` / `write_file` / `list_dir` | Work with the filesystem |
| `run_shell` | Run shell commands (dangerous patterns refused, timeouts enforced) |
| `python_eval` | Execute Python, capture output |
| `web_search` | Live web snippets, no API key |
| `web_fetch` | Fetch a URL's page text, HTML stripped (http/https only) |
| `todo_list` | In-memory todo list: add, list, complete, remove |
| `summarize_file` | File summary: counts, head/tail lines, top keywords |
| `recall_memory` | Semantic recall over long-term facts + past episodes (mid-task) |

Adding a tool is one decorator:

```python
from aura.tools import tool

@tool("roll_dice", "Roll an n-sided die.", {"sides": {"type": "integer"}})
def roll_dice(sides: int = 6) -> str:
    import random
    return str(random.randint(1, sides))
```

The agent discovers it automatically on the next run.

### Least-privilege tool allowlists

Confine an agent to the tools its task actually needs — a first step
toward full sandboxing. Disallowed tools are refused before they run,
and the system prompt only lists the tools the agent may use:

```python
from aura.agent import Agent
# Read-only agent: can never write files or run shell commands.
agent = Agent(llm, allowed_tools=["read_file", "list_dir", "recall_memory"])
```

Unknown names in `allowed_tools` raise `ValueError` (fail-closed).
Omit it and every registered tool is available, as before.

## Project layout

```
aura/
├── aura/
│   ├── agent.py     # the ReAct loop
│   ├── crew.py      # multi-agent crews: planner + least-privilege workers
│   ├── tools.py     # tool registry + built-ins
│   ├── llm.py       # OpenAI-compatible client + scripted demo model
│   ├── memory.py    # persistent facts + episode log
│   ├── vector_memory.py # zero-dep semantic recall (hashed embeddings)
│   └── cli.py       # `python -m aura.cli "task"`
├── examples/demo.py # zero-key end-to-end demo
└── tests/           # pytest suite
```

## Run the tests

```bash
pip install pytest && python -m pytest tests/ -q
```

## Memory and semantic recall

AURA remembers long-term facts (`memory.remember("pet", "cat named Milo")`)
and logs every run as an episode in `~/.aura/memory.json`. `memory.recall(key)`
is exact lookup; `memory.recall_semantic(query, k=5)` is semantic recall:
memories are embedded with zero-dependency hashed token/bigram vectors
(`aura/vector_memory.py`, no downloads, no GPU) and ranked by cosine
similarity, so a rephrased query still finds the right memory where keyword
search misses. From the shell:

```bash
python -m aura.cli --recall "what animal do I own"
# [0.243] fact:pet: pet: has a cat named Milo
```

## Multi-agent crews (planner + workers)

One agent is good at a focused job; a crew is good at a *big* job. A tool-less
planner decomposes the task into named subtasks, each subtask runs on a
worker `Agent` confined to a least-privilege tool allowlist, and the planner
synthesizes the final answer from the workers' results. All agents share one
memory, so every subtask episode is logged in one place.

```python
from aura import Crew, WorkerSpec, Memory
from aura.llm import OpenAICompatibleLLM

crew = Crew(OpenAICompatibleLLM, memory=Memory(), workers=[
    WorkerSpec("researcher",
               allowed_tools=["read_file", "list_dir", "web_search"]),
    WorkerSpec("writer", allowed_tools=["write_file", "todo_list"]),
])
result = crew.run("research drift-detection libraries and draft a summary")
print(result.final)          # the planner's synthesis
print(result.answers)        # per-subtask worker answers
print(result.assignments)    # subtask -> worker actually used (round-robin resolved)
```

A malformed planner plan (not JSON, empty, duplicate names, unknown worker)
fails closed with `ValueError` — silently dropping subtasks would be worse
than refusing to run. From the shell:

```bash
python -m aura.cli crew "explain the repo"        # scripted demo, no key
AURA_LLM=openai python -m aura.cli crew "your task"
```

## Roadmap

- [ ] Streaming thoughts to a web UI
- [ ] Tool sandboxing with containers
- [x] Multi-agent crews (planner + workers) — `aura/crew.py`: tool-less planner
      decomposes the task into named subtasks, each dispatched to a worker
      Agent confined to a least-privilege allowlist; planner synthesizes the
      final answer; shared memory logs every subtask episode; malformed
      plans fail closed
- [x] Vector-memory recall

---

Built to be read. If the ReAct loop clicks for you after 10 minutes with this
code, it did its job. PRs welcome.
