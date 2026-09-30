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

Adding a tool is one decorator:

```python
from aura.tools import tool

@tool("roll_dice", "Roll an n-sided die.", {"sides": {"type": "integer"}})
def roll_dice(sides: int = 6) -> str:
    import random
    return str(random.randint(1, sides))
```

The agent discovers it automatically on the next run.

## Project layout

```
aura/
├── aura/
│   ├── agent.py     # the ReAct loop
│   ├── tools.py     # tool registry + built-ins
│   ├── llm.py       # OpenAI-compatible client + scripted demo model
│   ├── memory.py    # persistent facts + episode log
│   └── cli.py       # `python -m aura.cli "task"`
├── examples/demo.py # zero-key end-to-end demo
└── tests/           # pytest suite
```

## Run the tests

```bash
pip install pytest && python -m pytest tests/ -q
```

## Roadmap

- [ ] Streaming thoughts to a web UI
- [ ] Tool sandboxing with containers
- [ ] Multi-agent crews (planner + workers)
- [ ] Vector-memory recall

---

Built to be read. If the ReAct loop clicks for you after 10 minutes with this
code, it did its job. PRs welcome.
