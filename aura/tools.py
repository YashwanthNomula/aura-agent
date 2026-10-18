"""Tool registry. A tool is just a documented Python function.

Add your own with the @tool decorator — the agent picks it up automatically.
"""
from __future__ import annotations

import builtins
import io
import os
import re
import subprocess
import urllib.parse
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from collections import Counter
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., str]


REGISTRY: dict[str, Tool] = {}


def tool(name: str, description: str, parameters: dict[str, Any]):
    """Register a function as an agent tool."""
    def deco(func: Callable[..., str]) -> Callable[..., str]:
        REGISTRY[name] = Tool(name, description, parameters, func)
        return func
    return deco


def describe_tools(only: list[str] | None = None) -> str:
    lines = []
    for t in REGISTRY.values():
        if only is not None and t.name not in only:
            continue
        params = ", ".join(f"{k} ({v.get('type', '?')})" for k, v in t.parameters.items())
        lines.append(f"- {t.name}({params}): {t.description}")
    return "\n".join(lines)


# ------------------------------------------------------- memory-aware tools
# Most tools are stateless, but a few (recall_memory) need the agent's
# Memory. Rather than a singleton, the active memory is bound explicitly:
# pass memory= to run_tool (what the Agent does), or use memory_scope()
# around a call. With neither, a default Memory() is used.

_BOUND_MEMORY: Any = None


@contextmanager
def memory_scope(memory: Any) -> Iterator[None]:
    """Temporarily bind *memory* as the active memory for tool calls."""
    global _BOUND_MEMORY
    prev = _BOUND_MEMORY
    _BOUND_MEMORY = memory
    try:
        yield
    finally:
        _BOUND_MEMORY = prev


def _active_memory() -> Any:
    if _BOUND_MEMORY is not None:
        return _BOUND_MEMORY
    from .memory import Memory  # local import: memory.py never imports tools
    return Memory()


def run_tool(name: str, args: dict[str, Any], memory: Any = None,
             allowed: list[str] | None = None) -> str:
    """Run a registered tool by name.

    ``allowed`` is an allowlist (e.g. from ``Agent(allowed_tools=...)``):
    when given, any tool outside it is refused without running, so an
    agent can be confined to the least privilege it needs. ``None`` means
    every registered tool is available (the historical default).
    """
    if allowed is not None and name not in allowed:
        return (f"ERROR: tool '{name}' is not allowed for this agent. "
                f"Allowed: {', '.join(sorted(allowed)) or '(none)'}")
    t = REGISTRY.get(name)
    if t is None:
        return f"ERROR: unknown tool '{name}'. Available: {', '.join(sorted(REGISTRY))}"
    try:
        if memory is not None:
            with memory_scope(memory):
                return str(t.func(**args))
        return str(t.func(**args))
    except TypeError as e:
        return f"ERROR: bad arguments for {name}: {e}"
    except Exception as e:  # tools must never crash the reasoning loop
        return f"ERROR: {name} failed: {e}"


# ---------------------------------------------------------------- built-in tools

@tool("read_file", "Read a text file and return its contents.",
      {"path": {"type": "string"}})
def read_file(path: str) -> str:
    with open(os.path.expanduser(path), "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    if len(content) > 8000:
        content = content[:8000] + "\n...[truncated]"
    return content


@tool("write_file", "Write text content to a file (creates parent dirs).",
      {"path": {"type": "string"}, "content": {"type": "string"}})
def write_file(path: str, content: str) -> str:
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Wrote {len(content)} chars to {path}"


@tool("list_dir", "List files in a directory.", {"path": {"type": "string"}})
def list_dir(path: str) -> str:
    path = os.path.expanduser(path)
    return "\n".join(sorted(os.listdir(path)))


@tool("run_shell", "Run a shell command; return stdout/stderr (truncated).",
      {"command": {"type": "string"}, "timeout": {"type": "integer"}})
def run_shell(command: str, timeout: int = 20) -> str:
    banned = ["rm -rf", "mkfs", ":(){", "dd if=", "> /dev/"]
    if any(b in command for b in banned):
        return "ERROR: command looks dangerous, refused."
    p = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
    out = (p.stdout + p.stderr).strip()
    if len(out) > 4000:
        out = out[:4000] + "\n...[truncated]"
    return out or f"(exit {p.returncode}, no output)"


@tool("python_eval", "Execute Python code and return printed output.",
      {"code": {"type": "string"}})
def python_eval(code: str) -> str:
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            exec(compile(code, "<agent>", "exec"), {"__builtins__": builtins})
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"
    return buf.getvalue().strip() or "(no output)"


@tool("web_search", "Search the web; return short text snippets.",
      {"query": {"type": "string"}})
def web_search(query: str) -> str:
    url = "https://lite.duckduckgo.com/lite/?" + urllib.parse.urlencode({"q": query})
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        return f"ERROR: search failed: {e}"
    snippets = re.findall(r'class="result-snippet"[^>]*>(.*?)</a>', html, re.S)
    clean = [re.sub(r"<[^>]+>", "", s).strip() for s in snippets[:5]]
    return "\n---\n".join(c for c in clean if c) or "No snippets found."


# ---------------------------------------------------------------- helpers

_TAG_RE = re.compile(r"<[^>]+>")
_RUN_RE = re.compile(r"[ \t\xa0]+")
_BLANK_RE = re.compile(r"\n[ \t]*\n+")


@tool("web_fetch", "Fetch an http(s) URL and return its page text (HTML stripped).",
      {"url": {"type": "string"}, "max_chars": {"type": "integer"}})
def web_fetch(url: str, max_chars: int = 8000) -> str:
    scheme = urllib.parse.urlparse(url).scheme.lower()
    if scheme not in ("http", "https"):
        return "ERROR: only http:// and https:// URLs are allowed."
    max_chars = min(max(max_chars, 100), 50000)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (AURA agent)"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read(2_000_000)  # hard cap on download size
    except Exception as e:
        return f"ERROR: fetch failed: {e}"
    text = raw.decode("utf-8", errors="replace")
    text = _TAG_RE.sub(" ", text)
    text = _RUN_RE.sub(" ", text)
    text = _BLANK_RE.sub("\n\n", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"
    return text or "(page contained no readable text)"


_TODOS: list[dict[str, Any]] = []
_TODO_NEXT_ID = 1


@tool("todo_list", "Manage a simple in-memory todo list.",
      {"action": {"type": "string"}, "text": {"type": "string"}, "id": {"type": "integer"}})
def todo_list(action: str, text: str = "", id: int = 0) -> str:
    """Actions: add (needs text), list, complete (needs id), remove (needs id)."""
    global _TODO_NEXT_ID
    action = action.lower().strip()
    if action == "add":
        if not text.strip():
            return "ERROR: 'add' needs non-empty text."
        item = {"id": _TODO_NEXT_ID, "text": text.strip(), "done": False}
        _TODOS.append(item)
        _TODO_NEXT_ID += 1
        return f"Added todo #{item['id']}: {item['text']}"
    if action == "list":
        if not _TODOS:
            return "(todo list is empty)"
        return "\n".join(
            f"[{'x' if t['done'] else ' '}] #{t['id']} {t['text']}" for t in _TODOS)
    if action in ("complete", "remove"):
        item = next((t for t in _TODOS if t["id"] == id), None)
        if item is None:
            return f"ERROR: no todo with id {id}."
        if action == "complete":
            item["done"] = True
            return f"Completed todo #{id}."
        _TODOS.remove(item)
        return f"Removed todo #{id}."
    return "ERROR: unknown action. Use add, list, complete, or remove."


_STOPWORDS = frozenset(
    "a an the and or but if of to in on for with as is are was were be been "
    "this that it its they their we you he she not no do does did will would "
    "can could should has have had at by from our your his her them".split())
_HEAD_TAIL = 5
_MAX_SUMMARY_BYTES = 1_000_000


@tool("summarize_file", "Summarize a text file: counts, head/tail lines, top keywords.",
      {"path": {"type": "string"}, "top_n": {"type": "integer"}})
def summarize_file(path: str, top_n: int = 10) -> str:
    path = os.path.expanduser(path)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read(_MAX_SUMMARY_BYTES)
        truncated = bool(f.read(1))
    lines = content.splitlines()
    words = content.split()
    tokens = [w.strip(".,!?;:\"'()[]{}<>").lower() for w in words]
    tokens = [t for t in tokens if len(t) >= 3 and t not in _STOPWORDS and t.isalpha()]
    top_n = min(max(top_n, 1), 30)
    keywords = Counter(tokens).most_common(top_n)
    parts = [
        f"File: {path}",
        f"Lines: {len(lines)}  Words: {len(words)}  Chars: {len(content)}",
        "Top keywords: " + (", ".join(f"{k}({n})" for k, n in keywords) or "(none)"),
        "First lines:",
    ]
    for i, ln in enumerate(lines[:_HEAD_TAIL], 1):
        parts.append(f"  {i}: {ln}")
    parts.append("Last lines:")
    tail = lines[-_HEAD_TAIL:]
    start = len(lines) - len(tail) + 1
    for i, ln in enumerate(tail, start):
        parts.append(f"  {i}: {ln}")
    if truncated:
        parts.append("(file larger than 1MB — summarized the first 1MB)")
    return "\n".join(parts)


@tool("recall_memory", "Recall relevant long-term facts and past episodes from memory "
      "using semantic search. Use it mid-task when the task may depend on something "
      "remembered earlier (preferences, names, past runs).",
      {"query": {"type": "string"}, "k": {"type": "integer"}})
def recall_memory(query: str, k: int = 5) -> str:
    """Semantic recall over the active memory (bound via run_tool's memory=
    argument or memory_scope); falls back to a default Memory() otherwise."""
    if not query or not str(query).strip():
        return "ERROR: recall_memory needs a non-empty query."
    try:
        k = max(1, min(int(k), 20))
    except (TypeError, ValueError):
        k = 5
    try:
        hits = _active_memory().recall_semantic(str(query), k=k)
    except Exception as e:
        return f"ERROR: recall_memory failed: {e}"
    if not hits:
        return "(no memories matched that query)"
    return "\n".join(f"[{score:.3f}] {doc_id}: {text}" for doc_id, text, score in hits)
