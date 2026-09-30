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
from contextlib import redirect_stdout
from dataclasses import dataclass
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


def describe_tools() -> str:
    lines = []
    for t in REGISTRY.values():
        params = ", ".join(f"{k} ({v.get('type', '?')})" for k, v in t.parameters.items())
        lines.append(f"- {t.name}({params}): {t.description}")
    return "\n".join(lines)


def run_tool(name: str, args: dict[str, Any]) -> str:
    t = REGISTRY.get(name)
    if t is None:
        return f"ERROR: unknown tool '{name}'. Available: {', '.join(sorted(REGISTRY))}"
    try:
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
