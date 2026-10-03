import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aura import tools

PAGE = (
    b"<html><head><title>Test</title></head>"
    b"<body><h1>Hello</h1><p>world &amp; friends</p>"
    b"<p>" + b"lorem ipsum dolor sit amet. " * 40 + b"</p></body></html>"
)


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(PAGE)))
        self.end_headers()
        self.wfile.write(PAGE)

    def log_message(self, *args):
        pass


@pytest.fixture
def http_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.fixture(autouse=True)
def reset_todos():
    tools._TODOS.clear()
    tools._TODO_NEXT_ID = 1
    yield
    tools._TODOS.clear()
    tools._TODO_NEXT_ID = 1


# ------------------------------------------------------------- web_fetch

def test_web_fetch_strips_html(http_server):
    out = tools.run_tool("web_fetch", {"url": http_server})
    assert "Hello" in out and "world" in out
    assert "<h1>" not in out and "<p>" not in out


def test_web_fetch_truncates(http_server):
    out = tools.run_tool("web_fetch", {"url": http_server, "max_chars": 100})
    assert out.endswith("...[truncated]")


def test_web_fetch_refuses_non_http():
    assert "only http" in tools.run_tool("web_fetch", {"url": "file:///etc/hostname"})
    assert "only http" in tools.run_tool("web_fetch", {"url": "ftp://x.example/file"})


def test_web_fetch_bad_host():
    out = tools.run_tool("web_fetch", {"url": "http://127.0.0.1:1/"})
    assert "ERROR" in out


# ------------------------------------------------------------- todo_list

def test_todo_add_list():
    assert "Added todo #1" in tools.run_tool("todo_list", {"action": "add", "text": "buy milk"})
    assert "Added todo #2" in tools.run_tool("todo_list", {"action": "add", "text": "ship aura"})
    out = tools.run_tool("todo_list", {"action": "list"})
    assert "[ ] #1 buy milk" in out and "[ ] #2 ship aura" in out


def test_todo_complete_and_remove():
    tools.run_tool("todo_list", {"action": "add", "text": "write tests"})
    assert "Completed todo #1" in tools.run_tool("todo_list", {"action": "complete", "id": 1})
    assert "[x] #1 write tests" in tools.run_tool("todo_list", {"action": "list"})
    assert "Removed todo #1" in tools.run_tool("todo_list", {"action": "remove", "id": 1})
    assert "empty" in tools.run_tool("todo_list", {"action": "list"})


def test_todo_errors():
    assert "ERROR" in tools.run_tool("todo_list", {"action": "add", "text": "  "})
    assert "ERROR" in tools.run_tool("todo_list", {"action": "complete", "id": 99})
    assert "ERROR" in tools.run_tool("todo_list", {"action": "dance"})


# ---------------------------------------------------------- summarize_file

SAMPLE = (
    "Python is a great language.\n"
    "The agent loves Python because Python is simple.\n"
    "Write Python code, test Python code, ship Python code.\n"
    "line four here\n"
    "line five here\n"
    "line six here\n"
    "line seven here\n"
    "the final line"
)


def test_summarize_file(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text(SAMPLE)
    out = tools.run_tool("summarize_file", {"path": str(p)})
    assert f"Lines: {len(SAMPLE.splitlines())}" in out
    assert f"Words: {len(SAMPLE.split())}" in out
    assert "python(" in out  # top keyword
    assert "First lines:" in out and "1: Python is a great language." in out
    assert "Last lines:" in out and "the final line" in out


def test_summarize_file_missing():
    out = tools.run_tool("summarize_file", {"path": "/tmp/aura_definitely_missing_xyz.txt"})
    assert "ERROR" in out


def test_tools_registered():
    for name in ("web_fetch", "todo_list", "summarize_file"):
        assert name in tools.REGISTRY
        assert name in tools.describe_tools()
