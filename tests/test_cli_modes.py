import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from aura.cli import main

BANNER = "demo mode"


@pytest.fixture(autouse=True)
def _hermetic(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AURA_LLM", "mock")


def _answer_from(captured):
    return captured.out


def test_json_mode_emits_parseable_json(capsys):
    assert main(["--json", "hello world"]) == 0
    data = json.loads(_answer_from(capsys.readouterr()))
    for key in ("task", "answer", "backend", "duration_ms"):
        assert key in data, f"missing key {key}"
    assert data["task"] == "hello world"
    assert data["backend"] == "mock"
    assert isinstance(data["duration_ms"], int)
    assert data["answer"], "answer should be non-empty"


def test_json_mode_prints_only_the_json(capsys):
    main(["--json", "ping"])
    out = _answer_from(capsys.readouterr())
    assert BANNER not in out
    assert out.strip().count("\n") == 0, "expected a single JSON line on stdout"


def test_quiet_mode_prints_only_the_answer(capsys):
    assert main(["--quiet", "hello"]) == 0
    out = _answer_from(capsys.readouterr())
    assert BANNER not in out
    assert "AURA starting" not in out
    # ScriptedLLM with an empty script answers directly
    assert "Done" in out


def test_default_mode_still_prints_banner(capsys):
    assert main(["hello"]) == 0
    out = _answer_from(capsys.readouterr())
    assert BANNER in out


def test_task_starting_with_dashes_is_not_a_flag(capsys):
    # "--" ends flag parsing; a task token beginning with -- must survive.
    assert main(["--json", "--", "--json"]) == 0
    data = json.loads(_answer_from(capsys.readouterr()))
    assert data["task"] == "--json"
