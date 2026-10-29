"""Tests for multi-agent crews (aura/crew.py)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aura.crew import Crew, CrewResult, WorkerSpec, default_workers
from aura.llm import ScriptedLLM
from aura.memory import Memory


@pytest.fixture
def memory(tmp_path):
    # Isolated path: the default Memory() persists to ~/.aura/memory.json.
    return Memory(path=str(tmp_path / "mem.json"))


def make_crew(scripts, workers=None, memory=None, **kwargs):
    def factory(script=iter([s if isinstance(s, list) else [s] for s in scripts])):
        return ScriptedLLM(next(script))
    kwargs.setdefault("verbose", False)
    if memory is not None:
        kwargs["memory"] = memory
    return Crew(factory, workers=workers, **kwargs)


PLAN = ('ANSWER: [{"name": "research", "worker": "researcher", '
        '"brief": "List the project components."}, '
        '{"name": "notes", "worker": "writer", '
        '"brief": "Write a 3-bullet summary to notes.txt."}]')


def test_full_crew_run_routes_and_synthesizes(memory):
    crew = make_crew(
        [[PLAN, "ANSWER: final combined answer"],
         ["ANSWER: research done: agent, tools, memory"],
         ["ANSWER: notes written"]],
        workers=[WorkerSpec("researcher", allowed_tools=["read_file"]),
                 WorkerSpec("writer", allowed_tools=["write_file"])],
        memory=memory,
    )
    result = crew.run("explain the repo")
    assert isinstance(result, CrewResult)
    assert [s["name"] for s in result.plan] == ["research", "notes"]
    assert result.answers["research"] == "research done: agent, tools, memory"
    assert result.answers["notes"] == "notes written"
    assert result.final == "final combined answer"


def test_round_robin_when_worker_omitted(memory):
    plan = ('ANSWER: [{"name": "one", "brief": "do one"}, '
            '{"name": "two", "brief": "do two"}]')
    crew = make_crew(
        [plan, ["ANSWER: first"], ["ANSWER: second"],
         ["ANSWER: synthesized"]],
        workers=[WorkerSpec("alpha", allowed_tools=["read_file"]),
                 WorkerSpec("beta", allowed_tools=["read_file"])],
        memory=memory,
    )
    result = crew.run("task")
    assert result.answers == {"one": "first", "two": "second"}
    # round-robin: subtask 'one' ran on alpha, 'two' on beta
    assert list(crew._agents) == ["alpha", "beta"]


def test_unknown_worker_rejected(memory):
    plan = 'ANSWER: [{"name": "x", "worker": "ghost", "brief": "boo"}]'
    crew = make_crew([plan], workers=[WorkerSpec("real")], memory=memory)
    with pytest.raises(ValueError, match="unknown worker"):
        crew.run("task")


def test_malformed_plan_rejected(memory):
    crew = make_crew(["ANSWER: I think we should do research first."],
                     memory=memory)
    with pytest.raises(ValueError, match="did not return a JSON plan"):
        crew.run("task")


def test_empty_plan_rejected(memory):
    crew = make_crew(["ANSWER: []"], memory=memory)
    with pytest.raises(ValueError, match="empty or non-list"):
        crew.run("task")


def test_subtask_missing_brief_rejected(memory):
    crew = make_crew(['ANSWER: [{"name": "x"}]'], memory=memory)
    with pytest.raises(ValueError, match="missing its 'brief'"):
        crew.run("task")


def test_duplicate_subtask_names_rejected(memory):
    plan = ('ANSWER: [{"name": "dup", "brief": "a"}, '
            '{"name": "dup", "brief": "b"}]')
    crew = make_crew([plan], memory=memory)
    with pytest.raises(ValueError, match="duplicate subtask name"):
        crew.run("task")


def test_worker_confined_to_allowlist(tmp_path, memory):
    target = tmp_path / "should-not-exist.txt"
    script = (f'THOUGHT: trying a write\n'
              f'ACTION: write_file {{"path": "{target}", "content": "x"}}\n')
    crew = make_crew(
        ['ANSWER: [{"name": "try", "worker": "reader", "brief": "attempt"}]',
         [script, "ANSWER: done"],
         ["ANSWER: synthesized"]],
        workers=[WorkerSpec("reader", allowed_tools=["read_file"])],
        memory=memory,
    )
    result = crew.run("task")
    assert result.answers["try"] == "done"
    assert not target.exists()  # the disallowed write never ran


def test_worker_can_use_allowed_tool(tmp_path, memory):
    target = tmp_path / "notes.txt"
    script = (f'THOUGHT: writing notes\n'
              f'ACTION: write_file {{"path": "{target}", "content": "hi"}}\n')
    crew = make_crew(
        ['ANSWER: [{"name": "write", "worker": "writer", "brief": "write"}]',
         [script, "ANSWER: wrote it"],
         ["ANSWER: synthesized"]],
        workers=[WorkerSpec("writer", allowed_tools=["write_file"])],
        memory=memory,
    )
    result = crew.run("task")
    assert result.answers["write"] == "wrote it"
    assert target.read_text() == "hi"


def test_episodes_logged_to_shared_memory(memory):
    crew = make_crew(
        [[PLAN, "ANSWER: s"],
         ["ANSWER: r"], ["ANSWER: w"]],
        workers=[WorkerSpec("researcher"), WorkerSpec("writer")],
        memory=memory,
    )
    crew.run("explain the repo")
    # planner plan + 2 workers + planner synthesis + crew summary
    assert len(memory.episodes) == 5


def test_worker_spec_validation():
    with pytest.raises(ValueError, match="non-empty name"):
        WorkerSpec("")
    with pytest.raises(ValueError, match="max_steps"):
        WorkerSpec("x", max_steps=0)
    with pytest.raises(ValueError, match="duplicate worker"):
        Crew(lambda: ScriptedLLM([]),
             workers=[WorkerSpec("a"), WorkerSpec("a")])
    with pytest.raises(ValueError, match="unknown tool"):
        Crew(lambda: ScriptedLLM([]),
             workers=[WorkerSpec("a", allowed_tools=["nope_tool"])])
    with pytest.raises(ValueError, match="at least one worker"):
        Crew(lambda: ScriptedLLM([]), workers=[])
    with pytest.raises(ValueError, match="callable"):
        Crew("not-callable")


def test_plan_defaults_missing_name(memory):
    plan = 'ANSWER: [{"brief": "do the thing"}]'
    crew = make_crew(
        [plan, ["ANSWER: ok"], ["ANSWER: s"]],
        workers=[WorkerSpec("solo")], memory=memory)
    assert crew.plan("task")[0]["name"] == "subtask-1"


def test_default_workers_sensible():
    workers = default_workers()
    assert {w.name for w in workers} == {"researcher", "writer"}
    assert "write_file" not in workers[0].allowed_tools  # read-only
    assert "write_file" in workers[1].allowed_tools


def test_assignments_record_explicit_workers(memory):
    crew = make_crew(
        [[PLAN, "ANSWER: final combined answer"],
         ["ANSWER: research done"],
         ["ANSWER: notes written"]],
        workers=[WorkerSpec("researcher", allowed_tools=["read_file"]),
                 WorkerSpec("writer", allowed_tools=["write_file"])],
        memory=memory,
    )
    result = crew.run("explain the repo")
    assert result.assignments == {"research": "researcher", "notes": "writer"}


def test_assignments_record_round_robin_workers(memory):
    plan = ('ANSWER: [{"name": "one", "brief": "do one"}, '
            '{"name": "two", "brief": "do two"}]')
    crew = make_crew(
        [plan, ["ANSWER: first"], ["ANSWER: second"],
         ["ANSWER: synthesized"]],
        workers=[WorkerSpec("alpha", allowed_tools=["read_file"]),
                 WorkerSpec("beta", allowed_tools=["read_file"])],
        memory=memory,
    )
    result = crew.run("task")
    assert result.assignments == {"one": "alpha", "two": "beta"}


def test_durations_ms_cover_exactly_the_subtasks(memory):
    crew = make_crew(
        [[PLAN, "ANSWER: final combined answer"],
         ["ANSWER: research done"],
         ["ANSWER: notes written"]],
        workers=[WorkerSpec("researcher", allowed_tools=["read_file"]),
                 WorkerSpec("writer", allowed_tools=["write_file"])],
        memory=memory,
    )
    result = crew.run("explain the repo")
    assert set(result.durations_ms) == {"research", "notes"}
    for value in result.durations_ms.values():
        assert isinstance(value, float)
        assert value >= 0.0


def test_cli_crew_json_includes_durations_ms(tmp_path, monkeypatch, capsys):
    # crew_main builds its own Memory(); point it at an isolated path so
    # the real ~/.aura/memory.json is never touched.
    import json as _json

    import aura.cli as cli

    isolated = Memory(path=str(tmp_path / "cli-mem.json"))
    monkeypatch.setattr(cli, "Memory", lambda *a, **k: isolated)
    monkeypatch.setenv("AURA_LLM", "mock")
    # main() passes quiet=True alongside --json; without it the demo banner
    # would precede the JSON on stdout.
    assert cli.crew_main(["explain the repo"], quiet=True, json_mode=True) == 0
    payload = _json.loads(capsys.readouterr().out)
    assert set(payload["durations_ms"]) == {"research", "notes"}
    for value in payload["durations_ms"].values():
        assert isinstance(value, float)
        assert value >= 0.0
