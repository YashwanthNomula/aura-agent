import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import aura.cli as cli
from aura.memory import Memory
from aura.vector_memory import VectorIndex, cosine, embed, index_memory, tokenize


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Fine-Tune! it 100%") == ["fine", "tune", "it", "100"]


def test_embed_deterministic_and_normalized():
    a, b = embed("the agent wrote a haiku"), embed("the agent wrote a haiku")
    assert a == b
    assert math.isclose(math.sqrt(sum(v * v for v in a)), 1.0, rel_tol=1e-6)
    assert a != embed("completely different words about zebras")


def test_query_ranks_relevant_above_irrelevant():
    idx = VectorIndex()
    idx.add("pizza", "user loves pizza for dinner")
    idx.add("physics", "attended a quantum physics lecture")
    results = idx.query("favorite italian dinner")
    assert results[0][0] == "pizza"
    assert all(results[i][2] >= results[i + 1][2] for i in range(len(results) - 1))


def test_query_top_k_and_empty_index():
    idx = VectorIndex()
    assert idx.query("anything") == []
    for i in range(10):
        idx.add(f"d{i}", f"memory number {i} about pizza")
    assert len(idx.query("pizza", k=3)) == 3
    assert idx.query("pizza", k=0) == []


def test_index_memory_covers_facts_and_episodes(tmp_path):
    m = Memory(path=str(tmp_path / "m.json"))
    m.remember("food", "Yashwanth loves dosa for breakfast")
    m.log_episode("Write haiku", "wrote a recursion haiku")
    idx = index_memory(m)
    assert len(idx) == 2
    ids = {doc_id for doc_id, _, _ in idx.query("breakfast", k=5)}
    assert "fact:food" in ids


def test_recall_semantic_prefers_rephrased_fact(tmp_path):
    m = Memory(path=str(tmp_path / "m.json"))
    m.remember("pet", "has a cat named Milo")
    m.remember("work", "works on quarterly financial spreadsheets")
    hits = m.recall_semantic("what animal does the user own", k=2)
    assert hits[0][0] == "fact:pet"
    assert "cat" in hits[0][1]


def test_cosine_known_values():
    assert math.isclose(cosine([1.0, 0.0], [1.0, 0.0]), 1.0)
    assert math.isclose(cosine([1.0, 0.0], [0.0, 1.0]), 0.0)


def test_cli_recall_flag(tmp_path, capsys, monkeypatch):
    mp = str(tmp_path / "m.json")
    m = Memory(path=mp)
    m.remember("pet", "has a cat named Milo")
    monkeypatch.setattr(cli, "Memory", lambda: Memory(path=mp))
    assert cli.main(["--recall", "what animal"]) == 0
    out = capsys.readouterr().out
    assert "cat" in out and "fact:pet" in out


def test_cli_recall_empty_memory(tmp_path, capsys, monkeypatch):
    mp = str(tmp_path / "m.json")
    monkeypatch.setattr(cli, "Memory", lambda: Memory(path=mp))
    assert cli.main(["--recall", "anything"]) == 0
    assert "No memories stored yet" in capsys.readouterr().out


def test_split_flags_recall_consumes_query():
    quiet, json_mode, recall_query, args = cli._split_flags(["--recall", "my cat", "do thing"])
    assert recall_query == "my cat"
    assert args == ["do thing"]
    assert not quiet and not json_mode
