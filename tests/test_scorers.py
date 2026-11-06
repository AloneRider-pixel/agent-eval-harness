"""Scorer tests: deterministic and key-free. The LLM judge is exercised only
through injected scripted judges; no credentialed provider is ever called."""
import sys

import pytest

from app import agents
from app.harness import AgentResult, Task, run_eval
from app.scorers import (KeywordScorer, LLMJudgeScorer, ScorerUnavailable,
                         SemanticScorer, _parse_judgement, get_scorer, groq_judge)


# ---- keyword ----

def test_keyword_scorer_parity_with_legacy_score():
    s = KeywordScorer()
    assert s.score_task(Task("t", "q", ["paris"]), "The capital is Paris.").passed
    assert not s.score_task(Task("t", "q", ["london"]), "The capital is Paris.").passed
    assert s.score_task(Task("t", "q", []), "anything").passed  # no expectations


def test_keyword_partial_credit_and_missing_detail():
    r = KeywordScorer().score_task(Task("t", "q", ["a", "b", "c"]), "only a and b")
    assert not r.passed and r.score == round(2 / 3, 4)
    assert "missing: ['c']" in r.detail


# ---- semantic (stub embedder keeps this offline + deterministic) ----

def stub_embedder(texts):
    """Two-topic toy embedding space: 'paris' vs everything else."""
    return [[1.0, 0.0] if "paris" in t.lower() else [0.0, 1.0] for t in texts]


def test_semantic_matching_reference_passes():
    s = SemanticScorer(embedder=stub_embedder, threshold=0.6)
    task = Task("t", "capital of france?", reference="The capital of France is Paris.")
    r = s.score_task(task, "Paris is the capital of France.")
    assert r.passed and r.score == 1.0


def test_semantic_unrelated_output_fails():
    s = SemanticScorer(embedder=stub_embedder, threshold=0.6)
    task = Task("t", "capital of france?", reference="The capital of France is Paris.")
    r = s.score_task(task, "I enjoy cycling in the rain.")
    assert not r.passed and r.score == 0.0


def test_semantic_requires_reference_and_nonempty_output():
    s = SemanticScorer(embedder=stub_embedder)
    assert not s.score_task(Task("t", "q"), "answer").passed  # no reference
    assert not s.score_task(Task("t", "q", reference="ref"), "   ").passed  # empty output


def test_semantic_default_embedder_reports_missing_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ScorerUnavailable, match="sentence-transformers"):
        SemanticScorer().score_task(Task("t", "q", reference="ref"), "out")


# ---- LLM judge (scripted judge injected; provider never called) ----

def test_judge_pass_and_fail_against_threshold():
    good = LLMJudgeScorer(judge=lambda i, r, o: (0.92, "correct and complete"))
    bad = LLMJudgeScorer(judge=lambda i, r, o: (0.31, "misses the point"))
    task = Task("t", "q", reference="ref answer")
    r1 = good.score_task(task, "some output")
    r2 = bad.score_task(task, "some output")
    assert r1.passed and r1.score == 0.92 and "correct and complete" in r1.detail
    assert not r2.passed and r2.score == 0.31


def test_judge_requires_reference():
    s = LLMJudgeScorer(judge=lambda i, r, o: (1.0, "x"))
    assert not s.score_task(Task("t", "q"), "out").passed


def test_groq_judge_requires_explicit_key(monkeypatch):
    monkeypatch.delenv("EVAL_JUDGE_API_KEY", raising=False)
    with pytest.raises(ScorerUnavailable, match="key-free"):
        groq_judge()


def test_parse_judgement_accepts_two_line_format():
    assert _parse_judgement("SCORE: 0.85\nWHY: mostly right") == (0.85, "mostly right")
    with pytest.raises(ValueError):
        _parse_judgement("no score here")
    with pytest.raises(ValueError):
        _parse_judgement("SCORE: 1.7\nWHY: out of range")


# ---- factory + run_eval wiring ----

def test_get_scorer_factory():
    assert get_scorer("keyword").name == "keyword"
    assert get_scorer("semantic").name == "semantic"
    assert get_scorer("judge").name == "judge"
    with pytest.raises(ValueError, match="unknown scorer"):
        get_scorer("vibes")


def test_run_eval_default_is_keyword_and_backward_compatible():
    tasks = [Task(**t) for t in agents.BUILTIN_TASKS]
    _, results, summary = run_eval(tasks, agents.scripted_agent)
    assert summary["success_rate"] == round(2 / 3, 4)
    assert all(r.scorer == "keyword" for r in results)
    assert summary["scorers"] == {"keyword": {"tasks": 3, "passed": 2}}
    assert 0.0 <= summary["mean_score"] <= 1.0


def test_run_eval_per_task_scorer_override():
    def paris_agent(input: str) -> AgentResult:
        return AgentResult(output="Paris is the capital of France.", model="mock")

    tasks = [
        Task("kw", "capital?", ["paris"], scorer="keyword"),
        Task("sem", "capital?", reference="The capital of France is Paris."),
    ]
    sem = SemanticScorer(embedder=stub_embedder)
    _, results, summary = run_eval(tasks, paris_agent, scorer=sem)
    by_id = {r.task_id: r for r in results}
    # run default is the injected semantic instance; the keyword task overrides it
    assert by_id["kw"].scorer == "keyword" and by_id["kw"].success
    assert by_id["sem"].scorer == "semantic" and by_id["sem"].success
    assert summary["scorers"]["keyword"]["tasks"] == 1
    assert summary["scorers"]["semantic"]["tasks"] == 1


def test_run_eval_agent_error_still_measured_failure():
    _, results, _ = run_eval([Task("t", "boom", ["x"], scorer="semantic")],
                             agents.always_fail_agent)
    assert results[0].success is False and results[0].score == 0.0
    assert results[0].error == "agent exploded"
