"""Retrieval metric tests: precision@k and the faithfulness proxy, offline."""

import pytest

from app.harness import AgentResult, Task, run_eval
from app.retrieval import content_words, faithfulness_proxy, precision_at_k

# ---- precision@k ----

def test_precision_at_k_math():
    exp = {"a", "b", "c"}
    assert precision_at_k(["a", "b", "x", "y", "z"], exp, 5) == 0.4
    assert precision_at_k(["a", "b", "c"], exp, 3) == 1.0
    assert precision_at_k(["x"], exp, 1) == 0.0
    assert precision_at_k([], exp, 5) == 0.0
    # k larger than the retrieved list: precision over what was returned
    assert precision_at_k(["a", "x"], exp, 5) == 0.5
    with pytest.raises(ValueError):
        precision_at_k(["a"], exp, 0)


# ---- faithfulness proxy ----

def test_content_words_strip_stopwords_and_short_tokens():
    assert content_words("The CPU is at 98% on a node") == {"cpu", "node"}  # "98" is too short to be a content word


def test_faithfulness_proxy_coverage():
    ctx = ["The connection pool is exhausted because connections leak."]
    assert faithfulness_proxy("The connection pool is exhausted by a leak.", ctx) == 1.0
    assert faithfulness_proxy("Mercury is the smallest planet.", ctx) == 0.0
    assert faithfulness_proxy("anything at all", []) == 0.0
    assert faithfulness_proxy("...", ctx) == 1.0  # no content words: trivially faithful


def test_run_eval_records_retrieval_metrics():
    corpus = {
        "doc-pool": "The connection pool is exhausted because connections leak.",
        "doc-cpu": "CPU saturation comes from hot partitions or runaway processes.",
        "doc-noise": "Unrelated operations handbook cover page.",
    }

    def retrieving_agent(input: str) -> AgentResult:
        if "pool" in input:
            return AgentResult(
                output="The connection pool is exhausted because connections leak.",
                model="mock", retrieved=["doc-pool", "doc-noise"],
                contexts=[corpus["doc-pool"], corpus["doc-noise"]])
        return AgentResult(output="I do not know.", model="mock",
                           retrieved=["doc-noise"], contexts=[corpus["doc-noise"]])

    tasks = [
        Task("hit", "why is the pool exhausted", ["pool"],
             expected_sources=["doc-pool"], retrieval_k=2),
        Task("miss", "something else", ["x"],
             expected_sources=["doc-cpu"], retrieval_k=2),
    ]
    _, results, summary = run_eval(tasks, retrieving_agent)
    by_id = {r.task_id: r for r in results}
    assert by_id["hit"].retrieval == {"k": 2, "precision_at_k": 0.5, "faithfulness": 1.0}
    assert by_id["miss"].retrieval["precision_at_k"] == 0.0
    assert summary["retrieval"]["tasks"] == 2
    assert summary["retrieval"]["mean_precision_at_k"] == 0.25
    # tasks without expected_sources carry no retrieval block
    _, r2, s2 = run_eval([Task("plain", "q", ["a"])], retrieving_agent)
    assert r2[0].retrieval is None and s2["retrieval"] is None
