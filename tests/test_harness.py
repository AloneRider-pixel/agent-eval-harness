"""Offline + real-Postgres tests for the eval harness."""
import pytest

from app import agents, store
from app.harness import AgentResult, Task, estimate_cost, run_eval, score


def test_score_keyword_matching():
    assert score("The capital of France is Paris.", ["paris"])
    assert score("STRUCTURED Query Language", ["structured", "language"])
    assert not score("answer without the word", ["missing"])
    assert score("anything", [])  # no expectations: trivially true


def test_cost_estimation():
    cost, priced = estimate_cost("llama-3.3-70b-versatile", 1000, 1000)
    assert priced and abs(cost - 0.00138) < 1e-9
    cost, priced = estimate_cost("unknown-model", 5000, 5000)
    assert not priced and cost == 0.0  # reported as unpriced, never guessed


def test_run_eval_measures_success_latency_cost():
    tasks = [Task(**t) for t in agents.BUILTIN_TASKS]
    run_id, results, summary = run_eval(tasks, agents.scripted_agent)
    assert summary["tasks"] == 3 and summary["successes"] == 2
    assert summary["success_rate"] == round(2 / 3, 4)
    assert all(r.latency_s >= 0 for r in results)
    assert summary["total_cost_usd"] == 0.0  # mock model is free


def test_agent_exception_is_a_measured_failure():
    run_id, results, summary = run_eval([Task("t1", "boom", ["x"])], agents.always_fail_agent)
    assert results[0].success is False and results[0].error == "agent exploded"
    assert summary["success_rate"] == 0.0


# ---------- persistence + API on REAL embedded Postgres ----------

@pytest.fixture(scope="module")
def pg():
    pgserver = pytest.importorskip("pgserver")
    server = pgserver.get_server("/tmp/eval-harness-test-pg")
    import os
    os.environ["DATABASE_URL"] = server.get_uri()
    yield server
    server.cleanup()


def test_store_roundtrip(pg):
    with store.connect() as conn:
        store.init_db(conn)
    tasks = [Task(**t) for t in agents.BUILTIN_TASKS]
    run_id, results, summary = run_eval(tasks, agents.scripted_agent)
    with store.connect() as conn:
        store.save_run(conn, run_id, "scripted", summary, results)
        got_summary, got_results = store.get_run(conn, run_id)
        runs = store.list_runs(conn)
    assert got_summary["success_rate"] == summary["success_rate"]
    assert len(got_results) == 3
    assert runs[0]["run_id"] == run_id


def test_api(pg):
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    with client:  # triggers startup (init_db)
        r = client.post("/runs", json={"agent": "scripted"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["success_rate"] == round(2 / 3, 4)
        got = client.get(f"/runs/{body['run_id']}")
        assert got.status_code == 200 and len(got.json()["results"]) == 3
        assert client.get("/runs/does-not-exist").status_code == 404
        bad = client.post("/runs", json={"agent": "nope"})
        assert bad.status_code == 400
