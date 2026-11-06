"""Golden dataset tests: loading, validation, versioning, API wiring."""
import json

import pytest

from app import datasets
from app.datasets import load_dataset, parse_dataset_ref

BOTH = ["enterprise-support", "aegis", "adversarial"]


def test_manifest_datasets_load_and_validate():
    manifest = json.loads((datasets.DATASETS_DIR / "manifest.json").read_text())
    assert sorted(manifest["datasets"]) == sorted(BOTH)
    for name in BOTH:
        ds = load_dataset(name)
        assert ds.version == manifest["datasets"][name]["latest"] == "v1"
        assert len(ds.tasks) >= 8
        assert ds.target_agent and ds.provenance
        ids = [t.id for t in ds.tasks]
        assert len(ids) == len(set(ids)), "unique task ids"
        for t in ds.tasks:
            assert t.input and t.reference  # every task is judge/semantic-ready


def test_version_resolution_and_pinning():
    assert load_dataset("aegis", "v1").version == "v1"
    with pytest.raises(KeyError, match="no version"):
        load_dataset("aegis", "v9")
    with pytest.raises(KeyError, match="unknown dataset"):
        load_dataset("nope")


def test_parse_dataset_ref():
    assert parse_dataset_ref("aegis@v1") == ("aegis", "v1")
    assert parse_dataset_ref("aegis") == ("aegis", None)
    with pytest.raises(ValueError):
        parse_dataset_ref("@v1")


def _write_dataset(root, name="x", version="v1", tasks=None):
    d = root / name
    d.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps(
        {"datasets": {name: {"versions": [version], "latest": version}}}))
    (d / f"{version}.json").write_text(json.dumps({
        "name": name, "version": version, "target_agent": "t", "provenance": "p",
        "tasks": tasks if tasks is not None else
        [{"id": "t1", "input": "q", "expected_keywords": ["a"]}]}))


def test_validation_rejects_duplicate_ids(tmp_path):
    _write_dataset(tmp_path, tasks=[
        {"id": "dup", "input": "q1", "expected_keywords": ["a"]},
        {"id": "dup", "input": "q2", "expected_keywords": ["b"]}])
    with pytest.raises(ValueError, match="duplicate task id"):
        load_dataset("x", root=tmp_path)


def test_validation_rejects_task_without_any_expectation(tmp_path):
    _write_dataset(tmp_path, tasks=[{"id": "t1", "input": "q"}])
    with pytest.raises(ValueError, match="expected_keywords and/or a reference"):
        load_dataset("x", root=tmp_path)


def test_validation_rejects_name_version_mismatch(tmp_path):
    _write_dataset(tmp_path)
    raw = json.loads((tmp_path / "x" / "v1.json").read_text())
    raw["version"] = "v2"
    (tmp_path / "x" / "v1.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="claims"):
        load_dataset("x", root=tmp_path)


# ---------- API wiring on real embedded Postgres ----------

@pytest.fixture(scope="module")
def pg():
    pgserver = pytest.importorskip("pgserver")
    server = pgserver.get_server("/tmp/eval-harness-test-pg-datasets")
    import os
    os.environ["DATABASE_URL"] = server.get_uri()
    yield server
    server.cleanup()


def test_api_runs_datasets_and_scorer_validation(pg):
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    with client:  # startup: init_db
        listed = client.get("/datasets").json()
        assert sorted(listed) == sorted(BOTH)

        r = client.post("/runs", json={"agent": "scripted", "dataset": "aegis@v1"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["tasks"] == 8 and body["scorers"]["keyword"]["tasks"] == 8
        # the scripted stand-in is NOT the aegis agent: results are harness
        # validation only, so we assert structure, not a flattering success rate.
        assert 0.0 <= body["success_rate"] <= 1.0

        assert client.post("/runs", json={"agent": "scripted",
                                          "dataset": "nope"}).status_code == 400
        assert client.post("/runs", json={"agent": "scripted",
                                          "dataset": "aegis@v9"}).status_code == 400
        assert client.post("/runs", json={"agent": "scripted",
                                          "scorer": "nope"}).status_code == 400
        # tasks without references degrade to scored failures, not errors:
        r = client.post("/runs", json={"agent": "scripted", "scorer": "judge"})
        assert r.status_code == 200 and r.json()["success_rate"] == 0.0
        # with a reference the judge needs its credentialed backend, which is
        # disabled in key-free CI: the run refuses loudly instead of guessing.
        r = client.post("/runs", json={
            "agent": "scripted", "scorer": "judge",
            "tasks": [{"id": "t", "input": "q", "reference": "r"}]})
        assert r.status_code == 400 and "key-free" in r.json()["detail"]
