"""FastAPI surface for the eval harness."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import agents, datasets, store
from .harness import Task, run_eval
from .scorers import ScorerUnavailable

app = FastAPI(title="Agent Eval Harness", version="1.0.0")

REGISTRY = {"scripted": agents.scripted_agent, "always_fail": agents.always_fail_agent}


class RunRequest(BaseModel):
    agent: str = "scripted"
    tasks: list[dict] | None = None  # [{"id","input","expected_keywords","reference"?,"scorer"?}]
    dataset: str | None = None       # "name" or "name@version"; overrides tasks
    scorer: str | None = None        # run default: keyword|semantic|judge


@app.on_event("startup")
def startup():
    with store.connect() as conn:
        store.init_db(conn)


@app.get("/health")
def health():
    return {"status": "ok", "agents": list(REGISTRY)}


@app.get("/runs")
def runs():
    with store.connect() as conn:
        return store.list_runs(conn)


@app.get("/datasets")
def list_datasets():
    import json
    return json.loads((datasets.DATASETS_DIR / "manifest.json").read_text())["datasets"]


@app.post("/runs")
def create_run(body: RunRequest):
    if body.agent not in REGISTRY:
        raise HTTPException(400, f"unknown agent {body.agent!r}; options: {list(REGISTRY)}")
    if body.dataset:
        name, version = datasets.parse_dataset_ref(body.dataset)
        try:
            tasks = datasets.load_dataset(name, version).tasks
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(400, str(exc))
    else:
        raw = body.tasks or agents.BUILTIN_TASKS
        tasks = [Task(id=t["id"], input=t["input"],
                      expected_keywords=t.get("expected_keywords", []),
                      reference=t.get("reference"), scorer=t.get("scorer"),
                      forbidden_keywords=t.get("forbidden_keywords", []),
                      expected_sources=t.get("expected_sources", []),
                      retrieval_k=t.get("retrieval_k", 5)) for t in raw]
    try:
        run_id, results, summary = run_eval(tasks, REGISTRY[body.agent], scorer=body.scorer)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except ScorerUnavailable as exc:
        raise HTTPException(400, f"scorer backend unavailable: {exc}")
    with store.connect() as conn:
        store.save_run(conn, run_id, body.agent, summary, results)
    return summary


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    with store.connect() as conn:
        summary, results = store.get_run(conn, run_id)
    if summary is None:
        raise HTTPException(404, "run not found")
    return {"summary": summary, "results": results}
