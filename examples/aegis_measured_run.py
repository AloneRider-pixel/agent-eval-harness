"""Measured run of the real aegis investigation agent through this harness.

Key-free by construction: on the audited aegis commit the investigation
graph contains no LLM call at all (deterministic node logic over simulated
telemetry tools), so there is no provider to mock and no credential in the
path. This script wires the real app code as the agent callable and records
exactly what happens - success or failure - with full disclosure.

Usage:
    python examples/aegis_measured_run.py [--aegis-backend PATH]
        [--dataset aegis@v1] [--scorer keyword] [--out PATH]

Disclosure recorded in the run file: app commit, harness commit, dataset
version, scorer, config (no LLM in path), timestamp, and per-task errors.
The numbers measure the app's current revision as-is; they are not a model
quality score.
"""
import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.datasets import load_dataset  # noqa: E402
from app.harness import AgentResult, run_eval  # noqa: E402


def _commit(path: str) -> str:
    return subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def run_aegis_batch(backend: Path, venv_python: str, tasks) -> dict:
    """Run every task against the real aegis graph in one subprocess with a
    clean interpreter - the two repos both name their package `app`, so
    in-process import would collide. Returns {task_id: {output, error}}."""
    runner = Path(__file__).resolve().parent / "aegis_adapter_runner.py"
    payload = json.dumps([{"id": t.id, "input": t.input} for t in tasks])
    proc = subprocess.run(
        [venv_python, str(runner), str(backend)], input=payload, capture_output=True,
        text=True, cwd=backend, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"aegis runner failed: {proc.stderr[-500:]}")
    return {r["id"]: r for r in json.loads(proc.stdout)}


def build_aegis_agent(backend: Path, venv_python: str, tasks):
    """Batch-run all tasks through the real aegis graph once (subprocess),
    then serve the harness's per-input calls from the recorded results.
    An errored task re-raises inside the harness so it is recorded as a
    measured agent failure with the real error text."""
    batch = run_aegis_batch(backend, venv_python, tasks)
    by_input = {t.input: batch[t.id] for t in tasks}
    aegis_agent_degraded = {}

    def aegis_agent(task_input: str) -> AgentResult:
        r = by_input[task_input]
        if r["error"]:
            raise RuntimeError(r["error"])
        aegis_agent_degraded[task_input] = r.get("degraded", [])
        return AgentResult(output=r["output"],
                           prompt_tokens=len(task_input.split()),
                           completion_tokens=len(r["output"].split()),
                           model="deterministic-graph (no LLM in path)")

    return aegis_agent, aegis_agent_degraded


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aegis-backend",
                    default=str(Path.home() / "build/aegis/backend"))
    ap.add_argument("--dataset", default="aegis@v1")
    ap.add_argument("--scorer", default="keyword")
    ap.add_argument("--out", default=None)
    ap.add_argument("--note", default=None,
                    help="extra disclosure recorded in the run file")
    ap.add_argument("--aegis-python",
                    default=str(Path.home() / "venv-aegis/bin/python"),
                    help="interpreter with aegis's runtime deps (langgraph)")
    args = ap.parse_args()

    backend = Path(args.aegis_backend)
    name, version = args.dataset.split("@")
    ds = load_dataset(name, version)
    agent, aegis_agent_degraded = build_aegis_agent(backend, args.aegis_python, ds.tasks)
    _, results, summary = run_eval(ds.tasks, agent, scorer=args.scorer)
    by_id = {t.id: t.input for t in ds.tasks}
    degradations = sorted({d for degs in aegis_agent_degraded.values()
                           for d in degs})

    record = {
        "label": "measured run (real app, current revision as-is)",
        "app": "aegis (AloneRider-pixel/aegis)",
        "app_commit": _commit(backend),
        "harness_commit": _commit(str(Path(__file__).resolve().parent.parent)),
        "dataset": ds.name, "dataset_version": ds.version,
        "scorer": args.scorer,
        "config": {"llm": "none in agent path on this commit; key-free",
                   "approval_gating": "unchanged (no execution path invoked)"},
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "note": args.note,
        "capability_degradations": degradations,
        "gate_eligible": not degradations,
        "gate_note": (None if not degradations else
                      "capabilities were degraded (unavailable services are "
                      "listed, not mistaken for empty results); a baseline "
                      "from this run would bake in the missing capability"),
        "summary": summary,
        "task_results": [{
            "task_id": r.task_id, "success": r.success, "error": r.error,
            "detail": r.detail, "latency_s": r.latency_s,
            "degraded": aegis_agent_degraded.get(by_id[r.task_id], []),
        } for r in results],
    }
    out = Path(args.out or f"runs/aegis-measured-{ds.version}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"run record: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
