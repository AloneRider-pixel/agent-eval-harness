"""Aegis-side runner: executes inside the aegis backend tree with its own
interpreter (see aegis_measured_run.py). Reads task inputs as JSON on stdin,
invokes the real investigation graph per task, writes results as JSON on
stdout. Never swallows errors: a crashed investigation is reported as an
error string so the harness records a measured failure, not a silent pass.
"""
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])  # the aegis backend tree
from app.ai.agent.graph import build_investigation_graph  # noqa: E402


class _WarningCollector(logging.Handler):
    """Captures WARNING+ log records so capability degradations the app only
    logs (knowledge search unavailable, incident history unavailable) become
    structured per-task facts in the run record, not just log lines."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


async def run_one(graph, task_input: str) -> dict:
    state = {
        "incident_id": "eval-task",
        "incident_title": task_input[:120],
        "incident_description": task_input,
        "severity": "high",
        "service_name": "eval-service",
        "symptoms": [task_input],
        "current_phase": "understand",
        "metrics_data": [], "logs_data": [], "traces_data": [],
        "deployment_data": [], "runbook_results": [],
        "historical_incidents": [], "hypotheses": [],
        "root_cause": None, "confidence": 0.0,
        "remediation_steps": [], "requires_approval": True,
        "approved": False, "investigation_trace": [], "iteration": 0,
    }
    out = await graph.ainvoke(state)
    parts = [f"Root cause: {out.get('root_cause')}",
             f"Confidence: {out.get('confidence')}"]
    for step in out.get("remediation_steps", []):
        parts.append(f"Remediation: {step}")
    for h in out.get("hypotheses", [])[:3]:
        parts.append(f"Hypothesis: {h.get('hypothesis')}")
    return {"output": "\n".join(parts), "error": None,
            "runbook_results": out.get("runbook_results"),
            "historical_incidents": out.get("historical_incidents")}


async def main() -> None:
    tasks = json.load(sys.stdin)
    collector = _WarningCollector()
    logging.getLogger().addHandler(collector)
    graph = build_investigation_graph()
    results = []
    for t in tasks:
        before = len(collector.records)
        try:
            r = await run_one(graph, t["input"])
            r["degraded"] = collector.records[before:]
            results.append({"id": t["id"], **r})
        except Exception as exc:  # noqa: BLE001 - agent failure is a measured failure
            results.append({"id": t["id"], "output": "",
                            "error": f"{type(exc).__name__}: {exc}",
                            "degraded": collector.records[before:]})
    json.dump(results, sys.stdout)


if __name__ == "__main__":
    asyncio.run(main())
