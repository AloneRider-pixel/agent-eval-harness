"""Aegis-side runner: executes inside the aegis backend tree with its own
interpreter (see aegis_measured_run.py). Reads task inputs as JSON on stdin,
invokes the real investigation graph per task, writes results as JSON on
stdout. Never swallows errors: a crashed investigation is reported as an
error string so the harness records a measured failure, not a silent pass.

Evidence alignment: the aegis investigation agent diagnoses from telemetry
in its simulator store, which the product flow seeds via trigger_scenario()
(the same call the simulator API endpoint makes). Tasks whose incident maps
to a built-in scenario are seeded through that real code path and run under
the scenario's service name; tasks with no matching scenario run unseeded,
exactly as an operator free-text question with no telemetry would. Seeding
never touches scoring: it supplies the environment evidence the task
describes, not the expected answer.
"""
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])  # the aegis backend tree
from app.ai.agent.graph import build_investigation_graph  # noqa: E402
from app.simulator.scenarios import trigger_scenario  # noqa: E402

# Dataset task -> the built-in aegis scenario whose telemetry matches the
# incident the task describes. Tasks absent here have no matching scenario
# in the app and run against an unseeded store (an honest empty environment).
SCENARIO_MAP = {
    "ag-error-rate-spike": "http-500-spike",
    "ag-cpu-saturation": "cpu-saturation",
    "ag-db-pool-exhaustion": "db-connection-exhaustion",
    "ag-memory-growth": "memory-pressure",
    "ag-latency-after-deploy": "api-latency-spike",
    "ag-approval-boundary": "db-connection-exhaustion",
}


class _WarningCollector(logging.Handler):
    """Captures WARNING+ log records so capability degradations the app only
    logs (knowledge search unavailable, incident history unavailable) become
    structured per-task facts in the run record, not just log lines."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


async def run_one(graph, task_id: str, task_input: str) -> dict:
    scenario_id = SCENARIO_MAP.get(task_id)
    service = "eval-service"
    if scenario_id:
        service = trigger_scenario(scenario_id)["service"]
    state = {
        "incident_id": "eval-task",
        "incident_title": task_input[:120],
        "incident_description": task_input,
        "severity": "high",
        "service_name": service,
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
    remediation = out.get("recommended_remediation") or {}
    if remediation:
        parts.append(
            f"Remediation: {remediation.get('action')}: "
            f"{remediation.get('description')} "
            f"(requires_approval={remediation.get('requires_approval')})")
    for h in out.get("hypotheses", [])[:3]:
        parts.append(f"Hypothesis: {h.get('hypothesis')}")
    return {"output": "\n".join(parts), "error": None,
            "seeded_scenario": scenario_id,
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
            r = await run_one(graph, t["id"], t["input"])
            r["degraded"] = collector.records[before:]
            results.append({"id": t["id"], **r})
        except Exception as exc:  # noqa: BLE001 - agent failure is a measured failure
            results.append({"id": t["id"], "output": "",
                            "error": f"{type(exc).__name__}: {exc}",
                            "seeded_scenario": SCENARIO_MAP.get(t["id"]),
                            "degraded": collector.records[before:]})
    json.dump(results, sys.stdout)


if __name__ == "__main__":
    asyncio.run(main())
