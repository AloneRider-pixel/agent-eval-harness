"""Release gate: stored baselines, budgets, and a compare/check command.

A baseline is a recorded run summary for one (agent, dataset) pair, stored with
full disclosure metadata: model config, dataset version, sample count, timestamp
and commit. `check` re-runs the same eval and fails when a tolerated metric
regresses against baseline or a budget threshold is breached.

Fixture runs (scripted agents, golden datasets without a real app) validate the
gate itself and are labelled harness validation. An app's baseline becomes a
published number only after a real run against that app.

CLI:
    python -m app.gate record --agent scripted [--dataset aegis@v1] [--scorer keyword]
    python -m app.gate check  --agent scripted [--dataset aegis@v1] [--config gate.config.json]
"""
import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

from . import agents as builtin_agents
from .datasets import load_dataset, parse_dataset_ref
from .harness import Task, run_eval

BASELINES_DIR = Path(__file__).resolve().parent.parent / "baselines"

# budget name -> (summary key, direction)
BUDGET_KEYS = {
    "max_avg_latency_s": ("avg_latency_s", "max"),
    "max_p95_latency_s": ("p95_latency_s", "max"),
    "max_total_cost_usd": ("total_cost_usd", "max"),
    "min_success_rate": ("success_rate", "min"),
    "min_mean_score": ("mean_score", "min"),
}

REGISTRY = {"scripted": builtin_agents.scripted_agent,
            "always_fail": builtin_agents.always_fail_agent}


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 - metadata must never break the gate
        return "unknown"


def baseline_path(agent: str, dataset: "str | None", root: Path = BASELINES_DIR) -> Path:
    name = parse_dataset_ref(dataset)[0] if dataset else "builtin"
    return root / agent / f"{name}.json"


def _tasks_for(dataset: "str | None"):
    if dataset:
        name, version = parse_dataset_ref(dataset)
        ds = load_dataset(name, version)
        return ds.tasks, {"dataset": ds.name, "dataset_version": ds.version}
    tasks = [Task(**t) for t in builtin_agents.BUILTIN_TASKS]
    return tasks, {"dataset": "builtin", "dataset_version": None}


def record(agent: str, dataset: "str | None", scorer: "str | None",
           root: Path = BASELINES_DIR) -> dict:
    tasks, meta = _tasks_for(dataset)
    _, _, summary = run_eval(tasks, REGISTRY[agent], scorer=scorer)
    baseline = {
        "agent": agent,
        **meta,
        "sample_count": summary["tasks"],
        "scorer": scorer or "keyword",
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "commit": _git_commit(),
        "summary": summary,
        "label": ("harness validation (fixture agent)" if agent in REGISTRY
                  else "measured run"),
    }
    path = baseline_path(agent, dataset, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, indent=2) + "\n")
    return baseline


def compare(current: dict, baseline: dict, tolerance: dict, budgets: dict):
    """Returns (passed: bool, reasons: list[str])."""
    reasons = []
    base = baseline.get("summary", {})
    for metric, tol in (tolerance or {}).items():
        cur_v, base_v = current.get(metric), base.get(metric)
        if cur_v is None or base_v is None:
            continue
        if cur_v < base_v - tol:
            reasons.append(
                f"regression: {metric} {cur_v:.6g} < baseline {base_v:.6g} - tolerance {tol}")
    for budget, (key, direction) in BUDGET_KEYS.items():
        if budget not in (budgets or {}):
            continue
        limit = budgets[budget]
        cur_v = current.get(key)
        if cur_v is None:
            continue
        if direction == "max" and cur_v > limit:
            reasons.append(f"budget breach: {key} {cur_v:.6g} > {limit:.6g}")
        if direction == "min" and cur_v < limit:
            reasons.append(f"budget breach: {key} {cur_v:.6g} < {limit:.6g}")
    return (not reasons), reasons


def check(agent: str, dataset: "str | None", scorer: "str | None",
          config_path: "str | None", root: Path = BASELINES_DIR):
    cfg = {"tolerance": {}, "budgets": {}}
    if config_path:
        cfg = {**cfg, **json.loads(Path(config_path).read_text())}
    path = baseline_path(agent, dataset, root)
    if not path.exists():
        return 2, [f"no baseline at {path}; run `record` first"]
    baseline = json.loads(path.read_text())
    tasks, _ = _tasks_for(dataset)
    _, _, current = run_eval(tasks, REGISTRY[agent], scorer=scorer)
    passed, reasons = compare(current, baseline, cfg["tolerance"], cfg["budgets"])
    return (0 if passed else 1), reasons


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="app.gate", description=__doc__)
    p.add_argument("command", choices=["record", "check"])
    p.add_argument("--agent", required=True, choices=sorted(REGISTRY))
    p.add_argument("--dataset", default=None, help="name or name@version")
    p.add_argument("--scorer", default=None)
    p.add_argument("--config", default=None, help="gate config JSON (check)")
    args = p.parse_args(argv)

    if args.command == "record":
        baseline = record(args.agent, args.dataset, args.scorer)
        print(f"baseline recorded: {baseline['agent']} / {baseline['dataset']} "
              f"success_rate={baseline['summary']['success_rate']} "
              f"({baseline['label']})")
        return 0
    code, reasons = check(args.agent, args.dataset, args.scorer, args.config)
    if code == 0:
        print("GATE PASS: no regression against baseline, budgets hold")
    else:
        print("GATE FAIL:" if code == 1 else "GATE ERROR:")
        for r in reasons:
            print(f"  - {r}")
    return code


if __name__ == "__main__":
    sys.exit(main())
