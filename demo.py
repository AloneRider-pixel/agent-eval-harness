"""Offline demo: evaluate the scripted agent on 3 tasks against a real Postgres store."""
import os

os.environ.setdefault("DATABASE_URL", "")

import pgserver

from app import agents, store
from app.harness import Task, run_eval

if __name__ == "__main__":
    server = pgserver.get_server("/tmp/eval-harness-demo-pg")
    os.environ["DATABASE_URL"] = server.get_uri()
    with store.connect() as conn:
        store.init_db(conn)
    tasks = [Task(**t) for t in agents.BUILTIN_TASKS]
    run_id, results, summary = run_eval(tasks, agents.scripted_agent)
    with store.connect() as conn:
        store.save_run(conn, run_id, "scripted", summary, results)
    print(f"run {run_id}")
    for r in results:
        mark = "PASS" if r.success else "FAIL"
        print(f"  [{mark}] {r.task_id:<10} {r.latency_s:.4f}s  cost ${r.cost_usd}  out: {r.output!r}")
    print("summary:", summary)
    server.cleanup()
