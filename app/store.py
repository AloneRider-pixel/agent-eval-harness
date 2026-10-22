"""Postgres persistence for eval runs and per-task results."""
import json
import os

import psycopg2

DDL = """
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id text PRIMARY KEY,
    created_at timestamptz DEFAULT now(),
    agent_name text,
    summary jsonb NOT NULL
);
CREATE TABLE IF NOT EXISTS eval_results (
    id serial PRIMARY KEY,
    run_id text REFERENCES eval_runs(run_id),
    task_id text,
    success boolean,
    latency_s real,
    prompt_tokens int,
    completion_tokens int,
    cost_usd double precision,
    output text,
    error text
);
"""


def connect(url=None):
    return psycopg2.connect(url or os.getenv("DATABASE_URL", ""))


def init_db(conn):
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()


def save_run(conn, run_id, agent_name, summary, results):
    with conn.cursor() as cur:
        cur.execute("INSERT INTO eval_runs (run_id, agent_name, summary) VALUES (%s, %s, %s)",
                    (run_id, agent_name, json.dumps(summary)))
        for r in results:
            cur.execute(
                "INSERT INTO eval_results (run_id, task_id, success, latency_s,"
                " prompt_tokens, completion_tokens, cost_usd, output, error)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (run_id, r.task_id, r.success, r.latency_s, r.prompt_tokens,
                 r.completion_tokens, r.cost_usd, r.output, r.error))
    conn.commit()


def get_run(conn, run_id):
    with conn.cursor() as cur:
        cur.execute("SELECT summary FROM eval_runs WHERE run_id = %s", (run_id,))
        row = cur.fetchone()
        if not row:
            return None, []
        cur.execute("SELECT task_id, success, latency_s, cost_usd, error"
                    " FROM eval_results WHERE run_id = %s ORDER BY id", (run_id,))
        results = [{"task_id": t, "success": s, "latency_s": l, "cost_usd": c, "error": e}
                   for t, s, l, c, e in cur.fetchall()]
    return row[0], results


def list_runs(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT run_id, agent_name, created_at, summary->>'success_rate'"
                    " FROM eval_runs ORDER BY created_at DESC LIMIT 50")
        return [{"run_id": r, "agent": a, "created_at": str(t), "success_rate": s}
                for r, a, t, s in cur.fetchall()]
