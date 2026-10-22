# Agent Eval Harness

Measure agent quality like an engineering metric: success rate, latency, and cost
per task, persisted in Postgres, exposed over FastAPI.

## What it does

- `POST /runs` - runs a named agent against a task set (built-in or supplied in the
  request body), scoring each task, timing it, and pricing its tokens.
- `GET /runs` / `GET /runs/{id}` - list past runs or fetch one run's summary +
  per-task results.
- Tasks are `{"id", "input", "expected_keywords"}`; an agent is any callable
  `input -> AgentResult(output, prompt_tokens, completion_tokens, model)`, so you
  can wrap Groq/Gemini/Ollama agents without changing the harness.

## Metrics

- **Success:** every expected keyword appears in the agent output (case-insensitive).
  An agent exception is a measured failure, not a crash.
- **Latency:** avg and p95 per run, wall-clock per task.
- **Cost:** per-1K-token pricing table for known models; unknown models are reported
  as unpriced (cost 0 with `priced=false` in code) instead of being guessed.

## Storage

Postgres (embedded via pgserver in tests/demo; `DATABASE_URL` in production):
`eval_runs` (summary JSONB) + `eval_results` (one row per task).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export DATABASE_URL=postgresql://user:pass@host/db   # optional locally; demo embeds one
uvicorn app.main:app --reload
```

## Demo + tests

```bash
python demo.py    # evaluates the scripted agent on 3 tasks, persists to a real Postgres
pytest tests -q   # 6 tests incl. real-Postgres store roundtrip and API
```

Verified locally: scripted agent scores 2/3 on the builtin set (success_rate 0.6667),
agent exceptions recorded as failures, store roundtrip on embedded Postgres 15,
API creates/lists/fetches runs and 404s unknown ones.

## Honest limitations

- Keyword scoring is a simple, transparent judge; semantic/LLM-judge scoring is a
  documented extension point, not included.
- The bundled agents are deterministic stand-ins; point `run_eval` at your real agent
  callable to evaluate it.
- Token counts come from the agent's own report; the harness trusts but records them.
