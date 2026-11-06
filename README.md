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

## Scorers

Each task is scored by one of three pluggable scorers (`scorer` per task or per run;
default `keyword`):

- **keyword** - deterministic substring checks, always available, fully offline.
- **semantic** - cosine similarity between sentence embeddings of the agent output
  and the task's `reference` answer. Needs the optional extra
  `pip install sentence-transformers` (or an injected embedder); never required for
  CI, which stays offline and deterministic.
- **judge** - an LLM judge grades the output against the reference. Credentialed:
  it activates only when `EVAL_JUDGE_API_KEY` is set, which is reserved for
  explicitly approved runs. A saved key existing is not approval. Default CI never
  builds or calls the judge; its logic is tested with injected scripted judges.

```json
POST /runs {"agent": "scripted", "dataset": "aegis@v1", "scorer": "keyword"}
```

`GET /datasets` lists the available golden datasets.

## Golden datasets

Versioned task sets under `datasets/` (`manifest.json` + `<name>/<version>.json`)
for the real agents this harness evaluates: `enterprise-support` and `aegis`,
8 tasks each, with references for semantic/judge scoring and documented
provenance. Released versions are immutable; changes land as a new version so
runs stay comparable.

Integrity rule: results on these fixtures validate the harness and the scorer
wiring. They are labelled harness validation, never measured app quality - an
app's numbers appear only after a real run against that app, with model, dataset
version, sample count, config, timestamp and commit recorded.

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

- Keyword scoring is the deterministic default. The semantic scorer's default
  threshold (0.6) is a starting point, not a calibrated value; calibrate per agent
  before trusting it in a gate. The LLM judge's real provider path
  (`groq_judge`) is implemented but intentionally never exercised in CI -
  credentialed runs are separately approved and separately reported.
- The bundled agents are deterministic stand-ins; point `run_eval` at your real agent
  callable to evaluate it.
- Token counts come from the agent's own report; the harness trusts but records them.
