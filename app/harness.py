"""Core eval harness: run tasks against an agent, measure success, latency, cost."""
import time
import uuid
from dataclasses import dataclass, field

from .scorers import ScoreResult

# USD per 1K tokens (prompt, completion) - free-tier models cost 0.00 on free tier;
# prices shown for honesty when a paid key is used. Keep this table current manually.
PRICING = {
    "llama-3.3-70b-versatile": (0.00059, 0.00079),
    "gemini-2.0-flash": (0.00010, 0.00040),
    "llama3.2": (0.0, 0.0),            # local Ollama: no API cost
    "mock": (0.0, 0.0),
}


@dataclass
class Task:
    id: str
    input: str
    expected_keywords: list[str] = field(default_factory=list)
    reference: str | None = None   # reference answer for semantic / judge scorers
    scorer: str | None = None      # per-task scorer override (keyword|semantic|judge|guardrail)
    forbidden_keywords: list[str] = field(default_factory=list)  # guardrail: must not appear
    expected_sources: list[str] = field(default_factory=list)    # retrieval: relevant source ids
    retrieval_k: int = 5           # retrieval: k for precision@k


@dataclass
class AgentResult:
    output: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = "mock"
    retrieved: list[str] | None = None   # retrieval: source ids, ranked
    contexts: list[str] | None = None    # retrieval: chunk texts


@dataclass
class TaskResult:
    task_id: str
    success: bool
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    output: str
    error: str | None = None
    score: float = 0.0
    scorer: str = "keyword"
    detail: str = ""
    retrieval: dict | None = None        # {"precision_at_k","faithfulness","k"}


def estimate_cost(model, prompt_tokens, completion_tokens):
    """Returns (cost_usd, priced: bool). Unknown models are reported, not guessed."""
    if model in PRICING:
        p_in, p_out = PRICING[model]
        return (prompt_tokens / 1000) * p_in + (completion_tokens / 1000) * p_out, True
    return 0.0, False


def score(output: str, expected_keywords: list[str]) -> bool:
    low = output.lower()
    return all(k.lower() in low for k in expected_keywords)


def run_eval(tasks: list[Task], agent, run_id: str | None = None, scorer=None):
    """agent: callable(input: str) -> AgentResult. Returns (run_id, results, summary).

    scorer: a scorer name ("keyword"|"semantic"|"judge"), a scorer instance, or
    None (keyword). A task's own .scorer overrides the run default. Scorer
    backend errors (ScorerUnavailable, provider failures) abort the run loudly -
    a misconfigured scorer must never masquerade as agent failures.
    """
    from .scorers import get_scorer

    run_id = run_id or uuid.uuid4().hex[:10]
    cache: dict = {}

    def resolve(task):
        if task.scorer:
            name = task.scorer
        elif isinstance(scorer, str):
            name = scorer
        elif scorer is not None:
            return scorer, getattr(scorer, "name", type(scorer).__name__)
        else:
            name = "keyword"
        if name not in cache:
            cache[name] = get_scorer(name)
        return cache[name], name

    results: list[TaskResult] = []
    for task in tasks:
        scr, scr_name = resolve(task)
        t0 = time.perf_counter()
        error = None
        try:
            res = agent(task.input)
        except Exception as exc:  # noqa: BLE001 - an agent failure is a measured failure
            res, error = AgentResult(output=""), str(exc)
        latency = time.perf_counter() - t0
        cost, _ = estimate_cost(res.model, res.prompt_tokens, res.completion_tokens)
        if error is not None:
            sres = ScoreResult(False, 0.0, "agent error")
        else:
            sres = scr.score_task(task, res.output)
        retrieval = None
        if error is None and task.expected_sources and res.retrieved is not None:
            from .retrieval import faithfulness_proxy, precision_at_k
            retrieval = {
                "k": task.retrieval_k,
                "precision_at_k": precision_at_k(
                    res.retrieved, set(task.expected_sources), task.retrieval_k),
                "faithfulness": faithfulness_proxy(res.output, res.contexts or []),
            }
        results.append(TaskResult(
            task_id=task.id,
            success=sres.passed,
            score=sres.score,
            scorer=scr_name,
            detail=sres.detail,
            retrieval=retrieval,
            latency_s=round(latency, 4),
            prompt_tokens=res.prompt_tokens,
            completion_tokens=res.completion_tokens,
            cost_usd=round(cost, 8),
            output=res.output,
            error=error))

    n = len(results)
    successes = sum(1 for r in results if r.success)
    latencies = sorted(r.latency_s for r in results)
    with_retrieval = [r.retrieval for r in results if r.retrieval]
    scorer_stats: dict = {}
    for r in results:
        st = scorer_stats.setdefault(r.scorer, {"tasks": 0, "passed": 0})
        st["tasks"] += 1
        st["passed"] += 1 if r.success else 0
    summary = {
        "run_id": run_id,
        "tasks": n,
        "successes": successes,
        "mean_score": round(sum(r.score for r in results) / n, 4) if n else 0.0,
        "scorers": scorer_stats,
        "retrieval": ({
            "tasks": len(with_retrieval),
            "mean_precision_at_k": round(
                sum(r["precision_at_k"] for r in with_retrieval) / len(with_retrieval), 4),
            "mean_faithfulness": round(
                sum(r["faithfulness"] for r in with_retrieval) / len(with_retrieval), 4),
        } if with_retrieval else None),
        "success_rate": round(successes / n, 4) if n else 0.0,
        "avg_latency_s": round(sum(latencies) / n, 4) if n else 0.0,
        "p95_latency_s": latencies[min(n - 1, int(0.95 * (n - 1)))] if n else 0.0,
        "total_prompt_tokens": sum(r.prompt_tokens for r in results),
        "total_completion_tokens": sum(r.completion_tokens for r in results),
        "total_cost_usd": round(sum(r.cost_usd for r in results), 8),
    }
    return run_id, results, summary
