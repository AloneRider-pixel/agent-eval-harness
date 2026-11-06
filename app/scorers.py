"""Pluggable scorers for eval tasks.

Backends:
- keyword: deterministic substring checks (the harness default, always available).
- semantic: cosine similarity between sentence embeddings of the agent output and
  a per-task reference answer. Dependency-free by injection; the default embedder
  uses the optional sentence-transformers package when installed, so default CI
  stays offline and deterministic.
- judge: an LLM judge grades the output against the reference. Credentialed judge
  calls can cost money, so the judge activates only when a provider key is set in
  the environment for an explicitly approved run. Default CI never runs it; tests
  inject a scripted judge instead.
"""
import math
import os
from dataclasses import dataclass


class ScorerUnavailable(RuntimeError):
    """Raised when a scorer's backend is not configured in this environment."""


@dataclass
class ScoreResult:
    passed: bool
    score: float  # 0.0 .. 1.0
    detail: str = ""


# ---- keyword ---------------------------------------------------------------


class KeywordScorer:
    name = "keyword"

    def score_task(self, task, output: str) -> ScoreResult:
        expected = task.expected_keywords
        if not expected:
            return ScoreResult(True, 1.0, "no expectations")
        low = output.lower()
        missing = [k for k in expected if k.lower() not in low]
        hits = len(expected) - len(missing)
        frac = hits / len(expected)
        detail = f"{hits}/{len(expected)} keywords"
        if missing:
            detail += f"; missing: {missing}"
        return ScoreResult(not missing, round(frac, 4), detail)


# ---- semantic --------------------------------------------------------------


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


def default_embedder():
    """sentence-transformers MiniLM embedder, when the optional extra is installed."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ScorerUnavailable(
            "semantic scoring needs the optional 'sentence-transformers' package "
            "(pip install sentence-transformers) or an injected embedder"
        ) from exc
    model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(texts):
        return [list(map(float, v)) for v in model.encode(texts)]

    return embed


class SemanticScorer:
    """Cosine similarity between the agent output and the task's reference answer.

    The default threshold (0.6) is a starting point for MiniLM embeddings, not a
    calibrated value; calibrate per agent before trusting it in a gate.
    """

    def __init__(self, embedder=None, threshold: float = 0.6):
        self._embedder = embedder  # callable(list[str]) -> list[list[float]]
        self.threshold = threshold

    @property
    def name(self):
        return "semantic"

    def _embed(self, texts):
        if self._embedder is None:
            self._embedder = default_embedder()
        return self._embedder(texts)

    def score_task(self, task, output: str) -> ScoreResult:
        if not task.reference:
            return ScoreResult(False, 0.0, "semantic scoring needs task.reference")
        if not output.strip():
            return ScoreResult(False, 0.0, "empty output")
        ref_vec, out_vec = self._embed([task.reference, output])
        sim = _cosine(ref_vec, out_vec)
        return ScoreResult(
            sim >= self.threshold, round(sim, 4),
            f"cosine {sim:.3f} vs threshold {self.threshold}")


# ---- LLM judge -------------------------------------------------------------

JUDGE_PROMPT = """You are grading an AI agent's answer against a reference answer.

Task given to the agent:
{task}

Reference answer:
{reference}

Agent's answer:
{output}

Grade factual correctness and completeness against the reference, not style.
Reply with exactly two lines:
SCORE: <0.0-1.0>
WHY: <one sentence>"""


def _parse_judgement(text: str):
    score, why = None, ""
    for line in text.splitlines():
        head, _, rest = line.partition(":")
        if head.strip().upper() == "SCORE":
            try:
                score = float(rest.strip())
            except ValueError:
                score = None
        elif head.strip().upper() == "WHY":
            why = rest.strip()
    if score is None or not 0.0 <= score <= 1.0:
        raise ValueError(f"judge returned no valid SCORE line: {text!r}")
    return score, why


def groq_judge(model: str = "llama-3.3-70b-versatile", api_key: "str | None" = None,
               base_url: str = "https://api.groq.com/openai/v1"):
    """Build a judge callable backed by an OpenAI-compatible chat endpoint.

    Credentialed: every call bills the configured key. Constructed only when
    EVAL_JUDGE_API_KEY is set, which is reserved for explicitly approved runs;
    default CI stays key-free and never builds this.
    """
    import httpx  # already a harness dependency; imported lazily

    key = api_key or os.getenv("EVAL_JUDGE_API_KEY")
    if not key:
        raise ScorerUnavailable(
            "LLM judge is credentialed and disabled: set EVAL_JUDGE_API_KEY only "
            "for an explicitly approved run; default CI stays key-free")

    def judge(task_input: str, reference: str, output: str):
        resp = httpx.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "temperature": 0,
                  "messages": [{"role": "user", "content": JUDGE_PROMPT.format(
                      task=task_input, reference=reference, output=output)}]},
            timeout=60)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        return _parse_judgement(text)

    return judge


class LLMJudgeScorer:
    """Grades output against the task's reference via an injected or credentialed judge."""

    def __init__(self, judge=None, threshold: float = 0.7):
        # judge: callable(task_input, reference, output) -> (score_0_to_1, rationale)
        self._judge = judge
        self.threshold = threshold

    @property
    def name(self):
        return "judge"

    def score_task(self, task, output: str) -> ScoreResult:
        if not task.reference:
            return ScoreResult(False, 0.0, "LLM judge needs task.reference")
        if self._judge is None:
            self._judge = groq_judge()
        value, rationale = self._judge(task.input, task.reference, output)
        value = round(float(value), 4)
        return ScoreResult(value >= self.threshold, value,
                           f"judge {value}: {rationale}")


# ---- factory ---------------------------------------------------------------

_SCORERS = {"keyword": KeywordScorer, "semantic": SemanticScorer, "judge": LLMJudgeScorer}


def get_scorer(name: str, **kwargs):
    """name: keyword | semantic | judge."""
    if name not in _SCORERS:
        raise ValueError(f"unknown scorer {name!r}; options: {sorted(_SCORERS)}")
    return _SCORERS[name](**kwargs)
