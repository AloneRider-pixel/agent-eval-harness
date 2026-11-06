"""Retrieval metrics for RAG agents: precision@k and a faithfulness proxy.

An agent under evaluation can report what it retrieved by returning
AgentResult(retrieved=[source ids], contexts=[chunk texts]). When a task also
carries expected_sources, run_eval records per-task retrieval metrics.

- precision@k is exact: the fraction of the k highest-ranked retrieved source
  ids that appear in the task's expected_sources.
- faithfulness here is a deterministic lexical proxy: the share of content
  words in the agent's answer that are covered by the retrieved contexts. It is
  cheap, offline and explainable - and it is a proxy, not neural entailment.
  The credentialed LLM judge is the accurate path when an approved run needs
  one; this proxy exists so default CI can still catch unsupported answers.

Both are recorded metrics. They become gate thresholds with the budgets and
baselines work, not pass/fail flags on individual tasks.
"""
import re

_WORD = re.compile(r"[a-z0-9]+")

_STOP = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "for", "to",
    "of", "in", "on", "at", "by", "with", "from", "as", "is", "are", "was",
    "were", "be", "been", "it", "its", "this", "that", "these", "those",
    "you", "your", "we", "our", "they", "their", "i", "me", "my", "do",
    "does", "did", "not", "no", "yes", "can", "could", "should", "would",
    "will", "shall", "may", "might", "must", "have", "has", "had", "what",
    "when", "where", "which", "who", "how", "why", "than", "so", "such",
}


def content_words(text: str) -> set:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


def precision_at_k(retrieved: list, expected: set, k: int) -> float:
    """Exact precision of the top-k retrieved source ids against expected ones."""
    if k <= 0:
        raise ValueError("k must be positive")
    top = list(retrieved)[:k]
    if not top:
        return 0.0
    hits = sum(1 for r in top if r in expected)
    return round(hits / len(top), 4)


def faithfulness_proxy(output: str, contexts: list) -> float:
    """Share of answer content words covered by the retrieved contexts (0..1)."""
    words = content_words(output)
    if not words:
        return 1.0  # nothing to support: trivially faithful
    covered: set = set()
    for ctx in contexts or []:
        covered |= content_words(ctx)
    if not covered:
        return 0.0
    return round(len(words & covered) / len(words), 4)
