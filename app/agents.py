"""Built-in example agents to evaluate. Real agents wrap an LLM; these are
deterministic so the harness itself is testable offline."""
from .harness import AgentResult

BUILTIN_TASKS = [
    {"id": "capital-fr", "input": "What is the capital of France?",
     "expected_keywords": ["paris"]},
    {"id": "sum", "input": "What is 17 + 25?", "expected_keywords": ["42"]},
    {"id": "sql-def", "input": "In one sentence, what does SQL stand for?",
     "expected_keywords": ["structured", "query", "language"]},
]


def scripted_agent(input: str) -> AgentResult:
    """Deterministic stand-in agent (offline). Right on 2 of 3 builtin tasks,
    so success_rate is meaningfully not 1.0 in demos."""
    low = input.lower()
    if "capital of france" in low:
        out = "The capital of France is Paris."
    elif "17 + 25" in low:
        out = "17 + 25 = 42."
    else:
        out = "I do not know."
    return AgentResult(output=out, prompt_tokens=len(input.split()),
                       completion_tokens=len(out.split()), model="mock")


def always_fail_agent(input: str) -> AgentResult:
    raise RuntimeError("agent exploded")
