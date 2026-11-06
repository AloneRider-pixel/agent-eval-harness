# Golden datasets

Versioned task sets for evaluating real agents through the harness. Each dataset
pins the behaviour the target agent is expected to have, with references for the
semantic and LLM-judge scorers.

## Layout

- `manifest.json` - dataset names, known versions, and the latest version.
- `<name>/<version>.json` - the tasks: `id`, `input`, `expected_keywords`,
  `reference` (for semantic / judge scoring), `tags`, plus dataset-level
  `target_agent` and `provenance`.

## Datasets

| Name | Target agent | Version | Tasks | Provenance |
|---|---|---|---|---|
| `enterprise-support` | enterprise-ai-agent | v1 | 8 | Synthetic support scenarios authored 2026-11-04 against the agent's documented scope (grounded RAG answers with citations, bounded support tools, safety boundaries). No production tickets or customer data. |
| `aegis` | aegis | v1 | 8 | Synthetic incident scenarios authored 2026-11-04 against the platform's documented operating model (evidence-backed investigation, remediation behind explicit human approval, audit trail). No production telemetry. |
| `adversarial` | any (run once per agent) | v1 | 10 | Synthetic attack prompts authored 2026-11-04 in three categories: prompt injection, fabrication pressure, off-lane requests. Each task pins lane markers (`expected_keywords`) and compromise indicators (`forbidden_keywords`), scored pass/fail by the `guardrail` scorer. No real prompts or secrets. |

## Versioning rules

- New tasks or changed references are a new version (`v2`, ...); released versions
  are immutable so runs stay comparable over time.
- `manifest.json` always points `latest` at the version new runs should use.

## Integrity

Results on these fixtures validate the harness and the scorer wiring. They are
labelled as harness validation, never as measured app quality: an app's published
numbers come only from a real run against that app, with model, dataset version,
sample count, config, timestamp and commit recorded.
