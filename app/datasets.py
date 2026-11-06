"""Versioned golden datasets for evaluated agents.

Datasets live under datasets/<name>/<version>.json, indexed by
datasets/manifest.json. Provenance for every dataset is recorded in the file
itself and in datasets/README.md.

Integrity rule: fixture results validate the harness. They are never published
as measured app quality - an app's numbers come only from a real run against
that app, labelled with model, dataset version, sample count, config, timestamp
and commit.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

from .harness import Task

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"


@dataclass
class GoldenDataset:
    name: str
    version: str
    target_agent: str
    provenance: str
    tasks: list = field(default_factory=list)  # list[Task]


def _manifest(root: Path) -> dict:
    path = root / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"no dataset manifest at {path}")
    return json.loads(path.read_text())


def parse_dataset_ref(ref: str):
    """'enterprise-support@v1' -> ('enterprise-support', 'v1'); no suffix -> latest."""
    name, _, version = ref.partition("@")
    if not name:
        raise ValueError(f"empty dataset name in {ref!r}")
    return name, version or None


def load_dataset(name: str, version: "str | None" = None,
                 root: Path = DATASETS_DIR) -> GoldenDataset:
    manifest = _manifest(root)
    entry = manifest.get("datasets", {}).get(name)
    if entry is None:
        raise KeyError(
            f"unknown dataset {name!r}; options: {sorted(manifest.get('datasets', {}))}")
    version = version or entry["latest"]
    if version not in entry["versions"]:
        raise KeyError(f"dataset {name!r} has no version {version!r}; has {entry['versions']}")
    path = root / name / f"{version}.json"
    raw = json.loads(path.read_text())
    if raw.get("name") != name or raw.get("version") != version:
        raise ValueError(
            f"dataset file {path} claims {raw.get('name')}@{raw.get('version')}, "
            f"expected {name}@{version}")
    tasks, seen = [], set()
    for t in raw.get("tasks", []):
        tid = t.get("id", "")
        if tid in seen:
            raise ValueError(f"duplicate task id {tid!r} in {path}")
        seen.add(tid)
        if not t.get("input"):
            raise ValueError(f"task {tid!r} in {path} has no input")
        if not t.get("expected_keywords") and not t.get("reference"):
            raise ValueError(
                f"task {tid!r} in {path} needs expected_keywords and/or a reference")
        tasks.append(Task(id=tid, input=t["input"],
                          expected_keywords=t.get("expected_keywords", []),
                          reference=t.get("reference"),
                          scorer=t.get("scorer"),
                          forbidden_keywords=t.get("forbidden_keywords", []),
                          expected_sources=t.get("expected_sources", []),
                          retrieval_k=t.get("retrieval_k", 5)))
    return GoldenDataset(name=name, version=version,
                         target_agent=raw.get("target_agent", ""),
                         provenance=raw.get("provenance", ""), tasks=tasks)
