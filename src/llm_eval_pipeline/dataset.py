"""Golden dataset loading, validation, and versioning.

Datasets are JSONL files: one :class:`EvalCase` per line. The version
fingerprint is a SHA-256 over the canonical re-encoding of the parsed
cases, so whitespace-only differences in the file do not create phantom
versions, but any semantic change does.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .schemas import DatasetManifest, EvalCase


def load_dataset(path: str | Path) -> list[EvalCase]:
    """Load and validate a JSONL golden dataset.

    Raises ``ValueError`` with the offending line number on bad input, and
    on duplicate case ids (ids must be unique for slice/gate reporting to
    be meaningful).
    """
    path = Path(path)
    cases: list[EvalCase] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            case = EvalCase.model_validate(raw)
            if case.id in seen:
                raise ValueError(f"{path}:{lineno}: duplicate case id '{case.id}'")
            seen.add(case.id)
            cases.append(case)
    if not cases:
        raise ValueError(f"{path}: dataset contains no cases")
    return cases


def canonical_jsonl(cases: list[EvalCase]) -> str:
    """Deterministic JSONL encoding used for hashing and writing."""
    lines = [
        json.dumps(case.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        for case in cases
    ]
    return "\n".join(lines) + "\n"


def content_hash(cases: list[EvalCase]) -> str:
    return hashlib.sha256(canonical_jsonl(cases).encode("utf-8")).hexdigest()


def build_manifest(
    cases: list[EvalCase], name: str, version: str, description: str = ""
) -> DatasetManifest:
    return DatasetManifest(
        name=name,
        version=version,
        description=description,
        case_count=len(cases),
        content_hash=content_hash(cases),
    )


def write_dataset(cases: list[EvalCase], path: str | Path) -> None:
    Path(path).write_text(canonical_jsonl(cases), encoding="utf-8")
