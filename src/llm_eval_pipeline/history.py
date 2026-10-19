"""Trend history: one JSON file holding a summary entry per run.

History is deliberately boring — a JSON array on disk — so it can be
checked into a repo, diffed in code review, and rendered by the report
module without any database.
"""

from __future__ import annotations

import json
from pathlib import Path

from .schemas import GateResult, HistoryEntry, RunReport


def entry_from_report(report: RunReport, gate: GateResult | None = None) -> HistoryEntry:
    m = report.metrics
    assert m is not None
    return HistoryEntry(
        run_id=report.run_id,
        dataset_name=report.dataset.name,
        dataset_hash=report.dataset.content_hash,
        sut_name=report.sut_config.name,
        overall_mean=m.overall_mean,
        per_scorer_mean={k: v.mean for k, v in m.per_scorer.items()},
        latency_p95_ms=m.latency_p95_ms,
        total_cost_usd=m.total_cost_usd,
        gate_passed=(gate.passed if gate is not None else None),
    )


def load_history(path: str | Path) -> list[HistoryEntry]:
    path = Path(path)
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [HistoryEntry.model_validate(item) for item in raw]


def append_history(path: str | Path, entry: HistoryEntry) -> None:
    path = Path(path)
    entries = load_history(path)
    entries.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([e.model_dump(mode="json") for e in entries], indent=2),
        encoding="utf-8",
    )
