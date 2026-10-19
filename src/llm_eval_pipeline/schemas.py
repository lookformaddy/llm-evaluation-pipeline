"""Pydantic schemas for datasets, runs, scores, and reports.

Everything that crosses a module boundary in this pipeline is a pydantic
model, so malformed datasets, configs, or reports fail fast with a clear
validation error instead of corrupting a run.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------
# Golden dataset
# --------------------------------------------------------------------------


class EvalCase(BaseModel):
    """One golden test case.

    ``expected_output`` is the reference answer used by exact/semantic
    scorers. ``reference_facts`` are the atomic facts a correct, grounded
    answer should contain; the groundedness scorer checks them against the
    produced output and its citations.
    """

    id: str
    input: str = Field(..., description="User prompt / question sent to the SUT.")
    expected_output: Optional[str] = None
    expected_patterns: list[str] = Field(
        default_factory=list,
        description="Regex patterns that must all appear in the output.",
    )
    reference_facts: list[str] = Field(default_factory=list)
    available_docs: list[str] = Field(
        default_factory=list,
        description="Document ids the SUT is allowed to cite for this case.",
    )
    tags: dict[str, str] = Field(
        default_factory=dict,
        description="Slice labels, e.g. {'category': 'returns', 'difficulty': 'easy'}.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "input")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must be a non-empty string")
        return v


class DatasetManifest(BaseModel):
    """Version identity of a golden dataset.

    The ``content_hash`` is a SHA-256 over the canonical JSONL encoding of
    the cases, so any edit to any case produces a new version fingerprint.
    """

    name: str
    version: str
    description: str = ""
    case_count: int
    content_hash: str
    created_at: datetime = Field(default_factory=utcnow)


# --------------------------------------------------------------------------
# System under test
# --------------------------------------------------------------------------


class SUTConfig(BaseModel):
    """Configuration of the system under test for one run.

    ``options`` is intentionally free-form: each SUT implementation reads
    the knobs it understands (e.g. ``max_facts``, ``drop_citations``).
    Cost rates make latency/cost accounting reproducible offline.
    """

    name: str = "baseline"
    model: str = "offline-kb-assistant"
    temperature: float = 0.0
    input_cost_per_1k_tokens: float = 0.003
    output_cost_per_1k_tokens: float = 0.015
    base_latency_ms: float = 120.0
    latency_ms_per_output_token: float = 4.0
    options: dict[str, Any] = Field(default_factory=dict)


class TraceEvent(BaseModel):
    step: str
    detail: str = ""
    latency_ms: float = 0.0


class CaseOutput(BaseModel):
    """Raw output of the system under test for one case, with its trace."""

    case_id: str
    output_text: str
    citations: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    trace: list[TraceEvent] = Field(default_factory=list)
    error: Optional[str] = None


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


class ScoreResult(BaseModel):
    scorer: str
    score: float = Field(..., ge=0.0, le=1.0)
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)


class CaseResult(BaseModel):
    case: EvalCase
    output: CaseOutput
    scores: dict[str, ScoreResult] = Field(default_factory=dict)

    @property
    def mean_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.score for s in self.scores.values()) / len(self.scores)


class ScorerAggregate(BaseModel):
    mean: float
    min: float
    max: float
    pass_rate: float


class SliceAggregate(BaseModel):
    slice_key: str  # e.g. "category=returns"
    case_count: int
    mean_score: float
    per_scorer: dict[str, float] = Field(default_factory=dict)


class AggregateMetrics(BaseModel):
    case_count: int
    overall_mean: float
    per_scorer: dict[str, ScorerAggregate] = Field(default_factory=dict)
    slices: list[SliceAggregate] = Field(default_factory=list)
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_mean_ms: float = 0.0
    total_cost_usd: float = 0.0
    mean_cost_usd: float = 0.0
    error_count: int = 0


class RunReport(BaseModel):
    run_id: str
    started_at: datetime = Field(default_factory=utcnow)
    dataset: DatasetManifest
    sut_config: SUTConfig
    case_results: list[CaseResult] = Field(default_factory=list)
    metrics: Optional[AggregateMetrics] = None


# --------------------------------------------------------------------------
# Regression gating
# --------------------------------------------------------------------------


class Thresholds(BaseModel):
    """CI gate thresholds.

    A scorer fails the gate if its current mean is below ``min_scores`` or
    if it dropped more than ``max_drop`` versus the baseline run.
    """

    min_scores: dict[str, float] = Field(default_factory=dict)
    max_drop: dict[str, float] = Field(default_factory=dict)
    default_max_drop: float = 0.05
    max_latency_p95_ms: Optional[float] = None
    max_total_cost_usd: Optional[float] = None


class GateCheck(BaseModel):
    name: str
    passed: bool
    current: Optional[float] = None
    baseline: Optional[float] = None
    threshold: Optional[float] = None
    message: str = ""


class GateResult(BaseModel):
    passed: bool
    checks: list[GateCheck] = Field(default_factory=list)

    @property
    def failures(self) -> list[GateCheck]:
        return [c for c in self.checks if not c.passed]


# --------------------------------------------------------------------------
# Trend history
# --------------------------------------------------------------------------


class HistoryEntry(BaseModel):
    run_id: str
    recorded_at: datetime = Field(default_factory=utcnow)
    dataset_name: str
    dataset_hash: str
    sut_name: str
    overall_mean: float
    per_scorer_mean: dict[str, float] = Field(default_factory=dict)
    latency_p95_ms: float = 0.0
    total_cost_usd: float = 0.0
    gate_passed: Optional[bool] = None
