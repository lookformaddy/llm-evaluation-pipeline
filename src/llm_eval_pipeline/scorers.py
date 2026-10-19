"""Scorers: turn a (case, output) pair into a 0..1 score with a verdict.

Every scorer implements the :class:`Scorer` protocol and is independently
configurable, so a team can gate on exactly the signals it trusts:

* :class:`ExactMatchScorer` — normalized string equality with the reference.
* :class:`RegexMatchScorer` — required patterns (all must match).
* :class:`SemanticSimilarityScorer` — embedding similarity to the reference
  answer, using the local deterministic embedder by default.
* :class:`GroundednessScorer` — are the reference facts actually present,
  are the citations valid, and is anything cited that wasn't allowed
  (a hallucination signal)?
* :class:`JudgeScorer` — rubric-based LLM-as-judge behind the
  :class:`Judge` protocol, with :class:`OfflineRubricJudge`, a
  deterministic stub that scores rubric dimensions from observable
  signals so the pipeline runs (and is testable) without an API key.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Protocol, runtime_checkable

from .embedder import Embedder, HashEmbedder, cosine_similarity, tokenize
from .schemas import CaseOutput, EvalCase, ScoreResult


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).rstrip(".")


def _fact_coverage(facts: list[str], output_text: str) -> tuple[int, list[str]]:
    """A fact counts as present if >=70% of its content tokens appear."""
    out_tokens = set(tokenize(output_text))
    covered, missing = 0, []
    for fact in facts:
        f_tokens = set(tokenize(fact))
        if not f_tokens:
            continue
        overlap = len(f_tokens & out_tokens) / len(f_tokens)
        if overlap >= 0.7:
            covered += 1
        else:
            missing.append(fact)
    return covered, missing


@runtime_checkable
class Scorer(Protocol):
    name: str

    def score(self, case: EvalCase, output: CaseOutput) -> ScoreResult:
        ...


# --------------------------------------------------------------------------
# Lexical scorers
# --------------------------------------------------------------------------


class ExactMatchScorer:
    name = "exact_match"

    def score(self, case: EvalCase, output: CaseOutput) -> ScoreResult:
        if case.expected_output is None:
            return ScoreResult(
                scorer=self.name, score=0.0, passed=False,
                details={"skipped": "no expected_output on case"},
            )
        match = _normalize(output.output_text) == _normalize(case.expected_output)
        return ScoreResult(
            scorer=self.name, score=1.0 if match else 0.0, passed=match,
            details={"expected": case.expected_output},
        )


class RegexMatchScorer:
    name = "regex_match"

    def score(self, case: EvalCase, output: CaseOutput) -> ScoreResult:
        patterns = case.expected_patterns
        if not patterns:
            return ScoreResult(
                scorer=self.name, score=1.0, passed=True,
                details={"skipped": "no expected_patterns on case"},
            )
        hits = {p: bool(re.search(p, output.output_text, re.IGNORECASE)) for p in patterns}
        matched = sum(hits.values())
        score = matched / len(patterns)
        return ScoreResult(
            scorer=self.name, score=score, passed=matched == len(patterns),
            details={"patterns": hits},
        )


# --------------------------------------------------------------------------
# Semantic similarity
# --------------------------------------------------------------------------


class SemanticSimilarityScorer:
    name = "semantic_similarity"

    def __init__(self, embedder: Optional[Embedder] = None, threshold: float = 0.35) -> None:
        self.embedder = embedder or HashEmbedder()
        self.threshold = threshold

    def score(self, case: EvalCase, output: CaseOutput) -> ScoreResult:
        if case.expected_output is None:
            return ScoreResult(
                scorer=self.name, score=0.0, passed=False,
                details={"skipped": "no expected_output on case"},
            )
        sim = cosine_similarity(
            self.embedder.embed(output.output_text),
            self.embedder.embed(case.expected_output),
        )
        sim = max(0.0, min(1.0, sim))
        return ScoreResult(
            scorer=self.name, score=round(sim, 4), passed=sim >= self.threshold,
            details={"threshold": self.threshold},
        )


# --------------------------------------------------------------------------
# Groundedness / citations
# --------------------------------------------------------------------------


class GroundednessScorer:
    """Score = fact coverage, penalized by invalid/hallucinated citations.

    * coverage: fraction of reference facts present in the output.
    * a citation to a doc not in ``case.available_docs`` is treated as a
      hallucinated citation and multiplies the score by 0.5.
    * if the case expects grounding (has reference facts) and the output
      cites nothing, the score is multiplied by 0.8 — facts stated with
      no source are weaker evidence than facts with a valid citation.
    """

    name = "groundedness"

    def score(self, case: EvalCase, output: CaseOutput) -> ScoreResult:
        facts = case.reference_facts
        if not facts:
            return ScoreResult(
                scorer=self.name, score=1.0, passed=True,
                details={"skipped": "no reference_facts on case"},
            )
        covered, missing = _fact_coverage(facts, output.output_text)
        score = covered / len(facts)
        allowed = set(case.available_docs)
        invalid = [c for c in output.citations if allowed and c not in allowed]
        penalties: list[str] = []
        if invalid:
            score *= 0.5
            penalties.append(f"hallucinated_citations={invalid}")
        if not output.citations:
            score *= 0.8
            penalties.append("no_citations")
        return ScoreResult(
            scorer=self.name, score=round(score, 4),
            passed=score >= 0.8 and not invalid,
            details={
                "facts_covered": covered,
                "facts_total": len(facts),
                "missing_facts": missing,
                "citations": output.citations,
                "penalties": penalties,
            },
        )


# --------------------------------------------------------------------------
# LLM-as-judge (rubric) with a deterministic offline stub
# --------------------------------------------------------------------------


class JudgeVerdict:
    def __init__(self, dimensions: dict[str, float], rationale: str) -> None:
        self.dimensions = dimensions
        self.rationale = rationale

    @property
    def overall(self) -> float:
        if not self.dimensions:
            return 0.0
        return sum(self.dimensions.values()) / len(self.dimensions)


@runtime_checkable
class Judge(Protocol):
    """An LLM-as-judge: grade an output against a rubric.

    Production implementations call a judge model with the case, the
    reference answer, the produced output, and the rubric dimensions,
    and parse the model's structured grades into a JudgeVerdict.
    """

    def judge(
        self,
        case: EvalCase,
        output: CaseOutput,
        rubric: list[str],
    ) -> JudgeVerdict:
        ...


class OfflineRubricJudge:
    """Deterministic judge stub — no API key, fully reproducible.

    It approximates each rubric dimension from observable signals
    (reference-fact coverage, valid citations, expected-pattern hits,
    answer-length sanity). It exists so pipelines and tests can exercise
    the judge *interface* offline; treat its grades as smoke-test
    signal, not as a substitute for a real judge model.
    """

    KNOWN_DIMENSIONS = ("relevance", "completeness", "grounding", "clarity")

    def judge(
        self,
        case: EvalCase,
        output: CaseOutput,
        rubric: list[str],
    ) -> JudgeVerdict:
        dims: dict[str, float] = {}
        facts = case.reference_facts
        covered, _ = _fact_coverage(facts, output.output_text) if facts else (0, [])
        coverage = (covered / len(facts)) if facts else 0.0

        q_tokens = set(tokenize(case.input))
        o_tokens = set(tokenize(output.output_text))
        relevance = (
            len(q_tokens & o_tokens) / len(q_tokens) if q_tokens else 0.0
        )
        words = len(output.output_text.split())
        clarity = 1.0 if 5 <= words <= 120 else (0.6 if words > 0 else 0.0)
        grounding = 0.0
        if output.citations:
            allowed = set(case.available_docs)
            valid = [c for c in output.citations if not allowed or c in allowed]
            grounding = len(valid) / len(output.citations)

        signals = {
            "relevance": round(relevance, 4),
            "completeness": round(coverage, 4),
            "grounding": round(grounding, 4),
            "clarity": clarity,
        }
        for dim in rubric:
            dims[dim] = signals.get(dim, round((coverage + relevance) / 2, 4))
        rationale = (
            f"offline stub: coverage={coverage:.2f} relevance={relevance:.2f} "
            f"grounding={grounding:.2f} words={words}"
        )
        return JudgeVerdict(dims, rationale)


class JudgeScorer:
    name = "llm_judge"

    def __init__(
        self,
        judge: Optional[Judge] = None,
        rubric: Optional[list[str]] = None,
        threshold: float = 0.6,
    ) -> None:
        self.judge = judge or OfflineRubricJudge()
        self.rubric = rubric or ["relevance", "completeness", "grounding", "clarity"]
        self.threshold = threshold

    def score(self, case: EvalCase, output: CaseOutput) -> ScoreResult:
        verdict = self.judge.judge(case, output, self.rubric)
        overall = verdict.overall
        return ScoreResult(
            scorer=self.name, score=round(overall, 4),
            passed=overall >= self.threshold,
            details={
                "dimensions": verdict.dimensions,
                "rationale": verdict.rationale,
                "judge": type(self.judge).__name__,
            },
        )


def default_scorers() -> list[Scorer]:
    """The standard scorer bundle used by the CLI demo and API."""
    return [
        RegexMatchScorer(),
        SemanticSimilarityScorer(),
        GroundednessScorer(),
        JudgeScorer(),
    ]


def scorer_by_name(name: str) -> Scorer:
    bundle: dict[str, Any] = {s.name: s for s in default_scorers()}
    bundle[ExactMatchScorer.name] = ExactMatchScorer()
    if name not in bundle:
        raise ValueError(f"unknown scorer '{name}'; known: {sorted(bundle)}")
    return bundle[name]
