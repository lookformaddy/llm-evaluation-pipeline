"""EvalRunner: execute a SUT over a golden dataset and score every case."""

from __future__ import annotations

import uuid

from . import metrics
from .schemas import (
    CaseOutput,
    CaseResult,
    DatasetManifest,
    EvalCase,
    RunReport,
    SUTConfig,
)
from .scorers import Scorer
from .sut import SystemUnderTest


class EvalRunner:
    def __init__(self, sut: SystemUnderTest, scorers: list[Scorer]) -> None:
        if not scorers:
            raise ValueError("at least one scorer is required")
        self.sut = sut
        self.scorers = scorers

    def run(
        self,
        cases: list[EvalCase],
        manifest: DatasetManifest,
        config: SUTConfig,
        run_id: str | None = None,
    ) -> RunReport:
        run_id = run_id or uuid.uuid4().hex[:12]
        results: list[CaseResult] = []
        for case in cases:
            try:
                output = self.sut.run(case, config)
            except Exception as exc:  # a crashing SUT must not kill the run
                output = CaseOutput(
                    case_id=case.id, output_text="", error=f"{type(exc).__name__}: {exc}"
                )
            scores = {}
            for scorer in self.scorers:
                try:
                    scores[scorer.name] = scorer.score(case, output)
                except Exception as exc:
                    from .schemas import ScoreResult

                    scores[scorer.name] = ScoreResult(
                        scorer=scorer.name, score=0.0, passed=False,
                        details={"error": f"{type(exc).__name__}: {exc}"},
                    )
            results.append(CaseResult(case=case, output=output, scores=scores))

        report = RunReport(
            run_id=run_id,
            dataset=manifest,
            sut_config=config,
            case_results=results,
        )
        report.metrics = metrics.aggregate(report)
        return report
