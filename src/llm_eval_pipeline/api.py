"""Optional FastAPI surface over the pipeline.

Run with:  ``uvicorn llm_eval_pipeline.api:app --reload``
(or ``python -m llm_eval_pipeline.api`` for a quick local server)

Endpoints
---------
GET  /health            liveness + bundled dataset fingerprint
POST /validate-dataset  body: {"cases": [...]} -> validity + content hash
POST /score             body: {"case": {...}, "output_text": str,
                        "citations": [...]} -> per-scorer results
POST /runs              body: {"config": {...sut config...}} -> runs the
                        bundled golden dataset and returns the report
GET  /history           trend history recorded by /runs calls
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .dataset import build_manifest, content_hash, load_dataset
from .history import append_history, entry_from_report, load_history
from .runner import EvalRunner
from .schemas import CaseOutput, EvalCase, SUTConfig
from .scorers import default_scorers
from .sut import KnowledgeBaseSUT

_HERE = Path(__file__).resolve().parent
_DATA = _HERE.parents[1] / "data"
KB_PATH = _DATA / "kb.json"
DATASET_PATH = _DATA / "golden" / "support_qa_v1.jsonl"
HISTORY_PATH = _HERE.parents[1] / "out" / "history.json"

app = FastAPI(title="LLM Evaluation Pipeline", version="1.0.0")


class ValidateRequest(BaseModel):
    cases: list[EvalCase]


class ScoreRequest(BaseModel):
    case: EvalCase
    output_text: str
    citations: list[str] = []


class RunRequest(BaseModel):
    config: Optional[SUTConfig] = None


def _runner() -> EvalRunner:
    return EvalRunner(KnowledgeBaseSUT.from_file(KB_PATH), default_scorers())


@app.get("/health")
def health() -> dict[str, Any]:
    cases = load_dataset(DATASET_PATH)
    return {
        "status": "ok",
        "dataset_cases": len(cases),
        "dataset_hash": content_hash(cases),
    }


@app.post("/validate-dataset")
def validate_dataset(req: ValidateRequest) -> dict[str, Any]:
    ids = [c.id for c in req.cases]
    if len(ids) != len(set(ids)):
        raise HTTPException(status_code=422, detail="duplicate case ids")
    return {"valid": True, "case_count": len(req.cases),
            "content_hash": content_hash(req.cases)}


@app.post("/score")
def score(req: ScoreRequest) -> dict[str, Any]:
    output = CaseOutput(
        case_id=req.case.id, output_text=req.output_text, citations=req.citations
    )
    results = {
        s.name: s.score(req.case, output).model_dump() for s in default_scorers()
    }
    return {"case_id": req.case.id, "scores": results}


@app.post("/runs")
def create_run(req: RunRequest) -> dict[str, Any]:
    cases = load_dataset(DATASET_PATH)
    manifest = build_manifest(cases, "support-qa", "1.0.0", "bundled golden dataset")
    config = req.config or SUTConfig(name="api-run")
    report = _runner().run(cases, manifest, config)
    append_history(HISTORY_PATH, entry_from_report(report))
    return report.model_dump(mode="json")


@app.get("/history")
def history() -> list[dict[str, Any]]:
    return [e.model_dump(mode="json") for e in load_history(HISTORY_PATH)]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
