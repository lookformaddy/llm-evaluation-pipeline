from __future__ import annotations

from pathlib import Path

import pytest

from llm_eval_pipeline import (
    EvalRunner,
    KnowledgeBaseSUT,
    build_manifest,
    default_scorers,
    load_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
KB_PATH = ROOT / "data" / "kb.json"
DATASET_PATH = ROOT / "data" / "golden" / "support_qa_v1.jsonl"


@pytest.fixture(scope="session")
def cases():
    return load_dataset(DATASET_PATH)


@pytest.fixture(scope="session")
def manifest(cases):
    return build_manifest(cases, "support-qa", "1.0.0")


@pytest.fixture(scope="session")
def sut():
    return KnowledgeBaseSUT.from_file(KB_PATH)


@pytest.fixture(scope="session")
def runner(sut):
    return EvalRunner(sut, default_scorers())
