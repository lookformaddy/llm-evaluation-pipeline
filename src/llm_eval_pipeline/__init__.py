"""LLM Evaluation Pipeline — a CI-grade evaluation harness for LLM apps.

Public API re-exports for convenience; see the module docstrings and
``docs/architecture.md`` for the full picture.
"""

from .dataset import build_manifest, content_hash, load_dataset, write_dataset
from .gating import compare_runs
from .runner import EvalRunner
from .schemas import (
    EvalCase,
    GateResult,
    RunReport,
    SUTConfig,
    Thresholds,
)
from .scorers import default_scorers
from .sut import KnowledgeBaseSUT, SystemUnderTest

__version__ = "1.0.0"

__all__ = [
    "EvalCase",
    "EvalRunner",
    "GateResult",
    "KnowledgeBaseSUT",
    "RunReport",
    "SUTConfig",
    "SystemUnderTest",
    "Thresholds",
    "build_manifest",
    "compare_runs",
    "content_hash",
    "default_scorers",
    "load_dataset",
    "write_dataset",
]
