from __future__ import annotations

import json

import pytest

from llm_eval_pipeline import content_hash, load_dataset, write_dataset
from llm_eval_pipeline.dataset import build_manifest, canonical_jsonl
from llm_eval_pipeline.schemas import EvalCase


def test_bundled_dataset_loads(cases):
    assert len(cases) == 12
    assert all(c.id for c in cases)
    categories = {c.tags["category"] for c in cases}
    assert categories == {"returns", "shipping", "warranty", "account", "loyalty", "pricing"}


def test_content_hash_is_stable(cases):
    assert content_hash(cases) == content_hash(list(cases))


def test_content_hash_changes_on_edit(cases):
    edited = [cases[0].model_copy(update={"input": "a different question"})] + cases[1:]
    assert content_hash(edited) != content_hash(cases)


def test_canonical_jsonl_ignores_source_formatting(tmp_path):
    case = EvalCase(id="x", input="q?")
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text(json.dumps({"id": "x", "input": "q?"}) + "\n")
    b.write_text('{ "input" : "q?" , "id" : "x" }\n')
    assert content_hash(load_dataset(a)) == content_hash(load_dataset(b))
    assert canonical_jsonl([case]).endswith("\n")


def test_duplicate_ids_rejected(tmp_path):
    p = tmp_path / "dup.jsonl"
    line = json.dumps({"id": "x", "input": "q?"})
    p.write_text(line + "\n" + line + "\n")
    with pytest.raises(ValueError, match="duplicate case id"):
        load_dataset(p)


def test_invalid_json_reports_line(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id": "x", "input": "q?"}\nnot json\n')
    with pytest.raises(ValueError, match=":2:"):
        load_dataset(p)


def test_empty_dataset_rejected(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("\n\n")
    with pytest.raises(ValueError, match="no cases"):
        load_dataset(p)


def test_write_load_roundtrip(tmp_path, cases):
    p = tmp_path / "roundtrip.jsonl"
    write_dataset(cases, p)
    assert load_dataset(p) == cases


def test_manifest_fields(cases):
    m = build_manifest(cases, "support-qa", "1.0.0", "desc")
    assert m.case_count == 12
    assert m.content_hash == content_hash(cases)
    assert len(m.content_hash) == 64
