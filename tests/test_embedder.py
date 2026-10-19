from __future__ import annotations

import pytest

from llm_eval_pipeline.embedder import HashEmbedder, cosine_similarity, tokenize


def test_tokenize_lowercases_and_splits():
    assert tokenize("Refunds, WITHIN 5 Days!") == ["refunds", "within", "5", "days"]


def test_embedder_is_deterministic():
    emb = HashEmbedder()
    assert emb.embed("standard shipping is free") == emb.embed("standard shipping is free")


def test_identical_text_similarity_is_one():
    emb = HashEmbedder()
    v = emb.embed("the warranty lasts two years")
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_paraphrase_beats_unrelated():
    emb = HashEmbedder()
    ref = emb.embed("refunds are issued within five business days")
    para = emb.embed("you will get your refund within five business days")
    unrel = emb.embed("loyalty points expire after twelve months of inactivity")
    assert cosine_similarity(ref, para) > cosine_similarity(ref, unrel)


def test_empty_text_zero_vector():
    emb = HashEmbedder()
    v = emb.embed("")
    assert all(x == 0.0 for x in v)
    assert cosine_similarity(v, v) == 0.0


def test_dimension_mismatch_rejected():
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 0.0], [1.0])
