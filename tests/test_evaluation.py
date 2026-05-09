import pytest

from document_intelligence_rag.evaluation import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"b", "d"}, k=2) == 0.5


def test_precision_at_k():
    assert precision_at_k(["a", "b", "c"], {"b", "c"}, k=2) == 0.5


def test_mean_reciprocal_rank():
    score = mean_reciprocal_rank(
        [["a", "b", "c"], ["x", "y"]],
        [{"b"}, {"z"}],
    )

    assert score == 0.25


def test_metric_k_must_be_positive():
    with pytest.raises(ValueError, match="greater than 0"):
        recall_at_k(["a"], {"a"}, k=0)
