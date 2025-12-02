"""Tests for core metrics module."""

import pytest

from biovector_eval.core.metrics import (
    mean_reciprocal_rank,
    recall_at_k,
)


class TestRecallAtK:
    """Tests for recall_at_k function."""

    def test_perfect_recall(self) -> None:
        """All ground truth items in predictions."""
        ground_truth = {"A", "B", "C"}
        predictions = ["A", "B", "C", "D", "E"]
        assert recall_at_k(ground_truth, predictions, k=5) == 1.0

    def test_partial_recall(self) -> None:
        """Some ground truth items in predictions."""
        ground_truth = {"A", "B", "C"}
        predictions = ["A", "D", "E", "F", "G"]
        assert recall_at_k(ground_truth, predictions, k=5) == pytest.approx(1 / 3)

    def test_zero_recall(self) -> None:
        """No ground truth items in predictions."""
        ground_truth = {"A", "B", "C"}
        predictions = ["D", "E", "F", "G", "H"]
        assert recall_at_k(ground_truth, predictions, k=5) == 0.0

    def test_empty_ground_truth(self) -> None:
        """Empty ground truth returns 0."""
        ground_truth: set[str] = set()
        predictions = ["A", "B", "C"]
        assert recall_at_k(ground_truth, predictions, k=3) == 0.0

    def test_k_limit(self) -> None:
        """Only considers top-k predictions."""
        ground_truth = {"E"}
        predictions = ["A", "B", "C", "D", "E"]
        assert recall_at_k(ground_truth, predictions, k=3) == 0.0
        assert recall_at_k(ground_truth, predictions, k=5) == 1.0

    def test_multiple_ground_truth_partial_match(self) -> None:
        """Multiple ground truth items with partial match."""
        ground_truth = {"A", "B", "C", "D"}
        predictions = ["A", "B", "X", "Y", "Z"]
        assert recall_at_k(ground_truth, predictions, k=5) == pytest.approx(0.5)


class TestMRR:
    """Tests for mean_reciprocal_rank function."""

    def test_first_position(self) -> None:
        """Correct result at position 1."""
        ground_truth = {"A"}
        predictions = ["A", "B", "C"]
        assert mean_reciprocal_rank(ground_truth, predictions) == 1.0

    def test_second_position(self) -> None:
        """Correct result at position 2."""
        ground_truth = {"B"}
        predictions = ["A", "B", "C"]
        assert mean_reciprocal_rank(ground_truth, predictions) == pytest.approx(0.5)

    def test_third_position(self) -> None:
        """Correct result at position 3."""
        ground_truth = {"C"}
        predictions = ["A", "B", "C", "D"]
        assert mean_reciprocal_rank(ground_truth, predictions) == pytest.approx(1 / 3)

    def test_not_found(self) -> None:
        """No correct result in predictions."""
        ground_truth = {"X"}
        predictions = ["A", "B", "C"]
        assert mean_reciprocal_rank(ground_truth, predictions) == 0.0

    def test_multiple_correct(self) -> None:
        """Multiple correct items - returns reciprocal of first."""
        ground_truth = {"B", "D"}
        predictions = ["A", "B", "C", "D"]
        assert mean_reciprocal_rank(ground_truth, predictions) == pytest.approx(1 / 2)

    def test_empty_predictions(self) -> None:
        """Empty predictions returns 0."""
        ground_truth = {"A"}
        predictions: list[str] = []
        assert mean_reciprocal_rank(ground_truth, predictions) == 0.0
