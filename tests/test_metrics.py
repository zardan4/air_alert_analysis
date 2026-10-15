"""Unit tests for evaluation metrics engine."""

import pytest
import numpy as np

from src.evaluation.metrics import (
    compute_brier_score,
    compute_log_loss,
    compute_pr_auc,
    compute_precision_recall_f1,
    compute_roc_auc,
    evaluate_predictions,
    find_optimal_f1_threshold,
)


def test_perfect_roc_auc():
    """Perfect separation should yield ROC-AUC = 1.0."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert compute_roc_auc(y_true, y_score) == pytest.approx(1.0, abs=0.01)


def test_random_roc_auc():
    """Identical scores for all samples should yield ROC-AUC ≈ 0.5."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=1000)
    y_score = rng.random(size=1000)
    auc = compute_roc_auc(y_true, y_score)
    assert 0.4 < auc < 0.6


def test_perfect_pr_auc():
    """Perfect ranking should yield PR-AUC close to 1.0."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert compute_pr_auc(y_true, y_score) == pytest.approx(1.0, abs=0.01)


def test_brier_score_perfect():
    """Perfect predictions should give Brier score = 0."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])
    assert compute_brier_score(y_true, y_prob) == pytest.approx(0.0)


def test_brier_score_worst():
    """Completely inverted predictions should give Brier score = 1."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([1.0, 1.0, 0.0, 0.0])
    assert compute_brier_score(y_true, y_prob) == pytest.approx(1.0)


def test_log_loss_perfect():
    """Near-perfect predictions should have log-loss close to 0."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.001, 0.001, 0.999, 0.999])
    assert compute_log_loss(y_true, y_prob) < 0.01


def test_precision_recall_f1_calculation():
    """Verify precision, recall, F1 arithmetic."""
    y_true = np.array([1, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1])
    # TP=2, FP=1, FN=1
    p, r, f = compute_precision_recall_f1(y_true, y_pred)
    assert p == pytest.approx(2 / 3, abs=0.01)
    assert r == pytest.approx(2 / 3, abs=0.01)
    assert f == pytest.approx(2 / 3, abs=0.01)


def test_optimal_f1_threshold():
    """Verify threshold search finds a reasonable operating point."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    threshold, p, r, f1 = find_optimal_f1_threshold(y_true, y_score)
    assert 0.3 < threshold < 0.7
    assert f1 > 0.8


def test_evaluate_predictions_full_pipeline():
    """Verify full evaluation pipeline produces valid MetricsResult."""
    rng = np.random.default_rng(123)
    y_true = rng.integers(0, 2, size=500)
    y_score = rng.random(size=500)

    result = evaluate_predictions(
        y_true=y_true,
        y_score=y_score,
        model_name="test_model",
        target_name="target_active_1h",
        horizon_hours=1,
    )
    assert result.model_name == "test_model"
    assert result.n_samples == 500
    assert 0 <= result.roc_auc <= 1
    assert 0 <= result.pr_auc <= 1
    assert 0 <= result.brier_score <= 1
    assert result.log_loss > 0
    assert 0 <= result.f1_score <= 1
    assert 0 < result.threshold < 1


def test_degenerate_all_positive():
    """ROC-AUC should return NaN when all labels are positive."""
    y_true = np.array([1, 1, 1, 1])
    y_score = np.array([0.3, 0.5, 0.7, 0.9])
    assert np.isnan(compute_roc_auc(y_true, y_score))
