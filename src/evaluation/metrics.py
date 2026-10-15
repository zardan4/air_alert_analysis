"""Evaluation metrics suite for probabilistic binary classification of alert events."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class MetricsResult:
    """Container for a complete evaluation of a single model on a single target."""
    model_name: str
    target_name: str
    horizon_hours: Optional[int]
    region_slug: Optional[str]  # None = national aggregate

    roc_auc: float
    pr_auc: float
    brier_score: float
    log_loss: float
    precision: float
    recall: float
    f1_score: float
    threshold: float       # Decision threshold used for precision/recall/f1

    n_samples: int
    n_positive: int
    prevalence: float

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


def _safe_log(p: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    return np.log(np.clip(p, eps, 1.0 - eps))


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC without sklearn dependency (pure numpy/manual trapezoidal)."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)

    if len(np.unique(y_true)) < 2:
        return float("nan")

    desc_indices = np.argsort(-y_score)
    y_sorted = y_true[desc_indices]
    scores_sorted = y_score[desc_indices]

    # Deduplicate thresholds
    distinct_idx = np.where(np.diff(scores_sorted))[0]
    threshold_indices = np.concatenate([distinct_idx, [len(y_sorted) - 1]])

    tps = np.cumsum(y_sorted)[threshold_indices]
    fps = (threshold_indices + 1) - tps

    total_pos = y_true.sum()
    total_neg = len(y_true) - total_pos

    if total_pos == 0 or total_neg == 0:
        return float("nan")

    tpr = np.concatenate([[0], tps / total_pos])
    fpr = np.concatenate([[0], fps / total_neg])

    trapezoid_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trapezoid_fn(tpr, fpr))


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Precision-Recall AUC (Average Precision) without sklearn."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)

    if len(np.unique(y_true)) < 2:
        return float("nan")

    desc_indices = np.argsort(-y_score)
    y_sorted = y_true[desc_indices]

    tps = np.cumsum(y_sorted)
    total_pos = y_true.sum()

    if total_pos == 0:
        return float("nan")

    precision_arr = tps / np.arange(1, len(y_sorted) + 1)
    recall_arr = tps / total_pos

    # Average precision: sum of precision at each positive prediction weighted by recall delta
    ap = 0.0
    prev_recall = 0.0
    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            ap += precision_arr[i] * (recall_arr[i] - prev_recall)
            prev_recall = recall_arr[i]

    return float(ap)


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score: mean squared error between predicted probability and observed outcome."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    return float(np.mean((y_prob - y_true) ** 2))


def compute_log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    """Binary log-loss (cross-entropy)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


def compute_precision_recall_f1(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1 from binary predictions."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def find_optimal_f1_threshold(
    y_true: np.ndarray, y_score: np.ndarray, n_thresholds: int = 200
) -> Tuple[float, float, float, float]:
    """Search for the threshold that maximizes F1 score.

    Returns: (best_threshold, precision_at_best, recall_at_best, best_f1)
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_f1 = 0.0
    best_threshold = 0.5
    best_precision = 0.0
    best_recall = 0.0

    for t in thresholds:
        y_pred = (y_score >= t).astype(np.int32)
        p, r, f = compute_precision_recall_f1(y_true, y_pred)
        if f > best_f1:
            best_f1 = f
            best_threshold = t
            best_precision = p
            best_recall = r

    return best_threshold, best_precision, best_recall, best_f1


def evaluate_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    model_name: str,
    target_name: str,
    horizon_hours: Optional[int] = None,
    region_slug: Optional[str] = None,
    threshold: Optional[float] = None,
) -> MetricsResult:
    """Full evaluation pipeline returning all metrics for a single model+target+region."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)

    roc_auc = compute_roc_auc(y_true, y_score)
    pr_auc = compute_pr_auc(y_true, y_score)
    brier = compute_brier_score(y_true, y_score)
    logloss = compute_log_loss(y_true, y_score)

    if threshold is None:
        threshold, precision, recall, f1 = find_optimal_f1_threshold(y_true, y_score)
    else:
        y_pred = (y_score >= threshold).astype(np.int32)
        precision, recall, f1 = compute_precision_recall_f1(y_true, y_pred)

    n_samples = len(y_true)
    n_positive = int(y_true.sum())
    prevalence = n_positive / n_samples if n_samples > 0 else 0.0

    return MetricsResult(
        model_name=model_name,
        target_name=target_name,
        horizon_hours=horizon_hours,
        region_slug=region_slug,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        brier_score=brier,
        log_loss=logloss,
        precision=precision,
        recall=recall,
        f1_score=f1,
        threshold=threshold,
        n_samples=n_samples,
        n_positive=n_positive,
        prevalence=prevalence,
    )


def format_metrics_table(results: List[MetricsResult]) -> pd.DataFrame:
    """Convert a list of MetricsResult into a formatted comparison DataFrame."""
    rows = [r.to_dict() for r in results]
    df = pd.DataFrame(rows)
    float_cols = ["roc_auc", "pr_auc", "brier_score", "log_loss", "precision", "recall", "f1_score", "threshold", "prevalence"]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].round(4)
    return df
