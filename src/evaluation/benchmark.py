"""Benchmark runner — fits all baselines on train, evaluates on test, produces comparison table."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.baselines.naive_baselines import BaselineModel, get_all_baselines
from src.evaluation.metrics import MetricsResult, evaluate_predictions, format_metrics_table
from src.utils.regions import ALL_REGION_SLUGS, filter_by_region


@dataclass
class BenchmarkReport:
    """Complete benchmark results across all baselines, horizons, and regions."""
    national_results: pd.DataFrame
    per_oblast_results: Optional[pd.DataFrame]
    raw_metrics: List[MetricsResult]


class BenchmarkRunner:
    """Orchestrates end-to-end baseline benchmarking across horizons and regions."""

    def __init__(
        self,
        horizons_hours: Optional[List[int]] = None,
        evaluate_per_oblast: bool = False,
        target_type: str = "active",  # "active" or "onset"
    ):
        self.horizons_hours = horizons_hours or [1, 2, 3]
        self.evaluate_per_oblast = evaluate_per_oblast
        self.target_type = target_type

    def run(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        baselines: Optional[List[BaselineModel]] = None,
    ) -> BenchmarkReport:
        """Execute full benchmark pipeline."""
        if baselines is None:
            baselines = get_all_baselines()

        all_metrics: List[MetricsResult] = []

        # Fit all baselines on training data
        for baseline in baselines:
            baseline.fit(train_df)

        # National aggregate evaluation per horizon
        for horizon in self.horizons_hours:
            target_col = f"target_{self.target_type}_{horizon}h"
            if target_col not in test_df.columns:
                continue

            valid_mask = test_df[target_col].notna()
            eval_df = test_df[valid_mask]
            if len(eval_df) == 0:
                continue

            y_true = eval_df[target_col].values.astype(np.int32)

            for baseline in baselines:
                y_score = baseline.predict_proba(eval_df, horizon_hours=horizon)
                y_score = np.clip(y_score, 0.0, 1.0)

                result = evaluate_predictions(
                    y_true=y_true,
                    y_score=y_score,
                    model_name=baseline.name,
                    target_name=target_col,
                    horizon_hours=horizon,
                    region_slug=None,  # National
                )
                all_metrics.append(result)

        # Per-oblast evaluation
        per_oblast_results = None
        if self.evaluate_per_oblast:
            oblast_metrics: List[MetricsResult] = []
            for region_slug in ALL_REGION_SLUGS:
                try:
                    region_test = filter_by_region(test_df, region_slug)
                except KeyError:
                    continue
                if len(region_test) < 10:
                    continue

                for horizon in self.horizons_hours:
                    target_col = f"target_{self.target_type}_{horizon}h"
                    if target_col not in region_test.columns:
                        continue
                    valid_mask = region_test[target_col].notna()
                    eval_df = region_test[valid_mask]
                    if len(eval_df) == 0:
                        continue

                    y_true = eval_df[target_col].values.astype(np.int32)

                    for baseline in baselines:
                        y_score = baseline.predict_proba(eval_df, horizon_hours=horizon)
                        y_score = np.clip(y_score, 0.0, 1.0)

                        result = evaluate_predictions(
                            y_true=y_true,
                            y_score=y_score,
                            model_name=baseline.name,
                            target_name=target_col,
                            horizon_hours=horizon,
                            region_slug=region_slug,
                        )
                        oblast_metrics.append(result)

            if oblast_metrics:
                per_oblast_results = format_metrics_table(oblast_metrics)

        national_results = format_metrics_table(all_metrics)

        return BenchmarkReport(
            national_results=national_results,
            per_oblast_results=per_oblast_results,
            raw_metrics=all_metrics,
        )


def print_benchmark_report(report: BenchmarkReport) -> None:
    """Pretty-print benchmark results to stdout."""
    print("\n" + "=" * 90)
    print("  BASELINE BENCHMARK REPORT — NATIONAL AGGREGATE")
    print("=" * 90)

    if report.national_results.empty:
        print("  No results available.")
        return

    display_cols = [
        "model_name", "horizon_hours", "roc_auc", "pr_auc",
        "brier_score", "log_loss", "f1_score", "precision",
        "recall", "threshold", "prevalence"
    ]
    available = [c for c in display_cols if c in report.national_results.columns]
    print(report.national_results[available].to_string(index=False))

    if report.per_oblast_results is not None and not report.per_oblast_results.empty:
        print("\n" + "-" * 90)
        print("  PER-OBLAST TOP-LEVEL SUMMARY (F1 by Region for 1h Horizon)")
        print("-" * 90)

        subset = report.per_oblast_results[report.per_oblast_results["horizon_hours"] == 1]
        if not subset.empty:
            pivot = subset.pivot_table(
                index="region_slug",
                columns="model_name",
                values="f1_score",
                aggfunc="first",
            )
            print(pivot.round(3).to_string())

    print("=" * 90)
