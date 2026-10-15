"""Unit tests for naive baseline models and benchmark runner."""

import pytest
import numpy as np
import pandas as pd

from src.baselines.naive_baselines import (
    ClimatologyBaseline,
    PersistenceBaseline,
    RollingRateBaseline,
    SpatialInflowBaseline,
    get_all_baselines,
)
from src.data.grid_converter import IntervalToGridConverter
from src.data.mock_source import MockAlertDataSource
from src.evaluation.benchmark import BenchmarkRunner, print_benchmark_report
from src.features.feature_engineer import PointInTimeFeatureEngineer
from src.splits.temporal_splitter import PurgedWalkForwardSplitter
from src.utils.regions import ALL_REGION_SLUGS


@pytest.fixture(scope="module")
def prepared_data():
    """Generate mock data → grid → features → train/test split for benchmark testing."""
    mock_src = MockAlertDataSource(seed=42)
    raw = mock_src.load_raw_alerts(
        start_date="2024-01-01T00:00:00Z",
        end_date="2024-03-31T23:59:59Z",
        regions=ALL_REGION_SLUGS,
    )
    converter = IntervalToGridConverter(frequency="1h")
    grid = converter.convert(
        raw,
        start_date="2024-01-01T00:00:00Z",
        end_date="2024-03-31T23:59:59Z",
        regions=ALL_REGION_SLUGS,
    )

    engineer = PointInTimeFeatureEngineer()
    result = engineer.transform(grid, include_targets=True, drop_na_targets=False)
    feat_df = result.data

    splitter = PurgedWalkForwardSplitter(
        max_horizon_hours=3,
        embargo_hours=3,
        test_window_days=30,
    )
    holdout = splitter.create_holdout_split(feat_df)
    return holdout.train_val_df, holdout.test_df


def test_persistence_baseline_output_range(prepared_data):
    """Persistence baseline should output only 0.0 or 1.0 (binary state)."""
    train_df, test_df = prepared_data
    model = PersistenceBaseline()
    model.fit(train_df)
    y_score = model.predict_proba(test_df, horizon_hours=1)

    assert len(y_score) == len(test_df)
    unique_vals = set(np.unique(y_score))
    assert unique_vals.issubset({0.0, 1.0})


def test_climatology_baseline_probability_range(prepared_data):
    """Climatology baseline should output probabilities in [0, 1]."""
    train_df, test_df = prepared_data
    model = ClimatologyBaseline()
    model.fit(train_df)
    y_score = model.predict_proba(test_df, horizon_hours=2)

    assert len(y_score) == len(test_df)
    assert np.all(y_score >= 0.0)
    assert np.all(y_score <= 1.0)
    # Should not be all zeros or all ones — climatology captures variance
    assert 0.0 < y_score.mean() < 1.0


def test_rolling_rate_baseline_uses_features(prepared_data):
    """Rolling rate baseline should produce valid probabilities from rolling features."""
    train_df, test_df = prepared_data
    model = RollingRateBaseline()
    model.fit(train_df)
    y_score = model.predict_proba(test_df, horizon_hours=1)

    assert len(y_score) == len(test_df)
    assert np.all(y_score >= 0.0)
    assert np.all(y_score <= 1.0)


def test_spatial_inflow_baseline_output_range(prepared_data):
    """Spatial inflow baseline should output probabilities in [0, 1]."""
    train_df, test_df = prepared_data
    model = SpatialInflowBaseline()
    model.fit(train_df)
    y_score = model.predict_proba(test_df, horizon_hours=1)

    assert len(y_score) == len(test_df)
    assert np.all(y_score >= 0.0)
    assert np.all(y_score <= 1.0)


def test_full_benchmark_runner_national(prepared_data):
    """Run full national benchmark pipeline across all 4 baselines and 3 horizons."""
    train_df, test_df = prepared_data

    runner = BenchmarkRunner(
        horizons_hours=[1, 2, 3],
        evaluate_per_oblast=False,
        target_type="active",
    )
    report = runner.run(train_df, test_df)

    # 4 baselines × 3 horizons = 12 national evaluation rows
    assert len(report.national_results) == 12

    # All metrics should be finite real numbers
    for col in ["roc_auc", "pr_auc", "brier_score", "log_loss", "f1_score"]:
        values = report.national_results[col].values
        assert np.all(np.isfinite(values)), f"Non-finite values in {col}"

    # Print for human review (visible in pytest -v -s)
    print_benchmark_report(report)


def test_full_benchmark_runner_per_oblast(prepared_data):
    """Run per-oblast benchmark to verify region-level isolation in evaluation."""
    train_df, test_df = prepared_data

    runner = BenchmarkRunner(
        horizons_hours=[1],
        evaluate_per_oblast=True,
        target_type="active",
    )
    report = runner.run(train_df, test_df)

    assert report.per_oblast_results is not None
    assert not report.per_oblast_results.empty

    # Should have up to 25 regions × 4 baselines × 1 horizon
    assert len(report.per_oblast_results) <= 25 * 4

    # Every region_slug should be a valid registered region
    for slug in report.per_oblast_results["region_slug"].unique():
        assert slug in ALL_REGION_SLUGS


def test_get_all_baselines_factory():
    """Verify factory function returns all 4 baseline types."""
    baselines = get_all_baselines()
    assert len(baselines) == 4
    names = {b.name for b in baselines}
    assert names == {"Persistence", "Climatology", "RollingRate7d", "SpatialInflow"}
