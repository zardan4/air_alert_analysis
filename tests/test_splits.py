"""Unit tests verifying strict anti-data leakage, purging, and walk-forward cross-validation."""

import pytest
import numpy as np
import pandas as pd

from src.data.grid_converter import IntervalToGridConverter
from src.data.mock_source import MockAlertDataSource
from src.splits.temporal_splitter import (
    HoldoutSplitResult,
    PurgedWalkForwardSplitter,
    TemporalLeakageError,
    verify_temporal_integrity,
)
from src.utils.regions import ALL_REGION_SLUGS


@pytest.fixture(scope="module")
def sample_grid_90_days():
    """Generate 90 days of uniform 1h grid for all 25 regions."""
    mock_src = MockAlertDataSource(seed=42)
    raw = mock_src.load_raw_alerts(
        start_date="2024-01-01T00:00:00Z",
        end_date="2024-03-31T23:59:59Z",
        regions=ALL_REGION_SLUGS,
    )
    converter = IntervalToGridConverter(frequency="1h")
    return converter.convert(
        raw,
        start_date="2024-01-01T00:00:00Z",
        end_date="2024-03-31T23:59:59Z",
        regions=ALL_REGION_SLUGS,
    )


def test_holdout_split_strictly_anti_leakage(sample_grid_90_days):
    """Verify holdout split strictly enforces separation buffer and zero leakage."""
    splitter = PurgedWalkForwardSplitter(
        max_horizon_hours=3,
        embargo_hours=3,
        test_window_days=30,
    )
    res = splitter.create_holdout_split(sample_grid_90_days)

    train_val = res.train_val_df
    test = res.test_df

    assert not train_val.empty
    assert not test.empty

    max_train_ts = train_val["timestamp"].max()
    min_test_ts = test["timestamp"].min()

    # Gap must be AT LEAST (3h horizon + 3h embargo) = 6h
    gap_hours = (min_test_ts - max_train_ts).total_seconds() / 3600.0
    assert gap_hours >= 6.0
    assert res.separation_buffer_hours == 6.0

    # Ensure zero timestamp overlap
    train_ts_set = set(train_val["timestamp"])
    test_ts_set = set(test["timestamp"])
    assert len(train_ts_set.intersection(test_ts_set)) == 0

    # Verify purged rows correspond to discrete interior grid points:
    # Gap of 6 hours has (6 - 1) = 5 hourly timestamps, each having 25 regions = 125 rows.
    expected_purged_points = int(gap_hours - 1) * 25
    assert res.purged_rows_count == expected_purged_points
    assert res.purged_rows_count > 0


def test_cv_splits_expanding_window_and_purging(sample_grid_90_days):
    """Verify CV folds implement expanding train windows and strict purging."""
    splitter = PurgedWalkForwardSplitter(
        max_horizon_hours=3,
        embargo_hours=3,
        val_window_days=10,
        min_train_days=30,
        step_days=10,
    )
    splits = splitter.create_cv_splits(sample_grid_90_days)

    assert len(splits) >= 3

    prev_train_len = 0
    for fold in splits:
        train_sub = sample_grid_90_days.iloc[fold.train_idx]
        val_sub = sample_grid_90_days.iloc[fold.val_idx]

        # 1. Zero index overlap
        assert len(set(fold.train_idx).intersection(set(fold.val_idx))) == 0

        # 2. Strict chronological separation: min(val) - max(train) >= 6.0 hours
        gap = (val_sub["timestamp"].min() - train_sub["timestamp"].max()).total_seconds() / 3600.0
        assert gap >= 6.0

        # 3. Expanding window check: each subsequent fold has more training data
        assert len(fold.train_idx) > prev_train_len
        prev_train_len = len(fold.train_idx)

        # 4. Zero timestamp intersection
        assert len(set(train_sub["timestamp"]).intersection(set(val_sub["timestamp"]))) == 0

        # 5. Purged rows existed
        assert fold.purged_rows_count > 0


def test_atomicity_all_regions_together_per_timestamp(sample_grid_90_days):
    """Verify that at any timestamp t, all 25 regions are allocated atomically to the same partition."""
    splitter = PurgedWalkForwardSplitter(
        max_horizon_hours=3,
        embargo_hours=3,
        val_window_days=14,
        min_train_days=30,
    )
    splits = splitter.create_cv_splits(sample_grid_90_days)
    fold0 = splits[0]

    train_df = sample_grid_90_days.iloc[fold0.train_idx]
    val_df = sample_grid_90_days.iloc[fold0.val_idx]

    # Every timestamp in train must have all 25 regions
    train_counts = train_df.groupby("timestamp")["region_slug"].count()
    assert (train_counts == 25).all()

    # Every timestamp in val must have all 25 regions
    val_counts = val_df.groupby("timestamp")["region_slug"].count()
    assert (val_counts == 25).all()


def test_temporal_integrity_catches_future_leakage():
    """Verify verify_temporal_integrity catches timestamp inversion or overlap."""
    t0 = pd.Timestamp("2024-01-01T00:00:00Z")
    t1 = pd.Timestamp("2024-01-01T05:00:00Z")
    t2 = pd.Timestamp("2024-01-01T08:00:00Z")

    # Inverted: eval is before train
    with pytest.raises(TemporalLeakageError, match="Strict temporal ordering violated"):
        verify_temporal_integrity(
            train_timestamps=[t2],
            eval_timestamps=[t0],
            min_separation_hours=3.0,
        )

    # Overlapping: same timestamp in train and eval
    with pytest.raises(TemporalLeakageError, match="Strict temporal ordering violated"):
        verify_temporal_integrity(
            train_timestamps=[t0, t1],
            eval_timestamps=[t1, t2],
            min_separation_hours=3.0,
        )


def test_temporal_integrity_catches_insufficient_buffer():
    """Verify verify_temporal_integrity catches buffer smaller than (H + embargo)."""
    t_train = pd.Timestamp("2024-01-01T10:00:00Z")
    t_eval = pd.Timestamp("2024-01-01T12:00:00Z")  # Only 2 hours later

    # Requires 6 hours buffer
    with pytest.raises(TemporalLeakageError, match="Separation buffer violated"):
        verify_temporal_integrity(
            train_timestamps=[t_train],
            eval_timestamps=[t_eval],
            min_separation_hours=6.0,
        )


def test_sklearn_split_generator_interface(sample_grid_90_days):
    """Verify splitter.split() generator yields valid (train_idx, val_idx) tuples."""
    splitter = PurgedWalkForwardSplitter(
        max_horizon_hours=1,
        embargo_hours=1,
        val_window_days=7,
        min_train_days=30,
        step_days=14,
    )
    generator = splitter.split(sample_grid_90_days)
    folds = list(generator)

    assert len(folds) >= 3
    for train_idx, val_idx in folds:
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(val_idx, np.ndarray)
        assert len(train_idx) > 0
        assert len(val_idx) > 0
        assert len(set(train_idx).intersection(set(val_idx))) == 0
