"""Unit tests for causal point-in-time feature engineering and multi-horizon targets."""

import pytest
import numpy as np
import pandas as pd

from src.features.feature_engineer import PointInTimeFeatureEngineer
from src.features.spatial import get_neighbors
from src.utils.regions import ALL_REGION_SLUGS


def create_minimal_grid(timestamps, regions, active_map=None, onset_map=None):
    """Helper to construct minimal grid for feature testing."""
    records = []
    active_map = active_map or {}
    onset_map = onset_map or {}

    for t in timestamps:
        for r in regions:
            records.append({
                "timestamp": pd.Timestamp(t, tz="UTC"),
                "region_slug": r,
                "region_id": 1,
                "alert_active": active_map.get((t, r), 0),
                "alert_onset": onset_map.get((t, r), 0),
                "active_duration_sec": 3600.0 if active_map.get((t, r), 0) else 0.0,
                "active_ratio": 1.0 if active_map.get((t, r), 0) else 0.0,
                "alerts_count": active_map.get((t, r), 0),
            })
    return pd.DataFrame(records)


def test_causal_invariance_no_future_leakage():
    """Mathematically verify that changing future values (t+1..t+k) does NOT affect features at t."""
    timestamps = [
        f"2024-01-01T{h:02d}:00:00Z" for h in range(24)
    ]
    regions = ["kyiv_city", "kharkiv", "lviv"]

    # Base dataset
    base_grid = create_minimal_grid(
        timestamps, regions,
        active_map={("2024-01-01T05:00:00Z", "kyiv_city"): 1},
        onset_map={("2024-01-01T05:00:00Z", "kyiv_city"): 1},
    )

    engineer = PointInTimeFeatureEngineer(
        horizons_hours=[1, 2, 3],
        lags_hours=[0, 1, 2],
        rolling_windows_hours=[6, 12],
    )
    res_base = engineer.transform(base_grid, include_targets=False)

    # Future altered dataset: dramatically alter everything after t=10
    altered_grid = base_grid.copy()
    future_mask = altered_grid["timestamp"] >= pd.Timestamp("2024-01-01T11:00:00Z", tz="UTC")
    altered_grid.loc[future_mask, "alert_active"] = 1
    altered_grid.loc[future_mask, "alert_onset"] = 1
    altered_grid.loc[future_mask, "active_duration_sec"] = 3600.0
    altered_grid.loc[future_mask, "active_ratio"] = 1.0

    res_altered = engineer.transform(altered_grid, include_targets=False)

    # Check cutoff t = 10:00:00
    cutoff_ts = pd.Timestamp("2024-01-01T10:00:00Z", tz="UTC")

    feat_cols = res_base.feature_columns
    base_t10 = res_base.data[res_base.data["timestamp"] <= cutoff_ts].sort_values(by=["timestamp", "region_slug"])
    altered_t10 = res_altered.data[res_altered.data["timestamp"] <= cutoff_ts].sort_values(by=["timestamp", "region_slug"])

    # All features at or before cutoff MUST BE 100% BIT-FOR-BIT IDENTICAL
    for col in feat_cols:
        np.testing.assert_array_equal(
            base_t10[col].values,
            altered_t10[col].values,
            err_msg=f"Causal leak detected in feature '{col}' at or before cutoff timestamp!",
        )


def test_multi_horizon_target_construction():
    """Verify forward targets accurately capture events at t+1, t+2, t+3."""
    timestamps = [
        "2024-01-01T10:00:00Z",
        "2024-01-01T11:00:00Z",
        "2024-01-01T12:00:00Z",
        "2024-01-01T13:00:00Z",
        "2024-01-01T14:00:00Z",
    ]
    # Alert occurs only at 12:00 (which is t+2 from 10:00)
    grid = create_minimal_grid(
        timestamps, ["kyiv_city"],
        active_map={("2024-01-01T12:00:00Z", "kyiv_city"): 1},
        onset_map={("2024-01-01T12:00:00Z", "kyiv_city"): 1},
    )

    engineer = PointInTimeFeatureEngineer(horizons_hours=[1, 2, 3])
    res = engineer.transform(grid, include_targets=True, drop_na_targets=False)
    data = res.data

    row_10 = data[data["timestamp"] == pd.Timestamp("2024-01-01T10:00:00Z", tz="UTC")].iloc[0]
    # At 10:00:
    # 1h ahead (11:00): quiet -> 0
    # 2h ahead (11:00, 12:00): active at 12:00 -> 1
    # 3h ahead (11:00, 12:00, 13:00): active at 12:00 -> 1
    assert row_10["target_active_1h"] == 0
    assert row_10["target_active_2h"] == 1
    assert row_10["target_active_3h"] == 1
    assert row_10["target_onset_1h"] == 0
    assert row_10["target_onset_2h"] == 1
    assert row_10["target_onset_3h"] == 1


def test_spatial_neighborhood_features():
    """Verify neighbor active count reflects immediate geographic adjacency."""
    timestamps = ["2024-01-01T10:00:00Z"]
    # Kyiv Oblast in alert, Kyiv City quiet
    grid = create_minimal_grid(
        timestamps, ["kyiv_city", "kyiv_oblast", "lviv"],
        active_map={("2024-01-01T10:00:00Z", "kyiv_oblast"): 1},
    )

    engineer = PointInTimeFeatureEngineer()
    res = engineer.transform(grid, include_targets=False)
    df = res.data

    kyiv_city_row = df[df["region_slug"] == "kyiv_city"].iloc[0]
    lviv_row = df[df["region_slug"] == "lviv"].iloc[0]

    # Kyiv City neighbor is Kyiv Oblast (which is active) -> count = 1
    assert kyiv_city_row["feat_neighbors_active_count"] == 1.0
    assert kyiv_city_row["feat_neighbors_active_ratio"] == 1.0

    # Lviv has no active neighbors in this minimal slice -> count = 0
    assert lviv_row["feat_neighbors_active_count"] == 0.0


def test_diurnal_and_calendar_features():
    """Verify sin/cos bounds and nighttime window detection."""
    timestamps = [
        "2024-01-01T02:00:00Z",  # Local Kyiv ~04:00 (Night window!)
        "2024-01-01T12:00:00Z",  # Local Kyiv ~14:00 (Daytime)
    ]
    grid = create_minimal_grid(timestamps, ["kharkiv"])

    engineer = PointInTimeFeatureEngineer()
    res = engineer.transform(grid, include_targets=False)
    df = res.data

    night_row = df[df["timestamp"] == pd.Timestamp("2024-01-01T02:00:00Z", tz="UTC")].iloc[0]
    day_row = df[df["timestamp"] == pd.Timestamp("2024-01-01T12:00:00Z", tz="UTC")].iloc[0]

    assert night_row["feat_is_night_strike_window"] == 1
    assert day_row["feat_is_night_strike_window"] == 0

    assert -1.0 <= night_row["feat_hour_sin"] <= 1.0
    assert -1.0 <= night_row["feat_hour_cos"] <= 1.0
