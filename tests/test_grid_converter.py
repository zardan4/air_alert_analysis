"""Unit tests for IntervalToGridConverter."""

import pytest
import pandas as pd

from src.data.grid_converter import IntervalToGridConverter
from src.utils.regions import ALL_REGION_SLUGS, get_region


def create_sample_raw(records):
    """Helper to construct raw alert DataFrame."""
    df = pd.DataFrame(records)
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True)
    df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True)
    if "alert_type" not in df.columns:
        df["alert_type"] = "air_raid"
    if "source" not in df.columns:
        df["source"] = "test"
    if "region_id" not in df.columns:
        df["region_id"] = [get_region(r).id for r in df["region_slug"]]
    if "alert_id" not in df.columns:
        df["alert_id"] = [f"test_{i}" for i in range(len(df))]
    return df


def test_single_alert_within_slot():
    """Verify single alert contained completely within a 1-hour slot."""
    raw = create_sample_raw([{
        "region_slug": "kyiv_city",
        "started_at": "2024-03-01T10:15:00Z",
        "finished_at": "2024-03-01T10:45:00Z",
    }])

    converter = IntervalToGridConverter(frequency="1h", min_alert_overlap_sec=60)
    grid = converter.convert(
        raw,
        start_date="2024-03-01T09:00:00Z",
        end_date="2024-03-01T12:00:00Z",
        regions=["kyiv_city"],
    )

    assert len(grid) == 3
    # Slot 09:00 - 10:00: inactive
    row_9 = grid[grid["timestamp"] == pd.Timestamp("2024-03-01T09:00:00Z")].iloc[0]
    assert row_9["alert_active"] == 0
    assert row_9["alert_onset"] == 0
    assert row_9["active_duration_sec"] == 0

    # Slot 10:00 - 11:00: active (30 mins = 1800s), onset = 1
    row_10 = grid[grid["timestamp"] == pd.Timestamp("2024-03-01T10:00:00Z")].iloc[0]
    assert row_10["alert_active"] == 1
    assert row_10["alert_onset"] == 1
    assert row_10["active_duration_sec"] == 1800.0
    assert pytest.approx(row_10["active_ratio"], 0.01) == 0.5

    # Slot 11:00 - 12:00: inactive
    row_11 = grid[grid["timestamp"] == pd.Timestamp("2024-03-01T11:00:00Z")].iloc[0]
    assert row_11["alert_active"] == 0
    assert row_11["alert_onset"] == 0


def test_multi_hour_spanning_alert():
    """Verify alert that spans across multiple consecutive hours."""
    raw = create_sample_raw([{
        "region_slug": "kharkiv",
        "started_at": "2024-03-01T10:30:00Z",
        "finished_at": "2024-03-01T12:30:00Z",
    }])

    converter = IntervalToGridConverter(frequency="1h")
    grid = converter.convert(
        raw,
        start_date="2024-03-01T10:00:00Z",
        end_date="2024-03-01T13:00:00Z",
        regions=["kharkiv"],
    )

    assert len(grid) == 3
    # 10:00 - 11:00: 30 mins active, onset=1
    r0 = grid.iloc[0]
    assert r0["alert_active"] == 1
    assert r0["alert_onset"] == 1
    assert r0["active_duration_sec"] == 1800.0

    # 11:00 - 12:00: full hour (3600s), onset=0 (continuation)
    r1 = grid.iloc[1]
    assert r1["alert_active"] == 1
    assert r1["alert_onset"] == 0
    assert r1["active_duration_sec"] == 3600.0
    assert r1["active_ratio"] == 1.0

    # 12:00 - 13:00: 30 mins active, onset=0
    r2 = grid.iloc[2]
    assert r2["alert_active"] == 1
    assert r2["alert_onset"] == 0
    assert r2["active_duration_sec"] == 1800.0


def test_overlapping_intervals_no_double_counting():
    """Verify overlapping alerts in the same region do not double-count active seconds."""
    # Alert A: 10:10 to 10:40 (30 min)
    # Alert B: 10:30 to 10:50 (20 min, overlaps 10:30-10:40)
    # Total merged active: 10:10 to 10:50 = 40 minutes (2400s)
    raw = create_sample_raw([
        {
            "region_slug": "odesa",
            "started_at": "2024-03-01T10:10:00Z",
            "finished_at": "2024-03-01T10:40:00Z",
        },
        {
            "region_slug": "odesa",
            "started_at": "2024-03-01T10:30:00Z",
            "finished_at": "2024-03-01T10:50:00Z",
        },
    ])

    converter = IntervalToGridConverter(frequency="1h")
    grid = converter.convert(
        raw,
        start_date="2024-03-01T10:00:00Z",
        end_date="2024-03-01T11:00:00Z",
        regions=["odesa"],
    )

    assert len(grid) == 1
    row = grid.iloc[0]
    assert row["active_duration_sec"] == 2400.0  # Exactly 40 mins
    assert row["alert_active"] == 1
    assert row["alert_onset"] == 1


def test_min_overlap_sec_threshold():
    """Verify alert under min_overlap_sec does not trigger alert_active."""
    # 45-second blip
    raw = create_sample_raw([{
        "region_slug": "lviv",
        "started_at": "2024-03-01T10:00:00Z",
        "finished_at": "2024-03-01T10:00:45Z",
    }])

    converter = IntervalToGridConverter(frequency="1h", min_alert_overlap_sec=60)
    grid = converter.convert(
        raw,
        start_date="2024-03-01T10:00:00Z",
        end_date="2024-03-01T11:00:00Z",
        regions=["lviv"],
    )

    assert len(grid) == 1
    row = grid.iloc[0]
    assert row["active_duration_sec"] == 45.0
    assert row["alert_active"] == 0  # Under 60 sec
    assert row["alert_onset"] == 1  # Still ignited here


def test_spatio_temporal_completeness_all_25_regions():
    """Verify grid produces entries for all 25 regions across every timestamp even with zero alerts."""
    # Data only in 1 region
    raw = create_sample_raw([{
        "region_slug": "kharkiv",
        "started_at": "2024-03-01T10:00:00Z",
        "finished_at": "2024-03-01T11:00:00Z",
    }])

    converter = IntervalToGridConverter(frequency="1h")
    grid = converter.convert(
        raw,
        start_date="2024-03-01T10:00:00Z",
        end_date="2024-03-01T14:00:00Z",
        regions=ALL_REGION_SLUGS,
    )

    expected_rows = 4 * 25  # 4 hours * 25 regions = 100 rows
    assert len(grid) == expected_rows
    assert set(grid["region_slug"].unique()) == set(ALL_REGION_SLUGS)

    # Kharkiv has 1 active slot, other 24 regions have 0 active slots
    kharkiv_rows = grid[grid["region_slug"] == "kharkiv"]
    assert kharkiv_rows["alert_active"].sum() == 1

    kyiv_rows = grid[grid["region_slug"] == "kyiv_city"]
    assert len(kyiv_rows) == 4
    assert kyiv_rows["alert_active"].sum() == 0


def test_configurable_frequency_15min():
    """Verify grid converter handles arbitrary sub-hourly frequencies (e.g. 15min)."""
    raw = create_sample_raw([{
        "region_slug": "kyiv_city",
        "started_at": "2024-03-01T10:00:00Z",
        "finished_at": "2024-03-01T11:00:00Z",
    }])

    converter = IntervalToGridConverter(frequency="15min")
    grid = converter.convert(
        raw,
        start_date="2024-03-01T10:00:00Z",
        end_date="2024-03-01T11:00:00Z",
        regions=["kyiv_city"],
    )

    # 1 hour = 4 slots of 15 min
    assert len(grid) == 4
    assert (grid["alert_active"] == 1).all()
    assert (grid["active_duration_sec"] == 900.0).all()  # 15 mins = 900s
    assert list(grid["alert_onset"]) == [1, 0, 0, 0]  # Only first slot has onset
