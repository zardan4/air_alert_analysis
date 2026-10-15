"""Unit tests for user-friendly civilian statistics and heatmap aggregations."""

import pytest
import pandas as pd
import numpy as np

from src.data.grid_converter import IntervalToGridConverter
from src.data.mock_source import MockAlertDataSource
from src.ui.stats import (
    DAY_ABBR,
    DAY_NAMES,
    get_calm_and_downtime_stats,
    get_day_of_week_profile,
    get_disruption_scores,
    get_duration_spectrum,
    get_heatmap_matrix,
    get_hromada_breakdown,
    get_peak_and_safe_windows,
    get_threat_level_stats,
    get_safe_window_recommendation,
    get_duration_cdf_percentiles,
    get_annual_evolution_stats,
)
from src.utils.regions import ALL_REGION_SLUGS


@pytest.fixture(scope="module")
def sample_data_ui():
    """Generate 30 days of raw and continuous grid data for testing (using modern Sept 2026 period)."""
    mock_src = MockAlertDataSource(seed=42)
    raw = mock_src.load_raw_alerts(
        start_date="2026-09-01T00:00:00Z",
        end_date="2026-09-30T23:59:59Z",
        regions=["kyiv_city", "kharkiv", "lviv"],
    )
    converter = IntervalToGridConverter(frequency="1h")
    grid = converter.convert(
        raw,
        start_date="2026-09-01T00:00:00Z",
        end_date="2026-09-30T23:59:59Z",
        regions=["kyiv_city", "kharkiv", "lviv"],
    )
    return raw, grid


def test_get_heatmap_matrix_dimensions(sample_data_ui):
    """Verify heatmap matrix is exactly 7 days x 24 hours."""
    raw, grid = sample_data_ui
    heatmap = get_heatmap_matrix(grid, region_slug="kyiv_city")
    assert heatmap.shape == (7, 24)
    assert list(heatmap.index) == DAY_ABBR
    assert list(heatmap.columns) == [f"{h:02d}:00" for h in range(24)]
    assert (heatmap.values >= 0.0).all()
    assert (heatmap.values <= 1.0).all()


def test_get_peak_and_safe_windows(sample_data_ui):
    """Verify peak and safe windows detection."""
    raw, grid = sample_data_ui
    heatmap = get_heatmap_matrix(grid, region_slug="kharkiv")
    windows = get_peak_and_safe_windows(heatmap)
    assert "peak_hour" in windows
    assert "safe_hour" in windows
    assert "peak_day" in windows
    assert windows["peak_hourly_rate"] >= windows["safe_hourly_rate"]


def test_get_day_of_week_profile(sample_data_ui):
    """Verify weekly attack profile returns metrics for all 7 days."""
    raw, grid = sample_data_ui
    profile = get_day_of_week_profile(raw, grid, region_slug="kyiv_city")
    assert len(profile) == 7
    assert list(profile.index) == DAY_NAMES
    for col in ["Average Alerts / Day", "Average Duration (mins)", "Median Duration (mins)", "Active Hours / Day"]:
        assert col in profile.columns
        assert (profile[col].values >= 0).all()


def test_get_duration_spectrum(sample_data_ui):
    """Verify duration spectrum percentages sum to ~1.0."""
    raw, grid = sample_data_ui
    spectrum = get_duration_spectrum(raw, region_slug="kyiv_city")
    assert spectrum["total_alerts"] > 0
    pct_sum = spectrum["pct_short"] + spectrum["pct_moderate"] + spectrum["pct_extended"]
    assert pct_sum == pytest.approx(1.0, abs=0.01)
    assert spectrum["median_min"] > 0
    assert spectrum["mean_min"] > 0


def test_get_calm_and_downtime_stats(sample_data_ui):
    """Verify calm window and daily shelter downtime calculations."""
    raw, grid = sample_data_ui
    calm_stats = get_calm_and_downtime_stats(raw, grid, region_slug="kyiv_city")
    assert calm_stats["avg_calm_hours"] > 0
    assert calm_stats["current_calm_hours"] >= 0
    assert calm_stats["daily_shelter_hours"] > 0


def test_get_disruption_scores(sample_data_ui):
    """Verify sleep and workday disruption percentages."""
    raw, grid = sample_data_ui
    scores = get_disruption_scores(grid, region_slug="kyiv_city")
    assert 0.0 <= scores["sleep_disruption_pct"] <= 1.0
    assert 0.0 <= scores["workday_disruption_pct"] <= 1.0


def test_get_threat_level_stats(sample_data_ui):
    """Verify threat level separation (Red vs Yellow vs Unspecified)."""
    raw, _ = sample_data_ui
    stats = get_threat_level_stats(raw, region_slug="kharkiv")

    assert stats["total_alerts"] > 0
    assert stats["has_threat_data"] is True
    # Red, Yellow, and Unspecified must sum to total
    assert stats["red_count"] + stats["yellow_count"] + stats["unspecified_count"] == stats["total_alerts"]
    pct_sum = stats["red_pct"] + stats["yellow_pct"] + stats["unspecified_pct"]
    assert pct_sum == pytest.approx(1.0, abs=0.01)

    assert stats["red_avg_min"] > 0
    assert stats["red_total_hours"] > 0
    assert stats["yellow_avg_min"] > 0


def test_get_threat_level_stats_backward_compatibility():
    """Verify threat level calculation when data completely lacks threat_level column."""
    legacy_df = pd.DataFrame([{
        "alert_id": "1",
        "region_slug": "lviv",
        "region_id": 14,
        "started_at": "2022-03-01T10:00:00Z",
        "finished_at": "2022-03-01T11:00:00Z",
        "alert_type": "air_raid",
        "source": "legacy",
    }])

    stats = get_threat_level_stats(legacy_df, region_slug="lviv")
    assert stats["has_threat_data"] is False
    assert stats["total_alerts"] == 1
    assert stats["red_count"] == 0
    assert stats["yellow_count"] == 0
    assert stats["unspecified_count"] == 1
    assert stats["unspecified_pct"] == 1.0


def test_get_hromada_breakdown(sample_data_ui):
    """Verify hromada drilldown produces granular breakdown for registered communities."""
    raw, _ = sample_data_ui
    breakdown = get_hromada_breakdown(raw, region_slug="kharkiv")

    assert breakdown["has_differentiation"] is True
    assert breakdown["total_alerts"] > 0
    assert breakdown["hromadas_count"] > 0
    assert breakdown["total_localized"] + breakdown["total_oblast_wide"] == breakdown["total_alerts"]

    summary_df = breakdown["summary_df"]
    assert not summary_df.empty
    assert "Hromada (EN)" in summary_df.columns
    assert "Громада (UK)" in summary_df.columns
    assert "Total Alerts" in summary_df.columns
    assert "Localized Alerts" in summary_df.columns
    assert "Oblast-Wide Alerts" in summary_df.columns
    assert (summary_df["Total Alerts"] >= summary_df["Localized Alerts"]).all()


def test_get_hromada_breakdown_historical_compatibility():
    """Verify backward compatibility when all alerts in interval are oblast-wide (no hromadas)."""
    historical_raw = pd.DataFrame([{
        "alert_id": "h1",
        "region_slug": "odesa",
        "region_id": 16,
        "started_at": "2022-05-01T10:00:00Z",
        "finished_at": "2022-05-01T11:00:00Z",
        "alert_type": "air_raid",
        "source": "archive",
        # Notice: no hromada column or threat_level column!
    }])

    breakdown = get_hromada_breakdown(historical_raw, region_slug="odesa")
    assert breakdown["has_differentiation"] is False
    assert breakdown["total_alerts"] == 1
    assert breakdown["total_localized"] == 0
    assert breakdown["total_oblast_wide"] == 1
    assert not breakdown["summary_df"].empty


def test_get_safe_window_recommendation(sample_data_ui):
    """Verify safe window recommendation calculation and scoring."""
    _, grid = sample_data_ui
    rec = get_safe_window_recommendation(grid, day_of_week=1, start_hour=14, duration_hours=2, region_slug="kyiv_city")
    assert 0 <= rec["safety_score"] <= 100
    assert 0.0 <= rec["danger_prob"] <= 1.0
    assert rec["verdict"] in ["SAFE", "MODERATE", "DANGER"]
    assert "safest_alternative_hour" in rec
    assert "verdict_label" in rec
    assert "recommendation" in rec


def test_get_duration_cdf_percentiles(sample_data_ui):
    """Verify empirical CDF duration percentiles and clearance probability milestones."""
    raw, _ = sample_data_ui
    cdf = get_duration_cdf_percentiles(raw, region_slug="kyiv_city")
    assert cdf["p25_min"] <= cdf["p50_min"] <= cdf["p75_min"] <= cdf["p90_min"]
    assert 0.0 <= cdf["prob_clear_30m"] <= 1.0
    assert 0.0 <= cdf["prob_clear_60m"] <= 1.0
    assert 0.0 <= cdf["prob_clear_120m"] <= 1.0
    assert cdf["prob_clear_30m"] <= cdf["prob_clear_60m"] <= cdf["prob_clear_120m"]


def test_get_annual_evolution_stats(sample_data_ui):
    """Verify annual tactical evolution aggregation across years."""
    raw, _ = sample_data_ui
    evo_df = get_annual_evolution_stats(raw, region_slug="kyiv_city")
    assert not evo_df.empty
    assert "Year" in evo_df.columns
    assert "Total Alerts" in evo_df.columns
    assert "Avg Duration (min)" in evo_df.columns
