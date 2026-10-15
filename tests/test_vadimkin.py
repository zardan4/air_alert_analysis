"""Comprehensive test suite for Vadimkin open air raid sirens dataset connector."""

from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest

from src.data import ALL_RAW_COLUMNS, VadimkinDataSource, get_alert_data_source
from src.utils.regions import ALL_REGION_SLUGS, TOTAL_REGIONS_COUNT, get_region


def test_vadimkin_data_source_factory():
    """Verify get_alert_data_source instantiates VadimkinDataSource when requested."""
    src = get_alert_data_source(source_type="vadimkin")
    assert isinstance(src, VadimkinDataSource)


def test_vadimkin_packaged_dump_loads():
    """Verify the packaged data/raw/vadimkin_official.csv loads all 25 regions cleanly."""
    dump_path = Path("data/raw/vadimkin_official.csv")
    assert dump_path.exists(), "Packaged Vadimkin CSV should exist"

    source = VadimkinDataSource(csv_path=dump_path)
    df = source.load_raw_alerts(
        start_date="2024-01-01T00:00:00Z",
        end_date="2024-01-07T23:59:59Z",
    )

    assert not df.empty
    assert (df["source"] == "vadimkin").all()
    assert (df["alert_type"] == "air_raid").all()
    # Check that required columns match ALL_RAW_COLUMNS
    for col in ALL_RAW_COLUMNS:
        assert col in df.columns
    # Check regional representation
    unique_slugs = df["region_slug"].unique()
    assert len(unique_slugs) >= 20, f"Expected coverage across Ukraine, got {len(unique_slugs)} regions"


def test_vadimkin_schema_normalization_variants(tmp_path):
    """Test normalization with various column naming conventions and languages."""
    # Test English schema
    csv_en = tmp_path / "vadimkin_en.csv"
    df_en = pd.DataFrame([
        {
            "id": 101,
            "region_title": "Kyiv City",
            "settlement": "",
            "start": "2024-02-01 10:00:00",
            "end": "2024-02-01 11:30:00",
            "clean_end": "2024-02-01 11:30:00",
            "calc_type": "air_raid_red",
        },
        {
            "id": 102,
            "region_title": "Kharkiv Oblast",
            "settlement": "Chuhuiv",
            "start": "2024-02-01 12:00:00",
            "end": "2024-02-01 13:00:00",
            "clean_end": "2024-02-01 13:00:00",
            "calc_type": "yellow_aviation",
        },
        {
            "id": 103,
            "region_title": "Lviv Oblast",
            "settlement": "",
            "start": "2024-02-01 14:00:00",
            "end": "2024-02-01 14:45:00",
            "clean_end": "",
            "calc_type": "unspecified",
        },
    ])
    df_en.to_csv(csv_en, index=False)

    src = VadimkinDataSource(csv_path=csv_en)
    res = src.load_raw_alerts()

    assert len(res) == 3
    assert res.iloc[0]["region_slug"] == "kyiv_city"
    assert res.iloc[0]["threat_level"] == "red"
    assert res.iloc[0]["hromada"] == "oblast_wide"

    assert res.iloc[1]["region_slug"] == "kharkiv"
    assert res.iloc[1]["threat_level"] == "yellow"
    assert res.iloc[1]["hromada"] == "Chuhuiv"

    assert res.iloc[2]["region_slug"] == "lviv"
    assert res.iloc[2]["threat_level"] == "unspecified"


def test_vadimkin_threat_level_mapping(tmp_path):
    """Test threat level extraction into 'red', 'yellow', and 'unspecified'."""
    csv_file = tmp_path / "threats.csv"
    data = [
        {"id": 1, "region": "м. Київ", "start": "2024-01-01 01:00:00", "end": "2024-01-01 02:00:00", "threat_level": "червона (сирена)"},
        {"id": 2, "region": "Одеська область", "start": "2024-01-01 03:00:00", "end": "2024-01-01 04:00:00", "threat_level": "можлива загроза (жовтий)"},
        {"id": 3, "region": "Полтавська область", "start": "2024-01-01 05:00:00", "end": "2024-01-01 06:00:00", "threat_level": "інше / невідомо"},
    ]
    pd.DataFrame(data).to_csv(csv_file, index=False)

    src = VadimkinDataSource(csv_path=csv_file)
    res = src.load_raw_alerts()
    assert res.loc[res["region_slug"] == "kyiv_city", "threat_level"].iloc[0] == "red"
    assert res.loc[res["region_slug"] == "odesa", "threat_level"].iloc[0] == "yellow"
    assert res.loc[res["region_slug"] == "poltava", "threat_level"].iloc[0] == "unspecified"


def test_vadimkin_filtering_dates_and_regions(tmp_path):
    """Test temporal and regional filtering."""
    csv_file = tmp_path / "filter_test.csv"
    data = [
        {"id": 1, "region_title": "Київська область", "start": "2024-01-01 10:00:00", "end": "2024-01-01 11:00:00"},
        {"id": 2, "region_title": "Київська область", "start": "2024-01-05 10:00:00", "end": "2024-01-05 11:00:00"},
        {"id": 3, "region_title": "Харківська область", "start": "2024-01-05 10:00:00", "end": "2024-01-05 11:00:00"},
        {"id": 4, "region_title": "Львівська область", "start": "2024-01-10 10:00:00", "end": "2024-01-10 11:00:00"},
    ]
    pd.DataFrame(data).to_csv(csv_file, index=False)

    src = VadimkinDataSource(csv_path=csv_file)
    res = src.load_raw_alerts(
        start_date="2024-01-04T00:00:00Z",
        end_date="2024-01-06T23:59:59Z",
        regions=["kyiv_oblast"],
    )
    assert len(res) == 1
    assert res.iloc[0]["alert_id"] == "2"
    assert res.iloc[0]["region_slug"] == "kyiv_oblast"


def test_vadimkin_fallback_to_mock_when_missing(tmp_path):
    """Verify fallback to mock source when CSV does not exist and download is disabled/offline."""
    non_existent = tmp_path / "does_not_exist.csv"
    src = VadimkinDataSource(
        csv_path=non_existent,
        download_if_missing=False,
        fallback_to_mock=True,
    )
    df = src.load_raw_alerts(start_date="2024-01-01", end_date="2024-01-02")
    assert not df.empty
    assert "mock" in df["source"].iloc[0]


def test_vadimkin_raises_when_fallback_disabled(tmp_path):
    """Verify FileNotFoundError is raised when fallback_to_mock=False and CSV is missing."""
    non_existent = tmp_path / "does_not_exist.csv"
    src = VadimkinDataSource(
        csv_path=non_existent,
        download_if_missing=False,
        fallback_to_mock=False,
    )
    with pytest.raises(FileNotFoundError):
        src.load_raw_alerts()


def test_cli_status_shows_vadimkin_mode():
    """Verify 'python main.py status --vadimkin' outputs [VADIMKIN MODE]."""
    cmd = [sys.executable, "main.py", "status", "--vadimkin"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "[VADIMKIN MODE]" in res.stdout
    assert "Active Source:    vadimkin" in res.stdout
    assert "Vadimkin Dump:" in res.stdout


def test_cli_generate_data_with_vadimkin_flag(tmp_path):
    """Verify 'python main.py generate-data --vadimkin' uses Vadimkin dataset."""
    raw_out = tmp_path / "raw_vadimkin.parquet"
    grid_out = tmp_path / "grid_vadimkin.parquet"

    cmd = [
        sys.executable,
        "main.py",
        "generate-data",
        "--vadimkin",
        "--raw-output", str(raw_out),
        "--grid-output", str(grid_out),
        "--start-date", "2024-01-01",
        "--end-date", "2024-01-03",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "[VADIMKIN MODE]" in res.stdout
    assert "source: vadimkin" in res.stdout
    assert raw_out.exists()
    assert grid_out.exists()

    df_raw = pd.read_parquet(raw_out)
    assert (df_raw["source"] == "vadimkin").all()
    df_grid = pd.read_parquet(grid_out)
    assert len(df_grid) > 0


def test_cli_generate_data_with_source_vadimkin(tmp_path):
    """Verify 'python main.py generate-data --source vadimkin' activates Vadimkin ingestion."""
    raw_out = tmp_path / "raw_vadimkin_src.parquet"
    grid_out = tmp_path / "grid_vadimkin_src.parquet"

    cmd = [
        sys.executable,
        "main.py",
        "generate-data",
        "--source", "vadimkin",
        "--raw-output", str(raw_out),
        "--grid-output", str(grid_out),
        "--start-date", "2024-01-01",
        "--end-date", "2024-01-03",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "[VADIMKIN MODE]" in res.stdout
    assert "source: vadimkin" in res.stdout
