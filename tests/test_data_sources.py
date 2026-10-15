"""Unit tests for alert data sources and schema validation."""

import pytest
import pandas as pd

from src.data.alerts_in_ua import AlertsInUaDataSource
from src.data.base import ALL_RAW_COLUMNS, AlertDataSource, REQUIRED_RAW_COLUMNS
from src.data.mock_source import MockAlertDataSource
from src.utils.regions import ALL_REGION_SLUGS, get_region


def test_mock_alert_data_source_generation():
    """Verify mock generator produces valid data conforming to schema."""
    source = MockAlertDataSource(seed=42)
    df = source.load_raw_alerts(
        start_date="2024-01-01T00:00:00Z",
        end_date="2024-01-07T23:59:59Z",
    )
    assert not df.empty
    assert list(df.columns) == ALL_RAW_COLUMNS
    assert set(REQUIRED_RAW_COLUMNS).issubset(set(df.columns))

    # Verify threat levels and hromadas
    assert set(df["threat_level"].unique()).issubset({"red", "yellow", "unspecified"})
    assert "threat_level" in df.columns
    assert "hromada" in df.columns

    # Ensure finished_at is strictly after started_at
    assert (df["finished_at"] > df["started_at"]).all()
    # Ensure UTC timezone
    assert df["started_at"].dt.tz is not None
    assert df["finished_at"].dt.tz is not None
    # Ensure all region slugs are valid
    for slug in df["region_slug"].unique():
        assert slug in ALL_REGION_SLUGS


@pytest.mark.parametrize("slug", ["kyiv_city", "kharkiv", "lviv", "odesa"])
def test_mock_single_oblast_isolation(slug: str):
    """Verify mock generator supports isolating a single oblast."""
    source = MockAlertDataSource(seed=123)
    df = source.load_raw_alerts(
        start_date="2024-01-01T00:00:00Z",
        end_date="2024-01-14T23:59:59Z",
        regions=[slug],
    )
    assert not df.empty
    assert (df["region_slug"] == slug).all()
    assert (df["region_id"] == get_region(slug).id).all()


def test_validate_raw_schema_rejects_inverted_interval():
    """Verify strict validation raises ValueError when finished_at < started_at."""
    source = MockAlertDataSource()
    corrupt_df = pd.DataFrame([{
        "alert_id": "bad_1",
        "region_slug": "kyiv_city",
        "region_id": 11,
        "started_at": "2024-01-02T10:00:00Z",
        "finished_at": "2024-01-02T09:00:00Z",  # Inverted!
        "alert_type": "air_raid",
        "source": "test",
    }])

    with pytest.raises(ValueError, match="finished_at < started_at"):
        source.validate_raw_schema(corrupt_df, strict=True)


def test_validate_raw_schema_rejects_missing_column():
    """Verify strict validation checks for missing required schema columns."""
    source = MockAlertDataSource()
    incomplete_df = pd.DataFrame([{
        "alert_id": "inc_1",
        "started_at": "2024-01-02T10:00:00Z",
    }])

    with pytest.raises(ValueError, match="missing required columns"):
        source.validate_raw_schema(incomplete_df)


def test_alerts_in_ua_normalization():
    """Verify alerts.in.ua mapper correctly maps raw Ukrainian names and formats."""
    connector = AlertsInUaDataSource()
    raw_data = pd.DataFrame([
        {
            "id": 101,
            "location_title": "Харківська область",
            "start": "2024-02-01 14:00:00+00:00",
            "end": "2024-02-01 14:45:00+00:00",
            "type": "air_raid",
        },
        {
            "id": 102,
            "location_title": "м. Київ",
            "start": "2024-02-01 15:00:00+00:00",
            "end": "2024-02-01 16:30:00+00:00",
            "type": "air_raid",
        },
    ])

    normalized = connector._normalize_alerts_in_ua_dataframe(raw_data)
    assert len(normalized) == 2
    assert normalized.iloc[0]["region_slug"] == "kharkiv"
    assert normalized.iloc[1]["region_slug"] == "kyiv_city"
    assert (normalized["source"] == "alerts_in_ua").all()
    # Backward compatibility: threat_level and hromada default gracefully
    assert (normalized["threat_level"] == "unspecified").all()
    assert (normalized["hromada"] == "oblast_wide").all()


def test_validate_raw_schema_backward_compatibility():
    """Verify that raw alert data missing threat_level and hromada defaults gracefully."""
    source = MockAlertDataSource()
    legacy_df = pd.DataFrame([{
        "alert_id": "legacy_1",
        "region_slug": "odesa",
        "region_id": 16,
        "started_at": "2022-03-01T10:00:00Z",
        "finished_at": "2022-03-01T11:00:00Z",
        "alert_type": "air_raid",
        "source": "legacy_archive",
    }])

    validated = source.validate_raw_schema(legacy_df, strict=True)
    assert len(validated) == 1
    assert validated.iloc[0]["threat_level"] == "unspecified"
    assert validated.iloc[0]["hromada"] == "oblast_wide"
    assert list(validated.columns) == ALL_RAW_COLUMNS


def test_threat_level_normalization_and_unspecified():
    """Verify that unrecognized or null threat levels default to 'unspecified'."""
    source = MockAlertDataSource()
    mixed_df = pd.DataFrame([
        {
            "alert_id": "t1",
            "region_slug": "kyiv_city",
            "region_id": 11,
            "started_at": "2024-03-01T10:00:00Z",
            "finished_at": "2024-03-01T11:00:00Z",
            "alert_type": "air_raid",
            "source": "test",
            "threat_level": "RED",  # Uppercase
            "hromada": "kyiv_city",
        },
        {
            "alert_id": "t2",
            "region_slug": "kyiv_city",
            "region_id": 11,
            "started_at": "2024-03-01T12:00:00Z",
            "finished_at": "2024-03-01T13:00:00Z",
            "alert_type": "air_raid",
            "source": "test",
            "threat_level": "yellow",  # Lowercase
            "hromada": "",
        },
        {
            "alert_id": "t3",
            "region_slug": "kyiv_city",
            "region_id": 11,
            "started_at": "2024-03-01T14:00:00Z",
            "finished_at": "2024-03-01T15:00:00Z",
            "alert_type": "air_raid",
            "source": "test",
            "threat_level": "CRITICAL_UNKNOWN",  # Unrecognized
            "hromada": None,
        },
    ])

    validated = source.validate_raw_schema(mixed_df)
    assert validated.iloc[0]["threat_level"] == "red"
    assert validated.iloc[1]["threat_level"] == "yellow"
    assert validated.iloc[2]["threat_level"] == "unspecified"
    assert validated.iloc[0]["hromada"] == "kyiv_city"
    assert validated.iloc[1]["hromada"] == "oblast_wide"
    assert validated.iloc[2]["hromada"] == "oblast_wide"


def test_alerts_in_ua_load_csv_dump(tmp_path):
    """Verify AlertsInUaDataSource parses alerts from CSV file dump."""
    csv_file = tmp_path / "alerts_dump.csv"
    csv_content = (
        "id,location_title,location_type,started_at,finished_at,updated_at,alert_type,location_uid,threat_level,hromada\n"
        "201,Харківська область,oblast,2024-03-01T10:00:00Z,2024-03-01T11:15:00Z,2024-03-01T11:15:00Z,air_raid,7,red,oblast_wide\n"
        "202,м. Київ,city,2024-03-01T12:00:00Z,2024-03-01T12:45:00Z,2024-03-01T12:45:00Z,air_raid,11,yellow,kyiv_city\n"
    )
    csv_file.write_text(csv_content, encoding="utf-8")

    source = AlertsInUaDataSource(dump_path=csv_file)
    df = source.load_from_local_dump(csv_file)
    assert len(df) == 2
    assert list(df.columns) == ALL_RAW_COLUMNS
    assert df.iloc[0]["region_slug"] == "kharkiv"
    assert df.iloc[0]["threat_level"] == "red"
    assert df.iloc[1]["region_slug"] == "kyiv_city"
    assert df.iloc[1]["hromada"] == "kyiv_city"
    assert (df["source"] == "alerts_in_ua").all()


def test_alerts_in_ua_load_json_dumps(tmp_path):
    """Verify AlertsInUaDataSource loads both list-based and nested JSON formats."""
    # 1. Standard list JSON
    list_json = tmp_path / "alerts_list.json"
    import json
    data_list = [
        {
            "id": 301,
            "location_title": "Одеська область",
            "started_at": "2024-04-01T02:00:00Z",
            "finished_at": "2024-04-01T03:30:00Z",
            "alert_type": "air_raid",
            "location_uid": 16,
            "threat_level": "red",
        }
    ]
    list_json.write_text(json.dumps(data_list), encoding="utf-8")

    source = AlertsInUaDataSource(dump_path=list_json)
    df_list = source.load_from_local_dump(list_json)
    assert len(df_list) == 1
    assert df_list.iloc[0]["region_slug"] == "odesa"
    assert df_list.iloc[0]["region_id"] == 16

    # 2. Nested alerts.in.ua API-style JSON: {"alerts": [...], "meta": ...}
    nested_json = tmp_path / "alerts_nested.json"
    nested_data = {
        "alerts": [
            {
                "id": 302,
                "location_title": "Львівська область",
                "started_at": "2024-04-01T04:00:00Z",
                "finished_at": "2024-04-01T05:00:00Z",
                "alert_type": "air_raid",
                "location_uid": 14,
            }
        ],
        "meta": {"last_updated_at": "2024-04-01T05:01:00Z"},
    }
    nested_json.write_text(json.dumps(nested_data), encoding="utf-8")

    df_nested = source.load_from_local_dump(nested_json)
    assert len(df_nested) == 1
    assert df_nested.iloc[0]["region_slug"] == "lviv"
    assert df_nested.iloc[0]["region_id"] == 14


def test_alerts_in_ua_sub_oblast_mapping_with_location_oblast():
    """Verify sub-oblast alerts (raions/communities) resolve to parent oblast."""
    source = AlertsInUaDataSource()
    raw_sub = pd.DataFrame([
        {
            "id": 401,
            "location_title": "Ізюмський район",
            "location_oblast": "Харківська область",
            "location_type": "raion",
            "start": "2024-05-01 10:00:00+00:00",
            "end": "2024-05-01 11:00:00+00:00",
            "type": "air_raid",
        }
    ])
    norm = source._normalize_alerts_in_ua_dataframe(raw_sub)
    assert len(norm) == 1
    assert norm.iloc[0]["region_slug"] == "kharkiv"
    assert norm.iloc[0]["region_id"] == 7
    assert norm.iloc[0]["hromada"] == "Ізюмський район"


def test_alerts_in_ua_region_slug_resolution_varieties():
    """Verify resolution of diverse region ID and name variants."""
    source = AlertsInUaDataSource()
    # Integer ID
    assert source._resolve_region_slug(11) == "kyiv_city"
    # String digit ID
    assert source._resolve_region_slug("7") == "kharkiv"
    # Float digit ID
    assert source._resolve_region_slug(16.0) == "odesa"
    # Exact Ukrainian name
    assert source._resolve_region_slug("Харківська область") == "kharkiv"
    # Variant shorthands
    assert source._resolve_region_slug("м.Київ") == "kyiv_city"
    assert source._resolve_region_slug("Київ") == "kyiv_city"
    assert source._resolve_region_slug("Київщина") == "kyiv_oblast"
    assert source._resolve_region_slug("Львівщина") == "lviv"
    # English names
    assert source._resolve_region_slug("Kyiv City") == "kyiv_city"
    assert source._resolve_region_slug("Kharkiv Oblast") == "kharkiv"
    assert source._resolve_region_slug("Lviv") == "lviv"
    # Non-existent or non-25 region
    assert source._resolve_region_slug("Unknown Region 999") is None
    assert source._resolve_region_slug(999) is None
    assert source._resolve_region_slug("") is None


def test_alerts_in_ua_filtering(tmp_path):
    """Verify date and region filtering in load_raw_alerts."""
    csv_file = tmp_path / "filter_test.csv"
    csv_content = (
        "id,location_title,location_type,started_at,finished_at,location_uid\n"
        "1,Харківська область,oblast,2024-01-02T10:00:00Z,2024-01-02T11:00:00Z,7\n"
        "2,м. Київ,city,2024-01-05T10:00:00Z,2024-01-05T11:00:00Z,11\n"
        "3,Львівська область,oblast,2024-01-10T10:00:00Z,2024-01-10T11:00:00Z,14\n"
    )
    csv_file.write_text(csv_content, encoding="utf-8")

    source = AlertsInUaDataSource(dump_path=csv_file)

    # Filter by date range
    df_date = source.load_raw_alerts(start_date="2024-01-03", end_date="2024-01-08")
    assert len(df_date) == 1
    assert df_date.iloc[0]["region_slug"] == "kyiv_city"

    # Filter by region
    df_reg = source.load_raw_alerts(regions=["lviv"])
    assert len(df_reg) == 1
    assert df_reg.iloc[0]["region_slug"] == "lviv"


def test_alerts_in_ua_fetch_active_alerts_api(monkeypatch):
    """Verify API connector parsing with mocked HTTP responses."""
    import requests

    class MockResponse:
        status_code = 200

        def json(self):
            return {
                "alerts": [
                    {
                        "id": 501,
                        "location_title": "Харківська область",
                        "started_at": "2024-06-01T12:00:00Z",
                        "finished_at": None,  # Active alert!
                        "alert_type": "air_raid",
                        "location_uid": 7,
                    }
                ]
            }

    monkeypatch.setattr(requests, "get", lambda url, headers, timeout: MockResponse())

    source = AlertsInUaDataSource(api_token="test_token_123")
    df = source.fetch_active_alerts()
    assert len(df) == 1
    assert df.iloc[0]["region_slug"] == "kharkiv"
    assert df.iloc[0]["source"] == "alerts_in_ua"
    assert pd.notna(df.iloc[0]["finished_at"])


def test_alerts_in_ua_fetch_active_alerts_status_codes(monkeypatch):
    """Verify API error handling for 304, 401, 403, 429."""
    import requests

    class MockStatusResponse:
        def __init__(self, code):
            self.status_code = code

        def raise_for_status(self):
            raise requests.HTTPError(f"HTTP {self.status_code}")

    source = AlertsInUaDataSource(api_token="test_tok")

    # 304 Not Modified -> empty DataFrame
    monkeypatch.setattr(requests, "get", lambda *a, **k: MockStatusResponse(304))
    empty_df = source.fetch_active_alerts()
    assert empty_df.empty

    # 401 Unauthorized -> PermissionError
    monkeypatch.setattr(requests, "get", lambda *a, **k: MockStatusResponse(401))
    with pytest.raises(PermissionError, match="401 Unauthorized"):
        source.fetch_active_alerts()

    # 403 Forbidden -> PermissionError
    monkeypatch.setattr(requests, "get", lambda *a, **k: MockStatusResponse(403))
    with pytest.raises(PermissionError, match="403 Forbidden"):
        source.fetch_active_alerts()

    # 429 Rate Limit -> RuntimeError
    monkeypatch.setattr(requests, "get", lambda *a, **k: MockStatusResponse(429))
    with pytest.raises(RuntimeError, match="rate limit exceeded"):
        source.fetch_active_alerts()


def test_alerts_in_ua_missing_token_raises():
    """Verify fetch_active_alerts raises ValueError when token is not provided."""
    source = AlertsInUaDataSource(api_token="")
    # Ensure env var is cleared for this test
    import os
    orig_env = os.environ.pop("ALERTS_IN_UA_TOKEN", None)
    try:
        with pytest.raises(ValueError, match="API token is missing"):
            source.fetch_active_alerts()
    finally:
        if orig_env:
            os.environ["ALERTS_IN_UA_TOKEN"] = orig_env


def test_alerts_in_ua_api_fallback_to_dump(tmp_path, monkeypatch):
    """Verify API failure falls back to dump when fallback_to_dump=True."""
    import requests

    dump_file = tmp_path / "fallback.csv"
    dump_file.write_text(
        "id,location_title,location_type,started_at,finished_at,location_uid\n"
        "99,Черкаська область,oblast,2024-01-01T01:00:00Z,2024-01-01T02:00:00Z,1\n",
        encoding="utf-8"
    )

    def mock_get_fail(*args, **kwargs):
        raise requests.RequestException("Connection refused")

    monkeypatch.setattr(requests, "get", mock_get_fail)

    source = AlertsInUaDataSource(
        dump_path=dump_file,
        api_token="valid_token",
        fallback_to_dump=True,
    )
    df = source.load_raw_alerts(use_api=True)
    assert len(df) == 1
    assert df.iloc[0]["region_slug"] == "cherkasy"


def test_alerts_in_ua_fallback_to_mock_when_dump_missing(tmp_path, monkeypatch):
    """Verify fallback to mock generator when neither API nor dump is accessible."""
    import requests

    def mock_get_fail(*args, **kwargs):
        raise requests.RequestException("No internet")

    monkeypatch.setattr(requests, "get", mock_get_fail)

    non_existent = tmp_path / "does_not_exist.csv"
    source = AlertsInUaDataSource(
        dump_path=non_existent,
        api_token="valid_token",
        fallback_to_dump=False,
        fallback_to_mock=True,
    )
    df = source.load_raw_alerts(
        start_date="2024-01-01T00:00:00Z",
        end_date="2024-01-03T00:00:00Z",
        use_api=True,
    )
    assert not df.empty
    assert "mock" in df.iloc[0]["source"]


def test_get_alert_data_source_factory():
    """Verify factory properly switches between mock and alerts_in_ua."""
    from src.data import get_alert_data_source

    mock_src = get_alert_data_source(mock=True)
    assert isinstance(mock_src, MockAlertDataSource)

    mock_src_named = get_alert_data_source(source_type="mock")
    assert isinstance(mock_src_named, MockAlertDataSource)

    real_src = get_alert_data_source(mock=False)
    assert isinstance(real_src, AlertsInUaDataSource)

    real_src_named = get_alert_data_source(source_type="alerts_in_ua")
    assert isinstance(real_src_named, AlertsInUaDataSource)


def test_packaged_dump_file_loads_successfully():
    """Verify the repository packaged alerts_in_ua_dump.csv loads all 25 regions."""
    source = AlertsInUaDataSource(dump_path="data/raw/alerts_in_ua_dump.csv")
    df = source.load_raw_alerts(
        start_date="2024-01-01T00:00:00Z",
        end_date="2024-01-14T23:59:59Z",
    )
    assert not df.empty
    assert (df["source"] == "alerts_in_ua").all()
    assert df["region_slug"].nunique() == 25
    assert set(df["threat_level"].unique()).issubset({"red", "yellow", "unspecified"})


def test_alerts_in_ua_official_uids_and_titles():
    """Verify official alerts.in.ua location_uids resolve accurately without misattribution."""
    source = AlertsInUaDataSource()
    raw_payload = pd.DataFrame([
        {
            "location_uid": 14,
            "location_title": "Київська область",
            "started_at": "2024-05-01T10:00:00Z",
            "finished_at": "2024-05-01T11:00:00Z",
        },
        {
            "location_uid": 22,
            "location_title": "Харківська область",
            "started_at": "2024-05-01T10:00:00Z",
            "finished_at": "2024-05-01T11:00:00Z",
        },
        {
            "location_uid": 16,
            "location_title": "Луганська область",
            "started_at": "2024-05-01T10:00:00Z",
            "finished_at": "2024-05-01T11:00:00Z",
        },
        {
            "location_uid": 31,
            "location_title": "м. Київ",
            "started_at": "2024-05-01T10:00:00Z",
            "finished_at": "2024-05-01T11:00:00Z",
        },
        {
            "location_uid": 28,
            "location_title": "Донецька область",
            "started_at": "2024-05-01T10:00:00Z",
            "finished_at": "2024-05-01T11:00:00Z",
        },
    ])

    norm = source._normalize_alerts_in_ua_dataframe(raw_payload)
    slug_map = dict(zip(norm["alert_id"], norm["region_slug"]))
    assert slug_map["alert_0"] == "kyiv_oblast"
    assert slug_map["alert_1"] == "kharkiv"
    assert slug_map["alert_2"] == "luhansk"
    assert slug_map["alert_3"] == "kyiv_city"
    assert slug_map["alert_4"] == "donetsk"


def test_alerts_in_ua_ukrainian_abbreviations():
    """Verify common Ukrainian administrative abbreviations normalize correctly."""
    source = AlertsInUaDataSource()
    test_cases = [
        ("Київська обл.", "kyiv_oblast"),
        ("Київська обл", "kyiv_oblast"),
        ("Харківська обл.", "kharkiv"),
        ("м.Харків", "kharkiv"),
        ("Львівська обл.", "lviv"),
        ("Одеська обл.", "odesa"),
        ("Дніпропетровська обл.", "dnipropetrovsk"),
        ("м. Одеса", "odesa"),
        ("м. Львів", "lviv"),
    ]
    for raw_name, expected_slug in test_cases:
        assert source._resolve_region_slug(raw_name) == expected_slug, f"Failed for {raw_name}"


def test_alerts_in_ua_jsonl_parsing(tmp_path):
    """Verify AlertsInUaDataSource parses JSONL files with comments and blank lines."""
    jsonl_file = tmp_path / "dump.jsonl"
    jsonl_content = (
        '{"id": 1, "location_title": "м. Київ", "started_at": "2024-01-01T00:00:00Z", "finished_at": "2024-01-01T01:00:00Z"}\n'
        '\n'  # blank line
        '# comment line\n'
        '{"id": 2, "location_title": "Харківська область", "started_at": "2024-01-01T02:00:00Z", "finished_at": "2024-01-01T03:00:00Z"}\n'
    )
    jsonl_file.write_text(jsonl_content, encoding="utf-8")

    source = AlertsInUaDataSource(dump_path=jsonl_file)
    df = source.load_from_local_dump(jsonl_file)
    assert len(df) == 2
    assert list(df["region_slug"]) == ["kyiv_city", "kharkiv"]


def test_alerts_in_ua_custom_column_aliases(tmp_path):
    """Verify AlertsInUaDataSource handles standard Kaggle/GitHub column variations."""
    csv_file = tmp_path / "custom_columns.csv"
    csv_content = (
        "id,region,alert_start,alert_end,alert_type\n"
        "101,Харківська обл.,2024-02-01 12:00:00+00:00,2024-02-01 13:00:00+00:00,air_raid\n"
        "102,Київська обл.,2024-02-01 14:00:00+00:00,2024-02-01 15:00:00+00:00,air_raid\n"
    )
    csv_file.write_text(csv_content, encoding="utf-8")

    source = AlertsInUaDataSource(dump_path=csv_file)
    df = source.load_from_local_dump(csv_file)
    assert len(df) == 2
    assert df.iloc[0]["region_slug"] == "kharkiv"
    assert df.iloc[1]["region_slug"] == "kyiv_oblast"


