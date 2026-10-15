"""Data package for alert loading, synthetic generation, and continuous grid conversion."""

from pathlib import Path
from typing import Optional, Union

from src.data.alerts_in_ua import AlertsInUaDataSource
from src.data.base import ALL_RAW_COLUMNS, AlertDataSource, OPTIONAL_RAW_COLUMNS, REQUIRED_RAW_COLUMNS
from src.data.grid_converter import IntervalToGridConverter
from src.data.mock_source import MockAlertDataSource
from src.data.vadimkin import VadimkinDataSource


def get_alert_data_source(
    mock: bool = False,
    source_type: Optional[str] = None,
    dump_path: Optional[Union[str, Path]] = None,
    api_token: Optional[str] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> AlertDataSource:
    """Instantiate the appropriate AlertDataSource based on mock flag or source name.

    Args:
        mock: If True, explicitly instantiates MockAlertDataSource.
        source_type: 'alerts_in_ua', 'vadimkin', or 'mock'.
        dump_path: Optional path to alerts.in.ua / Vadimkin CSV/JSON/Parquet dump file.
        api_token: Optional alerts.in.ua API token.
        seed: Seed for synthetic mock generation.

    Returns:
        AlertDataSource instance conforming to standard schema interface.
    """
    effective_source = source_type
    if effective_source is None and not mock:
        try:
            from src.config import load_config
            cfg = load_config()
            effective_source = getattr(getattr(cfg, "data", None), "default_source", "alerts_in_ua")
        except Exception:
            effective_source = "alerts_in_ua"

    if mock or effective_source == "mock":
        return MockAlertDataSource(seed=seed, **kwargs)
    elif effective_source == "vadimkin":
        return VadimkinDataSource(csv_path=dump_path, **kwargs)
    return AlertsInUaDataSource(dump_path=dump_path, api_token=api_token, **kwargs)


__all__ = [
    "AlertDataSource",
    "AlertsInUaDataSource",
    "VadimkinDataSource",
    "IntervalToGridConverter",
    "MockAlertDataSource",
    "get_alert_data_source",
    "REQUIRED_RAW_COLUMNS",
    "OPTIONAL_RAW_COLUMNS",
    "ALL_RAW_COLUMNS",
]
