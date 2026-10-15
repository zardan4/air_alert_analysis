"""Configuration management module."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml


@dataclass(frozen=True)
class HromadaInfo:
    slug: str
    name_en: str
    name_uk: str


@dataclass(frozen=True)
class RegionInfo:
    id: int
    slug: str
    name_en: str
    name_uk: str
    alerts_in_ua_id: int
    hromadas: Tuple[HromadaInfo, ...] = ()


@dataclass(frozen=True)
class GridConfig:
    frequency: str = "1h"
    min_alert_overlap_sec: int = 60


@dataclass(frozen=True)
class DataConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    regions_config_path: str = "config/regions.yaml"
    default_source: str = "alerts_in_ua"
    alerts_in_ua_dump_path: str = "data/raw/alerts_in_ua_dump.csv"
    vadimkin_dump_path: str = "data/raw/vadimkin_official.csv"


@dataclass(frozen=True)
class AlertsInUaConfig:
    api_base_url: str = "https://api.alerts.in.ua/v1"
    token_env_var: str = "ALERTS_IN_UA_TOKEN"
    api_token: str = ""
    timeout_sec: int = 10
    fallback_to_dump: bool = True
    fallback_to_mock: bool = True


@dataclass(frozen=True)
class VadimkinConfig:
    url_uk: str = "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/official_data_uk.csv"
    url_en: str = "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/official_data_en.csv"
    default_language: str = "uk"
    timeout_sec: int = 15
    fallback_to_mock: bool = True


@dataclass(frozen=True)
class MockGeneratorConfig:
    seed: int = 42
    start_date: str = "2024-01-01T00:00:00Z"
    end_date: str = "2024-06-30T23:59:59Z"
    base_daily_rate: float = 3.5
    night_multiplier: float = 2.2


@dataclass(frozen=True)
class BacktestingConfig:
    horizons_hours: List[int] = field(default_factory=lambda: [1, 2, 3])
    test_window_days: int = 60
    embargo_hours: int = 3


@dataclass(frozen=True)
class AppConfig:
    project_name: str
    version: str
    timezone: str
    grid: GridConfig
    data: DataConfig
    alerts_in_ua: AlertsInUaConfig
    vadimkin: VadimkinConfig
    mock_generator: MockGeneratorConfig
    backtesting: BacktestingConfig


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def load_regions(regions_path: Optional[Path | str] = None) -> Dict[str, RegionInfo]:
    """Load the standardized 25 Ukrainian regions from YAML.
    
    Returns a dictionary mapping region slug -> RegionInfo.
    """
    if regions_path is None:
        regions_path = get_project_root() / "config" / "regions.yaml"
    else:
        regions_path = Path(regions_path)

    if not regions_path.exists():
        raise FileNotFoundError(f"Regions configuration not found at {regions_path}")

    with open(regions_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    regions: Dict[str, RegionInfo] = {}
    for item in data.get("regions", []):
        hromadas_list = tuple(
            HromadaInfo(
                slug=h["slug"],
                name_en=h["name_en"],
                name_uk=h["name_uk"],
            )
            for h in item.get("hromadas", [])
        )
        info = RegionInfo(
            id=item["id"],
            slug=item["slug"],
            name_en=item["name_en"],
            name_uk=item["name_uk"],
            alerts_in_ua_id=item["alerts_in_ua_id"],
            hromadas=hromadas_list,
        )
        regions[info.slug] = info

    return regions


def load_config(config_path: Optional[Path | str] = None) -> AppConfig:
    """Load system configuration from YAML."""
    if config_path is None:
        config_path = get_project_root() / "config" / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f)

    project = data.get("project", {})
    grid_data = data.get("grid", {})
    data_data = data.get("data", {})
    alerts_data = data.get("alerts_in_ua", {})
    vadimkin_data = data.get("vadimkin", {})
    mock_data = data.get("mock_generator", {})
    backtesting_data = data.get("backtesting", {})

    return AppConfig(
        project_name=project.get("name", "ukraine-alert-forecast"),
        version=project.get("version", "0.1.0"),
        timezone=project.get("timezone", "Europe/Kyiv"),
        grid=GridConfig(
            frequency=grid_data.get("frequency", "1h"),
            min_alert_overlap_sec=grid_data.get("min_alert_overlap_sec", 60),
        ),
        data=DataConfig(
            raw_dir=data_data.get("raw_dir", "data/raw"),
            processed_dir=data_data.get("processed_dir", "data/processed"),
            regions_config_path=data_data.get("regions_config_path", "config/regions.yaml"),
            default_source=data_data.get("default_source", "alerts_in_ua"),
            alerts_in_ua_dump_path=data_data.get("alerts_in_ua_dump_path", "data/raw/alerts_in_ua_dump.csv"),
            vadimkin_dump_path=data_data.get("vadimkin_dump_path", "data/raw/vadimkin_official.csv"),
        ),
        alerts_in_ua=AlertsInUaConfig(
            api_base_url=alerts_data.get("api_base_url", "https://api.alerts.in.ua/v1"),
            token_env_var=alerts_data.get("token_env_var", "ALERTS_IN_UA_TOKEN"),
            api_token=alerts_data.get("api_token", ""),
            timeout_sec=alerts_data.get("timeout_sec", 10),
            fallback_to_dump=alerts_data.get("fallback_to_dump", True),
            fallback_to_mock=alerts_data.get("fallback_to_mock", True),
        ),
        vadimkin=VadimkinConfig(
            url_uk=vadimkin_data.get(
                "url_uk",
                "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/official_data_uk.csv",
            ),
            url_en=vadimkin_data.get(
                "url_en",
                "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/official_data_en.csv",
            ),
            default_language=vadimkin_data.get("default_language", "uk"),
            timeout_sec=vadimkin_data.get("timeout_sec", 15),
            fallback_to_mock=vadimkin_data.get("fallback_to_mock", True),
        ),
        mock_generator=MockGeneratorConfig(
            seed=mock_data.get("seed", 42),
            start_date=mock_data.get("start_date", "2024-01-01T00:00:00Z"),
            end_date=mock_data.get("end_date", "2024-06-30T23:59:59Z"),
            base_daily_rate=mock_data.get("base_daily_rate", 3.5),
            night_multiplier=mock_data.get("night_multiplier", 2.2),
        ),
        backtesting=BacktestingConfig(
            horizons_hours=backtesting_data.get("horizons_hours", [1, 2, 3]),
            test_window_days=backtesting_data.get("test_window_days", 60),
            embargo_hours=backtesting_data.get("embargo_hours", 3),
        ),
    )
