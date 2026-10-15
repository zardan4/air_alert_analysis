"""Connector for the open Vadimkin Ukrainian Air Raid Sirens Dataset.

Provides zero-token, open-access ingestion of official historical sirens records
from the Vadimkin GitHub dataset repository:
https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset

Supports local CSV files, automatic mirror downloading, schema normalization,
threat level mapping ('red', 'yellow', 'unspecified'), and hromada localization.
"""

from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import requests

from src.config import load_config
from src.data.alerts_in_ua import ALERTS_IN_UA_NAME_MAPPING
from src.data.base import ALL_RAW_COLUMNS, AlertDataSource
from src.data.mock_source import MockAlertDataSource
from src.utils.regions import get_all_regions, get_region, is_valid_region

logger = logging.getLogger(__name__)

# Pre-compiled case-folded lookup dictionary for fast O(1) matching
_NAME_MAPPING_LOWER: Dict[str, str] = {k.lower(): v for k, v in ALERTS_IN_UA_NAME_MAPPING.items()}


class VadimkinDataSource(AlertDataSource):
    """Data source for Vadimkin Ukrainian Air Raid Sirens open CSV datasets."""

    def __init__(
        self,
        csv_path: Optional[Union[str, Path]] = None,
        dataset_url: Optional[str] = None,
        language: str = "uk",
        timeout_sec: Optional[int] = None,
        download_if_missing: bool = True,
        fallback_to_mock: Optional[bool] = None,
    ):
        """Initialize Vadimkin dataset connector.

        Args:
            csv_path: Optional path to local CSV dump.
            dataset_url: Optional remote URL for raw dataset.
            language: Dataset language ('uk' for Ukrainian, 'en' for English).
            timeout_sec: Network timeout for remote fetch.
            download_if_missing: Whether to attempt remote download if local file is absent.
            fallback_to_mock: Whether to fall back to synthetic data if dataset is unavailable.
        """
        try:
            config = load_config()
            cfg_vadimkin = getattr(config, "vadimkin", None)
            cfg_data = getattr(config, "data", None)
        except Exception:
            config = None
            cfg_vadimkin = None
            cfg_data = None

        self.language = language.lower() if language else "uk"

        # Determine target local CSV path
        if csv_path is not None:
            self.csv_path = Path(csv_path)
        elif cfg_data and getattr(cfg_data, "vadimkin_dump_path", None):
            self.csv_path = Path(cfg_data.vadimkin_dump_path)
        else:
            self.csv_path = Path("data/raw/vadimkin_official.csv")

        # Determine download URL
        default_url = (
            getattr(cfg_vadimkin, "url_en", None)
            if self.language == "en" and cfg_vadimkin
            else getattr(cfg_vadimkin, "url_uk", None) if cfg_vadimkin else None
        )
        if not default_url:
            default_url = (
                "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/official_data_en.csv"
                if self.language == "en"
                else "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/official_data_uk.csv"
            )
        self.dataset_url = dataset_url or default_url

        self.timeout_sec = (
            timeout_sec
            if timeout_sec is not None
            else (getattr(cfg_vadimkin, "timeout_sec", 15) if cfg_vadimkin else 15)
        )
        self.download_if_missing = download_if_missing
        self.fallback_to_mock = (
            fallback_to_mock
            if fallback_to_mock is not None
            else (getattr(cfg_vadimkin, "fallback_to_mock", True) if cfg_vadimkin else True)
        )

    def download_dataset(self, target_path: Optional[Union[str, Path]] = None) -> Optional[Path]:
        """Download raw dataset from GitHub mirror into local destination."""
        dest = Path(target_path) if target_path else self.csv_path
        dest.parent.mkdir(parents=True, exist_ok=True)

        url = self.dataset_url
        logger.info(f"Downloading Vadimkin open sirens dataset from {url} to {dest}...")
        try:
            resp = requests.get(url, timeout=self.timeout_sec, stream=True)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
            logger.info(f"Successfully downloaded Vadimkin dataset to {dest} ({dest.stat().st_size:,} bytes).")
            return dest
        except Exception as exc:
            logger.warning(f"Failed to download Vadimkin dataset from {url}: {exc}")
            if dest.exists() and dest.stat().st_size == 0:
                dest.unlink()
            return None

    def _resolve_region_slug(self, raw_name: Any) -> Optional[str]:
        """Map raw Vadimkin region name to standardized 25 Ukrainian region slugs."""
        if raw_name is None or (isinstance(raw_name, float) and pd.isna(raw_name)):
            return None

        name_str = str(raw_name).strip()
        if not name_str:
            return None

        name_lower = name_str.lower()

        # 1. Exact match in name lookup dictionary
        if name_lower in _NAME_MAPPING_LOWER:
            return _NAME_MAPPING_LOWER[name_lower]

        # 2. Check canonical slugs and registered region names
        if is_valid_region(name_str):
            return get_region(name_str).slug

        for reg in get_all_regions().values():
            if (
                reg.name_uk.lower() == name_lower
                or reg.name_en.lower() == name_lower
                or reg.slug.lower() == name_lower
            ):
                return reg.slug

        # 3. Clean common prefixes/suffixes (e.g. "м. Київ", "Черкаська обл.", "Oblast")
        norm_name = re.sub(r"\b(обл\.?|область|oblast|region)\b", "", name_lower, flags=re.IGNORECASE).strip()
        norm_name = re.sub(r"\s+", " ", norm_name)
        if norm_name in _NAME_MAPPING_LOWER:
            return _NAME_MAPPING_LOWER[norm_name]

        # 4. Check known settlements/hromadas to identify parent oblast
        for reg in get_all_regions().values():
            for h in reg.hromadas:
                if (
                    h.slug.lower() == name_lower
                    or h.name_uk.lower() == name_lower
                    or h.name_en.lower() == name_lower
                ):
                    return reg.slug

        # 5. Substring prefix match for longer names
        for k_lower, slug in _NAME_MAPPING_LOWER.items():
            if len(k_lower) >= 6 and k_lower in name_lower:
                return slug

        return None

    def _map_threat_level(self, raw_threat: Any) -> str:
        """Map threat indicator to standardized Ukrainian alert levels: 'red', 'yellow', 'unspecified'.

        Ukrainian government regulations (enacted September 2026):
        - Yellow Alert: Drone danger (ударні БПЛА / Shahed), air reconnaissance, aviation alert.
        - Red Alert: Missile danger (ракетна небезпека / балістика / масований удар), immediate shelter required.
        - Unspecified: General siren alarms or historical intervals prior to government threat differentiation.

        Note: Generic alert terms such as 'air_raid', 'сирена', 'тривога' denote standard siren events
        and MUST NOT be misclassified as red alerts.
        """
        if raw_threat is None or (isinstance(raw_threat, float) and pd.isna(raw_threat)):
            return "unspecified"

        s = str(raw_threat).strip().lower()
        if not s or s in ("unspecified", "unknown", "інше", "невідомо", "air_raid", "сирена", "тривога", "siren", "0"):
            return "unspecified"

        # Explicit red indicators (missile / ballistic danger or explicit 'red' / 'червоний')
        if any(term in s for term in ("red", "червон", "ракет", "балістик", "missile", "ballistic")):
            return "red"

        # Explicit yellow indicators (drone / Shahed / aviation danger or explicit 'yellow' / 'жовтий')
        if any(term in s for term in ("yellow", "жовт", "можлива", "дрон", "шахед", "shahed", "бпла", "aviation", "авіац", "потенційна")):
            return "yellow"

        return "unspecified"

    def _normalize_vadimkin_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize raw Vadimkin DataFrame into standard schema."""
        if raw_df.empty:
            return pd.DataFrame(columns=ALL_RAW_COLUMNS)

        # Normalize column aliases
        col_aliases = {
            "id": "alert_id",
            "alert_id": "alert_id",
            "region_title": "raw_region",
            "region": "raw_region",
            "region_name": "raw_region",
            "location_title": "raw_region",
            "oblast": "raw_region",
            "location_oblast": "raw_region",
            "settlement": "raw_settlement",
            "community": "raw_settlement",
            "community_title": "raw_settlement",
            "hromada": "raw_settlement",
            "raion": "raw_settlement",
            "start": "raw_start",
            "started_at": "raw_start",
            "start_time": "raw_start",
            "alert_start": "raw_start",
            "end": "raw_end",
            "ended_at": "raw_end",
            "finished_at": "raw_end",
            "end_time": "raw_end",
            "clean_end": "raw_clean_end",
            "calc_type": "raw_threat",
            "threat_level": "raw_threat",
            "type": "raw_threat",
            "alert_type": "raw_threat",
        }

        # Case-insensitive column renaming
        lower_cols = {c.strip().lower(): c for c in raw_df.columns}
        rename_map = {}
        for alias, target in col_aliases.items():
            if alias in lower_cols:
                rename_map[lower_cols[alias]] = target

        df = raw_df.rename(columns=rename_map).copy()

        # Alert ID
        if "alert_id" not in df.columns:
            df["alert_id"] = [f"vadimkin_{i+1}" for i in range(len(df))]
        else:
            df["alert_id"] = df["alert_id"].astype(str)

        # Determine timestamps
        if "raw_start" not in df.columns:
            raise ValueError("Vadimkin dataset is missing 'start' or 'started_at' timestamp column.")

        # If raw_clean_end exists, prefer clean_end when available, otherwise fall back to raw_end
        if "raw_clean_end" in df.columns and "raw_end" in df.columns:
            df["raw_clean_end"] = df["raw_clean_end"].fillna(df["raw_end"])
            finished_col = "raw_clean_end"
        elif "raw_clean_end" in df.columns:
            finished_col = "raw_clean_end"
        elif "raw_end" in df.columns:
            finished_col = "raw_end"
        else:
            finished_col = None

        df["started_at"] = pd.to_datetime(df["raw_start"], utc=True)
        now_utc = pd.Timestamp.now(tz="UTC")
        if finished_col and finished_col in df.columns:
            df["finished_at"] = pd.to_datetime(df[finished_col], utc=True).fillna(now_utc)
        else:
            df["finished_at"] = now_utc

        # Resolve region slugs and territorial community
        has_region = "raw_region" in df.columns
        has_settlement = "raw_settlement" in df.columns
        has_threat = "raw_threat" in df.columns

        resolved_slugs = []
        resolved_ids = []
        resolved_hromadas = []
        threat_levels = []

        for _, row in df.iterrows():
            # 1. Resolve oblast slug
            slug = None
            if has_region and pd.notna(row["raw_region"]):
                slug = self._resolve_region_slug(row["raw_region"])

            # Fallback to settlement matching if region was blank
            if slug is None and has_settlement and pd.notna(row["raw_settlement"]):
                slug = self._resolve_region_slug(row["raw_settlement"])

            if slug and is_valid_region(slug):
                reg = get_region(slug)
                resolved_slugs.append(reg.slug)
                resolved_ids.append(reg.id)

                # Determine hromada / settlement
                if has_settlement and pd.notna(row["raw_settlement"]):
                    s_str = str(row["raw_settlement"]).strip()
                    if s_str and s_str.lower() != reg.name_uk.lower() and s_str.lower() != reg.name_en.lower():
                        resolved_hromadas.append(s_str)
                    else:
                        resolved_hromadas.append("oblast_wide")
                else:
                    resolved_hromadas.append("oblast_wide")
            else:
                resolved_slugs.append(None)
                resolved_ids.append(None)
                resolved_hromadas.append(None)

            # Determine threat level
            raw_threat_val = row["raw_threat"] if has_threat and pd.notna(row["raw_threat"]) else None
            threat_levels.append(self._map_threat_level(raw_threat_val))

        df["region_slug"] = resolved_slugs
        df["region_id"] = resolved_ids
        df["hromada"] = resolved_hromadas
        df["threat_level"] = threat_levels
        df["alert_type"] = "air_raid"
        df["source"] = "vadimkin"

        # Filter out records without valid region resolution or invalid temporal timestamps
        df = df.dropna(subset=["region_slug", "started_at", "finished_at"]).copy()

        return self.validate_raw_schema(df, strict=False)

    def load_from_csv(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Parse raw CSV or CSV.gz export from Vadimkin into standardized DataFrame."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Vadimkin dataset CSV not found: {file_path}")

        compression = "gzip" if file_path.name.lower().endswith(".gz") else None
        raw_df = pd.read_csv(file_path, compression=compression, encoding="utf-8-sig", low_memory=False)
        return self._normalize_vadimkin_dataframe(raw_df)

    def load_raw_alerts(
        self,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        regions: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Load raw alert records from Vadimkin open dataset.

        Args:
            start_date: Earliest alert start time.
            end_date: Latest alert start time.
            regions: Optional list of region slugs.

        Returns:
            pd.DataFrame matching ALL_RAW_COLUMNS.
        """
        target_csv = self.csv_path
        if not target_csv.exists() and self.download_if_missing:
            # Attempt to download from GitHub mirror
            downloaded = self.download_dataset(target_csv)
            if downloaded and downloaded.exists():
                target_csv = downloaded

        if not target_csv.exists():
            if self.fallback_to_mock:
                logger.warning(
                    f"Vadimkin dataset CSV not found at '{target_csv}' and download failed or offline. "
                    f"Falling back to MockAlertDataSource."
                )
                mock_source = MockAlertDataSource()
                return mock_source.load_raw_alerts(
                    start_date=start_date,
                    end_date=end_date,
                    regions=regions,
                )
            else:
                raise FileNotFoundError(
                    f"Vadimkin dataset file not found at '{target_csv}'. Run 'scripts/create_vadimkin_dump.py' "
                    f"or ensure network connectivity to download from GitHub."
                )

        df = self.load_from_csv(target_csv)

        # Apply filtering
        if start_date:
            start_ts = pd.to_datetime(start_date, utc=True)
            df = df[df["started_at"] >= start_ts]
        if end_date:
            end_ts = pd.to_datetime(end_date, utc=True)
            df = df[df["started_at"] <= end_ts]
        if regions:
            df = df[df["region_slug"].isin(regions)]

        return df.reset_index(drop=True)
