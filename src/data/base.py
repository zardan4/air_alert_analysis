"""Base interface and schema validation for air raid alert data sources."""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import pandas as pd

from src.utils.regions import get_all_region_slugs, is_valid_region

REQUIRED_RAW_COLUMNS = [
    "alert_id",
    "region_slug",
    "region_id",
    "started_at",
    "finished_at",
    "alert_type",
    "source",
]

OPTIONAL_RAW_COLUMNS = [
    "threat_level",
    "hromada",
]

ALL_RAW_COLUMNS = REQUIRED_RAW_COLUMNS + OPTIONAL_RAW_COLUMNS


class AlertDataSource(ABC):
    """Abstract interface defining ingestion and access to raw air raid alert intervals."""

    @abstractmethod
    def load_raw_alerts(
        self,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        regions: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Load raw alert interval records.
        
        Args:
            start_date: Earliest alert start time (inclusive).
            end_date: Latest alert start time (inclusive).
            regions: Optional list of region slugs. If None, all 25 regions are loaded.

        Returns:
            pd.DataFrame conforming to REQUIRED_RAW_COLUMNS, with started_at and finished_at in UTC.
        """
        pass

    def validate_raw_schema(self, df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
        """Validate, normalize types, and verify temporal sanity of raw alert intervals.
        
        Guarantees:
        - All REQUIRED_RAW_COLUMNS are present.
        - started_at and finished_at are datetime64[ns, UTC].
        - finished_at >= started_at (drops or errors on corrupted intervals).
        - region_slugs belong to valid registered regions.
        """
        if df.empty:
            return pd.DataFrame(columns=ALL_RAW_COLUMNS)

        missing = [col for col in REQUIRED_RAW_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Raw alerts DataFrame is missing required columns: {missing}")

        df = df.copy()

        # Enforce UTC datetime
        df["started_at"] = pd.to_datetime(df["started_at"], utc=True)
        df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True)

        # Sanity check: finished_at >= started_at
        invalid_temporal = df["finished_at"] < df["started_at"]
        if invalid_temporal.any():
            if strict:
                invalid_samples = df[invalid_temporal][["alert_id", "region_slug", "started_at", "finished_at"]]
                raise ValueError(
                    f"Found {invalid_temporal.sum()} alerts with finished_at < started_at:\n{invalid_samples.head()}"
                )
            else:
                df = df[~invalid_temporal]

        # Validate region slugs
        invalid_regions = [slug for slug in df["region_slug"].unique() if not is_valid_region(slug)]
        if invalid_regions:
            raise ValueError(f"Unrecognized region slugs found in raw data: {invalid_regions}")

        # Threat level validation with backward compatibility
        # Ukrainian official alert terminology: 'red' (direct threat / active siren),
        # 'yellow' (heightened danger / aviation activity / potential threat),
        # 'unspecified' (fallback for periods/datasets missing this differentiation).
        if "threat_level" not in df.columns:
            df["threat_level"] = "unspecified"
        else:
            df["threat_level"] = df["threat_level"].fillna("unspecified").astype(str).str.strip().str.lower()
            df["threat_level"] = df["threat_level"].apply(
                lambda x: x if x in ("red", "yellow") else "unspecified"
            )

        # Territorial community (hromada) validation with backward compatibility
        # Falls back to 'oblast_wide' when district/hromada differentiation was absent.
        if "hromada" not in df.columns:
            df["hromada"] = "oblast_wide"
        else:
            df["hromada"] = df["hromada"].fillna("oblast_wide").astype(str).str.strip()
            df["hromada"] = df["hromada"].replace("", "oblast_wide")

        # Ensure correct dtypes
        df["alert_id"] = df["alert_id"].astype(str)
        df["region_id"] = df["region_id"].astype(int)
        df["region_slug"] = df["region_slug"].astype(str)
        df["alert_type"] = df["alert_type"].astype(str)
        df["source"] = df["source"].astype(str)
        df["threat_level"] = df["threat_level"].astype(str)
        df["hromada"] = df["hromada"].astype(str)

        # Sort chronologically by start time and region
        df = df.sort_values(by=["started_at", "region_slug"]).reset_index(drop=True)
        return df[ALL_RAW_COLUMNS]

    def save_raw_parquet(self, df: pd.DataFrame, path: Union[str, Path]) -> Path:
        """Persist validated raw alert records to Parquet."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        validated_df = self.validate_raw_schema(df)
        validated_df.to_parquet(path, index=False)
        return path
