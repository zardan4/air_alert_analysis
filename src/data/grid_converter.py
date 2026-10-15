"""Converts raw air raid alert intervals into a continuous, uniform spatio-temporal grid."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.config import load_config
from src.utils.regions import get_all_region_slugs, get_region


class IntervalToGridConverter:
    """Discretizes raw alert intervals into a continuous uniform time-series grid.
    
    Guarantees:
    - Zero missing intervals: Every region has a contiguous row for every timestamp step.
    - Handles multiple overlapping/nested alerts in the same region without double-counting active time.
    - Accurately computes binary active status, onset events, active duration, and ratio.
    """

    def __init__(
        self,
        frequency: Optional[str] = None,
        min_alert_overlap_sec: Optional[int] = None,
    ):
        config = load_config()
        self.frequency = frequency or config.grid.frequency
        self.min_alert_overlap_sec = (
            min_alert_overlap_sec
            if min_alert_overlap_sec is not None
            else config.grid.min_alert_overlap_sec
        )

    @staticmethod
    def _merge_overlapping_intervals(intervals: List[Tuple[pd.Timestamp, pd.Timestamp]]) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Merge overlapping or adjacent intervals into disjoint active periods."""
        if not intervals:
            return []

        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        merged = [sorted_intervals[0]]

        for current_start, current_end in sorted_intervals[1:]:
            last_start, last_end = merged[-1]
            if current_start <= last_end:
                # Overlap or adjacent: extend last interval
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append((current_start, current_end))

        return merged

    def convert(
        self,
        raw_df: pd.DataFrame,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        regions: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Convert raw alert intervals into continuous uniform grid.

        Args:
            raw_df: DataFrame conforming to AlertDataSource schema.
            start_date: Grid start cutoff (floored to frequency).
            end_date: Grid end cutoff (ceiled to frequency).
            regions: List of region slugs to include. Defaults to all 25 regions.

        Returns:
            pd.DataFrame with columns:
                ['timestamp', 'region_slug', 'region_id', 'alert_active',
                 'alert_onset', 'active_duration_sec', 'active_ratio', 'alerts_count']
            Sorted by timestamp, then region_slug.
        """
        target_regions = regions if regions is not None else get_all_region_slugs()

        # Determine temporal boundary
        if start_date is not None:
            grid_start = pd.to_datetime(start_date, utc=True).floor(self.frequency)
        elif not raw_df.empty:
            grid_start = raw_df["started_at"].min().floor(self.frequency)
        else:
            grid_start = pd.Timestamp.now(tz="UTC").floor(self.frequency)

        if end_date is not None:
            grid_end = pd.to_datetime(end_date, utc=True).ceil(self.frequency)
        elif not raw_df.empty:
            grid_end = raw_df["finished_at"].max().ceil(self.frequency)
        else:
            grid_end = grid_start + pd.Timedelta(days=7)

        if grid_start >= grid_end:
            raise ValueError(f"grid_start ({grid_start}) must be before grid_end ({grid_end})")

        # Create uniform timestamps array
        timestamps = pd.date_range(start=grid_start, end=grid_end, freq=self.frequency, inclusive="left", tz="UTC")
        if len(timestamps) == 0:
            raise ValueError("Generated timestamp range is empty. Check start_date and end_date.")

        # Duration of one step in seconds
        step_duration_sec = (timestamps[1] - timestamps[0]).total_seconds() if len(timestamps) > 1 else pd.Timedelta(self.frequency).total_seconds()

        # Build full MultiIndex skeleton: (timestamp x target_regions)
        grid_index = pd.MultiIndex.from_product(
            [timestamps, target_regions],
            names=["timestamp", "region_slug"]
        )
        total_rows = len(grid_index)

        # Pre-allocate arrays
        active_duration_arr = np.zeros(total_rows, dtype=np.float64)
        alert_onset_arr = np.zeros(total_rows, dtype=np.int8)
        alerts_count_arr = np.zeros(total_rows, dtype=np.int16)

        # Map timestamps to index position
        ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}
        region_to_offset = {r: i for i, r in enumerate(target_regions)}
        num_regions = len(target_regions)

        # Filter raw_df to relevant regions and window
        if not raw_df.empty:
            valid_mask = (
                raw_df["region_slug"].isin(target_regions)
                & (raw_df["finished_at"] >= grid_start)
                & (raw_df["started_at"] <= grid_end)
            )
            filtered_raw = raw_df[valid_mask].copy()
        else:
            filtered_raw = pd.DataFrame(columns=raw_df.columns)

        # Process per region
        for region_slug in target_regions:
            region_offset = region_to_offset[region_slug]
            region_alerts = filtered_raw[filtered_raw["region_slug"] == region_slug]

            # 1. Onsets and alert count (based on raw alert starts)
            for _, row in region_alerts.iterrows():
                start_ts = row["started_at"]
                slot_start = start_ts.floor(self.frequency)
                if slot_start in ts_to_idx:
                    t_idx = ts_to_idx[slot_start]
                    row_idx = t_idx * num_regions + region_offset
                    alert_onset_arr[row_idx] = 1

            # 2. Overlap active duration (merging overlapping alerts first)
            raw_intervals = [
                (max(grid_start, r["started_at"]), min(grid_end, r["finished_at"]))
                for _, r in region_alerts.iterrows()
                if r["finished_at"] > grid_start and r["started_at"] < grid_end
            ]
            merged_intervals = self._merge_overlapping_intervals(raw_intervals)

            for alert_start, alert_end in merged_intervals:
                # Find all overlapping slots
                cur_slot = alert_start.floor(self.frequency)
                last_slot = alert_end.floor(self.frequency)

                while cur_slot <= last_slot and cur_slot < grid_end:
                    if cur_slot >= grid_start:
                        slot_next = cur_slot + pd.Timedelta(self.frequency)
                        overlap_start = max(alert_start, cur_slot)
                        overlap_end = min(alert_end, slot_next)
                        overlap_sec = max(0.0, (overlap_end - overlap_start).total_seconds())

                        if overlap_sec > 0 and cur_slot in ts_to_idx:
                            t_idx = ts_to_idx[cur_slot]
                            row_idx = t_idx * num_regions + region_offset
                            active_duration_arr[row_idx] += overlap_sec
                            alerts_count_arr[row_idx] += 1

                    cur_slot += pd.Timedelta(self.frequency)

        # Compute derived binary and ratio targets
        alert_active_arr = (active_duration_arr >= self.min_alert_overlap_sec).astype(np.int8)
        active_ratio_arr = np.clip(active_duration_arr / step_duration_sec, 0.0, 1.0)

        # Region IDs array
        region_ids_map = {r: get_region(r).id for r in target_regions}
        region_ids_arr = np.tile([region_ids_map[r] for r in target_regions], len(timestamps))

        # Construct final tidy DataFrame
        timestamps_expanded = np.repeat(timestamps, num_regions)
        regions_expanded = np.tile(target_regions, len(timestamps))

        grid_df = pd.DataFrame({
            "timestamp": timestamps_expanded,
            "region_slug": regions_expanded,
            "region_id": region_ids_arr,
            "alert_active": alert_active_arr,
            "alert_onset": alert_onset_arr,
            "active_duration_sec": active_duration_arr,
            "active_ratio": active_ratio_arr,
            "alerts_count": alerts_count_arr,
        })

        return grid_df.sort_values(by=["timestamp", "region_slug"]).reset_index(drop=True)

    def convert_and_save(
        self,
        raw_df: pd.DataFrame,
        output_path: Union[str, Path],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        regions: Optional[List[str]] = None,
    ) -> Path:
        """Convert intervals to continuous grid and persist to Parquet."""
        grid_df = self.convert(raw_df, start_date=start_date, end_date=end_date, regions=regions)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        grid_df.to_parquet(output_path, index=False)
        return output_path
