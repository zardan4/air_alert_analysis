"""Point-in-time causal feature engineering and multi-horizon target generator."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.features.spatial import REGION_NEIGHBORS, PRIMARY_INFLOW_REGIONS, get_neighbors
from src.utils.regions import ALL_REGION_SLUGS


@dataclass(frozen=True)
class FeatureEngineeringResult:
    """Container holding feature-engineered DataFrame with metadata."""
    data: pd.DataFrame
    feature_columns: List[str]
    target_columns: List[str]
    horizons_hours: List[int]


class PointInTimeFeatureEngineer:
    """Constructs causal point-in-time features (F_t) and forward multi-horizon targets.
    
    Guarantees:
    - All features at row (i, t) use strictly observations <= t.
    - Multi-horizon targets look ahead strictly into [t + 1, t + H].
    - Zero future lookahead leakage into any feature column.
    """

    def __init__(
        self,
        horizons_hours: Optional[List[int]] = None,
        lags_hours: Optional[List[int]] = None,
        rolling_windows_hours: Optional[List[int]] = None,
        timezone: str = "Europe/Kyiv",
    ):
        self.horizons_hours = sorted(horizons_hours or [1, 2, 3])
        self.lags_hours = sorted(lags_hours or [0, 1, 2, 3, 6, 12, 24])
        self.rolling_windows_hours = sorted(rolling_windows_hours or [6, 24, 72, 168])
        self.timezone = timezone

    def create_targets(
        self,
        df: pd.DataFrame,
        horizons_hours: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Create forward-looking multi-horizon targets for each region.

        For each horizon H:
        - target_active_{H}h: 1 if alert_active == 1 at any step in [t + 1, t + H]
        - target_onset_{H}h:  1 if alert_onset == 1 at any step in [t + 1, t + H]
        """
        horizons = horizons_hours or self.horizons_hours
        df = df.sort_values(by=["region_slug", "timestamp"]).copy()

        grouped = df.groupby("region_slug")

        for h in horizons:
            # Active in (t+1 .. t+h): take backward rolling max shifted backward
            # In pandas: shift(-1) is t+1. Rolling over next h steps:
            # We reverse, roll forward, reverse, or shift up to h steps.
            active_future_shifts = [grouped["alert_active"].shift(-k) for k in range(1, h + 1)]
            onset_future_shifts = [grouped["alert_onset"].shift(-k) for k in range(1, h + 1)]

            df[f"target_active_{h}h"] = pd.concat(active_future_shifts, axis=1).max(axis=1)
            df[f"target_onset_{h}h"] = pd.concat(onset_future_shifts, axis=1).max(axis=1)

        return df

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract cyclical calendar, diurnal, and local Kyiv war-time flight window features."""
        df = df.copy()
        ts = pd.to_datetime(df["timestamp"], utc=True)
        # Convert to local Ukrainian time for proper diurnal modeling
        ts_kyiv = ts.dt.tz_convert(self.timezone)

        hours = ts_kyiv.dt.hour
        dayofweek = ts_kyiv.dt.dayofweek

        # Cyclical diurnal encoding
        df["feat_hour_sin"] = np.sin(2.0 * np.pi * hours / 24.0)
        df["feat_hour_cos"] = np.cos(2.0 * np.pi * hours / 24.0)
        df["feat_day_sin"] = np.sin(2.0 * np.pi * dayofweek / 7.0)
        df["feat_day_cos"] = np.cos(2.0 * np.pi * dayofweek / 7.0)
        df["feat_is_weekend"] = (dayofweek >= 5).astype(np.int8)

        # High-risk Shahed night strike window (23:00 - 06:00 Kyiv time)
        df["feat_is_night_strike_window"] = ((hours >= 23) | (hours <= 6)).astype(np.int8)

        return df

    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create causal autoregressive lags strictly from timestamps <= t."""
        df = df.sort_values(by=["region_slug", "timestamp"]).copy()
        grouped = df.groupby("region_slug")

        for lag in self.lags_hours:
            if lag == 0:
                df[f"feat_active_lag_0"] = df["alert_active"].astype(np.float32)
                df[f"feat_onset_lag_0"] = df["alert_onset"].astype(np.float32)
                df[f"feat_ratio_lag_0"] = df["active_ratio"].astype(np.float32)
            else:
                df[f"feat_active_lag_{lag}"] = grouped["alert_active"].shift(lag).astype(np.float32)
                df[f"feat_onset_lag_{lag}"] = grouped["alert_onset"].shift(lag).astype(np.float32)
                if lag <= 3:
                    df[f"feat_ratio_lag_{lag}"] = grouped["active_ratio"].shift(lag).astype(np.float32)

        return df

    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling lookback statistics strictly closed on the left/present (<= t)."""
        df = df.sort_values(by=["region_slug", "timestamp"]).copy()
        grouped = df.groupby("region_slug")

        for w in self.rolling_windows_hours:
            # Trailing alert hours (sum over window of hourly slots)
            rolling_active = grouped["alert_active"].rolling(window=w, min_periods=1).sum().reset_index(0, drop=True)
            df[f"feat_rolling_active_hours_{w}h"] = rolling_active.astype(np.float32)
            df[f"feat_rolling_active_rate_{w}h"] = (rolling_active / float(w)).astype(np.float32)

            # Trailing onsets count
            rolling_onsets = grouped["alert_onset"].rolling(window=w, min_periods=1).sum().reset_index(0, drop=True)
            df[f"feat_rolling_onsets_count_{w}h"] = rolling_onsets.astype(np.float32)

        # Elapsed hours since last alert onset
        def _calc_hours_since_onset(series: pd.Series) -> pd.Series:
            onset_indices = series.index[series == 1]
            if len(onset_indices) == 0:
                return pd.Series(999.0, index=series.index)
            # Efficient forward fill elapsed counter
            is_onset = series == 1
            counter = (~is_onset).astype(int)
            # Cumulative count since reset
            group_key = is_onset.cumsum()
            elapsed = counter.groupby(group_key).cumsum()
            # Before first onset, cap at 999
            first_onset_idx = is_onset.idxmax() if is_onset.any() else None
            if first_onset_idx is not None:
                elapsed.loc[:first_onset_idx] = 999.0
                elapsed.loc[first_onset_idx] = 0.0
            return elapsed.astype(np.float32)

        df["feat_hours_since_last_onset"] = (
            grouped["alert_onset"].transform(_calc_hours_since_onset).astype(np.float32)
        )

        return df

    def create_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute cross-regional and neighbor alert statuses at cutoff timestamp t."""
        df = df.copy()

        # 1. Nationwide activity at timestamp t
        ts_group = df.groupby("timestamp")
        nationwide_active = ts_group["alert_active"].sum()
        nationwide_onsets = ts_group["alert_onset"].sum()

        df["feat_nationwide_active_count"] = df["timestamp"].map(nationwide_active).astype(np.float32)
        df["feat_nationwide_active_ratio"] = (df["feat_nationwide_active_count"] / float(len(ALL_REGION_SLUGS))).astype(np.float32)
        df["feat_nationwide_onsets_count"] = df["timestamp"].map(nationwide_onsets).astype(np.float32)

        # 2. Border / Inflow regions active at timestamp t
        inflow_df = df[df["region_slug"].isin(PRIMARY_INFLOW_REGIONS)]
        inflow_active = inflow_df.groupby("timestamp")["alert_active"].sum()
        df["feat_inflow_border_active_count"] = df["timestamp"].map(inflow_active).fillna(0.0).astype(np.float32)

        # 3. Immediate geographical neighbors active status at timestamp t
        # Vectorized graph convolution using adjacency matrix
        pivoted_active = df.pivot(index="timestamp", columns="region_slug", values="alert_active").fillna(0)
        regions_list = list(pivoted_active.columns)
        reg_to_idx = {r: i for i, r in enumerate(regions_list)}

        n_reg = len(regions_list)
        adj_matrix = np.zeros((n_reg, n_reg), dtype=np.float32)
        degree_vector = np.ones(n_reg, dtype=np.float32)

        for slug in regions_list:
            i = reg_to_idx[slug]
            neighbors = [n for n in get_neighbors(slug) if n in reg_to_idx]
            for n in neighbors:
                j = reg_to_idx[n]
                adj_matrix[i, j] = 1.0
            if neighbors:
                degree_vector[i] = float(len(neighbors))

        # Matrix multiplication: (T x N) @ (N x N)^T -> (T x N)
        active_matrix = pivoted_active.values.astype(np.float32)
        counts_matrix = active_matrix @ adj_matrix.T
        ratios_matrix = counts_matrix / degree_vector[np.newaxis, :]

        neighbor_counts_df = pd.DataFrame(counts_matrix, index=pivoted_active.index, columns=regions_list)
        neighbor_ratios_df = pd.DataFrame(ratios_matrix, index=pivoted_active.index, columns=regions_list)

        # Map back to df
        counts_stacked = neighbor_counts_df.stack().rename("feat_neighbors_active_count")
        ratios_stacked = neighbor_ratios_df.stack().rename("feat_neighbors_active_ratio")

        df = df.set_index(["timestamp", "region_slug"])
        df["feat_neighbors_active_count"] = counts_stacked
        df["feat_neighbors_active_ratio"] = ratios_stacked
        df = df.reset_index()

        return df

    def transform(
        self,
        grid_df: pd.DataFrame,
        include_targets: bool = True,
        drop_na_targets: bool = False,
    ) -> FeatureEngineeringResult:
        """Run complete causal feature pipeline on continuous grid DataFrame.

        Args:
            grid_df: Continuous uniform grid from IntervalToGridConverter.
            include_targets: If True, computes multi-horizon targets.
            drop_na_targets: If True, drops trailing rows with NaN future targets.

        Returns:
            FeatureEngineeringResult containing transformed DataFrame and metadata.
        """
        df = grid_df.copy()

        # Step 1: Temporal & calendar features
        df = self.create_temporal_features(df)

        # Step 2: Autoregressive lag features
        df = self.create_lag_features(df)

        # Step 3: Rolling lookback statistics
        df = self.create_rolling_features(df)

        # Step 4: Spatial and cross-region features
        df = self.create_spatial_features(df)

        target_columns: List[str] = []
        if include_targets:
            df = self.create_targets(df)
            for h in self.horizons_hours:
                target_columns.extend([f"target_active_{h}h", f"target_onset_{h}h"])

            if drop_na_targets:
                df = df.dropna(subset=target_columns).reset_index(drop=True)

        feature_columns = [col for col in df.columns if col.startswith("feat_")]

        # Sort chronologically by timestamp, then region
        df = df.sort_values(by=["timestamp", "region_slug"]).reset_index(drop=True)

        return FeatureEngineeringResult(
            data=df,
            feature_columns=feature_columns,
            target_columns=target_columns,
            horizons_hours=self.horizons_hours,
        )
