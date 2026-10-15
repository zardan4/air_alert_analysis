"""Four naive baseline models for air raid alert prediction benchmarking."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.features.spatial import REGION_NEIGHBORS, PRIMARY_INFLOW_REGIONS, get_neighbors


class BaselineModel(ABC):
    """Abstract interface for naive baseline predictors."""

    name: str

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> "BaselineModel":
        """Learn any parameters from training data (some baselines are stateless)."""
        pass

    @abstractmethod
    def predict_proba(self, df: pd.DataFrame, horizon_hours: int) -> np.ndarray:
        """Return predicted probability of alert in the next H hours for each row."""
        pass


class PersistenceBaseline(BaselineModel):
    """Baseline 1: State Inertia — predict current state persists into the future.

    p̂(alert in [t+1, t+H]) = 1.0 if alert_active(t) == 1 else 0.0

    This baseline cannot predict transitions; it tests whether the ML model
    captures onset/termination dynamics beyond mere state momentum.
    """

    name = "Persistence"

    def fit(self, train_df: pd.DataFrame) -> "PersistenceBaseline":
        return self  # Stateless

    def predict_proba(self, df: pd.DataFrame, horizon_hours: int) -> np.ndarray:
        if "feat_active_lag_0" in df.columns:
            return df["feat_active_lag_0"].fillna(0.0).values.astype(np.float64)
        elif "alert_active" in df.columns:
            return df["alert_active"].fillna(0.0).values.astype(np.float64)
        else:
            raise KeyError("Neither 'feat_active_lag_0' nor 'alert_active' found in DataFrame")


class ClimatologyBaseline(BaselineModel):
    """Baseline 2: Diurnal Climatology — historical base rate per (oblast, hour, day_of_week).

    p̂(i, d, h) = Σ historical alerts in region i at (day_of_week=d, hour=h)
                  / total observations at (d, h)

    Captures structural periodicity (night-time drone swarms, weekday/weekend patterns).
    Requires training data to estimate the conditional frequency table.
    """

    name = "Climatology"

    def __init__(self, timezone: str = "Europe/Kyiv", smoothing_alpha: float = 0.01):
        self.timezone = timezone
        self.smoothing_alpha = smoothing_alpha
        self._lookup: Dict[Tuple[str, int, int], float] = {}
        self._global_rate: float = 0.0

    def fit(self, train_df: pd.DataFrame) -> "ClimatologyBaseline":
        df = train_df.copy()
        ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(self.timezone)
        df["_hour"] = ts.dt.hour
        df["_dow"] = ts.dt.dayofweek

        # Global fallback rate
        self._global_rate = float(df["alert_active"].mean()) if len(df) > 0 else 0.0

        # Build stratified lookup table
        grouped = df.groupby(["region_slug", "_dow", "_hour"])["alert_active"]
        for (region, dow, hour), series in grouped:
            n = len(series)
            positive = series.sum()
            # Laplace smoothing
            rate = (positive + self.smoothing_alpha) / (n + 2 * self.smoothing_alpha)
            self._lookup[(region, int(dow), int(hour))] = float(rate)

        return self

    def predict_proba(self, df: pd.DataFrame, horizon_hours: int) -> np.ndarray:
        ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(self.timezone)
        hours = ts.dt.hour.values
        dows = ts.dt.dayofweek.values
        regions = df["region_slug"].values

        probas = np.empty(len(df), dtype=np.float64)
        for i in range(len(df)):
            # Average base rate across the next H hours to account for horizon span
            rates = []
            for h_offset in range(1, horizon_hours + 1):
                future_hour = (hours[i] + h_offset) % 24
                key = (regions[i], int(dows[i]), int(future_hour))
                rates.append(self._lookup.get(key, self._global_rate))

            # P(at least one alert in H slots) = 1 - Π(1 - p_h)
            prob_none = 1.0
            for r in rates:
                prob_none *= (1.0 - r)
            probas[i] = 1.0 - prob_none

        return probas


class RollingRateBaseline(BaselineModel):
    """Baseline 3: Trailing Rolling Window — empirical alert frequency over recent 7 days.

    p̂(i, t) = (alert hours in region i in [t - 7d, t]) / (7 × 24)

    Accounts for temporal non-stationarity (escalation periods vs. quiet weeks)
    without structural diurnal modeling.
    """

    name = "RollingRate7d"

    def __init__(self, lookback_hours: int = 168):
        self.lookback_hours = lookback_hours

    def fit(self, train_df: pd.DataFrame) -> "RollingRateBaseline":
        return self  # Uses rolling features already computed

    def predict_proba(self, df: pd.DataFrame, horizon_hours: int) -> np.ndarray:
        col_name = f"feat_rolling_active_rate_{self.lookback_hours}h"
        if col_name in df.columns:
            return df[col_name].fillna(0.0).values.astype(np.float64)

        # Fallback: compute from raw data
        if "alert_active" not in df.columns:
            raise KeyError(f"Neither '{col_name}' nor 'alert_active' found")

        df_sorted = df.sort_values(by=["region_slug", "timestamp"]).copy()
        grouped = df_sorted.groupby("region_slug")["alert_active"]
        rolling_rate = grouped.rolling(window=self.lookback_hours, min_periods=1).mean()
        return rolling_rate.reset_index(0, drop=True).reindex(df.index).fillna(0.0).values.astype(np.float64)


class SpatialInflowBaseline(BaselineModel):
    """Baseline 4: Upstream Spatial Cascading Heuristic.

    Alerts propagate along ballistic/drone flight paths from border entry regions.
    If any immediate neighbor (or upstream inflow region) has an active alert at t,
    the downstream region's probability is set to a calibrated conditional probability.

    p̂(j, t) = 1 - Π_{i ∈ neighbors(j)} (1 - w_i × active_i(t))

    where w_i is the learned transition probability P(alert_j | active_i) from training.
    """

    name = "SpatialInflow"

    def __init__(self):
        self._transition_probs: Dict[Tuple[str, str], float] = {}
        self._fallback_prob: float = 0.0

    def fit(self, train_df: pd.DataFrame) -> "SpatialInflowBaseline":
        """Learn pairwise conditional transition probabilities from training data."""
        df = train_df.copy()

        self._fallback_prob = float(df["alert_active"].mean()) if len(df) > 0 else 0.0

        # Pivot to get contemporaneous status
        pivoted = df.pivot(index="timestamp", columns="region_slug", values="alert_active").fillna(0)

        for target_region in pivoted.columns:
            neighbors = get_neighbors(target_region)
            valid_neighbors = [n for n in neighbors if n in pivoted.columns]

            for neighbor in valid_neighbors:
                # P(target active | neighbor active)
                neighbor_active_mask = pivoted[neighbor] == 1
                if neighbor_active_mask.sum() > 10:
                    conditional_prob = float(
                        pivoted.loc[neighbor_active_mask, target_region].mean()
                    )
                else:
                    conditional_prob = self._fallback_prob
                self._transition_probs[(neighbor, target_region)] = conditional_prob

        return self

    def predict_proba(self, df: pd.DataFrame, horizon_hours: int) -> np.ndarray:
        # Pivot for contemporaneous neighbor status
        pivoted = df.pivot(index="timestamp", columns="region_slug", values="alert_active").fillna(0)

        probas = np.full(len(df), self._fallback_prob, dtype=np.float64)
        regions = df["region_slug"].values
        timestamps = df["timestamp"].values

        ts_to_idx = {ts: i for i, ts in enumerate(pivoted.index)}

        for i in range(len(df)):
            region = regions[i]
            ts = timestamps[i]
            ts_key = pd.Timestamp(ts)

            if ts_key not in ts_to_idx:
                continue

            neighbors = get_neighbors(region)
            valid_neighbors = [n for n in neighbors if n in pivoted.columns]

            if not valid_neighbors:
                continue

            # p = 1 - Π(1 - w_ij * active_j)
            prob_none = 1.0
            for neighbor in valid_neighbors:
                if pivoted.loc[ts_key, neighbor] == 1:
                    w = self._transition_probs.get((neighbor, region), self._fallback_prob)
                    prob_none *= (1.0 - w)

            probas[i] = 1.0 - prob_none

        return probas


def get_all_baselines(timezone: str = "Europe/Kyiv") -> List[BaselineModel]:
    """Factory returning all four baseline models."""
    return [
        PersistenceBaseline(),
        ClimatologyBaseline(timezone=timezone),
        RollingRateBaseline(),
        SpatialInflowBaseline(),
    ]
