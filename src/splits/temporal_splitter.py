"""Strict anti-leakage temporal splitter with purging, embargoing, and walk-forward CV."""

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.config import load_config


class TemporalLeakageError(ValueError):
    """Raised when a temporal split violates causal anti-leakage constraints."""
    pass


@dataclass(frozen=True)
class SplitIndices:
    """Stores integer index positions and metadata for a single CV fold."""
    fold_idx: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    purged_rows_count: int
    separation_buffer_hours: float


@dataclass(frozen=True)
class HoldoutSplitResult:
    """Stores train/validation and test DataFrames with purge metadata."""
    train_val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_val_cutoff: pd.Timestamp
    test_start: pd.Timestamp
    purged_rows_count: int
    separation_buffer_hours: float


def verify_temporal_integrity(
    train_timestamps: Union[pd.Series, pd.DatetimeIndex, np.ndarray],
    eval_timestamps: Union[pd.Series, pd.DatetimeIndex, np.ndarray],
    min_separation_hours: float,
    context_label: str = "Split",
) -> None:
    """Mathematically verify that no temporal leakage exists between train and evaluation sets.
    
    Enforces:
    1. Zero index or timestamp overlap between train and eval.
    2. Strict causal ordering: all train timestamps precede all eval timestamps.
    3. Separation gap: (min(eval) - max(train)) >= min_separation_hours.
    """
    train_ts = pd.to_datetime(train_timestamps, utc=True)
    eval_ts = pd.to_datetime(eval_timestamps, utc=True)

    if len(train_ts) == 0 or len(eval_ts) == 0:
        raise TemporalLeakageError(f"[{context_label}] Empty train or evaluation set provided.")

    max_train = train_ts.max()
    min_eval = eval_ts.min()

    # 1. Monotonic chronological ordering
    if max_train >= min_eval:
        overlap = set(train_ts).intersection(set(eval_ts))
        raise TemporalLeakageError(
            f"[{context_label}] Strict temporal ordering violated! "
            f"Max train timestamp ({max_train}) >= Min eval timestamp ({min_eval}). "
            f"Overlapping timestamps count: {len(overlap)}."
        )

    # 2. Separation buffer (Purge + Embargo) verification
    gap_hours = (min_eval - max_train).total_seconds() / 3600.0
    if gap_hours < min_separation_hours:
        raise TemporalLeakageError(
            f"[{context_label}] Separation buffer violated! "
            f"Required separation: {min_separation_hours:.2f} hours, "
            f"Actual gap: {gap_hours:.2f} hours (between {max_train} and {min_eval})."
        )


class PurgedWalkForwardSplitter:
    """Strict anti-leakage temporal cross-validation and holdout splitter.
    
    Implements:
    - Dedicated chronological holdout test set (most recent chronological period).
    - Expanding-window walk-forward cross-validation.
    - Purging: Eliminates training records whose forward target lookahead window [t, t+H]
      intersects the evaluation set.
    - Embargoing: Enforces an additional buffer to eliminate autoregressive / rolling feature bleed.
    - Atomicity across regions: All 25 regions at timestamp t are grouped together atomically.
    """

    def __init__(
        self,
        max_horizon_hours: Optional[int] = None,
        embargo_hours: Optional[int] = None,
        test_window_days: Optional[int] = None,
        val_window_days: int = 14,
        min_train_days: int = 30,
        step_days: Optional[int] = None,
    ):
        config = load_config()
        self.max_horizon_hours = (
            max_horizon_hours
            if max_horizon_hours is not None
            else max(config.backtesting.horizons_hours)
        )
        self.embargo_hours = (
            embargo_hours
            if embargo_hours is not None
            else config.backtesting.embargo_hours
        )
        self.test_window_days = (
            test_window_days
            if test_window_days is not None
            else config.backtesting.test_window_days
        )
        self.val_window_days = val_window_days
        self.min_train_days = min_train_days
        self.step_days = step_days if step_days is not None else val_window_days

        # Total separation buffer required between latest train sample and earliest eval sample
        self.min_separation_hours = float(self.max_horizon_hours + self.embargo_hours)

    def create_holdout_split(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
    ) -> HoldoutSplitResult:
        """Partition continuous grid into (train_val) and (test) sets with strict purging.

        The test set covers the most recent `test_window_days`.
        Any observations within the purge/embargo buffer preceding test start are strictly dropped.
        """
        if timestamp_col not in df.columns:
            raise KeyError(f"Timestamp column '{timestamp_col}' not found in DataFrame")

        timestamps = pd.to_datetime(df[timestamp_col], utc=True)
        max_ts = timestamps.max()
        test_start = max_ts - pd.Timedelta(days=self.test_window_days)

        # Buffer: any train sample whose lookahead [t, t + H] touches test or embargo is purged
        buffer_duration = pd.Timedelta(hours=self.min_separation_hours)
        train_val_cutoff = test_start - buffer_duration

        train_val_mask = timestamps <= train_val_cutoff
        test_mask = timestamps >= test_start
        purged_mask = (timestamps > train_val_cutoff) & (timestamps < test_start)

        train_val_df = df[train_val_mask].copy().reset_index(drop=True)
        test_df = df[test_mask].copy().reset_index(drop=True)
        purged_rows_count = int(purged_mask.sum())

        # Mathematical verification
        verify_temporal_integrity(
            train_timestamps=train_val_df[timestamp_col],
            eval_timestamps=test_df[timestamp_col],
            min_separation_hours=self.min_separation_hours,
            context_label="HoldoutSplit",
        )

        return HoldoutSplitResult(
            train_val_df=train_val_df,
            test_df=test_df,
            train_val_cutoff=train_val_cutoff,
            test_start=test_start,
            purged_rows_count=purged_rows_count,
            separation_buffer_hours=self.min_separation_hours,
        )

    def create_cv_splits(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
    ) -> List[SplitIndices]:
        """Generate expanding-window walk-forward folds with purging and embargoing.

        For each fold:
        - Train window expands: [min_ts, train_cutoff]
        - Purge buffer: (train_cutoff, val_start) where (val_start - train_cutoff) >= separation_hours
        - Validation window: [val_start, val_end)
        """
        if timestamp_col not in df.columns:
            raise KeyError(f"Timestamp column '{timestamp_col}' not found in DataFrame")

        timestamps = pd.to_datetime(df[timestamp_col], utc=True)
        unique_ts = np.sort(timestamps.unique())

        min_ts = unique_ts[0]
        max_ts = unique_ts[-1]
        total_span = (max_ts - min_ts).total_seconds() / 86400.0

        min_required_span = self.min_train_days + (self.min_separation_hours / 24.0) + self.val_window_days
        if total_span < min_required_span:
            raise ValueError(
                f"Insufficient data span for purged CV: total {total_span:.1f} days, "
                f"need at least {min_required_span:.1f} days."
            )

        splits: List[SplitIndices] = []
        buffer_duration = pd.Timedelta(hours=self.min_separation_hours)
        val_duration = pd.Timedelta(days=self.val_window_days)
        step_duration = pd.Timedelta(days=self.step_days)

        # First validation start is after min_train_days + buffer
        first_val_start = min_ts + pd.Timedelta(days=self.min_train_days) + buffer_duration
        current_val_start = first_val_start
        fold_idx = 1

        # Precompute timestamp mapping to dataframe indices for high performance
        ts_series = pd.Series(timestamps.values, index=np.arange(len(df)))

        while current_val_start + val_duration <= max_ts + pd.Timedelta(hours=1):
            current_val_end = current_val_start + val_duration
            current_train_cutoff = current_val_start - buffer_duration

            train_mask = (timestamps <= current_train_cutoff).values
            val_mask = ((timestamps >= current_val_start) & (timestamps < current_val_end)).values
            purged_mask = ((timestamps > current_train_cutoff) & (timestamps < current_val_start)).values

            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]
            purged_count = int(np.sum(purged_mask))

            if len(train_idx) == 0 or len(val_idx) == 0:
                break

            # Verify integrity on every generated fold
            verify_temporal_integrity(
                train_timestamps=timestamps.iloc[train_idx],
                eval_timestamps=timestamps.iloc[val_idx],
                min_separation_hours=self.min_separation_hours,
                context_label=f"CV_Fold_{fold_idx}",
            )

            splits.append(
                SplitIndices(
                    fold_idx=fold_idx,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    train_start=timestamps.iloc[train_idx].min(),
                    train_end=timestamps.iloc[train_idx].max(),
                    val_start=timestamps.iloc[val_idx].min(),
                    val_end=timestamps.iloc[val_idx].max(),
                    purged_rows_count=purged_count,
                    separation_buffer_hours=self.min_separation_hours,
                )
            )

            fold_idx += 1
            current_val_start += step_duration

        if not splits:
            raise ValueError("No valid CV folds could be constructed with current parameters.")

        return splits

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        groups: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Scikit-learn compatible generator yielding (train_idx, val_idx) for cross-validation."""
        splits = self.create_cv_splits(X)
        for s in splits:
            yield s.train_idx, s.val_idx
