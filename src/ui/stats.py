"""User-friendly statistical aggregations for civil and operational air raid analytics."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.utils.regions import get_region_hromadas

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_ABBR = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def get_heatmap_matrix(
    grid_df: pd.DataFrame,
    region_slug: Optional[str] = None,
    timezone: str = "Europe/Kyiv",
) -> pd.DataFrame:
    """Compute 7x24 empirical alert probability matrix (Day of Week x Hour of Day).

    Returns DataFrame indexed by day names (Mon..Sun) with columns '00:00'..'23:00'.
    Values are percentages (0.0 to 1.0).
    """
    df = grid_df if region_slug is None else grid_df[grid_df["region_slug"] == region_slug]
    if df.empty:
        return pd.DataFrame(0.0, index=DAY_ABBR, columns=[f"{h:02d}:00" for h in range(24)])

    ts_local = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(timezone)
    df_calc = pd.DataFrame({
        "dow": ts_local.dt.dayofweek,
        "hour": ts_local.dt.hour,
        "alert_active": df["alert_active"].values,
    })

    grouped = df_calc.groupby(["dow", "hour"])["alert_active"].mean().unstack(level="hour", fill_value=0.0)

    # Reindex to ensure all 7 days and 24 hours exist
    grouped = grouped.reindex(index=range(7), columns=range(24), fill_value=0.0)
    grouped.index = DAY_ABBR
    grouped.columns = [f"{h:02d}:00" for h in range(24)]

    return grouped


def get_peak_and_safe_windows(heatmap_df: pd.DataFrame) -> Dict[str, Any]:
    """Identify the top peak threat hours and safest hours from the 7x24 heatmap."""
    # Average across all 7 days for each hour
    hourly_mean = heatmap_df.mean(axis=0)

    peak_hour = str(hourly_mean.idxmax())
    peak_rate = float(hourly_mean.max())

    safe_hour = str(hourly_mean.idxmin())
    safe_rate = float(hourly_mean.min())

    # Find highest risk day
    daily_mean = heatmap_df.mean(axis=1)
    peak_day = str(daily_mean.idxmax())
    peak_day_rate = float(daily_mean.max())

    return {
        "peak_hour": peak_hour,
        "peak_hourly_rate": peak_rate,
        "safe_hour": safe_hour,
        "safe_hourly_rate": safe_rate,
        "peak_day": peak_day,
        "peak_day_rate": peak_day_rate,
    }


def get_day_of_week_profile(
    raw_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    region_slug: Optional[str] = None,
    timezone: str = "Europe/Kyiv",
) -> pd.DataFrame:
    """Compute day-of-week profile: average alert count, average duration, and downtime.

    Returns DataFrame indexed by day name with columns:
        ['Average Alerts / Day', 'Average Duration (mins)', 'Median Duration (mins)', 'Active Hours / Day']
    """
    raw_sub = raw_df if region_slug is None else raw_df[raw_df["region_slug"] == region_slug]
    grid_sub = grid_df if region_slug is None else grid_df[grid_df["region_slug"] == region_slug]

    if raw_sub.empty or grid_sub.empty:
        return pd.DataFrame(index=DAY_NAMES, columns=[
            "Average Alerts / Day", "Average Duration (mins)", "Median Duration (mins)", "Active Hours / Day"
        ]).fillna(0.0)

    # Raw alerts duration in minutes
    durations_min = (
        pd.to_datetime(raw_sub["finished_at"], utc=True) - pd.to_datetime(raw_sub["started_at"], utc=True)
    ).dt.total_seconds() / 60.0

    raw_local_ts = pd.to_datetime(raw_sub["started_at"], utc=True).dt.tz_convert(timezone)
    raw_with_dow = pd.DataFrame({
        "dow": raw_local_ts.dt.dayofweek,
        "date": raw_local_ts.dt.date,
        "duration_min": durations_min.values,
    })

    # Grid hours for total downtime
    grid_local_ts = pd.to_datetime(grid_sub["timestamp"], utc=True).dt.tz_convert(timezone)
    grid_with_dow = pd.DataFrame({
        "dow": grid_local_ts.dt.dayofweek,
        "date": grid_local_ts.dt.date,
        "alert_active": grid_sub["alert_active"].values,
    })

    rows = []
    # Distinct dates per day of week to normalize averages
    for dow in range(7):
        dow_raw = raw_with_dow[raw_with_dow["dow"] == dow]
        dow_grid = grid_with_dow[grid_with_dow["dow"] == dow]

        num_dates = max(1, dow_grid["date"].nunique())

        avg_alerts_per_day = len(dow_raw) / num_dates
        avg_dur = float(dow_raw["duration_min"].mean()) if not dow_raw.empty else 0.0
        med_dur = float(dow_raw["duration_min"].median()) if not dow_raw.empty else 0.0
        active_hours_per_day = float(dow_grid["alert_active"].sum()) / num_dates

        rows.append({
            "Day": DAY_NAMES[dow],
            "Average Alerts / Day": round(avg_alerts_per_day, 1),
            "Average Duration (mins)": round(avg_dur, 1),
            "Median Duration (mins)": round(med_dur, 1),
            "Active Hours / Day": round(active_hours_per_day, 1),
        })

    profile_df = pd.DataFrame(rows).set_index("Day")
    return profile_df


def get_duration_spectrum(
    raw_df: pd.DataFrame,
    region_slug: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute alert duration spectrum: short (<30m), moderate (30m-2h), extended (>2h)."""
    raw_sub = raw_df if region_slug is None else raw_df[raw_df["region_slug"] == region_slug]
    if raw_sub.empty:
        return {
            "median_min": 0.0,
            "mean_min": 0.0,
            "pct_short": 0.0,
            "pct_moderate": 0.0,
            "pct_extended": 0.0,
            "max_hours": 0.0,
            "total_alerts": 0,
        }

    durations_min = (
        pd.to_datetime(raw_sub["finished_at"], utc=True) - pd.to_datetime(raw_sub["started_at"], utc=True)
    ).dt.total_seconds() / 60.0

    total = len(durations_min)
    short_count = (durations_min < 30.0).sum()
    moderate_count = ((durations_min >= 30.0) & (durations_min <= 120.0)).sum()
    extended_count = (durations_min > 120.0).sum()

    return {
        "median_min": float(durations_min.median()),
        "mean_min": float(durations_min.mean()),
        "pct_short": float(short_count / total),
        "pct_moderate": float(moderate_count / total),
        "pct_extended": float(extended_count / total),
        "max_hours": float(durations_min.max() / 60.0),
        "total_alerts": total,
    }


def get_calm_and_downtime_stats(
    raw_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    region_slug: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute quiet window between alerts, current calm streak, and daily shelter time."""
    raw_sub = raw_df if region_slug is None else raw_df[raw_df["region_slug"] == region_slug]
    grid_sub = grid_df if region_slug is None else grid_df[grid_df["region_slug"] == region_slug]

    if raw_sub.empty or grid_sub.empty:
        return {
            "avg_calm_hours": 0.0,
            "median_calm_hours": 0.0,
            "max_calm_hours": 0.0,
            "current_calm_hours": 0.0,
            "is_active_now": False,
            "daily_shelter_hours": 0.0,
        }

    # Sort chronologically
    sorted_raw = raw_sub.sort_values(by="started_at").copy()
    sorted_raw["started_at"] = pd.to_datetime(sorted_raw["started_at"], utc=True)
    sorted_raw["finished_at"] = pd.to_datetime(sorted_raw["finished_at"], utc=True)

    # Time between alerts: next start minus current end
    calm_intervals = (
        sorted_raw["started_at"].iloc[1:].values - sorted_raw["finished_at"].iloc[:-1].values
    )
    calm_hours = pd.to_timedelta(calm_intervals).total_seconds() / 3600.0
    calm_hours_valid = calm_hours[calm_hours > 0]
    avg_calm = float(np.mean(calm_hours_valid)) if len(calm_hours_valid) > 0 else 12.0
    median_calm = float(np.median(calm_hours_valid)) if len(calm_hours_valid) > 0 else 8.0
    max_calm = float(np.max(calm_hours_valid)) if len(calm_hours_valid) > 0 else 24.0

    # Current streak
    latest_finish = sorted_raw["finished_at"].max()
    latest_start = sorted_raw["started_at"].max()
    max_grid_ts = pd.to_datetime(grid_sub["timestamp"], utc=True).max()

    is_active_now = latest_start > latest_finish and latest_start >= max_grid_ts - pd.Timedelta(hours=1)
    if is_active_now:
        current_calm = 0.0
    else:
        current_calm = max(0.0, (max_grid_ts - latest_finish).total_seconds() / 3600.0)

    # Daily shelter time
    dates_count = max(1, pd.to_datetime(grid_sub["timestamp"]).dt.date.nunique())
    daily_shelter = float(grid_sub["alert_active"].sum()) / dates_count

    return {
        "avg_calm_hours": round(avg_calm, 1),
        "median_calm_hours": round(median_calm, 1),
        "max_calm_hours": round(max_calm, 1),
        "current_calm_hours": round(current_calm, 1),
        "is_active_now": is_active_now,
        "daily_shelter_hours": round(daily_shelter, 1),
    }


def get_disruption_scores(
    grid_df: pd.DataFrame,
    region_slug: Optional[str] = None,
    timezone: str = "Europe/Kyiv",
) -> Dict[str, float]:
    """Compute civilian disruption scores: sleep disruption index and business workday loss."""
    df = grid_df if region_slug is None else grid_df[grid_df["region_slug"] == region_slug]
    if df.empty:
        return {"sleep_disruption_pct": 0.0, "workday_disruption_pct": 0.0}

    ts_local = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(timezone)
    hours = ts_local.dt.hour
    dows = ts_local.dt.dayofweek

    # Sleep hours: 23:00 to 06:00
    is_sleep_hour = (hours >= 23) | (hours <= 6)
    sleep_active = df.loc[is_sleep_hour, "alert_active"].sum()
    total_active = max(1, df["alert_active"].sum())
    sleep_disruption_pct = float(sleep_active / total_active)

    # Workday hours: Mon-Fri (dow < 5), 09:00 - 18:00
    is_workday_hour = (dows < 5) & (hours >= 9) & (hours < 18)
    workday_slice = df[is_workday_hour]
    workday_disruption_pct = (
        float(workday_slice["alert_active"].mean()) if not workday_slice.empty else 0.0
    )

    return {
        "sleep_disruption_pct": round(sleep_disruption_pct, 3),
        "workday_disruption_pct": round(workday_disruption_pct, 3),
    }


def get_threat_level_stats(
    raw_df: pd.DataFrame,
    region_slug: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute threat level breakdown (Red vs. Yellow alerts) with backward compatibility.

    Uses Ukrainian official government terminology:
    - 'Red': Direct strike threat / immediate air raid sirens (ballistic missiles, drone swarms, artillery).
    - 'Yellow': Heightened danger / aviation activity / potential threat (e.g. MiG-31K airborne, tactical aviation).
    - 'Unspecified': Fallback for intervals or datasets missing threat-level classification.

    Returns:
        Dict containing counts, percentages, average durations, and total hours for each level,
        along with `has_threat_data` flag indicating whether differentiation exists in this interval.
    """
    raw_sub = raw_df if region_slug is None else raw_df[raw_df["region_slug"] == region_slug]
    if raw_sub.empty:
        return {
            "has_threat_data": False,
            "total_alerts": 0,
            "red_count": 0,
            "red_pct": 0.0,
            "red_avg_min": 0.0,
            "red_total_hours": 0.0,
            "yellow_count": 0,
            "yellow_pct": 0.0,
            "yellow_avg_min": 0.0,
            "yellow_total_hours": 0.0,
            "unspecified_count": 0,
            "unspecified_pct": 0.0,
            "unspecified_avg_min": 0.0,
            "unspecified_total_hours": 0.0,
        }

    durations_min = (
        pd.to_datetime(raw_sub["finished_at"], utc=True) - pd.to_datetime(raw_sub["started_at"], utc=True)
    ).dt.total_seconds() / 60.0

    if "threat_level" in raw_sub.columns:
        threats = raw_sub["threat_level"].fillna("unspecified").astype(str).str.strip().str.lower()
        threats = threats.apply(lambda x: x if x in ("red", "yellow") else "unspecified")
    else:
        threats = pd.Series(["unspecified"] * len(raw_sub), index=raw_sub.index)

    total_alerts = len(raw_sub)
    red_mask = threats == "red"
    yellow_mask = threats == "yellow"
    unspec_mask = threats == "unspecified"

    red_count = int(red_mask.sum())
    yellow_count = int(yellow_mask.sum())
    unspec_count = int(unspec_mask.sum())

    classified_count = red_count + yellow_count
    has_threat_data = classified_count > 0

    return {
        "has_threat_data": bool(has_threat_data),
        "total_alerts": total_alerts,
        "classified_count": classified_count,
        "red_count": red_count,
        "red_pct": float(red_count / total_alerts) if total_alerts > 0 else 0.0,
        "red_share": float(red_count / classified_count) if classified_count > 0 else 0.0,
        "red_avg_min": round(float(durations_min[red_mask].mean()), 1) if red_count > 0 else 0.0,
        "red_total_hours": round(float(durations_min[red_mask].sum() / 60.0), 1),
        "yellow_count": yellow_count,
        "yellow_pct": float(yellow_count / total_alerts) if total_alerts > 0 else 0.0,
        "yellow_share": float(yellow_count / classified_count) if classified_count > 0 else 0.0,
        "yellow_avg_min": round(float(durations_min[yellow_mask].mean()), 1) if yellow_count > 0 else 0.0,
        "yellow_total_hours": round(float(durations_min[yellow_mask].sum() / 60.0), 1),
        "unspecified_count": unspec_count,
        "unspecified_pct": float(unspec_count / total_alerts) if total_alerts > 0 else 0.0,
        "unspecified_avg_min": round(float(durations_min[unspec_mask].mean()), 1) if unspec_count > 0 else 0.0,
        "unspecified_total_hours": round(float(durations_min[unspec_mask].sum() / 60.0), 1),
    }


def get_hromada_breakdown(
    raw_df: pd.DataFrame,
    region_slug: str,
) -> Dict[str, Any]:
    """Compute granular territorial community (hromada) statistics within an individual oblast.

    Backward compatibility:
    If the selected interval contains no granular hromada-level data (e.g. historical periods
    prior to decentralization or oblast-wide alerts), `has_differentiation` is False.

    Returns:
        Dict containing:
        - 'has_differentiation': bool
        - 'total_alerts': int
        - 'total_localized': int
        - 'total_oblast_wide': int
        - 'summary_df': pd.DataFrame with per-hromada metrics
    """
    registered_hromadas = get_region_hromadas(region_slug)
    raw_sub = raw_df[raw_df["region_slug"] == region_slug].copy()

    if raw_sub.empty or not registered_hromadas:
        empty_df = pd.DataFrame(columns=[
            "Hromada (EN)", "Громада (UK)", "Total Alerts", "Localized Alerts",
            "Oblast-Wide Alerts", "Avg Duration (mins)", "Red Alerts", "Yellow Alerts"
        ])
        return {
            "has_differentiation": False,
            "total_alerts": 0,
            "total_localized": 0,
            "total_oblast_wide": 0,
            "hromadas_count": len(registered_hromadas),
            "summary_df": empty_df,
        }

    # Extract hromada column
    if "hromada" in raw_sub.columns:
        hromadas_series = raw_sub["hromada"].fillna("oblast_wide").astype(str).str.strip()
    else:
        hromadas_series = pd.Series(["oblast_wide"] * len(raw_sub), index=raw_sub.index)

    # Extract threat level
    if "threat_level" in raw_sub.columns:
        threats_series = raw_sub["threat_level"].fillna("unspecified").astype(str).str.strip().str.lower()
    else:
        threats_series = pd.Series(["unspecified"] * len(raw_sub), index=raw_sub.index)

    durations_min = (
        pd.to_datetime(raw_sub["finished_at"], utc=True) - pd.to_datetime(raw_sub["started_at"], utc=True)
    ).dt.total_seconds() / 60.0

    raw_sub = raw_sub.assign(
        _clean_hromada=hromadas_series,
        _clean_threat=threats_series,
        _duration_min=durations_min,
    )

    clean_h = hromadas_series.str.lower()
    oblast_wide_mask = clean_h.isin(["oblast_wide", "unspecified", "nan", "none", ""])
    is_localized = ~oblast_wide_mask
    has_differentiation = bool(is_localized.any())

    total_oblast_wide = int(oblast_wide_mask.sum())
    total_localized = int(is_localized.sum())

    oblast_wide_subset = raw_sub[oblast_wide_mask]

    rows = []
    for h in registered_hromadas:
        stem = h.name_uk.lower().rstrip("ська").rstrip("івська").rstrip("инська")
        h_mask = (
            (clean_h == h.slug.lower()) |
            (clean_h == h.name_uk.lower()) |
            (clean_h == h.name_en.lower()) |
            (clean_h.str.startswith(stem) if len(stem) >= 4 else False)
        ) & is_localized
        h_localized = raw_sub[h_mask]
        loc_count = len(h_localized)
        tot_count = loc_count + total_oblast_wide

        all_affecting = pd.concat([h_localized, oblast_wide_subset], ignore_index=True)
        avg_dur = round(float(all_affecting["_duration_min"].mean()), 1) if not all_affecting.empty else 0.0

        red_count = int((all_affecting["_clean_threat"] == "red").sum())
        yellow_count = int((all_affecting["_clean_threat"] == "yellow").sum())

        rows.append({
            "Hromada (EN)": h.name_en,
            "Громада (UK)": h.name_uk,
            "Total Alerts": tot_count,
            "Localized Alerts": loc_count,
            "Oblast-Wide Alerts": total_oblast_wide,
            "Localized Share": f"{(loc_count / max(1, tot_count)):.1%}",
            "Avg Duration (mins)": avg_dur,
            "Red Alerts": red_count,
            "Yellow Alerts": yellow_count,
        })

    summary_df = pd.DataFrame(rows).sort_values(by=["Localized Alerts", "Total Alerts"], ascending=False).reset_index(drop=True)

    return {
        "has_differentiation": has_differentiation,
        "total_alerts": len(raw_sub),
        "total_localized": total_localized,
        "total_oblast_wide": total_oblast_wide,
        "hromadas_count": len(registered_hromadas),
        "summary_df": summary_df,
    }


def get_safe_window_recommendation(
    grid_df: pd.DataFrame,
    day_of_week: int,
    start_hour: int,
    duration_hours: int = 1,
    region_slug: Optional[str] = None,
    timezone: str = "Europe/Kyiv",
) -> Dict[str, Any]:
    """Evaluate civil safety for a planned meeting, travel, or outdoor window.

    Args:
        grid_df: Continuous hourly observations grid.
        day_of_week: Day index (0=Monday, 6=Sunday).
        start_hour: Window starting hour in Kyiv local time (0-23).
        duration_hours: Duration of planned activity (1-4 hours).
        region_slug: Optional region slug. If None, computes national aggregate.
        timezone: Local timezone for diurnal interpretation.

    Returns:
        Dict containing safety_score (0-100%), danger_prob, verdict, plain-text advice,
        and safest alternative hour.
    """
    df = grid_df if region_slug is None else grid_df[grid_df["region_slug"] == region_slug]
    if df.empty:
        return {
            "safety_score": 100.0,
            "danger_prob": 0.0,
            "verdict": "SAFE",
            "verdict_label": "HIGH SAFETY WINDOW",
            "recommendation": "No historical alerts recorded in this interval.",
            "safest_alternative_hour": 14,
        }

    ts_local = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(timezone)
    df_temp = df.assign(_dow=ts_local.dt.dayofweek, _hour=ts_local.dt.hour)

    # Filter to the chosen day
    day_slice = df_temp[df_temp["_dow"] == day_of_week]
    if day_slice.empty:
        return {
            "safety_score": 100.0,
            "danger_prob": 0.0,
            "verdict": "SAFE",
            "verdict_label": "HIGH SAFETY WINDOW",
            "recommendation": "Insufficient data for selected day.",
            "safest_alternative_hour": 14,
        }

    # Evaluate target hours
    target_hours = [(start_hour + i) % 24 for i in range(duration_hours)]
    target_slice = day_slice[day_slice["_hour"].isin(target_hours)]

    danger_prob = float(target_slice["alert_active"].mean()) if not target_slice.empty else 0.15
    safety_score = max(0.0, min(100.0, round((1.0 - danger_prob) * 100.0, 1)))

    # Find the safest hour on this day
    hourly_threat = day_slice.groupby("_hour")["alert_active"].mean()
    safest_hour = int(hourly_threat.idxmin())

    # Formulate verdict & human-friendly advice
    day_name = DAY_NAMES[day_of_week]
    if safety_score >= 82.0:
        verdict = "SAFE"
        verdict_label = "OPTIMAL SAFETY WINDOW"
        recommendation = (
            f"Statistically optimal window on {day_name}. Historical alert probability is only "
            f"{danger_prob:.0%}. Highly recommended for in-person meetings, shopping, doctor visits, and outdoor travel."
        )
    elif safety_score >= 60.0:
        verdict = "MODERATE"
        verdict_label = "MODERATE ALERT RISK WINDOW (CAUTION)"
        recommendation = (
            f"Acceptable window with moderate alert risk ({danger_prob:.0%} chance of sirens). "
            f"Meetings can proceed, but verify proximity to certified bomb shelters or safe interior rooms."
        )
    else:
        verdict = "DANGER"
        verdict_label = "HIGH ALERT RISK WINDOW (DANGER)"
        recommendation = (
            f"Elevated alert risk window on {day_name}! Coincides with frequent wartime strike sorties or drone loitering "
            f"({danger_prob:.0%} historical alert rate). Consider rescheduling to around {safest_hour:02d}:00."
        )

    return {
        "safety_score": safety_score,
        "danger_prob": round(danger_prob, 3),
        "verdict": verdict,
        "verdict_label": verdict_label,
        "recommendation": recommendation,
        "safest_alternative_hour": safest_hour,
        "day_name": day_name,
        "target_hours": target_hours,
    }


def get_duration_cdf_percentiles(
    raw_df: pd.DataFrame,
    region_slug: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute empirical cumulative distribution percentiles for expected siren duration.

    Answers civilian question: 'If a siren sounds now, how long will it statistically last?'
    """
    df = raw_df if region_slug is None else raw_df[raw_df["region_slug"] == region_slug]
    empty_cdf = pd.DataFrame([
        {"minutes": m, "probability": round(min(1.0, m / 120.0), 3), "pct_label": f"{min(1.0, m / 120.0):.0%}"}
        for m in range(0, 185, 5)
    ])
    if df.empty:
        return {
            "p25_min": 20.0, "p50_min": 45.0, "p75_min": 90.0, "p90_min": 150.0,
            "prob_clear_30m": 0.35, "prob_clear_60m": 0.65, "prob_clear_120m": 0.85,
            "cdf_df": empty_cdf,
        }

    durations_min = (
        pd.to_datetime(df["finished_at"], utc=True) - pd.to_datetime(df["started_at"], utc=True)
    ).dt.total_seconds() / 60.0
    valid_durations = durations_min[durations_min > 0].values

    if len(valid_durations) == 0:
        return {
            "p25_min": 20.0, "p50_min": 45.0, "p75_min": 90.0, "p90_min": 150.0,
            "prob_clear_30m": 0.35, "prob_clear_60m": 0.65, "prob_clear_120m": 0.85,
            "cdf_df": empty_cdf,
        }

    p25 = float(np.percentile(valid_durations, 25))
    p50 = float(np.percentile(valid_durations, 50))
    p75 = float(np.percentile(valid_durations, 75))
    p90 = float(np.percentile(valid_durations, 90))

    prob_30m = float(np.mean(valid_durations <= 30.0))
    prob_60m = float(np.mean(valid_durations <= 60.0))
    prob_120m = float(np.mean(valid_durations <= 120.0))

    cdf_points = []
    for m in range(0, 185, 5):
        p = float(np.mean(valid_durations <= float(m)))
        cdf_points.append({"minutes": m, "probability": round(p, 3), "pct_label": f"{p:.0%}"})
    cdf_df = pd.DataFrame(cdf_points)

    return {
        "p25_min": round(p25, 0),
        "p50_min": round(p50, 0),
        "p75_min": round(p75, 0),
        "p90_min": round(p90, 0),
        "prob_clear_30m": round(prob_30m, 2),
        "prob_clear_60m": round(prob_60m, 2),
        "prob_clear_120m": round(prob_120m, 2),
        "cdf_df": cdf_df,
    }


def get_threat_level_hourly_profile(
    raw_df: pd.DataFrame,
    region_slug: Optional[str] = None,
    timezone: str = "Europe/Kyiv",
) -> pd.DataFrame:
    """Compute diurnal hour-by-hour distribution of Yellow vs Red alerts.

    Returns DataFrame with columns:
        ['Hour', 'Yellow (Drones)', 'Red (Missiles)', 'Total Classified']
    """
    df = raw_df if region_slug is None else raw_df[raw_df["region_slug"] == region_slug]
    if df.empty or "threat_level" not in df.columns:
        return pd.DataFrame(columns=["Hour", "Yellow (Drones)", "Red (Missiles)", "Total Classified"])

    threats = df["threat_level"].fillna("unspecified").astype(str).str.lower().str.strip()
    classified_mask = threats.isin(["yellow", "red"])
    if not classified_mask.any():
        return pd.DataFrame(columns=["Hour", "Yellow (Drones)", "Red (Missiles)", "Total Classified"])

    sub = df[classified_mask].copy()
    sub_local_ts = pd.to_datetime(sub["started_at"], utc=True).dt.tz_convert(timezone)
    sub["_hour"] = sub_local_ts.dt.hour
    sub["_threat"] = threats[classified_mask]

    records = []
    for h in range(24):
        h_sub = sub[sub["_hour"] == h]
        y_count = int((h_sub["_threat"] == "yellow").sum())
        r_count = int((h_sub["_threat"] == "red").sum())
        records.append({
            "Hour": f"{h:02d}:00",
            "Yellow (Drones)": y_count,
            "Red (Missiles)": r_count,
            "Total Classified": y_count + r_count,
        })

    return pd.DataFrame(records)


def get_annual_evolution_stats(
    raw_df: pd.DataFrame,
    region_slug: Optional[str] = None,
) -> pd.DataFrame:
    """Compute longitudinal tactical shift metrics across wartime years (2022 to 2026)."""
    df = raw_df if region_slug is None else raw_df[raw_df["region_slug"] == region_slug]
    if df.empty:
        return pd.DataFrame()

    df_copy = df.copy()
    start_dt = pd.to_datetime(df_copy["started_at"], utc=True)
    finish_dt = pd.to_datetime(df_copy["finished_at"], utc=True)
    df_copy["_year"] = start_dt.dt.year
    df_copy["_duration_min"] = (finish_dt - start_dt).dt.total_seconds() / 60.0
    df_copy["_is_night"] = (start_dt.dt.hour >= 23) | (start_dt.dt.hour <= 6)

    tactical_descriptions = {
        2022: "Cruise missile strikes & tactical aviation sorties",
        2023: "Introduction of Shahed-136 & combined barrages",
        2024: "Massed nighttime drone swarms & ballistic strikes",
        2025: "Sustained attrition drone warfare across central/west",
        2026: "Differentiated Yellow (drones) vs Red (missiles) system",
    }

    records = []
    for year, grp in df_copy.groupby("_year"):
        total_alerts = len(grp)
        # Approximate days active in year
        days_in_year = max(1, grp["started_at"].dt.date.nunique())
        daily_rate = round(total_alerts / days_in_year, 1)
        avg_dur = round(float(grp["_duration_min"].mean()), 1)
        median_dur = round(float(grp["_duration_min"].median()), 1)
        night_pct = round(float(grp["_is_night"].mean()) * 100.0, 1)
        total_hours = round(float(grp["_duration_min"].sum() / 60.0), 0)

        records.append({
            "Year": int(year),
            "Total Alerts": total_alerts,
            "Avg Alerts / Day": daily_rate,
            "Avg Duration (min)": avg_dur,
            "Median Duration (min)": median_dur,
            "Nighttime Alerts %": f"{night_pct:.1f}%",
            "Total Shelter Hours": int(total_hours),
            "Tactical Attack Profile": tactical_descriptions.get(int(year), "Standard Wartime Alert Distribution"),
        })

    return pd.DataFrame(records).sort_values(by="Year").reset_index(drop=True)

