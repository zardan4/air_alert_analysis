"""Interactive User-Friendly Air Raid Alert Forecasting & Civil Safety Platform.

Featuring:
- Citizen Safe Window & Meeting Planner (instant safety verdicts & risk mitigation advice)
- Empirical Duration Expectation Curve ('What to expect if a siren sounds now')
- Government Threat Level Differentiation (Yellow drone alerts vs Red missile alerts)
- Interactive Altair Diurnal Heatmaps & Tactical Attack Profiles
- Longitudinal Wartime Shift Analytics (2022 to 2026 attack evolution)
- Territorial Community (Hromada) Granularity Drilldowns
- Multi-Horizon Predictive AI Forecasts (1h, 2h, 3h ahead)
- Multi-Source Integration (Vadimkin open dataset, alerts.in.ua official dump/API, synthetic mock)
"""

from datetime import date, timedelta
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path for cloud deployment
_ROOT_DIR = Path(__file__).resolve().parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.baselines.naive_baselines import get_all_baselines
from src.config import load_config
from src.ui.map_component import render_ukraine_svg_map
from src.ui.stats import (
    DAY_ABBR,
    DAY_NAMES,
    get_annual_evolution_stats,
    get_calm_and_downtime_stats,
    get_day_of_week_profile,
    get_disruption_scores,
    get_duration_cdf_percentiles,
    get_duration_spectrum,
    get_heatmap_matrix,
    get_hromada_breakdown,
    get_peak_and_safe_windows,
    get_safe_window_recommendation,
    get_threat_level_hourly_profile,
    get_threat_level_stats,
)
from src.utils.regions import ALL_REGION_SLUGS, TOTAL_REGIONS_COUNT, get_all_regions, get_region

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION & MODERN DESIGN SYSTEM
# ==============================================================================
st.set_page_config(
    page_title="Ukraine Air Raid Alert Forecasting & Alert Risk Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Global clean typography & dark slate background */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Status Badges */
    .status-badge-active {
        background-color: #ef4444;
        color: white;
        padding: 8px 18px;
        border-radius: 8px;
        font-weight: 800;
        font-size: 1.15rem;
        display: inline-block;
        box-shadow: 0 0 14px rgba(239, 68, 68, 0.4);
    }
    .status-badge-yellow {
        background-color: #f59e0b;
        color: white;
        padding: 8px 18px;
        border-radius: 8px;
        font-weight: 800;
        font-size: 1.15rem;
        display: inline-block;
        box-shadow: 0 0 14px rgba(245, 158, 11, 0.4);
    }
    .status-badge-clear {
        background-color: #10b981;
        color: white;
        padding: 8px 18px;
        border-radius: 8px;
        font-weight: 800;
        font-size: 1.15rem;
        display: inline-block;
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.3);
    }

    /* Glassmorphic KPI Cards with uniform height and alignment */
    .kpi-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(71, 85, 105, 0.4);
        backdrop-filter: blur(8px);
        border-radius: 10px;
        padding: 14px 10px;
        text-align: center;
        transition: transform 0.15s ease, border-color 0.15s ease;
        height: 118px;
        min-height: 118px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .kpi-card:hover {
        border-color: #38bdf8;
        transform: translateY(-2px);
    }
    .kpi-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: #94a3b8;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
        line-height: 1.2;
        text-align: center;
    }
    .kpi-value {
        font-size: 1.45rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.2;
        text-align: center;
    }
    .kpi-sub {
        font-size: 0.73rem;
        color: #cbd5e1;
        margin-top: 4px;
        line-height: 1.25;
        text-align: center;
    }

    /* Threat Cards with matching uniform card geometry */
    .threat-card-red {
        background: rgba(69, 10, 10, 0.65);
        border: 1px solid #ef4444;
        border-radius: 10px;
        padding: 14px 10px;
        text-align: center;
        height: 118px;
        min-height: 118px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .threat-card-yellow {
        background: rgba(120, 53, 15, 0.65);
        border: 1px solid #f59e0b;
        border-radius: 10px;
        padding: 14px 10px;
        text-align: center;
        height: 118px;
        min-height: 118px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }

    /* Safety Verdict Cards */
    .verdict-card-safe {
        background: rgba(6, 78, 59, 0.75);
        border: 1px solid #10b981;
        border-radius: 10px;
        padding: 18px 20px;
    }
    .verdict-card-caution {
        background: rgba(120, 53, 15, 0.75);
        border: 1px solid #f59e0b;
        border-radius: 10px;
        padding: 18px 20px;
    }
    .verdict-card-danger {
        background: rgba(127, 29, 29, 0.75);
        border: 1px solid #ef4444;
        border-radius: 10px;
        padding: 18px 20px;
    }
</style>
""", unsafe_allow_html=True)

# Detect CLI parameters
is_mock_cli = "--mock" in sys.argv
is_vadimkin_cli = "--vadimkin" in sys.argv


@st.cache_data
def load_cached_data(mock_mode: bool = False, vadimkin_mode: bool = False):
    """Load raw alerts, continuous grid, and features. Automatically falls back if files are absent."""
    raw_path = _ROOT_DIR / "data/raw/alerts_raw.parquet"
    grid_path = _ROOT_DIR / "data/processed/alerts_grid.parquet"
    feat_path = _ROOT_DIR / "data/processed/alerts_features.parquet"
    bench_path = _ROOT_DIR / "data/processed/benchmark_results.csv"

    if not raw_path.exists():
        raw_path = Path("data/raw/alerts_raw.parquet")
    if not grid_path.exists():
        grid_path = Path("data/processed/alerts_grid.parquet")
    if not feat_path.exists():
        feat_path = Path("data/processed/alerts_features.parquet")
    if not bench_path.exists():
        bench_path = Path("data/processed/benchmark_results.csv")

    raw = pd.read_parquet(raw_path) if raw_path.exists() else None
    grid = pd.read_parquet(grid_path) if grid_path.exists() else None
    feat = pd.read_parquet(feat_path) if feat_path.exists() else None
    bench = pd.read_csv(bench_path) if bench_path.exists() else None

    if raw is None or grid is None:
        from src.data.grid_converter import IntervalToGridConverter
        if mock_mode:
            from src.data.mock_source import MockAlertDataSource
            source = MockAlertDataSource(seed=42)
            raw = source.load_raw_alerts(start_date="2024-01-01", end_date="2024-03-31")
            converter = IntervalToGridConverter(frequency="1h")
            grid = converter.convert(raw_df=raw, start_date="2024-01-01", end_date="2024-03-31")
        elif vadimkin_mode:
            from src.data.vadimkin import VadimkinDataSource
            source = VadimkinDataSource()
            raw = source.load_raw_alerts(start_date="2024-01-01", end_date="2024-03-31")
            converter = IntervalToGridConverter(frequency="1h")
            grid = converter.convert(raw_df=raw, start_date="2024-01-01", end_date="2024-03-31")
        else:
            alerts_dump_csv = _ROOT_DIR / "data/raw/alerts_in_ua_dump.csv"
            vadimkin_csv = _ROOT_DIR / "data/raw/vadimkin_official.csv"
            if alerts_dump_csv.exists():
                from src.data.alerts_in_ua import AlertsInUaDataSource
                source = AlertsInUaDataSource(csv_path=alerts_dump_csv)
                raw = source.load_raw_alerts()
                grid = IntervalToGridConverter(frequency="1h").convert(raw_df=raw)
            elif vadimkin_csv.exists():
                from src.data.vadimkin import VadimkinDataSource
                source = VadimkinDataSource(csv_path=vadimkin_csv)
                raw = source.load_raw_alerts()
                grid = IntervalToGridConverter(frequency="1h").convert(raw_df=raw)
            else:
                from src.data.mock_source import MockAlertDataSource
                source = MockAlertDataSource(seed=42)
                raw = source.load_raw_alerts(start_date="2024-01-01", end_date="2024-03-31")
                grid = IntervalToGridConverter(frequency="1h").convert(raw_df=raw, start_date="2024-01-01", end_date="2024-03-31")

    return raw, grid, feat, bench


raw_df, grid_df, feat_df, bench_df = load_cached_data(mock_mode=is_mock_cli, vadimkin_mode=is_vadimkin_cli)
all_regions_dict = get_all_regions()

if grid_df is None or raw_df is None:
    st.error("Data assets not found. Please run `python main.py generate-data` first, or launch with `--vadimkin` or `--mock`.")
    st.stop()

# Detect active data source
sources_in_data = list(raw_df["source"].unique()) if raw_df is not None and "source" in raw_df.columns else []
is_mock_source = all("mock" in str(s).lower() for s in sources_in_data) if sources_in_data else False
is_vadimkin_source = any("vadimkin" in str(s).lower() for s in sources_in_data) if sources_in_data else False
is_mock_mode = is_mock_cli or is_mock_source
is_vadimkin_mode = is_vadimkin_cli or is_vadimkin_source


# ==============================================================================
# ==============================================================================
# LEFT SIDEBAR: MENU & CONTROLS
# ==============================================================================
st.sidebar.title("Air Raid Intelligence System")

# Mode indicator
if is_mock_mode:
    st.sidebar.caption("Data Source: Synthetic Simulation")
elif is_vadimkin_mode:
    st.sidebar.caption("Data Source: Vadimkin Sirens Archive")
else:
    st.sidebar.caption("Data Source: alerts.in.ua Official Archive")

# 1. Location Selector
oblast_options = ["ALL_UKRAINE"] + ALL_REGION_SLUGS

def format_oblast_option(opt: str) -> str:
    if opt == "ALL_UKRAINE":
        return "National Overview (All Ukraine)"
    reg = all_regions_dict[opt]
    return f"{reg.name_en} ({reg.name_uk})"

# Synchronize with URL query parameter (?region=...)
query_param_region = st.query_params.get("region", None)
if query_param_region and query_param_region in oblast_options:
    initial_target = query_param_region
else:
    initial_target = "ALL_UKRAINE"

if "selected_target" not in st.session_state:
    st.session_state["selected_target"] = initial_target
elif query_param_region and query_param_region in oblast_options and query_param_region != st.session_state["selected_target"]:
    st.session_state["selected_target"] = query_param_region

if "region_selector_key" not in st.session_state:
    st.session_state["region_selector_key"] = st.session_state["selected_target"]
elif st.session_state["region_selector_key"] != st.session_state["selected_target"]:
    st.session_state["region_selector_key"] = st.session_state["selected_target"]

def handle_region_selection_change():
    new_sel = st.session_state["region_selector_key"]
    st.session_state["selected_target"] = new_sel
    if new_sel == "ALL_UKRAINE":
        st.query_params.pop("region", None)
    else:
        st.query_params["region"] = new_sel

selected_target = st.sidebar.selectbox(
    "Region:",
    oblast_options,
    format_func=format_oblast_option,
    key="region_selector_key",
    on_change=handle_region_selection_change,
)

st.sidebar.divider()

# 2. Forecast Lookahead Horizon
horizon_selected = st.sidebar.radio(
    "Forecast Horizon:",
    [1, 2, 3],
    index=0,
    format_func=lambda h: f"{h}h Ahead",
    horizontal=True,
)
st.sidebar.caption(f"Period used when calculating the air alert risk level for the upcoming hours.")

st.sidebar.divider()

# 3. Time Interval Selector
min_avail_date = pd.to_datetime(grid_df["timestamp"]).min().date()
max_avail_date = pd.to_datetime(grid_df["timestamp"]).max().date()

interval_mode = st.sidebar.radio(
    "Time Window:",
    [
        "Full History (2022-2026)",
        "2026 (YTD)",
        "2025",
        "2024",
        "2023",
        "2022",
        "Last 30 Days",
        "Last 90 Days",
        "Custom Date Range",
    ],
    index=0,
)

if interval_mode == "Full History (2022-2026)":
    sel_start_date, sel_end_date = min_avail_date, max_avail_date
elif interval_mode == "2026 (YTD)":
    sel_start_date = max(min_avail_date, date(2026, 1, 1))
    sel_end_date = max_avail_date
elif interval_mode == "2025":
    sel_start_date = max(min_avail_date, date(2025, 1, 1))
    sel_end_date = min(max_avail_date, date(2025, 12, 31))
elif interval_mode == "2024":
    sel_start_date = max(min_avail_date, date(2024, 1, 1))
    sel_end_date = min(max_avail_date, date(2024, 12, 31))
elif interval_mode == "2023":
    sel_start_date = max(min_avail_date, date(2023, 1, 1))
    sel_end_date = min(max_avail_date, date(2023, 12, 31))
elif interval_mode == "2022":
    sel_start_date = max(min_avail_date, date(2022, 2, 24))
    sel_end_date = min(max_avail_date, date(2022, 12, 31))
elif interval_mode == "Last 30 Days":
    sel_start_date = max(min_avail_date, max_avail_date - timedelta(days=30))
    sel_end_date = max_avail_date
elif interval_mode == "Last 90 Days":
    sel_start_date = max(min_avail_date, max_avail_date - timedelta(days=90))
    sel_end_date = max_avail_date
else:
    picked = st.sidebar.date_input(
        "Choose start and end dates:",
        value=(max(min_avail_date, max_avail_date - timedelta(days=90)), max_avail_date),
        min_value=min_avail_date,
        max_value=max_avail_date,
    )
    if isinstance(picked, (list, tuple)) and len(picked) == 2:
        sel_start_date, sel_end_date = picked
    elif isinstance(picked, (list, tuple)) and len(picked) == 1:
        sel_start_date = sel_end_date = picked[0]
    else:
        sel_start_date, sel_end_date = min_avail_date, max_avail_date

if sel_start_date > sel_end_date:
    sel_start_date, sel_end_date = sel_end_date, sel_start_date

num_days_selected = max(1, (sel_end_date - sel_start_date).days + 1)
st.sidebar.caption(f"{sel_start_date} to {sel_end_date} ({num_days_selected} days)")

st.sidebar.divider()

# ==============================================================================
# DATA FILTERING BY SELECTED WINDOW
# ==============================================================================
start_ts = pd.Timestamp(sel_start_date, tz="UTC")
end_ts = pd.Timestamp(sel_end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

filtered_grid_df = grid_df[
    (grid_df["timestamp"] >= start_ts) & (grid_df["timestamp"] <= end_ts)
].copy()

filtered_raw_df = raw_df[
    (pd.to_datetime(raw_df["finished_at"], utc=True) >= start_ts)
    & (pd.to_datetime(raw_df["started_at"], utc=True) <= end_ts)
].copy()

filtered_feat_df = (
    feat_df[(feat_df["timestamp"] >= start_ts) & (feat_df["timestamp"] <= end_ts)].copy()
    if feat_df is not None
    else None
)

if filtered_grid_df.empty:
    st.warning("No alert records found in the selected interval. Please select a broader date range in the sidebar.")
    st.stop()


# Helper function to render Altair heatmap
def render_altair_heatmap(matrix_df: pd.DataFrame) -> alt.Chart:
    """Build an interactive Altair 7x24 heatmap with rich tooltips and zoom/pan."""
    melted = matrix_df.reset_index().melt(id_vars="index", var_name="hour", value_name="probability")
    melted.rename(columns={"index": "day"}, inplace=True)
    melted["probability"] = melted["probability"].astype(float)

    def format_slot(h_str):
        try:
            h = int(str(h_str).split(":")[0])
            return f"{h:02d}:00 - {(h + 1) % 24:02d}:00"
        except Exception:
            return str(h_str)

    melted["time_window"] = melted["hour"].apply(format_slot)
    three_hour_ticks = [f"{h:02d}:00" for h in range(0, 24, 3)]

    chart = (
        alt.Chart(melted)
        .mark_rect(cornerRadius=3)
        .encode(
            x=alt.X(
                "hour:O",
                title="Hour of Day (Europe/Kyiv Time)",
                sort=list(matrix_df.columns),
                axis=alt.Axis(labelAngle=0, values=three_hour_ticks),
            ),
            y=alt.Y("day:O", title="Day of Week", sort=list(matrix_df.index)),
            color=alt.Color(
                "probability:Q",
                scale=alt.Scale(scheme="yelloworangered", domain=[0.0, 1.0]),
                legend=alt.Legend(title="Alert Risk", format=".0%"),
            ),
            tooltip=[
                alt.Tooltip("day:O", title="Day of Week"),
                alt.Tooltip("time_window:N", title="Time Slot (Kyiv Time)"),
                alt.Tooltip("probability:Q", title="Alert Risk Probability", format=".1%"),
            ],
        )
        .properties(height=260)
        .configure_axis(
            labelFontSize=11,
            titleFontSize=12,
            labelColor="#cbd5e1",
            titleColor="#e2e8f0",
        )
        .configure_view(strokeWidth=0)
    )
    return chart


# ==============================================================================
# VIEW A: INDIVIDUAL OBLAST WORKSPACE
# ==============================================================================
if selected_target != "ALL_UKRAINE":
    reg_info = all_regions_dict[selected_target]
    region_slug = selected_target

    # Return button to National Overview
    col_back, _ = st.columns([1.5, 4])
    with col_back:
        if st.button("Back to All Ukraine", key="btn_return_national", use_container_width=True):
            st.session_state["selected_target"] = "ALL_UKRAINE"
            st.query_params.pop("region", None)
            st.rerun()

    # Top Situation Banner
    col_head, col_state = st.columns([2.4, 1.0])
    with col_head:
        st.title(reg_info.name_en)
        st.caption(f"{reg_info.name_uk} · {sel_start_date} to {sel_end_date} ({num_days_selected} days)")

    # Compute key statistics
    calm_stats = get_calm_and_downtime_stats(filtered_raw_df, filtered_grid_df, region_slug=region_slug)
    spectrum = get_duration_spectrum(filtered_raw_df, region_slug=region_slug)
    disruption = get_disruption_scores(filtered_grid_df, region_slug=region_slug)
    heatmap_df = get_heatmap_matrix(filtered_grid_df, region_slug=region_slug)
    windows = get_peak_and_safe_windows(heatmap_df)
    threat_stats = get_threat_level_stats(filtered_raw_df, region_slug=region_slug)
    hromada_stats = get_hromada_breakdown(filtered_raw_df, region_slug=region_slug)
    cdf_stats = get_duration_cdf_percentiles(filtered_raw_df, region_slug=region_slug)
    evolution_df = get_annual_evolution_stats(filtered_raw_df, region_slug=region_slug)

    # National benchmark comparisons
    nat_dates_count = max(1, pd.to_datetime(filtered_grid_df["timestamp"]).dt.date.nunique())
    nat_daily_shelter = float(filtered_grid_df["alert_active"].sum()) / nat_dates_count / TOTAL_REGIONS_COUNT
    nat_sleep_pct = float(get_disruption_scores(filtered_grid_df)["sleep_disruption_pct"])

    shelter_diff = calm_stats["daily_shelter_hours"] - nat_daily_shelter
    shelter_sub = f"{shelter_diff:+.1f} h vs Nat'l ({nat_daily_shelter:.1f} h)"

    sleep_diff = disruption["sleep_disruption_pct"] - nat_sleep_pct
    sleep_sub = f"{sleep_diff:+.0%} vs Nat'l avg ({nat_sleep_pct:.0%})"

    # State Badge (hide current status in vadimkin mode since data is not live)
    region_raw = filtered_raw_df[filtered_raw_df["region_slug"] == region_slug]
    latest_alert = region_raw.sort_values(by="started_at", ascending=False).iloc[0] if not region_raw.empty else None
    current_threat = str(latest_alert["threat_level"]).lower() if latest_alert is not None and "threat_level" in latest_alert else "unspecified"
    current_hromada = str(latest_alert["hromada"]) if latest_alert is not None and "hromada" in latest_alert else "oblast_wide"

    with col_state:
        st.write("")
        if is_vadimkin_mode:
            st.markdown(
                '<div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(71, 85, 105, 0.5); '
                'border-radius: 8px; padding: 10px 14px; text-align: center;">'
                '<span style="color: #94a3b8; font-size: 0.8rem; font-weight: 700; letter-spacing: 0.05em;">HISTORICAL ARCHIVE</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        elif calm_stats["is_active_now"]:
            if current_threat == "yellow":
                st.markdown('<div class="status-badge-yellow">ACTIVE YELLOW ALERT</div>', unsafe_allow_html=True)
                st.caption("Drone Threat / Heightened Air Reconnaissance")
            else:
                st.markdown('<div class="status-badge-active">ACTIVE RED ALERT</div>', unsafe_allow_html=True)
                st.caption("Direct Strike Threat / Sirens Sounding")
            if current_hromada not in ("oblast_wide", "unspecified"):
                st.caption(f"Localized to: **{current_hromada}**")
            else:
                st.caption("Scope: Oblast-wide siren")
        else:
            st.markdown('<div class="status-badge-clear">ALL CLEAR (NO ACTIVE ALERT)</div>', unsafe_allow_html=True)
            st.caption(f"Current calm: **{calm_stats['current_calm_hours']:.1f} hours** without sirens")

    st.write("")

    # ==========================================================================
    # TABBED INTERFACE (4 PURPOSE-BUILT WORKSPACES)
    # ==========================================================================
    tab_safety, tab_tactics, tab_hromada, tab_ai = st.tabs([
        "Civil Defense & Planner",
        "Temporal Patterns",
        "Hromadas",
        "Predictive Forecasts",
    ])

    # --------------------------------------------------------------------------
    # TAB 1: CIVIL SAFETY & MEETING PLANNER
    # --------------------------------------------------------------------------
    with tab_safety:
        st.caption(f"Baseline operational civil safety indicators for {reg_info.name_en} over the selected {num_days_selected}-day window.")
        # Top KPI Metric Strip
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">TYPICAL DURATION</div>
                <div class="kpi-value">{int(spectrum['median_min'])} min</div>
                <div class="kpi-sub">Avg: {int(spectrum['mean_min'])} min</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">DAILY SHELTER TIME</div>
                <div class="kpi-value">{calm_stats['daily_shelter_hours']} h/day</div>
                <div class="kpi-sub">{shelter_sub}</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">QUIET WINDOW</div>
                <div class="kpi-value">{calm_stats['median_calm_hours']} h</div>
                <div class="kpi-sub">Avg: {calm_stats['avg_calm_hours']} h | Max: {calm_stats['max_calm_hours']:.0f} h</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">SLEEP DISRUPTION</div>
                <div class="kpi-value">{disruption['sleep_disruption_pct']:.0%}</div>
                <div class="kpi-sub">{sleep_sub}</div>
            </div>
            """, unsafe_allow_html=True)
        with m5:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">WORKDAY INTERRUPTED</div>
                <div class="kpi-value">{disruption['workday_disruption_pct']:.0%}</div>
                <div class="kpi-sub">Mon-Fri (09:00 - 18:00)</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # CITIZEN SAFE WINDOW & MEETING PLANNER
        st.subheader("Activity & Meeting Safety Assessment")
        st.caption(
            "Civilian schedule risk planner. Select your intended schedule parameters below to evaluate "
            "historical siren probability and receive an empirical safety score with recommended alternative windows."
        )

        col_p1, col_p2, col_p3 = st.columns([1.2, 1.2, 1.0])
        with col_p1:
            plan_dow = st.selectbox(
                "Select Day of the Week:",
                options=list(range(7)),
                index=date.today().weekday(),
                format_func=lambda i: DAY_NAMES[i],
            )
        with col_p2:
            plan_hour = st.slider("Planned Start Time (Kyiv Time):", 0, 23, 14, format="%02d:00")
        with col_p3:
            plan_dur = st.select_slider(
                "Duration:",
                options=[1, 2, 3, 4],
                value=1,
                format_func=lambda h: f"{h} hour{'s' if h > 1 else ''}",
            )

        verdict_res = get_safe_window_recommendation(
            filtered_grid_df,
            day_of_week=plan_dow,
            start_hour=plan_hour,
            duration_hours=plan_dur,
            region_slug=region_slug,
        )

        v_class = "verdict-card-safe" if verdict_res["verdict"] == "SAFE" else (
            "verdict-card-caution" if verdict_res["verdict"] == "MODERATE" else "verdict-card-danger"
        )

        st.markdown(f"""
        <div class="{v_class}">
            <div style="font-size:1.25rem; font-weight:800; margin-bottom:6px;">{verdict_res['verdict_label']} — {verdict_res['safety_score']:.0f}% Civil Safety Score</div>
            <div style="font-size:0.95rem; line-height:1.5;">{verdict_res['recommendation']}</div>
            <div style="font-size:0.85rem; color:#bae6fd; margin-top:8px;">
                <b>Statistical Benchmark:</b> Alert incidence for this slot is <b>{verdict_res['danger_prob']:.0%}</b>. 
                Safest window on {verdict_res['day_name']}: around <b>{verdict_res['safest_alternative_hour']:02d}:00</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        # DURATION EXPECTATION CURVE
        st.subheader("Siren Duration Expectations")
        st.caption(
            f"Empirical survival distribution of historical sirens in {reg_info.name_en}. "
            "Estimates how quickly an active siren is resolved, helping determine whether to wait out an alert or reschedule."
        )
        col_cdf1, col_cdf2 = st.columns([1.1, 1.0])
        with col_cdf1:
            st.markdown("**Duration Percentiles:**")
            st.markdown(f"- **Fast Intercept (25%):** Cleared in under **{cdf_stats['p25_min']:.0f} mins**")
            st.markdown(f"- **Typical Alert (50% Median):** Cleared in under **{cdf_stats['p50_min']:.0f} mins**")
            st.markdown(f"- **Extended Sortie (75%):** Cleared in under **{cdf_stats['p75_min']:.0f} mins**")
            st.markdown(f"- **Severe Barrage (90%):** Lasts up to **{cdf_stats['p90_min']:.0f} mins**")

        with col_cdf2:
            st.markdown("**Cumulative All-Clear Probability by Elapsed Time:**")
            st.progress(cdf_stats["prob_clear_30m"], text=f"Within 30 min: {cdf_stats['prob_clear_30m']:.0%}")
            st.progress(cdf_stats["prob_clear_60m"], text=f"Within 1 hour: {cdf_stats['prob_clear_60m']:.0%}")
            st.progress(cdf_stats["prob_clear_120m"], text=f"Within 2 hours: {cdf_stats['prob_clear_120m']:.0%}")

        if "cdf_df" in cdf_stats and not cdf_stats["cdf_df"].empty:
            st.write("")
            cdf_chart = (
                alt.Chart(cdf_stats["cdf_df"])
                .mark_area(
                    line={"color": "#38bdf8", "strokeWidth": 2.5},
                    color="rgba(56, 189, 248, 0.20)",
                )
                .encode(
                    x=alt.X("minutes:Q", title="Elapsed Siren Duration (Minutes)", scale=alt.Scale(domain=[0, 180])),
                    y=alt.Y("probability:Q", title="Probability of All-Clear", axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, 1.0])),
                    tooltip=[
                        alt.Tooltip("minutes:Q", title="Minutes Elapsed"),
                        alt.Tooltip("probability:Q", title="All-Clear Probability", format=".1%"),
                    ],
                )
                .properties(height=180)
            )
            rule_df = pd.DataFrame([
                {"m": 30, "label": "30m"},
                {"m": 60, "label": "60m"},
                {"m": int(cdf_stats["p50_min"]), "label": f"Median ({int(cdf_stats['p50_min'])}m)"},
            ])
            rules = alt.Chart(rule_df).mark_rule(color="#94a3b8", strokeDash=[4, 4]).encode(x="m:Q")
            text = alt.Chart(rule_df).mark_text(align="left", dx=5, dy=-65, color="#cbd5e1", fontSize=11).encode(
                x="m:Q", y=alt.value(80), text="label:N"
            )
            st.altair_chart(cdf_chart + rules + text, use_container_width=True)
            st.caption("Cumulative Distribution Function (CDF): Probability of alert termination by elapsed duration.")

        st.divider()

        # GOVERNMENT THREAT LEVEL DIFFERENTIATION SECTION
        st.subheader("Government Threat Classification")
        st.caption(
            "Official Ukrainian air defense categorization enacted in September 2026. "
            "Yellow denotes aerial reconnaissance and drone vectors (Shahed loitering UAVs), while Red denotes direct missile or ballistic strike threats requiring mandatory shelter."
        )

        if not threat_stats["has_threat_data"]:
            st.info("Historical Interval: Alerts in this period were recorded without red/yellow differentiation.")
        else:
            has_unspec = threat_stats["unspecified_count"] > 0
            red_pct_str = f"{threat_stats['red_share']:.0%} of classified" if has_unspec else f"{threat_stats['red_pct']:.0%}"
            yellow_pct_str = f"{threat_stats['yellow_share']:.0%} of classified" if has_unspec else f"{threat_stats['yellow_pct']:.0%}"

            t_c1, t_c2, t_c3 = st.columns(3)
            with t_c1:
                st.markdown(f"""
                <div class="threat-card-red">
                    <div style="font-size:0.9rem; color:#fca5a5; font-weight:700;">RED ALERTS (MISSILE / DIRECT)</div>
                    <div style="font-size:1.6rem; font-weight:800; color:#ffffff; margin:4px 0;">{threat_stats['red_count']:,} ({red_pct_str})</div>
                    <div style="font-size:0.8rem; color:#fecaca;">Avg: {threat_stats['red_avg_min']:.0f} min | Total: {threat_stats['red_total_hours']:.1f} hrs</div>
                </div>
                """, unsafe_allow_html=True)
            with t_c2:
                st.markdown(f"""
                <div class="threat-card-yellow">
                    <div style="font-size:0.9rem; color:#fde047; font-weight:700;">YELLOW ALERTS (DRONE DANGER)</div>
                    <div style="font-size:1.6rem; font-weight:800; color:#ffffff; margin:4px 0;">{threat_stats['yellow_count']:,} ({yellow_pct_str})</div>
                    <div style="font-size:0.8rem; color:#fef08a;">Avg: {threat_stats['yellow_avg_min']:.0f} min | Total: {threat_stats['yellow_total_hours']:.1f} hrs</div>
                </div>
                """, unsafe_allow_html=True)
            with t_c3:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">UNCLASSIFIED / HISTORICAL</div>
                    <div class="kpi-value">{threat_stats['unspecified_count']:,} ({threat_stats['unspecified_pct']:.0%})</div>
                    <div class="kpi-sub">Avg: {threat_stats['unspecified_avg_min']:.0f} min | Total: {threat_stats['unspecified_total_hours']:.1f} hrs</div>
                </div>
                """, unsafe_allow_html=True)

            threat_hourly_df = get_threat_level_hourly_profile(filtered_raw_df, region_slug=region_slug)
            if not threat_hourly_df.empty and threat_hourly_df["Total Classified"].sum() > 0:
                st.write("")
                st.caption(f"Diurnal Strike Vector Profile: 24-hour distribution of loitering drones (Yellow) versus direct missile strikes (Red).")
                melted_threat = threat_hourly_df.melt(
                    id_vars=["Hour", "Total Classified"],
                    value_vars=["Yellow (Drones)", "Red (Missiles)"],
                    var_name="Threat Vector",
                    value_name="Alert Count",
                )
                threat_chart = (
                    alt.Chart(melted_threat)
                    .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
                    .encode(
                        x=alt.X(
                            "Hour:O",
                            title="Hour of Day (Europe/Kyiv Time)",
                            axis=alt.Axis(labelAngle=0, values=[f"{h:02d}:00" for h in range(0, 24, 3)]),
                        ),
                        y=alt.Y("Alert Count:Q", title="Total Alerts Recorded", stack="zero"),
                        color=alt.Color(
                            "Threat Vector:N",
                            scale=alt.Scale(domain=["Yellow (Drones)", "Red (Missiles)"], range=["#eab308", "#ef4444"]),
                            legend=alt.Legend(title="Threat Classification", orient="top"),
                        ),
                        tooltip=["Hour:O", "Threat Vector:N", "Alert Count:Q"],
                    )
                    .properties(height=200)
                )
                st.altair_chart(threat_chart, use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 2: DIURNAL HEATMAPS & TACTICAL SHIFTS
    # --------------------------------------------------------------------------
    with tab_tactics:
        st.subheader("Diurnal Attack Frequency (24×7)")
        st.caption(
            "Heatmap of historical alert probability across all 168 hours of the week (Europe/Kyiv timezone). "
            "Darker cells highlight recurring tactical attack windows when sirens are statistically most frequent."
        )

        # Danger & Safe Window Highlight Cards (Reverted to previous styling)
        h_col1, h_col2 = st.columns(2)
        with h_col1:
            st.markdown(f"""
            <div style="background:#450a0a; border:1px solid #ef4444; border-radius:8px; padding:12px 16px;">
                <b>PEAK DANGER WINDOW:</b> Around <b>{windows['peak_hour']}</b> ({windows['peak_hourly_rate']:.0%} threat rate)<br>
                <span style="font-size:0.85rem; color:#fca5a5;">Highest risk day: <b>{windows['peak_day']}</b> ({windows['peak_day_rate']:.0%} daily average)</span>
            </div>
            """, unsafe_allow_html=True)
        with h_col2:
            st.markdown(f"""
            <div style="background:#064e3b; border:1px solid #10b981; border-radius:8px; padding:12px 16px;">
                <b>STATISTICAL CALM WINDOW:</b> Around <b>{windows['safe_hour']}</b> ({windows['safe_hourly_rate']:.0%} threat rate)<br>
                <span style="font-size:0.85rem; color:#6ee7b7;">Statistically lowest probability of alerts throughout the week in this period.</span>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.altair_chart(render_altair_heatmap(heatmap_df), use_container_width=True)

        st.divider()

        # Day of Week & Duration Breakdown
        col_dow, col_dur = st.columns([1.3, 1.0])
        with col_dow:
            st.subheader("Day of Week Profile")
            st.caption(
                "Aggregated weekday siren volume and shelter duration. Bars indicate average daily hours spent in shelters; the trendline tracks average alert activations."
            )
            dow_df = get_day_of_week_profile(filtered_raw_df, filtered_grid_df, region_slug=region_slug).reset_index()
            # Layered Altair chart: bar for Active Hours / Day and line for Average Alerts / Day
            bar_c = alt.Chart(dow_df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color="rgba(56, 189, 248, 0.45)").encode(
                x=alt.X("Day:O", sort=DAY_NAMES, title="Day of Week"),
                y=alt.Y("Active Hours / Day:Q", title="Daily Shelter Hours"),
                tooltip=["Day:O", "Active Hours / Day:Q", "Average Alerts / Day:Q", "Average Duration (mins):Q"],
            )
            line_c = alt.Chart(dow_df).mark_line(point=True, color="#f59e0b", strokeWidth=2.5).encode(
                x=alt.X("Day:O", sort=DAY_NAMES),
                y=alt.Y("Average Alerts / Day:Q", title="Average Alerts per Day"),
                tooltip=["Day:O", "Active Hours / Day:Q", "Average Alerts / Day:Q", "Average Duration (mins):Q"],
            )
            layered_dow = alt.layer(bar_c, line_c).resolve_scale(y="independent").properties(height=220)
            st.altair_chart(layered_dow, use_container_width=True)

            st.dataframe(
                dow_df.set_index("Day")[["Average Alerts / Day", "Average Duration (mins)", "Active Hours / Day"]],
                use_container_width=True,
            )
        with col_dur:
            st.subheader("Duration Spectrum")
            st.caption(
                "Distribution of siren lengths: short tactical alarms (<30m), medium intercepts (30m-2h), and protracted drone swarms (>2h)."
            )
            st.progress(spectrum["pct_short"], text=f"<30m: {spectrum['pct_short']:.0%}")
            st.progress(spectrum["pct_moderate"], text=f"30m-2h: {spectrum['pct_moderate']:.0%}")
            st.progress(spectrum["pct_extended"], text=f">2h: {spectrum['pct_extended']:.0%}")
            st.write("")
            st.markdown(f"- **Max recorded duration:** `{spectrum['max_hours']:.1f} hours`")
            st.markdown(f"- **Typical quiet window:** `{calm_stats['median_calm_hours']} hours` (Avg: `{calm_stats['avg_calm_hours']} h`)")

        st.divider()

        # LONGITUDINAL WARTIME SHIFT (2022 to 2026)
        st.subheader("Wartime Evolution by Year")
        st.caption(
            "Multi-year progression of strike patterns in this region, demonstrating shifts in daily alert frequency, alarm duration, and nighttime attack concentration since 2022."
        )
        if not evolution_df.empty:
            st.dataframe(
                evolution_df,
                use_container_width=True,
                hide_index=True,
            )
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                chart_evo1 = (
                    alt.Chart(evolution_df)
                    .mark_line(point=True, color="#f59e0b", strokeWidth=2.5)
                    .encode(
                        x=alt.X("Year:O", title="Wartime Year"),
                        y=alt.Y("Avg Alerts / Day:Q", title="Daily Alert Frequency"),
                        tooltip=["Year:O", "Avg Alerts / Day:Q", "Total Alerts:Q", "Tactical Attack Profile:N"],
                    )
                    .properties(height=200, title="Daily Alert Frequency Trend")
                )
                st.altair_chart(chart_evo1, use_container_width=True)
            with col_e2:
                chart_evo2 = (
                    alt.Chart(evolution_df)
                    .mark_line(point=True, color="#38bdf8", strokeWidth=2.5)
                    .encode(
                        x=alt.X("Year:O", title="Wartime Year"),
                        y=alt.Y("Avg Duration (min):Q", title="Avg Duration (min)"),
                        tooltip=["Year:O", "Avg Duration (min):Q", "Nighttime Alerts %:N", "Tactical Attack Profile:N"],
                    )
                    .properties(height=200, title="Average Alert Duration Shift")
                )
                st.altair_chart(chart_evo2, use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 3: TERRITORIAL COMMUNITIES (HROMADAS)
    # --------------------------------------------------------------------------
    with tab_hromada:
        st.subheader("Territorial Communities (Hromadas)")
        st.caption(
            "Granular breakdown of sirens across administrative communities. Differentiates between targeted local sirens affecting individual hromadas versus oblast-wide sirens triggered by strategic missile or aviation threats."
        )

        if not hromada_stats["has_differentiation"]:
            st.info("Historical Period: Alerts for this region were issued oblast-wide without per-community localization.")
        else:
            hc1, hc2, hc3 = st.columns(3)
            with hc1:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">MONITORED HROMADAS</div>
                    <div class="kpi-value">{hromada_stats['hromadas_count']}</div>
                    <div class="kpi-sub">Official communities in {reg_info.name_en}</div>
                </div>
                """, unsafe_allow_html=True)
            with hc2:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">LOCALIZED ALERTS</div>
                    <div class="kpi-value">{hromada_stats['total_localized']:,}</div>
                    <div class="kpi-sub">Confined to a specific community</div>
                </div>
                """, unsafe_allow_html=True)
            with hc3:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">OBLAST-WIDE SIRENS</div>
                    <div class="kpi-value">{hromada_stats['total_oblast_wide']:,}</div>
                    <div class="kpi-sub">Affecting all hromadas simultaneously</div>
                </div>
                """, unsafe_allow_html=True)

            summary_view = hromada_stats["summary_df"]
            top_hromadas = summary_view[summary_view["Localized Alerts"] > 0].head(8)
            if not top_hromadas.empty:
                st.write("")
                st.caption("Top targeted communities by localized sirens (indicating front-line or tactical engagement zones):")
                hromada_chart = (
                    alt.Chart(top_hromadas)
                    .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3, color="rgba(56, 189, 248, 0.75)")
                    .encode(
                        y=alt.Y("Hromada (EN):O", sort="-x", title="Community"),
                        x=alt.X("Localized Alerts:Q", title="Localized Alert Count"),
                        tooltip=["Hromada (EN):O", "Громада (UK):O", "Localized Alerts:Q", "Total Alerts:Q", "Avg Duration (mins):Q"],
                    )
                    .properties(height=max(140, 26 * len(top_hromadas) + 30))
                )
                st.altair_chart(hromada_chart, use_container_width=True)

            st.write("")
            search_hromada = st.text_input("Search Community:", placeholder="Search by English or Ukrainian name...")
            summary_view = hromada_stats["summary_df"]
            if search_hromada:
                q = search_hromada.strip().lower()
                summary_view = summary_view[
                    summary_view["Hromada (EN)"].str.lower().str.contains(q)
                    | summary_view["Громада (UK)"].str.lower().str.contains(q)
                ]

            st.dataframe(summary_view, use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # TAB 4: PREDICTIVE AI & MACHINE LEARNING
    # --------------------------------------------------------------------------
    with tab_ai:
        st.subheader("Predictive Horizon Forecast")
        st.caption(
            "Machine learning lookahead probabilities for siren activation over forward 1-hour, 2-hour, and 3-hour horizons, based on temporal climatology and recent operational activity."
        )

        if filtered_feat_df is not None:
            latest_ts = filtered_feat_df["timestamp"].max()
            reg_feat = filtered_feat_df[
                (filtered_feat_df["region_slug"] == region_slug) & (filtered_feat_df["timestamp"] == latest_ts)
            ]

            if not reg_feat.empty:
                row = reg_feat.iloc[0]
                climatology = get_all_baselines()[1].fit(filtered_feat_df)

                f_c1, f_c2, f_c3 = st.columns(3)
                with f_c1:
                    p1 = float(climatology.predict_proba(pd.DataFrame([row]), horizon_hours=1)[0])
                    st.metric("Alert Risk Next 1 Hour", f"{p1:.1%}", delta="Elevated" if p1 >= 0.5 else "Normal", delta_color="inverse")
                    st.caption("Lookahead [t, t+1h]")
                with f_c2:
                    p2 = float(climatology.predict_proba(pd.DataFrame([row]), horizon_hours=2)[0])
                    st.metric("Alert Risk Next 2 Hours", f"{p2:.1%}", delta="Elevated" if p2 >= 0.5 else "Normal", delta_color="inverse")
                    st.caption("Lookahead [t, t+2h]")
                with f_c3:
                    p3 = float(climatology.predict_proba(pd.DataFrame([row]), horizon_hours=3)[0])
                    st.metric("Alert Risk Next 3 Hours", f"{p3:.1%}", delta="High" if p3 >= 0.6 else "Moderate", delta_color="inverse")
                    st.caption("Lookahead [t, t+3h]")

        st.divider()

        st.subheader("Benchmark Evaluation Matrix")
        st.caption(
            "Out-of-sample evaluation on chronological holdout set, comparing baseline models (Persistent, Hourly Climatology, Day-of-Week Climatology) across Brier Score, ROC AUC, Precision, and Recall."
        )
        if bench_df is not None:
            st.dataframe(bench_df, use_container_width=True, hide_index=True)
        else:
            st.info("Run `python main.py run-benchmark` to generate comparative benchmark metrics.")


# ==============================================================================
# VIEW B: ALL UKRAINE (NATIONAL OVERVIEW)
# ==============================================================================
else:
    st.title("National Situation")
    # st.caption(f"All 25 Regions · {sel_start_date} to {sel_end_date} ({num_days_selected} days)")

    # Pre-calculate active_map and prob_map for national metrics & regional grid
    if filtered_feat_df is not None:
        latest_ts = filtered_feat_df["timestamp"].max()
        latest_slice = filtered_feat_df[filtered_feat_df["timestamp"] == latest_ts].copy()
        climatology = get_all_baselines()[1].fit(filtered_feat_df)
        clim_probs = climatology.predict_proba(latest_slice, horizon_hours=horizon_selected)
        prob_map = dict(zip(latest_slice["region_slug"], clim_probs))
        active_map = dict(zip(latest_slice["region_slug"], latest_slice["feat_active_lag_0"] == 1.0))
    else:
        prob_map = {}
        active_map = {}
        for slug in ALL_REGION_SLUGS:
            sub = filtered_grid_df[filtered_grid_df["region_slug"] == slug]
            prob_map[slug] = float(sub["alert_active"].mean()) if not sub.empty else 0.2
            active_map[slug] = bool(sub.iloc[-1]["alert_active"] == 1) if not sub.empty else False

    if not is_vadimkin_mode:
        active_count = sum(1 for a in active_map.values() if a)
        footprint_pct = active_count / TOTAL_REGIONS_COUNT
        footprint_val = f"{footprint_pct:.0%}"
        footprint_sub = f"{active_count} of {TOTAL_REGIONS_COUNT} regions active"
    else:
        elevated_count = sum(1 for p in prob_map.values() if p >= 0.35)
        footprint_pct = elevated_count / TOTAL_REGIONS_COUNT
        footprint_val = f"{footprint_pct:.0%}"
        footprint_sub = f"{elevated_count} of {TOTAL_REGIONS_COUNT} elevated threat"

    # National summary metrics
    nat_heatmap = get_heatmap_matrix(filtered_grid_df)
    nat_windows = get_peak_and_safe_windows(nat_heatmap)
    nat_spectrum = get_duration_spectrum(filtered_raw_df)
    nat_disruption = get_disruption_scores(filtered_grid_df)
    nat_threat = get_threat_level_stats(filtered_raw_df)
    nat_evolution = get_annual_evolution_stats(filtered_raw_df)

    n1, n2, n3, n4, n5 = st.columns(5)
    with n1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">ACTIVE THREAT FOOTPRINT</div>
            <div class="kpi-value">{footprint_val}</div>
            <div class="kpi-sub">{footprint_sub}</div>
        </div>
        """, unsafe_allow_html=True)
    with n2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">PEAK ATTACK HOUR</div>
            <div class="kpi-value">{nat_windows['peak_hour']}</div>
            <div class="kpi-sub">Nationwide {nat_windows['peak_hourly_rate']:.0%} alert risk</div>
        </div>
        """, unsafe_allow_html=True)
    with n3:
        if nat_threat["has_threat_data"]:
            has_unspec = nat_threat["unspecified_count"] > 0
            if has_unspec:
                threat_label = f"{nat_threat['yellow_share']:.0%} Yellow / {nat_threat['red_share']:.0%} Red"
            else:
                threat_label = f"{nat_threat['yellow_pct']:.0%} Yellow / {nat_threat['red_pct']:.0%} Red"
        else:
            threat_label = "Unclassified"

        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">THREAT LEVEL SPLIT</div>
            <div class="kpi-value" style="font-size:1.25rem;">{threat_label}</div>
            <div class="kpi-sub">Drones (Yellow) vs Missiles (Red)</div>
        </div>
        """, unsafe_allow_html=True)
    with n4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">MEAN DURATION</div>
            <div class="kpi-value">{int(nat_spectrum['mean_min'])} min</div>
            <div class="kpi-sub">{nat_spectrum['pct_short']:.0%} short / {nat_spectrum['pct_extended']:.0%} swarms</div>
        </div>
        """, unsafe_allow_html=True)
    with n5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">SLEEP DISRUPTION</div>
            <div class="kpi-value">{nat_disruption['sleep_disruption_pct']:.0%}</div>
            <div class="kpi-sub">Alarms between 23:00-06:00</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # REGIONAL DANGER GRID (QUICK DIRECT ACCESS TO ALL 25 OBLASTS)
    # st.subheader("Regional Threat")

    if filtered_grid_df is not None:
        # Precompute duration metrics across all regions in a single vectorized pass
        if not filtered_raw_df.empty:
            raw_with_dur = filtered_raw_df.copy()
            dur_mins = (
                pd.to_datetime(raw_with_dur["finished_at"], utc=True) - pd.to_datetime(raw_with_dur["started_at"], utc=True)
            ).dt.total_seconds() / 60.0
            raw_with_dur["duration_min"] = dur_mins
            raw_with_dur["duration_hours"] = dur_mins / 60.0

            agg_by_region = raw_with_dur.groupby("region_slug").agg(
                total_hours=("duration_hours", "sum"),
                mean_duration_min=("duration_min", "mean"),
                alert_count=("duration_min", "count"),
            ).to_dict("index")
        else:
            agg_by_region = {}

        table_rows = []
        for slug in ALL_REGION_SLUGS:
            info = all_regions_dict[slug]
            is_active = active_map.get(slug, False)
            prob = float(prob_map.get(slug, 0.25))

            reg_spectrum = get_duration_spectrum(filtered_raw_df, region_slug=slug)
            reg_threat = get_threat_level_stats(filtered_raw_df, region_slug=slug)
            reg_agg = agg_by_region.get(slug, {"total_hours": 0.0, "mean_duration_min": 0.0, "alert_count": 0})
            total_hours = float(reg_agg["total_hours"])
            mean_min = float(reg_agg["mean_duration_min"])
            alert_count = int(reg_agg["alert_count"])

            has_unspec = reg_threat["unspecified_count"] > 0
            if reg_threat["has_threat_data"]:
                red_pct_str = f"{reg_threat['red_share']:.0%}" if has_unspec else f"{reg_threat['red_pct']:.0%}"
                yellow_pct_str = f"{reg_threat['yellow_share']:.0%}" if has_unspec else f"{reg_threat['yellow_pct']:.0%}"
            else:
                red_pct_str = "Unclassified"
                yellow_pct_str = "Unclassified"

            row_dict = {
                "region_slug": slug,
                "Oblast (EN)": info.name_en,
                "Область (UK)": info.name_uk,
                f"{horizon_selected}h Ahead Alert Risk": f"{prob:.0%}",
                f"{horizon_selected}h Ahead Risk": f"{prob:.0%}",
                "Alert Risk Tier": "High" if prob >= 0.6 else ("Moderate" if prob >= 0.35 else "Low"),
                "Risk Tier": "High" if prob >= 0.6 else ("Moderate" if prob >= 0.35 else "Low"),
                "Total Alert Hours": round(total_hours, 1),
                "Total Alert Hours Formatted": f"{total_hours:,.1f} hrs",
                "Average Alert Time": f"{mean_min:.0f} mins",
                "Mean Alert Time": f"{mean_min:.0f} mins",
                "Alerts Count": f"{alert_count:,}",
                "Yellow (Drones) %": yellow_pct_str,
                "Red (Missiles) %": red_pct_str,
                "Typical Duration": f"{int(reg_spectrum['median_min'])} mins",
                "Short (<30m) %": f"{reg_spectrum['pct_short']:.0%}",
                "Swarm (>2h) %": f"{reg_spectrum['pct_extended']:.0%}",
                "_prob": prob,
                "_total_hours": total_hours,
                "_mean_min": mean_min,
            }
            if not is_vadimkin_mode:
                row_dict["Current Status"] = "SIREN ACTIVE" if is_active else "CLEAR"

            table_rows.append(row_dict)

        # ----------------------------------------------------------------------
        # GEOGRAPHIC MAP & TRACKED OBLAST BURDEN (SIDE-BY-SIDE LAYOUT)
        # ----------------------------------------------------------------------
        st.subheader("Geographic Alert Risk & Tracked Oblasts")
        st.caption("Interactive territorial map alongside tracked oblast cumulative siren downtime and average alert time metrics.")

        col_map, col_list = st.columns([1.35, 1.0], gap="medium")

        with col_map:
            # INTERACTIVE SVG MAP OF UKRAINE (BIDIRECTIONAL COMPONENT)
            clicked_slug = render_ukraine_svg_map(
                table_rows,
                is_vadimkin_mode=is_vadimkin_mode,
                horizon_hours=horizon_selected,
            )
            if clicked_slug and clicked_slug in ALL_REGION_SLUGS:
                st.session_state["selected_target"] = clicked_slug
                st.query_params["region"] = clicked_slug
                st.rerun()

            st.caption("Click any region on the map or in the list to inspect its detailed workspace.")

        with col_list:
            max_hours = max((r["_total_hours"] for r in table_rows), default=0.0)
            max_region = next((r["Oblast (EN)"] for r in table_rows if r["_total_hours"] == max_hours), "N/A")
            nat_avg_dur = np.mean([r["_mean_min"] for r in table_rows if r["_mean_min"] > 0]) if table_rows else 0.0

            # st.markdown("##### Tracked Oblasts Alert Burden")
            c_kpi1, c_kpi2 = st.columns(2)
            with c_kpi1:
                st.markdown(f"""
                <div class="kpi-card" style="height:76px; min-height:76px; padding:6px 10px; margin-bottom:12px;">
                    <div class="kpi-title">MAX TOTAL ALERT TIME</div>
                    <div class="kpi-value" style="font-size:1.15rem;">{max_hours:,.0f} hrs</div>
                    <div class="kpi-sub">{max_region}</div>
                </div>
                """, unsafe_allow_html=True)
            with c_kpi2:
                st.markdown(f"""
                <div class="kpi-card" style="height:76px; min-height:76px; padding:6px 10px; margin-bottom:12px;">
                    <div class="kpi-title">MEAN ALERT DURATION</div>
                    <div class="kpi-value" style="font-size:1.15rem;">{nat_avg_dur:.0f} mins</div>
                    <div class="kpi-sub">Across 25 oblasts</div>
                </div>
                """, unsafe_allow_html=True)

            ranking_df = pd.DataFrame(table_rows).sort_values(by="_total_hours", ascending=False).reset_index(drop=True)
            list_cols = [
                "Oblast (EN)",
                "Total Alert Hours",
                "Average Alert Time",
                "Alert Risk Tier",
            ]

            # Color table cells by risk tier and status
            def style_risk_cells(val: str) -> str:
                if val == "High":
                    return "background-color: rgba(239, 68, 68, 0.35); color: #fca5a5; font-weight: 600;"
                elif val == "Moderate":
                    return "background-color: rgba(245, 158, 11, 0.35); color: #fcd34d; font-weight: 600;"
                elif val == "Low":
                    return "background-color: rgba(16, 185, 129, 0.30); color: #6ee7b7; font-weight: 600;"
                return ""

            styled_ranking = ranking_df[list_cols].style.map(style_risk_cells, subset=["Alert Risk Tier"])

            list_column_config = {
                "Oblast (EN)": st.column_config.TextColumn("Tracked Oblast", width="medium"),
                "Total Alert Hours": st.column_config.ProgressColumn(
                    "Total Alert Hours",
                    help="Cumulative air raid siren hours in the selected period",
                    format="%.1f hrs",
                    min_value=0,
                    max_value=float(max_hours) if max_hours > 0 else 100,
                ),
                "Average Alert Time": st.column_config.TextColumn(
                    "Mean Alert Time",
                    help="Average duration per air raid siren",
                    width="small",
                ),
                "Alert Risk Tier": st.column_config.TextColumn(
                    "Alert Risk",
                    help="Forecasted alert risk tier",
                    width="small",
                ),
            }

            list_event = st.dataframe(
                styled_ranking,
                column_config=list_column_config,
                use_container_width=True,
                height=490,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="main_page_oblast_list",
            )

            if (
                list_event
                and hasattr(list_event, "selection")
                and list_event.selection
                and list_event.selection.get("rows")
            ):
                sel_idx = list_event.selection["rows"][0]
                if 0 <= sel_idx < len(ranking_df):
                    selected_slug = ranking_df.iloc[sel_idx]["region_slug"]
                    if selected_slug in ALL_REGION_SLUGS:
                        st.session_state["selected_target"] = selected_slug
                        st.query_params["region"] = selected_slug
                        st.rerun()

        st.divider()

        # ----------------------------------------------------------------------
        # COMPREHENSIVE REGIONAL THREAT MATRIX
        # ----------------------------------------------------------------------
        st.subheader("Regional Threat Matrix")
        st.caption("Comprehensive tactical breakdown across all 25 regions, including threat classifications, duration spectrums, and live operational status.")

        summary_df = pd.DataFrame(table_rows).sort_values(by="_prob", ascending=False).reset_index(drop=True)
        cols_to_show = ["Oblast (EN)", "Область (UK)"]
        if not is_vadimkin_mode:
            cols_to_show.append("Current Status")
        cols_to_show.extend([
            f"{horizon_selected}h Ahead Alert Risk",
            "Alert Risk Tier",
            "Total Alert Hours",
            "Average Alert Time",
            "Yellow (Drones) %",
            "Red (Missiles) %",
            "Typical Duration",
            "Short (<30m) %",
            "Swarm (>2h) %",
        ])

        def style_status_cells(val: str) -> str:
            if "ACTIVE" in str(val):
                return "background-color: rgba(239, 68, 68, 0.40); color: #fca5a5; font-weight: 700;"
            elif "CLEAR" in str(val):
                return "background-color: rgba(16, 185, 129, 0.25); color: #6ee7b7; font-weight: 600;"
            return ""

        display_df = summary_df[cols_to_show]
        styled_df = display_df.style.map(style_risk_cells, subset=["Alert Risk Tier"])
        if not is_vadimkin_mode and "Current Status" in cols_to_show:
            styled_df = styled_df.map(style_status_cells, subset=["Current Status"])

        table_column_config = {
            "Total Alert Hours": st.column_config.NumberColumn("Total Alert Hours", format="%.1f hrs"),
            "Average Alert Time": st.column_config.TextColumn("Mean Alert Time"),
        }

        table_event = st.dataframe(
            styled_df,
            column_config=table_column_config,
            use_container_width=True,
            height=460,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="national_matrix_table",
        )

        if (
            table_event
            and hasattr(table_event, "selection")
            and table_event.selection
            and table_event.selection.get("rows")
        ):
            selected_row_idx = table_event.selection["rows"][0]
            if 0 <= selected_row_idx < len(summary_df):
                row_slug = summary_df.iloc[selected_row_idx]["region_slug"]
                if row_slug in ALL_REGION_SLUGS:
                    st.session_state["selected_target"] = row_slug
                    st.query_params["region"] = row_slug
                    st.rerun()

    st.divider()

    # NATIONWIDE 7X24 INTERACTIVE HEATMAP
    st.subheader("Nationwide Density Heatmap")
    st.caption("Aggregated alert frequency across all regions by hour and day of week.")
    st.altair_chart(render_altair_heatmap(nat_heatmap), use_container_width=True)

    st.divider()

    # LONGITUDINAL WARTIME EVOLUTION NATIONWIDE
    st.subheader("Nationwide Wartime Evolution")
    if not nat_evolution.empty:
        st.dataframe(nat_evolution, use_container_width=True, hide_index=True)
