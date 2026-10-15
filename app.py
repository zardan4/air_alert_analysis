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
    page_title="Ukraine Air Raid Alert Forecasting & Risk Analysis Platform",
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
        background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%);
        color: white;
        padding: 8px 18px;
        border-radius: 8px;
        font-weight: 800;
        font-size: 1.15rem;
        display: inline-block;
        box-shadow: 0 0 16px rgba(239, 68, 68, 0.6);
        animation: pulse-red 2s infinite;
    }
    .status-badge-yellow {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 8px 18px;
        border-radius: 8px;
        font-weight: 800;
        font-size: 1.15rem;
        display: inline-block;
        box-shadow: 0 0 16px rgba(245, 158, 11, 0.5);
    }
    .status-badge-clear {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 8px 18px;
        border-radius: 8px;
        font-weight: 800;
        font-size: 1.15rem;
        display: inline-block;
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.4);
    }

    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
        70% { box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }

    /* Glassmorphic Cards */
    .kpi-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(71, 85, 105, 0.4);
        backdrop-filter: blur(8px);
        border-radius: 10px;
        padding: 16px 14px;
        text-align: center;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .kpi-card:hover {
        border-color: #38bdf8;
        transform: translateY(-2px);
    }
    .kpi-title { font-size: 0.8rem; font-weight: 600; color: #94a3b8; letter-spacing: 0.05em; margin-bottom: 6px; }
    .kpi-value { font-size: 1.5rem; font-weight: 800; color: #f8fafc; }
    .kpi-sub { font-size: 0.75rem; color: #cbd5e1; margin-top: 4px; }

    /* Threat Cards */
    .threat-card-red {
        background: linear-gradient(145deg, rgba(69, 10, 10, 0.8) 0%, rgba(40, 10, 10, 0.8) 100%);
        border: 1px solid #ef4444;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .threat-card-yellow {
        background: linear-gradient(145deg, rgba(69, 26, 3, 0.8) 0%, rgba(45, 20, 5, 0.8) 100%);
        border: 1px solid #f59e0b;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }

    /* Safety Verdict Cards */
    .verdict-card-safe {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.8) 0%, rgba(4, 47, 46, 0.8) 100%);
        border: 1px solid #10b981;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.2);
    }
    .verdict-card-caution {
        background: linear-gradient(135deg, rgba(120, 53, 15, 0.8) 0%, rgba(69, 26, 3, 0.8) 100%);
        border: 1px solid #f59e0b;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 0 20px rgba(245, 158, 11, 0.2);
    }
    .verdict-card-danger {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.8) 0%, rgba(69, 10, 10, 0.8) 100%);
        border: 1px solid #ef4444;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.25);
    }

    /* Window Highlight Pill */
    .interval-pill {
        background-color: #0f172a;
        border: 1px solid #38bdf8;
        color: #bae6fd;
        padding: 5px 12px;
        border-radius: 6px;
        font-size: 0.85rem;
        display: inline-block;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Detect CLI parameters
is_mock_cli = "--mock" in sys.argv
is_vadimkin_cli = "--vadimkin" in sys.argv


@st.cache_data
def load_cached_data(mock_mode: bool = False, vadimkin_mode: bool = False):
    """Load raw alerts, continuous grid, and features. Automatically generates data if files are absent."""
    raw_path = Path("data/raw/alerts_raw.parquet")
    grid_path = Path("data/processed/alerts_grid.parquet")
    feat_path = Path("data/processed/alerts_features.parquet")
    bench_path = Path("data/processed/benchmark_results.csv")

    raw = pd.read_parquet(raw_path) if raw_path.exists() else None
    grid = pd.read_parquet(grid_path) if grid_path.exists() else None
    feat = pd.read_parquet(feat_path) if feat_path.exists() else None
    bench = pd.read_csv(bench_path) if bench_path.exists() else None

    if (raw is None or grid is None) and mock_mode:
        from src.data.mock_source import MockAlertDataSource
        from src.data.grid_converter import IntervalToGridConverter
        source = MockAlertDataSource(seed=42)
        raw = source.load_raw_alerts(start_date="2024-01-01", end_date="2024-03-31")
        converter = IntervalToGridConverter(frequency="1h")
        grid = converter.convert(raw_df=raw, start_date="2024-01-01", end_date="2024-03-31")
    elif (raw is None or grid is None) and vadimkin_mode:
        from src.data.vadimkin import VadimkinDataSource
        from src.data.grid_converter import IntervalToGridConverter
        source = VadimkinDataSource()
        raw = source.load_raw_alerts(start_date="2024-01-01", end_date="2024-03-31")
        converter = IntervalToGridConverter(frequency="1h")
        grid = converter.convert(raw_df=raw, start_date="2024-01-01", end_date="2024-03-31")

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
# LEFT SIDEBAR: MENU & CONTROLS
# ==============================================================================
# LEFT SIDEBAR: MENU & CONTROLS
# ==============================================================================
st.sidebar.title("Air Raid Intelligence System")

# Mode indicator badge
if is_mock_mode:
    st.sidebar.info("**Mode: Synthetic Baseline Simulation**\nOperating in synthetic demonstration mode.")
elif is_vadimkin_mode:
    st.sidebar.success("**Mode: Historical Archive (Vadimkin Sirens Dataset)**\nPublic wartime records.")
else:
    st.sidebar.success("**Mode: Official Archive (alerts.in.ua)**\nOfficial Ukrainian air raid records.")

# 1. Location Selector
st.sidebar.markdown("### Administrative Region Selection")
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
    "Choose Region to Analyze:",
    oblast_options,
    format_func=format_oblast_option,
    key="region_selector_key",
    on_change=handle_region_selection_change,
    help="Select an individual oblast to view its activity planner, threat heatmap, and community breakdown."
)

st.sidebar.divider()

# 2. Time Interval Selector
st.sidebar.markdown("### Temporal Window Selection")
min_avail_date = pd.to_datetime(grid_df["timestamp"]).min().date()
max_avail_date = pd.to_datetime(grid_df["timestamp"]).max().date()

interval_mode = st.sidebar.radio(
    "Select Interval Range:",
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
st.sidebar.caption(f"Active window: **{sel_start_date}** to **{sel_end_date}** ({num_days_selected} days)")

st.sidebar.divider()

# 3. Forecast Lookahead Horizon
# st.sidebar.markdown("### Forecast Horizon Selection")
horizon_selected = st.sidebar.radio(
    "Target Prediction Horizon:",
    [1, 2, 3],
    index=0,
    format_func=lambda h: f"{h} Hour Ahead ({h}h)",
    horizontal=True,
)
st.sidebar.caption(f"Used in order to calculate the risk of the air alert in the closest hours.")

st.sidebar.divider()
st.sidebar.caption("Coverage: 24 administrative oblasts and Kyiv City.")


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
                legend=alt.Legend(title="Siren Risk", format=".0%"),
            ),
            tooltip=[
                alt.Tooltip("day:O", title="Day of Week"),
                alt.Tooltip("time_window:N", title="Time Slot (Kyiv Time)"),
                alt.Tooltip("probability:Q", title="Historical Alert Probability", format=".1%"),
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
    col_back, _ = st.columns([1.8, 4])
    with col_back:
        if st.button("Return to National Overview (All Ukraine)", key="btn_return_national", use_container_width=True):
            st.session_state["selected_target"] = "ALL_UKRAINE"
            st.query_params.pop("region", None)
            st.rerun()

    # Top Situation Banner
    col_head, col_state = st.columns([2.2, 1.0])
    with col_head:
        st.title(reg_info.name_en)
        st.subheader(f"{reg_info.name_uk} | Regional Threat Assessment & Risk Analytics")
        st.markdown(
            f'<div class="interval-pill">Analysis Interval: <b>{sel_start_date}</b> to <b>{sel_end_date}</b> ({num_days_selected} days)</div>',
            unsafe_allow_html=True,
        )

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
                '<span style="color: #94a3b8; font-size: 0.8rem; font-weight: 700; letter-spacing: 0.05em;">HISTORICAL DATASET</span><br>'
                '<span style="color: #cbd5e1; font-size: 0.75rem;">Vadimkin Open Sirens Dataset</span>'
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
        "Civil Defense & Activity Planner",
        "Temporal & Diurnal Attack Patterns",
        "Territorial Communities (Hromadas)",
        "Statistical & Machine Learning Forecasts",
    ])

    # --------------------------------------------------------------------------
    # TAB 1: CIVIL SAFETY & MEETING PLANNER
    # --------------------------------------------------------------------------
    with tab_safety:
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
        st.markdown(
            "Plan transit, essential business meetings, facility operations, and family commutes safely. "
            "Select intended schedule parameters below to receive an empirical risk verdict and statistical guidance."
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
        col_cdf1, col_cdf2 = st.columns([1.1, 1.0])
        with col_cdf1:
            st.subheader("Empirical Siren Duration Expectations")
            st.markdown(
                f"Empirical survival distribution for alarms recorded in **{reg_info.name_en}**:"
            )
            st.markdown(f"- **Fast Intercept (25th Percentile):** Cleared in under **{cdf_stats['p25_min']:.0f} mins**")
            st.markdown(f"- **Median Alert Duration (50th Percentile):** Cleared in under **{cdf_stats['p50_min']:.0f} mins**")
            st.markdown(f"- **Extended Sortie (75th Percentile):** Cleared in under **{cdf_stats['p75_min']:.0f} mins**")
            st.markdown(f"- **Severe Barrage (90th Percentile):** Lasts up to **{cdf_stats['p90_min']:.0f} mins**")

        with col_cdf2:
            st.subheader("Probability of All-Clear by Milestone")
            st.progress(cdf_stats["prob_clear_30m"], text=f"Cleared within 30 min: {cdf_stats['prob_clear_30m']:.0%}")
            st.progress(cdf_stats["prob_clear_60m"], text=f"Cleared within 1 hour: {cdf_stats['prob_clear_60m']:.0%}")
            st.progress(cdf_stats["prob_clear_120m"], text=f"Cleared within 2 hours: {cdf_stats['prob_clear_120m']:.0%}")
            st.caption("Based on all historical alarms recorded for this region.")

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

        st.divider()

        # GOVERNMENT THREAT LEVEL DIFFERENTIATION SECTION
        st.subheader(f"Government Threat Level Classification for {reg_info.name_en}")
        st.markdown(
            "Ukrainian government regulations (Decree enacted September 2026) distinguish between **Yellow Alerts** "
            "(дронова небезпека / ударні БПЛА / Shahed drones, aviation activity) and **Red Alerts** "
            "(ракетна та балістична небезпека, mandatory immediate shelter). "
            "Prior to enactment in September 2026 (including all of 2022–2025), alerts were strictly general sirens without color differentiation."
        )

        if not threat_stats["has_threat_data"]:
            st.info(
                "Historical / Unclassified Interval: For this analyzed date range (2022–2025), "
                "alerts in Ukraine were recorded without red/yellow government differentiation. "
                "All events default to unclassified standard sirens."
            )
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
                    <div style="font-size:0.75rem; color:#fda4af; margin-top:4px;">Ракетна / балістична небезпека (обов'язкове укриття)</div>
                </div>
                """, unsafe_allow_html=True)
            with t_c2:
                st.markdown(f"""
                <div class="threat-card-yellow">
                    <div style="font-size:0.9rem; color:#fde047; font-weight:700;">YELLOW ALERTS (DRONE DANGER)</div>
                    <div style="font-size:1.6rem; font-weight:800; color:#ffffff; margin:4px 0;">{threat_stats['yellow_count']:,} ({yellow_pct_str})</div>
                    <div style="font-size:0.8rem; color:#fef08a;">Avg: {threat_stats['yellow_avg_min']:.0f} min | Total: {threat_stats['yellow_total_hours']:.1f} hrs</div>
                    <div style="font-size:0.75rem; color:#fef9c3; margin-top:4px;">Дронова небезпека (БПЛА / «Шахед»), підвищена увага</div>
                </div>
                """, unsafe_allow_html=True)
            with t_c3:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">UNCLASSIFIED / HISTORICAL</div>
                    <div class="kpi-value">{threat_stats['unspecified_count']:,} ({threat_stats['unspecified_pct']:.0%})</div>
                    <div class="kpi-sub">Avg: {threat_stats['unspecified_avg_min']:.0f} min | Total: {threat_stats['unspecified_total_hours']:.1f} hrs</div>
                    <div style="font-size:0.75rem; color:#94a3b8; margin-top:4px;">Historical intervals prior to Sept 2026 reform</div>
                </div>
                """, unsafe_allow_html=True)

            threat_hourly_df = get_threat_level_hourly_profile(filtered_raw_df, region_slug=region_slug)
            if not threat_hourly_df.empty and threat_hourly_df["Total Classified"].sum() > 0:
                st.write("")
                st.caption(f"Diurnal Strike Vector Profile for {reg_info.name_en} (Kyiv Time) — Drones (Yellow) vs Missiles (Red):")
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
        st.subheader(f"Diurnal Attack Frequency Matrix (24x7) for {reg_info.name_en}")
        st.markdown(
            "Historical probability of an active air raid siren for every hour of the week (`Europe/Kyiv` local time). "
            "Hover over any square to view exact threat probability."
        )

        # Danger & Safe Window Highlight Cards
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
            st.subheader("Attack Frequency & Shelter Burden by Day of Week")
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
            st.subheader("Duration Spectrum Analysis")
            st.markdown(f"Tactical profile across **{spectrum['total_alerts']:,}** alerts in {reg_info.name_en}:")
            st.progress(spectrum["pct_short"], text=f"Short (<30m): {spectrum['pct_short']:.0%} (Ballistics / MiG-31K)")
            st.progress(spectrum["pct_moderate"], text=f"Moderate (30m-2h): {spectrum['pct_moderate']:.0%} (Missile strikes)")
            st.progress(spectrum["pct_extended"], text=f"Extended (>2h): {spectrum['pct_extended']:.0%} (Shahed drone barrages)")
            st.write("")
            st.markdown(f"- **Max recorded alert duration:** `{spectrum['max_hours']:.1f} hours`")
            st.markdown(f"- **Typical quiet window:** `{calm_stats['median_calm_hours']} hours` (Avg: `{calm_stats['avg_calm_hours']} h`)")

        st.divider()

        # LONGITUDINAL WARTIME SHIFT (2022 to 2026)
        st.subheader("Longitudinal Wartime Evolution (2022 to 2026)")
        st.markdown(
            f"Analyzing how threat profiles, night-escalation, and siren duration in **{reg_info.name_en}** evolved across wartime years:"
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
        st.subheader(f"Territorial Community (Hromada) Threat Distribution in {reg_info.name_en}")
        st.markdown(
            f"Granular territorial analysis for registered communities within **{reg_info.name_en}** (*{reg_info.name_uk}*). "
            "Alerts can be localized to a specific community or announced oblast-wide (affecting all communities simultaneously)."
        )

        if not hromada_stats["has_differentiation"]:
            st.info(
                f"Historical / Oblast-Wide Alert Period: In this analyzed date range ({sel_start_date} to {sel_end_date}), "
                f"all alerts for {reg_info.name_en} were issued strictly oblast-wide. "
                "Granular territorial community (hromada) alerting was either not yet implemented during this historical period "
                "or not recorded for this date range."
            )
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
                st.caption(f"Top Targeted Communities by Localized Sirens in {reg_info.name_en}:")
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
            search_hromada = st.text_input("Search Community (English or Ukrainian):", placeholder="e.g. Uman, Bucha, Brovary...")
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
        st.subheader(f"Probabilistic Forward Horizon Forecast for {reg_info.name_en}")
        st.markdown(
            "Causal point-in-time probabilistic model assessing alert risk over upcoming lookahead windows:"
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
                    st.metric("Risk Next 1 Hour", f"{p1:.1%}", delta="Elevated" if p1 >= 0.5 else "Normal", delta_color="inverse")
                    st.caption("Probability of siren starting or active in [t, t+1h]")
                with f_c2:
                    p2 = float(climatology.predict_proba(pd.DataFrame([row]), horizon_hours=2)[0])
                    st.metric("Risk Next 2 Hours", f"{p2:.1%}", delta="Elevated" if p2 >= 0.5 else "Normal", delta_color="inverse")
                    st.caption("Probability of siren within next 2 hours")
                with f_c3:
                    p3 = float(climatology.predict_proba(pd.DataFrame([row]), horizon_hours=3)[0])
                    st.metric("Risk Next 3 Hours", f"{p3:.1%}", delta="High" if p3 >= 0.6 else "Moderate", delta_color="inverse")
                    st.caption("Probability of siren within next 3 hours")

        st.divider()

        st.subheader("Benchmark Evaluation Matrix")
        st.markdown(
            "Strict out-of-sample evaluation on chronologically purged holdout window (zero temporal leakage):"
        )
        if bench_df is not None:
            st.dataframe(bench_df, use_container_width=True, hide_index=True)
        else:
            st.info("Run `python main.py run-benchmark` to generate comparative benchmark metrics.")


# ==============================================================================
# VIEW B: ALL UKRAINE (NATIONAL OVERVIEW)
# ==============================================================================
else:
    st.title("Ukraine Air Raid Alerts — National Situation Room")
    st.markdown(
        f'<div class="interval-pill">Nationwide Interval: <b>{sel_start_date}</b> to <b>{sel_end_date}</b> ({num_days_selected} days)</div>',
        unsafe_allow_html=True,
    )

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
            <div class="kpi-sub">Nationwide {nat_windows['peak_hourly_rate']:.0%} risk</div>
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
            <div class="kpi-value" style="font-size:1.15rem;">{threat_label}</div>
            <div class="kpi-sub">Drones (Yellow) vs Missiles (Red)</div>
        </div>
        """, unsafe_allow_html=True)
    with n4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">MEDIAN DURATION</div>
            <div class="kpi-value">{int(nat_spectrum['median_min'])} min</div>
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
    st.subheader("All 25 Ukrainian Administrative Regions")
    if is_vadimkin_mode:
        st.markdown(
            "Historical and forecast risk profile across all 24 oblasts and Kyiv City "
            "(Vadimkin open dataset — real-time monitoring inactive):"
        )
    else:
        st.markdown("Current real-time and forecast risk status across all 24 oblasts and Kyiv City:")

    if filtered_grid_df is not None:
        table_rows = []
        for slug in ALL_REGION_SLUGS:
            info = all_regions_dict[slug]
            is_active = active_map.get(slug, False)
            prob = float(prob_map.get(slug, 0.25))

            reg_spectrum = get_duration_spectrum(filtered_raw_df, region_slug=slug)
            reg_threat = get_threat_level_stats(filtered_raw_df, region_slug=slug)

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
                f"{horizon_selected}h Ahead Risk": f"{prob:.0%}",
                "Risk Tier": "High" if prob >= 0.6 else ("Moderate" if prob >= 0.35 else "Low"),
                "Yellow (Drones) %": yellow_pct_str,
                "Red (Missiles) %": red_pct_str,
                "Typical Duration": f"{int(reg_spectrum['median_min'])} mins",
                "Short (<30m) %": f"{reg_spectrum['pct_short']:.0%}",
                "Swarm (>2h) %": f"{reg_spectrum['pct_extended']:.0%}",
                "_prob": prob,
            }
            if not is_vadimkin_mode:
                row_dict["Current Status"] = "SIREN ACTIVE" if is_active else "CLEAR"

            table_rows.append(row_dict)

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

        st.write("")
        st.caption("Regional Risk Intelligence Matrix (Click any oblast on the map above or select a row in the table below to open its workspace):")

        summary_df = pd.DataFrame(table_rows).sort_values(by="_prob", ascending=False).reset_index(drop=True)
        cols_to_show = ["Oblast (EN)", "Область (UK)"]
        if not is_vadimkin_mode:
            cols_to_show.append("Current Status")
        cols_to_show.extend([
            f"{horizon_selected}h Ahead Risk",
            "Risk Tier",
            "Yellow (Drones) %",
            "Red (Missiles) %",
            "Typical Duration",
            "Short (<30m) %",
            "Swarm (>2h) %",
        ])

        # Color table cells by risk tier and status
        def style_risk_cells(val: str) -> str:
            if val == "High":
                return "background-color: rgba(239, 68, 68, 0.35); color: #fca5a5; font-weight: 600;"
            elif val == "Moderate":
                return "background-color: rgba(245, 158, 11, 0.35); color: #fcd34d; font-weight: 600;"
            elif val == "Low":
                return "background-color: rgba(16, 185, 129, 0.30); color: #6ee7b7; font-weight: 600;"
            return ""

        def style_status_cells(val: str) -> str:
            if "ACTIVE" in str(val):
                return "background-color: rgba(239, 68, 68, 0.40); color: #fca5a5; font-weight: 700;"
            elif "CLEAR" in str(val):
                return "background-color: rgba(16, 185, 129, 0.25); color: #6ee7b7; font-weight: 600;"
            return ""

        display_df = summary_df[cols_to_show]
        styled_df = display_df.style.map(style_risk_cells, subset=["Risk Tier"])
        if not is_vadimkin_mode and "Current Status" in cols_to_show:
            styled_df = styled_df.map(style_status_cells, subset=["Current Status"])

        table_event = st.dataframe(
            styled_df,
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
    st.subheader("Nationwide 7×24 Alert Density Heatmap")
    st.markdown(
        f"Aggregated alert incidence across all Ukrainian regions from **{sel_start_date}** to **{sel_end_date}**. "
        "Hover over any square to view time slot risk."
    )
    st.altair_chart(render_altair_heatmap(nat_heatmap), use_container_width=True)

    st.divider()

    # LONGITUDINAL WARTIME EVOLUTION NATIONWIDE
    st.subheader("Nationwide Wartime Shift Timeline (2022 to 2026)")
    st.markdown("Annual breakdown of total alerts, daily rate, and tactical shift nationwide:")
    if not nat_evolution.empty:
        st.dataframe(nat_evolution, use_container_width=True, hide_index=True)
