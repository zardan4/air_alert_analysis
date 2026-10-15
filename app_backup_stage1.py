"""Interactive User-Friendly Air Raid Alert Forecasting & Civil Safety Dashboard."""

from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st

from src.baselines.naive_baselines import get_all_baselines
from src.config import load_config
from src.ui.stats import (
    DAY_ABBR,
    DAY_NAMES,
    get_calm_and_downtime_stats,
    get_day_of_week_profile,
    get_disruption_scores,
    get_duration_spectrum,
    get_heatmap_matrix,
    get_hromada_breakdown,
    get_peak_and_safe_windows,
    get_threat_level_stats,
)
from src.utils.regions import ALL_REGION_SLUGS, TOTAL_REGIONS_COUNT, get_all_regions, get_region

# Page configuration
st.set_page_config(
    page_title="Ukraine Air Raid Forecast & Civil Safety",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .status-badge-active {
        background-color: #dc2626;
        color: white;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        box-shadow: 0 0 12px rgba(220, 38, 38, 0.5);
    }
    .status-badge-yellow {
        background-color: #d97706;
        color: white;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        box-shadow: 0 0 12px rgba(217, 119, 6, 0.5);
    }
    .status-badge-clear {
        background-color: #16a34a;
        color: white;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .kpi-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .threat-card-red {
        background-color: #450a0a;
        border: 1px solid #dc2626;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .threat-card-yellow {
        background-color: #451a03;
        border: 1px solid #d97706;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .kpi-title { font-size: 0.85rem; color: #94a3b8; margin-bottom: 4px; }
    .kpi-value { font-size: 1.4rem; font-weight: 700; color: #f8fafc; }
    .kpi-sub { font-size: 0.75rem; color: #cbd5e1; }
    .danger-card {
        background-color: #450a0a;
        border: 1px solid #991b1b;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    .safe-card {
        background-color: #064e3b;
        border: 1px solid #065f46;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    .interval-pill {
        background-color: #0f172a;
        border: 1px solid #3b82f6;
        color: #93c5fd;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        display: inline-block;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


import sys

# Detect data mode from CLI arguments
is_mock_cli = "--mock" in sys.argv
is_vadimkin_cli = "--vadimkin" in sys.argv


@st.cache_data
def load_cached_data(mock_mode: bool = False, vadimkin_mode: bool = False):
    """Load raw alerts, continuous grid, and features. Automatically generates data if CLI flags are passed and files are absent."""
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

# Check for data availability
if grid_df is None or raw_df is None:
    st.error(" Data files not found. Please run `python main.py generate-data` first, or launch with `--mock` or `--vadimkin`.")
    st.stop()

# Detect active source mode (mock vs alerts.in.ua vs vadimkin)
sources_in_data = list(raw_df["source"].unique()) if raw_df is not None and "source" in raw_df.columns else []
is_mock_source = all("mock" in str(s).lower() for s in sources_in_data) if sources_in_data else False
is_vadimkin_source = any("vadimkin" in str(s).lower() for s in sources_in_data) if sources_in_data else False
is_mock_mode = is_mock_cli or is_mock_source
is_vadimkin_mode = is_vadimkin_cli or is_vadimkin_source

# ==============================================================================
# LEFT SIDEBAR: MENU & CONTROLS
# ==============================================================================
st.sidebar.title("🛡️ Air Raid Analytics")

# Mode indicator badge
if is_mock_mode:
    st.sidebar.info("🧪 **Mode: Mock Synthetic Data**\nOperating in mock/synthetic data mode.")
elif is_vadimkin_mode:
    st.sidebar.success("📊 **Mode: Vadimkin Open Sirens Dataset**\nOperating with Vadimkin GitHub dataset (zero-token open access).")
else:
    st.sidebar.success("🛡️ **Mode: alerts.in.ua Official Archive**\nOperating with authentic Ukrainian air raid records.")

# 1. Region Selector
st.sidebar.markdown("### 📍 Location")
oblast_options = ["ALL_UKRAINE"] + ALL_REGION_SLUGS

def format_oblast_option(opt: str) -> str:
    if opt == "ALL_UKRAINE":
        return " All Ukraine (National Overview)"
    reg = all_regions_dict[opt]
    return f"{reg.name_en} ({reg.name_uk})"

selected_target = st.sidebar.selectbox(
    "Choose Region to Analyze:",
    oblast_options,
    index=0,
    format_func=format_oblast_option,
    help="Select an individual oblast to view its danger heatmap, duration spectrum, and forecast."
)

st.sidebar.divider()

# 2. Time Interval Selector
st.sidebar.markdown("###  Analysis Time Interval")

# Find dataset date boundaries
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
    sel_start_date = min_avail_date
    sel_end_date = max_avail_date
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
else:  # Custom Date Range
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

# Date validation
if sel_start_date > sel_end_date:
    sel_start_date, sel_end_date = sel_end_date, sel_start_date

num_days_selected = max(1, (sel_end_date - sel_start_date).days + 1)
st.sidebar.caption(f"Active window: **{sel_start_date}** to **{sel_end_date}** ({num_days_selected} days)")

st.sidebar.divider()

# 3. Horizon Selector
st.sidebar.markdown("###  Forecast Lookahead")
horizon_selected = st.sidebar.radio(
    "Forecast Window:",
    [1, 2, 3],
    index=0,
    format_func=lambda h: f"{h} Hour Ahead ({h}h)",
    horizontal=True,
)

st.sidebar.divider()
st.sidebar.caption(" Data covers all 24 oblasts and Kyiv City.")

# ==============================================================================
# DATA FILTERING BY SELECTED INTERVAL
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
    st.warning(" No alert records found in the selected interval. Please select a broader date range in the left sidebar.")
    st.stop()


# ==============================================================================
# VIEW A: SPECIFIC OBLAST DRILLDOWN
# ==============================================================================
if selected_target != "ALL_UKRAINE":
    reg_info = all_regions_dict[selected_target]
    region_slug = selected_target

    # 1. Header Banner & Interval Pill
    col_title, col_status = st.columns([2.2, 1.0])
    with col_title:
        st.title(f"{reg_info.name_en}")
        st.subheader(f"*{reg_info.name_uk}* — Civil Safety & Risk Analysis")
        st.markdown(
            f'<div class="interval-pill"> Analyzing data interval: <b>{sel_start_date}</b> to <b>{sel_end_date}</b> ({num_days_selected} days)</div>',
            unsafe_allow_html=True,
        )

    # 2. Compute stats for this region within the selected interval
    calm_stats = get_calm_and_downtime_stats(filtered_raw_df, filtered_grid_df, region_slug=region_slug)
    spectrum = get_duration_spectrum(filtered_raw_df, region_slug=region_slug)
    disruption = get_disruption_scores(filtered_grid_df, region_slug=region_slug)
    heatmap_df = get_heatmap_matrix(filtered_grid_df, region_slug=region_slug)
    windows = get_peak_and_safe_windows(heatmap_df)
    threat_stats = get_threat_level_stats(filtered_raw_df, region_slug=region_slug)
    hromada_stats = get_hromada_breakdown(filtered_raw_df, region_slug=region_slug)

    # Real-time state badge & threat categorization
    region_raw = filtered_raw_df[filtered_raw_df["region_slug"] == region_slug]
    latest_alert = region_raw.sort_values(by="started_at", ascending=False).iloc[0] if not region_raw.empty else None
    current_threat = "red"
    current_hromada = "oblast_wide"
    if latest_alert is not None:
        if "threat_level" in latest_alert:
            current_threat = str(latest_alert["threat_level"]).lower()
        if "hromada" in latest_alert:
            current_hromada = str(latest_alert["hromada"])

    with col_status:
        st.write("")
        if calm_stats["is_active_now"]:
            if current_threat == "yellow":
                st.markdown('<div class="status-badge-yellow">🟡 ACTIVE YELLOW ALERT</div>', unsafe_allow_html=True)
                st.caption("Heightened Danger / Aviation Activity")
            else:
                st.markdown('<div class="status-badge-active">🔴 ACTIVE RED ALERT</div>', unsafe_allow_html=True)
                st.caption("Direct Strike Threat / Sirens Sounding")
            if current_hromada not in ("oblast_wide", "unspecified"):
                st.caption(f"📍 Localized to: **{current_hromada}**")
            else:
                st.caption("📢 Scope: Oblast-wide siren")
        else:
            st.markdown('<div class="status-badge-clear">✅ CLEAR (NO ACTIVE ALERT)</div>', unsafe_allow_html=True)
            st.caption(f"Cutoff: {filtered_grid_df['timestamp'].max().strftime('%Y-%m-%d %H:%M UTC')}")

    st.write("")

    # 3. Top KPI Metric Strip
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title"> TYPICAL DURATION</div>
            <div class="kpi-value">{int(spectrum['median_min'])} min</div>
            <div class="kpi-sub">Avg: {int(spectrum['mean_min'])} min</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title"> DAILY SHELTER TIME</div>
            <div class="kpi-value">{calm_stats['daily_shelter_hours']} h/day</div>
            <div class="kpi-sub">Time under sirens</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title"> CALM BETWEEN ALERTS</div>
            <div class="kpi-value">{calm_stats['avg_calm_hours']} h</div>
            <div class="kpi-sub">Current: {calm_stats['current_calm_hours']}h calm</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title"> SLEEP DISRUPTION</div>
            <div class="kpi-value">{disruption['sleep_disruption_pct']:.0%}</div>
            <div class="kpi-sub">Alerts between 23:00-06:00</div>
        </div>
        """, unsafe_allow_html=True)
    with m5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title"> WORKDAY INTERRUPTED</div>
            <div class="kpi-value">{disruption['workday_disruption_pct']:.0%}</div>
            <div class="kpi-sub">Mon-Fri (09:00 - 18:00)</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # 3.5 SECTION: GOVERNMENT THREAT LEVEL DIFFERENTIATION
    st.subheader(f"🔴 / 🟡 Government Threat Level Breakdown for {reg_info.name_en}")
    st.markdown(
        "Ukrainian government regulations (enacted September 2026) distinguish between **🟡 Yellow Alerts** "
        "(дронова небезпека / ударні БПЛА / Shahed drones, aviation activity) and **🔴 Red Alerts** "
        "(ракетна та балістична небезпека, mandatory immediate shelter). "
        "Prior to the enactment in September 2026 (including all of 2022–2025), alerts were strictly general sirens without color differentiation."
    )

    if not threat_stats["has_threat_data"]:
        st.info(
            "ℹ️ **Historical / Unclassified Interval**: For this analyzed date range (e.g. 2022–2025), "
            "alerts in Ukraine were recorded without red/yellow government differentiation. "
            "All events correctly default to unclassified standard sirens."
        )
    else:
        # Determine whether to show classified share vs total percentage
        has_unspec = threat_stats["unspecified_count"] > 0
        red_pct_str = f"{threat_stats['red_share']:.0%} of classified" if has_unspec else f"{threat_stats['red_pct']:.0%}"
        yellow_pct_str = f"{threat_stats['yellow_share']:.0%} of classified" if has_unspec else f"{threat_stats['yellow_pct']:.0%}"

        t_c1, t_c2, t_c3 = st.columns(3)
        with t_c1:
            st.markdown(f"""
            <div class="threat-card-red">
                <div style="font-size:0.9rem; color:#fca5a5; font-weight:700;">🔴 RED ALERTS (MISSILE / DIRECT)</div>
                <div style="font-size:1.6rem; font-weight:800; color:#ffffff; margin:4px 0;">{threat_stats['red_count']:,} ({red_pct_str})</div>
                <div style="font-size:0.8rem; color:#fecaca;">Avg: {threat_stats['red_avg_min']:.0f} min | Total: {threat_stats['red_total_hours']:.1f} hrs</div>
                <div style="font-size:0.75rem; color:#fda4af; margin-top:4px;">Ракетна / балістична небезпека (обов'язкове укриття)</div>
            </div>
            """, unsafe_allow_html=True)
        with t_c2:
            st.markdown(f"""
            <div class="threat-card-yellow">
                <div style="font-size:0.9rem; color:#fde047; font-weight:700;">🟡 YELLOW ALERTS (DRONE DANGER)</div>
                <div style="font-size:1.6rem; font-weight:800; color:#ffffff; margin:4px 0;">{threat_stats['yellow_count']:,} ({yellow_pct_str})</div>
                <div style="font-size:0.8rem; color:#fef08a;">Avg: {threat_stats['yellow_avg_min']:.0f} min | Total: {threat_stats['yellow_total_hours']:.1f} hrs</div>
                <div style="font-size:0.75rem; color:#fef9c3; margin-top:4px;">Дронова небезпека (БПЛА / «Шахед»), підвищена увага</div>
            </div>
            """, unsafe_allow_html=True)
        with t_c3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">⚪ UNCLASSIFIED / HISTORICAL</div>
                <div class="kpi-value">{threat_stats['unspecified_count']:,} ({threat_stats['unspecified_pct']:.0%})</div>
                <div class="kpi-sub">Avg: {threat_stats['unspecified_avg_min']:.0f} min | Total: {threat_stats['unspecified_total_hours']:.1f} hrs</div>
                <div style="font-size:0.75rem; color:#94a3b8; margin-top:4px;">Historical intervals prior to Sept 2026 reform</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # 4. SECTION 1: 7x24 DIURNAL RISK HEATMAP
    st.subheader(f" When are sirens most likely in {reg_info.name_en}?")
    st.markdown(
        f"This heatmap displays the **historical probability of an air raid alert** for every hour of the week in {reg_info.name_en} "
        f"over the selected period (**{sel_start_date}** to **{sel_end_date}**, `Europe/Kyiv` time). "
        "Darker red/orange colors represent high-probability strike windows."
    )

    # Highlight cards for peak vs safe windows
    h_col1, h_col2 = st.columns(2)
    with h_col1:
        st.markdown(f"""
        <div class="danger-card">
            <b> PEAK DANGER WINDOW:</b> Around <b>{windows['peak_hour']}</b> ({windows['peak_hourly_rate']:.0%} threat rate)<br>
            <span style="font-size:0.85rem;">Highest risk day: <b>{windows['peak_day']}</b> ({windows['peak_day_rate']:.0%} daily average)</span>
        </div>
        """, unsafe_allow_html=True)
    with h_col2:
        st.markdown(f"""
        <div class="safe-card">
            <b> SAFEST CALM WINDOW:</b> Around <b>{windows['safe_hour']}</b> ({windows['safe_hourly_rate']:.0%} threat rate)<br>
            <span style="font-size:0.85rem;">Statistically lowest probability of alerts throughout the week in this period.</span>
        </div>
        """, unsafe_allow_html=True)

    # Render Styled Heatmap Table
    st.dataframe(
        heatmap_df.style.background_gradient(cmap="YlOrRd", vmin=0.0, vmax=1.0).format("{:.0%}"),
        use_container_width=True,
    )

    st.divider()

    # 5. SECTION 2: DAY OF WEEK PROFILE & DURATION SPECTRUM
    col_dow, col_spectrum = st.columns([1.3, 1.0])

    with col_dow:
        st.subheader(" Weekly Attack Profile (By Day of Week)")
        st.caption(f"Calculated over the {num_days_selected}-day interval ({sel_start_date} to {sel_end_date})")
        dow_profile = get_day_of_week_profile(filtered_raw_df, filtered_grid_df, region_slug=region_slug)

        # Plot bar chart of Average Alerts / Day
        st.bar_chart(dow_profile["Average Alerts / Day"], color="#3b82f6")
        st.dataframe(
            dow_profile[["Average Alerts / Day", "Average Duration (mins)", "Active Hours / Day"]],
            use_container_width=True,
        )

    with col_spectrum:
        st.subheader(" Alert Duration Breakdown")
        st.markdown(f"Distribution of all `{spectrum['total_alerts']:,}` alerts in {reg_info.name_en} during this period:")

        pct_s = spectrum["pct_short"]
        pct_m = spectrum["pct_moderate"]
        pct_e = spectrum["pct_extended"]

        st.progress(pct_s, text=f" Short (< 30 min): {pct_s:.0%} (Tactical sorties / MiG-31K)")
        st.progress(pct_m, text=f" Moderate (30m - 2h): {pct_m:.0%} (Missile strikes / cruise)")
        st.progress(pct_e, text=f" Extended (> 2h): {pct_e:.0%} (Shahed drone swarm barrages)")

        st.write("")
        st.markdown(f"- **Total alerts in selected interval:** `{spectrum['total_alerts']:,}`")
        st.markdown(f"- **Longest single alert recorded:** `{spectrum['max_hours']:.1f} hours`")
        st.markdown(f"- **Typical calm between sirens:** `{calm_stats['avg_calm_hours']} hours`")

    st.divider()

    # 6. SECTION 3: MULTI-HORIZON FORWARD FORECAST
    st.subheader(f" Forward Probability Forecast for {reg_info.name_en}")
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
                st.metric("Risk Next 1 Hour", f"{p1:.1%}", delta="Elevated" if p1>=0.5 else "Normal", delta_color="inverse")
                st.caption("Probability of siren starting or active in [t, t+1h]")
            with f_c2:
                p2 = float(climatology.predict_proba(pd.DataFrame([row]), horizon_hours=2)[0])
                st.metric("Risk Next 2 Hours", f"{p2:.1%}", delta="Elevated" if p2>=0.5 else "Normal", delta_color="inverse")
                st.caption("Probability of siren within next 2 hours")
            with f_c3:
                p3 = float(climatology.predict_proba(pd.DataFrame([row]), horizon_hours=3)[0])
                st.metric("Risk Next 3 Hours", f"{p3:.1%}", delta="High" if p3>=0.6 else "Moderate", delta_color="inverse")
                st.caption("Probability of siren within next 3 hours")

    st.divider()

    # 7. SECTION 4: TERRITORIAL COMMUNITY (HROMADA) DIFFERENTIATION
    st.subheader(f"🏛️ Territorial Community (Hromada) Risk Breakdown in {reg_info.name_en}")
    st.markdown(
        f"Detailed drilldown for territorial communities within **{reg_info.name_en}** (*{reg_info.name_uk}*). "
        "Alerts can be localized to a specific community or announced oblast-wide (affecting all communities simultaneously)."
    )

    if not hromada_stats["has_differentiation"]:
        st.info(
            f"ℹ️ **Historical / Oblast-Wide Alert Period**: In this analyzed date range ({sel_start_date} to {sel_end_date}), "
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
                <div class="kpi-sub">Sirens confined to specific community</div>
            </div>
            """, unsafe_allow_html=True)
        with hc3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">OBLAST-WIDE SIRENS</div>
                <div class="kpi-value">{hromada_stats['total_oblast_wide']:,}</div>
                <div class="kpi-sub">Affecting all hromadas in the oblast</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.dataframe(
            hromada_stats["summary_df"],
            use_container_width=True,
            hide_index=True,
        )


# ==============================================================================
# VIEW B: ALL UKRAINE (NATIONAL OVERVIEW)
# ==============================================================================
else:
    st.title("🛡️ Ukraine Air Raid Alerts — National Overview")
    st.markdown(
        f'<div class="interval-pill"> Analyzing nationwide data interval: <b>{sel_start_date}</b> to <b>{sel_end_date}</b> ({num_days_selected} days)</div>',
        unsafe_allow_html=True,
    )

    # National summary stats for the filtered period
    nat_heatmap = get_heatmap_matrix(filtered_grid_df)
    nat_windows = get_peak_and_safe_windows(nat_heatmap)
    nat_spectrum = get_duration_spectrum(filtered_raw_df)
    nat_disruption = get_disruption_scores(filtered_grid_df)
    nat_threat = get_threat_level_stats(filtered_raw_df)

    n1, n2, n3, n4, n5 = st.columns(5)
    with n1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">MONITORED OBLASTS</div>
            <div class="kpi-value">{TOTAL_REGIONS_COUNT}</div>
            <div class="kpi-sub">24 Oblasts + Kyiv City</div>
        </div>
        """, unsafe_allow_html=True)
    with n2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">PEAK THREAT HOUR</div>
            <div class="kpi-value">{nat_windows['peak_hour']}</div>
            <div class="kpi-sub">Avg {nat_windows['peak_hourly_rate']:.0%} alert incidence</div>
        </div>
        """, unsafe_allow_html=True)
    with n3:
        threat_label = f"{nat_threat['red_pct']:.0%} Red / {nat_threat['yellow_pct']:.0%} Yellow" if nat_threat["has_threat_data"] else "Unclassified"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">THREAT LEVEL RATIO</div>
            <div class="kpi-value" style="font-size:1.15rem;">{threat_label}</div>
            <div class="kpi-sub">🔴 Direct Sirens vs 🟡 Patrols</div>
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
            <div class="kpi-sub">Alerts between 23:00-06:00</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Table of all 25 regions with user-friendly metrics for the selected interval
    st.subheader(f"📊 All 25 Oblasts Risk & Shelter Overview ({sel_start_date} to {sel_end_date})")
    st.caption("Top-level administrative overview. Drill down into any individual oblast using the left sidebar to explore granular territorial communities (hromadas).")

    if filtered_feat_df is not None:
        latest_ts = filtered_feat_df["timestamp"].max()
        latest_slice = filtered_feat_df[filtered_feat_df["timestamp"] == latest_ts].copy()

        climatology = get_all_baselines()[1].fit(filtered_feat_df)
        clim_probs = climatology.predict_proba(latest_slice, horizon_hours=horizon_selected)
        latest_slice["pred_prob"] = clim_probs

        table_rows = []
        for _, r in latest_slice.iterrows():
            slug = r["region_slug"]
            info = all_regions_dict[slug]
            is_active = bool(r["feat_active_lag_0"] == 1.0)
            prob = float(r["pred_prob"])

            # Regional spectrum & threat stats for this interval
            reg_spectrum = get_duration_spectrum(filtered_raw_df, region_slug=slug)
            reg_threat = get_threat_level_stats(filtered_raw_df, region_slug=slug)

            red_pct_str = f"{reg_threat['red_pct']:.0%}" if reg_threat["has_threat_data"] else "N/A"
            yellow_pct_str = f"{reg_threat['yellow_pct']:.0%}" if reg_threat["has_threat_data"] else "N/A"

            table_rows.append({
                "Oblast (EN)": info.name_en,
                "Область (UK)": info.name_uk,
                "Current Status": "🚨 SIREN" if is_active else "✅ CLEAR",
                f"{horizon_selected}h Forecast": f"{prob:.0%}",
                "Risk Level": "HIGH" if prob >= 0.7 else ("MODERATE" if prob >= 0.4 else "LOW"),
                "Red Alerts %": red_pct_str,
                "Yellow Alerts %": yellow_pct_str,
                "Typical Duration": f"{int(reg_spectrum['median_min'])} mins",
                "Short (<30m) %": f"{reg_spectrum['pct_short']:.0%}",
                "Swarm (>2h) %": f"{reg_spectrum['pct_extended']:.0%}",
                "_prob": prob,
            })

        summary_df = pd.DataFrame(table_rows).sort_values(by="_prob", ascending=False)
        st.dataframe(
            summary_df[[
                "Oblast (EN)", "Область (UK)", "Current Status", f"{horizon_selected}h Forecast",
                "Risk Level", "Red Alerts %", "Yellow Alerts %", "Typical Duration", "Short (<30m) %", "Swarm (>2h) %"
            ]],
            use_container_width=True,
            height=600,
            hide_index=True,
        )

    st.divider()

    # National Heatmap
    st.subheader("Nationwide Alert Density Heatmap")
    st.markdown(f"Average alert probability across all regions of Ukraine from **{sel_start_date}** to **{sel_end_date}**:")
    st.dataframe(
        nat_heatmap.style.background_gradient(cmap="YlOrRd", vmin=0.0, vmax=1.0).format("{:.0%}"),
        use_container_width=True,
    )
