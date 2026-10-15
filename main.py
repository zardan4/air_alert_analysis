"""Main CLI application entrypoint for Ukraine Air Raid Alert Forecasting."""

import argparse
import sys
from pathlib import Path
import pandas as pd

from src.baselines.naive_baselines import get_all_baselines
from src.config import load_config
from src.data import (
    AlertsInUaDataSource,
    IntervalToGridConverter,
    MockAlertDataSource,
    VadimkinDataSource,
    get_alert_data_source,
)
from src.evaluation.benchmark import BenchmarkRunner, print_benchmark_report
from src.features.feature_engineer import PointInTimeFeatureEngineer
from src.splits.temporal_splitter import PurgedWalkForwardSplitter
from src.utils.regions import ALL_REGION_SLUGS, TOTAL_REGIONS_COUNT, get_region


def is_mock_mode(args) -> bool:
    """Check if mock mode was activated via root or subcommand flags."""
    return (
        getattr(args, "cmd_mock", False)
        or getattr(args, "global_mock", False)
        or (getattr(args, "source", None) == "mock")
    )


def is_vadimkin_mode(args) -> bool:
    """Check if Vadimkin open dataset mode was activated via root or subcommand flags."""
    return (
        getattr(args, "cmd_vadimkin", False)
        or getattr(args, "global_vadimkin", False)
        or (getattr(args, "source", None) == "vadimkin")
    )


def cmd_status(args):
    """Display system status, configuration, and data summary."""
    import os
    config = load_config()
    is_mock = is_mock_mode(args)
    is_vadimkin = is_vadimkin_mode(args)
    print("=" * 65)
    print("  UKRAINE AIR RAID ALERT FORECASTING SYSTEM")
    print("=" * 65)
    print(f"Project Name:        {config.project_name} (v{config.version})")
    if is_mock:
        active_mode_str = "[MOCK MODE] (Operating with synthetic alert data)"
    elif is_vadimkin:
        active_mode_str = "[VADIMKIN MODE] (Operating with Vadimkin open sirens dataset)"
    else:
        active_mode_str = "[ALERTS.IN.UA MODE] (Operating with authentic alerts.in.ua records)"
    print(f"Active Mode:         {active_mode_str}")
    print(f"Default Frequency:   {config.grid.frequency}")
    print(f"Timezone:            {config.timezone}")
    print(f"Registered Regions:  {TOTAL_REGIONS_COUNT} (24 oblasts + Kyiv City)")
    print(f"Prediction Horizons: {config.backtesting.horizons_hours} hours ahead")
    print(f"Test Window:         {config.backtesting.test_window_days} days (chronological holdout)")
    print(f"Embargo Buffer:      {config.backtesting.embargo_hours} hours")
    print("-" * 65)

    dump_path = Path(config.data.alerts_in_ua_dump_path)
    vadimkin_dump_path = Path(config.data.vadimkin_dump_path)
    api_token_configured = bool(config.alerts_in_ua.api_token or os.environ.get(config.alerts_in_ua.token_env_var))
    print("Data Connectors & Ingestion:")
    active_source = "mock" if is_mock else ("vadimkin" if is_vadimkin else config.data.default_source)
    print(f" - Active Source:    {active_source}")
    print(f" - alerts.in.ua Dump:{' [EXISTS] ' + str(dump_path) if dump_path.exists() else ' [NOT FOUND] ' + str(dump_path)}")
    print(f" - alerts.in.ua API: {' [CONFIGURED]' if api_token_configured else ' [NOT SET] (set ALERTS_IN_UA_TOKEN)'}")
    print(f" - Vadimkin Dump:    {' [EXISTS] ' + str(vadimkin_dump_path) if vadimkin_dump_path.exists() else ' [NOT FOUND] ' + str(vadimkin_dump_path)}")
    print("-" * 65)

    raw_path = Path("data/raw/alerts_raw.parquet")
    grid_path = Path("data/processed/alerts_grid.parquet")
    features_path = Path("data/processed/alerts_features.parquet")

    print("Data Assets:")
    raw_status = '[NOT FOUND]'
    if raw_path.exists():
        raw_df_preview = pd.read_parquet(raw_path)
        sources = list(raw_df_preview['source'].unique()) if 'source' in raw_df_preview.columns else ['unknown']
        raw_status = f"[EXISTS] ({len(raw_df_preview):,} intervals, source: {', '.join(sources)})"
    print(f" - Raw Alerts:       {raw_status}")
    print(f" - Continuous Grid:  {'[EXISTS] ' + str(grid_path) if grid_path.exists() else '[NOT FOUND]'}")
    print(f" - Feature Matrix:   {'[EXISTS] ' + str(features_path) if features_path.exists() else '[NOT FOUND]'}")

    if grid_path.exists():
        df = pd.read_parquet(grid_path)
        print("-" * 65)
        print(f"Continuous Grid Summary:")
        print(f" - Time Span:        {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f" - Total Observations: {len(df):,}")
        print(f" - Active Incidence: {df['alert_active'].mean():.2%}")
        print(f" - Onset Incidence:  {df['alert_onset'].mean():.2%}")

    print("=" * 65)


def cmd_generate_data(args):
    """Generate raw alert intervals and convert to continuous uniform grid."""
    config = load_config()
    freq = args.frequency or config.grid.frequency
    target_regions = [get_region(args.region).slug] if args.region else ALL_REGION_SLUGS

    is_mock = is_mock_mode(args)
    is_vadimkin = is_vadimkin_mode(args)

    if is_mock:
        print(f"[*] [MOCK MODE] Ingesting/Generating synthetic alert data for {len(target_regions)} region(s)...")
        source = MockAlertDataSource()
        raw_df = source.load_raw_alerts(
            start_date=args.start_date,
            end_date=args.end_date,
            regions=target_regions,
        )
    elif is_vadimkin:
        print(f"[*] [VADIMKIN MODE] Ingesting actual alert data from Vadimkin open dataset for {len(target_regions)} region(s)...")
        source = VadimkinDataSource(
            csv_path=getattr(args, "vadimkin_path", None) or getattr(args, "dump_path", None),
        )
        raw_df = source.load_raw_alerts(
            start_date=args.start_date,
            end_date=args.end_date,
            regions=target_regions,
        )
    else:
        print(f"[*] [ALERTS.IN.UA MODE] Ingesting actual alert data from alerts.in.ua for {len(target_regions)} region(s)...")
        source = AlertsInUaDataSource(
            dump_path=getattr(args, "dump_path", None),
            api_token=getattr(args, "api_token", None),
        )
        raw_df = source.load_raw_alerts(
            start_date=args.start_date,
            end_date=args.end_date,
            regions=target_regions,
            use_api=getattr(args, "live", False),
        )

    print(f"[+] Loaded {len(raw_df):,} raw alert intervals (source: {raw_df['source'].iloc[0] if not raw_df.empty else 'empty'}).")

    raw_out = Path(args.raw_output)
    source.save_raw_parquet(raw_df, raw_out)
    print(f"[+] Raw data saved to {raw_out}")

    print(f"[*] Converting raw intervals into uniform continuous grid ({freq} resolution)...")
    converter = IntervalToGridConverter(frequency=freq)
    grid_out = Path(args.grid_output)
    grid_df = converter.convert_and_save(
        raw_df=raw_df,
        output_path=grid_out,
        start_date=args.start_date,
        end_date=args.end_date,
        regions=target_regions,
    )
    print(f"[+] Continuous grid successfully built: {grid_out}")
    print(f"[+] Grid rows: {len(pd.read_parquet(grid_out)):,}")


def cmd_build_features(args):
    """Execute causal point-in-time feature engineering pipeline."""
    grid_path = Path(args.input_grid)
    if not grid_path.exists():
        print(f"[!] Error: Grid file not found at {grid_path}. Run 'generate-data' first.")
        sys.exit(1)

    print(f"[*] Loading continuous grid from {grid_path}...")
    grid_df = pd.read_parquet(grid_path)

    print(f"[*] Constructing causal point-in-time features & multi-horizon targets...")
    engineer = PointInTimeFeatureEngineer()
    res = engineer.transform(grid_df, include_targets=True, drop_na_targets=False)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res.data.to_parquet(out_path, index=False)

    print(f"[+] Features successfully constructed and saved to {out_path}")
    print(f"[+] Total Samples:   {len(res.data):,}")
    print(f"[+] Feature Count:   {len(res.feature_columns)} features")
    print(f"[+] Target Count:    {len(res.target_columns)} targets across horizons {res.horizons_hours}h")
    print("\nFeature Categories Generated:")
    print(" - Temporal:  hour_sin, hour_cos, day_sin, day_cos, is_weekend, is_night_strike_window")
    print(" - Lags:      active_lag_[0,1,2,3,6,12,24], onset_lag_[0,1,2,3,6], ratio_lag_[0,1,2,3]")
    print(" - Rolling:   active_hours_[6,24,72,168]h, active_rate_[6,24,72,168]h, onsets_count_[6,24,72,168]h, hours_since_last_onset")
    print(" - Spatial:   nationwide_active_count, neighbors_active_count, inflow_border_active_count")


def cmd_check_splits(args):
    """Run walk-forward anti-leakage splitting engine and verify temporal integrity."""
    grid_path = Path(args.input_grid)
    if not grid_path.exists():
        print(f"[!] Error: Grid file not found at {grid_path}. Run 'generate-data' first.")
        sys.exit(1)

    print(f"[*] Loading continuous grid from {grid_path}...")
    grid_df = pd.read_parquet(grid_path)

    splitter = PurgedWalkForwardSplitter(
        max_horizon_hours=3,
        embargo_hours=3,
        test_window_days=args.test_days,
        val_window_days=args.val_days,
        min_train_days=args.min_train_days,
    )

    print(f"[*] Validating dedicated chronological holdout split ({args.test_days} days test)...")
    holdout = splitter.create_holdout_split(grid_df)
    print(f"[+] Holdout Train/Val Window: {holdout.train_val_df['timestamp'].min()} to {holdout.train_val_df['timestamp'].max()}")
    print(f"[+] Purged Buffer:           {holdout.separation_buffer_hours:.1f} hours ({holdout.purged_rows_count} rows dropped)")
    print(f"[+] Dedicated Test Window:   {holdout.test_df['timestamp'].min()} to {holdout.test_df['timestamp'].max()}")
    print("[+] Zero temporal leakage verified between Train/Val and Test!")

    print(f"\n[*] Generating Purged Walk-Forward CV Folds (Validation: {args.val_days} days)...")
    folds = splitter.create_cv_splits(holdout.train_val_df)
    for f in folds:
        print(f"  Fold {f.fold_idx}: Train [{f.train_start.strftime('%Y-%m-%d')} .. {f.train_end.strftime('%Y-%m-%d')}] "
              f"({len(f.train_idx):,} rows) -> Buffer ({f.separation_buffer_hours:.1f}h) -> "
              f"Val [{f.val_start.strftime('%Y-%m-%d')} .. {f.val_end.strftime('%Y-%m-%d')}] ({len(f.val_idx):,} rows)")
    print(f"[+] All {len(folds)} folds passed mathematical anti-leakage integrity verification!")


def main():
    parser = argparse.ArgumentParser(
        description="Ukraine Air Raid Alert Forecasting Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mock", dest="global_mock", action="store_true", default=False, help="Run program in mock synthetic mode")
    parser.add_argument("--vadimkin", dest="global_vadimkin", action="store_true", default=False, help="Run program in Vadimkin open sirens dataset mode")

    common_mode_parser = argparse.ArgumentParser(add_help=False)
    common_mode_parser.add_argument(
        "--mock",
        dest="cmd_mock",
        action="store_true",
        default=False,
        help="Run in mock mode using synthetic data generator",
    )
    common_mode_parser.add_argument(
        "--vadimkin",
        dest="cmd_vadimkin",
        action="store_true",
        default=False,
        help="Run in Vadimkin open sirens dataset mode (no API token required)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Command: status
    subparsers.add_parser("status", parents=[common_mode_parser], help="Show system status and data asset overview")

    # Command: generate-data
    p_gen = subparsers.add_parser("generate-data", parents=[common_mode_parser], help="Ingest/generate alert records and build continuous grid")
    p_gen.add_argument("--source", type=str, choices=["alerts_in_ua", "vadimkin", "mock"], default=None, help="Explicit data source to ingest from ('alerts_in_ua', 'vadimkin', or 'mock')")
    p_gen.add_argument("--dump-path", type=str, default=None, help="Path to local alerts.in.ua dump file (CSV, JSON, Parquet)")
    p_gen.add_argument("--vadimkin-path", type=str, default=None, help="Path to custom Vadimkin open sirens CSV dataset")
    p_gen.add_argument("--api-token", type=str, default=None, help="API token for alerts.in.ua live connector")
    p_gen.add_argument("--live", action="store_true", default=False, help="Fetch real-time active alerts directly from alerts.in.ua API")
    p_gen.add_argument("--start-date", type=str, default=None, help="Start date (ISO 8601, e.g. 2024-01-01)")
    p_gen.add_argument("--end-date", type=str, default=None, help="End date (ISO 8601, e.g. 2024-04-30)")
    p_gen.add_argument("--frequency", type=str, default=None, help="Temporal grid resolution (e.g. '1h', '15min')")
    p_gen.add_argument("--region", type=str, default=None, help="Isolate a specific oblast slug (optional)")
    p_gen.add_argument("--raw-output", type=str, default="data/raw/alerts_raw.parquet", help="Raw Parquet output")
    p_gen.add_argument("--grid-output", type=str, default="data/processed/alerts_grid.parquet", help="Grid Parquet output")

    # Command: build-features
    p_feat = subparsers.add_parser("build-features", parents=[common_mode_parser], help="Construct causal point-in-time features & targets")
    p_feat.add_argument("--input-grid", type=str, default="data/processed/alerts_grid.parquet", help="Input grid path")
    p_feat.add_argument("--output", type=str, default="data/processed/alerts_features.parquet", help="Output feature matrix")

    # Command: check-splits
    p_split = subparsers.add_parser("check-splits", parents=[common_mode_parser], help="Test anti-leakage purging and walk-forward CV splits")
    p_split.add_argument("--input-grid", type=str, default="data/processed/alerts_grid.parquet", help="Input grid path")
    p_split.add_argument("--test-days", type=int, default=30, help="Holdout test window duration in days")
    p_split.add_argument("--val-days", type=int, default=14, help="Validation fold duration in days")
    p_split.add_argument("--min-train-days", type=int, default=14, help="Minimum initial training window duration in days")

    # Command: run-benchmark
    p_bench = subparsers.add_parser("run-benchmark", parents=[common_mode_parser], help="Run all naive baselines and print evaluation report")
    p_bench.add_argument("--input-features", type=str, default="data/processed/alerts_features.parquet", help="Input feature matrix")
    p_bench.add_argument("--test-days", type=int, default=14, help="Holdout test window duration in days")
    p_bench.add_argument("--target-type", type=str, default="active", choices=["active", "onset"], help="Target type")
    p_bench.add_argument("--per-oblast", action="store_true", help="Include per-oblast evaluation")

    # Command: ui
    p_ui = subparsers.add_parser("ui", parents=[common_mode_parser], help="Launch interactive Streamlit dashboard")
    p_ui.add_argument("--port", type=int, default=8501, help="Port to run Streamlit on (default: 8501)")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "status": cmd_status,
        "generate-data": cmd_generate_data,
        "build-features": cmd_build_features,
        "check-splits": cmd_check_splits,
        "run-benchmark": cmd_run_benchmark,
        "ui": cmd_ui,
    }
    dispatch[args.command](args)


def cmd_ui(args):
    """Launch interactive Streamlit dashboard."""
    import subprocess
    is_mock = is_mock_mode(args)
    is_vadimkin = is_vadimkin_mode(args)
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.port",
        str(args.port),
    ]
    if is_mock:
        cmd.extend(["--", "--mock"])
    elif is_vadimkin:
        cmd.extend(["--", "--vadimkin"])
    mode_descr = "mock mode" if is_mock else ("vadimkin mode" if is_vadimkin else "alerts.in.ua mode")
    print(f"[*] Launching Streamlit dashboard on http://localhost:{args.port} ({mode_descr}) ...")
    subprocess.run(cmd)


def cmd_run_benchmark(args):
    """Run all naive baselines on test set and print comparative evaluation report."""
    feat_path = Path(args.input_features)
    if not feat_path.exists():
        print(f"[!] Error: Feature file not found at {feat_path}. Run 'build-features' first.")
        sys.exit(1)

    print(f"[*] Loading feature matrix from {feat_path}...")
    feat_df = pd.read_parquet(feat_path)

    splitter = PurgedWalkForwardSplitter(
        max_horizon_hours=3,
        embargo_hours=3,
        test_window_days=args.test_days,
    )
    holdout = splitter.create_holdout_split(feat_df)
    train_df = holdout.train_val_df
    test_df = holdout.test_df
    print(f"[+] Train/Val: {len(train_df):,} rows | Test: {len(test_df):,} rows | Purged buffer: {holdout.separation_buffer_hours:.1f}h")

    runner = BenchmarkRunner(
        horizons_hours=[1, 2, 3],
        evaluate_per_oblast=args.per_oblast,
        target_type=args.target_type,
    )
    report = runner.run(train_df, test_df)
    print_benchmark_report(report)

    # Save to CSV
    out_path = Path("data/processed/benchmark_results.csv")
    report.national_results.to_csv(out_path, index=False)
    print(f"\n[+] Results saved to {out_path}")


if __name__ == "__main__":
    main()
