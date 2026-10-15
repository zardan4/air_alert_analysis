"""Script to generate raw alerts data (mock or dump) and convert into a continuous uniform grid."""

import argparse
from pathlib import Path
import pandas as pd

from src.config import load_config
from src.data import AlertsInUaDataSource, IntervalToGridConverter, MockAlertDataSource
from src.utils.regions import ALL_REGION_SLUGS, get_region


def main():
    parser = argparse.ArgumentParser(description="Generate air raid alert data and build continuous time-series grid")
    parser.add_argument("--mock", action="store_true", default=False, help="Run in mock mode using synthetic generator")
    parser.add_argument("--source", type=str, choices=["alerts_in_ua", "mock"], default=None, help="Explicit data source ('alerts_in_ua' or 'mock')")
    parser.add_argument("--dump-path", type=str, default=None, help="Path to alerts.in.ua dump file")
    parser.add_argument("--api-token", type=str, default=None, help="API token for alerts.in.ua")
    parser.add_argument("--start-date", type=str, default=None, help="Start date (ISO 8601, e.g. 2024-01-01)")
    parser.add_argument("--end-date", type=str, default=None, help="End date (ISO 8601, e.g. 2024-03-31)")
    parser.add_argument("--frequency", type=str, default=None, help="Grid temporal frequency (e.g. '1h', '15min')")
    parser.add_argument("--region", type=str, default=None, help="Specific region slug to isolate (defaults to all 25 regions)")
    parser.add_argument("--output-raw", type=str, default="data/raw/alerts_raw.parquet", help="Path to save raw alerts")
    parser.add_argument("--output-grid", type=str, default="data/processed/alerts_grid.parquet", help="Path to save continuous grid")
    args = parser.parse_args()

    config = load_config()
    target_regions = [get_region(args.region).slug] if args.region else ALL_REGION_SLUGS
    freq = args.frequency or config.grid.frequency
    is_mock = args.mock or (args.source == "mock")

    if is_mock:
        print(f"[MOCK MODE] Generating alert data for {len(target_regions)} region(s)...")
        source = MockAlertDataSource()
    else:
        print(f"[ALERTS.IN.UA MODE] Ingesting alert data for {len(target_regions)} region(s)...")
        source = AlertsInUaDataSource(dump_path=args.dump_path, api_token=args.api_token)

    raw_df = source.load_raw_alerts(
        start_date=args.start_date,
        end_date=args.end_date,
        regions=target_regions,
    )
    print(f"Loaded {len(raw_df)} raw alert intervals (source: {raw_df['source'].iloc[0] if not raw_df.empty else 'empty'}).")

    raw_path = Path(args.output_raw)
    source.save_raw_parquet(raw_df, raw_path)
    print(f"Saved raw alerts to {raw_path}")

    print(f"Converting intervals to continuous uniform grid (frequency: {freq})...")
    converter = IntervalToGridConverter(frequency=freq)
    grid_path = Path(args.output_grid)
    grid_df = converter.convert(
        raw_df=raw_df,
        start_date=args.start_date,
        end_date=args.end_date,
        regions=target_regions,
    )
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    grid_df.to_parquet(grid_path, index=False)
    print(f"Continuous grid built with {len(grid_df)} rows. Saved to {grid_path}")
    print(f"Alert active incidence rate: {grid_df['alert_active'].mean():.2%}")
    print(f"Alert onset incidence rate: {grid_df['alert_onset'].mean():.2%}")


if __name__ == "__main__":
    main()
