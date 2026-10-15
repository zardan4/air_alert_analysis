"""Realistic synthetic alert data source for local development, testing, and benchmarking."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.config import load_config
from src.data.base import ALL_RAW_COLUMNS, AlertDataSource
from src.utils.regions import get_all_region_slugs, get_region, get_region_hromadas

# Base regional risk multipliers based on observed war-time air raid intensity
# (Frontline/border regions experience higher frequency, western regions lower)
REGIONAL_RISK_WEIGHTS: Dict[str, float] = {
    # Border & Frontline (High activity)
    "kharkiv": 2.2,
    "donetsk": 2.4,
    "zaporizhzhia": 2.1,
    "kherson": 2.0,
    "sumy": 1.9,
    "dnipropetrovsk": 1.7,
    "mykolaiv": 1.5,
    "luhansk": 2.5,
    # Central / Capital (Targeted by missile & drone barrages)
    "kyiv_city": 1.3,
    "kyiv_oblast": 1.4,
    "poltava": 1.3,
    "odesa": 1.5,
    "cherkasy": 1.1,
    "kirovohrad": 1.2,
    "chernihiv": 1.3,
    "vinnytsia": 1.0,
    "zhytomyr": 1.0,
    # Western (Lower frequency, usually missile fly-overs or western infrastructure strikes)
    "khmelnytskyi": 0.8,
    "rivne": 0.7,
    "volyn": 0.6,
    "lviv": 0.7,
    "ternopil": 0.6,
    "ivano_frankivsk": 0.5,
    "chernivtsi": 0.4,
    "zakarpattia": 0.3,
}

# Cascading flight paths: upstream region -> potential downstream alerts
CASCADING_PROPAGATION: Dict[str, List[str]] = {
    "sumy": ["poltava", "chernihiv"],
    "kharkiv": ["poltava", "dnipropetrovsk"],
    "dnipropetrovsk": ["poltava", "kirovohrad", "cherkasy"],
    "poltava": ["cherkasy", "kyiv_oblast", "kyiv_city"],
    "kherson": ["mykolaiv", "dnipropetrovsk", "zaporizhzhia"],
    "mykolaiv": ["odesa", "kirovohrad", "vinnytsia"],
    "kyiv_oblast": ["kyiv_city", "zhytomyr", "cherkasy"],
    "zhytomyr": ["rivne", "khmelnytskyi", "vinnytsia"],
    "vinnytsia": ["khmelnytskyi", "chernivtsi"],
}


class MockAlertDataSource(AlertDataSource):
    """Generates synthetic, physically plausible air raid alert histories."""

    def __init__(
        self,
        seed: Optional[int] = None,
        base_daily_rate: Optional[float] = None,
        night_multiplier: Optional[float] = None,
    ):
        config = load_config()
        self.seed = seed if seed is not None else config.mock_generator.seed
        self.base_daily_rate = (
            base_daily_rate if base_daily_rate is not None else config.mock_generator.base_daily_rate
        )
        self.night_multiplier = (
            night_multiplier if night_multiplier is not None else config.mock_generator.night_multiplier
        )

    def load_raw_alerts(
        self,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        regions: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Generate synthetic raw alert records for the specified period and regions."""
        config = load_config()
        start = (
            pd.to_datetime(start_date, utc=True)
            if start_date
            else pd.to_datetime(config.mock_generator.start_date, utc=True)
        )
        end = (
            pd.to_datetime(end_date, utc=True)
            if end_date
            else pd.to_datetime(config.mock_generator.end_date, utc=True)
        )

        if start >= end:
            raise ValueError(f"start_date ({start}) must precede end_date ({end})")

        target_regions = regions if regions is not None else get_all_region_slugs()
        for r in target_regions:
            get_region(r)  # validates slug

        rng = np.random.default_rng(self.seed)
        total_days = max(1, (end - start).days)

        records: List[Dict] = []
        alert_counter = 1

        # Simulate day by day
        current_day_start = start.floor("D")

        while current_day_start < end:
            day_end = min(current_day_start + timedelta(days=1), end)

            # 1. Generate independent regional alerts
            for region_slug in target_regions:
                region_info = get_region(region_slug)
                weight = REGIONAL_RISK_WEIGHTS.get(region_slug, 1.0)
                expected_alerts = (self.base_daily_rate * weight) / 24.0

                # Sample per hour of day
                current_hour = current_day_start
                while current_hour < day_end:
                    hour_of_day = current_hour.hour
                    # Night multiplier in local Kyiv time (approx UTC+2 or UTC+3, say 20:00 to 04:00 UTC)
                    is_night = hour_of_day >= 20 or hour_of_day <= 4
                    hourly_lambda = expected_alerts * (self.night_multiplier if is_night else 0.7)
                    
                    num_alerts = rng.poisson(hourly_lambda)
                    for _ in range(num_alerts):
                        minute_offset = int(rng.integers(0, 59))
                        second_offset = int(rng.integers(0, 59))
                        alert_start = current_hour + timedelta(minutes=minute_offset, seconds=second_offset)

                        if alert_start >= end:
                            continue

                        # Alert duration distribution:
                        # 65% short sorties / ballistic threats (20 - 45 mins)
                        # 35% drone swarms / extended strikes (1.5 - 5.5 hours)
                        is_drone_swarm = is_night and (rng.random() < 0.7) or (rng.random() < 0.25)
                        if is_drone_swarm:
                            duration_mins = rng.normal(loc=180, scale=45)
                            duration_mins = max(45.0, min(duration_mins, 420.0))
                            alert_type = "air_raid_swarm"
                        else:
                            duration_mins = rng.normal(loc=32, scale=8)
                            duration_mins = max(15.0, min(duration_mins, 70.0))
                            alert_type = "air_raid_tactical"

                        alert_finish = alert_start + timedelta(minutes=float(duration_mins))

                        # Threat level generation:
                        # The Ukrainian government enacted yellow/red alert differentiation in September 2026.
                        # Prior to 2026-09-01 (including 2022-2025 and Jan-Aug 2026), alerts had no red/yellow
                        # differentiation (unspecified / general sirens).
                        # From September 2026 onwards:
                        # - Yellow (дронова небезпека / ударні БПЛА / heightened caution) represents the vast majority (~85%).
                        # - Red (ракетна та балістична небезпека) is much less frequent (~15%).
                        if alert_start < pd.Timestamp("2026-09-01", tz="UTC") or rng.random() < 0.02:
                            threat_level = "unspecified"
                        elif is_drone_swarm:
                            # Drone swarms correspond directly to yellow alert (дронова небезпека) under government rules
                            threat_level = "yellow" if rng.random() < 0.90 else "red"
                        else:
                            # Much fewer red alerts than yellow alerts (~15% red, ~85% yellow)
                            threat_level = "red" if rng.random() < 0.15 else "yellow"

                        # Hromada differentiation:
                        # Prior to decentralization (pre-2023), alerts were strictly oblast-wide.
                        # In the modern period, a portion of alerts are localized to specific territorial communities.
                        region_hromadas = get_region_hromadas(region_slug)
                        if alert_start < pd.Timestamp("2023-01-01", tz="UTC") or not region_hromadas or rng.random() < 0.40:
                            hromada_slug = "oblast_wide"
                        else:
                            chosen_hromada = region_hromadas[int(rng.integers(0, len(region_hromadas)))]
                            hromada_slug = chosen_hromada.slug

                        records.append({
                            "alert_id": f"mock_{alert_counter}",
                            "region_slug": region_slug,
                            "region_id": region_info.id,
                            "started_at": alert_start,
                            "finished_at": alert_finish,
                            "alert_type": alert_type,
                            "source": "mock_generator",
                            "threat_level": threat_level,
                            "hromada": hromada_slug,
                        })
                        alert_counter += 1

                        # Cascading downstream alert
                        downstream_candidates = CASCADING_PROPAGATION.get(region_slug, [])
                        for downstream in downstream_candidates:
                            if downstream in target_regions and rng.random() < 0.45:
                                lag_mins = rng.integers(20, 60)
                                cascaded_start = alert_start + timedelta(minutes=int(lag_mins))
                                if cascaded_start < end:
                                    cascaded_duration = max(20.0, duration_mins - float(rng.integers(10, 30)))
                                    downstream_info = get_region(downstream)

                                    downstream_hromadas = get_region_hromadas(downstream)
                                    if cascaded_start < pd.Timestamp("2023-01-01", tz="UTC") or not downstream_hromadas or rng.random() < 0.45:
                                        downstream_hromada = "oblast_wide"
                                    else:
                                        downstream_hromada = downstream_hromadas[int(rng.integers(0, len(downstream_hromadas)))].slug

                                    if threat_level == "unspecified":
                                        downstream_threat = "unspecified"
                                    else:
                                        downstream_threat = "yellow" if rng.random() < 0.85 else "red"

                                    records.append({
                                        "alert_id": f"mock_{alert_counter}",
                                        "region_slug": downstream,
                                        "region_id": downstream_info.id,
                                        "started_at": cascaded_start,
                                        "finished_at": cascaded_start + timedelta(minutes=cascaded_duration),
                                        "alert_type": alert_type,
                                        "source": "mock_generator_cascaded",
                                        "threat_level": downstream_threat,
                                        "hromada": downstream_hromada,
                                    })
                                    alert_counter += 1

                    current_hour += timedelta(hours=1)

            current_day_start += timedelta(days=1)

        if not records:
            return pd.DataFrame(columns=ALL_RAW_COLUMNS)

        df = pd.DataFrame(records)
        return self.validate_raw_schema(df, strict=True)
