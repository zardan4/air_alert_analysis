"""Connector for alerts.in.ua historical dumps and live API integration."""

from datetime import datetime, timezone
import gzip
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import requests

from src.config import load_config
from src.data.base import ALL_RAW_COLUMNS, AlertDataSource
from src.data.mock_source import MockAlertDataSource
from src.utils.regions import get_all_regions, get_region, is_valid_region

logger = logging.getLogger(__name__)

# Official alerts.in.ua numeric location_uids for Ukrainian oblasts & Kyiv City
OFFICIAL_ALERTS_IN_UA_UIDS: Dict[int, str] = {
    3: "khmelnytskyi",
    4: "vinnytsia",
    5: "rivne",
    8: "volyn",
    9: "dnipropetrovsk",
    10: "zhytomyr",
    11: "zakarpattia",
    12: "zaporizhzhia",
    13: "ivano_frankivsk",
    14: "kyiv_oblast",
    15: "kirovohrad",
    16: "luhansk",
    17: "mykolaiv",
    18: "odesa",
    19: "poltava",
    20: "sumy",
    21: "ternopil",
    22: "kharkiv",
    23: "kherson",
    24: "cherkasy",
    25: "chernivtsi",
    26: "chernihiv",
    27: "lviv",
    28: "donetsk",
    31: "kyiv_city",
}

# Internal sequential region IDs (1 to 25)
INTERNAL_REGION_ID_MAP: Dict[int, str] = {
    1: "cherkasy",
    2: "chernihiv",
    3: "chernivtsi",
    4: "dnipropetrovsk",
    5: "donetsk",
    6: "ivano_frankivsk",
    7: "kharkiv",
    8: "kherson",
    9: "khmelnytskyi",
    10: "kirovohrad",
    11: "kyiv_city",
    12: "kyiv_oblast",
    13: "luhansk",
    14: "lviv",
    15: "mykolaiv",
    16: "odesa",
    17: "poltava",
    18: "rivne",
    19: "sumy",
    20: "ternopil",
    21: "vinnytsia",
    22: "volyn",
    23: "zakarpattia",
    24: "zaporizhzhia",
    25: "zhytomyr",
}

# Mapping from common alerts.in.ua Ukrainian names & shorthands to standardized slugs
ALERTS_IN_UA_NAME_MAPPING: Dict[str, str] = {
    # Cherkasy
    "Черкаська область": "cherkasy",
    "Черкаська обл.": "cherkasy",
    "Черкаська обл": "cherkasy",
    "Черкащина": "cherkasy",
    "Черкаси": "cherkasy",
    "м. Черкаси": "cherkasy",
    "м.Черкаси": "cherkasy",
    "Cherkasy Oblast": "cherkasy",
    "Cherkasy": "cherkasy",

    # Chernihiv
    "Чернігівська область": "chernihiv",
    "Чернігівська обл.": "chernihiv",
    "Чернігівська обл": "chernihiv",
    "Чернігівщина": "chernihiv",
    "Сіверщина": "chernihiv",
    "Чернігів": "chernihiv",
    "м. Чернігів": "chernihiv",
    "м.Чернігів": "chernihiv",
    "Chernihiv Oblast": "chernihiv",
    "Chernihiv": "chernihiv",

    # Chernivtsi
    "Чернівецька область": "chernivtsi",
    "Чернівецька обл.": "chernivtsi",
    "Чернівецька обл": "chernivtsi",
    "Чернівеччина": "chernivtsi",
    "Буковина": "chernivtsi",
    "Чернівці": "chernivtsi",
    "м. Чернівці": "chernivtsi",
    "м.Чернівці": "chernivtsi",
    "Chernivtsi Oblast": "chernivtsi",
    "Chernivtsi": "chernivtsi",

    # Dnipropetrovsk
    "Дніпропетровська область": "dnipropetrovsk",
    "Дніпропетровська обл.": "dnipropetrovsk",
    "Дніпропетровська обл": "dnipropetrovsk",
    "Дніпропетровщина": "dnipropetrovsk",
    "Дніпро": "dnipropetrovsk",
    "м. Дніпро": "dnipropetrovsk",
    "м.Дніпро": "dnipropetrovsk",
    "Dnipropetrovsk Oblast": "dnipropetrovsk",
    "Dnipro": "dnipropetrovsk",

    # Donetsk
    "Донецька область": "donetsk",
    "Донецька обл.": "donetsk",
    "Донецька обл": "donetsk",
    "Донеччина": "donetsk",
    "Донбас": "donetsk",
    "Донецьк": "donetsk",
    "м. Донецьк": "donetsk",
    "м.Донецьк": "donetsk",
    "Donetsk Oblast": "donetsk",
    "Donetsk": "donetsk",

    # Ivano-Frankivsk
    "Івано-Франківська область": "ivano_frankivsk",
    "Івано-Франківська обл.": "ivano_frankivsk",
    "Івано-Франківська обл": "ivano_frankivsk",
    "Івано-Франківщина": "ivano_frankivsk",
    "Прикарпаття": "ivano_frankivsk",
    "Івано-Франківськ": "ivano_frankivsk",
    "м. Івано-Франківськ": "ivano_frankivsk",
    "м.Івано-Франківськ": "ivano_frankivsk",
    "Ivano-Frankivsk Oblast": "ivano_frankivsk",
    "Ivano-Frankivsk": "ivano_frankivsk",

    # Kharkiv
    "Харківська область": "kharkiv",
    "Харківська обл.": "kharkiv",
    "Харківська обл": "kharkiv",
    "Харківщина": "kharkiv",
    "Слобожанщина": "kharkiv",
    "Харків": "kharkiv",
    "м. Харків": "kharkiv",
    "м.Харків": "kharkiv",
    "Kharkiv Oblast": "kharkiv",
    "Kharkiv": "kharkiv",

    # Kherson
    "Херсонська область": "kherson",
    "Херсонська обл.": "kherson",
    "Херсонська обл": "kherson",
    "Херсонщина": "kherson",
    "Херсон": "kherson",
    "м. Херсон": "kherson",
    "м.Херсон": "kherson",
    "Kherson Oblast": "kherson",
    "Kherson": "kherson",

    # Khmelnytskyi
    "Хмельницька область": "khmelnytskyi",
    "Хмельницька обл.": "khmelnytskyi",
    "Хмельницька обл": "khmelnytskyi",
    "Хмельниччина": "khmelnytskyi",
    "Поділля": "khmelnytskyi",
    "Хмельницький": "khmelnytskyi",
    "м. Хмельницький": "khmelnytskyi",
    "м.Хмельницький": "khmelnytskyi",
    "Khmelnytskyi Oblast": "khmelnytskyi",
    "Khmelnytskyi": "khmelnytskyi",

    # Kirovohrad
    "Кіровоградська область": "kirovohrad",
    "Кіровоградська обл.": "kirovohrad",
    "Кіровоградська обл": "kirovohrad",
    "Кіровоградщина": "kirovohrad",
    "Кропивницький": "kirovohrad",
    "м. Кропивницький": "kirovohrad",
    "м.Кропивницький": "kirovohrad",
    "Kirovohrad Oblast": "kirovohrad",
    "Kropyvnytskyi": "kirovohrad",

    # Kyiv City (Distinct from Kyiv Oblast)
    "м. Київ": "kyiv_city",
    "м.Київ": "kyiv_city",
    "Київ": "kyiv_city",
    "місто Київ": "kyiv_city",
    "Київ (місто)": "kyiv_city",
    "Київ місто": "kyiv_city",
    "Kyiv City": "kyiv_city",
    "City of Kyiv": "kyiv_city",

    # Kyiv Oblast
    "Київська область": "kyiv_oblast",
    "Київська обл.": "kyiv_oblast",
    "Київська обл": "kyiv_oblast",
    "Київщина": "kyiv_oblast",
    "Kyiv Oblast": "kyiv_oblast",
    "Kyiv Region": "kyiv_oblast",

    # Luhansk
    "Луганська область": "luhansk",
    "Луганська обл.": "luhansk",
    "Луганська обл": "luhansk",
    "Луганщина": "luhansk",
    "Луганськ": "luhansk",
    "м. Луганськ": "luhansk",
    "м.Луганськ": "luhansk",
    "Luhansk Oblast": "luhansk",
    "Luhansk": "luhansk",

    # Lviv
    "Львівська область": "lviv",
    "Львівська обл.": "lviv",
    "Львівська обл": "lviv",
    "Львівщина": "lviv",
    "Львів": "lviv",
    "м. Львів": "lviv",
    "м.Львів": "lviv",
    "Lviv Oblast": "lviv",
    "Lviv": "lviv",

    # Mykolaiv
    "Миколаївська область": "mykolaiv",
    "Миколаївська обл.": "mykolaiv",
    "Миколаївська обл": "mykolaiv",
    "Миколаївщина": "mykolaiv",
    "Миколаїв": "mykolaiv",
    "м. Миколаїв": "mykolaiv",
    "м.Миколаїв": "mykolaiv",
    "Mykolaiv Oblast": "mykolaiv",
    "Mykolaiv": "mykolaiv",

    # Odesa
    "Одеська область": "odesa",
    "Одеська обл.": "odesa",
    "Одеська обл": "odesa",
    "Одещина": "odesa",
    "Одеса": "odesa",
    "м. Одеса": "odesa",
    "м.Одеса": "odesa",
    "Odesa Oblast": "odesa",
    "Odesa": "odesa",

    # Poltava
    "Полтавська область": "poltava",
    "Полтавська обл.": "poltava",
    "Полтавська обл": "poltava",
    "Полтавщина": "poltava",
    "Полтава": "poltava",
    "м. Полтава": "poltava",
    "м.Полтава": "poltava",
    "Poltava Oblast": "poltava",
    "Poltava": "poltava",

    # Rivne
    "Рівненська область": "rivne",
    "Рівненська обл.": "rivne",
    "Рівненська обл": "rivne",
    "Рівненщина": "rivne",
    "Рівне": "rivne",
    "м. Рівне": "rivne",
    "м.Рівне": "rivne",
    "Rivne Oblast": "rivne",
    "Rivne": "rivne",

    # Sumy
    "Сумська область": "sumy",
    "Сумська обл.": "sumy",
    "Сумська обл": "sumy",
    "Сумщина": "sumy",
    "Суми": "sumy",
    "м. Суми": "sumy",
    "м.Суми": "sumy",
    "Sumy Oblast": "sumy",
    "Sumy": "sumy",

    # Ternopil
    "Тернопільська область": "ternopil",
    "Тернопільська обл.": "ternopil",
    "Тернопільська обл": "ternopil",
    "Тернопільщина": "ternopil",
    "Тернопіль": "ternopil",
    "м. Тернопіль": "ternopil",
    "м.Тернопіль": "ternopil",
    "Ternopil Oblast": "ternopil",
    "Ternopil": "ternopil",

    # Vinnytsia
    "Вінницька область": "vinnytsia",
    "Вінницька обл.": "vinnytsia",
    "Вінницька обл": "vinnytsia",
    "Вінниччина": "vinnytsia",
    "Вінниця": "vinnytsia",
    "м. Вінниця": "vinnytsia",
    "м.Вінниця": "vinnytsia",
    "Vinnytsia Oblast": "vinnytsia",
    "Vinnytsia": "vinnytsia",

    # Volyn
    "Волинська область": "volyn",
    "Волинська обл.": "volyn",
    "Волинська обл": "volyn",
    "Волинь": "volyn",
    "Луцьк": "volyn",
    "м. Луцьк": "volyn",
    "м.Луцьк": "volyn",
    "Volyn Oblast": "volyn",
    "Volyn": "volyn",

    # Zakarpattia
    "Закарпатська область": "zakarpattia",
    "Закарпатська обл.": "zakarpattia",
    "Закарпатська обл": "zakarpattia",
    "Закарпаття": "zakarpattia",
    "Ужгород": "zakarpattia",
    "м. Ужгород": "zakarpattia",
    "м.Ужгород": "zakarpattia",
    "Zakarpattia Oblast": "zakarpattia",
    "Zakarpattia": "zakarpattia",

    # Zaporizhzhia
    "Запорізька область": "zaporizhzhia",
    "Запорізька обл.": "zaporizhzhia",
    "Запорізька обл": "zaporizhzhia",
    "Запоріжжя": "zaporizhzhia",
    "м. Запоріжжя": "zaporizhzhia",
    "м.Запоріжжя": "zaporizhzhia",
    "Zaporizhzhia Oblast": "zaporizhzhia",
    "Zaporizhzhia": "zaporizhzhia",

    # Zhytomyr
    "Житомирська область": "zhytomyr",
    "Житомирська обл.": "zhytomyr",
    "Житомирська обл": "zhytomyr",
    "Житомирщина": "zhytomyr",
    "Житомир": "zhytomyr",
    "м. Житомир": "zhytomyr",
    "м.Житомир": "zhytomyr",
    "Zhytomyr Oblast": "zhytomyr",
    "Zhytomyr": "zhytomyr",
}

# Pre-compile case-folded lookup for fast O(1) matching
_NAME_MAPPING_LOWER: Dict[str, str] = {k.lower(): v for k, v in ALERTS_IN_UA_NAME_MAPPING.items()}


class AlertsInUaDataSource(AlertDataSource):
    """Data source for alerts.in.ua datasets (local CSV/JSON/Parquet archives or live API)."""

    def __init__(
        self,
        dump_path: Optional[Union[str, Path]] = None,
        api_token: Optional[str] = None,
        api_base_url: Optional[str] = None,
        token_env_var: Optional[str] = None,
        timeout_sec: Optional[int] = None,
        fallback_to_dump: Optional[bool] = None,
        fallback_to_mock: Optional[bool] = None,
    ):
        try:
            config = load_config()
            cfg_alerts = getattr(config, "alerts_in_ua", None)
            cfg_data = getattr(config, "data", None)
        except Exception:
            config = None
            cfg_alerts = None
            cfg_data = None

        # Determine dump path
        if dump_path is not None:
            self.dump_path = Path(dump_path)
        elif cfg_data and getattr(cfg_data, "alerts_in_ua_dump_path", None):
            candidate = Path(cfg_data.alerts_in_ua_dump_path)
            self.dump_path = candidate if candidate.exists() else None
        else:
            default_candidate = Path("data/raw/alerts_in_ua_dump.csv")
            self.dump_path = default_candidate if default_candidate.exists() else None

        # Token and API endpoints
        self.token_env_var = (
            token_env_var
            or (getattr(cfg_alerts, "token_env_var", None) if cfg_alerts else None)
            or "ALERTS_IN_UA_TOKEN"
        )
        self.api_token = (
            api_token
            or os.environ.get(self.token_env_var, "")
            or (getattr(cfg_alerts, "api_token", "") if cfg_alerts else "")
        )
        self.api_base_url = (
            api_base_url
            or (getattr(cfg_alerts, "api_base_url", None) if cfg_alerts else None)
            or "https://api.alerts.in.ua/v1"
        )
        self.timeout_sec = (
            timeout_sec
            if timeout_sec is not None
            else (getattr(cfg_alerts, "timeout_sec", 10) if cfg_alerts else 10)
        )
        self.fallback_to_dump = (
            fallback_to_dump
            if fallback_to_dump is not None
            else (getattr(cfg_alerts, "fallback_to_dump", True) if cfg_alerts else True)
        )
        self.fallback_to_mock = (
            fallback_to_mock
            if fallback_to_mock is not None
            else (getattr(cfg_alerts, "fallback_to_mock", True) if cfg_alerts else True)
        )

    def _resolve_region_slug(self, raw_name_or_id: Union[str, int, float]) -> Optional[str]:
        """Map raw alerts.in.ua title or ID to standardized region slug.

        String/name matching takes precedence to avoid misclassifying alerts with numeric UIDs.
        """
        if raw_name_or_id is None or (isinstance(raw_name_or_id, float) and pd.isna(raw_name_or_id)):
            return None

        # Handle pure numeric types or string digits
        is_num = isinstance(raw_name_or_id, (int, float))
        name_str = str(raw_name_or_id).strip()
        if not name_str:
            return None

        if is_num or name_str.isdigit():
            try:
                num_id = int(raw_name_or_id)
                return self._resolve_location_uid(num_id)
            except (ValueError, OverflowError):
                return None

        # 1. Exact string match against dictionary (case-insensitive)
        name_lower = name_str.lower()
        if name_lower in _NAME_MAPPING_LOWER:
            return _NAME_MAPPING_LOWER[name_lower]

        # 2. Check if already a valid registered slug or name in regions registry
        if is_valid_region(name_str):
            return get_region(name_str).slug

        for reg in get_all_regions().values():
            if (
                reg.name_uk.lower() == name_lower
                or reg.name_en.lower() == name_lower
                or reg.slug.lower() == name_lower
            ):
                return reg.slug

        # 3. Normalize common abbreviations (e.g. "обл." -> "область")
        norm_name = re.sub(r"\bобл\.?", "область", name_lower).strip()
        norm_name = re.sub(r"\s+", " ", norm_name)
        if norm_name in _NAME_MAPPING_LOWER:
            return _NAME_MAPPING_LOWER[norm_name]

        # 4. Check known hromadas from registry
        hromada_match = self._resolve_hromada_to_oblast(name_str)
        if hromada_match:
            return hromada_match[0]

        # 5. Targeted prefix check (avoiding false partials like "київ" in "київська область")
        for k_lower, slug in _NAME_MAPPING_LOWER.items():
            if len(k_lower) >= 5 and k_lower in name_lower:
                return slug

        return None

    def _resolve_location_uid(self, num_id: int) -> Optional[str]:
        """Map numeric location UID (checks internal 1-25 ID, then official alerts.in.ua UID)."""
        # 1. Check internal sequential ID (1 to 25)
        if num_id in INTERNAL_REGION_ID_MAP:
            return INTERNAL_REGION_ID_MAP[num_id]

        # 2. Check official alerts.in.ua UID (e.g. 26, 27, 28, 31)
        if num_id in OFFICIAL_ALERTS_IN_UA_UIDS:
            return OFFICIAL_ALERTS_IN_UA_UIDS[num_id]

        # 3. Check region alerts_in_ua_id in registry
        for reg in get_all_regions().values():
            if reg.alerts_in_ua_id == num_id:
                return reg.slug

        return None

    def _resolve_official_location_uid(self, num_id: int) -> Optional[str]:
        """Map official alerts.in.ua numeric UID first."""
        if num_id in OFFICIAL_ALERTS_IN_UA_UIDS:
            return OFFICIAL_ALERTS_IN_UA_UIDS[num_id]
        return self._resolve_location_uid(num_id)

    def _resolve_hromada_to_oblast(self, title: str) -> Optional[tuple[str, str]]:
        """Attempt to match a raion or community title to (oblast_slug, hromada_slug)."""
        title_lower = title.strip().lower()
        for reg in get_all_regions().values():
            for h in reg.hromadas:
                if (
                    h.slug.lower() == title_lower
                    or h.name_uk.lower() in title_lower
                    or h.name_en.lower() in title_lower
                ):
                    return reg.slug, h.slug
        return None

    def load_from_local_dump(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Parse raw CSV, JSON, JSONL, or Parquet export from alerts.in.ua into standardized format."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Alerts dump file not found: {file_path}")

        name = file_path.name.lower()

        if name.endswith(".parquet"):
            raw_df = pd.read_parquet(file_path)

        elif name.endswith(".csv.gz") or name.endswith(".csv"):
            compression = "gzip" if name.endswith(".gz") else None
            raw_df = pd.read_csv(file_path, compression=compression)

        elif name.endswith(".jsonl.gz") or name.endswith(".jsonl"):
            # Robust line-by-line JSONL parser supporting blank and malformed lines
            records = []
            open_fn = gzip.open if name.endswith(".gz") else open
            with open_fn(file_path, "rt", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, start=1):
                    line_str = line.strip()
                    if not line_str or line_str.startswith("#"):
                        continue
                    try:
                        record = json.loads(line_str)
                        if isinstance(record, dict):
                            records.append(record)
                    except json.JSONDecodeError as err:
                        logger.warning(f"Skipping malformed JSON line {line_idx} in {file_path}: {err}")
            raw_df = pd.DataFrame(records)

        elif name.endswith(".json.gz") or name.endswith(".json"):
            open_fn = gzip.open if name.endswith(".gz") else open
            with open_fn(file_path, "rt", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                alerts = data.get("alerts", [])
                raw_df = pd.DataFrame(alerts)
            elif isinstance(data, list):
                raw_df = pd.DataFrame(data)
            else:
                raise ValueError(f"Unexpected JSON structure in {file_path}")

        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        return self._normalize_alerts_in_ua_dataframe(raw_df)

    def fetch_active_alerts(self, api_token: Optional[str] = None) -> pd.DataFrame:
        """Fetch active air raid alerts directly from alerts.in.ua API.

        Endpoint: GET https://api.alerts.in.ua/v1/alerts/active.json
        Authentication: Bearer token via api_token parameter or ALERTS_IN_UA_TOKEN env var.
        """
        token = api_token or self.api_token or os.environ.get(self.token_env_var, "")
        if not token:
            raise ValueError(
                f"alerts.in.ua API token is missing. Please provide api_token, configure alerts_in_ua.api_token, "
                f"or set {self.token_env_var} environment variable."
            )

        endpoint = f"{self.api_base_url.rstrip('/')}/alerts/active.json"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": "UkraineAlertForecasting/0.1.0",
        }

        response = requests.get(endpoint, headers=headers, timeout=self.timeout_sec)
        if response.status_code == 200:
            data = response.json()
            alerts_list = data.get("alerts", [])
            if not alerts_list:
                return pd.DataFrame(columns=ALL_RAW_COLUMNS)

            raw_df = pd.DataFrame(alerts_list)
            # Active alerts in alerts.in.ua have finished_at = null
            now_iso = datetime.now(timezone.utc).isoformat()
            if "finished_at" in raw_df.columns:
                raw_df["finished_at"] = raw_df["finished_at"].fillna(now_iso)
            elif "ended_at" in raw_df.columns:
                raw_df["ended_at"] = raw_df["ended_at"].fillna(now_iso)
            else:
                raw_df["finished_at"] = now_iso

            return self._normalize_alerts_in_ua_dataframe(raw_df)

        elif response.status_code == 304:
            return pd.DataFrame(columns=ALL_RAW_COLUMNS)
        elif response.status_code == 401:
            raise PermissionError("alerts.in.ua API authentication failed (401 Unauthorized): invalid or expired token.")
        elif response.status_code == 403:
            raise PermissionError("alerts.in.ua API access forbidden (403 Forbidden): IP blocked or region restricted.")
        elif response.status_code == 429:
            raise RuntimeError("alerts.in.ua API rate limit exceeded (429 Too Many Requests).")
        else:
            response.raise_for_status()
            return pd.DataFrame(columns=ALL_RAW_COLUMNS)

    def _normalize_alerts_in_ua_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Map heterogeneous raw columns from alerts.in.ua to standardized schema."""
        if raw_df.empty:
            return pd.DataFrame(columns=ALL_RAW_COLUMNS)

        # Comprehensive column alias dictionary
        col_map = {
            "id": "alert_id",
            "alert_id": "alert_id",
            "start": "started_at",
            "started_at": "started_at",
            "alert_start": "started_at",
            "end": "finished_at",
            "ended_at": "finished_at",
            "finished_at": "finished_at",
            "alert_end": "finished_at",
            "type": "alert_type",
            "alert_type": "alert_type",
            "location_title": "raw_region",
            "region": "raw_region",
            "region_name": "raw_region",
            "region_title": "raw_region",
            "location": "raw_region",
            "oblast": "raw_region",
            "location_title_en": "raw_region_en",
            "region_en": "raw_region_en",
            "location_oblast": "raw_oblast",
            "oblast_title": "raw_oblast",
            "location_raion": "raw_raion",
            "raion": "raw_raion",
            "location_settlement": "raw_settlement",
            "settlement": "raw_settlement",
            "location_uid": "raw_location_uid",
            "region_id": "raw_region_id",
            "location_oblast_uid": "raw_oblast_id",
            "threat_level": "threat_level",
            "level": "threat_level",
            "color": "threat_level",
            "hromada": "hromada",
            "community": "hromada",
        }

        # Rename matching columns
        df = raw_df.rename(columns={c: col_map[c] for c in raw_df.columns if c in col_map}).copy()

        # Fallback alert_id and alert_type
        if "alert_id" not in df.columns:
            df["alert_id"] = [f"alert_{i}" for i in range(len(df))]
        if "alert_type" not in df.columns:
            df["alert_type"] = "air_raid"

        has_raw_region = "raw_region" in df.columns
        has_raw_region_en = "raw_region_en" in df.columns
        has_raw_oblast = "raw_oblast" in df.columns
        has_raw_settlement = "raw_settlement" in df.columns
        has_raw_location_uid = "raw_location_uid" in df.columns
        has_raw_region_id = "raw_region_id" in df.columns
        has_raw_oblast_id = "raw_oblast_id" in df.columns
        has_hromada = "hromada" in df.columns

        resolved_slugs = []
        resolved_ids = []
        resolved_hromadas = []

        for idx, row in df.iterrows():
            slug = None
            hromada_val = row["hromada"] if has_hromada and pd.notna(row["hromada"]) else None

            # 1. Match by Ukrainian region name / title (Primary authoritative source)
            if has_raw_region and pd.notna(row["raw_region"]):
                slug = self._resolve_region_slug(row["raw_region"])

            # 2. Match by English region title
            if slug is None and has_raw_region_en and pd.notna(row["raw_region_en"]):
                slug = self._resolve_region_slug(row["raw_region_en"])

            # 3. Match by parent oblast title (raion or territorial community alert)
            if slug is None and has_raw_oblast and pd.notna(row["raw_oblast"]):
                slug = self._resolve_region_slug(row["raw_oblast"])
                if slug and not hromada_val and has_raw_region and pd.notna(row["raw_region"]):
                    hromada_val = str(row["raw_region"]).strip()

            # 4. Check known hromadas or settlement name
            if slug is None and has_raw_settlement and pd.notna(row["raw_settlement"]):
                h_match = self._resolve_hromada_to_oblast(str(row["raw_settlement"]))
                if h_match:
                    slug, hromada_val = h_match

            # 5. Match by location_uid (official alerts.in.ua UID)
            if slug is None and has_raw_location_uid and pd.notna(row["raw_location_uid"]):
                val = row["raw_location_uid"]
                if isinstance(val, (int, float)) or (isinstance(val, str) and val.strip().isdigit()):
                    try:
                        slug = self._resolve_official_location_uid(int(val))
                    except (ValueError, OverflowError):
                        slug = self._resolve_region_slug(val)
                else:
                    slug = self._resolve_region_slug(val)

            # 6. Match by internal region_id
            if slug is None and has_raw_region_id and pd.notna(row["raw_region_id"]):
                slug = self._resolve_region_slug(row["raw_region_id"])

            # 7. Match by parent oblast UID
            if slug is None and has_raw_oblast_id and pd.notna(row["raw_oblast_id"]):
                val = row["raw_oblast_id"]
                if isinstance(val, (int, float)) or (isinstance(val, str) and val.strip().isdigit()):
                    try:
                        slug = self._resolve_official_location_uid(int(val))
                    except (ValueError, OverflowError):
                        slug = self._resolve_region_slug(val)
                else:
                    slug = self._resolve_region_slug(val)

            if slug and is_valid_region(slug):
                resolved_slugs.append(slug)
                resolved_ids.append(get_region(slug).id)
                resolved_hromadas.append(hromada_val if hromada_val else "oblast_wide")
            else:
                resolved_slugs.append(None)
                resolved_ids.append(None)
                resolved_hromadas.append(None)

        df["region_slug"] = resolved_slugs
        df["region_id"] = resolved_ids
        df["hromada"] = resolved_hromadas
        df["source"] = "alerts_in_ua"

        # Active alert handling: fill missing finished_at with current UTC timestamp
        now_utc = pd.Timestamp.now(tz="UTC")
        if "finished_at" in df.columns:
            df["finished_at"] = df["finished_at"].fillna(now_utc)
        else:
            df["finished_at"] = now_utc

        # Filter out records that could not be mapped to the 25 oblasts
        df = df.dropna(subset=["region_slug", "started_at", "finished_at"]).copy()

        return self.validate_raw_schema(df, strict=False)

    def load_raw_alerts(
        self,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        regions: Optional[List[str]] = None,
        use_api: bool = False,
    ) -> pd.DataFrame:
        """Load alerts from local dump or live endpoint, with robust fallback mechanisms."""
        df: Optional[pd.DataFrame] = None

        # 1. API ingestion mode
        if use_api or (self.api_token and not (self.dump_path and self.dump_path.exists())):
            try:
                df = self.fetch_active_alerts()
            except Exception as e:
                logger.warning(f"alerts.in.ua live API fetch failed: {e}")
                if self.fallback_to_dump and self.dump_path and self.dump_path.exists():
                    logger.info(f"Falling back to local dump: {self.dump_path}")
                    df = self.load_from_local_dump(self.dump_path)
                elif self.fallback_to_mock:
                    logger.warning("Falling back to MockAlertDataSource due to API failure.")
                    mock_source = MockAlertDataSource()
                    return mock_source.load_raw_alerts(
                        start_date=start_date,
                        end_date=end_date,
                        regions=regions,
                    )
                else:
                    raise

        # 2. Local dump mode
        if df is None:
            if self.dump_path and self.dump_path.exists():
                df = self.load_from_local_dump(self.dump_path)
            else:
                default_dump = Path("data/raw/alerts_in_ua_dump.csv")
                if default_dump.exists():
                    df = self.load_from_local_dump(default_dump)
                elif self.api_token:
                    try:
                        df = self.fetch_active_alerts()
                    except Exception as e:
                        if self.fallback_to_mock:
                            logger.warning(f"API fetch failed ({e}). Falling back to MockAlertDataSource.")
                            mock_source = MockAlertDataSource()
                            return mock_source.load_raw_alerts(
                                start_date=start_date,
                                end_date=end_date,
                                regions=regions,
                            )
                        raise
                elif self.fallback_to_mock:
                    logger.warning(
                        "No alerts.in.ua dump file found and no API token configured. "
                        "Falling back to MockAlertDataSource (mock mode)."
                    )
                    mock_source = MockAlertDataSource()
                    return mock_source.load_raw_alerts(
                        start_date=start_date,
                        end_date=end_date,
                        regions=regions,
                    )
                else:
                    raise FileNotFoundError(
                        f"Alerts dump file not found at '{self.dump_path}' and no API credentials configured. "
                        f"Provide a valid dump_path, configure ALERTS_IN_UA_TOKEN, or run with '--mock' for synthetic data."
                    )

        # Apply filtering
        if start_date:
            start_ts = pd.to_datetime(start_date, utc=True)
            df = df[df["started_at"] >= start_ts]
        if end_date:
            end_ts = pd.to_datetime(end_date, utc=True)
            df = df[df["started_at"] <= end_ts]
        if regions:
            df = df[df["region_slug"].isin(regions)]

        return df.reset_index(drop=True)
