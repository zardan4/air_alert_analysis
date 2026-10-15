from typing import Dict, List, Optional, Union
import pandas as pd

from src.config import HromadaInfo, RegionInfo, load_regions

_REGIONS_CACHE: Optional[Dict[str, RegionInfo]] = None
_ID_TO_SLUG_CACHE: Optional[Dict[int, str]] = None


def _ensure_cache() -> None:
    global _REGIONS_CACHE, _ID_TO_SLUG_CACHE
    if _REGIONS_CACHE is None:
        _REGIONS_CACHE = load_regions()
        _ID_TO_SLUG_CACHE = {r.id: r.slug for r in _REGIONS_CACHE.values()}


def get_all_regions() -> Dict[str, RegionInfo]:
    """Return all 25 registered regions (24 oblasts + Kyiv city)."""
    _ensure_cache()
    assert _REGIONS_CACHE is not None
    return dict(_REGIONS_CACHE)


def get_region(identifier: Union[str, int]) -> RegionInfo:
    """Look up a region by its slug (e.g. 'kyiv_city', 'kharkiv') or numeric ID (1-25).
    
    Raises KeyError or ValueError if not found.
    """
    _ensure_cache()
    assert _REGIONS_CACHE is not None
    assert _ID_TO_SLUG_CACHE is not None

    if isinstance(identifier, int):
        slug = _ID_TO_SLUG_CACHE.get(identifier)
        if slug is None:
            raise ValueError(f"Region ID {identifier} not recognized (valid range: 1-{len(_REGIONS_CACHE)})")
        return _REGIONS_CACHE[slug]

    slug = str(identifier).strip().lower()
    if slug in _REGIONS_CACHE:
        return _REGIONS_CACHE[slug]

    # Also search by English or Ukrainian name
    for region in _REGIONS_CACHE.values():
        if (
            region.name_en.lower() == slug
            or region.name_uk.lower() == slug
            or region.slug == slug.replace(" ", "_").replace("-", "_")
        ):
            return region

    raise KeyError(f"Region '{identifier}' not found in registered Ukrainian regions")


def is_valid_region(identifier: Union[str, int]) -> bool:
    """Check if identifier matches a valid region."""
    try:
        get_region(identifier)
        return True
    except (KeyError, ValueError):
        return False


def get_all_region_slugs() -> List[str]:
    """Return sorted list of all 25 region slugs."""
    _ensure_cache()
    assert _REGIONS_CACHE is not None
    return sorted(_REGIONS_CACHE.keys())


def filter_by_region(df: pd.DataFrame, region_identifier: Union[str, int], region_col: str = "region_slug") -> pd.DataFrame:
    """Filter any dataframe (raw intervals or continuous grid) to a single specific oblast.
    
    This is used for per-oblast unit testing, isolated model evaluation, and diagnostics.
    """
    region = get_region(region_identifier)
    
    # Check if region is in the multi-index
    if isinstance(df.index, pd.MultiIndex) and region_col in df.index.names:
        idx_level = df.index.names.index(region_col)
        mask = df.index.get_level_values(idx_level) == region.slug
        return df[mask].copy()

    if region_col not in df.columns:
        raise KeyError(f"Column '{region_col}' not found in dataframe (columns: {list(df.columns)})")

    filtered = df[df[region_col] == region.slug].copy()
    return filtered


def get_region_hromadas(region_identifier: Union[str, int]) -> List[HromadaInfo]:
    """Return the list of representative territorial communities (hromadas) for the specified oblast."""
    region = get_region(region_identifier)
    return list(region.hromadas)


def get_hromada(region_identifier: Union[str, int], hromada_slug: str) -> Optional[HromadaInfo]:
    """Look up a specific hromada by slug within a given oblast."""
    hromadas = get_region_hromadas(region_identifier)
    slug = str(hromada_slug).strip().lower()
    for h in hromadas:
        if h.slug == slug:
            return h
    return None


def is_valid_hromada(region_identifier: Union[str, int], hromada_slug: str) -> bool:
    """Check if hromada exists in the specified oblast."""
    return get_hromada(region_identifier, hromada_slug) is not None


_ensure_cache()
assert _REGIONS_CACHE is not None
ALL_REGION_SLUGS: List[str] = sorted(_REGIONS_CACHE.keys())
TOTAL_REGIONS_COUNT: int = len(ALL_REGION_SLUGS)  # 25
