"""Spatial adjacency graph and geographical neighborhood definitions for Ukraine."""

from typing import Dict, List, Set
from src.utils.regions import ALL_REGION_SLUGS

# Explicit bilateral geographical neighbors for all 24 oblasts + Kyiv city
REGION_NEIGHBORS: Dict[str, List[str]] = {
    "kyiv_city": ["kyiv_oblast"],
    "kyiv_oblast": ["kyiv_city", "chernihiv", "poltava", "cherkasy", "vinnytsia", "zhytomyr"],
    "chernihiv": ["kyiv_oblast", "sumy", "poltava"],
    "sumy": ["chernihiv", "poltava", "kharkiv"],
    "kharkiv": ["sumy", "poltava", "dnipropetrovsk", "donetsk", "luhansk"],
    "luhansk": ["kharkiv", "donetsk"],
    "donetsk": ["luhansk", "kharkiv", "dnipropetrovsk", "zaporizhzhia"],
    "zaporizhzhia": ["donetsk", "dnipropetrovsk", "kherson"],
    "dnipropetrovsk": ["poltava", "kharkiv", "donetsk", "zaporizhzhia", "kherson", "mykolaiv", "kirovohrad"],
    "poltava": ["chernihiv", "sumy", "kharkiv", "dnipropetrovsk", "kirovohrad", "cherkasy", "kyiv_oblast"],
    "cherkasy": ["kyiv_oblast", "poltava", "kirovohrad", "vinnytsia"],
    "kirovohrad": ["cherkasy", "poltava", "dnipropetrovsk", "mykolaiv", "odesa", "vinnytsia"],
    "mykolaiv": ["kirovohrad", "dnipropetrovsk", "kherson", "odesa"],
    "kherson": ["mykolaiv", "dnipropetrovsk", "zaporizhzhia"],
    "odesa": ["mykolaiv", "kirovohrad", "vinnytsia"],
    "vinnytsia": ["zhytomyr", "kyiv_oblast", "cherkasy", "kirovohrad", "odesa", "chernivtsi", "khmelnytskyi"],
    "zhytomyr": ["kyiv_oblast", "vinnytsia", "khmelnytskyi", "rivne"],
    "khmelnytskyi": ["zhytomyr", "vinnytsia", "chernivtsi", "ternopil", "rivne"],
    "rivne": ["volyn", "zhytomyr", "khmelnytskyi", "ternopil", "lviv"],
    "volyn": ["rivne", "lviv"],
    "lviv": ["volyn", "rivne", "ternopil", "ivano_frankivsk", "zakarpattia"],
    "ternopil": ["rivne", "khmelnytskyi", "chernivtsi", "ivano_frankivsk", "lviv"],
    "ivano_frankivsk": ["lviv", "ternopil", "chernivtsi", "zakarpattia"],
    "chernivtsi": ["vinnytsia", "khmelnytskyi", "ternopil", "ivano_frankivsk"],
    "zakarpattia": ["lviv", "ivano_frankivsk"],
}

# Key primary border/coastal entry and frontline corridors
PRIMARY_INFLOW_REGIONS: Set[str] = {
    "kharkiv", "sumy", "donetsk", "luhansk", "zaporizhzhia", "kherson", "mykolaiv", "odesa"
}


def get_neighbors(region_slug: str) -> List[str]:
    """Get immediate geographical neighbors for an oblast."""
    return REGION_NEIGHBORS.get(region_slug, [])


def is_primary_inflow_region(region_slug: str) -> bool:
    """Check if region is a primary border or frontline entry corridor."""
    return region_slug in PRIMARY_INFLOW_REGIONS
