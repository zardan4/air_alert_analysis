"""Interactive SVG Map Component for Ukraine Air Raid Alerts.

Renders an interactive SVG map of Ukraine covering all 24 oblasts,
Kyiv City, and Crimea/Sevastopol with hover intelligence cards and one-click
navigation to individual oblast workspaces.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import streamlit.components.v1 as components

import json

# Declare Streamlit custom component for Ukraine SVG map
_COMPONENT_DIR = Path(__file__).resolve().parent / "assets" / "map_component"
_INDEX_HTML_PATH = _COMPONENT_DIR / "index.html"

_ukraine_map_custom_component = components.declare_component(
    "ukraine_svg_map",
    path=str(_COMPONENT_DIR),
)


def load_map_assets() -> tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    """Load SVG path strings and geographic centers for Ukrainian regions."""
    asset_dir = Path(__file__).resolve().parent / "assets"
    paths_file = asset_dir / "ukraine_svg_paths.json"
    centers_file = asset_dir / "ukraine_svg_centers.json"

    paths: Dict[str, str] = {}
    centers: Dict[str, Dict[str, Any]] = {}

    if paths_file.exists():
        with open(paths_file, "r", encoding="utf-8") as f:
            paths = json.load(f)

    if centers_file.exists():
        with open(centers_file, "r", encoding="utf-8") as f:
            centers = json.load(f)

    return paths, centers


def generate_interactive_ukraine_svg_html(
    region_rows: List[Dict[str, Any]],
    is_vadimkin_mode: bool = False,
    horizon_hours: int = 1,
) -> str:
    """Generate standalone HTML document string from index.html template."""
    if _INDEX_HTML_PATH.exists():
        with open(_INDEX_HTML_PATH, "r", encoding="utf-8") as f:
            html = f.read()
    else:
        html = "<!DOCTYPE html><html><body>Map component template missing</body></html>"
    return html


def render_ukraine_svg_map(
    region_rows: List[Dict[str, Any]],
    is_vadimkin_mode: bool = False,
    horizon_hours: int = 1,
    key: Optional[str] = "ukraine_svg_map_interactive",
) -> Optional[str]:
    """Render interactive Ukraine SVG map component.

    Args:
        region_rows: List of dicts with region metrics.
        is_vadimkin_mode: If True, omits live status indicators.
        horizon_hours: Target lookahead prediction horizon.
        key: Streamlit component key.

    Returns:
        The clicked region slug (str) if an oblast was clicked, or None.
    """
    return _ukraine_map_custom_component(
        region_data=region_rows,
        is_vadimkin_mode=is_vadimkin_mode,
        horizon_hours=horizon_hours,
        default=None,
        key=key,
    )
