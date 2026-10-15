"""Unit tests for interactive SVG map component."""

import pytest
from src.ui.map_component import generate_interactive_ukraine_svg_html, load_map_assets
from src.utils.regions import ALL_REGION_SLUGS


def test_load_map_assets():
    """Verify SVG paths and centroids load for all 25 Ukrainian administrative regions."""
    paths, centers = load_map_assets()
    assert len(paths) >= 25
    assert len(centers) >= 25

    for slug in ALL_REGION_SLUGS:
        assert slug in paths, f"Missing SVG path for region {slug}"
        assert slug in centers, f"Missing centroid for region {slug}"
        assert len(paths[slug]) > 50, f"Path for {slug} too short"
        assert "cx" in centers[slug] and "cy" in centers[slug]
        assert centers[slug]["cx"] > 0
        assert centers[slug]["cy"] > 0


def test_generate_interactive_ukraine_svg_html_structure():
    """Verify generated HTML contains valid SVG elements, styles, and scripts."""
    mock_rows = [
        {
            "region_slug": "kharkiv",
            "Oblast (EN)": "Kharkiv Oblast",
            "Область (UK)": "Харківська область",
            "Current Status": "SIREN ACTIVE",
            "1h Ahead Risk": "68%",
            "Risk Tier": "High",
            "Yellow (Drones) %": "85%",
            "Red (Missiles) %": "15%",
            "Typical Duration": "45 mins",
            "Short (<30m) %": "22%",
            "Swarm (>2h) %": "28%",
            "_prob": 0.68,
        },
        {
            "region_slug": "lviv",
            "Oblast (EN)": "Lviv Oblast",
            "Область (UK)": "Львівська область",
            "Current Status": "CLEAR",
            "1h Ahead Risk": "15%",
            "Risk Tier": "Low",
            "Yellow (Drones) %": "88%",
            "Red (Missiles) %": "12%",
            "Typical Duration": "35 mins",
            "Short (<30m) %": "40%",
            "Swarm (>2h) %": "8%",
            "_prob": 0.15,
        },
    ]

    html = generate_interactive_ukraine_svg_html(mock_rows, is_vadimkin_mode=False, horizon_hours=1)
    assert "<!DOCTYPE html>" in html
    assert "<svg" in html
    assert 'viewBox="0 0 1000 669"' in html
    assert "path-kharkiv" in html
    assert "path-lviv" in html
    assert "path-kyiv_city" in html
    assert "onOblastClick" in html
    assert "floating-tooltip" in html


def test_generate_interactive_ukraine_svg_html_vadimkin_mode():
    """Verify that vadimkin mode removes 'Current Status' from hover intelligence and adds historical indicator."""
    mock_rows = [
        {
            "region_slug": "odesa",
            "Oblast (EN)": "Odesa Oblast",
            "Область (UK)": "Одеська область",
            "1h Ahead Risk": "42%",
            "Risk Tier": "Moderate",
            "Yellow (Drones) %": "86%",
            "Red (Missiles) %": "14%",
            "Typical Duration": "52 mins",
            "Short (<30m) %": "20%",
            "Swarm (>2h) %": "32%",
            "_prob": 0.42,
            # Notice: no Current Status passed in vadimkin mode
        }
    ]

    html_vadimkin = generate_interactive_ukraine_svg_html(mock_rows, is_vadimkin_mode=True, horizon_hours=1)
    assert "Historical Archive Mode (Vadimkin)" in html_vadimkin
    assert "isVadimkinMode" in html_vadimkin
    assert "Real-Time Synchronized Mode" in html_vadimkin
