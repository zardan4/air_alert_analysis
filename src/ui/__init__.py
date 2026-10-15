"""UI helper package."""

from src.ui.stats import (
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
    get_threat_level_stats,
)
from src.ui.map_component import (
    generate_interactive_ukraine_svg_html,
    load_map_assets,
    render_ukraine_svg_map,
)

__all__ = [
    "render_ukraine_svg_map",
    "generate_interactive_ukraine_svg_html",
    "load_map_assets",
    "get_annual_evolution_stats",
    "get_calm_and_downtime_stats",
    "get_day_of_week_profile",
    "get_disruption_scores",
    "get_duration_cdf_percentiles",
    "get_duration_spectrum",
    "get_heatmap_matrix",
    "get_hromada_breakdown",
    "get_peak_and_safe_windows",
    "get_safe_window_recommendation",
    "get_threat_level_stats",
]
