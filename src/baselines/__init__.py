"""Baselines package."""

from src.baselines.naive_baselines import (
    BaselineModel,
    ClimatologyBaseline,
    PersistenceBaseline,
    RollingRateBaseline,
    SpatialInflowBaseline,
    get_all_baselines,
)

__all__ = [
    "BaselineModel",
    "ClimatologyBaseline",
    "PersistenceBaseline",
    "RollingRateBaseline",
    "SpatialInflowBaseline",
    "get_all_baselines",
]
