# Ukrainian Air Raid Alert Forecasting (`ukraine-alert-forecast`)

A modular, anti-leakage time-series forecasting framework to model and predict Ukrainian air raid alerts across all 24 oblasts and Kyiv City.

## Key Design Principles
- **Configurable Spatio-Temporal Discretization**: Standard uniform time grid (defaulting to 1h, configurable down to 15m) across all 25 administrative regions.
- **Strict Anti-Data Leakage**:
  - Point-in-time feature extraction ($t \le \text{cutoff}$).
  - Walk-forward expanding-window cross-validation with label embargoing.
  - Dedicated recent chronological test set.
- **Decoupled Data Ingestion**: Clean interface abstraction (`AlertDataSource`) allowing instant switching between realistic synthetic/mock generators and live `alerts.in.ua` data archives.
- **Isolated Per-Oblast Unit Testing**: Full programmatic test suite verifying each individual oblast and the unified 25-region system.

## Project Structure
```
├── config/
│   ├── config.yaml          # System, backtesting, and temporal grid settings
│   └── regions.yaml         # Complete 25-region administrative registry
├── data/
│   ├── raw/                 # Raw alert interval records
│   └── processed/           # Standardized uniform continuous grids
├── src/
│   ├── __init__.py
│   ├── config.py            # Typed configuration and region managers
│   ├── data/
│   │   ├── base.py          # AlertDataSource abstract interface & validator
│   │   ├── mock_source.py   # High-fidelity synthetic generator
│   │   ├── alerts_in_ua.py  # alerts.in.ua dump/API ingestor
│   │   └── grid_converter.py# Continuous spatio-temporal grid transformer
│   └── utils/
│       └── regions.py       # Region lookup and isolation helpers
├── tests/
│   ├── test_regions.py      # Individual oblast isolation & registry tests
│   ├── test_data_sources.py # Data source interface & mock generator tests
│   └── test_grid_converter.py # Grid interval discretization tests
└── pyproject.toml
```
