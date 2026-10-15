"""Unit tests verifying CLI execution in both mock and alerts.in.ua modes."""

import subprocess
import sys
from pathlib import Path
import pandas as pd
import pytest


def test_cli_generate_data_alerts_in_ua_default(tmp_path):
    """Verify 'python main.py generate-data' uses alerts.in.ua by default."""
    raw_out = tmp_path / "raw_real.parquet"
    grid_out = tmp_path / "grid_real.parquet"

    cmd = [
        sys.executable,
        "main.py",
        "generate-data",
        "--raw-output", str(raw_out),
        "--grid-output", str(grid_out),
        "--start-date", "2024-01-01",
        "--end-date", "2024-01-03",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "[ALERTS.IN.UA MODE]" in res.stdout
    assert "source: alerts_in_ua" in res.stdout
    assert raw_out.exists()
    assert grid_out.exists()

    df = pd.read_parquet(raw_out)
    assert (df["source"] == "alerts_in_ua").all()


def test_cli_generate_data_mock_flag(tmp_path):
    """Verify 'python main.py generate-data --mock' invokes synthetic generator."""
    raw_out = tmp_path / "raw_mock.parquet"
    grid_out = tmp_path / "grid_mock.parquet"

    cmd = [
        sys.executable,
        "main.py",
        "generate-data",
        "--mock",
        "--raw-output", str(raw_out),
        "--grid-output", str(grid_out),
        "--start-date", "2024-01-01",
        "--end-date", "2024-01-03",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "[MOCK MODE]" in res.stdout
    assert "source: mock_generator" in res.stdout
    assert raw_out.exists()

    df = pd.read_parquet(raw_out)
    assert "mock" in df.iloc[0]["source"]


def test_cli_global_mock_flag(tmp_path):
    """Verify 'python main.py --mock generate-data' honors root mock argument."""
    raw_out = tmp_path / "raw_global_mock.parquet"
    grid_out = tmp_path / "grid_global_mock.parquet"

    cmd = [
        sys.executable,
        "main.py",
        "--mock",
        "generate-data",
        "--raw-output", str(raw_out),
        "--grid-output", str(grid_out),
        "--start-date", "2024-01-01",
        "--end-date", "2024-01-03",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "[MOCK MODE]" in res.stdout
    assert raw_out.exists()


def test_cli_generate_data_source_argument(tmp_path):
    """Verify '--source mock' and '--source alerts_in_ua' arguments."""
    # Test --source mock
    raw_mock = tmp_path / "src_mock.parquet"
    cmd_mock = [
        sys.executable,
        "main.py",
        "generate-data",
        "--source", "mock",
        "--raw-output", str(raw_mock),
        "--grid-output", str(tmp_path / "grid1.parquet"),
        "--start-date", "2024-01-01",
        "--end-date", "2024-01-02",
    ]
    res_mock = subprocess.run(cmd_mock, capture_output=True, text=True, check=True)
    assert "[MOCK MODE]" in res_mock.stdout

    # Test --source alerts_in_ua
    raw_real = tmp_path / "src_real.parquet"
    cmd_real = [
        sys.executable,
        "main.py",
        "generate-data",
        "--source", "alerts_in_ua",
        "--raw-output", str(raw_real),
        "--grid-output", str(tmp_path / "grid2.parquet"),
        "--start-date", "2024-01-01",
        "--end-date", "2024-01-02",
    ]
    res_real = subprocess.run(cmd_real, capture_output=True, text=True, check=True)
    assert "[ALERTS.IN.UA MODE]" in res_real.stdout


def test_cli_status_display():
    """Verify 'python main.py status' reports data source configuration."""
    cmd = [sys.executable, "main.py", "status"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "alerts.in.ua Dump:" in res.stdout
    assert "alerts.in.ua API:" in res.stdout
    assert "Registered Regions:  25" in res.stdout


def test_scripts_generate_data_mock_and_default(tmp_path):
    """Verify scripts/generate_data.py supports both mock mode and alerts_in_ua."""
    raw_real = tmp_path / "script_real.parquet"
    grid_real = tmp_path / "script_grid_real.parquet"
    cmd_real = [
        sys.executable,
        "scripts/generate_data.py",
        "--output-raw", str(raw_real),
        "--output-grid", str(grid_real),
        "--start-date", "2024-01-01",
        "--end-date", "2024-01-02",
    ]
    res_real = subprocess.run(cmd_real, capture_output=True, text=True, check=True)
    assert "[ALERTS.IN.UA MODE]" in res_real.stdout
    assert raw_real.exists()

    raw_mock = tmp_path / "script_mock.parquet"
    grid_mock = tmp_path / "script_grid_mock.parquet"
    cmd_mock = [
        sys.executable,
        "scripts/generate_data.py",
        "--mock",
        "--output-raw", str(raw_mock),
        "--output-grid", str(grid_mock),
        "--start-date", "2024-01-01",
        "--end-date", "2024-01-02",
    ]
    res_mock = subprocess.run(cmd_mock, capture_output=True, text=True, check=True)
    assert "[MOCK MODE]" in res_mock.stdout
    assert raw_mock.exists()


def test_cli_status_with_mock_flag():
    """Verify 'python main.py status --mock' correctly reports mock active mode."""
    cmd = [sys.executable, "main.py", "status", "--mock"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "[MOCK MODE]" in res.stdout
    assert "Active Source:    mock" in res.stdout


def test_cli_subcommands_mock_flag():
    """Verify all CLI subcommands accept --mock flag."""
    for subcmd in ["build-features", "check-splits", "run-benchmark", "ui"]:
        cmd = [sys.executable, "main.py", subcmd, "--help"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        assert "--mock" in res.stdout, f"{subcmd} missing --mock flag"


def test_cmd_ui_mock_argument_formatting():
    """Verify cmd_ui constructs valid streamlit command with --server.port before '--'."""
    from unittest.mock import patch
    import main as cli_main

    args = cli_main.argparse.Namespace(
        command="ui",
        cmd_mock=True,
        global_mock=False,
        port=8502,
    )
    with patch("subprocess.run") as mock_run:
        cli_main.cmd_ui(args)
        called_cmd = mock_run.call_args[0][0]
        # '--server.port' and '8502' must appear BEFORE '--'
        assert "--server.port" in called_cmd
        assert "8502" in called_cmd
        port_idx = called_cmd.index("--server.port")
        dash_idx = called_cmd.index("--")
        mock_idx = called_cmd.index("--mock")
        assert port_idx < dash_idx
        assert dash_idx < mock_idx

