"""
Tests for plot_results.py.

load() and _grouped_bars() are tested with synthetic DataFrames.
The six plot_* functions are smoke-tested (run without error) by mocking
_save so no PNG files are written during the test suite.
"""

import io
import sys
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from benchmarks.plot_results import (
    _grouped_bars,
    load,
    plot_approx_ratio,
    plot_bitstring,
    plot_combined,
    plot_portfolio_return,
    plot_portfolio_variance,
    plot_runtime,
    plot_success_prob,
)

# ---------------------------------------------------------------------------
# Shared synthetic data
# ---------------------------------------------------------------------------

_CSV = """\
n,tickers,k,lam,p,solver,bitstring,selected_stocks,portfolio_return,portfolio_variance,objective,approx_ratio,success_prob,runtime
4,"AAPL,MSFT,GOOGL,AMZN",2,1.0,,brute_force,1010,AAPL|GOOGL,0.50,0.10,-0.40,1.000,,0.001
4,"AAPL,MSFT,GOOGL,AMZN",2,1.0,,greedy,1010,AAPL|GOOGL,0.45,0.12,-0.33,0.825,,0.0002
4,"AAPL,MSFT,GOOGL,AMZN",2,1.0,,simulated_annealing,1010,AAPL|GOOGL,0.45,0.12,-0.33,0.825,,0.005
4,"AAPL,MSFT,GOOGL,AMZN",2,1.0,1,qaoa_aer,1010,AAPL|GOOGL,0.40,0.14,-0.26,0.650,0.12,2.50
4,"AAPL,MSFT,GOOGL,AMZN",2,1.0,2,qaoa_aer,1010,AAPL|GOOGL,0.42,0.13,-0.29,0.725,0.18,3.80
4,"AAPL,MSFT,GOOGL,AMZN",2,1.0,3,qaoa_aer,1010,AAPL|GOOGL,0.44,0.11,-0.33,0.825,0.25,5.10
"""


def _make_df() -> pd.DataFrame:
    return load(io.StringIO(_CSV))  # load accepts any path-like / file-like


# Override load to accept StringIO for testing
def _synthetic_df() -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(_CSV))
    df["p"] = pd.to_numeric(df["p"], errors="coerce")

    def _label(row):
        if row["solver"] == "qaoa_aer":
            return f"QAOA p={int(row['p'])}"
        return {
            "brute_force":         "Brute Force",
            "greedy":              "Greedy",
            "simulated_annealing": "Sim. Annealing",
        }.get(row["solver"], row["solver"])

    df["solver_label"] = df.apply(_label, axis=1)
    return df


# ---------------------------------------------------------------------------
# Tests for load()
# ---------------------------------------------------------------------------

class TestLoad:
    def test_returns_dataframe(self, tmp_path):
        csv_file = tmp_path / "results.csv"
        csv_file.write_text(_CSV)
        df = load(csv_file)
        assert isinstance(df, pd.DataFrame)

    def test_adds_solver_label_column(self, tmp_path):
        csv_file = tmp_path / "results.csv"
        csv_file.write_text(_CSV)
        df = load(csv_file)
        assert "solver_label" in df.columns

    def test_brute_force_label(self, tmp_path):
        csv_file = tmp_path / "results.csv"
        csv_file.write_text(_CSV)
        df = load(csv_file)
        bf = df[df["solver"] == "brute_force"]
        assert (bf["solver_label"] == "Brute Force").all()

    def test_greedy_label(self, tmp_path):
        csv_file = tmp_path / "results.csv"
        csv_file.write_text(_CSV)
        df = load(csv_file)
        gr = df[df["solver"] == "greedy"]
        assert (gr["solver_label"] == "Greedy").all()

    def test_sim_annealing_label(self, tmp_path):
        csv_file = tmp_path / "results.csv"
        csv_file.write_text(_CSV)
        df = load(csv_file)
        sa = df[df["solver"] == "simulated_annealing"]
        assert (sa["solver_label"] == "Sim. Annealing").all()

    def test_qaoa_label_includes_depth(self, tmp_path):
        csv_file = tmp_path / "results.csv"
        csv_file.write_text(_CSV)
        df = load(csv_file)
        qaoa_p1 = df[(df["solver"] == "qaoa_aer") & (df["p"] == 1)]
        assert (qaoa_p1["solver_label"] == "QAOA p=1").all()

    def test_p_column_is_numeric(self, tmp_path):
        csv_file = tmp_path / "results.csv"
        csv_file.write_text(_CSV)
        df = load(csv_file)
        assert pd.api.types.is_float_dtype(df["p"]) or pd.api.types.is_integer_dtype(df["p"])

    def test_row_count_preserved(self, tmp_path):
        csv_file = tmp_path / "results.csv"
        csv_file.write_text(_CSV)
        df = load(csv_file)
        assert len(df) == 6


# ---------------------------------------------------------------------------
# Tests for _grouped_bars()
# ---------------------------------------------------------------------------

class TestGroupedBars:
    def test_runs_without_error(self):
        df = _synthetic_df()
        fig, ax = plt.subplots()
        _grouped_bars(ax, df, "approx_ratio", [4])
        plt.close(fig)

    def test_log_scale_option(self):
        df = _synthetic_df()
        fig, ax = plt.subplots()
        _grouped_bars(ax, df, "runtime", [4], log_scale=True)
        assert ax.get_yscale() == "log"
        plt.close(fig)

    def test_linear_scale_by_default(self):
        df = _synthetic_df()
        fig, ax = plt.subplots()
        _grouped_bars(ax, df, "approx_ratio", [4])
        assert ax.get_yscale() == "linear"
        plt.close(fig)

    def test_x_tick_labels_set(self):
        df = _synthetic_df()
        fig, ax = plt.subplots()
        _grouped_bars(ax, df, "portfolio_return", [4])
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert len(labels) > 0
        plt.close(fig)


# ---------------------------------------------------------------------------
# Smoke tests for plot_*() — _save is mocked to skip file writes
# ---------------------------------------------------------------------------

@pytest.fixture
def df():
    return _synthetic_df()


@patch("benchmarks.plot_results._save")
def test_plot_approx_ratio_runs(mock_save, df):
    plot_approx_ratio(df)
    assert mock_save.called


@patch("benchmarks.plot_results._save")
def test_plot_portfolio_return_runs(mock_save, df):
    plot_portfolio_return(df)
    assert mock_save.called


@patch("benchmarks.plot_results._save")
def test_plot_portfolio_variance_runs(mock_save, df):
    plot_portfolio_variance(df)
    assert mock_save.called


@patch("benchmarks.plot_results._save")
def test_plot_runtime_runs(mock_save, df):
    plot_runtime(df)
    assert mock_save.called


@patch("benchmarks.plot_results._save")
def test_plot_success_prob_runs(mock_save, df):
    plot_success_prob(df)
    assert mock_save.called


@patch("benchmarks.plot_results._save")
def test_plot_bitstring_runs(mock_save, df):
    plot_bitstring(df)
    assert mock_save.called


@patch("benchmarks.plot_results._save")
def test_plot_combined_runs(mock_save, df):
    plot_combined(df)
    assert mock_save.called


@patch("benchmarks.plot_results._save")
def test_plot_success_prob_skips_gracefully_with_no_qaoa(mock_save):
    df = _synthetic_df()
    df_no_qaoa = df[df["solver"] != "qaoa_aer"].copy()
    plot_success_prob(df_no_qaoa)  # should not raise
