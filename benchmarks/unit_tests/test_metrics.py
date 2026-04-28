import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from benchmarks.metrics import (
    approximation_ratio,
    format_bitstring,
    portfolio_return,
    portfolio_variance,
    selected_stocks,
    success_probability,
)

TICKERS_4 = ["AAPL", "MSFT", "GOOGL", "AMZN"]


class TestPortfolioReturn:
    def test_zero_bitstring_gives_zero(self):
        mu = np.array([0.1, 0.2, 0.3])
        x = np.zeros(3)
        assert portfolio_return(x, mu) == 0.0

    def test_returns_float(self):
        result = portfolio_return(np.array([1.0, 0.0]), np.array([0.1, 0.2]))
        assert isinstance(result, float)

    def test_single_asset_selected(self):
        mu = np.array([0.1, 0.5, 0.3])
        x = np.array([0.0, 1.0, 0.0])
        assert abs(portfolio_return(x, mu) - 0.5) < 1e-12

    def test_all_assets_selected(self):
        mu = np.array([0.1, 0.2, 0.3])
        x = np.ones(3)
        assert abs(portfolio_return(x, mu) - 0.6) < 1e-12

    def test_matches_dot_product(self):
        mu = np.array([0.05, 0.10, 0.08, 0.15])
        x = np.array([1.0, 0.0, 1.0, 0.0])
        assert abs(portfolio_return(x, mu) - float(mu @ x)) < 1e-12


class TestPortfolioVariance:
    def test_zero_bitstring_gives_zero(self):
        Sigma = np.eye(3) * 0.1
        x = np.zeros(3)
        assert portfolio_variance(x, Sigma) == 0.0

    def test_returns_float(self):
        result = portfolio_variance(np.array([1.0, 0.0]), np.eye(2) * 0.05)
        assert isinstance(result, float)

    def test_diagonal_sigma_single_asset(self):
        Sigma = np.diag([0.04, 0.09, 0.16])
        x = np.array([0.0, 1.0, 0.0])
        assert abs(portfolio_variance(x, Sigma) - 0.09) < 1e-12

    def test_nonnegative_for_psd_sigma(self):
        Sigma = np.eye(4) * 0.05
        x = np.array([1.0, 0.0, 1.0, 1.0])
        assert portfolio_variance(x, Sigma) >= 0.0

    def test_matches_quadratic_form(self):
        Sigma = np.array([[0.04, 0.01], [0.01, 0.09]])
        x = np.array([1.0, 1.0])
        assert abs(portfolio_variance(x, Sigma) - float(x @ Sigma @ x)) < 1e-12


class TestApproximationRatio:
    def test_same_values_gives_one(self):
        assert abs(approximation_ratio(-0.5, -0.5) - 1.0) < 1e-12

    def test_returns_float(self):
        result = approximation_ratio(-0.4, -0.5)
        assert isinstance(result, float)

    def test_ratio_formula(self):
        assert abs(approximation_ratio(-0.4, -0.5) - 0.8) < 1e-12

    def test_both_near_zero_gives_one(self):
        assert abs(approximation_ratio(1e-15, 1e-15) - 1.0) < 1e-10

    def test_optimal_near_zero_solver_nonzero_gives_inf(self):
        result = approximation_ratio(0.5, 0.0)
        assert result == float("inf")

    def test_ratio_is_obj_divided_by_optimal(self):
        # approximation_ratio is a plain division: obj / optimal_obj
        assert abs(approximation_ratio(-0.3, -0.5) - 0.6) < 1e-12

    def test_optimal_solver_ratio_is_one(self):
        assert approximation_ratio(-1.23, -1.23) == 1.0


class TestSuccessProbability:
    def test_empty_counts_gives_zero(self):
        assert success_probability({}, np.array([1.0, 0.0])) == 0.0

    def test_returns_float(self):
        counts = {"01": 1000}
        result = success_probability(counts, np.array([1.0, 0.0]))
        assert isinstance(result, float)

    def test_all_shots_optimal_gives_one(self):
        # optimal [1, 0] → big-endian Qiskit key is "01"
        counts = {"01": 1000}
        assert abs(success_probability(counts, np.array([1.0, 0.0])) - 1.0) < 1e-12

    def test_no_shots_optimal_gives_zero(self):
        counts = {"11": 500, "00": 500}
        assert success_probability(counts, np.array([1.0, 0.0])) == 0.0

    def test_half_shots_optimal(self):
        counts = {"01": 500, "10": 500}
        prob = success_probability(counts, np.array([1.0, 0.0]))
        assert abs(prob - 0.5) < 1e-12

    def test_little_endian_to_big_endian_conversion(self):
        # optimal = [1, 0, 1] (little-endian) → reversed = "101" (big-endian)
        counts = {"101": 800, "010": 200}
        prob = success_probability(counts, np.array([1.0, 0.0, 1.0]))
        assert abs(prob - 0.8) < 1e-12

    def test_zero_total_shots_gives_zero(self):
        counts = {"01": 0, "10": 0}
        assert success_probability(counts, np.array([1.0, 0.0])) == 0.0


class TestFormatBitstring:
    def test_all_zeros(self):
        assert format_bitstring(np.zeros(4)) == "0000"

    def test_all_ones(self):
        assert format_bitstring(np.ones(4)) == "1111"

    def test_mixed(self):
        assert format_bitstring(np.array([1.0, 0.0, 1.0, 0.0])) == "1010"

    def test_single_asset_on(self):
        assert format_bitstring(np.array([0.0, 1.0, 0.0])) == "010"

    def test_returns_string(self):
        assert isinstance(format_bitstring(np.array([1.0, 0.0])), str)

    def test_length_matches_n(self):
        for n in [2, 3, 4, 6]:
            bits = np.zeros(n)
            assert len(format_bitstring(bits)) == n


class TestSelectedStocks:
    def test_none_selected_returns_none_label(self):
        x = np.zeros(4)
        assert selected_stocks(x, TICKERS_4) == "(none)"

    def test_returns_string(self):
        x = np.array([1.0, 0.0, 0.0, 0.0])
        assert isinstance(selected_stocks(x, TICKERS_4), str)

    def test_single_asset_selected(self):
        x = np.array([1.0, 0.0, 0.0, 0.0])
        assert selected_stocks(x, TICKERS_4) == "AAPL"

    def test_multiple_assets_pipe_separated(self):
        x = np.array([1.0, 0.0, 1.0, 0.0])
        assert selected_stocks(x, TICKERS_4) == "AAPL|GOOGL"

    def test_all_assets_selected(self):
        x = np.ones(4)
        assert selected_stocks(x, TICKERS_4) == "AAPL|MSFT|GOOGL|AMZN"

    def test_order_follows_ticker_order(self):
        x = np.array([0.0, 1.0, 0.0, 1.0])
        result = selected_stocks(x, TICKERS_4)
        assert result == "MSFT|AMZN"
