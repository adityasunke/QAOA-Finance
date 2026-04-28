import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from classical.brute_force import PortfolioResult
from classical.run_classical import result_to_dict

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN"]

# Minimal PortfolioResult with 4 assets, selecting assets 0 and 2
_BITS = np.array([1.0, 0.0, 1.0, 0.0])
_RESULT = PortfolioResult(
    bitstring=_BITS,
    objective=-0.123456789,
    ret=0.25,
    variance=0.05,
)


class TestResultToDict:
    def test_returns_dict(self):
        d = result_to_dict(_RESULT, TICKERS, "brute_force")
        assert isinstance(d, dict)

    def test_solver_field(self):
        d = result_to_dict(_RESULT, TICKERS, "greedy")
        assert d["solver"] == "greedy"

    def test_selected_assets_are_correct(self):
        d = result_to_dict(_RESULT, TICKERS, "brute_force")
        assert d["selected_assets"] == ["AAPL", "GOOGL"]

    def test_selected_assets_empty_when_none_chosen(self):
        result_zero = PortfolioResult(
            bitstring=np.zeros(4),
            objective=0.0,
            ret=0.0,
            variance=0.0,
        )
        d = result_to_dict(result_zero, TICKERS, "brute_force")
        assert d["selected_assets"] == []

    def test_bitstring_is_int_list(self):
        d = result_to_dict(_RESULT, TICKERS, "brute_force")
        assert d["bitstring"] == [1, 0, 1, 0]
        assert all(isinstance(b, int) for b in d["bitstring"])

    def test_objective_is_rounded(self):
        d = result_to_dict(_RESULT, TICKERS, "brute_force")
        assert d["objective"] == round(-0.123456789, 6)

    def test_return_is_rounded(self):
        d = result_to_dict(_RESULT, TICKERS, "brute_force")
        assert d["annualised_return"] == round(0.25, 6)

    def test_variance_is_rounded(self):
        d = result_to_dict(_RESULT, TICKERS, "brute_force")
        assert d["annualised_variance"] == round(0.05, 6)

    def test_all_expected_keys_present(self):
        d = result_to_dict(_RESULT, TICKERS, "simulated_annealing")
        expected_keys = {
            "solver", "selected_assets", "bitstring",
            "objective", "annualised_return", "annualised_variance",
        }
        assert expected_keys.issubset(d.keys())

    def test_all_assets_selected(self):
        result_all = PortfolioResult(
            bitstring=np.ones(4),
            objective=-1.0,
            ret=1.0,
            variance=0.2,
        )
        d = result_to_dict(result_all, TICKERS, "greedy")
        assert d["selected_assets"] == TICKERS
