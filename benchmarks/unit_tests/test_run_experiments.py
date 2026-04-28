import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from benchmarks.run_experiments import (
    CSV_COLUMNS,
    K_BY_N,
    LAM,
    P_VALUES,
    TICKERS_BY_N,
    _row,
)

MU_4 = np.array([0.10, 0.15, 0.08, 0.20])
SIGMA_4 = np.eye(4) * 0.05
TICKERS_4 = ["AAPL", "MSFT", "GOOGL", "AMZN"]


class TestConstants:
    def test_lam_is_one(self):
        assert LAM == 1.0

    def test_p_values_contains_1_2_3(self):
        assert set(P_VALUES) == {1, 2, 3}

    def test_k_by_n_keys(self):
        assert set(K_BY_N.keys()) == {4, 6}

    def test_k_by_n_values_are_half_n(self):
        for n, k in K_BY_N.items():
            assert k == n // 2

    def test_tickers_by_n_keys(self):
        assert set(TICKERS_BY_N.keys()) == {4, 6}

    def test_tickers_by_n_lengths(self):
        for n, tickers in TICKERS_BY_N.items():
            assert len(tickers) == n

    def test_csv_columns_contains_required_fields(self):
        required = {
            "n", "tickers", "k", "lam", "p", "solver",
            "bitstring", "objective", "approx_ratio", "runtime",
        }
        assert required.issubset(set(CSV_COLUMNS))

    def test_csv_columns_is_list(self):
        assert isinstance(CSV_COLUMNS, list)


class TestRowFunction:
    def _call(self, **kwargs):
        defaults = dict(
            n=4,
            tickers=TICKERS_4,
            k=2,
            p=1,
            solver="brute_force",
            x=np.array([1.0, 0.0, 1.0, 0.0]),
            obj=-0.25,
            approx=1.0,
            succ=None,
            runtime=0.002,
            mu=MU_4,
            Sigma=SIGMA_4,
        )
        defaults.update(kwargs)
        return _row(**defaults)

    def test_returns_dict(self):
        assert isinstance(self._call(), dict)

    def test_n_field(self):
        assert self._call(n=4)["n"] == 4

    def test_tickers_field_is_comma_joined(self):
        d = self._call(tickers=["AAPL", "MSFT", "GOOGL", "AMZN"])
        assert d["tickers"] == "AAPL,MSFT,GOOGL,AMZN"

    def test_k_field(self):
        assert self._call(k=2)["k"] == 2

    def test_lam_field_uses_module_constant(self):
        assert self._call()["lam"] == LAM

    def test_p_field_none_for_classical(self):
        assert self._call(p=None)["p"] is None

    def test_p_field_integer_for_qaoa(self):
        assert self._call(p=2)["p"] == 2

    def test_solver_field(self):
        assert self._call(solver="greedy")["solver"] == "greedy"

    def test_bitstring_field_is_formatted_string(self):
        x = np.array([1.0, 0.0, 1.0, 0.0])
        assert self._call(x=x)["bitstring"] == "1010"

    def test_selected_stocks_field(self):
        x = np.array([1.0, 0.0, 1.0, 0.0])
        d = self._call(x=x, tickers=TICKERS_4)
        assert d["selected_stocks"] == "AAPL|GOOGL"

    def test_portfolio_return_field(self):
        x = np.array([1.0, 0.0, 0.0, 0.0])
        d = self._call(x=x, mu=MU_4)
        assert abs(d["portfolio_return"] - MU_4[0]) < 1e-12

    def test_portfolio_variance_field(self):
        x = np.array([1.0, 0.0, 0.0, 0.0])
        d = self._call(x=x, Sigma=SIGMA_4)
        assert abs(d["portfolio_variance"] - 0.05) < 1e-12

    def test_objective_field(self):
        assert self._call(obj=-0.42)["objective"] == -0.42

    def test_approx_ratio_field(self):
        assert abs(self._call(approx=1.25)["approx_ratio"] - 1.25) < 1e-12

    def test_success_prob_none_for_classical(self):
        assert self._call(succ=None)["success_prob"] is None

    def test_success_prob_set_for_qaoa(self):
        assert abs(self._call(succ=0.37)["success_prob"] - 0.37) < 1e-12

    def test_runtime_field(self):
        assert abs(self._call(runtime=3.14)["runtime"] - 3.14) < 1e-12

    def test_all_csv_columns_present(self):
        d = self._call()
        for col in CSV_COLUMNS:
            assert col in d, f"Missing column: {col}"
