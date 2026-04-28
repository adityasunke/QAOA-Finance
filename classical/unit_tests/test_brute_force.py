import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from classical.brute_force import PortfolioResult, brute_force, objective

# n=2: asset 1 dominates (high return, same variance)
MU_2 = np.array([0.0, 1.0])
SIGMA_2 = np.eye(2) * 0.001

# n=3: diagonal, increasing returns
MU_3 = np.array([0.05, 0.10, 0.15])
SIGMA_3 = np.eye(3) * 0.01


class TestPortfolioResult:
    def test_instantiation(self):
        r = PortfolioResult(
            bitstring=np.array([1.0, 0.0]),
            objective=-0.5,
            ret=0.1,
            variance=0.04,
        )
        assert r is not None

    def test_field_values(self):
        bits = np.array([0.0, 1.0])
        r = PortfolioResult(bitstring=bits, objective=-1.0, ret=2.0, variance=0.5)
        np.testing.assert_array_equal(r.bitstring, bits)
        assert r.objective == -1.0
        assert r.ret == 2.0
        assert r.variance == 0.5


class TestObjective:
    def test_zero_vector_gives_zero(self):
        x = np.zeros(2)
        assert objective(x, MU_2, SIGMA_2, lam=1.0) == 0.0

    def test_returns_float(self):
        result = objective(np.array([1.0, 0.0]), MU_2, SIGMA_2, lam=1.0)
        assert isinstance(result, float)

    def test_known_value_asset0_only(self):
        # x=[1,0]: obj = 0.001 - 0.0 = 0.001
        x = np.array([1.0, 0.0])
        assert abs(objective(x, MU_2, SIGMA_2, lam=1.0) - 0.001) < 1e-12

    def test_known_value_asset1_only(self):
        # x=[0,1]: obj = 0.001 - 1.0 = -0.999
        x = np.array([0.0, 1.0])
        assert abs(objective(x, MU_2, SIGMA_2, lam=1.0) - (-0.999)) < 1e-12

    def test_known_value_both_assets(self):
        # x=[1,1]: obj = 0.002 - 1.0 = -0.998
        x = np.array([1.0, 1.0])
        assert abs(objective(x, MU_2, SIGMA_2, lam=1.0) - (-0.998)) < 1e-12

    def test_lambda_scales_return_term(self):
        x = np.array([0.0, 1.0])
        obj_l1 = objective(x, MU_2, SIGMA_2, lam=1.0)
        obj_l2 = objective(x, MU_2, SIGMA_2, lam=2.0)
        # Doubling lam subtracts an extra mu^T x = 1.0 from the objective
        assert abs(obj_l2 - (obj_l1 - 1.0)) < 1e-12

    def test_zero_sigma_gives_negative_return(self):
        mu = np.array([0.1, 0.2])
        Sigma = np.zeros((2, 2))
        x = np.array([1.0, 1.0])
        assert abs(objective(x, mu, Sigma, lam=1.0) - (-0.3)) < 1e-12

    def test_matches_manual_formula(self):
        x = np.array([1.0, 0.0, 1.0])
        obj = objective(x, MU_3, SIGMA_3, lam=1.0)
        expected = float(x @ SIGMA_3 @ x - 1.0 * MU_3 @ x)
        assert abs(obj - expected) < 1e-12


class TestBruteForce:
    def test_returns_portfolio_result(self):
        assert isinstance(brute_force(MU_2, SIGMA_2), PortfolioResult)

    def test_bitstring_length_matches_n(self):
        assert len(brute_force(MU_2, SIGMA_2).bitstring) == 2
        assert len(brute_force(MU_3, SIGMA_3).bitstring) == 3

    def test_bitstring_is_binary(self):
        for b in brute_force(MU_2, SIGMA_2).bitstring:
            assert b in (0.0, 1.0)

    def test_known_optimal_unconstrained(self):
        # x=[0,1] has lowest obj (-0.999) among all 4 portfolios
        result = brute_force(MU_2, SIGMA_2, lam=1.0)
        np.testing.assert_array_equal(result.bitstring, np.array([0.0, 1.0]))
        assert abs(result.objective - (-0.999)) < 1e-9

    def test_known_optimal_k1(self):
        result = brute_force(MU_2, SIGMA_2, lam=1.0, k=1)
        assert int(result.bitstring.sum()) == 1
        np.testing.assert_array_equal(result.bitstring, np.array([0.0, 1.0]))

    def test_known_optimal_k2(self):
        # Only feasible k=2 portfolio is [1,1] with obj=-0.998
        result = brute_force(MU_2, SIGMA_2, lam=1.0, k=2)
        assert int(result.bitstring.sum()) == 2
        np.testing.assert_array_equal(result.bitstring, np.array([1.0, 1.0]))

    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_cardinality_satisfied(self, k):
        result = brute_force(MU_3, SIGMA_3, lam=1.0, k=k)
        assert int(result.bitstring.sum()) == k

    def test_result_is_minimum_over_feasible_set(self):
        k = 1
        result = brute_force(MU_3, SIGMA_3, lam=1.0, k=k)
        for i in range(3):
            x = np.zeros(3)
            x[i] = 1.0
            assert result.objective <= objective(x, MU_3, SIGMA_3, 1.0) + 1e-12

    def test_ret_consistent_with_bitstring(self):
        result = brute_force(MU_2, SIGMA_2, lam=1.0, k=1)
        assert abs(result.ret - float(MU_2 @ result.bitstring)) < 1e-12

    def test_variance_consistent_with_bitstring(self):
        result = brute_force(MU_2, SIGMA_2, lam=1.0, k=1)
        assert abs(result.variance - float(result.bitstring @ SIGMA_2 @ result.bitstring)) < 1e-12

    def test_variance_is_nonnegative(self):
        assert brute_force(MU_3, SIGMA_3, lam=1.0, k=2).variance >= 0.0

    def test_infeasible_k_raises_value_error(self):
        with pytest.raises(ValueError):
            brute_force(MU_2, SIGMA_2, lam=1.0, k=5)

    def test_objective_field_matches_formula(self):
        result = brute_force(MU_3, SIGMA_3, lam=1.0, k=2)
        expected = objective(result.bitstring, MU_3, SIGMA_3, 1.0)
        assert abs(result.objective - expected) < 1e-12

    def test_lam_zero_minimizes_variance_only(self):
        # With lam=0, objective = xᵀΣx ≥ 0; minimum feasible is all-zeros (or k=0)
        result = brute_force(MU_3, SIGMA_3, lam=0.0)
        assert result.objective >= 0.0
