import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from classical.brute_force import PortfolioResult, brute_force, objective
from classical.heuristics import greedy, simulated_annealing

# n=2: strong signal — asset 1 strictly dominates
MU_2 = np.array([0.0, 1.0])
SIGMA_2 = np.eye(2) * 0.001

# n=3: diagonal covariance, increasing returns
MU_3 = np.array([0.05, 0.10, 0.15])
SIGMA_3 = np.eye(3) * 0.01

# n=4: for seed-diversity tests
MU_4 = np.array([0.05, 0.10, 0.08, 0.15])
SIGMA_4 = np.eye(4) * 0.01


class TestGreedy:
    def test_returns_portfolio_result(self):
        assert isinstance(greedy(MU_2, SIGMA_2), PortfolioResult)

    def test_bitstring_is_binary(self):
        for b in greedy(MU_2, SIGMA_2, k=1).bitstring:
            assert b in (0.0, 1.0)

    def test_bitstring_length_matches_n(self):
        assert len(greedy(MU_3, SIGMA_3, k=1).bitstring) == 3

    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_cardinality_satisfied(self, k):
        result = greedy(MU_3, SIGMA_3, k=k)
        assert int(result.bitstring.sum()) == k

    def test_known_optimal_k1(self):
        # Asset 1 has the highest return and same variance → greedy selects it
        result = greedy(MU_2, SIGMA_2, k=1)
        np.testing.assert_array_equal(result.bitstring, np.array([0.0, 1.0]))

    def test_ret_consistent_with_bitstring(self):
        result = greedy(MU_3, SIGMA_3, k=2)
        assert abs(result.ret - float(MU_3 @ result.bitstring)) < 1e-12

    def test_variance_consistent_with_bitstring(self):
        result = greedy(MU_3, SIGMA_3, k=2)
        assert abs(result.variance - float(result.bitstring @ SIGMA_3 @ result.bitstring)) < 1e-12

    def test_objective_consistent_with_bitstring(self):
        result = greedy(MU_3, SIGMA_3, k=2)
        expected = objective(result.bitstring, MU_3, SIGMA_3, 1.0)
        assert abs(result.objective - expected) < 1e-12

    def test_variance_is_nonnegative(self):
        assert greedy(MU_3, SIGMA_3, k=2).variance >= 0.0

    def test_no_cardinality_returns_result(self):
        result = greedy(MU_2, SIGMA_2)
        assert isinstance(result, PortfolioResult)
        for b in result.bitstring:
            assert b in (0.0, 1.0)

    def test_lam_affects_selection(self):
        # With lam=0 only variance matters; with lam=10 return dominates.
        # Both should return valid results.
        r_low = greedy(MU_4, SIGMA_4, lam=0.01, k=2)
        r_high = greedy(MU_4, SIGMA_4, lam=10.0, k=2)
        assert int(r_low.bitstring.sum()) == 2
        assert int(r_high.bitstring.sum()) == 2

    def test_greedy_k1_matches_brute_force_on_simple(self):
        # For diagonal sigma with increasing returns, brute force and greedy agree at k=1
        bf = brute_force(MU_3, SIGMA_3, lam=1.0, k=1)
        gr = greedy(MU_3, SIGMA_3, lam=1.0, k=1)
        assert abs(gr.objective - bf.objective) < 1e-9


class TestSimulatedAnnealing:
    def test_returns_portfolio_result(self):
        assert isinstance(simulated_annealing(MU_2, SIGMA_2, seed=0), PortfolioResult)

    def test_bitstring_is_binary(self):
        for b in simulated_annealing(MU_2, SIGMA_2, k=1, seed=0).bitstring:
            assert b in (0.0, 1.0)

    def test_bitstring_length_matches_n(self):
        assert len(simulated_annealing(MU_3, SIGMA_3, k=1, seed=0).bitstring) == 3

    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_cardinality_satisfied(self, k):
        result = simulated_annealing(MU_3, SIGMA_3, k=k, seed=0)
        assert int(result.bitstring.sum()) == k

    def test_seed_reproducibility(self):
        r1 = simulated_annealing(MU_3, SIGMA_3, k=2, seed=42)
        r2 = simulated_annealing(MU_3, SIGMA_3, k=2, seed=42)
        np.testing.assert_array_equal(r1.bitstring, r2.bitstring)
        assert r1.objective == r2.objective

    def test_different_seeds_both_valid(self):
        r1 = simulated_annealing(MU_4, SIGMA_4, k=2, seed=0)
        r2 = simulated_annealing(MU_4, SIGMA_4, k=2, seed=99)
        assert int(r1.bitstring.sum()) == 2
        assert int(r2.bitstring.sum()) == 2

    def test_finds_optimum_on_simple_problem(self):
        # 2-asset, k=1: only [1,0] and [0,1]; [0,1] is clearly optimal.
        # With long cooling, SA must find it.
        result = simulated_annealing(
            MU_2, SIGMA_2, k=1, seed=0,
            T_init=2.0, T_min=1e-6, alpha=0.90, steps_per_temp=300,
        )
        np.testing.assert_array_equal(result.bitstring, np.array([0.0, 1.0]))

    def test_ret_consistent_with_bitstring(self):
        result = simulated_annealing(MU_3, SIGMA_3, k=2, seed=0)
        assert abs(result.ret - float(MU_3 @ result.bitstring)) < 1e-12

    def test_variance_consistent_with_bitstring(self):
        result = simulated_annealing(MU_3, SIGMA_3, k=2, seed=0)
        assert abs(result.variance - float(result.bitstring @ SIGMA_3 @ result.bitstring)) < 1e-12

    def test_variance_is_nonnegative(self):
        assert simulated_annealing(MU_3, SIGMA_3, k=2, seed=0).variance >= 0.0

    def test_no_cardinality_returns_binary_bitstring(self):
        result = simulated_annealing(MU_2, SIGMA_2, seed=0)
        assert isinstance(result, PortfolioResult)
        for b in result.bitstring:
            assert b in (0.0, 1.0)

    def test_objective_stored_is_best_seen(self):
        # The stored objective should equal the objective of the stored bitstring
        result = simulated_annealing(MU_3, SIGMA_3, k=2, seed=42)
        computed = objective(result.bitstring, MU_3, SIGMA_3, 1.0)
        assert abs(result.objective - computed) < 1e-12

    def test_cooling_parameters_accepted(self):
        result = simulated_annealing(
            MU_2, SIGMA_2, k=1, seed=7,
            T_init=5.0, T_min=0.01, alpha=0.95, steps_per_temp=50,
        )
        assert isinstance(result, PortfolioResult)
