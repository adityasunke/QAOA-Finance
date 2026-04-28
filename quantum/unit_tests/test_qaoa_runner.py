import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from quantum.qaoa_runner import QAOAResult, run_qaoa

# Tiny 2-asset problem for fast tests
MU_2 = np.array([0.10, 0.20])
SIGMA_2 = np.array([[0.04, 0.01],
                    [0.01, 0.09]])

# Slightly larger 3-asset problem
MU_3 = np.array([0.05, 0.12, 0.18])
SIGMA_3 = np.diag([0.03, 0.05, 0.08])


class TestQAOAResult:
    def test_instantiation(self):
        result = QAOAResult(
            bitstring=np.array([1.0, 0.0]),
            objective=-0.5,
            ret=0.1,
            variance=0.04,
            expectation=-0.4,
            n_iters=10,
            p=1,
            runtime=0.1,
        )
        assert result is not None

    def test_default_counts_is_empty_dict(self):
        result = QAOAResult(
            bitstring=np.array([1.0, 0.0]),
            objective=0.0,
            ret=0.0,
            variance=0.0,
            expectation=0.0,
            n_iters=0,
            p=1,
            runtime=0.0,
        )
        assert result.counts == {}

    def test_counts_can_be_set(self):
        counts = {'01': 500, '10': 300}
        result = QAOAResult(
            bitstring=np.array([1.0, 0.0]),
            objective=0.0,
            ret=0.0,
            variance=0.0,
            expectation=0.0,
            n_iters=5,
            p=2,
            runtime=1.5,
            counts=counts,
        )
        assert result.counts == counts

    def test_field_types(self):
        bits = np.array([0.0, 1.0])
        result = QAOAResult(
            bitstring=bits,
            objective=-0.1,
            ret=0.2,
            variance=0.09,
            expectation=-0.05,
            n_iters=42,
            p=3,
            runtime=7.3,
        )
        assert isinstance(result.bitstring, np.ndarray)
        assert isinstance(result.objective, float)
        assert isinstance(result.ret, float)
        assert isinstance(result.variance, float)
        assert isinstance(result.expectation, float)
        assert isinstance(result.n_iters, int)
        assert isinstance(result.p, int)
        assert isinstance(result.runtime, float)


class TestRunQaoa:
    """Integration tests using a tiny 2-asset problem with minimal iterations."""

    def test_returns_qaoa_result(self):
        result = run_qaoa(MU_2, SIGMA_2, lam=1.0, k=1, p=1, shots=100, seed=0, maxiter=10)
        assert isinstance(result, QAOAResult)

    def test_bitstring_length_matches_n(self):
        result = run_qaoa(MU_2, SIGMA_2, lam=1.0, k=1, p=1, shots=100, seed=0, maxiter=10)
        assert len(result.bitstring) == 2

    def test_bitstring_is_binary(self):
        result = run_qaoa(MU_2, SIGMA_2, lam=1.0, k=1, p=1, shots=100, seed=0, maxiter=10)
        for bit in result.bitstring:
            assert bit in (0.0, 1.0)

    def test_cardinality_constraint_satisfied(self):
        k = 1
        result = run_qaoa(MU_2, SIGMA_2, lam=1.0, k=k, p=1, shots=200, seed=0, maxiter=10)
        assert int(result.bitstring.sum()) == k

    def test_p_field_matches_input(self):
        for p in [1, 2]:
            result = run_qaoa(MU_2, SIGMA_2, lam=1.0, k=1, p=p, shots=100, seed=0, maxiter=5)
            assert result.p == p

    def test_runtime_is_positive(self):
        result = run_qaoa(MU_2, SIGMA_2, lam=1.0, k=1, p=1, shots=100, seed=0, maxiter=10)
        assert result.runtime > 0.0

    def test_n_iters_is_positive(self):
        result = run_qaoa(MU_2, SIGMA_2, lam=1.0, k=1, p=1, shots=100, seed=0, maxiter=10)
        assert result.n_iters > 0

    def test_counts_dict_is_nonempty(self):
        result = run_qaoa(MU_2, SIGMA_2, lam=1.0, k=1, p=1, shots=100, seed=0, maxiter=10)
        assert len(result.counts) > 0

    def test_counts_sum_to_shots(self):
        shots = 200
        result = run_qaoa(MU_2, SIGMA_2, lam=1.0, k=1, p=1, shots=shots, seed=0, maxiter=10)
        assert sum(result.counts.values()) == shots

    def test_variance_is_nonnegative(self):
        result = run_qaoa(MU_2, SIGMA_2, lam=1.0, k=1, p=1, shots=100, seed=0, maxiter=10)
        assert result.variance >= 0.0

    def test_ret_consistent_with_mu_and_bitstring(self):
        result = run_qaoa(MU_2, SIGMA_2, lam=1.0, k=1, p=1, shots=100, seed=0, maxiter=10)
        expected_ret = float(MU_2 @ result.bitstring)
        assert abs(result.ret - expected_ret) < 1e-9

    def test_variance_consistent_with_sigma_and_bitstring(self):
        result = run_qaoa(MU_2, SIGMA_2, lam=1.0, k=1, p=1, shots=100, seed=0, maxiter=10)
        expected_var = float(result.bitstring @ SIGMA_2 @ result.bitstring)
        assert abs(result.variance - expected_var) < 1e-9

    def test_no_cardinality_constraint(self):
        result = run_qaoa(MU_2, SIGMA_2, lam=1.0, k=None, p=1, shots=100, seed=0, maxiter=10)
        assert isinstance(result, QAOAResult)
        assert len(result.bitstring) == 2

    def test_seed_produces_reproducible_expectation(self):
        # The expectation value is derived from the statevector (no shot noise),
        # so it should be identical for the same seed and maxiter.
        r1 = run_qaoa(MU_2, SIGMA_2, lam=1.0, k=1, p=1, shots=100, seed=7, maxiter=5)
        r2 = run_qaoa(MU_2, SIGMA_2, lam=1.0, k=1, p=1, shots=100, seed=7, maxiter=5)
        assert r1.expectation == r2.expectation

    def test_three_asset_problem(self):
        result = run_qaoa(MU_3, SIGMA_3, lam=1.0, k=1, p=1, shots=100, seed=0, maxiter=10)
        assert len(result.bitstring) == 3
        assert int(result.bitstring.sum()) == 1
