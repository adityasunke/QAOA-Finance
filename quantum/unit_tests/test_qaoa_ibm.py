"""
Unit tests for qaoa_ibm.py.

Tests cover the two pure/simulator functions:
  - decode_counts: pure function, no hardware needed
  - optimize_on_simulator: uses Aer, runs quickly on tiny systems

The main() function and IBM hardware execution are not tested here as
they require a live IBM Quantum token and network access.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from quantum.qaoa_ibm import decode_counts, optimize_on_simulator
from quantum.hamiltonian import qubo_to_ising
from quantum.qubo import build_qubo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_h_j(n: int = 2) -> tuple[np.ndarray, np.ndarray, float]:
    mu = np.array([0.1, 0.2])[:n]
    sigma = np.eye(n) * 0.05
    Q, _ = build_qubo(mu, sigma, k=1)
    h, J, offset = qubo_to_ising(Q)
    return h, J, offset


# ---------------------------------------------------------------------------
# Tests for decode_counts
# ---------------------------------------------------------------------------

class TestDecodeCounts:
    def test_returns_numpy_array(self):
        counts = {'01': 600, '10': 400}
        result = decode_counts(counts, k=1)
        assert isinstance(result, np.ndarray)

    def test_selects_most_frequent_valid(self):
        # '10' reversed = [0, 1], sum=1. '01' reversed = [1, 0], sum=1.
        # Most frequent is '10' with count 700.
        counts = {'10': 700, '01': 300}
        result = decode_counts(counts, k=1)
        np.testing.assert_array_equal(result, np.array([0.0, 1.0]))

    def test_selects_valid_over_higher_frequency_invalid(self):
        # '11' has highest count but sum=2, violates k=1.
        # '10' is next with count 400, sum=1 (reversed: [0,1]).
        counts = {'11': 500, '10': 400, '01': 100}
        result = decode_counts(counts, k=1)
        assert int(result.sum()) == 1

    def test_no_cardinality_returns_most_frequent(self):
        counts = {'11': 800, '00': 200}
        result = decode_counts(counts, k=None)
        # '11' reversed = [1, 1]
        np.testing.assert_array_equal(result, np.array([1.0, 1.0]))

    def test_falls_back_when_no_valid_bitstring(self):
        # k=3 but all bitstrings have sum != 3
        counts = {'00': 500, '11': 500}
        result = decode_counts(counts, k=3)
        assert isinstance(result, np.ndarray)
        assert len(result) == 2

    def test_output_is_binary(self):
        counts = {'0101': 300, '1010': 400, '0110': 300}
        result = decode_counts(counts, k=2)
        for bit in result:
            assert bit in (0.0, 1.0)

    def test_output_length_matches_bitstring_length(self):
        counts = {'0110': 600, '1001': 400}
        result = decode_counts(counts, k=2)
        assert len(result) == 4

    def test_little_endian_reversal(self):
        # Qiskit big-endian '10' means qubit 1 = 1, qubit 0 = 0.
        # Reversed to little-endian: [0, 1] (qubit 0 = 0, qubit 1 = 1).
        counts = {'10': 1000}
        result = decode_counts(counts, k=None)
        np.testing.assert_array_equal(result, np.array([0.0, 1.0]))

    def test_single_bitstring_in_counts(self):
        counts = {'01': 1000}
        result = decode_counts(counts, k=1)
        np.testing.assert_array_equal(result, np.array([1.0, 0.0]))

    def test_k_zero_selects_all_zero_bitstring(self):
        counts = {'00': 600, '11': 400}
        result = decode_counts(counts, k=0)
        assert int(result.sum()) == 0

    def test_three_qubit_cardinality_two(self):
        # '110' reversed = [0,1,1], sum=2; '101' reversed=[1,0,1] sum=2
        counts = {'011': 100, '110': 800, '101': 100}
        result = decode_counts(counts, k=2)
        assert int(result.sum()) == 2

    def test_ties_broken_by_sort_order(self):
        # Both have same count; sorted() is stable so first in sorted order wins.
        counts = {'01': 500, '10': 500}
        result = decode_counts(counts, k=1)
        assert int(result.sum()) == 1


# ---------------------------------------------------------------------------
# Tests for optimize_on_simulator
# ---------------------------------------------------------------------------

class TestOptimizeOnSimulator:
    def test_returns_three_values(self):
        h, J, offset = _make_h_j(n=2)
        result = optimize_on_simulator(h, J, offset, p=1, seed=0, maxiter=5)
        assert len(result) == 3

    def test_optimal_params_length_equals_2p(self):
        h, J, offset = _make_h_j(n=2)
        for p in [1, 2]:
            opt_params, _, _ = optimize_on_simulator(h, J, offset, p=p, seed=0, maxiter=5)
            assert len(opt_params) == 2 * p

    def test_expectation_is_float(self):
        h, J, offset = _make_h_j(n=2)
        _, expectation, _ = optimize_on_simulator(h, J, offset, p=1, seed=0, maxiter=5)
        assert isinstance(expectation, float)

    def test_iteration_count_is_positive(self):
        h, J, offset = _make_h_j(n=2)
        _, _, n_iters = optimize_on_simulator(h, J, offset, p=1, seed=0, maxiter=10)
        assert n_iters > 0

    def test_iteration_count_does_not_exceed_maxiter(self):
        h, J, offset = _make_h_j(n=2)
        maxiter = 8
        _, _, n_iters = optimize_on_simulator(h, J, offset, p=1, seed=0, maxiter=maxiter)
        assert n_iters <= maxiter

    def test_optimal_params_are_numeric(self):
        h, J, offset = _make_h_j(n=2)
        opt_params, _, _ = optimize_on_simulator(h, J, offset, p=1, seed=0, maxiter=5)
        assert np.all(np.isfinite(opt_params))

    def test_seed_reproducibility(self):
        h, J, offset = _make_h_j(n=2)
        params1, exp1, _ = optimize_on_simulator(h, J, offset, p=1, seed=42, maxiter=5)
        params2, exp2, _ = optimize_on_simulator(h, J, offset, p=1, seed=42, maxiter=5)
        np.testing.assert_array_equal(params1, params2)
        assert exp1 == exp2

    def test_different_seeds_may_differ(self):
        h, J, offset = _make_h_j(n=2)
        params1, _, _ = optimize_on_simulator(h, J, offset, p=1, seed=0, maxiter=5)
        params2, _, _ = optimize_on_simulator(h, J, offset, p=1, seed=99, maxiter=5)
        # Seeds differ so initial points differ; params should not always be equal
        # (this is probabilistic but almost certain to hold)
        assert not np.allclose(params1, params2)
