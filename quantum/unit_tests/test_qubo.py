import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from quantum.qubo import build_qubo, evaluate_qubo, verify_qubo

MU_2 = np.array([0.1, 0.2])
SIGMA_2 = np.array([[0.04, 0.01], [0.01, 0.09]])

MU_3 = np.array([0.05, 0.10, 0.15])
SIGMA_3 = np.eye(3) * 0.01


class TestBuildQubo:
    def test_output_shape(self):
        Q, offset = build_qubo(MU_2, SIGMA_2)
        assert Q.shape == (2, 2)
        assert isinstance(offset, float)

    def test_upper_triangular(self):
        Q, _ = build_qubo(MU_3, SIGMA_3)
        for i in range(3):
            for j in range(i + 1, 3):
                assert Q[j, i] == 0.0, f"Q[{j},{i}] should be zero (lower triangle)"

    def test_no_cardinality_offset_is_zero(self):
        _, offset = build_qubo(MU_2, SIGMA_2)
        assert offset == 0.0

    def test_cardinality_offset_equals_penalty_k_squared(self):
        penalty = 10.0
        k = 2
        _, offset = build_qubo(MU_3, SIGMA_3, k=k, penalty=penalty)
        assert abs(offset - penalty * k * k) < 1e-12

    def test_cardinality_offset_nonzero_when_k_set(self):
        _, offset = build_qubo(MU_3, SIGMA_3, k=1)
        assert offset > 0.0

    def test_diagonal_encodes_sigma_minus_lam_mu(self):
        lam = 1.0
        Q, _ = build_qubo(MU_2, SIGMA_2, lam=lam)
        for i in range(2):
            expected = SIGMA_2[i, i] - lam * MU_2[i]
            assert abs(Q[i, i] - expected) < 1e-12

    def test_off_diagonal_encodes_symmetric_sigma(self):
        Q, _ = build_qubo(MU_2, SIGMA_2)
        expected = SIGMA_2[0, 1] + SIGMA_2[1, 0]
        assert abs(Q[0, 1] - expected) < 1e-12

    def test_lambda_scaling_affects_diagonal(self):
        Q1, _ = build_qubo(MU_2, SIGMA_2, lam=1.0)
        Q2, _ = build_qubo(MU_2, SIGMA_2, lam=2.0)
        # Larger lambda reduces the diagonal (increases return contribution)
        for i in range(2):
            assert Q2[i, i] < Q1[i, i]

    def test_auto_penalty_changes_diagonal(self):
        Q_no_k, _ = build_qubo(MU_3, SIGMA_3, lam=1.0)
        Q_with_k, _ = build_qubo(MU_3, SIGMA_3, lam=1.0, k=1)
        assert not np.allclose(np.diag(Q_no_k), np.diag(Q_with_k))

    def test_cardinality_off_diagonal_increased(self):
        Q_no_k, _ = build_qubo(MU_3, SIGMA_3, lam=1.0)
        penalty = 5.0
        Q_with_k, _ = build_qubo(MU_3, SIGMA_3, lam=1.0, k=1, penalty=penalty)
        # Each off-diagonal entry gains 2*penalty
        assert abs(Q_with_k[0, 1] - (Q_no_k[0, 1] + 2 * penalty)) < 1e-12

    def test_n_equals_1(self):
        mu = np.array([0.1])
        sigma = np.array([[0.05]])
        Q, offset = build_qubo(mu, sigma)
        assert Q.shape == (1, 1)
        assert offset == 0.0

    def test_symmetric_sigma_gives_double_off_diagonal(self):
        # When Sigma is symmetric, Q[i,j] = 2 * Sigma[i,j] for i < j
        mu = np.zeros(3)
        sigma = np.array([[1.0, 0.5, 0.3],
                          [0.5, 2.0, 0.4],
                          [0.3, 0.4, 1.5]])
        Q, _ = build_qubo(mu, sigma)
        assert abs(Q[0, 1] - 2 * sigma[0, 1]) < 1e-12
        assert abs(Q[0, 2] - 2 * sigma[0, 2]) < 1e-12
        assert abs(Q[1, 2] - 2 * sigma[1, 2]) < 1e-12


class TestEvaluateQubo:
    def test_zero_vector_gives_zero(self):
        Q, _ = build_qubo(MU_2, SIGMA_2)
        assert evaluate_qubo(Q, np.zeros(2)) == 0.0

    def test_returns_float(self):
        Q, _ = build_qubo(MU_2, SIGMA_2)
        result = evaluate_qubo(Q, np.array([1.0, 0.0]))
        assert isinstance(result, float)

    def test_single_asset_selected(self):
        Q = np.array([[3.0, 2.0], [0.0, 5.0]])
        x = np.array([1.0, 0.0])
        assert abs(evaluate_qubo(Q, x) - 3.0) < 1e-12

    def test_both_assets_selected(self):
        Q = np.array([[1.0, 2.0], [0.0, 3.0]])
        x = np.ones(2)
        # x^T Q x = 1*1 + 1*2*1 + 0 + 1*3 = 6
        assert abs(evaluate_qubo(Q, x) - 6.0) < 1e-12

    def test_diagonal_only_qubo(self):
        Q = np.diag([4.0, 6.0, 8.0])
        x = np.array([1.0, 0.0, 1.0])
        assert abs(evaluate_qubo(Q, x) - 12.0) < 1e-12

    def test_matches_direct_computation(self):
        Q, _ = build_qubo(MU_3, SIGMA_3)
        x = np.array([1.0, 0.0, 1.0])
        expected = float(x @ Q @ x)
        assert abs(evaluate_qubo(Q, x) - expected) < 1e-12


class TestVerifyQubo:
    def test_passes_no_cardinality(self):
        Q, offset = build_qubo(MU_2, SIGMA_2, lam=1.0)
        assert verify_qubo(Q, offset, MU_2, SIGMA_2, lam=1.0) is True

    def test_passes_with_cardinality(self):
        Q, offset = build_qubo(MU_3, SIGMA_3, lam=1.0, k=1)
        assert verify_qubo(Q, offset, MU_3, SIGMA_3, lam=1.0, k=1) is True

    def test_passes_explicit_penalty(self):
        penalty = 5.0
        Q, offset = build_qubo(MU_2, SIGMA_2, lam=1.0, k=1, penalty=penalty)
        assert verify_qubo(Q, offset, MU_2, SIGMA_2, lam=1.0, k=1, penalty=penalty) is True

    def test_passes_lambda_two(self):
        Q, offset = build_qubo(MU_3, SIGMA_3, lam=2.0)
        assert verify_qubo(Q, offset, MU_3, SIGMA_3, lam=2.0) is True

    def test_corrupt_qubo_raises(self):
        Q, offset = build_qubo(MU_2, SIGMA_2)
        Q_bad = Q.copy()
        Q_bad[0, 0] += 999.0
        with pytest.raises(AssertionError):
            verify_qubo(Q_bad, offset, MU_2, SIGMA_2)
