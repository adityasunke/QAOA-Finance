import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from qiskit.quantum_info import SparsePauliOp

from quantum.qubo import build_qubo
from quantum.hamiltonian import (
    build_ising_hamiltonian,
    ising_energy,
    qubo_to_ising,
    verify_hamiltonian,
)

MU_2 = np.array([0.1, 0.2])
SIGMA_2 = np.array([[0.04, 0.01], [0.01, 0.09]])

MU_3 = np.array([0.05, 0.10, 0.15])
SIGMA_3 = np.eye(3) * 0.01


class TestQuboToIsing:
    def test_output_shapes(self):
        Q, _ = build_qubo(MU_2, SIGMA_2)
        h, J, offset = qubo_to_ising(Q)
        assert h.shape == (2,)
        assert J.shape == (2, 2)
        assert isinstance(offset, float)

    def test_j_is_upper_triangular(self):
        Q, _ = build_qubo(MU_3, SIGMA_3)
        _, J, _ = qubo_to_ising(Q)
        for i in range(3):
            for j in range(i + 1, 3):
                assert J[j, i] == 0.0, f"J[{j},{i}] should be zero"

    def test_diagonal_qubo_produces_zero_j(self):
        Q = np.diag([1.0, 2.0, 3.0])
        _, J, _ = qubo_to_ising(Q)
        np.testing.assert_array_equal(J, np.zeros((3, 3)))

    def test_offset_from_diagonal_qubo(self):
        # For diagonal Q, offset = sum(Q[i,i] / 2)
        Q = np.diag([2.0, 4.0])
        _, _, offset = qubo_to_ising(Q)
        assert abs(offset - 3.0) < 1e-12

    def test_h_from_diagonal_qubo(self):
        # For diagonal Q with no off-diagonal, h[i] = -Q[i,i] / 2
        Q = np.diag([2.0, 4.0])
        h, _, _ = qubo_to_ising(Q)
        np.testing.assert_allclose(h, [-1.0, -2.0])

    def test_j_coupling_from_off_diagonal(self):
        # Q[i,j] = c → J[i,j] = c/4
        Q = np.array([[0.0, 4.0], [0.0, 0.0]])
        _, J, _ = qubo_to_ising(Q)
        assert abs(J[0, 1] - 1.0) < 1e-12

    def test_h_contribution_from_off_diagonal(self):
        # Q[i,j] = c → h[i] -= c/4, h[j] -= c/4
        Q = np.array([[0.0, 4.0], [0.0, 0.0]])
        h, _, _ = qubo_to_ising(Q)
        assert abs(h[0] - (-1.0)) < 1e-12
        assert abs(h[1] - (-1.0)) < 1e-12

    def test_offset_includes_off_diagonal_contribution(self):
        # Q[i,j] = c → offset += c/4
        Q = np.array([[0.0, 4.0], [0.0, 0.0]])
        _, _, offset = qubo_to_ising(Q)
        assert abs(offset - 1.0) < 1e-12

    def test_energy_matches_qubo_for_known_state(self):
        Q = np.array([[1.0, 0.5], [0.0, 2.0]])
        h, J, offset = qubo_to_ising(Q)
        x = np.array([1.0, 0.0])
        z = 1 - 2 * x  # z = [-1, +1]
        qubo_val = float(x @ Q @ x)
        ising_val = ising_energy(z, h, J) + offset
        assert abs(qubo_val - ising_val) < 1e-12

    def test_energy_matches_qubo_for_all_2bit_states(self):
        Q, _ = build_qubo(MU_2, SIGMA_2)
        h, J, offset = qubo_to_ising(Q)
        for bits in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            x = np.array(bits, dtype=float)
            z = 1 - 2 * x
            qubo_val = float(x @ Q @ x)
            ising_val = ising_energy(z, h, J) + offset
            assert abs(qubo_val - ising_val) < 1e-9, f"Mismatch at x={bits}"


class TestBuildIsingHamiltonian:
    def test_returns_sparse_pauli_op(self):
        Q, _ = build_qubo(MU_2, SIGMA_2)
        h, J, _ = qubo_to_ising(Q)
        H = build_ising_hamiltonian(h, J)
        assert isinstance(H, SparsePauliOp)

    def test_num_qubits_equals_n(self):
        for n, mu, sigma in [
            (2, MU_2, SIGMA_2),
            (3, MU_3, SIGMA_3),
        ]:
            Q, _ = build_qubo(mu, sigma)
            h, J, _ = qubo_to_ising(Q)
            H = build_ising_hamiltonian(h, J)
            assert H.num_qubits == n

    def test_zero_h_and_j_returns_identity(self):
        n = 2
        h = np.zeros(n)
        J = np.zeros((n, n))
        H = build_ising_hamiltonian(h, J)
        assert isinstance(H, SparsePauliOp)
        assert H.num_qubits == n

    def test_nonzero_h_produces_z_terms(self):
        h = np.array([1.0, 0.0])
        J = np.zeros((2, 2))
        H = build_ising_hamiltonian(h, J)
        pauli_strs = [str(p) for p in H.paulis]
        assert any('Z' in s for s in pauli_strs)

    def test_nonzero_j_produces_zz_terms(self):
        h = np.zeros(2)
        J = np.array([[0.0, 1.0], [0.0, 0.0]])
        H = build_ising_hamiltonian(h, J)
        pauli_strs = [str(p) for p in H.paulis]
        zz_terms = [s for s in pauli_strs if s.count('Z') == 2]
        assert len(zz_terms) >= 1

    def test_coefficients_match_h_values(self):
        h = np.array([3.0, 0.0])
        J = np.zeros((2, 2))
        H = build_ising_hamiltonian(h, J)
        # Find the Z term coefficient for qubit 0
        for pauli, coeff in zip(H.paulis, H.coeffs):
            if str(pauli) == 'IZ':
                assert abs(coeff.real - 3.0) < 1e-12

    def test_coefficients_match_j_values(self):
        h = np.zeros(2)
        J = np.array([[0.0, 2.5], [0.0, 0.0]])
        H = build_ising_hamiltonian(h, J)
        for pauli, coeff in zip(H.paulis, H.coeffs):
            if str(pauli) == 'ZZ':
                assert abs(coeff.real - 2.5) < 1e-12


class TestIsingEnergy:
    def test_all_plus_one_spins_linear_term(self):
        h = np.array([1.0, 2.0])
        J = np.zeros((2, 2))
        z = np.array([1.0, 1.0])
        assert abs(ising_energy(z, h, J) - 3.0) < 1e-12

    def test_all_minus_one_spins_linear_term(self):
        h = np.array([1.0, 2.0])
        J = np.zeros((2, 2))
        z = np.array([-1.0, -1.0])
        assert abs(ising_energy(z, h, J) + 3.0) < 1e-12

    def test_coupling_parallel_spins(self):
        h = np.zeros(2)
        J = np.array([[0.0, 1.0], [0.0, 0.0]])
        z = np.array([1.0, 1.0])
        assert abs(ising_energy(z, h, J) - 1.0) < 1e-12

    def test_coupling_antiparallel_spins(self):
        h = np.zeros(2)
        J = np.array([[0.0, 1.0], [0.0, 0.0]])
        z = np.array([1.0, -1.0])
        assert abs(ising_energy(z, h, J) + 1.0) < 1e-12

    def test_zero_h_zero_j_gives_zero_energy(self):
        h = np.zeros(3)
        J = np.zeros((3, 3))
        z = np.array([1.0, -1.0, 1.0])
        assert ising_energy(z, h, J) == 0.0

    def test_single_spin_system(self):
        h = np.array([5.0])
        J = np.zeros((1, 1))
        assert abs(ising_energy(np.array([1.0]), h, J) - 5.0) < 1e-12
        assert abs(ising_energy(np.array([-1.0]), h, J) + 5.0) < 1e-12

    def test_three_qubit_combined(self):
        h = np.array([1.0, 2.0, 3.0])
        J = np.array([[0.0, 0.5, 0.0],
                      [0.0, 0.0, 1.0],
                      [0.0, 0.0, 0.0]])
        z = np.array([1.0, -1.0, 1.0])
        # E = 1*1 + 2*(-1) + 3*1 + 0.5*(1*-1) + 1.0*(-1*1)
        # E = 1 - 2 + 3 - 0.5 - 1 = 0.5
        expected = 1.0 - 2.0 + 3.0 + 0.5 * (1.0 * -1.0) + 1.0 * (-1.0 * 1.0)
        assert abs(ising_energy(z, h, J) - expected) < 1e-12


class TestVerifyHamiltonian:
    def test_small_system_no_cardinality(self):
        Q, _ = build_qubo(MU_2, SIGMA_2, lam=1.0)
        h, J, offset = qubo_to_ising(Q)
        assert verify_hamiltonian(h, J, offset, Q) is True

    def test_small_system_with_cardinality(self):
        Q, _ = build_qubo(MU_3, SIGMA_3, lam=1.0, k=1)
        h, J, offset = qubo_to_ising(Q)
        assert verify_hamiltonian(h, J, offset, Q) is True

    def test_corrupt_offset_raises(self):
        Q, _ = build_qubo(MU_2, SIGMA_2)
        h, J, offset = qubo_to_ising(Q)
        with pytest.raises(AssertionError):
            verify_hamiltonian(h, J, offset + 999.0, Q)

    def test_corrupt_h_raises(self):
        Q, _ = build_qubo(MU_2, SIGMA_2)
        h, J, offset = qubo_to_ising(Q)
        h_bad = h.copy()
        h_bad[0] += 999.0
        with pytest.raises(AssertionError):
            verify_hamiltonian(h_bad, J, offset, Q)
