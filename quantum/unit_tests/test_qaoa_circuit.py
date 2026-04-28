import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from qiskit.circuit import ParameterVector, QuantumCircuit

from quantum.hamiltonian import qubo_to_ising
from quantum.qaoa_circuit import build_qaoa_circuit
from quantum.qubo import build_qubo


def _make_h_j(n: int = 3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.05, 0.2, n)
    sigma = np.eye(n) * 0.05
    Q, _ = build_qubo(mu, sigma, k=max(1, n // 2))
    h, J, _ = qubo_to_ising(Q)
    return h, J


class TestBuildQaoaCircuitReturnTypes:
    def test_returns_three_objects(self):
        h, J = _make_h_j()
        result = build_qaoa_circuit(h, J, p=1)
        assert len(result) == 3

    def test_first_element_is_quantum_circuit(self):
        h, J = _make_h_j()
        qc, _, _ = build_qaoa_circuit(h, J, p=1)
        assert isinstance(qc, QuantumCircuit)

    def test_second_element_is_parameter_vector(self):
        h, J = _make_h_j()
        _, gammas, _ = build_qaoa_circuit(h, J, p=1)
        assert isinstance(gammas, ParameterVector)

    def test_third_element_is_parameter_vector(self):
        h, J = _make_h_j()
        _, _, betas = build_qaoa_circuit(h, J, p=1)
        assert isinstance(betas, ParameterVector)

    def test_gamma_vector_named_correctly(self):
        h, J = _make_h_j()
        _, gammas, _ = build_qaoa_circuit(h, J, p=1)
        assert gammas.name == 'γ'

    def test_beta_vector_named_correctly(self):
        h, J = _make_h_j()
        _, _, betas = build_qaoa_circuit(h, J, p=1)
        assert betas.name == 'β'


class TestQaoaCircuitStructure:
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_num_qubits_matches_n(self, n):
        h, J = _make_h_j(n)
        qc, _, _ = build_qaoa_circuit(h, J, p=1)
        assert qc.num_qubits == n

    @pytest.mark.parametrize("p", [1, 2, 3])
    def test_gamma_vector_length_equals_p(self, p):
        h, J = _make_h_j()
        _, gammas, _ = build_qaoa_circuit(h, J, p=p)
        assert len(gammas) == p

    @pytest.mark.parametrize("p", [1, 2, 3])
    def test_beta_vector_length_equals_p(self, p):
        h, J = _make_h_j()
        _, _, betas = build_qaoa_circuit(h, J, p=p)
        assert len(betas) == p

    @pytest.mark.parametrize("p", [1, 2, 3])
    def test_num_parameters_equals_2p(self, p):
        h, J = _make_h_j()
        qc, _, _ = build_qaoa_circuit(h, J, p=p)
        assert qc.num_parameters == 2 * p

    def test_circuit_has_gates(self):
        h, J = _make_h_j()
        qc, _, _ = build_qaoa_circuit(h, J, p=1)
        assert len(qc) > 0

    def test_depth_nondecreasing_with_p(self):
        h, J = _make_h_j()
        qc1, _, _ = build_qaoa_circuit(h, J, p=1)
        qc2, _, _ = build_qaoa_circuit(h, J, p=2)
        assert qc2.depth() >= qc1.depth()

    def test_gate_count_nondecreasing_with_p(self):
        h, J = _make_h_j()
        qc1, _, _ = build_qaoa_circuit(h, J, p=1)
        qc2, _, _ = build_qaoa_circuit(h, J, p=2)
        assert len(qc2) >= len(qc1)


class TestQaoaCircuitParameterBinding:
    def test_can_bind_all_parameters(self):
        h, J = _make_h_j()
        qc, gammas, betas = build_qaoa_circuit(h, J, p=1)
        param_dict = {gammas[0]: 0.5, betas[0]: 0.3}
        bound = qc.assign_parameters(param_dict)
        assert bound.num_parameters == 0

    def test_can_bind_multi_layer_parameters(self):
        h, J = _make_h_j()
        p = 3
        qc, gammas, betas = build_qaoa_circuit(h, J, p=p)
        param_dict = {g: 0.1 * (i + 1) for i, g in enumerate(gammas)}
        param_dict.update({b: 0.2 * (i + 1) for i, b in enumerate(betas)})
        bound = qc.assign_parameters(param_dict)
        assert bound.num_parameters == 0

    def test_partial_binding_leaves_free_parameters(self):
        h, J = _make_h_j()
        p = 2
        qc, gammas, betas = build_qaoa_circuit(h, J, p=p)
        partial = qc.assign_parameters({gammas[0]: 0.5, gammas[1]: 0.7})
        assert partial.num_parameters == p  # betas still free


class TestQaoaCircuitContent:
    def test_contains_hadamard_gates(self):
        h, J = _make_h_j()
        qc, _, _ = build_qaoa_circuit(h, J, p=1)
        gate_names = [instr.operation.name for instr in qc]
        assert 'h' in gate_names

    def test_contains_rx_gates_for_mixer(self):
        h, J = _make_h_j()
        qc, _, _ = build_qaoa_circuit(h, J, p=1)
        gate_names = [instr.operation.name for instr in qc]
        assert 'rx' in gate_names

    def test_nonzero_h_produces_rz_gates(self):
        # Use nonzero h to guarantee RZ gates appear
        h = np.array([1.0, 2.0])
        J = np.zeros((2, 2))
        qc, _, _ = build_qaoa_circuit(h, J, p=1)
        gate_names = [instr.operation.name for instr in qc]
        assert 'rz' in gate_names

    def test_nonzero_j_produces_rzz_gates(self):
        h = np.zeros(2)
        J = np.array([[0.0, 1.0], [0.0, 0.0]])
        qc, _, _ = build_qaoa_circuit(h, J, p=1)
        gate_names = [instr.operation.name for instr in qc]
        assert 'rzz' in gate_names

    def test_zero_h_skips_rz_gates(self):
        h = np.zeros(2)
        J = np.zeros((2, 2))
        qc, _, _ = build_qaoa_circuit(h, J, p=1)
        gate_names = [instr.operation.name for instr in qc]
        assert 'rz' not in gate_names

    def test_zero_j_skips_rzz_gates(self):
        h = np.array([1.0, 1.0])
        J = np.zeros((2, 2))
        qc, _, _ = build_qaoa_circuit(h, J, p=1)
        gate_names = [instr.operation.name for instr in qc]
        assert 'rzz' not in gate_names
