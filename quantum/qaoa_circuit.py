"""
QAOA circuit builder for binary portfolio optimization.

Constructs the standard QAOA ansatz of depth p:

    |ψ(γ,β)⟩ = exp(-iβₚ HB) exp(-iγₚ HC) ··· exp(-iβ₁ HB) exp(-iγ₁ HC) |+⟩^n

where:
    HC = Σᵢ hᵢ Zᵢ + Σᵢ<ⱼ Jᵢⱼ ZᵢZⱼ   (cost Hamiltonian, from Ising mapping)
    HB = Σᵢ Xᵢ                         (mixer Hamiltonian)

Circuit conventions:
    - exp(-iγ hᵢ Zᵢ)      → RZ(2γhᵢ) on qubit i
    - exp(-iγ Jᵢⱼ ZᵢZⱼ)  → CNOT(i,j), RZ(2γJᵢⱼ, j), CNOT(i,j)
    - exp(-iβ Xᵢ)          → RX(2β) on qubit i
"""

import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector


def build_qaoa_circuit(
    h: np.ndarray,
    J: np.ndarray,
    p: int = 1,
) -> QuantumCircuit:
    """
    Build a parameterized QAOA circuit of depth p.

    Parameters
    ----------
    h : (n,) Ising local field coefficients (hᵢ for Zᵢ terms)
    J : (n, n) upper-triangular Ising coupling matrix (Jᵢⱼ for ZᵢZⱼ terms)
    p : QAOA depth (number of cost+mixer repetitions)

    Returns
    -------
    QuantumCircuit with 2p free parameters:
        gamma[0..p-1]  — cost layer angles
        beta[0..p-1]   — mixer layer angles
    """
    n = len(h)

    gamma = ParameterVector("γ", p)
    beta = ParameterVector("β", p)

    qc = QuantumCircuit(n)

    # --- Initial state: uniform superposition ---
    qc.h(range(n))

    # --- p repetitions of cost + mixer layers ---
    for layer in range(p):
        g = gamma[layer]
        b = beta[layer]

        # Cost layer: exp(-i γ HC)
        # Single-qubit Z terms: RZ(2γhᵢ)
        for i in range(n):
            if h[i] != 0.0:
                qc.rz(2 * g * h[i], i)

        # Two-qubit ZZ terms: CNOT, RZ(2γJᵢⱼ), CNOT
        for i in range(n):
            for j in range(i + 1, n):
                if J[i, j] != 0.0:
                    qc.cx(i, j)
                    qc.rz(2 * g * J[i, j], j)
                    qc.cx(i, j)

        # Mixer layer: exp(-i β HB) = RX(2β) on each qubit
        for i in range(n):
            qc.rx(2 * b, i)

    qc.measure_all()
    return qc


def build_qaoa_circuit_no_measure(
    h: np.ndarray,
    J: np.ndarray,
    p: int = 1,
) -> QuantumCircuit:
    """
    Same as build_qaoa_circuit but without the terminal measurement.
    Used during optimization (statevector simulation needs no measurements).
    """
    n = len(h)

    gamma = ParameterVector("γ", p)
    beta = ParameterVector("β", p)

    qc = QuantumCircuit(n)
    qc.h(range(n))

    for layer in range(p):
        g = gamma[layer]
        b = beta[layer]

        for i in range(n):
            if h[i] != 0.0:
                qc.rz(2 * g * h[i], i)

        for i in range(n):
            for j in range(i + 1, n):
                if J[i, j] != 0.0:
                    qc.cx(i, j)
                    qc.rz(2 * g * J[i, j], j)
                    qc.cx(i, j)

        for i in range(n):
            qc.rx(2 * b, i)

    return qc


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from data.generate_data import generate_assets
    from quantum.qubo import build_qubo
    from quantum.hamiltonian import qubo_to_ising

    n, k = 4, 2
    mu, Sigma = generate_assets(n, seed=42)
    Q, _ = build_qubo(mu, Sigma, lam=1.0, k=k)
    h, J, _ = qubo_to_ising(Q)

    for p in [1, 2, 3]:
        qc = build_qaoa_circuit(h, J, p=p)
        print(f"\nQAOA circuit p={p}, n={n}:")
        print(f"  Qubits: {qc.num_qubits}")
        print(f"  Parameters: {qc.num_parameters}  (expect {2*p})")
        print(f"  Depth: {qc.depth()}")
        print(f"  Gates: {dict(qc.count_ops())}")
