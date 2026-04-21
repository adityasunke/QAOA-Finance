"""
QAOA circuit builder for binary portfolio optimization.

Constructs the standard QAOA ansatz of depth p with 2p free parameters
[γ₁,...,γₚ, β₁,...,βₚ].
"""

import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector


def build_qaoa_circuit(
    h: np.ndarray,
    J: np.ndarray,
    p: int = 1,
) -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """
    Build a parameterized QAOA circuit of depth p.

    Circuit structure:
      1. Initial state: H on all n qubits → uniform superposition |+>^n
      2. Repeat p times:
         - Cost layer: exp(-i γ H_C) via RZ(2γhᵢ) and RZZ(2γJᵢⱼ) gates
         - Mixer layer: exp(-i β H_B) = RX(2β) on each qubit

    Parameters
    ----------
    h : (n,) local field coefficients from the Ising Hamiltonian
    J : (n, n) upper-triangular coupling matrix from the Ising Hamiltonian
    p : circuit depth (number of QAOA layers)

    Returns
    -------
    qc     : parameterized QuantumCircuit with 2p free parameters
    gammas : ParameterVector of length p (cost layer angles)
    betas  : ParameterVector of length p (mixer layer angles)
    """
    n = len(h)
    gammas = ParameterVector('γ', p)
    betas = ParameterVector('β', p)

    qc = QuantumCircuit(n)

    # Initial state: uniform superposition
    qc.h(range(n))

    for layer in range(p):
        # --- Cost layer: exp(-i γ H_C) -----------------------------------
        # Single-qubit Z terms: exp(-i γ hᵢ Zᵢ) = RZ(2γhᵢ) on qubit i
        for i in range(n):
            if h[i] != 0.0:
                qc.rz(2.0 * gammas[layer] * float(h[i]), i)

        # Two-qubit ZZ terms: exp(-i γ Jᵢⱼ ZᵢZⱼ) = RZZ(2γJᵢⱼ) on (i, j)
        for i in range(n):
            for j in range(i + 1, n):
                if J[i, j] != 0.0:
                    qc.rzz(2.0 * gammas[layer] * float(J[i, j]), i, j)

        # --- Mixer layer: exp(-i β Σ Xᵢ) = Π RX(2β) on each qubit ------
        for i in range(n):
            qc.rx(2.0 * betas[layer], i)

    return qc, gammas, betas


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from data.generate_data import load_assets, DEFAULT_TICKERS
    from quantum.qubo import build_qubo
    from quantum.hamiltonian import qubo_to_ising

    tickers = DEFAULT_TICKERS[:4]
    mu, Sigma = load_assets(tickers)
    Q, _ = build_qubo(mu, Sigma, lam=1.0, k=2)
    h, J, _ = qubo_to_ising(Q)

    for p in [1, 2, 3]:
        qc, gammas, betas = build_qaoa_circuit(h, J, p=p)
        print(f"p={p}: {qc.num_qubits} qubits, "
              f"{qc.num_parameters} parameters, "
              f"{qc.depth()} depth, "
              f"{len(qc)} gates")
    print()
    print("p=1 circuit:")
    qc1, _, _ = build_qaoa_circuit(h, J, p=1)
    print(qc1.draw(fold=-1))
