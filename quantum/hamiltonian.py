"""
Ising Hamiltonian construction from a QUBO matrix.

Substitution: xᵢ = (1 − zᵢ) / 2,  zᵢ ∈ {−1, +1}

This maps the QUBO objective xᵀQx to an Ising energy:
    H = Σᵢ hᵢ Zᵢ + Σᵢ<ⱼ Jᵢⱼ ZᵢZⱼ  (+  constant offset)

The SparsePauliOp built here is used directly as the QAOA cost Hamiltonian.
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp


def qubo_to_ising(
    Q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Convert an upper-triangular QUBO matrix to Ising parameters.

    Derivation
    ----------
    For binary variables xᵢ ∈ {0, 1} and spin variables zᵢ ∈ {−1, +1}:

        xᵢ = (1 − zᵢ) / 2

    Substituting into f(x) = Σᵢ Qᵢᵢ xᵢ + Σᵢ<ⱼ Qᵢⱼ xᵢ xⱼ:

        xᵢ xⱼ = (1 − zᵢ)(1 − zⱼ) / 4
               = (1 − zᵢ − zⱼ + zᵢzⱼ) / 4

        xᵢ    = (1 − zᵢ) / 2

    Collecting spin-independent (constant), linear (hᵢ), and quadratic (Jᵢⱼ) terms:

        constant += Qᵢᵢ / 2         (from linear terms)
        hᵢ       -= Qᵢᵢ / 2         (linear in zᵢ)
        constant += Qᵢⱼ / 4         (from quadratic terms)
        hᵢ       -= Qᵢⱼ / 4         (linear in zᵢ)
        hⱼ       -= Qᵢⱼ / 4         (linear in zⱼ)
        Jᵢⱼ      += Qᵢⱼ / 4         (quadratic in zᵢ zⱼ)

    Parameters
    ----------
    Q : (n, n) upper-triangular QUBO matrix

    Returns
    -------
    h      : (n,) local fields (coefficient of Zᵢ)
    J      : (n, n) coupling matrix (upper-triangular; J[i,j] for i < j)
    offset : constant energy offset
    """
    n = Q.shape[0]
    h = np.zeros(n)
    J = np.zeros((n, n))
    offset = 0.0

    for i in range(n):
        # Linear (diagonal) term: Qᵢᵢ xᵢ = Qᵢᵢ (1 − zᵢ) / 2
        offset += Q[i, i] / 2
        h[i]   -= Q[i, i] / 2

    for i in range(n):
        for j in range(i + 1, n):
            # Quadratic term: Qᵢⱼ xᵢ xⱼ = Qᵢⱼ (1 − zᵢ − zⱼ + zᵢzⱼ) / 4
            offset += Q[i, j] / 4
            h[i]   -= Q[i, j] / 4
            h[j]   -= Q[i, j] / 4
            J[i, j] += Q[i, j] / 4

    return h, J, offset


def build_ising_hamiltonian(
    h: np.ndarray,
    J: np.ndarray,
) -> SparsePauliOp:
    """
    Build the Qiskit SparsePauliOp for the Ising Hamiltonian.

    H = Σᵢ hᵢ Zᵢ + Σᵢ<ⱼ Jᵢⱼ ZᵢZⱼ

    Qubit ordering: qubit 0 corresponds to spin z₀ (leftmost in Pauli string,
    but Qiskit uses little-endian convention so Pauli string index 0 is the
    least significant / rightmost qubit).  We adopt the convention that
    qubit i ↔ zᵢ throughout.

    Parameters
    ----------
    h : (n,) local field coefficients
    J : (n, n) upper-triangular coupling matrix

    Returns
    -------
    SparsePauliOp representing H (without the constant offset)
    """
    n = len(h)
    pauli_list = []

    # Single-qubit Z terms: hᵢ Zᵢ
    for i in range(n):
        if h[i] == 0.0:
            continue
        # Pauli string: identity on all qubits except qubit i which gets Z
        pauli = ["I"] * n
        pauli[i] = "Z"
        pauli_list.append(("".join(reversed(pauli)), h[i]))

    # Two-qubit ZZ terms: Jᵢⱼ ZᵢZⱼ
    for i in range(n):
        for j in range(i + 1, n):
            if J[i, j] == 0.0:
                continue
            pauli = ["I"] * n
            pauli[i] = "Z"
            pauli[j] = "Z"
            pauli_list.append(("".join(reversed(pauli)), J[i, j]))

    if not pauli_list:
        # Trivial zero Hamiltonian
        pauli_list = [("I" * n, 0.0)]

    return SparsePauliOp.from_list(pauli_list)


def ising_energy(z: np.ndarray, h: np.ndarray, J: np.ndarray) -> float:
    """
    Compute the Ising energy for a spin vector z ∈ {−1, +1}^n.

    E(z) = Σᵢ hᵢ zᵢ + Σᵢ<ⱼ Jᵢⱼ zᵢ zⱼ
    """
    energy = float(h @ z)
    for i in range(len(z)):
        for j in range(i + 1, len(z)):
            energy += J[i, j] * z[i] * z[j]
    return energy


def verify_hamiltonian(
    h: np.ndarray,
    J: np.ndarray,
    offset: float,
    Q: np.ndarray,
) -> bool:
    """
    Verify that the Ising energy + offset matches the QUBO energy for all 2^n states.

    Confirms:  E_ising(z) + offset  ==  xᵀQx   for all x ↔ z.

    Returns True if all states agree; raises AssertionError on mismatch.
    """
    from itertools import product as iproduct
    from quantum.qubo import evaluate_qubo

    n = len(h)
    max_err = 0.0

    for bits in iproduct([0, 1], repeat=n):
        x = np.array(bits, dtype=float)
        z = 1 - 2 * x  # xᵢ=0 → zᵢ=+1, xᵢ=1 → zᵢ=−1

        qubo_val = evaluate_qubo(Q, x)
        ising_val = ising_energy(z, h, J) + offset

        err = abs(qubo_val - ising_val)
        max_err = max(max_err, err)
        assert err < 1e-9, (
            f"Mismatch at x={bits}: QUBO={qubo_val:.6f}, Ising+offset={ising_val:.6f}, err={err:.2e}"
        )

    print(f"Ising verification passed for all 2^{n} states. Max error: {max_err:.2e}")
    return True


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from data.generate_data import generate_assets
    from quantum.qubo import build_qubo

    for n in [4, 6, 8]:
        mu, Sigma = generate_assets(n, seed=42)
        k = n // 2
        Q, _offset = build_qubo(mu, Sigma, lam=1.0, k=k)
        h, J, offset = qubo_to_ising(Q)

        print(f"\nn={n}, k={k}")
        print(f"h (local fields): {np.round(h, 4)}")
        print(f"J (couplings, upper-tri):\n{np.round(J, 4)}")
        print(f"offset: {offset:.4f}")

        verify_hamiltonian(h, J, offset, Q)

        H = build_ising_hamiltonian(h, J)
        print(f"SparsePauliOp terms: {len(H)}")
        print(H)
