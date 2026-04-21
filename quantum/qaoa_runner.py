"""
QAOA optimization loop for binary portfolio optimization.

Uses Qiskit Aer statevector simulator for exact simulation and COBYLA
for gradient-free classical parameter optimization.
"""

import sys
import time
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent.parent))

from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from quantum.qubo import build_qubo
from quantum.hamiltonian import qubo_to_ising, build_ising_hamiltonian
from quantum.qaoa_circuit import build_qaoa_circuit
from classical.brute_force import objective as portfolio_obj


@dataclass
class QAOAResult:
    bitstring: np.ndarray       # decoded binary selection vector (length n)
    objective: float            # f(x) = xᵀΣx − λμᵀx (portfolio obj, no penalty)
    ret: float                  # portfolio return: μᵀx
    variance: float             # portfolio variance: xᵀΣx
    expectation: float          # final QAOA cost expectation value ⟨H_C⟩ + offset
    n_iters: int                # number of COBYLA evaluations
    p: int                      # circuit depth used
    runtime: float              # wall-clock seconds
    counts: dict = field(default_factory=dict)  # measurement outcome counts


def run_qaoa(
    mu: np.ndarray,
    Sigma: np.ndarray,
    lam: float = 1.0,
    k: int | None = None,
    p: int = 1,
    shots: int = 4000,
    seed: int = 0,
) -> QAOAResult:
    """
    Run QAOA for portfolio optimization.

    Steps:
      1. Build QUBO → Ising Hamiltonian → QAOA circuit.
      2. Minimize ⟨H_C⟩ + ising_offset (= ⟨xᵀQx⟩) via COBYLA on the
         Aer statevector simulator.
      3. Sample the optimized state (4000 shots) and decode the most
         frequent bitstring satisfying the cardinality constraint.

    Parameters
    ----------
    mu    : (n,) annualized mean log-return vector
    Sigma : (n, n) annualized covariance matrix
    lam   : risk-aversion parameter λ
    k     : cardinality (number of assets to select); None = unconstrained
    p     : QAOA circuit depth
    shots : measurement shots for final sampling
    seed  : random seed for parameter initialization

    Returns
    -------
    QAOAResult with decoded portfolio and optimization metadata
    """
    t_start = time.perf_counter()

    # Build QUBO → Ising → circuit
    Q, _ = build_qubo(mu, Sigma, lam=lam, k=k)
    h, J, ising_offset = qubo_to_ising(Q)
    H_C = build_ising_hamiltonian(h, J)
    qc, gammas, betas = build_qaoa_circuit(h, J, p=p)

    # Parameters in a fixed order: γ₀,...,γₚ₋₁, β₀,...,βₚ₋₁
    all_params = list(gammas) + list(betas)

    # Aer statevector simulator
    sim = AerSimulator(method='statevector')

    # Compile circuit once (without measurements) for expectation evaluation
    qc_sv = qc.copy()
    qc_sv.save_statevector()
    qc_sv_t = transpile(qc_sv, sim)

    iter_count = [0]

    def cost(params: np.ndarray) -> float:
        iter_count[0] += 1
        param_dict = {p_obj: float(v) for p_obj, v in zip(all_params, params)}
        bound = qc_sv_t.assign_parameters(param_dict)
        sv = sim.run(bound).result().get_statevector()
        # ⟨H_C⟩ + ising_offset  ≡  ⟨xᵀQx⟩  (constant qubo_offset excluded)
        return float(Statevector(sv).expectation_value(H_C).real) + ising_offset

    # Random parameter initialization in [0, π]
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0.0, np.pi, 2 * p)

    opt = minimize(cost, x0, method='COBYLA', options={'maxiter': 1000, 'rhobeg': 0.5})
    final_expectation = float(opt.fun)

    # Sample the optimized state
    qc_meas = qc.copy()
    qc_meas.measure_all()
    qc_meas_t = transpile(qc_meas, sim)
    param_dict = {p_obj: float(v) for p_obj, v in zip(all_params, opt.x)}
    bound_meas = qc_meas_t.assign_parameters(param_dict)
    counts = sim.run(bound_meas, shots=shots).result().get_counts()

    # Decode: most frequent bitstring satisfying cardinality k
    # Qiskit count strings are big-endian: qubit n-1 is leftmost character
    best_bits = None
    for bitstr, _ in sorted(counts.items(), key=lambda kv: -kv[1]):
        x = np.array([int(b) for b in reversed(bitstr)], dtype=float)
        if k is None or int(x.sum()) == k:
            best_bits = x
            break

    # Fallback: no valid bitstring in counts — use most frequent overall
    if best_bits is None:
        top = max(counts, key=counts.get)
        best_bits = np.array([int(b) for b in reversed(top)], dtype=float)

    runtime = time.perf_counter() - t_start

    return QAOAResult(
        bitstring=best_bits,
        objective=portfolio_obj(best_bits, mu, Sigma, lam),
        ret=float(mu @ best_bits),
        variance=float(best_bits @ Sigma @ best_bits),
        expectation=final_expectation,
        n_iters=iter_count[0],
        p=p,
        runtime=runtime,
        counts=counts,
    )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from data.generate_data import load_assets, DEFAULT_TICKERS
    from classical.brute_force import brute_force

    tickers = DEFAULT_TICKERS[:4]   # n=4 for tractable simulation
    mu, Sigma = load_assets(tickers)
    k = 2

    # Ground truth
    bf = brute_force(mu, Sigma, lam=1.0, k=k)
    print(f"Brute-force: obj={bf.objective:.4f}  bits={bf.bitstring.astype(int).tolist()}")
    print()

    # Depth sweep p = 1, 2, 3
    for p in [1, 2, 3]:
        result = run_qaoa(mu, Sigma, lam=1.0, k=k, p=p, shots=4000, seed=42)
        ratio = result.objective / bf.objective if bf.objective != 0.0 else float('inf')
        match = result.bitstring.astype(int).tolist() == bf.bitstring.astype(int).tolist()
        print(f"QAOA p={p}:")
        print(f"  bits={result.bitstring.astype(int).tolist()}  {'✓ optimal' if match else '✗'}")
        print(f"  obj={result.objective:.4f}  approx_ratio={ratio:.4f}")
        print(f"  expectation={result.expectation:.4f}  iters={result.n_iters}  time={result.runtime:.1f}s")
        print()
