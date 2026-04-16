"""
QAOA classical optimization loop and result decoder.

Workflow
--------
1. Build QAOA circuit (parameterized, no measurement) from Ising h, J.
2. Optimize parameters θ = [γ₁,...,γₚ, β₁,...,βₚ] to minimize ⟨ψ(θ)|HC|ψ(θ)⟩
   using Qiskit Aer statevector simulator (exact, no shot noise).
3. After convergence, sample the final statevector (shots) and decode the
   most probable bitstring as the portfolio selection.
4. Return structured results for benchmarking.
"""

import time
import numpy as np
from scipy.optimize import minimize

from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimator

from quantum.qaoa_circuit import build_qaoa_circuit, build_qaoa_circuit_no_measure
from quantum.hamiltonian import build_ising_hamiltonian, qubo_to_ising
from quantum.qubo import build_qubo, evaluate_qubo


# ---------------------------------------------------------------------------
# Expectation value via statevector
# ---------------------------------------------------------------------------

def _expectation_statevector(
    qc_no_meas,
    params: np.ndarray,
    hamiltonian: SparsePauliOp,
) -> float:
    """
    Bind parameters and compute ⟨ψ|H|ψ⟩ exactly via statevector.

    Parameters are bound in the order: gamma[0..p-1], beta[0..p-1]
    (matching ParameterVector ordering in qaoa_circuit.py).
    """
    param_dict = dict(zip(sorted(qc_no_meas.parameters, key=lambda p: p.name), params))
    bound_qc = qc_no_meas.assign_parameters(param_dict)

    sv = Statevector(bound_qc)
    # expectation value: ⟨ψ|H|ψ⟩ (real part; imaginary should be ~0)
    exp_val = sv.expectation_value(hamiltonian).real
    return float(exp_val)


# ---------------------------------------------------------------------------
# Main QAOA runner
# ---------------------------------------------------------------------------

def run_qaoa(
    mu: np.ndarray,
    Sigma: np.ndarray,
    lam: float = 1.0,
    k: int | None = None,
    penalty: float | None = None,
    p: int = 1,
    shots: int = 2000,
    n_restarts: int = 3,
    seed: int = 0,
    optimizer: str = "COBYLA",
    max_iter: int = 1000,
) -> dict:
    """
    Run QAOA for portfolio optimization.

    Parameters
    ----------
    mu         : (n,) expected returns
    Sigma      : (n, n) covariance matrix
    lam        : risk-aversion parameter λ
    k          : cardinality constraint (select exactly k assets)
    penalty    : QUBO penalty coefficient (auto-set if None)
    p          : QAOA depth
    shots      : number of samples from the final statevector
    n_restarts : number of random parameter initializations (best is kept)
    seed       : random seed for reproducibility
    optimizer  : scipy optimizer name ('COBYLA' or 'BFGS')
    max_iter   : maximum optimizer iterations per restart

    Returns
    -------
    dict with keys:
        bitstring      : best binary string (str, length n)
        x              : best binary vector (np.ndarray)
        qubo_energy    : QUBO objective xᵀQx (without offset) for best bitstring
        portfolio_return  : μᵀx
        portfolio_variance: xᵀΣx
        opt_energy     : best ⟨HC⟩ achieved (Ising energy, no offset)
        n_iters        : total optimizer iterations across all restarts
        runtime_s      : wall-clock seconds
        p              : circuit depth used
        params         : optimized parameters [γ₁,..,γₚ, β₁,..,βₚ]
        counts         : {bitstring: count} from final sampling
    """
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    n = len(mu)

    # --- Build QUBO and Ising Hamiltonian ---
    Q, qubo_offset = build_qubo(mu, Sigma, lam=lam, k=k, penalty=penalty)
    h_ising, J_ising, ising_offset = qubo_to_ising(Q)
    H_cost = build_ising_hamiltonian(h_ising, J_ising)

    # --- Build circuit (no measurement, for statevector optimization) ---
    qc = build_qaoa_circuit_no_measure(h_ising, J_ising, p=p)

    # Parameter ordering: ParameterVector sorts lexicographically
    # "β[0]" < "β[1]" < ... < "γ[0]" < "γ[1]" < ...
    # We need to map our theta array [γ₀,..,γₚ₋₁, β₀,..,βₚ₋₁] to the sorted order.
    sorted_params = sorted(qc.parameters, key=lambda param: param.name)

    def _cost(theta: np.ndarray) -> float:
        param_dict = dict(zip(sorted_params, theta))
        bound_qc = qc.assign_parameters(param_dict)
        sv = Statevector(bound_qc)
        return float(sv.expectation_value(H_cost).real)

    # --- Multi-restart optimization ---
    # theta layout: [β₀,..,βₚ₋₁, γ₀,..,γₚ₋₁]  (sorted_params order)
    # Initial ranges: γ ∈ [0, π], β ∈ [0, π/2]
    best_energy = np.inf
    best_params = None
    total_iters = 0

    for _ in range(n_restarts):
        # Random init in [0, 2π] for all parameters
        theta0 = rng.uniform(0, 2 * np.pi, size=2 * p)

        if optimizer == "COBYLA":
            result = minimize(
                _cost,
                theta0,
                method="COBYLA",
                options={"maxiter": max_iter, "rhobeg": 0.5},
            )
        else:
            result = minimize(
                _cost,
                theta0,
                method=optimizer,
                options={"maxiter": max_iter},
            )

        total_iters += result.nfev
        if result.fun < best_energy:
            best_energy = result.fun
            best_params = result.x

    # --- Sample the optimized circuit ---
    qc_meas = build_qaoa_circuit(h_ising, J_ising, p=p)
    sorted_params_meas = sorted(qc_meas.parameters, key=lambda param: param.name)
    param_dict_meas = dict(zip(sorted_params_meas, best_params))
    bound_meas = qc_meas.assign_parameters(param_dict_meas)

    # Use statevector to get exact probabilities, then sample
    # (avoids AerSimulator shot-based transpilation complexity)
    qc_sv = build_qaoa_circuit_no_measure(h_ising, J_ising, p=p)
    sorted_params_sv = sorted(qc_sv.parameters, key=lambda param: param.name)
    param_dict_sv = dict(zip(sorted_params_sv, best_params))
    bound_sv = qc_sv.assign_parameters(param_dict_sv)
    sv_final = Statevector(bound_sv)

    probs = sv_final.probabilities()  # length 2^n, index = computational basis integer
    # Sample bitstrings
    basis_indices = rng.choice(len(probs), size=shots, p=probs)
    counts: dict[str, int] = {}
    for idx in basis_indices:
        bits = format(idx, f"0{n}b")  # big-endian bitstring
        counts[bits] = counts.get(bits, 0) + 1

    # Decode: pick the most frequent bitstring
    best_bitstring = max(counts, key=counts.__getitem__)
    x_best = np.array([int(b) for b in best_bitstring], dtype=float)

    # --- Compute portfolio metrics ---
    qubo_energy = evaluate_qubo(Q, x_best)
    port_return = float(mu @ x_best)
    port_variance = float(x_best @ Sigma @ x_best)

    runtime = time.perf_counter() - t0

    return {
        "bitstring": best_bitstring,
        "x": x_best,
        "qubo_energy": qubo_energy,
        "portfolio_return": port_return,
        "portfolio_variance": port_variance,
        "opt_energy": best_energy,
        "n_iters": total_iters,
        "runtime_s": runtime,
        "p": p,
        "params": best_params,
        "counts": counts,
    }


# ---------------------------------------------------------------------------
# Depth sweep helper
# ---------------------------------------------------------------------------

def depth_sweep(
    mu: np.ndarray,
    Sigma: np.ndarray,
    optimal_qubo_energy: float,
    p_values: list[int] = [1, 2, 3],
    **kwargs,
) -> list[dict]:
    """
    Run QAOA for multiple depths and compute approximation ratios.

    Parameters
    ----------
    mu, Sigma            : asset data
    optimal_qubo_energy  : brute-force optimal QUBO energy (ground truth)
    p_values             : list of QAOA depths to sweep
    **kwargs             : forwarded to run_qaoa (lam, k, penalty, shots, ...)

    Returns
    -------
    List of result dicts (one per depth), each augmented with:
        approx_ratio : qubo_energy / optimal_qubo_energy  (1.0 = optimal)
    """
    results = []
    for p in p_values:
        print(f"  Running QAOA p={p} ...", flush=True)
        res = run_qaoa(mu, Sigma, p=p, **kwargs)
        # Approximation ratio: for minimization, ratio = optimal / achieved
        # (values are negative for good portfolios, so use careful sign handling)
        # Approximation ratio for minimization: QAOA / optimal.
        # Both energies are typically negative (lower = better), so ratio ≤ 1
        # with 1.0 meaning QAOA matched the optimal solution exactly.
        if abs(optimal_qubo_energy) > 1e-12:
            res["approx_ratio"] = res["qubo_energy"] / optimal_qubo_energy
        else:
            res["approx_ratio"] = float("nan")
        results.append(res)
        print(
            f"    p={p}: energy={res['qubo_energy']:.4f}, "
            f"approx_ratio={res.get('approx_ratio', float('nan')):.4f}, "
            f"iters={res['n_iters']}, time={res['runtime_s']:.1f}s"
        )
    return results


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from data.generate_data import generate_assets
    from classical.brute_force import brute_force
    from quantum.qubo import build_qubo, evaluate_qubo

    n, k = 4, 2
    lam = 1.0
    mu, Sigma = generate_assets(n, seed=42)

    # Build QUBO to get the optimal QUBO energy (used for approximation ratio)
    Q, _ = build_qubo(mu, Sigma, lam=lam, k=k)

    print(f"Running brute-force for n={n}, k={k} ...")
    bf = brute_force(mu, Sigma, lam=lam, k=k)
    bf_bits = "".join(str(int(b)) for b in bf.bitstring)
    bf_qubo_energy = evaluate_qubo(Q, bf.bitstring)
    print(f"  Optimal bitstring: {bf_bits}")
    print(f"  Optimal objective (no penalty): {bf.objective:.4f}")
    print(f"  Optimal QUBO energy (with penalty): {bf_qubo_energy:.4f}")

    print(f"\nRunning QAOA depth sweep (p=1,2,3) for n={n}, k={k} ...")
    results = depth_sweep(
        mu, Sigma,
        optimal_qubo_energy=bf_qubo_energy,
        p_values=[1, 2, 3],
        lam=lam,
        k=k,
        shots=2000,
        n_restarts=3,
        seed=42,
    )

    print("\n--- Summary ---")
    print(f"{'p':>3}  {'bitstring':>10}  {'energy':>10}  {'approx_ratio':>13}  {'time(s)':>8}")
    for r in results:
        print(
            f"{r['p']:>3}  {r['bitstring']:>10}  {r['qubo_energy']:>10.4f}"
            f"  {r['approx_ratio']:>13.4f}  {r['runtime_s']:>8.2f}"
        )
    print(f"\nBrute-force: {bf_bits}  QUBO energy={bf_qubo_energy:.4f}")
