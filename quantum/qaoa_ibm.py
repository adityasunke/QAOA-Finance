"""
Run QAOA portfolio optimization on IBM Quantum hardware.

Strategy
--------
1. Optimize QAOA parameters on the Aer statevector simulator (fast, noiseless).
2. Bind the optimal parameters and transpile the circuit for the target IBM backend.
3. Execute on IBM hardware via qiskit-ibm-runtime SamplerV2.
4. Decode results and compare against the brute-force optimum.

Setup
-----
    pip install qiskit-ibm-runtime python-dotenv

    Add your token to the .env file in the project root:
    IBM_QUANTUM_TOKEN='<your token from quantum.ibm.com>'

Usage
-----
    python quantum/qaoa_ibm.py
    python quantum/qaoa_ibm.py --backend ibm_kyoto --p 1 --shots 4000
    python quantum/qaoa_ibm.py --backend least_busy --p 2 --shots 8000
"""

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent.parent))

from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from quantum.qubo import build_qubo
from quantum.hamiltonian import qubo_to_ising, build_ising_hamiltonian
from quantum.qaoa_circuit import build_qaoa_circuit
from classical.brute_force import brute_force, objective as portfolio_obj
from data.generate_data import load_assets, DEFAULT_TICKERS


# ---------------------------------------------------------------------------
# Parameter optimization on the Aer statevector simulator
# ---------------------------------------------------------------------------

def optimize_on_simulator(
    h: np.ndarray,
    J: np.ndarray,
    ising_offset: float,
    p: int,
    seed: int = 42,
) -> tuple[np.ndarray, float, int]:
    """
    Minimize ⟨H_C⟩ + ising_offset using COBYLA on the Aer statevector simulator.

    Returns optimal parameters, final expectation value, and iteration count.
    """
    H_C = build_ising_hamiltonian(h, J)
    qc, gammas, betas = build_qaoa_circuit(h, J, p=p)
    all_params = list(gammas) + list(betas)

    sim = AerSimulator(method='statevector')
    qc_sv = qc.copy()
    qc_sv.save_statevector()
    qc_sv_t = transpile(qc_sv, sim)

    iter_count = [0]

    def cost(params: np.ndarray) -> float:
        iter_count[0] += 1
        param_dict = {p_obj: float(v) for p_obj, v in zip(all_params, params)}
        bound = qc_sv_t.assign_parameters(param_dict)
        sv = sim.run(bound).result().get_statevector()
        return float(Statevector(sv).expectation_value(H_C).real) + ising_offset

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0.0, np.pi, 2 * p)
    opt = minimize(cost, x0, method='COBYLA', options={'maxiter': 1000, 'rhobeg': 0.5})

    return opt.x, float(opt.fun), iter_count[0]


# ---------------------------------------------------------------------------
# Decode measurement counts to the best valid bitstring
# ---------------------------------------------------------------------------

def decode_counts(counts: dict, k: int | None) -> np.ndarray:
    """
    Return the most-frequent bitstring satisfying cardinality k.
    Falls back to the global most-frequent if none satisfy k.

    Qiskit count keys are big-endian strings: index 0 = qubit n-1.
    We reverse to get little-endian (qubit 0 = index 0).
    """
    for bitstr, _ in sorted(counts.items(), key=lambda kv: -kv[1]):
        x = np.array([int(b) for b in reversed(bitstr)], dtype=float)
        if k is None or int(x.sum()) == k:
            return x
    top = max(counts, key=counts.get)
    return np.array([int(b) for b in reversed(top)], dtype=float)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="QAOA portfolio optimization on IBM hardware")
    parser.add_argument("--backend", default="least_busy",
                        help="IBM backend name, or 'least_busy' to auto-select (default: least_busy)")
    parser.add_argument("--p", type=int, default=1,
                        help="QAOA circuit depth (default: 1)")
    parser.add_argument("--shots", type=int, default=4000,
                        help="Measurement shots on hardware (default: 4000)")
    parser.add_argument("--n-assets", type=int, default=4,
                        help="Number of assets / qubits (default: 4)")
    parser.add_argument("--k", type=int, default=2,
                        help="Number of assets to select (default: 2)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # --- Load problem data -------------------------------------------------
    tickers = DEFAULT_TICKERS[: args.n_assets]
    mu, Sigma = load_assets(tickers)
    k = args.k

    print(f"Assets: {tickers}")
    print(f"Selecting k={k} of n={args.n_assets} assets, p={args.p}, shots={args.shots}")
    print()

    # --- Brute-force ground truth ------------------------------------------
    bf = brute_force(mu, Sigma, lam=1.0, k=k)
    print(f"Brute-force optimum: bits={bf.bitstring.astype(int).tolist()}  obj={bf.objective:.4f}")
    print()

    # --- Build QUBO → Ising -----------------------------------------------
    Q, _ = build_qubo(mu, Sigma, lam=1.0, k=k)
    h, J, ising_offset = qubo_to_ising(Q)

    # --- Phase 1: optimize parameters on Aer simulator --------------------
    print("Phase 1: optimizing parameters on Aer simulator...")
    t0 = time.perf_counter()
    opt_params, sim_expectation, n_iters = optimize_on_simulator(
        h, J, ising_offset, p=args.p, seed=args.seed
    )
    sim_time = time.perf_counter() - t0
    print(f"  Converged in {n_iters} iterations ({sim_time:.1f}s)")
    print(f"  Optimal parameters: {np.round(opt_params, 4).tolist()}")
    print(f"  Simulator expectation ⟨H_C⟩: {sim_expectation:.4f}")
    print()

    # --- Build and bind the final circuit ---------------------------------
    qc, gammas, betas = build_qaoa_circuit(h, J, p=args.p)
    all_params = list(gammas) + list(betas)
    param_dict = {p_obj: float(v) for p_obj, v in zip(all_params, opt_params)}
    qc_bound = qc.assign_parameters(param_dict)
    qc_bound.measure_all()

    # --- Phase 2: run on IBM hardware -------------------------------------
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    if not token:
        print("ERROR: IBM_QUANTUM_TOKEN environment variable not set.")
        print("  Get your token at https://quantum.ibm.com and run:")
        print("  export IBM_QUANTUM_TOKEN='<your token>'")
        sys.exit(1)

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    except ImportError:
        print("ERROR: qiskit-ibm-runtime not installed.")
        print("  pip install qiskit-ibm-runtime")
        sys.exit(1)

    print("Phase 2: connecting to IBM Quantum...")
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)

    if args.backend == "least_busy":
        backend = service.least_busy(
            operational=True,
            min_num_qubits=args.n_assets,
            simulator=False,
        )
        print(f"  Selected backend: {backend.name}")
    else:
        backend = service.backend(args.backend)
        print(f"  Using backend: {backend.name}")

    print(f"  Backend status: {backend.status().status_msg}")
    print(f"  Queue depth: {backend.status().pending_jobs} pending jobs")
    print()

    # Transpile for the target backend at highest optimization level
    print("  Transpiling circuit for hardware...")
    qc_t = transpile(qc_bound, backend=backend, optimization_level=3)
    print(f"  Transpiled depth: {qc_t.depth()}  (original: {qc_bound.depth()})")
    print(f"  Transpiled gate count: {qc_t.size()}")
    print()

    print(f"  Submitting job ({args.shots} shots)...")
    t0 = time.perf_counter()
    sampler = Sampler(backend)
    job = sampler.run([qc_t], shots=args.shots)
    print(f"  Job ID: {job.job_id()}")
    print("  Waiting for results (this may take several minutes)...")

    result = job.result()
    hw_time = time.perf_counter() - t0
    print(f"  Job completed in {hw_time:.1f}s")
    print()

    # --- Decode results ---------------------------------------------------
    counts = result[0].data.meas.get_counts()
    best_bits = decode_counts(counts, k)
    obj = portfolio_obj(best_bits, mu, Sigma, lam=1.0)
    approx_ratio = obj / bf.objective if bf.objective != 0.0 else float("inf")
    match = best_bits.astype(int).tolist() == bf.bitstring.astype(int).tolist()

    # --- Report -----------------------------------------------------------
    print("=" * 50)
    print("Results")
    print("=" * 50)
    print(f"Backend:          {backend.name}")
    print(f"QAOA depth p:     {args.p}")
    print(f"Shots:            {args.shots}")
    print()
    print(f"Brute-force:      bits={bf.bitstring.astype(int).tolist()}  obj={bf.objective:.4f}")
    print(f"IBM hardware:     bits={best_bits.astype(int).tolist()}  obj={obj:.4f}  {'✓ optimal' if match else '✗'}")
    print(f"Approx ratio:     {approx_ratio:.4f}  (1.0 = optimal)")
    print()

    # Top-5 most measured bitstrings
    print("Top-5 measured bitstrings:")
    for bs, cnt in sorted(counts.items(), key=lambda kv: -kv[1])[:5]:
        x = np.array([int(b) for b in reversed(bs)], dtype=float)
        card = int(x.sum())
        o = portfolio_obj(x, mu, Sigma, lam=1.0)
        freq = cnt / args.shots * 100
        print(f"  {bs}  count={cnt:5d}  ({freq:5.1f}%)  k={card}  obj={o:.4f}")


if __name__ == "__main__":
    main()
