# QAOA Portfolio Optimization

Applying the Quantum Approximate Optimization Algorithm (QAOA) to binary portfolio selection using real stock data from the "Magnificent 7" (AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA). Benchmarked against classical solvers (brute-force, greedy, simulated annealing) across problem sizes n ∈ {4, 6} and QAOA circuit depths p ∈ {1, 2, 3}.

---

## Problem

Given n assets with expected return vector μ and covariance matrix Σ, select exactly k assets to minimize the risk-adjusted objective:

```
f(x) = xᵀΣx − λ·μᵀx,   x ∈ {0,1}ⁿ,  Σᵢ xᵢ = k
```

This binary portfolio selection problem is NP-hard in general and maps naturally to QUBO/Ising form for quantum solvers.

---

## Results

### Approximation Ratio (f(x) / f\*, lower = better, 1.0 = optimal)

| Solver | n=4, k=2 | n=6, k=3 |
|---|---|---|
| Brute Force | **1.000** | **1.000** |
| Greedy | **1.000** | **1.000** |
| Simulated Annealing | **1.000** | **1.000** |
| QAOA p=1 | **1.000** | **1.000** |
| QAOA p=2 | 1.681 | **1.000** |
| QAOA p=3 | 1.143 | 1.459 |

QAOA at depth p=1 achieves the **optimal portfolio** for both problem sizes. Deeper circuits (p=2, 3) degrade — suggesting barren plateaus or poor parameter initialization at higher depth.

### QAOA Success Probability vs Depth

| p | n=4 | n=6 |
|---|---|---|
| 1 | 19.4% | 5.6% |
| 2 | 0.5% | 7.5% |
| 3 | 9.6% | 1.0% |

### Selected Portfolios (optimal)

| n | Selected Stocks | Return | Variance |
|---|---|---|---|
| 4 (k=2) | AAPL, MSFT | −0.474 | 0.177 |
| 6 (k=3) | AAPL, MSFT, GOOGL | −0.653 | 0.221 |

Negative returns reflect the risk-minimization dominance (λ=1.0) over the 100-day window used.

---

## Project Structure

```
QAOA-Finance/
├── data/
│   ├── *.csv                  # 100-day price histories for 7 tickers
│   ├── fetch_stock_data.py    # Alpha Vantage API data fetcher
│   └── generate_data.py       # Computes μ, Σ from CSVs
│
├── classical/
│   ├── brute_force.py         # Exact 2ⁿ enumeration (ground truth)
│   ├── heuristics.py          # Greedy and simulated annealing
│   └── run_classical.py       # Driver for classical solvers
│
├── quantum/
│   ├── qubo.py                # QUBO matrix construction + verification
│   ├── hamiltonian.py         # QUBO → Ising Hamiltonian (SparsePauliOp)
│   ├── qaoa_circuit.py        # Parameterized QAOA circuit builder
│   ├── qaoa_runner.py         # QAOA optimization loop (COBYLA + Aer)
│   └── qaoa_ibm.py            # IBM Quantum hardware execution
│
└── benchmarks/
    ├── run_experiments.py     # Full sweep over (n, p) combinations
    ├── metrics.py             # Approx ratio, success prob, return, variance
    ├── plot_results.py        # Generates 7 publication figures
    ├── results/
    │   ├── results.csv
    │   └── results.json
    └── figures/               # PNG visualizations (fig0–fig6)
```

---

## Quantum Pipeline

```
Stock CSVs → μ, Σ → QUBO matrix Q → Ising Hamiltonian H_C
                                              ↓
                              Parameterized QAOA circuit (p layers)
                                              ↓
                              COBYLA optimizer minimizes ⟨H_C⟩
                                              ↓
                              Sample optimized circuit (4000 shots)
                                              ↓
                              Decode most-frequent valid k-bitstring
```

**QAOA circuit structure (per layer):**
1. Hadamard on all n qubits → uniform superposition |+⟩ⁿ
2. Cost layer: `RZ(2γhᵢ)` + `RZZ(2γJᵢⱼ)` gates encoding H_C
3. Mixer layer: `RX(2β)` on each qubit

Parameters: 2p (γ₁…γₚ for cost, β₁…βₚ for mixer), optimized with COBYLA.

---

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies: `qiskit==2.3.1`, `qiskit-aer==0.17.2`, `qiskit-ibm-runtime==0.46.1`, `numpy`, `scipy`, `pandas`, `matplotlib`.

---

## Running

### Classical baselines
```bash
python classical/run_classical.py
```

### QAOA on simulator (n=4, p=1,2,3)
```bash
python quantum/qaoa_runner.py
```

### Full benchmark sweep
```bash
python benchmarks/run_experiments.py
python benchmarks/plot_results.py     # generates figures/
```

### IBM Quantum hardware
```bash
export IBM_QUANTUM_TOKEN='<token from quantum.ibm.com>'
python quantum/qaoa_ibm.py --backend ibm_kyoto --p 1 --shots 4000
```

---

## Configuration

Key parameters in `benchmarks/run_experiments.py`:

| Parameter | Value | Description |
|---|---|---|
| `N_ASSETS` | [4, 6] | Portfolio sizes (capped at 6 — 7 tickers available) |
| `P_VALUES` | [1, 2, 3] | QAOA circuit depths |
| `k` | n/2 | Assets to select |
| `λ` (lam) | 1.0 | Risk-aversion weight |
| `QAOA_SHOTS` | 4000 | Measurement shots |
| `QAOA_MAXITER` | 300 | COBYLA max iterations |

---

## Figures

| Figure | Description |
|---|---|
| `fig0_combined_summary.png` | 3×2 grid of all metrics |
| `fig1_bitstring_selection.png` | Heatmap of asset selection per solver |
| `fig2_approx_ratio.png` | f(x)/f\* grouped bars |
| `fig3_portfolio_return.png` | μᵀx grouped bars |
| `fig4_portfolio_variance.png` | xᵀΣx grouped bars |
| `fig5_runtime.png` | Wall-clock time (log scale) |
| `fig6_success_probability.png` | QAOA success probability vs depth p |
