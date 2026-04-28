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
│   ├── AAPL.csv … NVDA.csv    # 100-day OHLCV price histories (7 tickers)
│   ├── fetch_stock_data.py    # Alpha Vantage API data fetcher
│   ├── generate_data.py       # Computes annualized μ, Σ from CSVs
│   └── unit_tests/
│       ├── test_generate_data.py   # load_assets: shapes, PSD, annualization
│       └── test_fetch_stock_data.py # fetch_daily: mocked HTTP, parsing
│
├── classical/
│   ├── brute_force.py         # Exact 2ⁿ enumeration (ground truth)
│   ├── heuristics.py          # Greedy and simulated annealing
│   ├── run_classical.py       # Driver — runs all solvers, saves JSON
│   └── unit_tests/
│       ├── test_brute_force.py     # objective, brute_force, PortfolioResult
│       ├── test_heuristics.py      # greedy, simulated_annealing
│       └── test_run_classical.py   # result_to_dict
│
├── quantum/
│   ├── qubo.py                # QUBO matrix construction + verification
│   ├── hamiltonian.py         # QUBO → Ising Hamiltonian (SparsePauliOp)
│   ├── qaoa_circuit.py        # Parameterized QAOA circuit builder
│   ├── qaoa_runner.py         # QAOA optimization loop (COBYLA + Aer)
│   ├── qaoa_ibm.py            # IBM Quantum hardware execution
│   └── unit_tests/
│       ├── test_qubo.py            # build_qubo, evaluate_qubo, verify_qubo
│       ├── test_hamiltonian.py     # qubo_to_ising, build_ising_hamiltonian, ising_energy
│       ├── test_qaoa_circuit.py    # build_qaoa_circuit: structure, gates, binding
│       ├── test_qaoa_runner.py     # QAOAResult, run_qaoa (Aer simulator)
│       └── test_qaoa_ibm.py        # decode_counts, optimize_on_simulator
│
├── benchmarks/
│   ├── run_experiments.py     # Full sweep over (n, p) combinations
│   ├── metrics.py             # Approx ratio, success prob, return, variance
│   ├── plot_results.py        # Generates 7 publication figures
│   ├── unit_tests/
│   │   ├── test_metrics.py         # All 6 metrics functions (pure math)
│   │   ├── test_plot_results.py    # load(), _grouped_bars(), plot_* smoke tests
│   │   └── test_run_experiments.py # _row(), CSV_COLUMNS, K_BY_N constants
│   ├── results/
│   │   ├── results.csv
│   │   └── results.json
│   └── figures/               # PNG visualizations (fig0–fig6)
│
└── notebooks/
    └── analysis.ipynb         # Post-hoc analysis and supplementary plots
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

## Testing

Each module has a `unit_tests/` subfolder. Tests are discovered automatically by `pytest` from the project root.

### Run all tests (305 tests across all modules)
```bash
python -m pytest classical/unit_tests/ data/unit_tests/ quantum/unit_tests/ benchmarks/unit_tests/ -v
```

### Run one module at a time
```bash
python -m pytest classical/unit_tests/ -v    # 65 tests — brute force, greedy, SA
python -m pytest data/unit_tests/ -v         # 38 tests — data loading, API fetch
python -m pytest quantum/unit_tests/ -v      # 120 tests — QUBO, Hamiltonian, QAOA circuit/runner
python -m pytest benchmarks/unit_tests/ -v   # 82 tests — metrics, plotting, experiment row
```

### Run a specific file or test
```bash
python -m pytest quantum/unit_tests/test_hamiltonian.py -v
python -m pytest classical/unit_tests/test_brute_force.py::TestBruteForce::test_known_optimal_k1 -v
```

### Useful flags
| Flag | Effect |
|---|---|
| `-v` | Verbose — show each test name |
| `-x` | Stop on first failure |
| `--tb=short` | Shorter traceback on failures |
| `-k "cardinality"` | Run only tests whose name contains "cardinality" |

> **Note:** Tests in `quantum/unit_tests/` spin up the Qiskit Aer simulator and take ~5–10 s. All other tests complete in under 1 s.

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
python quantum/qaoa_ibm.py --backend ibm_fez --p 1 --shots 4000
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
