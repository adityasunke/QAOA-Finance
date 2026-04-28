#!/usr/bin/env python3
"""
Phase 4 experiment driver for QAOA portfolio optimization.

Sweeps n_assets ∈ {4, 6} × QAOA depth p ∈ {1, 2, 3}.  For each n, loads
real stock data from data/*.csv, runs brute-force, greedy, simulated
annealing, and QAOA (Qiskit Aer statevector simulator), then records all
metrics.  Classical solvers run once per n (they don't depend on p).

Results are saved to:
    benchmarks/results/results.csv
    benchmarks/results/results.json

Usage
-----
    python benchmarks/run_experiments.py            # full sweep
    python benchmarks/run_experiments.py --n 4      # single n
    python benchmarks/run_experiments.py --n 4 --p 1  # single (n, p)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.generate_data import load_assets, DEFAULT_TICKERS
from classical.brute_force import brute_force
from classical.heuristics import greedy, simulated_annealing
from quantum.qaoa_runner import run_qaoa
from benchmarks.metrics import (
    portfolio_return,
    portfolio_variance,
    approximation_ratio,
    success_probability,
    format_bitstring,
    selected_stocks,
)

# ── Sweep configuration ───────────────────────────────────────────────────────

DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Ticker subsets drawn from DEFAULT_TICKERS (AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA).
TICKERS_BY_N: dict[int, list[str]] = {
    4: DEFAULT_TICKERS[:4],   # AAPL, MSFT, GOOGL, AMZN
    6: DEFAULT_TICKERS[:6],   # AAPL, MSFT, GOOGL, AMZN, META, TSLA
}

K_BY_N: dict[int, int] = {4: 2, 6: 3}   # assets to select (half of n)

LAM: float = 1.0          # risk-aversion λ
P_VALUES: list[int] = [1, 2, 3]
QAOA_SHOTS: int = 4000
QAOA_MAXITER: int = 300   # COBYLA cap; 300 gives good quality with ~3–5× speedup
SA_SEED: int = 42

CSV_COLUMNS: list[str] = [
    "n", "tickers", "k", "lam", "p", "solver",
    "bitstring", "selected_stocks",
    "portfolio_return", "portfolio_variance", "objective",
    "approx_ratio", "success_prob", "runtime",
]

# ── Per-n runner ──────────────────────────────────────────────────────────────


def _row(
    n: int,
    tickers: list[str],
    k: int,
    p: int | None,
    solver: str,
    x: np.ndarray,
    obj: float,
    approx: float,
    succ: float | None,
    runtime: float,
    mu: np.ndarray,
    Sigma: np.ndarray,
) -> dict[str, Any]:
    return {
        "n": n,
        "tickers": ",".join(tickers),
        "k": k,
        "lam": LAM,
        "p": p,
        "solver": solver,
        "bitstring": format_bitstring(x),
        "selected_stocks": selected_stocks(x, tickers),
        "portfolio_return": portfolio_return(x, mu),
        "portfolio_variance": portfolio_variance(x, Sigma),
        "objective": obj,
        "approx_ratio": approx,
        "success_prob": succ,
        "runtime": runtime,
    }


def run_for_n(
    n: int,
    mu: np.ndarray,
    Sigma: np.ndarray,
    tickers: list[str],
    p_values: list[int],
) -> list[dict[str, Any]]:
    """
    Run all solvers for a given portfolio size and return result rows.

    Classical solvers (brute-force, greedy, SA) run once; QAOA runs once
    per depth p.
    """
    k = K_BY_N[n]
    rows: list[dict[str, Any]] = []

    print(f"\n── n={n}  k={k}  tickers={tickers} ──")

    # ── Brute force (ground truth; establishes C*) ────────────────────────
    t0 = time.perf_counter()
    bf = brute_force(mu, Sigma, lam=LAM, k=k)
    bf_rt = time.perf_counter() - t0
    c_star = bf.objective

    rows.append(_row(n, tickers, k, None, "brute_force",
                     bf.bitstring, bf.objective, 1.0, None, bf_rt, mu, Sigma))
    print(f"  brute_force : {format_bitstring(bf.bitstring)} "
          f"({selected_stocks(bf.bitstring, tickers)})  "
          f"obj={bf.objective:.4f}  rt={bf_rt:.3f}s")

    # ── Greedy ────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    gr = greedy(mu, Sigma, lam=LAM, k=k)
    gr_rt = time.perf_counter() - t0

    rows.append(_row(n, tickers, k, None, "greedy",
                     gr.bitstring, gr.objective,
                     approximation_ratio(gr.objective, c_star),
                     None, gr_rt, mu, Sigma))
    print(f"  greedy      : {format_bitstring(gr.bitstring)} "
          f"({selected_stocks(gr.bitstring, tickers)})  "
          f"obj={gr.objective:.4f}  approx={approximation_ratio(gr.objective, c_star):.4f}  "
          f"rt={gr_rt:.4f}s")

    # ── Simulated annealing ───────────────────────────────────────────────
    t0 = time.perf_counter()
    sa = simulated_annealing(mu, Sigma, lam=LAM, k=k, seed=SA_SEED)
    sa_rt = time.perf_counter() - t0

    rows.append(_row(n, tickers, k, None, "simulated_annealing",
                     sa.bitstring, sa.objective,
                     approximation_ratio(sa.objective, c_star),
                     None, sa_rt, mu, Sigma))
    print(f"  sim_anneal  : {format_bitstring(sa.bitstring)} "
          f"({selected_stocks(sa.bitstring, tickers)})  "
          f"obj={sa.objective:.4f}  approx={approximation_ratio(sa.objective, c_star):.4f}  "
          f"rt={sa_rt:.4f}s")

    # ── QAOA (one run per depth p) ────────────────────────────────────────
    for p in p_values:
        qaoa = run_qaoa(
            mu, Sigma, lam=LAM, k=k, p=p,
            shots=QAOA_SHOTS, seed=SA_SEED, maxiter=QAOA_MAXITER,
        )
        succ = success_probability(qaoa.counts, bf.bitstring)
        approx = approximation_ratio(qaoa.objective, c_star)

        rows.append(_row(n, tickers, k, p, "qaoa_aer",
                         qaoa.bitstring, qaoa.objective, approx, succ,
                         qaoa.runtime, mu, Sigma))
        print(f"  qaoa p={p}     : {format_bitstring(qaoa.bitstring)} "
              f"({selected_stocks(qaoa.bitstring, tickers)})  "
              f"obj={qaoa.objective:.4f}  approx={approx:.4f}  "
              f"succ={succ:.3f}  iters={qaoa.n_iters}  rt={qaoa.runtime:.1f}s")

    return rows


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 4 QAOA portfolio benchmark"
    )
    parser.add_argument(
        "--n", type=int, choices=[4, 6],
        help="Run only this portfolio size (default: both 4 and 6)",
    )
    parser.add_argument(
        "--p", type=int, choices=[1, 2, 3],
        help="Run only this QAOA depth (default: 1, 2, 3)",
    )
    args = parser.parse_args()

    n_values = [args.n] if args.n else [4, 6]
    p_values = [args.p] if args.p else P_VALUES

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load the 6 tickers needed for n=6 (n=4 is a subset).
    all_tickers = DEFAULT_TICKERS[:6]
    print(f"Loading data for {all_tickers} from {DATA_DIR} ...")
    mu_all, Sigma_all = load_assets(all_tickers, DATA_DIR)
    idx_map = {t: i for i, t in enumerate(all_tickers)}

    all_rows: list[dict[str, Any]] = []

    for n in n_values:
        tickers = TICKERS_BY_N[n]
        idx = [idx_map[t] for t in tickers]
        mu = mu_all[idx]
        Sigma = Sigma_all[np.ix_(idx, idx)]
        all_rows.extend(run_for_n(n, mu, Sigma, tickers, p_values))

    # ── Save results ──────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows, columns=CSV_COLUMNS)

    csv_path = RESULTS_DIR / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} rows → {csv_path}")

    json_path = RESULTS_DIR / "results.json"
    json_path.write_text(
        json.dumps(all_rows, indent=2, default=str), encoding="utf-8"
    )
    print(f"Saved → {json_path}")

    # ── Print summary table ───────────────────────────────────────────────
    print("\n── Results summary ──────────────────────────────────────────────")
    summary_cols = ["n", "p", "solver", "bitstring", "selected_stocks",
                    "approx_ratio", "success_prob", "runtime"]
    print(df[summary_cols].to_string(index=False))


if __name__ == "__main__":
    main()
