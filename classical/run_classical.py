"""
Run brute-force, greedy, and simulated annealing portfolio optimizers
on 100 days of real Magnificent 7 stock data and save results to classical/.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from classical.brute_force import brute_force
from classical.heuristics import greedy, simulated_annealing
from data.generate_data import load_assets

DATA_DIR = ROOT / "data"
OUT_DIR = Path(__file__).parent
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"]
LAM = 1.0
K = 4   # select 4 of 7 assets


def result_to_dict(result, tickers, solver_name: str) -> dict:
    selected = [tickers[i] for i, b in enumerate(result.bitstring) if b == 1.0]
    return {
        "solver": solver_name,
        "selected_assets": selected,
        "bitstring": result.bitstring.astype(int).tolist(),
        "objective": round(result.objective, 6),
        "annualised_return": round(result.ret, 6),
        "annualised_variance": round(result.variance, 6),
    }


def main():
    print("Loading real stock data (100 trading days)...")
    mu, Sigma = load_assets(TICKERS, DATA_DIR)
    print(f"  {len(TICKERS)} assets loaded\n")

    print(f"Running solvers (n={len(TICKERS)}, k={K}, λ={LAM})...\n")

    # ── Brute Force ───────────────────────────────────────────────────────────
    print("  Brute force...")
    bf_result = brute_force(mu, Sigma, lam=LAM, k=K)
    bf_dict = result_to_dict(bf_result, TICKERS, "brute_force")
    print(f"    selected: {bf_dict['selected_assets']}")
    print(f"    obj={bf_dict['objective']:.4f}  ret={bf_dict['annualised_return']:.4f}  var={bf_dict['annualised_variance']:.4f}")

    # ── Greedy ────────────────────────────────────────────────────────────────
    print("  Greedy...")
    gr_result = greedy(mu, Sigma, lam=LAM, k=K)
    gr_dict = result_to_dict(gr_result, TICKERS, "greedy")
    print(f"    selected: {gr_dict['selected_assets']}")
    print(f"    obj={gr_dict['objective']:.4f}  ret={gr_dict['annualised_return']:.4f}  var={gr_dict['annualised_variance']:.4f}")

    # ── Simulated Annealing ───────────────────────────────────────────────────
    print("  Simulated annealing...")
    sa_result = simulated_annealing(mu, Sigma, lam=LAM, k=K, seed=42)
    sa_dict = result_to_dict(sa_result, TICKERS, "simulated_annealing")
    print(f"    selected: {sa_dict['selected_assets']}")
    print(f"    obj={sa_dict['objective']:.4f}  ret={sa_dict['annualised_return']:.4f}  var={sa_dict['annualised_variance']:.4f}")

    # ── Save results ──────────────────────────────────────────────────────────
    metadata = {
        "tickers": TICKERS,
        "trading_days": 100,
        "lambda": LAM,
        "k": K,
        "annualized_mu": dict(zip(TICKERS, np.round(mu, 6).tolist())),
    }

    for result_dict, fname in [
        (bf_dict,  "brute_force_result.json"),
        (gr_dict,  "greedy_result.json"),
        (sa_dict,  "simulated_annealing_result.json"),
    ]:
        out = OUT_DIR / fname
        with open(out, "w") as f:
            json.dump({"metadata": metadata, "result": result_dict}, f, indent=2)
        print(f"\n  Saved → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
