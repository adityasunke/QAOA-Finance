"""
Metric computations for Phase 4 benchmarking.

Each function takes raw solver outputs (bitstring, mu, Sigma, counts) and
returns a single scalar or formatted string.  No I/O or solver logic here.
"""

from __future__ import annotations

import numpy as np


def portfolio_return(x: np.ndarray, mu: np.ndarray) -> float:
    """Annualized portfolio return μᵀx."""
    return float(mu @ x)


def portfolio_variance(x: np.ndarray, Sigma: np.ndarray) -> float:
    """Annualized portfolio variance xᵀΣx."""
    return float(x @ Sigma @ x)


def approximation_ratio(obj: float, optimal_obj: float) -> float:
    """
    Ratio of solver objective to brute-force optimal: f(x) / f*.

    Returns 1.0 when both are essentially zero; inf when f*≈0 but f(x) is not.
    For a minimisation problem, values closer to 1.0 indicate a better solution.
    """
    if abs(optimal_obj) < 1e-12:
        return 1.0 if abs(obj) < 1e-12 else float("inf")
    return obj / optimal_obj


def success_probability(counts: dict, optimal_bitstring: np.ndarray) -> float:
    """
    Fraction of measurement shots that returned the optimal bitstring.

    counts keys are Qiskit big-endian strings (qubit n-1 on the left);
    optimal_bitstring is little-endian (qubit 0 at index 0).
    """
    if not counts:
        return 0.0
    # Convert little-endian → Qiskit big-endian key
    target = "".join(str(int(b)) for b in reversed(optimal_bitstring))
    total = sum(counts.values())
    return counts.get(target, 0) / total if total > 0 else 0.0


def format_bitstring(x: np.ndarray) -> str:
    """Binary string representation, e.g. '1010' (qubit-0 on the left)."""
    return "".join(str(int(b)) for b in x)


def selected_stocks(x: np.ndarray, tickers: list[str]) -> str:
    """Pipe-separated list of selected ticker symbols, e.g. 'AAPL|GOOGL'."""
    chosen = [t for t, b in zip(tickers, x) if int(b) == 1]
    return "|".join(chosen) if chosen else "(none)"
