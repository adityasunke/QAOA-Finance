"""
Metrics and result structures for portfolio optimization benchmarking.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ExperimentResult:
    """One result record for a (method, n_assets, p) combination."""
    method: str               # 'brute_force', 'greedy', 'sim_anneal', 'qaoa_aer', 'qaoa_ibm'
    n_assets: int
    tickers: list[str]
    p: int                    # QAOA depth; 0 for classical methods
    k: int                    # cardinality constraint
    lam: float
    bitstring: list[int]
    selected_tickers: list[str]
    objective: float          # f(x) = xᵀΣx − λμᵀx  (lower = better)
    ret: float                # annualized portfolio return
    variance: float           # annualized portfolio variance
    approx_ratio: float       # f_method / f_optimal  (1.0 = optimal)
    runtime: float            # wall-clock seconds
    success_prob: float       # fraction of shots matching optimal (QAOA only)
    n_iters: int              # COBYLA iterations (QAOA only)
    is_optimal: bool          # whether bitstring matches brute-force optimal
    counts: dict = field(default_factory=dict)  # raw measurement counts (QAOA only)


def approx_ratio(obj: float, optimal_obj: float) -> float:
    """
    f_method / f_optimal.

    For negative objectives (good portfolios): ratio in (0, 1], 1.0 = optimal.
    For positive objectives: ratio >= 1.0.
    """
    if abs(optimal_obj) < 1e-12:
        return 1.0 if abs(obj) < 1e-12 else 0.0
    return obj / optimal_obj


def success_probability(counts: dict, optimal_bits: np.ndarray, shots: int) -> float:
    """
    Fraction of shots that returned the optimal bitstring.

    Qiskit count strings are big-endian, so the optimal bitstring is
    represented as reversed(optimal_bits).
    """
    if not counts or shots == 0:
        return 0.0
    opt_str = "".join(str(int(b)) for b in reversed(optimal_bits))
    return counts.get(opt_str, 0) / shots


def sharpe_ratio(ret: float, variance: float, risk_free: float = 0.05) -> float:
    """Annualized Sharpe ratio with a 5% risk-free rate."""
    std = np.sqrt(max(variance, 0.0))
    return (ret - risk_free) / std if std > 0 else 0.0


def efficient_frontier(
    mu: np.ndarray,
    Sigma: np.ndarray,
    k: int,
    n_points: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the efficient frontier by sweeping the risk-aversion parameter λ.

    Returns arrays of (variances, returns) along the frontier.
    Uses brute-force for exact solutions (tractable for n ≤ 10).
    """
    from classical.brute_force import brute_force as bf_solve

    lam_values = np.logspace(-2, 2, n_points)
    variances, returns = [], []

    seen = set()
    for lam in lam_values:
        res = bf_solve(mu, Sigma, lam=lam, k=k)
        key = tuple(res.bitstring.astype(int))
        if key not in seen:
            seen.add(key)
            variances.append(res.variance)
            returns.append(res.ret)

    # Sort by variance for a clean curve
    order = np.argsort(variances)
    return np.array(variances)[order], np.array(returns)[order]
