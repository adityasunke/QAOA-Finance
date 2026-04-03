"""
Brute-force exact solver for binary portfolio optimization.

Enumerates all 2^n binary portfolios and returns the one minimizing:
    f(x) = xᵀΣx - λμᵀx

subject to an optional cardinality constraint: sum(x) == k.
"""

import numpy as np
from itertools import product
from dataclasses import dataclass


@dataclass
class PortfolioResult:
    bitstring: np.ndarray   # binary selection vector (length n)
    objective: float        # f(x) = xᵀΣx − λμᵀx  (lower = better)
    ret: float              # portfolio return: μᵀx
    variance: float         # portfolio variance: xᵀΣx


def objective(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, lam: float) -> float:
    """Compute the portfolio objective f(x) = xᵀΣx − λμᵀx."""
    return float(x @ Sigma @ x - lam * mu @ x)


def brute_force(
    mu: np.ndarray,
    Sigma: np.ndarray,
    lam: float = 1.0,
    k: int | None = None,
) -> PortfolioResult:
    """
    Enumerate all 2^n portfolios and return the optimal one.

    Parameters
    ----------
    mu    : (n,) expected returns
    Sigma : (n, n) covariance matrix
    lam   : risk-aversion parameter (λ), scales the return term
    k     : if given, only consider portfolios with exactly k assets selected

    Returns
    -------
    PortfolioResult with the minimum-objective portfolio
    """
    n = len(mu)
    best: PortfolioResult | None = None

    for bits in product([0, 1], repeat=n):
        x = np.array(bits, dtype=float)

        if k is not None and int(x.sum()) != k:
            continue

        obj = objective(x, mu, Sigma, lam)

        if best is None or obj < best.objective:
            best = PortfolioResult(
                bitstring=x.copy(),
                objective=obj,
                ret=float(mu @ x),
                variance=float(x @ Sigma @ x),
            )

    if best is None:
        raise ValueError(f"No feasible portfolio found for k={k} with n={n} assets.")

    return best


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from data.generate_data import generate_assets

    for n in [4, 6, 8]:
        mu, Sigma = generate_assets(n, seed=42)
        result = brute_force(mu, Sigma, lam=1.0, k=n // 2)
        print(
            f"n={n}, k={n//2}: "
            f"bits={result.bitstring.astype(int).tolist()}, "
            f"obj={result.objective:.4f}, "
            f"ret={result.ret:.4f}, "
            f"var={result.variance:.4f}"
        )
