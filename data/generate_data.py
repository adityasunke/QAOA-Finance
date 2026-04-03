"""
Synthetic financial data generation for QAOA portfolio optimization.
"""

import numpy as np


def generate_assets(n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic expected returns and covariance matrix for n assets.

    Parameters
    ----------
    n    : number of assets
    seed : random seed for reproducibility

    Returns
    -------
    mu    : (n,) array of expected returns, drawn from N(0.1, 0.05)
    Sigma : (n, n) positive-semidefinite covariance matrix, constructed as AᵀA / n
    """
    rng = np.random.default_rng(seed)

    mu = rng.normal(loc=0.1, scale=0.05, size=n)

    A = rng.standard_normal((n, n))
    Sigma = (A.T @ A) / n  # guaranteed PSD

    return mu, Sigma


if __name__ == "__main__":
    for n in [4, 6, 8, 10]:
        mu, Sigma = generate_assets(n, seed=42)
        print(f"n={n}: mu={np.round(mu, 4)}, Sigma diagonal={np.round(np.diag(Sigma), 4)}")
