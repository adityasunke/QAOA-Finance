"""
QUBO matrix construction for binary portfolio optimization.

Objective:
    minimize  xᵀΣx − λμᵀx

With cardinality constraint (select exactly k assets) enforced via penalty:
    Penalty = A · (Σᵢ xᵢ − k)²

Combined QUBO:
    f(x) = xᵀQx    where Q encodes both objective and constraint
"""

import numpy as np


def build_qubo(
    mu: np.ndarray,
    Sigma: np.ndarray,
    lam: float = 1.0,
    k: int | None = None,
    penalty: float | None = None,
) -> tuple[np.ndarray, float]:
    """
    Build the QUBO matrix Q and constant offset such that:

        f(x) = xᵀQx + offset

    equals the portfolio objective plus cardinality penalty for all binary x.

    The QUBO is stored in upper-triangular form: diagonal Q[i,i] captures
    linear terms, and Q[i,j] for i < j captures quadratic terms.

    Parameters
    ----------
    mu      : (n,) expected returns
    Sigma   : (n, n) covariance matrix (PSD)
    lam     : risk-aversion parameter λ (scales the return term)
    k       : target cardinality (number of assets to select); if None,
              no cardinality penalty is added
    penalty : penalty coefficient A; if None and k is given, defaults to
              max(|Q_ij|) * n (computed from the objective part first)

    Returns
    -------
    Q      : (n, n) upper-triangular QUBO matrix
    offset : constant term A·k² (zero when k is None)
    """
    n = len(mu)

    # --- Objective: xᵀΣx − λμᵀx ----------------------------------------
    # xᵀΣx diagonal (i==j): Σᵢᵢ xᵢ² = Σᵢᵢ xᵢ  (since xᵢ ∈ {0,1})
    # off-diagonal: (Σᵢⱼ + Σⱼᵢ) xᵢ xⱼ stored in upper triangle as 2Σᵢⱼ
    # −λμᵀx: linear → diagonal

    Q = np.zeros((n, n))
    offset = 0.0

    for i in range(n):
        Q[i, i] += Sigma[i, i] - lam * mu[i]
        for j in range(i + 1, n):
            Q[i, j] += Sigma[i, j] + Sigma[j, i]  # symmetric contribution

    # --- Cardinality penalty: A·(Σᵢ xᵢ − k)² ---------------------------
    if k is not None:
        # Auto-set penalty if not provided
        if penalty is None:
            max_abs = np.max(np.abs(Q)) if np.any(Q != 0) else 1.0
            penalty = max_abs * n

        # Expand (Σᵢ xᵢ − k)² = Σᵢ xᵢ + 2 Σᵢ<ⱼ xᵢ xⱼ − 2k Σᵢ xᵢ + k²
        # The k² constant is tracked as offset (doesn't appear in xᵀQx)
        offset = penalty * k * k
        for i in range(n):
            Q[i, i] += penalty * (1 - 2 * k)
            for j in range(i + 1, n):
                Q[i, j] += penalty * 2

    return Q, offset


def evaluate_qubo(Q: np.ndarray, x: np.ndarray) -> float:
    """
    Evaluate the QUBO objective xᵀQx for a given binary vector x.

    For upper-triangular Q this equals:
        Σᵢ Q[i,i] xᵢ + Σᵢ<ⱼ Q[i,j] xᵢ xⱼ
    """
    return float(x @ Q @ x)


def verify_qubo(
    Q: np.ndarray,
    offset: float,
    mu: np.ndarray,
    Sigma: np.ndarray,
    lam: float = 1.0,
    k: int | None = None,
    penalty: float | None = None,
) -> bool:
    """
    Verify that xᵀQx + offset matches the direct objective + penalty for all 2^n states.

    Returns True if all states agree (within floating-point tolerance).
    Prints a summary; raises AssertionError on any mismatch.
    """
    from itertools import product as iproduct

    n = len(mu)

    # Infer penalty coefficient if needed for direct computation
    if k is not None and penalty is None:
        # Recover A from offset: offset = A * k²
        penalty = offset / (k * k) if k != 0 else 0.0

    max_err = 0.0
    for bits in iproduct([0, 1], repeat=n):
        x = np.array(bits, dtype=float)

        qubo_val = evaluate_qubo(Q, x) + offset

        # Direct computation
        obj = float(x @ Sigma @ x - lam * mu @ x)
        pen = float(penalty * (x.sum() - k) ** 2) if (k is not None and penalty is not None) else 0.0
        direct_val = obj + pen

        err = abs(qubo_val - direct_val)
        max_err = max(max_err, err)
        assert err < 1e-9, (
            f"Mismatch at x={bits}: QUBO+offset={qubo_val:.6f}, direct={direct_val:.6f}, err={err:.2e}"
        )

    print(f"QUBO verification passed for all 2^{n} states. Max error: {max_err:.2e}")
    return True


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from data.generate_data import generate_assets

    for n in [4, 6, 8]:
        mu, Sigma = generate_assets(n, seed=42)
        k = n // 2
        Q, offset = build_qubo(mu, Sigma, lam=1.0, k=k)
        print(f"\nn={n}, k={k}")
        print(f"Q diagonal: {np.round(np.diag(Q), 4)}")
        print(f"offset (A·k²): {offset:.4f}")
        verify_qubo(Q, offset, mu, Sigma, lam=1.0, k=k)
