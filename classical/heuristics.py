"""
Classical heuristic solvers for binary portfolio optimization.

Both solvers minimize:  f(x) = xᵀΣx - λμᵀx
subject to an optional cardinality constraint: sum(x) == k.

Returns are in the same PortfolioResult format as the brute-force solver.
"""

import numpy as np
from classical.brute_force import PortfolioResult, objective


# ---------------------------------------------------------------------------
# Greedy
# ---------------------------------------------------------------------------

def greedy(
    mu: np.ndarray,
    Sigma: np.ndarray,
    lam: float = 1.0,
    k: int | None = None,
) -> PortfolioResult:
    """
    Iteratively add the asset with the best marginal improvement in objective.

    If k is None, add assets as long as the objective improves.
    If k is given, add exactly k assets (always picking the best marginal).
    """
    n = len(mu)
    x = np.zeros(n, dtype=float)
    selected = []

    budget = k if k is not None else n

    for _ in range(budget):
        best_gain = np.inf
        best_i = -1

        for i in range(n):
            if i in selected:
                continue
            x[i] = 1.0
            obj = objective(x, mu, Sigma, lam)
            if obj < best_gain:
                best_gain = obj
                best_i = i
            x[i] = 0.0

        if best_i == -1:
            break

        # If no budget constraint, only commit if it actually helps
        if k is None and best_gain >= objective(x, mu, Sigma, lam):
            break

        x[best_i] = 1.0
        selected.append(best_i)

    return PortfolioResult(
        bitstring=x.copy(),
        objective=objective(x, mu, Sigma, lam),
        ret=float(mu @ x),
        variance=float(x @ Sigma @ x),
    )


# ---------------------------------------------------------------------------
# Simulated Annealing
# ---------------------------------------------------------------------------

def simulated_annealing(
    mu: np.ndarray,
    Sigma: np.ndarray,
    lam: float = 1.0,
    k: int | None = None,
    T_init: float = 1.0,
    T_min: float = 1e-4,
    alpha: float = 0.98,
    steps_per_temp: int = 100,
    seed: int = 0,
) -> PortfolioResult:
    """
    Random bit-flip search with Boltzmann acceptance criterion.

    When k is given, moves swap one selected bit off and one unselected bit on
    to preserve cardinality. When k is None, single-bit flips are used.
    """
    rng = np.random.default_rng(seed)
    n = len(mu)

    # --- initialise feasibly ---
    if k is not None:
        x = np.zeros(n, dtype=float)
        chosen = rng.choice(n, size=k, replace=False)
        x[chosen] = 1.0
    else:
        x = rng.integers(0, 2, size=n).astype(float)

    current_obj = objective(x, mu, Sigma, lam)
    best_x = x.copy()
    best_obj = current_obj

    T = T_init

    while T > T_min:
        for _ in range(steps_per_temp):
            x_new = x.copy()

            if k is not None:
                # swap move: flip one 1→0 and one 0→1
                ones = np.where(x_new == 1.0)[0]
                zeros = np.where(x_new == 0.0)[0]
                if len(ones) == 0 or len(zeros) == 0:
                    continue
                flip_off = rng.choice(ones)
                flip_on = rng.choice(zeros)
                x_new[flip_off] = 0.0
                x_new[flip_on] = 1.0
            else:
                # single bit flip
                i = rng.integers(n)
                x_new[i] = 1.0 - x_new[i]

            new_obj = objective(x_new, mu, Sigma, lam)
            delta = new_obj - current_obj

            if delta < 0 or rng.random() < np.exp(-delta / T):
                x = x_new
                current_obj = new_obj

                if current_obj < best_obj:
                    best_obj = current_obj
                    best_x = x.copy()

        T *= alpha

    return PortfolioResult(
        bitstring=best_x,
        objective=best_obj,
        ret=float(mu @ best_x),
        variance=float(best_x @ Sigma @ best_x),
    )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from data.generate_data import load_assets, DEFAULT_TICKERS
    from classical.brute_force import brute_force

    subsets = [
        DEFAULT_TICKERS[:4],
        DEFAULT_TICKERS[:6],
        DEFAULT_TICKERS,
    ]
    for tickers in subsets:
        n = len(tickers)
        k = n // 2
        mu, Sigma = load_assets(tickers)
        bf = brute_force(mu, Sigma, lam=1.0, k=k)
        gr = greedy(mu, Sigma, lam=1.0, k=k)
        sa = simulated_annealing(mu, Sigma, lam=1.0, k=k, seed=42)

        print(f"\nn={n}, k={k}")
        print(f"  Brute-force:  obj={bf.objective:.4f}  bits={bf.bitstring.astype(int).tolist()}")
        print(f"  Greedy:       obj={gr.objective:.4f}  bits={gr.bitstring.astype(int).tolist()}")
        print(f"  Sim. Anneal:  obj={sa.objective:.4f}  bits={sa.bitstring.astype(int).tolist()}")
