"""
Phase 4 Benchmarking: Compare brute-force, greedy, simulated annealing,
QAOA (Qiskit Aer), and QAOA (IBM Runtime / FakeNairobiV2 noise model).

Saves results to benchmarks/results/ and generates 7 comparison charts.

Usage
-----
    python benchmarks/run_experiments.py            # full sweep
    python benchmarks/run_experiments.py --fast     # n=4 only, quick demo
    python benchmarks/run_experiments.py --skip-ibm # skip IBM Runtime experiments
"""

from __future__ import annotations

import sys
import json
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import seaborn as sns

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.generate_data import load_assets, DEFAULT_TICKERS
from classical.brute_force import brute_force as bf_solve, objective as portfolio_obj
from classical.heuristics import greedy, simulated_annealing
from quantum.qaoa_runner import run_qaoa
from benchmarks.metrics import (
    ExperimentResult,
    approx_ratio,
    success_probability,
    efficient_frontier,
    sharpe_ratio,
)

# ── IBM Runtime ───────────────────────────────────────────────────────────────
IBM_AVAILABLE = False
try:
    from qiskit_ibm_runtime.fake_provider import FakeNairobiV2
    from qiskit_ibm_runtime import SamplerV2 as IBMSampler
    IBM_AVAILABLE = True
except ImportError:
    pass

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "savefig.dpi": 150})

METHOD_COLORS = {
    "brute_force": "#2c7bb6",
    "greedy":      "#4dac26",
    "sim_anneal":  "#f1a340",
    "qaoa_aer":    "#d7191c",
    "qaoa_ibm":    "#762a83",
}
METHOD_LABELS = {
    "brute_force": "Brute Force (optimal)",
    "greedy":      "Greedy",
    "sim_anneal":  "Sim. Annealing",
    "qaoa_aer":    "QAOA (Aer)",
    "qaoa_ibm":    "QAOA (IBM Runtime)",
}
METHOD_MARKERS = {
    "brute_force": "D",
    "greedy":      "s",
    "sim_anneal":  "^",
    "qaoa_aer":    "o",
    "qaoa_ibm":    "P",
}

# ── Experiment configuration ──────────────────────────────────────────────────
FULL_SWEEP = [
    (4, [1, 2, 3]),
    (6, [1, 2, 3]),
    (7, [1, 2]),
]
FAST_SWEEP = [
    (4, [1, 2, 3]),
]
# IBM Runtime only runs for the following (n, p) pairs to limit runtime
IBM_SWEEP = {(4, 1), (4, 2), (4, 3), (6, 1), (6, 2), (7, 1)}

LAM   = 1.0
SHOTS = 4000
SEED  = 42


# ═══════════════════════════════════════════════════════════════════════════════
#  Individual experiment runners
# ═══════════════════════════════════════════════════════════════════════════════

def run_brute_force(mu, Sigma, k, tickers) -> ExperimentResult:
    t0 = time.perf_counter()
    res = bf_solve(mu, Sigma, lam=LAM, k=k)
    rt = time.perf_counter() - t0
    n = len(mu)
    return ExperimentResult(
        method="brute_force",
        n_assets=n, tickers=tickers, p=0, k=k, lam=LAM,
        bitstring=res.bitstring.astype(int).tolist(),
        selected_tickers=[tickers[i] for i in range(n) if res.bitstring[i] == 1],
        objective=res.objective,
        ret=res.ret,
        variance=res.variance,
        approx_ratio=1.0,
        runtime=rt,
        success_prob=0.0, n_iters=0, is_optimal=True, counts={},
    )


def run_greedy_exp(mu, Sigma, k, tickers, optimal_res: ExperimentResult) -> ExperimentResult:
    t0 = time.perf_counter()
    res = greedy(mu, Sigma, lam=LAM, k=k)
    rt = time.perf_counter() - t0
    n = len(mu)
    return ExperimentResult(
        method="greedy",
        n_assets=n, tickers=tickers, p=0, k=k, lam=LAM,
        bitstring=res.bitstring.astype(int).tolist(),
        selected_tickers=[tickers[i] for i in range(n) if res.bitstring[i] == 1],
        objective=res.objective,
        ret=res.ret,
        variance=res.variance,
        approx_ratio=approx_ratio(res.objective, optimal_res.objective),
        runtime=rt,
        success_prob=0.0, n_iters=0,
        is_optimal=res.bitstring.astype(int).tolist() == optimal_res.bitstring,
        counts={},
    )


def run_sa_exp(mu, Sigma, k, tickers, optimal_res: ExperimentResult) -> ExperimentResult:
    t0 = time.perf_counter()
    res = simulated_annealing(mu, Sigma, lam=LAM, k=k, seed=SEED)
    rt = time.perf_counter() - t0
    n = len(mu)
    return ExperimentResult(
        method="sim_anneal",
        n_assets=n, tickers=tickers, p=0, k=k, lam=LAM,
        bitstring=res.bitstring.astype(int).tolist(),
        selected_tickers=[tickers[i] for i in range(n) if res.bitstring[i] == 1],
        objective=res.objective,
        ret=res.ret,
        variance=res.variance,
        approx_ratio=approx_ratio(res.objective, optimal_res.objective),
        runtime=rt,
        success_prob=0.0, n_iters=0,
        is_optimal=res.bitstring.astype(int).tolist() == optimal_res.bitstring,
        counts={},
    )


def run_qaoa_aer_exp(mu, Sigma, k, tickers, p, optimal_res: ExperimentResult) -> ExperimentResult:
    n = len(mu)
    res = run_qaoa(mu, Sigma, lam=LAM, k=k, p=p, shots=SHOTS, seed=SEED)
    return ExperimentResult(
        method="qaoa_aer",
        n_assets=n, tickers=tickers, p=p, k=k, lam=LAM,
        bitstring=res.bitstring.astype(int).tolist(),
        selected_tickers=[tickers[i] for i in range(n) if res.bitstring[i] == 1],
        objective=res.objective,
        ret=res.ret,
        variance=res.variance,
        approx_ratio=approx_ratio(res.objective, optimal_res.objective),
        runtime=res.runtime,
        success_prob=success_probability(res.counts, np.array(optimal_res.bitstring), SHOTS),
        n_iters=res.n_iters,
        is_optimal=res.bitstring.astype(int).tolist() == optimal_res.bitstring,
        counts=res.counts,
    )


def run_qaoa_ibm_exp(
    mu: np.ndarray,
    Sigma: np.ndarray,
    k: int,
    tickers: list[str],
    p: int,
    optimal_res: ExperimentResult,
) -> ExperimentResult | None:
    """
    Optimize QAOA parameters on Aer, then execute on FakeNairobiV2 (noise model).
    """
    if not IBM_AVAILABLE:
        return None

    from qiskit import transpile
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator
    from scipy.optimize import minimize as sp_minimize
    from quantum.qubo import build_qubo
    from quantum.hamiltonian import qubo_to_ising, build_ising_hamiltonian
    from quantum.qaoa_circuit import build_qaoa_circuit

    t0 = time.perf_counter()
    n = len(mu)

    # ── Phase 1: optimize parameters on Aer ──────────────────────────────────
    Q, _ = build_qubo(mu, Sigma, lam=LAM, k=k)
    h, J, ising_offset = qubo_to_ising(Q)
    H_C = build_ising_hamiltonian(h, J)
    qc, gammas, betas = build_qaoa_circuit(h, J, p=p)
    all_params = list(gammas) + list(betas)

    sim = AerSimulator(method="statevector")
    qc_sv = qc.copy()
    qc_sv.save_statevector()
    qc_sv_t = transpile(qc_sv, sim)

    iter_count = [0]

    def cost(params: np.ndarray) -> float:
        iter_count[0] += 1
        pd_ = {po: float(v) for po, v in zip(all_params, params)}
        sv = sim.run(qc_sv_t.assign_parameters(pd_)).result().get_statevector()
        return float(Statevector(sv).expectation_value(H_C).real) + ising_offset

    rng = np.random.default_rng(SEED)
    x0 = rng.uniform(0.0, np.pi, 2 * p)
    opt = sp_minimize(cost, x0, method="COBYLA", options={"maxiter": 1000, "rhobeg": 0.5})

    # ── Phase 2: execute on FakeNairobiV2 ─────────────────────────────────────
    param_dict = {po: float(v) for po, v in zip(all_params, opt.x)}
    qc_bound = qc.assign_parameters(param_dict)
    qc_bound.measure_all()

    fake_backend = FakeNairobiV2()
    qc_t = transpile(qc_bound, backend=fake_backend, optimization_level=1)

    sampler = IBMSampler(mode=fake_backend)
    job = sampler.run([qc_t], shots=SHOTS)
    ibm_result = job.result()
    counts = ibm_result[0].data.meas.get_counts()

    # Decode most frequent bitstring satisfying cardinality k
    best_bits = None
    for bitstr, _ in sorted(counts.items(), key=lambda kv: -kv[1]):
        x = np.array([int(b) for b in reversed(bitstr)], dtype=float)
        if k is None or int(x.sum()) == k:
            best_bits = x
            break
    if best_bits is None:
        top = max(counts, key=counts.get)
        best_bits = np.array([int(b) for b in reversed(top)], dtype=float)

    rt = time.perf_counter() - t0
    obj = portfolio_obj(best_bits, mu, Sigma, LAM)

    return ExperimentResult(
        method="qaoa_ibm",
        n_assets=n, tickers=tickers, p=p, k=k, lam=LAM,
        bitstring=best_bits.astype(int).tolist(),
        selected_tickers=[tickers[i] for i in range(n) if best_bits[i] == 1],
        objective=obj,
        ret=float(mu @ best_bits),
        variance=float(best_bits @ Sigma @ best_bits),
        approx_ratio=approx_ratio(obj, optimal_res.objective),
        runtime=rt,
        success_prob=success_probability(counts, np.array(optimal_res.bitstring), SHOTS),
        n_iters=iter_count[0],
        is_optimal=best_bits.astype(int).tolist() == optimal_res.bitstring,
        counts=counts,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Main experiment driver
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_experiments(
    sweep: list[tuple[int, list[int]]],
    skip_ibm: bool = False,
    output_dir: Path = ROOT / "benchmarks" / "results",
) -> list[ExperimentResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[ExperimentResult] = []

    for n, depths in sweep:
        tickers = DEFAULT_TICKERS[:n]
        k = n // 2
        print(f"\n{'='*60}")
        print(f"  n={n} assets  k={k}  tickers={tickers}")
        print(f"{'='*60}")

        mu, Sigma = load_assets(tickers)

        # ── Brute force ──────────────────────────────────────────────────────
        print(f"  [1/5] Brute force ...", end=" ", flush=True)
        bf_res = run_brute_force(mu, Sigma, k, tickers)
        print(f"obj={bf_res.objective:.4f}  time={bf_res.runtime:.3f}s")
        results.append(bf_res)

        # ── Greedy ───────────────────────────────────────────────────────────
        print(f"  [2/5] Greedy ...", end=" ", flush=True)
        gr_res = run_greedy_exp(mu, Sigma, k, tickers, bf_res)
        print(f"obj={gr_res.objective:.4f}  ratio={gr_res.approx_ratio:.4f}  time={gr_res.runtime:.4f}s")
        results.append(gr_res)

        # ── Simulated annealing ───────────────────────────────────────────────
        print(f"  [3/5] Simulated annealing ...", end=" ", flush=True)
        sa_res = run_sa_exp(mu, Sigma, k, tickers, bf_res)
        print(f"obj={sa_res.objective:.4f}  ratio={sa_res.approx_ratio:.4f}  time={sa_res.runtime:.3f}s")
        results.append(sa_res)

        # ── QAOA (Aer) ────────────────────────────────────────────────────────
        for p in depths:
            print(f"  [4/5] QAOA Aer p={p} ...", end=" ", flush=True)
            aer_res = run_qaoa_aer_exp(mu, Sigma, k, tickers, p, bf_res)
            print(
                f"obj={aer_res.objective:.4f}  ratio={aer_res.approx_ratio:.4f}"
                f"  iters={aer_res.n_iters}  time={aer_res.runtime:.1f}s"
            )
            results.append(aer_res)

        # ── QAOA (IBM Runtime) ────────────────────────────────────────────────
        if not skip_ibm and IBM_AVAILABLE:
            for p in depths:
                if (n, p) not in IBM_SWEEP:
                    continue
                print(f"  [5/5] QAOA IBM Runtime p={p} ...", end=" ", flush=True)
                ibm_res = run_qaoa_ibm_exp(mu, Sigma, k, tickers, p, bf_res)
                if ibm_res is not None:
                    print(
                        f"obj={ibm_res.objective:.4f}  ratio={ibm_res.approx_ratio:.4f}"
                        f"  iters={ibm_res.n_iters}  time={ibm_res.runtime:.1f}s"
                    )
                    results.append(ibm_res)
        elif skip_ibm:
            print("  [5/5] QAOA IBM Runtime  (skipped)")
        elif not IBM_AVAILABLE:
            print("  [5/5] QAOA IBM Runtime  (qiskit-ibm-runtime not installed)")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Serialisation
# ═══════════════════════════════════════════════════════════════════════════════

def to_dataframe(results: list[ExperimentResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "method": r.method,
            "n_assets": r.n_assets,
            "p": r.p,
            "k": r.k,
            "objective": r.objective,
            "ret": r.ret,
            "variance": r.variance,
            "approx_ratio": r.approx_ratio,
            "runtime": r.runtime,
            "success_prob": r.success_prob,
            "n_iters": r.n_iters,
            "is_optimal": r.is_optimal,
            "bitstring": r.bitstring,
            "selected_tickers": r.selected_tickers,
            "sharpe": sharpe_ratio(r.ret, r.variance),
        })
    return pd.DataFrame(rows)


def save_results(results: list[ExperimentResult], output_dir: Path) -> None:
    df = to_dataframe(results)
    df.to_csv(output_dir / "results.csv", index=False)

    # JSON with full details
    json_data = []
    for r in results:
        d = dict(r.__dict__)
        json_data.append(d)
    with open(output_dir / "results.json", "w") as f:
        json.dump(json_data, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}/")


# ═══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def _method_label(m: str) -> str:
    return METHOD_LABELS.get(m, m)


def _method_color(m: str) -> str:
    return METHOD_COLORS.get(m, "#888888")


def _method_marker(m: str) -> str:
    return METHOD_MARKERS.get(m, "o")


def plot_approx_ratio_vs_depth(df: pd.DataFrame, output_dir: Path) -> None:
    """Fig 1 – Approximation ratio vs QAOA depth, one subplot per n_assets."""
    ns = sorted(df["n_assets"].unique())
    n_cols = len(ns)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 5), sharey=True)
    if n_cols == 1:
        axes = [axes]

    for ax, n in zip(axes, ns):
        sub = df[df["n_assets"] == n]

        # QAOA lines
        for method in ["qaoa_aer", "qaoa_ibm"]:
            mdf = sub[sub["method"] == method].sort_values("p")
            if mdf.empty:
                continue
            ax.plot(
                mdf["p"], mdf["approx_ratio"],
                marker=_method_marker(method),
                color=_method_color(method),
                label=_method_label(method),
                linewidth=2, markersize=8,
            )

        # Classical baselines as horizontal dashed lines
        for method in ["greedy", "sim_anneal"]:
            mdf = sub[sub["method"] == method]
            if mdf.empty:
                continue
            ratio_val = mdf["approx_ratio"].values[0]
            ax.axhline(
                ratio_val, linestyle="--", linewidth=1.5,
                color=_method_color(method), label=_method_label(method),
            )

        # Optimal reference
        ax.axhline(1.0, linestyle=":", linewidth=1.5, color=_method_color("brute_force"),
                   label=_method_label("brute_force"))

        depths = sorted(sub[sub["method"] == "qaoa_aer"]["p"].unique())
        ax.set_xticks(depths)
        ax.set_xlabel("QAOA depth p", fontsize=12)
        ax.set_title(f"n = {n} assets", fontsize=13, fontweight="bold")
        all_ratios = sub[sub["method"].isin(["qaoa_aer", "qaoa_ibm"])]["approx_ratio"].dropna()
        y_top = max(all_ratios.max() if not all_ratios.empty else 1.0, 1.0) * 1.12
        ax.set_ylim(0, y_top)
        ax.legend(fontsize=9, loc="upper right")

    axes[0].set_ylabel("Approximation ratio  f / f*  (1.0 = optimal)", fontsize=12)
    fig.suptitle("Approximation Ratio vs. QAOA Circuit Depth\n(lower = better; 1.0 = brute-force optimal)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_approx_ratio_vs_depth.png")
    plt.close(fig)
    print("  Saved fig1_approx_ratio_vs_depth.png")


def plot_runtime_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Fig 2 – Wall-clock runtime comparison (log scale) for p=1 QAOA."""
    methods_order = ["brute_force", "greedy", "sim_anneal", "qaoa_aer", "qaoa_ibm"]
    ns = sorted(df["n_assets"].unique())

    # Use p=1 for QAOA methods; p=0 for classical
    runtime_data: dict[str, list[float]] = {m: [] for m in methods_order}

    for n in ns:
        sub = df[df["n_assets"] == n]
        for method in methods_order:
            if method in ("qaoa_aer", "qaoa_ibm"):
                row = sub[(sub["method"] == method) & (sub["p"] == 1)]
            else:
                row = sub[sub["method"] == method]
            runtime_data[method].append(row["runtime"].values[0] if not row.empty else np.nan)

    x = np.arange(len(ns))
    n_methods = len(methods_order)
    width = 0.14
    offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * width

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(methods_order):
        vals = runtime_data[method]
        mask = ~np.isnan(vals)
        if not any(mask):
            continue
        ax.bar(
            x[mask] + offsets[i], np.array(vals)[mask],
            width=width, label=_method_label(method),
            color=_method_color(method), alpha=0.85, edgecolor="white",
        )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}" for n in ns], fontsize=12)
    ax.set_xlabel("Portfolio size (n assets)", fontsize=12)
    ax.set_ylabel("Runtime (seconds, log scale)", fontsize=12)
    ax.set_title("Wall-Clock Runtime Comparison  (p=1 for QAOA)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.yaxis.set_major_formatter(mticker.LogFormatter())
    ax.grid(axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_runtime_comparison.png")
    plt.close(fig)
    print("  Saved fig2_runtime_comparison.png")


def plot_solution_quality_bars(df: pd.DataFrame, output_dir: Path) -> None:
    """Fig 3 – Portfolio return & variance for every method (best p per method)."""
    ns = sorted(df["n_assets"].unique())
    methods_order = ["brute_force", "greedy", "sim_anneal", "qaoa_aer", "qaoa_ibm"]
    fig, axes = plt.subplots(2, len(ns), figsize=(5.5 * len(ns), 9))
    if len(ns) == 1:
        axes = axes.reshape(2, 1)

    for col, n in enumerate(ns):
        sub = df[df["n_assets"] == n]
        labels, returns, variances, colors = [], [], [], []

        for method in methods_order:
            mdf = sub[sub["method"] == method]
            if mdf.empty:
                continue
            # best approximation ratio across depths
            best_row = mdf.loc[mdf["approx_ratio"].idxmax()]
            p_str = f" p={int(best_row['p'])}" if best_row["p"] > 0 else ""
            labels.append(_method_label(method) + p_str)
            returns.append(best_row["ret"])
            variances.append(best_row["variance"])
            colors.append(_method_color(method))

        x = np.arange(len(labels))
        # Returns
        ax_r = axes[0, col]
        bars = ax_r.bar(x, returns, color=colors, alpha=0.85, edgecolor="white")
        ax_r.set_xticks(x)
        ax_r.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax_r.set_ylabel("Ann. portfolio return", fontsize=11)
        ax_r.set_title(f"n={n} — Return", fontsize=12, fontweight="bold")
        for bar, val in zip(bars, returns):
            ax_r.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                      f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        # Variances
        ax_v = axes[1, col]
        bars = ax_v.bar(x, variances, color=colors, alpha=0.85, edgecolor="white")
        ax_v.set_xticks(x)
        ax_v.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax_v.set_ylabel("Ann. portfolio variance", fontsize=11)
        ax_v.set_title(f"n={n} — Variance", fontsize=12, fontweight="bold")
        for bar, val in zip(bars, variances):
            ax_v.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                      f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Portfolio Return & Variance by Method", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "fig3_solution_quality_bars.png")
    plt.close(fig)
    print("  Saved fig3_solution_quality_bars.png")


def plot_success_probability(df: pd.DataFrame, output_dir: Path) -> None:
    """Fig 4 – Success probability vs QAOA depth for Aer and IBM Runtime."""
    ns = sorted(df["n_assets"].unique())
    fig, axes = plt.subplots(1, len(ns), figsize=(5.5 * len(ns), 5), sharey=True)
    if len(ns) == 1:
        axes = [axes]

    for ax, n in zip(axes, ns):
        sub = df[df["n_assets"] == n]
        for method in ["qaoa_aer", "qaoa_ibm"]:
            mdf = sub[sub["method"] == method].sort_values("p")
            if mdf.empty:
                continue
            ax.plot(
                mdf["p"], mdf["success_prob"] * 100,
                marker=_method_marker(method),
                color=_method_color(method),
                label=_method_label(method),
                linewidth=2, markersize=9,
            )
        depths = sorted(sub[sub["method"] == "qaoa_aer"]["p"].unique())
        ax.set_xticks(depths)
        ax.set_xlabel("QAOA depth p", fontsize=12)
        ax.set_title(f"n = {n} assets", fontsize=13, fontweight="bold")
        ax.set_ylim(-2, 102)
        ax.set_ylabel("Success probability (%)", fontsize=12)
        ax.legend(fontsize=10)

    fig.suptitle("Success Probability vs. QAOA Depth", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "fig4_success_probability.png")
    plt.close(fig)
    print("  Saved fig4_success_probability.png")


def plot_measurement_distribution(
    results: list[ExperimentResult],
    output_dir: Path,
) -> None:
    """Fig 5 – Bitstring measurement distribution for n=4, p=1 (Aer vs IBM)."""
    # Find the n=4, p=1 results
    aer_res = next(
        (r for r in results if r.method == "qaoa_aer" and r.n_assets == 4 and r.p == 1), None
    )
    ibm_res = next(
        (r for r in results if r.method == "qaoa_ibm" and r.n_assets == 4 and r.p == 1), None
    )

    if aer_res is None:
        print("  Skipping fig5 (no n=4, p=1 Aer result)")
        return

    n_plots = 1 + (ibm_res is not None)
    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    def _plot_hist(ax, res: ExperimentResult, title: str) -> None:
        counts = res.counts
        if not counts:
            ax.text(0.5, 0.5, "No counts", ha="center", va="center", transform=ax.transAxes)
            return
        total = sum(counts.values())
        items = sorted(counts.items(), key=lambda kv: -kv[1])[:20]  # top 20
        labels = [bs for bs, _ in items]
        probs = [cnt / total for _, cnt in items]

        # Mark optimal bitstring
        opt_str = "".join(str(b) for b in reversed(res.bitstring))
        bar_colors = [
            "#d7191c" if lbl == opt_str else "#aaaaaa" for lbl in labels
        ]
        bars = ax.bar(range(len(labels)), probs, color=bar_colors, edgecolor="white")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.set_xlabel("Bitstring (big-endian)", fontsize=11)
        ax.set_ylabel("Probability", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")

        # Legend patch
        from matplotlib.patches import Patch
        ax.legend(
            handles=[
                Patch(color="#d7191c", label="Decoded optimal"),
                Patch(color="#aaaaaa", label="Other"),
            ],
            fontsize=9,
        )

    _plot_hist(axes[0], aer_res, "QAOA (Aer)  n=4, p=1")
    if ibm_res is not None:
        _plot_hist(axes[1], ibm_res, "QAOA (IBM Runtime / FakeNairobiV2)  n=4, p=1")

    fig.suptitle("Measurement Outcome Distribution", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "fig5_measurement_distribution.png")
    plt.close(fig)
    print("  Saved fig5_measurement_distribution.png")


def plot_efficient_frontier(
    results: list[ExperimentResult],
    output_dir: Path,
) -> None:
    """Fig 6 – Risk-return plane: all feasible portfolios + method solutions (n=4)."""
    from itertools import combinations

    n = 4
    tickers = DEFAULT_TICKERS[:n]
    k = n // 2
    mu, Sigma = load_assets(tickers)

    # Enumerate every C(n,k) feasible portfolio
    all_vars, all_rets, all_objs = [], [], []
    opt_var, opt_ret = None, None
    best_obj = np.inf
    for combo in combinations(range(n), k):
        x = np.zeros(n)
        x[list(combo)] = 1.0
        v = float(x @ Sigma @ x)
        r = float(mu @ x)
        o = portfolio_obj(x, mu, Sigma, LAM)
        all_vars.append(v * 100)
        all_rets.append(r * 100)
        all_objs.append(o)
        if o < best_obj:
            best_obj = o
            opt_var, opt_ret = v * 100, r * 100

    fig, ax = plt.subplots(figsize=(9, 6))

    # All feasible portfolios as light background dots
    ax.scatter(all_vars, all_rets, c="#cccccc", s=90, zorder=2, label="All feasible (k=2) portfolios")

    # Mark the globally optimal portfolio
    if opt_var is not None:
        ax.scatter(opt_var, opt_ret, c="#2c7bb6", s=260, marker="*",
                   zorder=6, label="Optimal (min objective)", edgecolors="white")

    # Efficient frontier curve (Pareto-optimal in variance-return space)
    ef_var, ef_ret = efficient_frontier(mu, Sigma, k=k)
    ax.plot(ef_var * 100, ef_ret * 100, color="#555555", linewidth=1.5,
            zorder=3, label="Efficient frontier", linestyle="--")

    # Overlay each method's BEST p solution for n=4
    jitter = {
        "brute_force": (0, 0), "greedy": (3, 3), "sim_anneal": (3, -4),
        "qaoa_aer": (-4, 4), "qaoa_ibm": (4, -4),
    }
    plotted_methods: set[str] = set()
    for method in ["brute_force", "greedy", "sim_anneal", "qaoa_aer", "qaoa_ibm"]:
        candidates = [rr for rr in results if rr.n_assets == n and rr.method == method]
        if not candidates:
            continue
        r = min(candidates, key=lambda rr: rr.approx_ratio)
        if r.method in plotted_methods:
            continue
        plotted_methods.add(r.method)
        jx, jy = jitter.get(method, (0, 0))
        ax.scatter(
            r.variance * 100, r.ret * 100,
            marker=_method_marker(method), s=200,
            color=_method_color(method),
            label=_method_label(method),
            zorder=7, edgecolors="white", linewidths=0.8,
        )
        short = _method_label(method).replace("(optimal)", "").replace("Brute Force", "BF").strip()
        ax.annotate(short, (r.variance * 100, r.ret * 100),
                    textcoords="offset points", xytext=(jx + 6, jy + 4), fontsize=8.5)

    ax.set_xlabel("Portfolio variance × 100 (%²)", fontsize=12)
    ax.set_ylabel("Portfolio return × 100 (%)", fontsize=12)
    ax.set_title(
        f"Risk-Return Plane — All Feasible Portfolios  (n={n}, k={k})\n"
        "Each gray dot = one C(4,2) portfolio; colored markers = solver solutions",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "fig6_efficient_frontier.png")
    plt.close(fig)
    print("  Saved fig6_efficient_frontier.png")


def plot_summary_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Fig 7 – Heatmap of approximation ratios across all (method × config) combinations."""
    methods_order = ["brute_force", "greedy", "sim_anneal", "qaoa_aer", "qaoa_ibm"]
    ns = sorted(df["n_assets"].unique())

    # Build pivot: rows = method+p config, columns = n_assets
    rows_data = []
    row_labels = []

    # Classical methods (one row each)
    for method in ["brute_force", "greedy", "sim_anneal"]:
        row = {}
        for n in ns:
            sub = df[(df["n_assets"] == n) & (df["method"] == method)]
            row[f"n={n}"] = sub["approx_ratio"].values[0] if not sub.empty else np.nan
        rows_data.append(row)
        row_labels.append(_method_label(method))

    # QAOA methods (one row per depth p)
    for method in ["qaoa_aer", "qaoa_ibm"]:
        all_p = sorted(df[df["method"] == method]["p"].unique())
        for p in all_p:
            row = {}
            for n in ns:
                sub = df[(df["n_assets"] == n) & (df["method"] == method) & (df["p"] == p)]
                row[f"n={n}"] = sub["approx_ratio"].values[0] if not sub.empty else np.nan
            rows_data.append(row)
            row_labels.append(f"{_method_label(method)}  p={p}")

    pivot = pd.DataFrame(rows_data, index=row_labels)

    # Convert approx ratios to a "quality" score (1/ratio) so that
    # 1.0 = optimal (green) and values < 1.0 = suboptimal (red/yellow).
    quality = pivot.copy().astype(float)
    for col in quality.columns:
        quality[col] = quality[col].apply(
            lambda x: 1.0 / x if (pd.notna(x) and x > 0) else np.nan
        )

    # Annotation text: show the raw approx ratio value
    annot_text = pivot.copy().astype(object)
    for col in annot_text.columns:
        annot_text[col] = annot_text[col].apply(
            lambda x: f"{x:.3f}" if pd.notna(x) else ""
        )

    fig, ax = plt.subplots(figsize=(max(7, 2.8 * len(ns)), max(6, 0.65 * len(row_labels))))
    sns.heatmap(
        quality,
        annot=annot_text, fmt="",
        cmap="RdYlGn", vmin=0.0, vmax=1.0,
        linewidths=0.5, linecolor="white",
        ax=ax, cbar_kws={"label": "Solution quality  (1.0 = optimal, lower = worse)"},
    )
    ax.set_xlabel("Portfolio size", fontsize=12)
    ax.set_ylabel("Method", fontsize=12)
    ax.set_title(
        "Approximation Ratio — All Methods & Configurations\n"
        "(annotated values = f/f*; colour = 1/(f/f*))",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "fig7_summary_heatmap.png")
    plt.close(fig)
    print("  Saved fig7_summary_heatmap.png")


def plot_optimizer_convergence(
    results: list[ExperimentResult],
    output_dir: Path,
) -> None:
    """Fig 8 – Approx ratio vs number of COBYLA iterations for QAOA methods."""
    qaoa_results = [r for r in results if r.method in ("qaoa_aer", "qaoa_ibm")]
    if not qaoa_results:
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    seen: dict[tuple, bool] = {}
    for r in sorted(qaoa_results, key=lambda r: (r.method, r.n_assets, r.p)):
        key = (r.method, r.n_assets, r.p)
        if key in seen:
            continue
        seen[key] = True
        ax.scatter(
            r.n_iters, r.approx_ratio,
            marker=_method_marker(r.method),
            s=160,
            color=_method_color(r.method),
            label=f"{_method_label(r.method)}  n={r.n_assets}, p={r.p}",
            edgecolors="white", linewidths=0.8,
        )

    ax.axhline(1.0, linestyle=":", linewidth=1.5, color="#555555", label="Optimal (ratio=1)")
    ax.set_xlabel("COBYLA iterations", fontsize=12)
    ax.set_ylabel("Approximation ratio", fontsize=12)
    ax.set_title("Approximation Ratio vs. Optimizer Iterations", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "fig8_optimizer_iterations.png")
    plt.close(fig)
    print("  Saved fig8_optimizer_iterations.png")


def plot_runtime_scaling(df: pd.DataFrame, output_dir: Path) -> None:
    """Fig 9 – Runtime vs n_assets for all methods (line plot, log scale)."""
    methods_order = ["brute_force", "greedy", "sim_anneal", "qaoa_aer", "qaoa_ibm"]
    ns = sorted(df["n_assets"].unique())

    fig, ax = plt.subplots(figsize=(9, 5))

    for method in methods_order:
        mdf = df[df["method"] == method].copy()
        if mdf.empty:
            continue
        # For QAOA methods, use p=1
        if method in ("qaoa_aer", "qaoa_ibm"):
            mdf = mdf[mdf["p"] == 1]
        # One point per n (min runtime if duplicates)
        pts = mdf.groupby("n_assets")["runtime"].min().reset_index()
        if pts.empty:
            continue
        ax.plot(
            pts["n_assets"], pts["runtime"],
            marker=_method_marker(method), linewidth=2, markersize=9,
            color=_method_color(method), label=_method_label(method),
        )

    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Number of assets (n)", fontsize=12)
    ax.set_ylabel("Runtime (seconds, log scale)", fontsize=12)
    ax.set_title("Runtime Scaling vs. Problem Size", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fig9_runtime_scaling.png")
    plt.close(fig)
    print("  Saved fig9_runtime_scaling.png")


def generate_all_plots(
    results: list[ExperimentResult],
    output_dir: Path,
) -> None:
    df = to_dataframe(results)
    print("\nGenerating charts ...")
    plot_approx_ratio_vs_depth(df, output_dir)
    plot_runtime_comparison(df, output_dir)
    plot_solution_quality_bars(df, output_dir)
    plot_success_probability(df, output_dir)
    plot_measurement_distribution(results, output_dir)
    plot_efficient_frontier(results, output_dir)
    plot_summary_heatmap(df, output_dir)
    plot_optimizer_convergence(results, output_dir)
    plot_runtime_scaling(df, output_dir)


# ═══════════════════════════════════════════════════════════════════════════════
#  Print summary table
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(results: list[ExperimentResult]) -> None:
    df = to_dataframe(results)
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    cols = ["method", "n_assets", "p", "objective", "approx_ratio", "ret",
            "variance", "success_prob", "runtime"]
    print(df[cols].to_string(index=False, float_format="{:.4f}".format))
    print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4 benchmarking: QAOA vs classical methods",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Quick demo: n=4 only",
    )
    parser.add_argument(
        "--skip-ibm", action="store_true",
        help="Skip IBM Runtime experiments",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "benchmarks" / "results",
        help="Directory for results and figures",
    )
    args = parser.parse_args()

    sweep = FAST_SWEEP if args.fast else FULL_SWEEP
    skip_ibm = args.skip_ibm

    print("QAOA Finance — Phase 4 Benchmarking")
    print(f"Sweep: {sweep}")
    print(f"IBM Runtime: {'skipped' if skip_ibm else 'enabled' if IBM_AVAILABLE else 'unavailable'}")

    results = run_all_experiments(sweep, skip_ibm=skip_ibm, output_dir=args.output_dir)

    print_summary(results)
    save_results(results, args.output_dir)
    generate_all_plots(results, args.output_dir)

    print(f"\nAll done. Results and figures in: {args.output_dir}/")


if __name__ == "__main__":
    main()
