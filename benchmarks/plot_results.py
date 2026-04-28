#!/usr/bin/env python3
"""
Phase 4 visualization: six comparison plots across all solvers.

Reads benchmarks/results/results.csv and saves one figure per metric
plus a combined summary figure to benchmarks/figures/.

Plots
-----
1. Bitstring Binary Selection  – heatmap of 0/1 stock picks per solver
2. Approximation Ratio         – grouped bars (n=4, n=6)
3. Portfolio Return             – grouped bars
4. Portfolio Variance           – grouped bars
5. Runtime                      – grouped bars, log scale
6. Success Probability          – QAOA-only line chart vs depth p

Usage
-----
    python benchmarks/plot_results.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"

SOLVER_ORDER = [
    "Brute Force", "Greedy", "Sim. Annealing",
    "QAOA p=1", "QAOA p=2", "QAOA p=3",
]

SOLVER_COLORS = {
    "Brute Force":    "#1565C0",
    "Greedy":         "#2E7D32",
    "Sim. Annealing": "#E65100",
    "QAOA p=1":       "#6A1B9A",
    "QAOA p=2":       "#8E24AA",
    "QAOA p=3":       "#CE93D8",
}

N_COLORS = {4: "#1976D2", 6: "#388E3C"}
SAVE_DPI = 150


# ── Data loading ──────────────────────────────────────────────────────────────

def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["p"] = pd.to_numeric(df["p"], errors="coerce")

    def _label(row: pd.Series) -> str:
        if row["solver"] == "qaoa_aer":
            return f"QAOA p={int(row['p'])}"
        return {
            "brute_force":         "Brute Force",
            "greedy":              "Greedy",
            "simulated_annealing": "Sim. Annealing",
        }.get(row["solver"], row["solver"])

    df["solver_label"] = df.apply(_label, axis=1)
    return df


# ── Shared helpers ────────────────────────────────────────────────────────────

def _style(ax: plt.Axes, title: str, ylabel: str, legend: bool = True) -> None:
    ax.set_title(title, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.spines[["top", "right"]].set_visible(False)
    if legend:
        ax.legend(framealpha=0.85, fontsize=9)


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _grouped_bars(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    n_vals: list[int],
    log_scale: bool = False,
) -> None:
    """
    Draw grouped bars: x positions = SOLVER_ORDER present in df,
    one bar group per n value, coloured by N_COLORS.
    """
    solvers = [s for s in SOLVER_ORDER if s in df["solver_label"].unique()]
    x = np.arange(len(solvers))
    n_grps = len(n_vals)
    width = 0.7 / n_grps
    offsets = np.linspace(-(n_grps - 1) / 2, (n_grps - 1) / 2, n_grps) * width

    for offset, n in zip(offsets, n_vals):
        sub = df[df["n"] == n]
        vals = []
        for s in solvers:
            row = sub[sub["solver_label"] == s]
            v = row[metric].values[0] if not row.empty else np.nan
            vals.append(float(v) if pd.notna(v) else 0.0)
        ax.bar(
            x + offset, vals, width * 0.92,
            color=N_COLORS[n], alpha=0.85, label=f"n={n}",
            edgecolor="white", linewidth=0.6,
        )
        # Value labels above bars
        for xi, v in zip(x + offset, vals):
            ax.text(
                xi, v * (1.02 if v >= 0 else 0.98),
                f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=7, rotation=0,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(solvers, rotation=30, ha="right", fontsize=9)
    if log_scale:
        ax.set_yscale("log")


# ── Plot 1: Bitstring Binary Selection ───────────────────────────────────────

def plot_bitstring(df: pd.DataFrame) -> None:
    n_vals = sorted(df["n"].unique().astype(int))
    fig, axes = plt.subplots(1, len(n_vals), figsize=(5 * len(n_vals), 5))
    if len(n_vals) == 1:
        axes = [axes]

    cmap = matplotlib.colors.ListedColormap(["#EEF2F7", "#1565C0"])

    for ax, n in zip(axes, n_vals):
        sub = df[df["n"] == n].copy()
        tickers = sub["tickers"].iloc[0].split(",")
        solvers = [s for s in SOLVER_ORDER if s in sub["solver_label"].unique()]

        # Build binary matrix (rows=solvers, cols=tickers)
        matrix = np.zeros((len(solvers), len(tickers)), dtype=int)
        for r, s in enumerate(solvers):
            row = sub[sub["solver_label"] == s]
            if not row.empty:
                bits = row["bitstring"].values[0]
                for c, b in enumerate(str(bits).zfill(len(tickers))):
                    matrix[r, c] = int(b)

        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

        # Cell annotations
        for r in range(len(solvers)):
            for c in range(len(tickers)):
                color = "white" if matrix[r, c] == 1 else "#555"
                ax.text(c, r, str(matrix[r, c]),
                        ha="center", va="center", fontsize=11,
                        fontweight="bold", color=color)

        ax.set_xticks(range(len(tickers)))
        ax.set_xticklabels(tickers, fontsize=10)
        ax.set_yticks(range(len(solvers)))
        ax.set_yticklabels(solvers, fontsize=9)
        ax.set_title(f"n={n}  (k={n // 2} selected)", fontweight="bold", pad=8)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["Not selected", "Selected"])
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle("Fig 1 – Bitstring Binary Selection", fontweight="bold", fontsize=13, y=1.01)
    _save(fig, "fig1_bitstring_selection.png")


# ── Plot 2: Approximation Ratio ───────────────────────────────────────────────

def plot_approx_ratio(df: pd.DataFrame) -> None:
    n_vals = sorted(df["n"].unique().astype(int))
    fig, ax = plt.subplots(figsize=(9, 5))
    _grouped_bars(ax, df, "approx_ratio", n_vals)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="Optimal (1.0)")
    _style(ax, "Fig 2 – Approximation Ratio  f(x) / f*", "Approximation ratio  (1.0 = optimal)")
    _save(fig, "fig2_approx_ratio.png")


# ── Plot 3: Portfolio Return ──────────────────────────────────────────────────

def plot_portfolio_return(df: pd.DataFrame) -> None:
    n_vals = sorted(df["n"].unique().astype(int))
    fig, ax = plt.subplots(figsize=(9, 5))
    _grouped_bars(ax, df, "portfolio_return", n_vals)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    _style(ax, "Fig 3 – Portfolio Return  μᵀx", "Annualized log-return")
    _save(fig, "fig3_portfolio_return.png")


# ── Plot 4: Portfolio Variance ────────────────────────────────────────────────

def plot_portfolio_variance(df: pd.DataFrame) -> None:
    n_vals = sorted(df["n"].unique().astype(int))
    fig, ax = plt.subplots(figsize=(9, 5))
    _grouped_bars(ax, df, "portfolio_variance", n_vals)
    _style(ax, "Fig 4 – Portfolio Variance  xᵀΣx", "Annualized variance  (lower = less risk)")
    _save(fig, "fig4_portfolio_variance.png")


# ── Plot 5: Runtime ───────────────────────────────────────────────────────────

def plot_runtime(df: pd.DataFrame) -> None:
    n_vals = sorted(df["n"].unique().astype(int))
    fig, ax = plt.subplots(figsize=(9, 5))
    _grouped_bars(ax, df, "runtime", n_vals, log_scale=True)
    _style(ax, "Fig 5 – Wall-clock Runtime", "Runtime (seconds, log scale)")
    _save(fig, "fig5_runtime.png")


# ── Plot 6: Success Probability ───────────────────────────────────────────────

def plot_success_prob(df: pd.DataFrame) -> None:
    qaoa = df[df["solver"] == "qaoa_aer"].dropna(subset=["success_prob", "p"]).copy()
    if qaoa.empty:
        print("  Fig 6: no QAOA data, skipping.")
        return

    n_vals = sorted(qaoa["n"].unique().astype(int))
    p_vals = sorted(qaoa["p"].unique().astype(int))

    fig, ax = plt.subplots(figsize=(7, 5))

    for n in n_vals:
        sub = qaoa[qaoa["n"] == n].sort_values("p")
        ax.plot(
            sub["p"], sub["success_prob"],
            color=N_COLORS[n], marker="o", linewidth=2.2,
            markersize=8, label=f"n={n}",
        )
        for _, row in sub.iterrows():
            ax.annotate(
                f"{row['success_prob']:.3f}",
                (row["p"], row["success_prob"]),
                textcoords="offset points", xytext=(6, 4),
                fontsize=9, color=N_COLORS[n],
            )

    ax.set_xticks(p_vals)
    ax.set_xticklabels([f"p={p}" for p in p_vals])
    ax.set_ylim(0, max(qaoa["success_prob"].max() * 1.25, 0.05))
    _style(
        ax,
        "Fig 6 – QAOA Success Probability vs Circuit Depth",
        "Fraction of shots returning optimal bitstring",
    )
    ax.set_xlabel("QAOA depth p", fontsize=10)
    _save(fig, "fig6_success_probability.png")


# ── Combined summary figure ───────────────────────────────────────────────────

def plot_combined(df: pd.DataFrame) -> None:
    n_vals = sorted(df["n"].unique().astype(int))

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "Portfolio Optimization — Method Comparison",
        fontweight="bold", fontsize=15, y=1.01,
    )
    gs = fig.add_gridspec(3, 2, hspace=0.55, wspace=0.35)

    # ── 1. Bitstring (left col, top row, split by n) ─────────────────────
    ax_bit_outer = fig.add_subplot(gs[0, 0])
    ax_bit_outer.set_visible(False)
    cmap = matplotlib.colors.ListedColormap(["#EEF2F7", "#1565C0"])
    sub_axes = [
        fig.add_axes(
            [0.04 + i * 0.26, 0.73, 0.23, 0.22]
        )
        for i, _ in enumerate(n_vals)
    ]
    for ax_b, n in zip(sub_axes, n_vals):
        sub = df[df["n"] == n]
        tickers = sub["tickers"].iloc[0].split(",")
        solvers = [s for s in SOLVER_ORDER if s in sub["solver_label"].unique()]
        matrix = np.zeros((len(solvers), len(tickers)), dtype=int)
        for r, s in enumerate(solvers):
            row = sub[sub["solver_label"] == s]
            if not row.empty:
                bits = row["bitstring"].values[0]
                for c, b in enumerate(str(bits).zfill(len(tickers))):
                    matrix[r, c] = int(b)
        ax_b.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for r in range(len(solvers)):
            for c in range(len(tickers)):
                color = "white" if matrix[r, c] == 1 else "#555"
                ax_b.text(c, r, str(matrix[r, c]),
                          ha="center", va="center", fontsize=9, fontweight="bold", color=color)
        ax_b.set_xticks(range(len(tickers)))
        ax_b.set_xticklabels(tickers, fontsize=7)
        ax_b.set_yticks(range(len(solvers)))
        ax_b.set_yticklabels(solvers, fontsize=7)
        ax_b.set_title(f"Bitstring n={n}", fontweight="bold", fontsize=9, pad=5)
        ax_b.tick_params(length=0)
        for sp in ax_b.spines.values():
            sp.set_visible(False)

    # ── 2–5. Metric bar charts ────────────────────────────────────────────
    metrics = [
        ("approx_ratio",       "Fig 2 – Approximation Ratio",   "f(x)/f*",              False),
        ("portfolio_return",   "Fig 3 – Portfolio Return",       "μᵀx",                  False),
        ("portfolio_variance", "Fig 4 – Portfolio Variance",     "xᵀΣx",                 False),
        ("runtime",            "Fig 5 – Runtime",                "seconds (log)",         True),
    ]
    positions = [(0, 1), (1, 0), (1, 1), (2, 0)]

    for (row_i, col_i), (metric, title, ylabel, log) in zip(positions, metrics):
        ax = fig.add_subplot(gs[row_i, col_i])
        _grouped_bars(ax, df, metric, n_vals, log_scale=log)
        if metric == "approx_ratio":
            ax.axhline(1.0, color="red", linestyle="--", linewidth=1.0)
        _style(ax, title, ylabel)

    # ── 6. Success probability ────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    qaoa = df[df["solver"] == "qaoa_aer"].dropna(subset=["success_prob", "p"])
    if not qaoa.empty:
        p_vals = sorted(qaoa["p"].unique().astype(int))
        for n in n_vals:
            sub = qaoa[qaoa["n"] == n].sort_values("p")
            ax6.plot(sub["p"], sub["success_prob"],
                     color=N_COLORS[n], marker="o", linewidth=2, markersize=7, label=f"n={n}")
            for _, r in sub.iterrows():
                ax6.annotate(f"{r['success_prob']:.3f}", (r["p"], r["success_prob"]),
                             textcoords="offset points", xytext=(5, 3), fontsize=8,
                             color=N_COLORS[n])
        ax6.set_xticks(p_vals)
        ax6.set_xticklabels([f"p={p}" for p in p_vals])
        ax6.set_ylim(0, max(qaoa["success_prob"].max() * 1.3, 0.05))
    _style(ax6, "Fig 6 – Success Probability", "Fraction of shots → optimal")
    ax6.set_xlabel("QAOA depth p", fontsize=9)

    # Shared legend for n colours
    patches = [mpatches.Patch(color=N_COLORS[n], label=f"n={n}") for n in n_vals]
    fig.legend(handles=patches, loc="upper right", fontsize=10, framealpha=0.85)

    _save(fig, "fig0_combined_summary.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    csv_path = RESULTS_DIR / "results.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.\nRun  python benchmarks/run_experiments.py  first.")
        raise SystemExit(1)

    print(f"Loading {csv_path} ...")
    df = load(csv_path)
    print(f"  {len(df)} rows  |  n values: {sorted(df['n'].unique())}  |  "
          f"solvers: {sorted(df['solver'].unique())}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("\nGenerating figures ...")

    plot_bitstring(df)
    plot_approx_ratio(df)
    plot_portfolio_return(df)
    plot_portfolio_variance(df)
    plot_runtime(df)
    plot_success_prob(df)
    plot_combined(df)

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
