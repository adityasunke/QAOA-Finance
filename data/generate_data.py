"""
Real financial data loader for QAOA portfolio optimization.

Loads daily closing prices from CSV files in data/, computes log daily returns,
and derives annualized μ (mean return vector) and Σ (covariance matrix).
"""

import numpy as np
import pandas as pd
from pathlib import Path

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"]
DEFAULT_DATA_DIR = Path(__file__).parent
TRADING_DAYS = 252  # annualization factor


def load_assets(
    tickers: list[str] | None = None,
    data_dir: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load stock CSVs and compute annualized μ and Σ from log daily returns.

    Parameters
    ----------
    tickers  : list of ticker symbols; defaults to the Magnificent 7
    data_dir : directory containing <TICKER>.csv files; defaults to data/

    Returns
    -------
    mu    : (n,) annualized mean log-return vector
    Sigma : (n, n) annualized sample covariance matrix (PSD)
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    data_dir = Path(data_dir)

    closes = {}
    for ticker in tickers:
        path = data_dir / f"{ticker}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run data/fetch_stock_data.py first."
            )
        df = pd.read_csv(path, index_col="date", parse_dates=True)
        closes[ticker] = df["close"]

    prices = pd.DataFrame(closes).sort_index()
    log_returns = np.log(prices / prices.shift(1)).dropna()

    mu = log_returns.mean().values * TRADING_DAYS
    Sigma = log_returns.cov().values * TRADING_DAYS

    return mu, Sigma


if __name__ == "__main__":
    subsets = [
        ["AAPL", "MSFT", "GOOGL", "AMZN"],
        ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"],
        DEFAULT_TICKERS,
    ]
    for tickers in subsets:
        mu, Sigma = load_assets(tickers)
        n = len(tickers)
        print(f"n={n} {tickers}:")
        print(f"  mu    = {np.round(mu, 4)}")
        print(f"  Sigma diagonal = {np.round(np.diag(Sigma), 4)}")
