"""
Fetch 2 months of daily adjusted close prices for the Magnificent 7
from Alpha Vantage and save each ticker as a CSV in data/.
"""

import os
import time
import datetime
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────
MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"]
MONTHS = 2
DATA_DIR = Path(__file__).parent
BASE_URL = "https://www.alphavantage.co/query"
# Alpha Vantage free tier: 25 req/day, ~5 req/min
REQUEST_DELAY = 15  # seconds between requests


def fetch_daily(ticker: str, api_key: str) -> pd.DataFrame:
    """Return a DataFrame of daily OHLCV + adjusted close for `ticker`."""
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "outputsize": "compact",   # last 100 trading days (~5 months)
        "datatype": "json",
        "apikey": api_key,
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if "Time Series (Daily)" not in data:
        note = data.get("Note") or data.get("Information") or data
        raise ValueError(f"Unexpected response for {ticker}: {note}")

    ts = data["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(ts, orient="index")
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df.columns = [c.split(". ", 1)[1] for c in df.columns]  # strip "1. " prefix
    df = df.astype(float)
    df.sort_index(inplace=True)
    return df


def filter_two_months(df: pd.DataFrame) -> pd.DataFrame:
    cutoff = datetime.date.today() - datetime.timedelta(days=MONTHS * 31)
    return df[df.index.date >= cutoff]


def main():
    load_dotenv(Path(__file__).parent.parent / ".env")
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    if not api_key or api_key == "your_api_key_here":
        raise RuntimeError("Set ALPHA_VANTAGE_API_KEY in .env before running.")

    for i, ticker in enumerate(MAG7):
        print(f"[{i+1}/{len(MAG7)}] Fetching {ticker}...", end=" ", flush=True)
        df = fetch_daily(ticker, api_key)
        df = filter_two_months(df)
        out = DATA_DIR / f"{ticker}.csv"
        df.to_csv(out)
        print(f"{len(df)} trading days → {out.name}")

        if i < len(MAG7) - 1:
            print(f"  Waiting {REQUEST_DELAY}s (rate limit)...")
            time.sleep(REQUEST_DELAY)

    print("\nDone. Files saved to data/")


if __name__ == "__main__":
    main()
