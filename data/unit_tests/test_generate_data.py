import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from data.generate_data import DEFAULT_TICKERS, TRADING_DAYS, load_assets

DATA_DIR = Path(__file__).parents[2] / "data"


class TestConstants:
    def test_trading_days_value(self):
        assert TRADING_DAYS == 252

    def test_default_tickers_length(self):
        assert len(DEFAULT_TICKERS) == 7

    def test_default_tickers_are_mag7(self):
        expected = {"AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"}
        assert set(DEFAULT_TICKERS) == expected

    def test_default_tickers_is_list(self):
        assert isinstance(DEFAULT_TICKERS, list)


class TestLoadAssets:
    def test_returns_two_arrays(self):
        result = load_assets(["AAPL", "MSFT"], DATA_DIR)
        assert len(result) == 2

    def test_mu_shape(self):
        mu, _ = load_assets(["AAPL", "MSFT"], DATA_DIR)
        assert mu.shape == (2,)

    def test_sigma_shape(self):
        _, Sigma = load_assets(["AAPL", "MSFT"], DATA_DIR)
        assert Sigma.shape == (2, 2)

    @pytest.mark.parametrize("n", [2, 4, 6, 7])
    def test_shapes_for_various_n(self, n):
        tickers = DEFAULT_TICKERS[:n]
        mu, Sigma = load_assets(tickers, DATA_DIR)
        assert mu.shape == (n,)
        assert Sigma.shape == (n, n)

    def test_sigma_is_symmetric(self):
        _, Sigma = load_assets(DEFAULT_TICKERS[:4], DATA_DIR)
        np.testing.assert_allclose(Sigma, Sigma.T, atol=1e-12)

    def test_sigma_diagonal_is_nonnegative(self):
        _, Sigma = load_assets(DEFAULT_TICKERS[:4], DATA_DIR)
        assert np.all(np.diag(Sigma) >= 0.0)

    def test_sigma_is_positive_semidefinite(self):
        _, Sigma = load_assets(DEFAULT_TICKERS[:4], DATA_DIR)
        eigenvalues = np.linalg.eigvalsh(Sigma)
        assert np.all(eigenvalues >= -1e-10)

    def test_mu_is_annualized(self):
        # Annualized returns for real equity data are typically in [-1, 5] range
        mu, _ = load_assets(DEFAULT_TICKERS[:4], DATA_DIR)
        assert np.all(np.abs(mu) < 10.0)

    def test_sigma_is_annualized(self):
        # Annualized variance for equities is typically < 5.0
        _, Sigma = load_assets(DEFAULT_TICKERS[:4], DATA_DIR)
        assert np.all(np.diag(Sigma) < 10.0)

    def test_returns_numpy_arrays(self):
        mu, Sigma = load_assets(["AAPL"], DATA_DIR)
        assert isinstance(mu, np.ndarray)
        assert isinstance(Sigma, np.ndarray)

    def test_default_tickers_used_when_none(self):
        mu, Sigma = load_assets(None, DATA_DIR)
        assert mu.shape == (7,)
        assert Sigma.shape == (7, 7)

    def test_missing_ticker_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_assets(["FAKE_TICKER_XYZ"], DATA_DIR)

    def test_single_ticker(self):
        mu, Sigma = load_assets(["AAPL"], DATA_DIR)
        assert mu.shape == (1,)
        assert Sigma.shape == (1, 1)

    def test_subset_matches_full_load(self):
        # Loading AAPL+MSFT together vs individually should give same mu
        mu_joint, _ = load_assets(["AAPL", "MSFT"], DATA_DIR)
        mu_aapl, _ = load_assets(["AAPL"], DATA_DIR)
        mu_msft, _ = load_assets(["MSFT"], DATA_DIR)
        assert abs(mu_joint[0] - mu_aapl[0]) < 1e-10
        assert abs(mu_joint[1] - mu_msft[0]) < 1e-10

    def test_with_synthetic_csv(self, tmp_path):
        # Build a minimal CSV with known returns and verify mu/Sigma
        dates = pd.date_range("2023-01-01", periods=6, freq="B")
        # Close prices that give exactly known log returns
        prices = np.array([100.0, 110.0, 121.0, 133.1, 146.41, 161.051])
        df = pd.DataFrame(
            {"open": prices, "high": prices, "low": prices,
             "close": prices, "volume": 1_000_000},
            index=dates,
        )
        df.index.name = "date"
        (tmp_path / "FAKE.csv").write_text(df.to_csv())

        mu, Sigma = load_assets(["FAKE"], tmp_path)
        # log returns ≈ log(1.1) for all 5 steps
        expected_daily_mu = np.log(1.1)
        assert abs(mu[0] - expected_daily_mu * TRADING_DAYS) < 1e-6
