"""
Tests for fetch_stock_data.py.

fetch_daily() makes live HTTP calls, so all tests use unittest.mock to
intercept requests.get and return controlled JSON payloads.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from data.fetch_stock_data import BASE_URL, MAG7, REQUEST_DELAY, fetch_daily

# ---------------------------------------------------------------------------
# Shared fixture: a minimal valid Alpha Vantage "Time Series (Daily)" response
# ---------------------------------------------------------------------------

VALID_RESPONSE = {
    "Time Series (Daily)": {
        "2024-01-03": {
            "1. open": "185.00",
            "2. high": "187.00",
            "3. low": "184.00",
            "4. close": "186.50",
            "5. volume": "50000000",
        },
        "2024-01-02": {
            "1. open": "182.00",
            "2. high": "186.00",
            "3. low": "181.00",
            "4. close": "185.20",
            "5. volume": "45000000",
        },
        "2024-01-01": {
            "1. open": "180.00",
            "2. high": "183.00",
            "3. low": "179.00",
            "4. close": "182.00",
            "5. volume": "40000000",
        },
    }
}


def _mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data
    mock.raise_for_status.return_value = None
    return mock


class TestConstants:
    def test_mag7_has_seven_tickers(self):
        assert len(MAG7) == 7

    def test_mag7_contains_expected_tickers(self):
        assert set(MAG7) == {"AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"}

    def test_base_url_is_alpha_vantage(self):
        assert "alphavantage" in BASE_URL.lower()

    def test_request_delay_is_positive(self):
        assert REQUEST_DELAY > 0


class TestFetchDaily:
    @patch("data.fetch_stock_data.requests.get")
    def test_returns_dataframe(self, mock_get):
        mock_get.return_value = _mock_response(VALID_RESPONSE)
        df = fetch_daily("AAPL", "fake_key")
        assert isinstance(df, pd.DataFrame)

    @patch("data.fetch_stock_data.requests.get")
    def test_row_count_matches_time_series(self, mock_get):
        mock_get.return_value = _mock_response(VALID_RESPONSE)
        df = fetch_daily("AAPL", "fake_key")
        assert len(df) == 3

    @patch("data.fetch_stock_data.requests.get")
    def test_columns_have_numeric_prefix_stripped(self, mock_get):
        mock_get.return_value = _mock_response(VALID_RESPONSE)
        df = fetch_daily("AAPL", "fake_key")
        # Original keys are "1. open", "2. high", etc.
        # After stripping: "open", "high", "low", "close", "volume"
        assert "close" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "volume" in df.columns
        assert not any(c.startswith("1.") for c in df.columns)

    @patch("data.fetch_stock_data.requests.get")
    def test_values_are_float(self, mock_get):
        mock_get.return_value = _mock_response(VALID_RESPONSE)
        df = fetch_daily("AAPL", "fake_key")
        assert df["close"].dtype == float

    @patch("data.fetch_stock_data.requests.get")
    def test_index_is_sorted_ascending(self, mock_get):
        mock_get.return_value = _mock_response(VALID_RESPONSE)
        df = fetch_daily("AAPL", "fake_key")
        assert df.index.is_monotonic_increasing

    @patch("data.fetch_stock_data.requests.get")
    def test_index_name_is_date(self, mock_get):
        mock_get.return_value = _mock_response(VALID_RESPONSE)
        df = fetch_daily("AAPL", "fake_key")
        assert df.index.name == "date"

    @patch("data.fetch_stock_data.requests.get")
    def test_correct_close_value(self, mock_get):
        mock_get.return_value = _mock_response(VALID_RESPONSE)
        df = fetch_daily("AAPL", "fake_key")
        assert abs(df["close"].iloc[-1] - 186.50) < 1e-9

    @patch("data.fetch_stock_data.requests.get")
    def test_missing_time_series_key_raises_value_error(self, mock_get):
        mock_get.return_value = _mock_response(
            {"Note": "API call frequency exceeded."}
        )
        with pytest.raises(ValueError, match="Unexpected response"):
            fetch_daily("AAPL", "fake_key")

    @patch("data.fetch_stock_data.requests.get")
    def test_http_error_propagates(self, mock_get):
        import requests as req_lib
        mock = _mock_response({})
        mock.raise_for_status.side_effect = req_lib.HTTPError("404 Not Found")
        mock_get.return_value = mock
        with pytest.raises(req_lib.HTTPError):
            fetch_daily("AAPL", "fake_key")

    @patch("data.fetch_stock_data.requests.get")
    def test_request_uses_correct_base_url(self, mock_get):
        mock_get.return_value = _mock_response(VALID_RESPONSE)
        fetch_daily("MSFT", "my_key")
        call_args = mock_get.call_args
        assert call_args[0][0] == BASE_URL

    @patch("data.fetch_stock_data.requests.get")
    def test_request_includes_ticker_symbol(self, mock_get):
        mock_get.return_value = _mock_response(VALID_RESPONSE)
        fetch_daily("TSLA", "my_key")
        params = mock_get.call_args[1]["params"]
        assert params["symbol"] == "TSLA"

    @patch("data.fetch_stock_data.requests.get")
    def test_request_includes_api_key(self, mock_get):
        mock_get.return_value = _mock_response(VALID_RESPONSE)
        fetch_daily("NVDA", "secret_key")
        params = mock_get.call_args[1]["params"]
        assert params["apikey"] == "secret_key"
