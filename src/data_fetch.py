# data_loader.py

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import ccxt
import pandas as pd


def get_exchange() -> ccxt.Exchange:
    """
    Create and return a Binance US CCXT exchange instance.
    """
    exchange = ccxt.binanceus({
        "enableRateLimit": True,
    })
    exchange.load_markets()
    return exchange


def fetch_ohlcv_chunk(
    exchange: ccxt.Exchange,
    symbol: str = "BTC/USDT",
    timeframe: str = "1m",
    since_ms: Optional[int] = None,
    limit: int = 1000,
) -> list[list]:
    """
    Fetch one OHLCV chunk from Binance US.

    Returns rows in CCXT format:
    [timestamp, open, high, low, close, volume]
    """
    return exchange.fetch_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        since=since_ms,
        limit=limit,
    )


def load_historical_data(
    symbol: str = "BTC/USDT",
    timeframe: str = "1m",
    lookback_days: int = 7,
    pause_seconds: float = 0.25,
) -> pd.DataFrame:
    """
    Load historical OHLCV data for the given symbol/timeframe by paginating
    forward from (now - lookback_days) until the present.

    Parameters
    ----------
    symbol : str
        CCXT symbol, e.g. 'BTC/USDT'
    timeframe : str
        CCXT timeframe, e.g. '1m'
    lookback_days : int
        Number of days of history to request
    pause_seconds : float
        Small pause between paginated requests

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, open, high, low, close, volume
    """
    exchange = get_exchange()

    now = datetime.now(timezone.utc)
    start_dt = now - timedelta(days=lookback_days)
    since_ms = int(start_dt.timestamp() * 1000)

    all_rows: list[list] = []

    while True:
        chunk = fetch_ohlcv_chunk(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            since_ms=since_ms,
            limit=1000,
        )

        if not chunk:
            break

        all_rows.extend(chunk)

        last_ts = chunk[-1][0]

        # Stop if the exchange is no longer giving newer candles
        if len(chunk) < 1000:
            break

        # Advance by one candle to avoid duplicates
        since_ms = last_ts + 60_000

        # Safety: don't request into the future
        if since_ms >= int(now.timestamp() * 1000):
            break

        time.sleep(pause_seconds)

    if not all_rows:
        raise ValueError("No OHLCV data returned from Binance US.")

    df = pd.DataFrame(
        all_rows,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )

    # Deduplicate in case any overlapping rows slipped in
    df = df.drop_duplicates(subset=["timestamp"]).sort_values(
        "timestamp").reset_index(drop=True)

    # Convert timestamp and numeric columns
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().reset_index(drop=True)
    return df


def save_to_csv(
    df: pd.DataFrame,
    filename: str = "btcusdt_1m.csv",
) -> None:
    """
    Save dataframe to CSV.
    """
    df.to_csv(filename, index=False)


if __name__ == "__main__":
    df = load_historical_data(
        symbol="BTC/USDT",
        timeframe="1m",
        lookback_days=7,
    )
    print(df.head())
    print(df.tail())
    print(f"Rows fetched: {len(df)}")
    save_to_csv(df, "data/btcusdt_1m_7d_binanceus.csv")
