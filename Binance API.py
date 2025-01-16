# 1) Install python-binance if you haven't already:
#    pip install python-binance

import os
from binance.client import Client
from binance.enums import *
import pandas as pd

# 2) Provide your Binance API credentials (optional for public data),
#    but recommended if you're pulling a lot of data to get higher rate limits.
API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# 3) Initialize the Binance client
client = Client(API_KEY, API_SECRET)

# 4) Fetch 24-hour ticker data to find top 20 pairs by volume (vs. USDT)
tickers_24h = client.get_ticker()
usdt_pairs = [t for t in tickers_24h if t['symbol'].endswith("USDT")]
usdt_pairs_sorted = sorted(usdt_pairs, key=lambda x: float(x['volume']), reverse=True)
top_20_by_volume = usdt_pairs_sorted[:20]

# 5) Define the time range (4 years).
#    Adjust these strings as needed. Make sure they're in a format the client library supports.
start_str = "1 Jan, 2019"
end_str = "1 Jan, 2023"

# 6) For each of the top 20 pairs, fetch historical 1-minute klines
all_data = {}

for ticker_info in top_20_by_volume:
    symbol = ticker_info['symbol']
    print(f"Fetching 4 years of 1-minute data for {symbol}...")

    # get_historical_klines will automatically loop and fetch data in chunks
    klines = client.get_historical_klines(
        symbol,
        Client.KLINE_INTERVAL_1MINUTE,  # 1-minute interval
        start_str,
        end_str
    )

    # Convert raw klines to a Pandas DataFrame
    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close",
        "volume", "close_time", "quote_asset_volume",
        "number_of_trades", "taker_buy_base_volume",
        "taker_buy_quote_volume", "ignore"
    ])

    # Convert numeric columns from string to float
    numeric_cols = ["open", "high", "low", "close", "volume",
                    "quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    # Convert timestamps (milliseconds) to human-readable datetime
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    # Optionally, set "open_time" as the DataFrame index
    df.set_index("open_time", inplace=True)

    # Store DataFrame in a dictionary keyed by symbol
    all_data[symbol] = df

    # Print a status update
    print(f"  Fetched {len(df)} rows for {symbol}.")

# 7) Now, 'all_data' contains DataFrames of 1-minute prices
#    for each of the top 20 coins vs USDT over the last 4 years.
#    You can then save to CSV, Parquet, or perform further analysis.

# Example: Save one symbol’s data to CSV
# all_data["BTCUSDT"].to_csv("btc_usdt_1m_4y.csv")
