"""Refresh the daily price cache in data/prices/daily.parquet from yfinance.

python scripts/download_prices.py
"""

from riskpulse.ingestion.prices import download_prices

if __name__ == "__main__":
    df = download_prices()
    summary = df.groupby("symbol")["date"].agg(["count", "min", "max"])
    print(summary.to_string())
    print(f"rows={len(df)}")
