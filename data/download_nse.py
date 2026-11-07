"""Download the NSE daily series used in the paper (Section X) with yfinance.

Yahoo Finance terms do not allow redistribution, so the raw prices are not in this repository.
This rebuilds data/raw/nse_daily.csv in the same layout the scripts expect
(yfinance multi-ticker CSV, auto_adjust=True, 2012-01-02 to 2025-12-30).
Yahoo revises adjusted prices over time, so a fresh download can differ slightly from the paper's copy
(downloaded November 2026). The derived forecasts the paper used are kept in
paper/experiments/fair/nse_vol_forecasts.csv, so most NSE results rerun without this file.

usage: pip install yfinance ; python data/download_nse.py
"""
from pathlib import Path
import yfinance as yf

TICKERS = ['^NSEI', 'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', '^INDIAVIX']
OUT = Path(__file__).resolve().parent / 'raw' / 'nse_daily.csv'

if __name__ == '__main__':
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = yf.download(TICKERS, start='2012-01-01', end='2025-12-31', auto_adjust=True, progress=False)
    df.to_csv(OUT)
    print('wrote', OUT, df.shape)
