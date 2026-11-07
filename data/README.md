# Data

No raw market data is stored in this repository. Each source has its own terms, so the scripts below
fetch or convert the data on your machine into `data/raw/` (ignored by git).

| Dataset | Used for | How to get it | Terms |
|---|---|---|---|
| FI-2010 limit order book (Ntakaris et al., 2018), NoAuction Z-score release | Paper Sections IX, XI (FI-2010 splits S1-S3) | Download from Fairdata/Etsin: https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649 and unzip, then run `python paper/experiments/real/fi2010_convert.py <BenchmarkDatasets/NoAuction> data/raw/fi2010` | As stated on the Etsin page; cite Ntakaris et al. (2018) |
| Bitstamp BTC/USD 1-minute bars | Section IX (BTC/USD, fixed protocol) | `python data/download_btc.py` (checks the SHA-256 recorded in `paper/experiments/round2/BTC_PREREGISTRATION.md`) | CC BY 4.0, "Bitstamp BTC/USD minute data, github.com/ff137/bitstamp-btcusd-minute-data" |
| NSE daily: NIFTY 50, India VIX, RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK, 2012-2025 | Sections X, XI | `pip install yfinance && python data/download_nse.py` | Yahoo Finance terms; personal and research use, no redistribution |

`fi2010_convert.py` writes `day01.npy` ... `day10.npy` (149 x N each). Day 1 is `Train_Dst_NoAuction_ZScore_CF_1.txt`
and day k+1 is `Test_Dst_NoAuction_ZScore_CF_k.txt`. Create the output folder first.

## What is included instead

Model outputs (not market data) are committed so every table can be rebuilt without the raw files:

- `paper/experiments/fair/fi_S{1,2,3}_forecasts.npz`: out-of-sample probabilities of the three FI-2010 agents and the binary label, per split.
- `paper/experiments/round2/btc_forecasts.npz`: probabilities of the four BTC/USD agents and the label.
- `paper/experiments/fair/nse_vol_forecasts.csv`: probabilities of the four NSE agents and the binary volatility label (no prices).
- `paper/experiments/real/fi2010_forecasts.npz`: forecasts for the earlier single-split FI-2010 run (used by `verify.py`).

Regenerating `btc_forecasts.npz` from the downloaded file was checked to reproduce the committed file exactly.
