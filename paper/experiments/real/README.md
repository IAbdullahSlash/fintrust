# Market-data experiments (FI-2010 and NSE daily)

All scripts replay forecasts through the unmodified prototype/fintrust.py (via replay.py).

FI-2010 (NoAuction, Z-score files from the FI-2010 download, see data/README.md):
1. Convert the 10 day files (Train_CF_1 = day 1, Test_CF_k = day k+1) to day01.npy..day10.npy (149 x N arrays).
2. `python fi2010_agents.py <folder with dayNN.npy>`: trains LOGIT40, GBT, MLP on days 1-5, Platt on day 6, writes fi2010_forecasts.npz.
3. `python fi2010_dev_check.py`: policy selection on days 7-8 only.
4. `python fi2010_replay.py` (default thresholds) and `python fi2010_replay.py variant` (W=500, minimum=2000, auto-clear statistical quarantines).

NSE daily:
1. `OMP_NUM_THREADS=1 python nse_agents.py <path to nse_daily.csv>`: yearly walk-forward forecasts, writes nse_forecasts.csv.
2. `python nse_replay.py`: clean + MOM inversion from 2021, Brier and illustrative long/flat backtest (10 bps).

Results: fi2010_results*.json, fi2010_dev_check.json, fi2010_agents.json, nse_results.json. Raw data are not redistributed.
