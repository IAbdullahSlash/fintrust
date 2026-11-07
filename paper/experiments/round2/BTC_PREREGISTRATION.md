# BTC/USD second dataset: protocol fixed before any result was computed

Written 2026-11-05, before any agent was trained or any combiner was run on this data.
Nothing below may change after results are seen; any deviation must be reported in the paper.

## Data
- Source: github.com/ff137/bitstamp-btcusd-minute-data, file `data/updates/btcusd_bitstamp_1min_latest.csv`
  (Bitstamp public API, CC BY 4.0, attribution "Bitstamp BTC/USD minute data, github.com/ff137/bitstamp-btcusd-minute-data").
- Downloaded 2026-11-05; 960,705 rows, 2025-01-07 00:01 to 2026-11-05 03:45 UTC;
  sha256 72b11143c6ddba74dd2604167bd1232b34e5bcbe5b44ed2af1f2b59247d24ddf. Raw file is not redistributed.
- Resampled to 5-minute bars (UTC, right-labelled): close = last, high = max, low = min, volume = sum.

## Task (overlapping labels)
- v_t = |log C_{t+6} - log C_t| (absolute 30-minute return), horizon h = 6 bars, one forecast per bar.
- y_t = 1 if v_t exceeds the mean of v over the 288 most recent matured labels (v_{t-6-287} .. v_{t-6}).
- Overlap factor m = h = 6.

## Agents (different inputs)
- RV: logistic regression on log mean |5-min return| over 1, 6, 24 and 288 bars.
- RANGE: logistic regression on log mean (high-low)/close over 1, 6, 24 and 288 bars.
- GBT: sklearn HistGradientBoosting on the RV, RANGE and log-volume (1, 6, 24, 288 bars) features plus hour-of-day sin/cos.
- TOD: logistic regression on hour-of-day sin/cos (two harmonics), weekend flag and log volume ratio (6 vs 288 bars).
- Features standardised on training data only. Each agent Platt-calibrated on the calibration period.

## Splits (UTC dates, purge h bars at each boundary)
- Train 2025-01-08 to 2025-09-30; calibration 2025-10; selection (warm-up and tuning) 2025-11-01 to 2025-12-31;
  locked test 2026-01-01 to 2026-11-04.

## Governance thresholds (not tuned)
- W = 50m = 300 outcomes, evidence minimum 200m = 1200, 3 consecutive windows, bad if excess > 0.05, good if < 0,
  probation cap 0.10, prior mass 20, two-tier quarantine (statistical quarantines self-clear).

## Methods and tuning
- Same methods, grids and two tuning rules (clean selection data; fault-aware selection data) as the FI-2010 round-2 runs.
- Faults: fixed fault on the best single agent (lowest selection Brier) from the first test bar (transient: first 30 days),
  plus 10 random draws per type (inversion, stuck at 0.95, overconfident), target uniform over agents,
  onset uniform in the first half of the test period, numpy default_rng(42).
- Primary comparison: FT vs tuned LW. Intervals: paired block bootstrap over days (288-bar blocks).
