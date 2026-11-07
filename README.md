# FinTrust

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24045037.svg)](https://doi.org/10.5281/zenodo.24045037)

Code and experiments for the paper **"FinTrust: Stateful Reliability Governance for Forecast Ensembles in Trading"**
by Ainaan Raza, Abdullah Azmi and Faizane Raza (Department of Computer Science and Engineering, Integral University, Lucknow).

FinTrust is a governance layer between several probabilistic forecasters and a trading decision. It scores each
forecaster on outcomes as they mature, moves it through an explicit lifecycle (shadow, active, probation, quarantine),
combines the admitted forecasts, and writes every decision to a hash-chained audit log.

> This is research code. It is not a trading system and makes no profitability claim. It has no broker
> connection, credentials or order management.

## Repository layout

```
prototype/                 governance core (standard library only) + 17 unit tests
paper/
  experiments/
    fair/                  vectorized engine, synthetic study, FI-2010 agents, NSE volatility task
    round2/                final study: all baselines, both tuning rules, BTC/USD, ablation, input faults,
                           pooled NSE evidence, risk outcome, table and figure generators
    real/                  earlier FI-2010 / NSE replay through the unmodified prototype (kept for reference;
                           real/replay.py is used by fair/verify.py)
    simulate.py ...        earlier synthetic study (first draft; simulate.py is still imported by fair/)
  figures/                 generated figures
  tab*.tex, tables_r2.tex  generated LaTeX table bodies
data/                      download / conversion scripts (raw data is not redistributed, see data/README.md)
```

Every experiment writes its result as JSON next to the script. All result files are committed, so the tables and
figures can be rebuilt in seconds without any market data.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10 or newer. Set `OMP_NUM_THREADS=1` for the agent-training scripts to get the same results as the paper.

## Quick check (no data needed, under a minute)

```bash
cd prototype && python -m unittest -v && cd ..                      # 17 tests
python paper/experiments/fair/verify.py                             # vectorized engine == prototype ("match")
cd paper/experiments/round2
python make_tables.py && python make_tables3.py                     # rebuild every table from the saved JSON
python plots2.py                                                    # rebuild Figs. 3 and 4
```

`make_tables*.py` overwrite `paper/tab*.tex`; `git diff` should show no change.

## Reproducing each table and figure

Paths are relative to `paper/experiments/`. "Saved" means the inputs are committed, so no raw data is needed.

| Paper item | Script(s) | Output | Inputs |
|---|---|---|---|
| Prototype behaviour (Sec. VI) | `prototype/test_fintrust.py` | test log | none |
| Table III, synthetic study | `fair/synth_fair.py`; `round2/synth_fs_ext.py` (Fixed Share extended grid) | `fair/synth_fair.json`, `fair/synth_traces.json`, `round2/synth_fs_ext.json` | none (simulated) |
| Table IV, fixed faults (FI-2010 S1-S3, BTC/USD) | `round2/run2.py S1`, `S2`, `S3`, `BTC`, then `round2/make_tables.py` | `round2/r2_*.json`, `paper/tabA.tex` | saved forecasts |
| Table V, random faults, both tuning rules | same as Table IV | `paper/tabB.tex` | saved forecasts |
| Fig. 3, tuning robustness grid | `round2/plots2.py` | `paper/figures/robust_grid.pdf` | `round2/r2_*.json` |
| Fig. 4, FI-2010 S1 influence trace | `round2/plots2.py` | `paper/figures/fi_influence_s1.pdf` | saved forecasts |
| Table VI, NSE volatility | `fair/nse_vol.py <nse_daily.csv>` (agents + round-1 tuning), `round2/nse2.py`, `round2/make_tables.py` | `fair/nse_vol_results.json`, `round2/r2_NSE.json`, `paper/tabNSE.tex` | `nse2.py` runs on the saved `fair/nse_vol_forecasts.csv` |
| Table VII, lifecycle ablation | `round2/ablate.py S1`, `S2`, `S3`, `BTC`, then `round2/make_tables3.py` | `round2/abl_*.json`, `paper/tabABL.tex` | saved forecasts |
| Table VIII, input-level faults | `round2/input_faults.py <fi2010 npy dir> <nse_daily.csv>`, then `make_tables3.py` | `round2/input_faults.json`, `paper/tabIF.tex` | raw FI-2010 + NSE (agents are retrained on corrupted inputs) |
| Pooled NSE evidence (Sec. XI text) | `round2/pool_nse.py` | `round2/pool_nse.json` | saved forecasts |
| Table IX, volatility targeting risk outcome | `round2/econ_nse.py <nse_daily.csv>`, then `make_tables3.py` | `round2/econ_nse.json`, `paper/tabECON.tex` | NSE prices |
| Table X (appendix), threshold sensitivity | `fair/synth_sens.py` | `fair/synth_sens.json` | `fair/synth_fair.json` |
| Detection delays and counts quoted in the text | `round2/summarize2.py S1 S2 S3 BTC` | printed | `round2/r2_*.json` |

Figs. 1 and 2 (architecture and lifecycle) are diagrams, not results.

### Rebuilding the agent forecasts from raw data

The saved forecast files can be regenerated from the public data (see `data/README.md` for downloads):

```bash
export OMP_NUM_THREADS=1
# FI-2010: convert the 10 day files, then train the three agents per split
mkdir -p data/raw/fi2010
python paper/experiments/real/fi2010_convert.py <path to BenchmarkDatasets/NoAuction> data/raw/fi2010
cd paper/experiments/fair
python fi_split_agents.py ../../../data/raw/fi2010 S1 2 3 4 6
python fi_split_agents.py ../../../data/raw/fi2010 S2 4 5 6 8
python fi_split_agents.py ../../../data/raw/fi2010 S3 5 6 7 10
cd ../../..

# BTC/USD (protocol fixed in advance: paper/experiments/round2/BTC_PREREGISTRATION.md)
python data/download_btc.py
python paper/experiments/round2/btc_agents.py data/raw/btcusd_bitstamp_1min.csv

# NSE daily (needs yfinance)
python data/download_nse.py
python paper/experiments/fair/nse_vol.py data/raw/nse_daily.csv
```

Then run the experiment scripts in the order of the table above (`run2.py` before `ablate.py`,
`make_tables.py` and `plots2.py`; `nse_vol.py` before `nse2.py`, `pool_nse.py` and `econ_nse.py`).
`run2.py` takes the longest (a few minutes to about an hour per dataset on a laptop CPU).

The downloaded BTC/USD file matches the SHA-256 in the pre-registration, and `btc_agents.py` reproduces the
committed `btc_forecasts.npz` exactly. Yahoo revises adjusted prices over time, so a fresh NSE download can
differ slightly from the copy used in the paper (downloaded November 2026).

## Using the governance core

```python
import sys; sys.path.insert(0, "prototype")
from fintrust import Governor, Config
```

See `prototype/README.md` for the API and its limits, and `paper/experiments/real/replay.py` for a complete
loop that feeds time-ordered forecasts and delayed outcomes through the governor.

## Citation

If you use this code, please cite the software release and the paper (see `CITATION.cff`):

> A. Raza, A. Azmi and F. Raza, *FinTrust: Stateful Reliability Governance for Forecast Ensembles in Trading (code and experiments)*, v1.0.0, Zenodo, 2026. doi:[10.5281/zenodo.24045037](https://doi.org/10.5281/zenodo.24045037)

## License

MIT, see `LICENSE`. The FI-2010, Bitstamp and Yahoo Finance data keep their own terms (see `data/README.md`).
