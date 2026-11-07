# Fair-baseline experiments (paper Sections VII-X)

Every method is tuned by the same rule: grid search on fault-free selection data, then frozen.

| Script | What it does | Output |
|---|---|---|
| `engine.py` | Vectorized FinTrust governor (`ft`), loss weighting (`lw`), Fixed Share, DWM, EW, exclusion oracle, fault injection | - |
| `verify.py` | Checks `engine.ft` against `prototype/fintrust.py` | prints match |
| `synth_fair.py` | Synthetic study: tune on seeds 100-104, test on seeds 0-29 | `synth_fair.json`, `synth_traces.json` |
| `synth_sens.py` | Window / consecutive-window sensitivity, 30 seeds | `synth_sens.json` |
| `fi_split_agents.py` | Trains LOGIT40, GBT, MLP for one FI-2010 split | `fi_S{1,2,3}_forecasts.npz` |
| `fi_fair.py` | FI-2010 fixed and randomized faults for one split | `fi_fair_S{1,2,3}.json` |
| `fi_hindsight_lw.py` | Stress test: LW tuned on the faulty test stream | `fi_hindsight_lw_S{1,2,3}.json` |
| `nse_vol.py` | NSE daily volatility task (needs `data/market/nse_daily.csv`) | `nse_vol_results.json` |
| `plots.py` | Figures `synth_band.pdf`, `fi_influence.pdf` | `../../figures/` |

Run order for FI-2010 (set `OMP_NUM_THREADS=1`):

```
python fi_split_agents.py <npy dir> S1 2 3 4 6
python fi_split_agents.py <npy dir> S2 4 5 6 8
python fi_split_agents.py <npy dir> S3 5 6 7 10
python fi_fair.py S1 ; python fi_fair.py S2 ; python fi_fair.py S3
python fi_hindsight_lw.py S1 ; python fi_hindsight_lw.py S2 ; python fi_hindsight_lw.py S3
```

The raw FI-2010 and Yahoo data are not redistributed. The `.npz` forecast files are derived model outputs.
