"""BTC/USD second dataset (see BTC_PREREGISTRATION.md). Builds 5-minute bars, the volatility-exceedance
task with h = 6, trains four agents on the training period, Platt-calibrates them on the calibration month,
and saves forecasts for the selection and test periods. usage: python btc_agents.py <btcusd_1min.csv>"""
import sys, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
OUT = Path(__file__).resolve().parent; H = 6
d = pd.read_csv(sys.argv[1]); d.index = pd.to_datetime(d.timestamp, unit='s')
b = pd.DataFrame({'close': d.close.resample('5min', label='right', closed='right').last(),
                  'high': d.high.resample('5min', label='right', closed='right').max(),
                  'low': d.low.resample('5min', label='right', closed='right').min(),
                  'vol': d.volume.resample('5min', label='right', closed='right').sum()}).dropna()
lc = np.log(b.close); r = lc.diff().abs(); rng_ = (b.high - b.low) / b.close
v = (lc.shift(-H) - lc).abs()                       # future 30-minute absolute return (label source)
ref = v.shift(H).rolling(288).mean()                # mean of the 288 most recent matured labels
y = (v > ref).astype(float); y[v.isna() | ref.isna()] = np.nan
F = pd.DataFrame(index=b.index)
for n in [1, 6, 24, 288]:
    F[f'rv{n}'] = np.log(r.rolling(n).mean() + 1e-7); F[f'rg{n}'] = np.log(rng_.rolling(n).mean() + 1e-7)
    F[f'lv{n}'] = np.log(b.vol.rolling(n).mean() + 1e-6)
hr = (b.index.hour + b.index.minute / 60) / 24 * 2 * np.pi
for j in [1, 2]: F[f's{j}'] = np.sin(j * hr); F[f'c{j}'] = np.cos(j * hr)
F['wkend'] = (b.index.dayofweek >= 5).astype(float); F['vratio'] = F.lv6 - F.lv288
cols = {'RV': [f'rv{n}' for n in [1, 6, 24, 288]], 'RANGE': [f'rg{n}' for n in [1, 6, 24, 288]],
        'GBT': [f'{p}{n}' for p in ['rv', 'rg', 'lv'] for n in [1, 6, 24, 288]] + ['s1', 'c1'],
        'TOD': ['s1', 'c1', 's2', 'c2', 'wkend', 'vratio']}
ok = F.notna().all(1) & y.notna(); F, y = F[ok], y[ok].astype(int)
def per(a, z): return (F.index >= a) & (F.index < z)
tr = per('2025-01-08', '2025-10-01'); ca = per('2025-10-01', '2025-11-01'); ev = per('2025-11-01', '2026-11-05')
def drop_last(m):  # remove the last H samples of a period (their labels reach into the next period)
    idx = np.where(m)[0]; m = m.copy(); m[idx[-H:]] = False; return m
tr, ca = drop_last(tr), drop_last(ca)
mu, sd = F[tr].mean(), F[tr].std() + 1e-9; Z = (F - mu) / sd
P = {}
for nm, c in cols.items():
    m = (HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, random_state=0) if nm == 'GBT'
         else LogisticRegression(max_iter=2000, C=1.0))
    m.fit(Z[tr][c].values, y[tr].values)
    def lg(X):
        q = np.clip(m.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6); return np.log(q / (1 - q))
    pl = LogisticRegression().fit(lg(Z[ca][c].values).reshape(-1, 1), y[ca].values)
    P[nm] = pl.predict_proba(lg(Z[ev][c].values).reshape(-1, 1))[:, 1]
ts = F.index[ev]; yy = y[ev].values
period = np.where(ts < pd.Timestamp('2026-01-01'), 0, 1)       # 0 = selection, 1 = test
dayid = ((ts - pd.Timestamp('2025-11-01')).days).astype(int)
np.savez_compressed(OUT / 'btc_forecasts.npz', y=yy, period=period, day=dayid, ts=ts.astype('int64'), **P)
print({k: round(float(np.mean((p - yy) ** 2)), 4) for k, p in P.items()}, 'base', round(float(yy.mean()), 3),
      'const', round(float(np.mean((yy - y[tr].mean()) ** 2)), 4), 'n sel', int((period == 0).sum()), 'n test', int((period == 1).sum()))
for k, p in P.items():
    for s in [0, 1]: print(k, 'period', s, round(float(np.mean((p[period == s] - yy[period == s]) ** 2)), 4))
