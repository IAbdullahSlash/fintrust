"""NSE daily: next-day direction forecasts from 4 agents, yearly expanding walk-forward 2016-2025.
Input: nse_daily.csv from yfinance (auto_adjust=True). Output: nse_forecasts.csv.
y_t = 1 if close_{t+1} > close_t. Features use data up to close t only.
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
OUT = Path(__file__).resolve().parent
raw = pd.read_csv(sys.argv[1], header=[0, 1], index_col=0, skiprows=[2], parse_dates=True)
close = raw['Close']; vix = close['^INDIAVIX'].ffill()
assets = ['^NSEI', 'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']
rows = []
for asset in assets:
    c = close[asset].dropna(); r = np.log(c).diff()
    df = pd.DataFrame(index=c.index)
    df['r1'] = r; df['r5'] = r.rolling(5).sum(); df['r20'] = r.rolling(20).sum(); df['r60'] = r.rolling(60).sum()
    df['z5'] = (c - c.rolling(5).mean()) / c.rolling(20).std(); df['z20'] = (c - c.rolling(20).mean()) / c.rolling(20).std()
    df['vol20'] = r.rolling(20).std(); df['vol5'] = r.rolling(5).std()
    v = vix.reindex(c.index).ffill(); df['vix'] = np.log(v); df['dvix5'] = np.log(v).diff(5); df['vix_r5'] = df['vix'] * df['r5']
    df['y'] = (c.shift(-1) > c).astype(float); df.loc[c.shift(-1).isna(), 'y'] = np.nan
    df['ret_next'] = r.shift(-1)
    df = df.dropna(subset=['r60', 'z20', 'vix', 'dvix5'])
    feats = {'MOM': ['r1', 'r5', 'r20', 'r60'], 'REV': ['r1', 'z5', 'z20'],
             'GBT': ['r1', 'r5', 'r20', 'r60', 'z5', 'z20', 'vol5', 'vol20', 'vix', 'dvix5'], 'VIX': ['vix', 'dvix5', 'vix_r5', 'r5']}
    for year in range(2016, 2026):
        tr = df[(df.index.year < year) & df['y'].notna()]; te = df[df.index.year == year]
        tr = tr.iloc[:-1]  # last training label uses the first test-year close: drop it
        out = te[['y', 'ret_next']].copy(); out['asset'] = asset
        for name, f in feats.items():
            if name == 'GBT':
                m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.03, max_depth=3, min_samples_leaf=100, random_state=0)
            else:
                m = LogisticRegression(C=1.0, max_iter=1000)
            mu, sd = tr[f].mean(), tr[f].std() + 1e-12
            m.fit(((tr[f] - mu) / sd).values, tr['y'].astype(int).values)
            out[name] = m.predict_proba(((te[f] - mu) / sd).values)[:, 1]
        rows.append(out)
res = pd.concat(rows); res.index.name = 'date'; res.to_csv(OUT / 'nse_forecasts.csv')
res = res.dropna(subset=['y'])
for name in ['MOM', 'REV', 'GBT', 'VIX']:
    print(name, round(((res[name] - res['y']) ** 2).mean(), 5))
print('climatology-free 0.5:', 0.25, 'base rate', round(res['y'].mean(), 4), 'n', len(res))
