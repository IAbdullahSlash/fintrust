"""NSE volatility-direction task: will tomorrow's absolute log return exceed its trailing 20-day mean?
Same instruments, walk-forward scheme and agent families as ../real/nse_agents.py; then fair-baseline
replay (parameters chosen on 2016-2018 clean, test 2019-2025) with injected faults from 2021-01-01.
usage: OMP_NUM_THREADS=1 python nse_vol.py <nse_daily.csv>"""
import sys, json, itertools, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from engine import ft, lw, fixed_share, dwm, ew, oracle_excl, corrupt_matrix, brier
D = Path(__file__).resolve().parent
raw = pd.read_csv(sys.argv[1], header=[0, 1], index_col=0, skiprows=[2], parse_dates=True)
close, high, low = raw['Close'], raw['High'], raw['Low']; vix = close['^INDIAVIX'].ffill()
assets = ['^NSEI', 'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']
AG = ['RV', 'RANGE', 'GBT', 'VIX']
feats = {'RV': ['a1', 'a5', 'a20'], 'RANGE': ['rg1', 'rg5', 'rg20'], 'GBT': ['a1', 'a5', 'a20', 'rg1', 'rg5', 'r5', 'vix', 'dvix5'], 'VIX': ['vix', 'dvix5', 'vixrel']}
frames = {}
for asset in assets:
    c = close[asset].dropna(); r = np.log(c).diff(); a = r.abs(); df = pd.DataFrame(index=c.index)
    rg = np.log(high[asset] / low[asset]).reindex(c.index)
    m20 = a.rolling(20).mean()
    df['a1'] = a / m20; df['a5'] = a.rolling(5).mean() / m20; df['a20'] = np.log(m20)
    df['rg1'] = rg / rg.rolling(20).mean(); df['rg5'] = rg.rolling(5).mean() / rg.rolling(20).mean(); df['rg20'] = np.log(rg.rolling(20).mean())
    df['r5'] = r.rolling(5).sum(); v = np.log(vix.reindex(c.index).ffill()); df['vix'] = v; df['dvix5'] = v.diff(5)
    df['vixrel'] = v - np.log(m20 * np.sqrt(252) * 100)
    df['y'] = (a.shift(-1) > m20).astype(float); df.loc[a.shift(-1).isna(), 'y'] = np.nan
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=sum(feats.values(), []))
    out = []
    for year in range(2016, 2026):
        tr = df[(df.index.year < year) & df['y'].notna()].iloc[:-1]; te = df[df.index.year == year].dropna(subset=['y'])
        o = te[['y']].copy()
        for nm, f in feats.items():
            mdl = HistGradientBoostingClassifier(max_iter=150, learning_rate=.03, max_depth=3, min_samples_leaf=100, random_state=0) if nm == 'GBT' else LogisticRegression(max_iter=1000)
            mu, sd = tr[f].mean(), tr[f].std() + 1e-12; mdl.fit(((tr[f] - mu) / sd).values, tr['y'].astype(int).values)
            o[nm] = mdl.predict_proba(((te[f] - mu) / sd).values)[:, 1]
        out.append(o)
    frames[asset] = pd.concat(out)
pd.concat([f.assign(asset=a) for a, f in frames.items()]).to_csv(D / 'nse_vol_forecasts.csv')
SEL_END = pd.Timestamp('2019-01-01'); FAULT = pd.Timestamp('2021-01-01')
def fill(o, Q): p = o['p'].copy(); m = np.isnan(p); p[m] = Q[m].mean(1); return p
RHO = [.9, .95, .98, .99, .995]; TAU = [.2, .05, .02, .01, .005]
grids = {'LW': (lw, [dict(decay=r, temperature=t) for r, t in itertools.product(RHO, TAU)]),
         'FT': (ft, [dict(decay=r, temperature=t) for r, t in itertools.product(RHO, TAU)]),
         'FS': (fixed_share, [dict(eta=e, alpha=a) for e, a in itertools.product([1, 2, 5, 10, 20, 50], [0, 1e-3, 1e-2, .05])]),
         'DWM': (dwm, [dict(beta=b, theta=th, period=p) for b, th, p in itertools.product([.5, .8, .9, .95, .99], [.01, .05, .1], [1, 5, 20])])}
data = {a: (f[AG].values, f['y'].values.astype(int), f.index) for a, f in frames.items()}
best = {}
for m, (fn, g) in grids.items():
    sc = []
    for kw in g:
        v = []
        for a, (P, y, idx) in data.items():
            s = idx < SEL_END; o = fn(P, y, 1, **kw); p = fill(o, P) if fn is ft else o['p']; v.append(((p[s] - y[s]) ** 2).tolist())
        sc.append(np.mean(sum(v, [])))
    best[m] = g[int(np.argmin(sc))]; print(m, best[m], round(min(sc), 5), flush=True)
methods = {'EW': None, 'LW-paper': (lw, dict(decay=.9, temperature=.2)), 'LW-tuned': (lw, best['LW']), 'FS-tuned': (fixed_share, best['FS']),
           'DWM-tuned': (dwm, best['DWM']), 'FT-paper': (ft, {}), 'FT-tuned': (ft, best['FT'])}
res = dict(best=best, agents={}, scenarios={})
for a, (P, y, idx) in data.items():
    t = idx >= SEL_END; res['agents'][a] = {nm: brier(P[t, i], y[t]) for i, nm in enumerate(AG)}; res['agents'][a]['base_rate'] = float(y[t].mean())
for sc in ['clean', 'inversion', 'stuck', 'overconfident']:
    for tgt in ([None] if sc == 'clean' else AG):
        rows = {}; det = []; fq = 0; cov = []
        for a, (P, y, idx) in data.items():
            k0 = int((idx < FAULT).sum()); fault = None if sc == 'clean' else (AG.index(tgt), sc, k0, None)
            Q, M = corrupt_matrix(P, fault); post = (idx >= FAULT) if fault else (idx >= SEL_END)
            for m, spec in methods.items():
                if spec is None: p = ew(Q)
                else:
                    o = spec[0](Q, y, 1, **spec[1]); p = fill(o, Q) if spec[0] is ft else o['p']
                    if m == 'FT-tuned':
                        cov.append(float(o['cov'][post].mean()))
                        if fault: det.append(next((tt - k0 for tt, k, x, n in o['trans'] if k == fault[0] and n == 'QUARANTINED' and tt >= k0), None))
                        fq += sum(1 for tt, k, x, n in o['trans'] if n == 'QUARANTINED' and post[tt] and (not fault or k != fault[0]))
                rows.setdefault(m, []).extend(((p[post] - y[post]) ** 2).tolist())
            rows.setdefault('ORACLE', []).extend(((oracle_excl(Q, M)[post] - y[post]) ** 2).tolist())
        key = sc if tgt is None else f'{sc}:{tgt}'
        res['scenarios'][key] = dict(brier={m: float(np.mean(v)) for m, v in rows.items()}, detect_days=det, false_quarantines=fq, coverage=float(np.mean(cov)))
        print(key, {m: round(float(np.mean(v)), 4) for m, v in rows.items()}, det, fq, flush=True)
(D / 'nse_vol_results.json').write_text(json.dumps(res, indent=1))
