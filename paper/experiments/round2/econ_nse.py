"""Economic link (review item 9): does governance change a RISK outcome? NSE volatility task, 2021-2025.
Each of the six instruments is held long alone with ordinary volatility targeting, and the combined probability of a
high-volatility day scales exposure down through the paper's exposure multiplier:
   x_t = min(1, s* / s_t) * g_t,   g_t = clip(exp(-gamma * z_t), g_min, 1),   z_t = logit(p_t) - logit(pi_t)
s_t: annualised 20-day realised volatility; s* = 10%; pi_t: running base rate of matured labels; gamma = 1, g_min = 0.25.
These four values were fixed before any run and are not tuned (gamma = 0.5 and 2 are reported as a check).
Position x_t is set at the close of day t and earns the log return of day t+1; cost 5 bp per unit of turnover.
We report risk outcomes only (realised volatility, its gap to target, maximum drawdown, 5% expected shortfall,
mean exposure, turnover). No return or profitability claim is made.
usage: python econ_nse.py <nse_daily.csv>"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
from engine2 import lw_fast, fs_fast, recal_fast, ft2, ew, corrupt_matrix
R = Path(__file__).resolve().parent; F = R.parent / 'fair'
raw = pd.read_csv(sys.argv[1], header=[0, 1], index_col=0, skiprows=[2], parse_dates=True); close = raw['Close']
df = pd.read_csv(F / 'nse_vol_forecasts.csv', index_col=0, parse_dates=True)
AG = ['RV', 'RANGE', 'GBT', 'VIX']; FAULT = pd.Timestamp('2021-01-01'); TGT, GMIN, COST = .10, .25, 5e-4
r1 = json.loads((F / 'nse_vol_results.json').read_text())['best']; r2 = json.loads((R / 'r2_NSE.json').read_text())['best']
data = []
for a, g in df.groupby('asset', sort=False):
    c = close[a].dropna(); r = np.log(c).diff()
    nxt = r.shift(-1).reindex(g.index).values; vol = (r.rolling(20).std() * np.sqrt(252)).reindex(g.index).values
    data.append((a, g[AG].values, g['y'].values.astype(int), g.index, nxt, vol))

def fill(o, Q): p = o['p'].copy(); m = np.isnan(p); p[m] = Q[m].mean(1); return p
def combos(Q, M, y):
    Rq = recal_fast(Q, y, 1, r2['REC_lr'])
    return {'EW': ew(Q), 'LW': lw_fast(Q, y, 1, **r1['LW'])['p'], 'FS': fs_fast(Q, y, 1, **r1['FS'])['p'],
            'FT': fill(ft2(Q, y, 1, **r1['FT']), Q), 'FTB': fill(ft2(Q, y, 1, ref='clim', **r2['FTB']), Q),
            'FTR': fill(ft2(Rq, y, 1, **r2['FTR']), Rq), 'ORACLE': (Q * ~M).sum(1) / (~M).sum(1)}

def metrics(x, nxt, m):
    x, nxt = x[m], nxt[m]; ok = ~np.isnan(nxt); x, nxt = x[ok], nxt[ok]
    turn = np.abs(np.diff(np.r_[0., x])); pnl = x * nxt - COST * turn
    eq = np.cumsum(pnl); dd = float(np.max(np.maximum.accumulate(np.r_[0., eq])[1:] - eq))
    q = np.quantile(pnl, .05); es = float(-pnl[pnl <= q].mean())
    v = float(pnl.std() * np.sqrt(252))
    return dict(vol=v, gap=abs(v - TGT), maxdd=dd, es5=es, exposure=float(x.mean()), turnover=float(turn.sum() / (len(x) / 252)))

def lg(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
res = {}
for gamma in [1., .5, 2.]:
    out = {}
    for sc in ['clean', 'inversion', 'stuck', 'overconfident']:
        acc = {}
        for tgt in ([None] if sc == 'clean' else AG):
            for a, P, y, ix, nxt, vol in data:
                k0 = int((ix < FAULT).sum()); f = None if sc == 'clean' else (AG.index(tgt), sc, k0, None)
                Q, M = corrupt_matrix(P, f); m = ix >= FAULT
                cs = np.cumsum(np.r_[0., y.astype(float)]); pi = ((cs + .5) / (np.arange(len(y) + 1) + 1))[:-1]   # outcomes 0..t-1
                base = np.minimum(1., TGT / np.maximum(vol, 1e-6)); base = np.nan_to_num(base, nan=0.)
                acc.setdefault('VT', []).append(metrics(base, nxt, m))
                for nm, p in combos(Q, M, y).items():
                    g = np.clip(np.exp(-gamma * (lg(p) - lg(pi))), GMIN, 1.); acc.setdefault(nm, []).append(metrics(base * g, nxt, m))
        out[sc] = {nm: {k: float(np.mean([d[k] for d in v])) for k in v[0]} for nm, v in acc.items()}
        print(gamma, sc, {nm: (round(d['vol'], 4), round(d['maxdd'], 4), round(d['es5'] * 1e4, 1)) for nm, d in out[sc].items()}, flush=True)
    res[str(gamma)] = out
(R / 'econ_nse.json').write_text(json.dumps(res, indent=1))
print('done')
