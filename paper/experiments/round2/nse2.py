"""NSE volatility task with the round-2 baselines (clean-rule tuning on 2016-2018, as in round 1).
Reads ../fair/nse_vol_forecasts.csv and ../fair/nse_vol_results.json (round-1 tuned LW/FT/FS/DWM).
Adds LW+CUSUM, stacking, FT with base-rate reference (FTB), LW/FT on online-recalibrated forecasts."""
import json, itertools, numpy as np, pandas as pd
from pathlib import Path
from engine2 import lw_fast, lw_cusum, stack, recal_fast, ft2, corrupt_matrix, brier
R = Path(__file__).resolve().parent; F = R.parent / 'fair'
df = pd.read_csv(F / 'nse_vol_forecasts.csv', index_col=0, parse_dates=True); r1 = json.loads((F / 'nse_vol_results.json').read_text())
AG = ['RV', 'RANGE', 'GBT', 'VIX']; SEL_END = pd.Timestamp('2019-01-01'); FAULT = pd.Timestamp('2021-01-01')
data = {a: (g[AG].values, g['y'].values.astype(int), g.index) for a, g in df.groupby('asset', sort=False)}
def fill(o, Q): p = o['p'].copy(); m = np.isnan(p); p[m] = Q[m].mean(1); return p
RT = [dict(decay=r, temperature=t) for r, t in itertools.product([.8, .9, .95, .98, .99, .995], [.2, .05, .02, .01, .005])]
lwb = r1['best']['LW']; RECLR = [1e-3, 1e-2, 3e-2]
def run(m, Q, y, kw, Rq=None):
    if m == 'LWC': return lw_cusum(Q, y, 1, **lwb, **kw)['p'], None
    if m == 'STK': return stack(Q, y, 1, **kw)['p'], None
    if m == 'LWR': return lw_fast(Rq, y, 1, **kw)['p'], None
    X = Rq if m == 'FTR' else Q; o = ft2(X, y, 1, ref='clim' if m == 'FTB' else 'half', **kw); return fill(o, X), o
G = {'LWC': [dict(k=k, H=h) for k, h in itertools.product([0, .02, .05, .1], [.5, 1, 2, 5, 10, 20])],
     'STK': [dict(lr=lr, l2=l2) for lr, l2 in itertools.product([1e-3, 3e-3, 1e-2, 3e-2], [0, 1e-3])], 'FTB': RT, 'LWR': RT, 'FTR': RT}
lr = min(RECLR, key=lambda l: np.mean([np.mean((recal_fast(P, y, 1, l)[idx < SEL_END] - y[idx < SEL_END, None]) ** 2) for P, y, idx in data.values()]))
recs = {a: recal_fast(P, y, 1, lr) for a, (P, y, idx) in data.items()}
best = {'REC_lr': lr}
for m, g in G.items():
    sc = [np.mean(np.concatenate([((run(m, P, y, kw, recs[a])[0] - y) ** 2)[idx < SEL_END] for a, (P, y, idx) in data.items()])) for kw in g]
    best[m] = g[int(np.argmin(sc))]; print(m, best[m], flush=True)
res = dict(best=best, scenarios={})
for sc in ['clean', 'inversion', 'stuck', 'overconfident']:
    for tgt in ([None] if sc == 'clean' else AG):
        rows = {}; det = {}; fq = {}
        for a, (P, y, idx) in data.items():
            k0 = int((idx < FAULT).sum()); fault = None if sc == 'clean' else (AG.index(tgt), sc, k0, None)
            Q, M = corrupt_matrix(P, fault); post = (idx >= FAULT) if fault else (idx >= SEL_END); Rq = recal_fast(Q, y, 1, lr)
            for m in G:
                p, o = run(m, Q, y, best[m], Rq); rows.setdefault(m, []).extend(((p[post] - y[post]) ** 2).tolist())
                if o is not None:
                    if fault: det.setdefault(m, []).append(next((tt - k0 for tt, k, x, n in o['trans'] if k == fault[0] and n == 'QUARANTINED' and tt >= k0), None))
                    fq[m] = fq.get(m, 0) + sum(1 for tt, k, x, n in o['trans'] if n == 'QUARANTINED' and post[tt] and (not fault or k != fault[0]))
        key = sc if tgt is None else f'{sc}:{tgt}'
        res['scenarios'][key] = dict(brier={m: float(np.mean(v)) for m, v in rows.items()}, detect=det, false_q=fq)
        print(key, {m: round(float(np.mean(v)), 4) for m, v in rows.items()}, {m: sum(x is not None for x in d) for m, d in det.items()}, fq, flush=True)
(R / 'r2_NSE.json').write_text(json.dumps(res, indent=1))
