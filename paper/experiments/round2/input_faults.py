"""Input-level faults (review item 5): the fault corrupts what goes INTO one model, and the model's own
(clean-calibrated) output is what the combiners see. Nothing is re-tuned: every method uses its clean-rule and
fault-rule parameters from r2_<split>.json / r2_NSE.json and ../fair/nse_vol_results.json.

FI-2010 (S1-S3; the agents are retrained exactly as in ../fair/fi_split_agents.py, and the clean forecasts are
checked against the saved ones). From the first test sample, on one agent at a time:
  delayed  the agent's feature row is 500 samples old (a lagging feed / stale cache)
  dropped  the 20 order-book volume columns (features 1,3,...,39) are zero (a dropped column block)
  scaled   the same volume columns are multiplied by 10 (a unit bug)
NSE volatility task (2021-2025, all six instruments):
  decay:<agent>  that agent is not refitted after 2020 (models trained on data up to 2020 are used to 2025)
  stale_vix      India VIX is frozen at its last 2020 value from 2021 on; hits GBT and VIX together
usage: OMP_NUM_THREADS=1 python input_faults.py <fi2010 dayNN.npy dir> <nse_daily.csv>"""
import sys, json, itertools, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from engine2 import lw_fast, lw_cusum, stack, recal_fast, fs_fast, ft2, dwm, ew, brier
R = Path(__file__).resolve().parent; F = R.parent / 'fair'; SRC = Path(sys.argv[1]); NSE = sys.argv[2]
VOL = list(range(1, 40, 2))

def fill(o, Q): p = o['p'].copy(); m = np.isnan(p); p[m] = Q[m].mean(1); return p

def combine(m, Q, y, H, kw, lwkw, R_, ftw):
    if m == 'LW': return lw_fast(Q, y, H, **kw)['p'], None
    if m == 'FS': return fs_fast(Q, y, H, **kw)['p'], None
    if m == 'DWM': return dwm(Q, y, H, **kw)['p'], None
    if m == 'STK': return stack(Q, y, H, **kw)['p'], None
    if m == 'LWC': return lw_cusum(Q, y, H, **lwkw, **kw)['p'], None
    if m == 'LWR': return lw_fast(R_, y, H, **kw)['p'], None
    X = R_ if m == 'FTR' else Q; o = ft2(X, y, H, ref='clim' if m == 'FTB' else 'half', **ftw, **kw); return fill(o, X), o

ORDER = ['LW', 'FS', 'DWM', 'STK', 'LWC', 'FT', 'FTB', 'FTR']

def evaluate(Q, y, H, a, bad, best, ftw, seg=None):
    """Q: corrupted matrix, a: onset index, bad: list of faulty agents, seg: optional array of segment starts
    (NSE: each instrument is its own stream; the combiners are run per segment and losses pooled)."""
    segs = [(0, len(y))] if seg is None else seg; out = {}; aux = {}
    for mode in ['clean', 'fault']:
        b = best[mode]; L = {m: [] for m in ORDER}; L['EW'] = []; L['ORACLE'] = []; det = []; hq = 0
        for s0, s1 in segs:
            q, yy = Q[s0:s1], y[s0:s1]; aa = a[s0] if isinstance(a, dict) else a - s0
            Rq = recal_fast(q, yy, H, b['REC_lr'])
            for m in ORDER:
                p, o = combine(m, q, yy, H, b[m], b['LW'], Rq, ftw); L[m].append((p[aa:] - yy[aa:]) ** 2)
                if m == 'FT' and o is not None:
                    det.append([next((t - aa for t, k, _, n in o['trans'] if k == j and n == 'QUARANTINED' and t >= aa), None) for j in bad])
                    hq += sum(1 for t, k, _, n in o['trans'] if n == 'QUARANTINED' and t >= aa and k not in bad)
            L['EW'].append((ew(q)[aa:] - yy[aa:]) ** 2)
            good = [k for k in range(q.shape[1]) if k not in bad]; L['ORACLE'].append((q[aa:, good].mean(1) - yy[aa:]) ** 2)
        for m, v in L.items():
            if mode == 'fault' and m in ('EW', 'ORACLE'): continue
            out[m if m in ('EW', 'ORACLE') else f'{m}|{mode}'] = float(np.mean(np.concatenate(v)))
        aux[f'FT|{mode}'] = dict(detect=det, healthy_q=hq)
    return out, aux

res = {}
# ------------------------------ FI-2010 ------------------------------
H = 10
days = {d: np.load(SRC / f'day{d:02d}.npy') for d in range(1, 11)}
X = {d: days[d][:144].T.astype(np.float32) for d in days}; Y = {d: (days[d][148] == 1).astype(int) for d in days}
for name, (tr_end, cal, d0, d1) in {'S1': (2, 3, 4, 6), 'S2': (4, 5, 6, 8), 'S3': (5, 6, 7, 10)}.items():
    Xtr = np.vstack([X[d][:-H] for d in range(1, tr_end + 1)]); ytr = np.concatenate([Y[d][:-H] for d in range(1, tr_end + 1)])
    Xca, yca = X[cal][:-H], Y[cal][:-H]
    rng = np.random.default_rng(0); sub = rng.choice(len(ytr), size=min(80000, len(ytr)), replace=False)
    models = {'LOGIT40': (LogisticRegression(max_iter=2000, C=0.1), slice(0, 40)),
              'GBT': (HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, random_state=0), slice(0, 144)),
              'MLP': (MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=60, early_stopping=True, random_state=0), slice(0, 144))}
    fd = list(range(d0, d1 + 1)); Xte = np.vstack([X[d] for d in fd]); y = np.concatenate([Y[d] for d in fd])
    AG = list(models); preds = {}
    for nm, (m, cols) in models.items():
        xs, ys = (Xtr, ytr) if nm == 'LOGIT40' else (Xtr[sub], ytr[sub]); m.fit(xs[:, cols], ys)
        def lg(Z, m=m, cols=cols):
            q = np.clip(m.predict_proba(Z[:, cols])[:, 1], 1e-6, 1 - 1e-6); return np.log(q / (1 - q))
        pl = LogisticRegression().fit(lg(Xca).reshape(-1, 1), yca); preds[nm] = lambda Z, lg=lg, pl=pl: pl.predict_proba(lg(Z).reshape(-1, 1))[:, 1]
    P = np.stack([preds[a](Xte) for a in AG], 1)
    z = np.load(F / f'fi_{name}_forecasts.npz'); dev = max(float(np.abs(P[:, i] - z[a]).max()) for i, a in enumerate(AG))
    r2 = json.loads((R / f'r2_{name}.json').read_text()); test0 = r2['n_sel']; best = r2['best']
    out = dict(max_dev_from_saved=dev, scen={})
    print(name, 'retrained, max deviation from saved forecasts', dev, flush=True)
    for kind in ['delayed', 'dropped', 'scaled']:
        Xc = Xte.copy()
        if kind == 'delayed': Xc[test0 + 500:] = Xte[test0:-500]
        elif kind == 'dropped': Xc[test0:, VOL] = 0.
        else: Xc[test0:, VOL] *= 10.
        for j, a in enumerate(AG):
            Q = P.copy(); Q[test0:, j] = preds[a](Xc[test0:])
            o, aux = evaluate(Q, y, H, test0, [j], best, dict(window=50 * H, minimum=200 * H))
            ag = brier(Q[test0:, j], y[test0:]); ag0 = brier(P[test0:, j], y[test0:])
            out['scen'][f'{kind}:{a}'] = dict(brier=o, aux=aux, agent_before=ag0, agent_after=ag)
            print(name, kind, a, f'agent {ag0:.4f}->{ag:.4f}', {k: round(v, 4) for k, v in o.items()}, aux['FT|clean'], flush=True)
    res[name] = out
# ------------------------------ NSE ------------------------------
raw = pd.read_csv(NSE, header=[0, 1], index_col=0, skiprows=[2], parse_dates=True)
close, high, low = raw['Close'], raw['High'], raw['Low']; vix_all = close['^INDIAVIX'].ffill()
assets = ['^NSEI', 'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']; AG = ['RV', 'RANGE', 'GBT', 'VIX']
feats = {'RV': ['a1', 'a5', 'a20'], 'RANGE': ['rg1', 'rg5', 'rg20'], 'GBT': ['a1', 'a5', 'a20', 'rg1', 'rg5', 'r5', 'vix', 'dvix5'], 'VIX': ['vix', 'dvix5', 'vixrel']}
FAULT = pd.Timestamp('2021-01-01')
def features(asset, vix):
    c = close[asset].dropna(); r = np.log(c).diff(); a = r.abs(); df = pd.DataFrame(index=c.index)
    rg = np.log(high[asset] / low[asset]).reindex(c.index); m20 = a.rolling(20).mean()
    df['a1'] = a / m20; df['a5'] = a.rolling(5).mean() / m20; df['a20'] = np.log(m20)
    df['rg1'] = rg / rg.rolling(20).mean(); df['rg5'] = rg.rolling(5).mean() / rg.rolling(20).mean(); df['rg20'] = np.log(rg.rolling(20).mean())
    df['r5'] = r.rolling(5).sum(); v = np.log(vix.reindex(c.index).ffill()); df['vix'] = v; df['dvix5'] = v.diff(5)
    df['vixrel'] = v - np.log(m20 * np.sqrt(252) * 100)
    df['y'] = (a.shift(-1) > m20).astype(float); df.loc[a.shift(-1).isna(), 'y'] = np.nan
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=sum(feats.values(), []))
def mk(nm): return (HistGradientBoostingClassifier(max_iter=150, learning_rate=.03, max_depth=3, min_samples_leaf=100, random_state=0)
                    if nm == 'GBT' else LogisticRegression(max_iter=1000))
vix_stale = vix_all.copy(); vix_stale[vix_stale.index >= FAULT] = vix_all[vix_all.index < FAULT].iloc[-1]
saved = pd.read_csv(F / 'nse_vol_forecasts.csv', index_col=0, parse_dates=True)
streams = {}  # scenario -> list of (Q, y, onset) per asset
dev = 0.
for asset in assets:
    df = features(asset, vix_all); dfs = features(asset, vix_stale)
    rows = {k: [] for k in ['clean', 'stale_vix'] + [f'decay:{a}' for a in AG]}; ys = []
    frozen = {}
    for year in range(2016, 2026):
        tr = df[(df.index.year < year) & df['y'].notna()].iloc[:-1]; te = df[df.index.year == year].dropna(subset=['y'])
        tes = dfs.loc[te.index]; ys.append(te['y'].values)
        cur = {}
        for nm, f in feats.items():
            mu, sd = tr[f].mean(), tr[f].std() + 1e-12; m = mk(nm).fit(((tr[f] - mu) / sd).values, tr['y'].astype(int).values); cur[nm] = (m, mu, sd)
            if year == 2021: frozen[nm] = (m, mu, sd)   # trained on data up to 2020
        pr = lambda spec, frame, f: spec[0].predict_proba(((frame[f] - spec[1]) / spec[2]).values)[:, 1]
        base = {nm: pr(cur[nm], te, f) for nm, f in feats.items()}
        rows['clean'].append(np.stack([base[a] for a in AG], 1))
        st = dict(base)
        if year >= 2021:
            for nm in ['GBT', 'VIX']: st[nm] = pr(cur[nm], tes, feats[nm])
        rows['stale_vix'].append(np.stack([st[a] for a in AG], 1))
        for j, nm in enumerate(AG):
            d = dict(base)
            if year > 2021: d[nm] = pr(frozen[nm], te, feats[nm])
            rows[f'decay:{nm}'].append(np.stack([d[a] for a in AG], 1))
    y = np.concatenate(ys).astype(int); idx = df[(df.index.year >= 2016) & (df.index.year <= 2025)].dropna(subset=['y']).index
    sv = saved[saved.asset == asset][AG].values; Pc = np.vstack(rows['clean']); dev = max(dev, float(np.abs(sv - Pc).max()))
    k0 = int((idx < FAULT).sum())
    for sc, lst in rows.items(): streams.setdefault(sc, []).append((np.vstack(lst), y, k0))
print('NSE max deviation from saved forecasts', dev, flush=True)
r1 = json.loads((F / 'nse_vol_results.json').read_text())['best']; r2n = json.loads((R / 'r2_NSE.json').read_text())['best']
bn = {**{m: r1[m] for m in ['LW', 'FS', 'DWM', 'FT']}, **{m: r2n[m] for m in ['LWC', 'STK', 'FTB', 'LWR', 'FTR']}, 'REC_lr': r2n['REC_lr']}
bestN = {'clean': bn, 'fault': bn}   # NSE has only the clean rule
res['NSE'] = dict(max_dev_from_saved=dev, scen={})
for sc, lst in streams.items():
    if sc == 'clean': continue
    bad = [AG.index('GBT'), AG.index('VIX')] if sc == 'stale_vix' else [AG.index(sc.split(':')[1])]
    Q = np.vstack([q for q, _, _ in lst]); y = np.concatenate([yy for _, yy, _ in lst])
    starts = np.cumsum([0] + [len(yy) for _, yy, _ in lst]); segs = list(zip(starts[:-1], starts[1:]))
    onset = {int(s0): int(k0) for (s0, _), (_, _, k0) in zip(segs, lst)}
    o, aux = evaluate(Q, y, 1, onset, bad, bestN, {}, seg=segs); o = {k: v for k, v in o.items() if not k.endswith('|fault')}
    post = np.concatenate([np.arange(s0 + k0, s1) for (s0, s1), (_, _, k0) in zip(segs, lst)])
    clean = np.vstack([q for q, _, _ in streams['clean']])
    ag = {AG[j]: [brier(clean[post, j], y[post]), brier(Q[post, j], y[post])] for j in bad}
    res['NSE']['scen'][sc] = dict(brier=o, aux={'FT|clean': aux['FT|clean']}, agents=ag)
    print('NSE', sc, ag, {k: round(v, 4) for k, v in o.items()}, aux['FT|clean'], flush=True)
(R / 'input_faults.json').write_text(json.dumps(res, indent=1, default=lambda x: int(x) if isinstance(x, np.integer) else float(x)))
print('done')
