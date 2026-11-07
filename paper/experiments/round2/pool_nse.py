"""Pooled evidence across instruments (review item 10), NSE volatility task.
The same four agent models run on six instruments. In the per-instrument design (the paper's), each instrument has
its own governor, so each sees one loss per agent per day. In the pooled design the agent's lifecycle state is shared:
every instrument's matured loss for agent k goes into one evaluation window for k, so windows fill about six times
faster, while scores and weights stay per instrument. Thresholds are unchanged; W and the evidence minimum are counted
in losses, as before (W = 50, minimum 200), and also tried at W = 300, minimum 1200 (= 50 and 200 trading days).
Losses on the same day are correlated across instruments, so a pooled window holds fewer than W independent labels;
we report healthy-agent quarantines to show what that costs. Parameters: the clean-rule FT values from round 1.
usage: python pool_nse.py"""
import json, math, numpy as np, pandas as pd
from pathlib import Path
from engine2 import ft2, corrupt_matrix, brier
R = Path(__file__).resolve().parent; F = R.parent / 'fair'
df = pd.read_csv(F / 'nse_vol_forecasts.csv', index_col=0, parse_dates=True)
AG = ['RV', 'RANGE', 'GBT', 'VIX']; SEL_END = pd.Timestamp('2019-01-01'); FAULT = pd.Timestamp('2021-01-01')
data = [(a, g[AG].values, g['y'].values.astype(int), g.index) for a, g in df.groupby('asset', sort=False)]
bp = json.loads((F / 'nse_vol_results.json').read_text())['best']['FT']

def ft_pooled(Ps, ys, idxs, decay, temperature, prior_mass=20, window=50, minimum=200, consecutive=3,
              bad_thr=.05, good_thr=0., probation_cap=.1, ref='half'):
    """Ps, ys, idxs: per-instrument forecast matrices, outcomes and dates (h = 1). Returns per-instrument outputs."""
    n = len(Ps); K = Ps[0].shape[1]
    dates = sorted(set().union(*[set(ix) for ix in idxs])); pos = [{d: t for t, d in enumerate(ix)} for ix in idxs]
    shift = []
    for y in ys:
        cs = np.cumsum(np.r_[0., y.astype(float)]); pi = (cs + .5) / (np.arange(len(y) + 1) + 1)
        shift.append(.25 - pi[1:] * (1 - pi[1:]) if ref == 'clim' else np.zeros(len(y)))
    ls = [[prior_mass * .25] * K for _ in range(n)]; ms = [[float(prior_mass)] * K for _ in range(n)]
    st = ['SHADOW'] * K; cnt = [0] * K; since = [0] * K; good = [0] * K; bad = [0] * K; win = [[] for _ in range(K)]
    outs = [np.full(len(y), np.nan) for y in ys]; trans = []
    for d in dates:
        live = [(i, pos[i][d]) for i in range(n) if d in pos[i]]
        for i, t in live:
            if t > 0:
                for k in range(K): ls[i][k] *= decay; ms[i][k] *= decay
            s = t - 1
            if s < 0: continue
            for k in range(K):
                loss = (Ps[i][s, k] - ys[i][s]) ** 2; ls[i][k] += loss; ms[i][k] += 1
                cnt[k] += 1; since[k] += 1; win[k].append(loss - .25 + shift[i][s])
                if len(win[k]) == window:
                    dd = sum(win[k]) / window; win[k] = []
                    good[k] = good[k] + 1 if dd < good_thr else 0; bad[k] = bad[k] + 1 if dd > bad_thr else 0; old = st[k]
                    if st[k] == 'ACTIVE' and bad[k] >= consecutive: st[k] = 'QUARANTINED'
                    elif st[k] == 'PROBATION' and bad[k]: st[k] = 'QUARANTINED'
                    elif st[k] == 'SHADOW' and cnt[k] >= minimum and good[k] >= consecutive: st[k] = 'ACTIVE'
                    elif st[k] == 'QUARANTINED' and since[k] >= minimum and good[k] >= consecutive: st[k] = 'PROBATION'
                    elif st[k] == 'PROBATION' and good[k] >= consecutive: st[k] = 'ACTIVE'
                    if old != st[k]: since[k] = good[k] = bad[k] = 0; trans.append((d, k, old, st[k]))
        el = [k for k in range(K) if st[k] in ('ACTIVE', 'PROBATION')]
        caps = {k: (probation_cap if st[k] == 'PROBATION' else 1.) for k in el}
        if not el or sum(caps.values()) < 1 - 1e-12: continue
        for i, t in live:
            sc = {k: (1 - ls[i][k] / ms[i][k]) / temperature for k in el}; mx = max(sc.values()); raw = {k: math.exp(v - mx) for k, v in sc.items()}
            rem = set(raw); budget = 1.; w = {}
            while rem:
                tot = sum(raw[k] for k in rem); trial = {k: budget * raw[k] / tot for k in rem}
                cl = {k for k in rem if trial[k] > caps[k] + 1e-12}
                if not cl: w.update(trial); break
                for k in cl: w[k] = caps[k]; budget -= caps[k]
                rem -= cl
            outs[i][t] = sum(w[k] * Ps[i][t, k] for k in w)
    return outs, trans

def run_all(fault_kind, tgt):
    Qs, Ms, ks = [], [], []
    for a, P, y, ix in data:
        k0 = int((ix < FAULT).sum()); f = None if fault_kind == 'clean' else (AG.index(tgt), fault_kind, k0, None)
        Q, M = corrupt_matrix(P, f); Qs.append(Q); ks.append(k0)
    ys = [y for _, _, y, _ in data]; ixs = [ix for _, _, _, ix in data]; post = [(ix >= FAULT) if fault_kind != 'clean' else (ix >= SEL_END) for ix in ixs]
    j = None if fault_kind == 'clean' else AG.index(tgt); t0 = FAULT if fault_kind != 'clean' else SEL_END; out = {}
    def summarise(ps, trans_list, det_fn):
        L = np.concatenate([((np.where(np.isnan(p), Q.mean(1), p) - y) ** 2)[m] for p, Q, y, m in zip(ps, Qs, ys, post)])
        return dict(brier=float(L.mean()), **det_fn(trans_list))
    def det_per(trs):   # per-instrument governors: list of (date, k, old, new) per instrument
        dl = [next(((d - FAULT).days for d, k, _, nw in tr if k == j and nw == 'QUARANTINED' and d >= FAULT), None) for tr in trs] if j is not None else []
        hq = sum(1 for tr in trs for d, k, _, nw in tr if nw == 'QUARANTINED' and d >= t0 and k != j)
        return dict(detect_days=dl, healthy_q=hq)
    for ref in ['half', 'clim']:
        ps, trs = [], []
        for Q, y, ix in zip(Qs, ys, ixs):
            o = ft2(Q, y, 1, ref=ref, **bp); ps.append(o['p']); trs.append([(ix[t], k, a, b) for t, k, a, b in o['trans']])
        out[f'per_{ref}'] = summarise(ps, trs, det_per)
        for W, mn in [(50, 200), (300, 1200)]:
            ps, tr = ft_pooled(Qs, ys, ixs, window=W, minimum=mn, ref=ref, **bp)
            out[f'pool{W}_{ref}'] = summarise(ps, [tr], det_per)
    return out

res = {}
for sc in ['clean', 'inversion', 'stuck', 'overconfident']:
    for tgt in ([None] if sc == 'clean' else AG):
        key = sc if tgt is None else f'{sc}:{tgt}'; res[key] = run_all(sc, tgt)
        print(key, {v: (round(d['brier'], 4), d['detect_days'], d['healthy_q']) for v, d in res[key].items()}, flush=True)
(R / 'pool_nse.json').write_text(json.dumps(res, indent=1, default=lambda x: int(x) if isinstance(x, np.integer) else float(x)))
print('done')
