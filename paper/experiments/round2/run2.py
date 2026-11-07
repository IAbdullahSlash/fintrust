"""Round-2 study on one dataset: S1 | S2 | S3 (FI-2010 splits) | BTC.

Two tuning rules, applied identically to every method:
  clean  : parameters minimise Brier loss on the fault-free selection period (as in round 1)
  fault  : parameters minimise the mean Brier loss over the clean selection stream plus six faulty copies of it
           (two inversion, two stuck, two overconfident; target agent uniform; onset uniform in the first half
           of the selection period; numpy default_rng(7)), each scored from its onset
Test scenarios are the round-1 ones: fixed faults on the best single agent from the first test sample (transient:
first test day for FI-2010, first 30 days for BTC) and 10 random draws per type (default_rng(42)).
Methods: EW, BEST1, LW, FS, DWM, LW+CUSUM (LWC), stacking (STK), LW and FT on online-recalibrated forecasts
(LWR, FTR), FT, FT with a base-rate reference (FTB), and the exclusion oracle. LW-ref and FT-ref use rho=.9, tau=.2.
usage: OMP_NUM_THREADS=1 python run2.py S1|S2|S3|BTC"""
import sys, json, itertools, time, numpy as np
from pathlib import Path
from engine2 import lw_fast, lw_cusum, stack, recal_fast, fs_fast, ft2, dwm, ew, oracle_excl, corrupt_matrix, brier
R = Path(__file__).resolve().parent; F = R.parent / 'fair'; name = sys.argv[1]
if name == 'BTC':
    z = dict(np.load(R / 'btc_forecasts.npz')); AG = ['RV', 'RANGE', 'GBT', 'TOD']; H = 6; BL = 288
    y = z['y'].astype(int); test0 = int(np.argmax(z['period'] == 1)); last_sel0 = int(np.argmax(z['day'] >= 30))
    tr_end = test0 + 30 * 288
else:
    z = dict(np.load(F / f'fi_{name}_forecasts.npz')); AG = ['LOGIT40', 'GBT', 'MLP']; H = 10; BL = 1000
    SEL = {'S1': [4], 'S2': [6], 'S3': [7, 8]}[name]; day = z['day']; y = z['y'].astype(int)
    test0 = int(np.argmax(~np.isin(day, SEL))); last_sel0 = int(np.argmax(day == SEL[-1]))
    tdays = sorted(set(day[test0:])); tr_end = int(np.argmax(day == tdays[1]))
P = np.stack([z[a] for a in AG], 1); T, K = P.shape
RHO = [.7, .8, .9, .97, .99, .995, .998, .999]; TAU = [.2, .05, .02, .01, .005, .002, .001]
RT = [dict(decay=r, temperature=t) for r, t in itertools.product(RHO, TAU)]
FTW = dict(window=50 * H, minimum=200 * H)
G = {'LW': RT, 'FT': RT, 'FTB': RT, 'LWR': RT, 'FTR': RT,
     'FS': [dict(eta=e, alpha=a) for e, a in itertools.product([.1, .3, 1, 2, 5, 10, 20, 50, 100], [0, 1e-4, 1e-3, 1e-2, .05, .1, .2])],
     'DWM': [dict(beta=b, theta=th, period=p) for b, th, p in itertools.product([.5, .8, .9, .95, .99], [.01, .05, .1], [1, 10, 100])],
     'LWC': [dict(k=k, H=hc) for k, hc in itertools.product([0, .02, .05, .1], [.5, 1, 2, 5, 10, 20, 50])],
     'STK': [dict(lr=lr, l2=l2) for lr, l2 in itertools.product([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2], [0, 1e-3])]}
RECLR = [1e-4, 1e-3, 1e-2, 3e-2]

def fill(o, Q):
    p = o['p'].copy(); m = np.isnan(p); p[m] = Q[m].mean(1); return p

def combine(m, Q, yy, kw, extra=None):
    """Returns (p, ft_output_or_None). Q is the (possibly corrupted) forecast matrix; extra holds tuned LW params for LWC
    and the recalibrated matrix for LWR/FTR."""
    if m == 'LW': return lw_fast(Q, yy, H, **kw)['p'], None
    if m == 'FS': return fs_fast(Q, yy, H, **kw)['p'], None
    if m == 'DWM': return dwm(Q, yy, H, **kw)['p'], None
    if m == 'STK': return stack(Q, yy, H, **kw)['p'], None
    if m == 'LWC': return lw_cusum(Q, yy, H, **extra['lw'], **kw)['p'], None
    if m == 'LWR': return lw_fast(extra['R'], yy, H, **kw)['p'], None
    if m in ('FT', 'FTB', 'FTR'):
        X = extra['R'] if m == 'FTR' else Q
        o = ft2(X, yy, H, ref='clim' if m == 'FTB' else 'half', **FTW, **kw); return fill(o, X), o
    raise KeyError(m)

# ---------------- tuning ----------------
Ps, ys = P[:test0], y[:test0]
r7 = np.random.default_rng(7); sel_faults = [None]
for kind in ['inversion', 'stuck', 'overconfident']:
    for _ in range(2): sel_faults.append((int(r7.integers(0, K)), kind, int(r7.integers(0, test0 // 2)), None))
streams = [corrupt_matrix(Ps, f)[0] for f in sel_faults]; starts = [0 if f is None else f[2] for f in sel_faults]
t0 = time.time(); obj = {}; best = {'clean': {}, 'fault': {}}
def score(p, i): return brier(p[starts[i]:], ys[starts[i]:])
# recalibration learning rate: mean per-agent Brier of recalibrated forecasts
recQ = {(i, lr): recal_fast(S, ys, H, lr) for i, S in enumerate(streams) for lr in RECLR}
rec_obj = {lr: [float(np.mean([brier(recQ[(i, lr)][starts[i]:, k], ys[starts[i]:]) for k in range(K)])) for i in range(len(streams))] for lr in RECLR}
for mode, idx in [('clean', [0]), ('fault', list(range(len(streams))))]:
    best[mode]['REC_lr'] = min(RECLR, key=lambda lr: np.mean([rec_obj[lr][i] for i in idx]))
order = ['LW', 'FS', 'DWM', 'STK', 'FT', 'FTB', 'LWR', 'FTR', 'LWC']
for m in order:
    # objectives per config per stream (computed once, both modes read from them)
    if m in ('LWR', 'FTR'):
        vals = {}
        for mode in ['clean', 'fault']:
            lr = best[mode]['REC_lr']
            for c, kw in enumerate(G[m]):
                for i in ([0] if mode == 'clean' else range(len(streams))):
                    if (c, i, lr) not in vals: vals[(c, i, lr)] = score(combine(m, streams[i], ys, kw, {'R': recQ[(i, lr)]})[0], i)
            idx = [0] if mode == 'clean' else list(range(len(streams)))
            o = [float(np.mean([vals[(c, i, lr)] for i in idx])) for c in range(len(G[m]))]
            best[mode][m] = G[m][int(np.argmin(o))]; obj[f'{m}_{mode}'] = o
    elif m == 'LWC':
        for mode in ['clean', 'fault']:
            idx = [0] if mode == 'clean' else list(range(len(streams)))
            o = [float(np.mean([score(combine(m, streams[i], ys, kw, {'lw': best[mode]['LW']})[0], i) for i in idx])) for kw in G[m]]
            best[mode][m] = G[m][int(np.argmin(o))]; obj[f'{m}_{mode}'] = o
    else:
        v = np.array([[score(combine(m, S, ys, kw)[0], i) for i, S in enumerate(streams)] for kw in G[m]])
        obj[m] = v.tolist()
        for mode, idx in [('clean', [0]), ('fault', list(range(len(streams))))]:
            best[mode][m] = G[m][int(np.argmin(v[:, idx].mean(1)))]
        if m in ('LW', 'FT'):   # selection-length check: clean objective on the last selection day only
            o1 = [brier(combine(m, Ps, ys, kw)[0][last_sel0:], ys[last_sel0:]) for kw in G[m]]
            best.setdefault('clean_lastday', {})[m] = G[m][int(np.argmin(o1))]
    print(name, m, {md: best[md][m] for md in ['clean', 'fault']}, f'{time.time() - t0:.0f}s', flush=True)
single = int(np.argmin([brier(Ps[:, k], ys) for k in range(K)]))

# ---------------- test ----------------
def block_ci(d, B=2000, seed=0):
    n = len(d) // BL; bl = np.asarray(d[:n * BL]).reshape(n, BL).mean(1); r = np.random.default_rng(seed)
    m = np.sort([bl[r.integers(0, n, n)].mean() for _ in range(B)]); return [float(bl.mean()), float(m[int(.025 * B)]), float(m[int(.975 * B)])]

def evaluate(fault):
    Q, M = corrupt_matrix(P, fault); a = fault[2] if fault else test0; b = (fault[3] or T) if fault else T
    out = {'EW': brier(ew(Q)[a:b], y[a:b]), 'BEST1': brier(Q[a:b, single], y[a:b]), 'ORACLE': brier(oracle_excl(Q, M)[a:b], y[a:b])}
    aux = {}; series = {}
    out['LW-ref'] = brier(lw_fast(Q, y, H, decay=.9, temperature=.2)['p'][a:b], y[a:b])
    p, o = combine('FT', Q, y, dict(decay=.9, temperature=.2)); out['FT-ref'] = brier(p[a:b], y[a:b])
    Rq = {lr: recal_fast(Q, y, H, lr) for lr in {best['clean']['REC_lr'], best['fault']['REC_lr']}}
    for mode in ['clean', 'fault']:
        for m in order:
            extra = {'lw': best[mode]['LW'], 'R': Rq[best[mode]['REC_lr']]}
            p, o = combine(m, Q, y, best[mode][m], extra); key = f'{m}|{mode}'; out[key] = brier(p[a:b], y[a:b])
            if m in ('LW', 'FT'): series[key] = (p - y) ** 2
            if o is not None:
                aux[key] = dict(coverage=float(o['cov'][a:b].mean()),
                                detect=next((t - a for t, k, _, n in o['trans'] if fault and k == fault[0] and n == 'QUARANTINED' and t >= a), None),
                                healthy_q=[(int(t - test0), AG[k]) for t, k, _, n in o['trans'] if n == 'QUARANTINED' and t >= test0 and (not fault or k != fault[0])])
    for mode in ['clean', 'fault']:
        aux[f'ci|{mode}'] = block_ci((series[f'FT|{mode}'] - series[f'LW|{mode}'])[a:b])
    return out, aux

res = dict(dataset=name, agents=AG, h=H, n_sel=test0, n_test=T - test0, best_single=AG[single], sel_faults=sel_faults,
           best=best, rec_obj=rec_obj, fixed={}, random={}, lastday={}, grid={})
fixed = {'clean': None, 'inversion': (single, 'inversion', test0, None), 'stuck': (single, 'stuck', test0, None),
         'overconfident': (single, 'overconfident', test0, None), 'transient': (single, 'inversion', test0, tr_end)}
for sc, f in fixed.items():
    o, a = evaluate(f); res['fixed'][sc] = dict(brier=o, aux=a)
    print(name, sc, {k: round(v, 4) for k, v in o.items()}, a.get('ci|clean'), a.get('ci|fault'), f'{time.time() - t0:.0f}s', flush=True)
rng = np.random.default_rng(42); L = T - test0
for kind in ['inversion', 'stuck', 'overconfident']:
    res['random'][kind] = []
    for _ in range(10):
        k = int(rng.integers(0, K)); on = int(test0 + rng.integers(0, L // 2))
        o, a = evaluate((k, kind, on, None)); res['random'][kind].append(dict(agent=AG[k], onset=on - test0, brier=o, aux=a))
    print(name, kind, 'FT-LW clean', [round(d['brier']['FT|clean'] - d['brier']['LW|clean'], 4) for d in res['random'][kind]],
          'fault', [round(d['brier']['FT|fault'] - d['brier']['LW|fault'], 4) for d in res['random'][kind]], flush=True)
# selection-length check (clean rule, last selection day only), LW and FT
for sc, f in fixed.items():
    Q, M = corrupt_matrix(P, f); a = f[2] if f else test0; b = (f[3] or T) if f else T
    res['lastday'][sc] = {m: brier(combine(m, Q, y, best['clean_lastday'][m])[0][a:b], y[a:b]) for m in ['LW', 'FT']}
# robustness grid: test loss of LW and FT for every (rho, tau)
for sc in ['clean', 'inversion', 'stuck']:
    f = fixed[sc]; Q, M = corrupt_matrix(P, f); a = f[2] if f else test0
    res['grid'][sc] = {m: [brier(combine(m, Q, y, kw)[0][a:], y[a:]) for kw in RT] for m in ['LW', 'FT']}
    print(name, 'grid', sc, f'{time.time() - t0:.0f}s', flush=True)
res['grid_params'] = RT; res['obj'] = obj
(R / f'r2_{name}.json').write_text(json.dumps(res, indent=1, default=lambda x: x if not isinstance(x, np.integer) else int(x)))
print(name, 'done', f'{time.time() - t0:.0f}s')
