"""Fair-baseline study on one FI-2010 split.
Every combiner's parameters are chosen by clean Brier loss on the split's selection days only; the
selected values are then frozen for all test scenarios. usage: python fi_fair.py S1|S2|S3"""
import sys, json, itertools, numpy as np
from pathlib import Path
from engine import ft, lw, fixed_share, dwm, ew, oracle_excl, corrupt_matrix, brier
D = Path(__file__).resolve().parent; name = sys.argv[1]
SEL = {'S1': [4], 'S2': [6], 'S3': [7, 8]}[name]
z = dict(np.load(D / f'fi_{name}_forecasts.npz')); AG = ['LOGIT40', 'GBT', 'MLP']
P = np.stack([z[a] for a in AG], 1); y = z['y'].astype(int); day = z['day']; H = 10
test0 = int(np.argmax(~np.isin(day, SEL))); T = len(y); tdays = sorted(set(day[test0:]))
sel = slice(0, test0)
def fill(o, Q):  # abstention -> equal-weight fallback (scoring convention), coverage reported separately
    p = o['p'].copy(); m = np.isnan(p); p[m] = Q[m].mean(1); return p
RHO = [.9, .97, .99, .995, .998, .999]; TAU = [.2, .05, .02, .01, .005, .002, .001]
grids = {
 'LW': [dict(decay=r, temperature=t) for r, t in itertools.product(RHO, TAU)],
 'FT': [dict(decay=r, temperature=t, window=50 * H, minimum=200 * H) for r, t in itertools.product(RHO, TAU)],
 'FS': [dict(eta=e, alpha=a) for e, a in itertools.product([1, 2, 5, 10, 20, 50, 100], [0, 1e-4, 1e-3, 1e-2, .05])],
 'DWM': [dict(beta=b, theta=th, period=p) for b, th, p in itertools.product([.5, .8, .9, .95, .99], [.01, .05, .1], [1, 10, 100])]}
fn = {'LW': lw, 'FT': ft, 'FS': fixed_share, 'DWM': dwm}
Psel, ysel = P[sel], y[sel]
best = {}
for m, g in grids.items():
    scores = [(brier(fill(fn[m](Psel, ysel, H, **kw), Psel) if m == 'FT' else fn[m](Psel, ysel, H, **kw)['p'], ysel), i) for i, kw in enumerate(g)]
    s, i = min(scores); best[m] = dict(params=g[i], sel_brier=s); print(name, m, best[m], flush=True)
single = int(np.argmin([brier(P[sel, k], y[sel]) for k in range(3)])); best['BEST1'] = dict(agent=AG[single])
methods = {'EW': None, 'BEST1': None, 'LW-paper': (lw, dict(decay=.9, temperature=.2)), 'LW-tuned': (lw, best['LW']['params']),
           'FS-tuned': (fixed_share, best['FS']['params']), 'DWM-tuned': (dwm, best['DWM']['params']),
           'FT-paper': (ft, dict(window=500, minimum=2000)), 'FT-tuned': (ft, best['FT']['params'])}
def run(fault, clear=None):
    Q, M = corrupt_matrix(P, fault); a = fault[2] if fault else test0; b = (fault[3] or T) if fault else T
    sl = slice(a, b); r = {}; aux = {}
    for m, spec in methods.items():
        if m == 'EW': p = ew(Q)
        elif m == 'BEST1': p = Q[:, single]
        else:
            o = spec[0](Q, y, H, **spec[1]); p = fill(o, Q) if spec[0] is ft else o['p']
            if spec[0] is ft:
                aux[m] = dict(coverage=float(o['cov'][sl].mean()), covered_brier=float(np.mean((o['p'][sl][o['cov'][sl]] - y[sl][o['cov'][sl]]) ** 2)) if o['cov'][sl].any() else None,
                              detect=next((t - a for t, k, o_, n in o['trans'] if fault and k == fault[0] and n == 'QUARANTINED' and t >= a), None),
                              n_quar_healthy=sum(1 for t, k, o_, n in o['trans'] if n == 'QUARANTINED' and t >= test0 and (not fault or k != fault[0])))
            if m in ('LW-tuned', 'FT-tuned') and fault: aux[m + '_w'] = float(o['W'][sl, fault[0]].mean())
        r[m] = brier(p[sl], y[sl])
    r['ORACLE'] = brier(oracle_excl(Q, M)[sl], y[sl]); return r, aux, (P[sl] if False else None)
def block_ci(d, L=1000, B=2000, seed=0):
    n = len(d) // L; bl = np.asarray(d[:n * L]).reshape(n, L).mean(1); r = np.random.default_rng(seed)
    m = np.sort([bl[r.integers(0, n, n)].mean() for _ in range(B)]); return float(bl.mean()), float(m[int(.025 * B)]), float(m[int(.975 * B)])
res = dict(split=name, selection_days=SEL, test_days=[int(d) for d in tdays], n_test=T - test0, best=best, fixed={}, random={})
mlp = AG.index('MLP'); first_end = int(np.argmax(day == tdays[1])) if len(tdays) > 1 else T
fixed = {'clean': None, 'inversion': (mlp, 'inversion', test0, None), 'stuck': (mlp, 'stuck', test0, None),
         'overconfident': (mlp, 'overconfident', test0, None), 'transient': (mlp, 'inversion', test0, first_end)}
for sc, f in fixed.items():
    r, aux, _ = run(f); res['fixed'][sc] = dict(brier=r, aux=aux)
    # paired block-bootstrap CI for FT-tuned minus LW-tuned on this scenario
    Q, M = corrupt_matrix(P, f); a = f[2] if f else test0; b = (f[3] or T) if f else T
    pf = fill(ft(Q, y, H, **best['FT']['params']), Q); pl = lw(Q, y, H, **best['LW']['params'])['p']
    res['fixed'][sc]['ft_minus_lw_ci'] = block_ci(((pf - y) ** 2 - (pl - y) ** 2)[a:b])
    print(name, sc, {k: round(v, 4) for k, v in r.items()}, aux, res['fixed'][sc]['ft_minus_lw_ci'], flush=True)
rng = np.random.default_rng(42); L = T - test0
for kind in ['inversion', 'stuck', 'overconfident']:
    draws = []
    for i in range(10):
        k = int(rng.integers(0, 3)); on = int(test0 + rng.integers(0, L // 2))
        r, aux, _ = run((k, kind, on, None)); draws.append(dict(agent=AG[k], onset=on - test0, brier=r, aux=aux))
    res['random'][kind] = draws
    print(name, kind, 'FT-tuned - LW-tuned', [round(d['brier']['FT-tuned'] - d['brier']['LW-tuned'], 4) for d in draws], flush=True)
(D / f'fi_fair_{name}.json').write_text(json.dumps(res, indent=1))
