"""Fair-baseline version of the synthetic study (paper Sec. VII).
Parameters of LW, FT, Fixed Share and DWM are chosen on clean streams with tuning seeds 100-104,
then frozen and evaluated on seeds 0-29 for every scenario. FT uses the two-tier rule (statistical
quarantines self-clear). Labels do not overlap here, so W stays 50."""
import sys, json, itertools, numpy as np
from pathlib import Path
D = Path(__file__).resolve().parent; sys.path.insert(0, str(D.parent))
from simulate import stream, T, FAULT, RECOVER, EVAL_START
from engine import ft, lw, fixed_share, dwm, ew, oracle_excl, corrupt_matrix, brier
def data(seed):
    s = stream(seed); y = np.array([a for a, _ in s]); P = np.array([[o[k] for k in 'ABCD'] for _, o in s]); return P, y
def fill(o, Q): p = o['p'].copy(); m = np.isnan(p); p[m] = Q[m].mean(1); return p
RHO = [.8, .9, .95, .98, .99, .995]; TAU = [.2, .1, .05, .02, .01]
grids = {'LW': (lw, [dict(decay=r, temperature=t) for r, t in itertools.product(RHO, TAU)]),
         'FT': (ft, [dict(decay=r, temperature=t) for r, t in itertools.product(RHO, TAU)]),
         'FS': (fixed_share, [dict(eta=e, alpha=a) for e, a in itertools.product([1, 2, 5, 10, 20, 50], [0, 1e-3, 1e-2, .05, .1])]),
         'DWM': (dwm, [dict(beta=b, theta=th, period=p) for b, th, p in itertools.product([.5, .8, .9, .95, .99], [.01, .05, .1], [1, 10, 50])])}
tune = [data(s) for s in range(100, 105)]; ev = slice(EVAL_START, T)
best = {}
for m, (f, g) in grids.items():
    sc = []
    for kw in g:
        sc.append(np.mean([brier((fill(f(P, y, 1, **kw), P) if f is ft else f(P, y, 1, **kw)['p'])[ev], y[ev]) for P, y in tune]))
    best[m] = g[int(np.argmin(sc))]; print(m, best[m], round(min(sc), 5), flush=True)
methods = {'EW': None, 'LW-paper': (lw, dict(decay=.9, temperature=.2)), 'LW-tuned': (lw, best['LW']), 'FS-tuned': (fixed_share, best['FS']),
           'DWM-tuned': (dwm, best['DWM']), 'FT-paper': (ft, {}), 'FT-tuned': (ft, best['FT'])}
SC = {'clean': (None, 1), 'inversion': ((3, 'inversion', FAULT, None), 1), 'stuck': ((3, 'stuck', FAULT, None), 1),
      'overconfident': ((3, 'overconfident', FAULT, None), 1), 'transient': ((3, 'inversion', FAULT, RECOVER), 1), 'inversion_delayed': ((3, 'inversion', FAULT, None), 5)}
res = dict(best=best, scenarios={}); traces = {}
for name, (fault, h) in SC.items():
    rows = []; tr = {'FT-tuned': [], 'LW-tuned': []}
    for seed in range(30):
        P, y = data(seed); Q, M = corrupt_matrix(P, fault)
        post = slice(FAULT, RECOVER if name == 'transient' else T) if fault else slice(FAULT, T)
        r = {}; aux = {}
        for m, spec in methods.items():
            if spec is None: p = ew(Q)
            else:
                o = spec[0](Q, y, h, **spec[1]); p = fill(o, Q) if spec[0] is ft else o['p']
                if m in tr: tr[m].append(o['W'][:, 3])
                if spec[0] is ft:
                    aux[m + '_detect'] = next((t - FAULT for t, k, a, b in o['trans'] if fault and k == 3 and b == 'QUARANTINED' and t >= FAULT), None)
                    aux[m + '_falseq'] = sum(1 for t, k, a, b in o['trans'] if b == 'QUARANTINED' and (k != 3 or not fault or t < FAULT))
            r[m] = brier(p[post], y[post])
        r['ORACLE'] = brier(oracle_excl(Q, M)[post], y[post]); rows.append(dict(seed=seed, brier=r, aux=aux))
    d = np.array([x['brier']['FT-tuned'] - x['brier']['LW-tuned'] for x in rows]); rr = np.random.default_rng(1)
    bs = np.sort([d[rr.integers(0, 30, 30)].mean() for _ in range(10000)])
    res['scenarios'][name] = dict(mean={m: float(np.mean([x['brier'][m] for x in rows])) for m in rows[0]['brier']},
        ft_minus_lw=(float(d.mean()), float(bs[250]), float(bs[9750])),
        detect=[x['aux'].get('FT-tuned_detect') for x in rows], falseq=int(sum(x['aux'].get('FT-tuned_falseq', 0) for x in rows)),
        detect_paper=[x['aux'].get('FT-paper_detect') for x in rows], falseq_paper=int(sum(x['aux'].get('FT-paper_falseq', 0) for x in rows)))
    if name in ('inversion', 'transient'):
        traces[name] = {m: dict(mean=np.mean(v, 0)[::10].tolist(), lo=np.percentile(v, 10, 0)[::10].tolist(), hi=np.percentile(v, 90, 0)[::10].tolist()) for m, v in tr.items()}
    s = res['scenarios'][name]; print(name, {k: round(v, 4) for k, v in s['mean'].items()}, [round(x, 4) for x in s['ft_minus_lw']], 'falseq', s['falseq'], flush=True)
(D / 'synth_fair.json').write_text(json.dumps(res, indent=1)); (D / 'synth_traces.json').write_text(json.dumps(traces))
