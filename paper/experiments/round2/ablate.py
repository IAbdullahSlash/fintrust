"""Lifecycle ablation (review item 6). FinTrust with its clean-rule parameters from r2_<name>.json, and one
component changed at a time; nothing is re-tuned. Same fixed and random test faults as run2.py.
Variants:
  full          the paper's FinTrust
  no_shadow     agents start ACTIVE (no evidence minimum before first use)
  no_cap        probation cap 1.0 (a recovering agent gets full weight at once)
  one_window    one window is enough for every state change (instead of three in a row)
  no_gap        good threshold = bad threshold = 0.05 (no hysteresis band)
  operator      every quarantine needs an operator (none is cleared in the test)
  short_window  W = 50, evidence minimum 200 (windows not scaled by the overlap factor)
  no_states     softmax weights only, no lifecycle (= loss weighting with FinTrust's rho, tau)
usage: OMP_NUM_THREADS=1 python ablate.py S1|S2|S3|BTC"""
import sys, json, time, numpy as np
from pathlib import Path
from engine2 import lw_fast, ft_ref, corrupt_matrix, brier
R = Path(__file__).resolve().parent; F = R.parent / 'fair'; name = sys.argv[1]
if name == 'BTC':
    z = dict(np.load(R / 'btc_forecasts.npz')); AG = ['RV', 'RANGE', 'GBT', 'TOD']; H = 6
    y = z['y'].astype(int); test0 = int(np.argmax(z['period'] == 1)); tr_end = test0 + 30 * 288
else:
    z = dict(np.load(F / f'fi_{name}_forecasts.npz')); AG = ['LOGIT40', 'GBT', 'MLP']; H = 10
    SEL = {'S1': [4], 'S2': [6], 'S3': [7, 8]}[name]; day = z['day']; y = z['y'].astype(int)
    test0 = int(np.argmax(~np.isin(day, SEL))); tdays = sorted(set(day[test0:])); tr_end = int(np.argmax(day == tdays[1]))
P = np.stack([z[a] for a in AG], 1); T, K = P.shape
r2 = json.loads((R / f'r2_{name}.json').read_text()); bp = r2['best']['clean']['FT']; single = AG.index(r2['best_single'])
base = dict(window=50 * H, minimum=200 * H, **bp)
V = {'full': {}, 'no_shadow': dict(start='ACTIVE'), 'no_cap': dict(probation_cap=1.), 'one_window': dict(consecutive=1),
     'no_gap': dict(good_thr=.05), 'operator': dict(auto_clear=False), 'short_window': dict(window=50, minimum=200), 'no_states': None}

def run(fault):
    Q, M = corrupt_matrix(P, fault); a = fault[2] if fault else test0; b = (fault[3] or T) if fault else T; out = {}
    for v, kw in V.items():
        if kw is None:
            out[v] = dict(brier=brier(lw_fast(Q, y, H, **bp)['p'][a:b], y[a:b])); continue
        o = ft_ref(Q, y, H, **{**base, **kw}); p = o['p'].copy(); m = np.isnan(p); p[m] = Q[m].mean(1)
        out[v] = dict(brier=brier(p[a:b], y[a:b]), abstain=float(m[a:b].mean()),
                      detect=next((t - a for t, k, _, n in o['trans'] if fault and k == fault[0] and n == 'QUARANTINED' and t >= a), None),
                      healthy_q=sum(1 for t, k, _, n in o['trans'] if n == 'QUARANTINED' and t >= test0 and (not fault or k != fault[0])))
    return out

t0 = time.time(); res = dict(dataset=name, params=bp, fixed={}, random={})
fixed = {'clean': None, 'inversion': (single, 'inversion', test0, None), 'stuck': (single, 'stuck', test0, None),
         'overconfident': (single, 'overconfident', test0, None), 'transient': (single, 'inversion', test0, tr_end)}
for sc, f in fixed.items():
    res['fixed'][sc] = run(f); print(name, sc, {v: round(d['brier'], 4) for v, d in res['fixed'][sc].items()}, f'{time.time() - t0:.0f}s', flush=True)
rng = np.random.default_rng(42); L = T - test0
for kind in ['inversion', 'stuck', 'overconfident']:
    res['random'][kind] = []
    for _ in range(10):
        k = int(rng.integers(0, K)); on = int(test0 + rng.integers(0, L // 2)); res['random'][kind].append(run((k, kind, on, None)))
    print(name, kind, {v: round(float(np.mean([d[v]['brier'] for d in res['random'][kind]])), 4) for v in V}, f'{time.time() - t0:.0f}s', flush=True)
(R / f'abl_{name}.json').write_text(json.dumps(res, indent=1))
print(name, 'done')
