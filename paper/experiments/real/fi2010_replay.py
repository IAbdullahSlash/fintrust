"""Replay FI-2010 forecasts through FinTrust with faults injected on day 9 (agent MLP)."""
import json, numpy as np, random
from pathlib import Path
import sys
from replay import replay, Config
VARIANT = len(sys.argv) > 1 and sys.argv[1] == 'variant'
CFG = Config(window=500, minimum=2000) if VARIANT else Config()
TAG = '_variant' if VARIANT else ''
D = Path(__file__).resolve().parent
z = dict(np.load(D / 'fi2010_forecasts.npz')); y = z['y'].astype(int).tolist(); day = z['day']
agents = ['LOGIT40', 'GBT', 'MLP']
F = [{a: float(z[a][t]) for a in agents} for t in range(len(y))]
start9 = int(np.argmax(day == 9)); start10 = int(np.argmax(day == 10)); T = len(y)
scen = {'clean': None, 'inversion': ('MLP', 'inversion', start9, None), 'stuck': ('MLP', 'stuck', start9, None),
        'overconfident': ('MLP', 'overconfident', start9, None), 'transient': ('MLP', 'inversion', start9, start10)}
def block_ci(d, B=2000, L=1000, seed=0):
    d = np.asarray(d); n = len(d) // L; blocks = d[:n * L].reshape(n, L).mean(1); r = np.random.default_rng(seed)
    m = np.sort([blocks[r.integers(0, n, n)].mean() for _ in range(B)])
    return float(blocks.mean()), float(m[int(.025 * B)]), float(m[int(.975 * B)])
res = {}; traces = {}
for name, fault in scen.items():
    o = replay(F, y, 10, agents, fault=fault, clear_at=start10 if name == 'transient' else None, cfg=CFG, auto_clear_statistical=VARIANT)
    end = start10 if name == 'transient' else T
    sl = slice(start9, end); yy = np.array(y[sl])
    b = {k: (np.array(o[k][sl]) - yy) ** 2 for k in ['p_eq', 'p_lw', 'p_ft']}
    healthy = [a for a in agents if fault is None or a != fault[0]]
    p_or = np.array([np.mean([F[t][a] for a in healthy]) for t in range(start9, end)])
    qt = [tr for tr in o['transitions'] if tr['agent'] == 'MLP' and tr['new'] == 'QUARANTINED' and tr['time'] >= start9]
    res[name] = dict(brier_eq=float(b['p_eq'].mean()), brier_lw=float(b['p_lw'].mean()), brier_ft=float(b['p_ft'].mean()),
        brier_oracle=float(((p_or - yy) ** 2).mean()), diff_ft_lw=block_ci(b['p_ft'] - b['p_lw']), diff_ft_eq=block_ci(b['p_ft'] - b['p_eq']),
        coverage=float(np.mean(o['cov'][sl])), wMLP_lw=float(np.mean([w['MLP'] for w in o['w_lw'][sl]])),
        wMLP_ft=float(np.mean([w.get('MLP', 0.) for w in o['w_ft'][sl]])),
        detect_delay=(qt[0]['time'] - start9) if qt and fault else None,
        transitions=o['transitions'], audit_ok=o['audit_ok'], audit_n=o['audit_n'], n_eval=end - start9,
        first_active={a: next((tr['time'] for tr in o['transitions'] if tr['agent'] == a and tr['new'] == 'ACTIVE'), None) for a in agents},
        coverage_day7=float(np.mean(o['cov'][:int(np.argmax(day == 8))])))
    if name in ('inversion', 'transient'):
        traces[name] = dict(lw=[w['MLP'] for w in o['w_lw'][::100]], ft=[w.get('MLP', 0.) for w in o['w_ft'][::100]])
    print(name, {k: v for k, v in res[name].items() if k != 'transitions'}, flush=True)
res['meta'] = dict(start9=start9, start10=start10, T=T)
(D / f'fi2010_results{TAG}.json').write_text(json.dumps(res, indent=1)); (D / f'fi2010_traces{TAG}.json').write_text(json.dumps(traces))
