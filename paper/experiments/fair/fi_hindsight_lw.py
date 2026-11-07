"""Stress test: LW with rho, tau chosen IN HINDSIGHT on each test scenario (not implementable),
compared with FT-tuned (validation-selected). usage: python fi_hindsight_lw.py S1|S2|S3"""
import sys, json, itertools, numpy as np
from pathlib import Path
from engine import lw, corrupt_matrix, brier
D = Path(__file__).resolve().parent; name = sys.argv[1]
r = json.loads((D / f'fi_fair_{name}.json').read_text())
z = dict(np.load(D / f'fi_{name}_forecasts.npz')); P = np.stack([z[a] for a in ['LOGIT40', 'GBT', 'MLP']], 1); y = z['y'].astype(int); day = z['day']
test0 = int(np.argmax(~np.isin(day, r['selection_days']))); T = len(y); first_end = int(np.argmax(day == r['test_days'][1]))
fx = {'inversion': (2, 'inversion', test0, None), 'stuck': (2, 'stuck', test0, None), 'overconfident': (2, 'overconfident', test0, None), 'transient': (2, 'inversion', test0, first_end)}
RHO = [.9, .97, .99, .995, .998, .999]; TAU = [.2, .05, .02, .01, .005, .002, .001]
out = {}
for sc, f in fx.items():
    Q, _ = corrupt_matrix(P, f); b = f[3] or T
    s = [(brier(lw(Q, y, 10, decay=a, temperature=t)['p'][test0:b], y[test0:b]), a, t) for a, t in itertools.product(RHO, TAU)]
    best = min(s); out[sc] = dict(lw_hindsight=best[0], rho=best[1], tau=best[2], ft_tuned=r['fixed'][sc]['brier']['FT-tuned'])
    print(name, sc, out[sc], flush=True)
(D / f'fi_hindsight_lw_{name}.json').write_text(json.dumps(out, indent=1))
