"""Threshold sensitivity with 30 seeds (same seeds as the main synthetic table), FT with tuned weighting."""
import sys, json, numpy as np
from pathlib import Path
D = Path(__file__).resolve().parent; sys.path.insert(0, str(D.parent))
from simulate import stream, T, FAULT
from engine import ft, corrupt_matrix, brier
best = json.loads((D / 'synth_fair.json').read_text())['best']['FT']
def data(seed):
    s = stream(seed); return np.array([[o[k] for k in 'ABCD'] for _, o in s]), np.array([a for a, _ in s])
D_ = [data(s) for s in range(30)]
out = []
for W, c in [(25, 3), (50, 2), (50, 3), (50, 5), (100, 3)]:
    row = dict(window=W, consecutive=c)
    for sc in ['clean', 'inversion', 'overconfident']:
        fq = 0; det = []; br = []
        for P, y in D_:
            f = None if sc == 'clean' else (3, sc, FAULT, None); Q, M = corrupt_matrix(P, f)
            o = ft(Q, y, 1, window=W, consecutive=c, **best); p = o['p'].copy(); m = np.isnan(p); p[m] = Q[m].mean(1)
            fq += sum(1 for t, k, a, b in o['trans'] if b == 'QUARANTINED' and (sc == 'clean' or k != 3 or t < FAULT))
            if f: det.append(next((t - FAULT for t, k, a, b in o['trans'] if k == 3 and b == 'QUARANTINED' and t >= FAULT), None))
            br.append(brier(p[FAULT:], y[FAULT:]))
        row[sc] = dict(false_q=fq, detected=sum(d is not None for d in det), delay_median=float(np.median([d for d in det if d is not None])) if any(d is not None for d in det) else None, brier=float(np.mean(br)))
    out.append(row); print(row, flush=True)
(D / 'synth_sens.json').write_text(json.dumps(out, indent=1))
