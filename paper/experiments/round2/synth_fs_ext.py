"""Grid-edge check for the synthetic study: Fixed Share re-tuned on an extended grid (eta down to 0.03, alpha up to 0.3)."""
import sys, json, itertools, numpy as np
from pathlib import Path
R = Path(__file__).resolve().parent; sys.path.insert(0, str(R.parent / 'fair')); sys.path.insert(0, str(R.parent))
from simulate import stream, T, FAULT, RECOVER, EVAL_START
from engine2 import fs_fast, corrupt_matrix, brier
def data(seed):
    s = stream(seed); return np.array([[o[k] for k in 'ABCD'] for _, o in s]), np.array([a for a, _ in s])
g = [dict(eta=e, alpha=a) for e, a in itertools.product([.03, .1, .3, 1, 2, 5, 10, 20, 50], [0, 1e-3, 1e-2, .05, .1, .2, .3])]
tune = [data(s) for s in range(100, 105)]; ev = slice(EVAL_START, T)
sc = [np.mean([brier(fs_fast(P, y, 1, **kw)['p'][ev], y[ev]) for P, y in tune]) for kw in g]; best = g[int(np.argmin(sc))]
SC = {'clean': (None, 1), 'inversion': ((3, 'inversion', FAULT, None), 1), 'stuck': ((3, 'stuck', FAULT, None), 1),
      'overconfident': ((3, 'overconfident', FAULT, None), 1), 'transient': ((3, 'inversion', FAULT, RECOVER), 1), 'inversion_delayed': ((3, 'inversion', FAULT, None), 5)}
out = dict(best=best, mean={})
for name, (f, h) in SC.items():
    v = []
    for seed in range(30):
        P, y = data(seed); Q, _ = corrupt_matrix(P, f); post = slice(FAULT, RECOVER if name == 'transient' else T)
        v.append(brier(fs_fast(Q, y, h, **best)['p'][post], y[post]))
    out['mean'][name] = float(np.mean(v))
print(out); (R / 'synth_fs_ext.json').write_text(json.dumps(out, indent=1))
