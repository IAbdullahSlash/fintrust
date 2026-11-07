"""Checks lw_fast against fair/engine.lw and times the round-2 methods."""
import time, numpy as np
from pathlib import Path
from engine2 import lw_fast, lw_cusum, stack, recal, ft2
import engine
D = Path(__file__).resolve().parent.parent / 'fair'
z = dict(np.load(D / 'fi_S3_forecasts.npz')); P = np.stack([z[a] for a in ['LOGIT40', 'GBT', 'MLP']], 1); y = z['y'].astype(int)
for r, t in [(.9, .2), (.995, .005)]:
    a = engine.lw(P, y, 10, decay=r, temperature=t)['p']; b = lw_fast(P, y, 10, decay=r, temperature=t)['p']
    print('lw_fast max abs diff', r, t, float(np.abs(a - b).max()))
for nm, f in [('lw_fast', lambda: lw_fast(P, y, 10)), ('lw_cusum', lambda: lw_cusum(P, y, 10)), ('stack', lambda: stack(P, y, 10)),
              ('recal', lambda: recal(P, y, 10)), ('ft2 clim', lambda: ft2(P, y, 10, ref='clim', window=500, minimum=2000))]:
    t0 = time.time(); f(); print(nm, round(time.time() - t0, 2), 's')
