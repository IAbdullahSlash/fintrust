"""Check engine.ft against the unmodified prototype (via ../real/replay.py) on FI-2010 days 7-8."""
import sys, numpy as np
from pathlib import Path
H = Path(__file__).resolve().parent
sys.path.insert(0, str(H.parent / 'real')); sys.path.insert(0, str(H))
from replay import replay, Config
from engine import ft, corrupt_matrix
z = dict(np.load(H.parent / 'real' / 'fi2010_forecasts.npz')); ag = ['LOGIT40', 'GBT', 'MLP']
n = 30000; P = np.stack([z[a][:n] for a in ag], 1); y = z['y'][:n].astype(int)
for cfg, kw, fault in [(Config(auto_clear_statistical=False), dict(auto_clear=False), None),
                       (Config(window=500, minimum=2000, decay=.99, temperature=.05), dict(window=500, minimum=2000, decay=.99, temperature=.05), (2, 'inversion', 15000, None))]:
    Q, _ = corrupt_matrix(P, fault)
    F = [{a: float(Q[t, i]) for i, a in enumerate(ag)} for t in range(n)]
    o = replay(F, y.tolist(), 10, ag, cfg=cfg)
    f = ft(Q, y, 10, **kw)
    ref = np.array([p if c else np.nan for p, c in zip(o['p_ft'], o['cov'])])
    same = np.allclose(np.nan_to_num(ref, nan=-1), np.nan_to_num(f['p'], nan=-1), atol=1e-12)
    print('match' if same else 'MISMATCH', len(o['transitions']), len(f['trans']))
    assert same
