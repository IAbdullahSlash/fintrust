"""Policy-selection check on FI-2010 days 7-8 only (no test data). Compares the default
thresholds with a horizon-scaled variant (W=500, minimum=2000, statistical quarantines auto-cleared)."""
import json, numpy as np
from pathlib import Path
from replay import replay, Config
D = Path(__file__).resolve().parent
z = dict(np.load(D / 'fi2010_forecasts.npz')); ag = ['LOGIT40', 'GBT', 'MLP']; n = int(np.argmax(z['day'] == 9))
y = z['y'][:n].astype(int).tolist(); F = [{a: float(z[a][t]) for a in ag} for t in range(n)]
res = {}
for name, cfg, ac in [('default', Config(), False), ('W50_autoclear', Config(), True), ('W500_autoclear', Config(window=500, minimum=2000), True)]:
    o = replay(F, y, 10, ag, cfg=cfg, auto_clear_statistical=ac)
    res[name] = dict(coverage_after_3000=float(np.mean(o['cov'][3000:])), quarantines=sum(t['new'] == 'QUARANTINED' for t in o['transitions']), transitions=len(o['transitions']))
    print(name, res[name], flush=True)
(D / 'fi2010_dev_check.json').write_text(json.dumps(res, indent=1))
