"""Threshold sensitivity sweep (window length, consecutive bad windows). Synthetic data only."""
import json, statistics as st
from pathlib import Path
from simulate import run, Config
D = Path(__file__).resolve().parent
grid = [(25, 3), (50, 2), (50, 3), (50, 5), (100, 3)]
out = []
for w, c in grid:
    cfg = Config(window=w, consecutive=c)
    for scn in ['clean', 'inversion', 'overconfident']:
        R = [run(scn, k, cfg) for k in range(10)]
        dd = [r['detect_delay'] for r in R if r['detect_delay'] is not None]
        out.append(dict(window=w, consecutive=c, scenario=scn,
            post_ft=st.mean(r['post_ft'] for r in R), post_hedge=st.mean(r['post_hedge'] for r in R),
            false_q=sum(r['false_quarantine'] for r in R), detected=len(dd),
            delay_median=st.median(dd) if dd else None))
        print(out[-1])
(D / 'sensitivity.json').write_text(json.dumps(out, indent=1))
