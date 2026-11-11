"""Sensitivity of the monitor comparison to settings NOT fixed in MONITORS_ADDENDUM.md (reported as exploratory):
drift window (10, 20, 40 days) and a threshold margin (x1, x1.5 the calibration maximum).
usage: python monitors_sensitivity.py -> monitors_sensitivity.json"""
import json
import numpy as np
import monitors as M

def summarise(res):
    out = {}
    for m in ('L1', 'L2', 'B1'):
        hit, n, d, fa = 0, 0, [], 0
        for c in res['configs'].values():
            for f, s in c['scenarios'].items():
                if f == 'clean': fa += s[m]['assets']; continue
                if s['mean_abs_change_vs_clean'] < 1e-9: continue
                hit += s[m]['assets']; n += 6; d += [x for x in s[m]['per_asset'] if x is not None]
        out[m] = dict(caught=hit, of=n, median_days=float(np.median(d)) if d else None, clean_false_alarms=fa)
    return out

rows = {}
for win in (10, 20, 40):
    for mult in (1.0, 1.5):
        M.WIN, M.THR_MULT, M.NSETS = win, mult, 10
        rows[f'win{win}_x{mult}'] = summarise(M.main('_tmp_sens.json', verbose=False)); print(win, mult, rows[f'win{win}_x{mult}'])
json.dump(rows, open(M.HERE / 'monitors_sensitivity.json', 'w'), indent=1)
(M.HERE / '_tmp_sens.json').unlink()
