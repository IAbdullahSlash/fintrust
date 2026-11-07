"""Replay NSE daily forecasts through FinTrust per asset (one asset-horizon task each).
Scenarios: clean; MOM agent sign-inverted from 2021-01-01 (injected fault on real outcomes).
Economics (illustrative): long/flat on combined p>0.5, 10 bps per position change; FinTrust is flat when it abstains.
"""
import json, numpy as np, pandas as pd
from pathlib import Path
from replay import replay
D = Path(__file__).resolve().parent
df = pd.read_csv(D / 'nse_forecasts.csv', parse_dates=['date']).dropna(subset=['y'])
agents = ['MOM', 'REV', 'GBT', 'VIX']; COST = 0.001; EVAL = pd.Timestamp('2017-01-01'); FAULT = pd.Timestamp('2021-01-01')
def stats(r):
    r = np.asarray(r); eq = np.cumsum(r); mdd = float(np.max(np.maximum.accumulate(eq) - eq))
    return dict(ann_ret=float(r.mean() * 252), sharpe=float(r.mean() / (r.std() + 1e-12) * np.sqrt(252)), mdd=mdd)
def block_ci(d, dates, B=2000, seed=0):
    s = pd.Series(d, index=dates).groupby(pd.Grouper(freq='ME')).mean().dropna().values
    r = np.random.default_rng(seed); m = np.sort([s[r.integers(0, len(s), len(s))].mean() for _ in range(B)])
    return float(s.mean()), float(m[int(.025 * B)]), float(m[int(.975 * B)])
out = {}
for scn in ['clean', 'mom_inversion']:
    per = []; daily = {}
    for asset, g in df.groupby('asset'):
        g = g.sort_values('date').reset_index(drop=True)
        F = g[agents].to_dict('records'); y = g['y'].astype(int).tolist()
        fault = ('MOM', 'inversion', int((g['date'] < FAULT).sum()), None) if scn == 'mom_inversion' else None
        o = replay(F, y, 1, agents, fault=fault)
        m = (g['date'] >= EVAL).values; yy = g['y'].values[m]
        rec = dict(asset=asset, n=int(m.sum()))
        for k in ['p_eq', 'p_lw', 'p_ft']:
            p = np.array(o[k])[m]; rec['brier_' + k[2:]] = float(((p - yy) ** 2).mean())
            pos = (p > .5).astype(float)
            if k == 'p_ft': pos = pos * np.array(o['cov'])[m]
            ret = pos * g['ret_next'].values[m] - COST * np.abs(np.diff(np.concatenate([[0], pos])))
            daily.setdefault(k, []).append(pd.Series(ret, index=g['date'][m]))
            rec['exposure_' + k[2:]] = float(pos.mean())
        daily.setdefault('bh', []).append(pd.Series(g['ret_next'].values[m], index=g['date'][m]))
        rec['coverage'] = float(np.mean(np.array(o['cov'])[m]))
        st = pd.DataFrame(o['states'])[m]
        rec['frac_active'] = {a: float((st[a].isin(['ACTIVE', 'PROBATION'])).mean()) for a in agents}
        rec['n_transitions'] = len(o['transitions']); rec['n_quarantines'] = sum(t['new'] == 'QUARANTINED' for t in o['transitions'])
        if fault:
            q = [t for t in o['transitions'] if t['agent'] == 'MOM' and t['new'] == 'QUARANTINED' and t['time'] >= fault[2]]
            post = (g['date'] >= FAULT).values
            rec['detect_delay_days'] = (q[0]['time'] - fault[2]) if q else None
            rec['wMOM_lw_post'] = float(np.mean([w['MOM'] for w, k in zip(o['w_lw'], post) if k]))
            rec['wMOM_ft_post'] = float(np.mean([w.get('MOM', 0.) for w, k in zip(o['w_ft'], post) if k]))
            for k in ['p_lw', 'p_ft', 'p_eq']:
                rec['post_brier_' + k[2:]] = float(((np.array(o[k])[post] - g['y'].values[post]) ** 2).mean())
        rec['indiv_brier'] = {a: float(((g[a].values[m] - yy) ** 2).mean()) for a in agents}
        rec['audit_ok'] = o['audit_ok']; rec['_bft'] = ((np.array(o['p_ft'])[m] - yy) ** 2).tolist(); rec['_blw'] = ((np.array(o['p_lw'])[m] - yy) ** 2).tolist(); rec['_dates'] = g['date'][m].astype(str).tolist()
        per.append(rec); print(scn, {k: v for k, v in rec.items() if not k.startswith('_')}, flush=True)
    port = {k: pd.concat(v, axis=1).mean(axis=1) for k, v in daily.items()}
    d_all = np.concatenate([np.array(r['_bft']) - np.array(r['_blw']) for r in per]); dates = pd.to_datetime(sum([r['_dates'] for r in per], []))
    out[scn] = dict(per_asset=[{k: v for k, v in r.items() if not k.startswith('_')} for r in per],
                    portfolio={k: stats(v.values) for k, v in port.items()},
                    exposure={k: float(np.mean([r['exposure_' + k[2:]] for r in per])) for k in ['p_eq', 'p_lw', 'p_ft']},
                    diff_ft_lw_monthly_ci=block_ci(d_all, dates))
    print(scn, out[scn]['portfolio'], out[scn]['exposure'], out[scn]['diff_ft_lw_monthly_ci'], flush=True)
(D / 'nse_results.json').write_text(json.dumps(out, indent=1))
