"""Behaviour monitors versus loss monitors for the LLM agent (protocol: MONITORS_ADDENDUM.md).

usage: python monitors.py            -> monitors_results.json
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; FAIR = HERE.parent / 'fair'
sys.path.insert(0, str(FAIR)); sys.path.insert(0, str(HERE))
from engine import corrupt_matrix  # noqa: E402
from run_llm import load_outputs  # noqa: E402

R1 = ('2024-01-01', '2024-06-30'); R2 = ('2024-07-01', '2024-12-31'); ONSET = pd.Timestamp('2025-01-01')
CONFIGS = {'E1_llama8b': ('llama31_8b', ''), 'E2_llama8b_signals': ('llama31_8b', 'feat_'), 'E3_phi3': ('phi3_medium', '')}
SWAP = 'llama32_1b'; WIN = 20; BINS = np.linspace(0, 1, 11); PSI_RULE = .25; TOL = .10; KS = (5, 10, 20); NSETS = 1000

def hist(p):
    c, _ = np.histogram(np.clip(p, 0, 1), BINS); c = c + .5; return c / c.sum()

def psi_series(p, ref):
    r = hist(ref); out = np.full(len(p), np.nan)
    for t in range(WIN - 1, len(p)):
        q = hist(p[t - WIN + 1:t + 1]); out[t] = float(np.sum((q - r) * np.log(q / r)))
    return out

def cusum(b, mu, sd, two_sided):
    """b: daily losses aligned so that b[t] is usable on day t (already lagged). Returns max(S+, S-) path."""
    sp = sm = 0.; out = np.zeros(len(b))
    for t, x in enumerate(b):
        if np.isnan(x): out[t] = max(sp, sm); continue
        z = (x - mu) / sd; sp = max(0., sp + z - .5)
        if two_sided: sm = max(0., sm - z - .5)
        out[t] = max(sp, sm)
    return out

def fmt_series(ok):
    s = pd.Series(~ok.astype(bool)).rolling(WIN, min_periods=WIN).mean().values; return s

def first(alarm, idx):
    ev = np.where((idx >= ONSET) & alarm)[0]
    return int(ev[0] - np.searchsorted(idx, ONSET)) if len(ev) else None

def main():
    lab = pd.read_csv(FAIR / 'nse_vol_forecasts.csv', index_col=0, parse_dates=True)
    o = load_outputs(HERE / 'outputs')
    def get(tag, cond):
        s = o[(o.tag == tag) & (o.condition == cond)]
        return {a: g.set_index('date')[['p', 'parse_ok']] for a, g in s.groupby('asset')}
    res = dict(protocol='MONITORS_ADDENDUM.md', configs={})
    for cname, (tag, pre) in CONFIGS.items():
        clean = get(tag, pre + 'clean')
        late = {'clean': None, 'news_alarm': get(tag, pre + 'news_alarm'), 'news_calm': get(tag, pre + 'news_calm'),
                'persona': get(tag, pre + 'persona'), 'swap_1b': get(SWAP, pre + 'clean' if pre else 'clean'),
                'inversion': 'inversion', 'stuck': 'stuck', 'overconfident': 'overconfident'}
        # per asset clean streams and reference statistics
        A = {}
        for asset, g in clean.items():
            y = lab[lab.asset == asset]['y']; idx = g.index[(g.index >= R1[0])]; g = g.loc[idx]
            yy = y.reindex(idx).values; p = g['p'].astype(float).values; ok = g['parse_ok'].values
            p = np.where(np.isnan(p), .5, p)        # unparsed answers count as 0.5 for loss and PSI; B3 sees them
            r1 = (idx >= R1[0]) & (idx <= R1[1]); r2 = (idx >= R2[0]) & (idx <= R2[1])
            b = (p - yy) ** 2; A[asset] = dict(idx=idx, y=yy, p=p, ok=ok, r1=r1, r2=r2, mu=b[r1].mean(), sd=max(b[r1].std(), 1e-6))
        def stats(asset, p, ok):
            a = A[asset]; b = (p - a['y']) ** 2; bl = np.r_[np.nan, b[:-1]]          # label known one day late
            return dict(L1=cusum(bl, a['mu'], a['sd'], False), L2=cusum(bl, a['mu'], a['sd'], True),
                        B1=psi_series(p, a['p'][a['r1']]), B3=fmt_series(ok))
        # thresholds on R2: restart statistics at R2 start
        thr = {}
        for m in ('L1', 'L2', 'B1'):
            mx = 0.
            for asset, a in A.items():
                sl = a['r2'] | (a['idx'] < R2[0])        # keep R1 for PSI windows, restart CUSUM at R2
                st = stats(asset, a['p'], a['ok'])
                if m == 'B1': v = np.nanmax(st[m][a['r2']])
                else:
                    b = (a['p'] - a['y']) ** 2; bl = np.r_[np.nan, b[:-1]]; k = np.where(a['r2'])[0]
                    v = cusum(bl[k], a['mu'], a['sd'], m == 'L2').max()
                mx = max(mx, float(v))
            thr[m] = mx
        thr['B1_rule'] = PSI_RULE; thr['B3'] = .05
        out = dict(thresholds=thr, scenarios={})
        rng = np.random.default_rng(0)
        for sname, lt in late.items():
            if isinstance(lt, dict) and not lt: out['scenarios'][sname] = 'missing outputs'; continue
            row = {m: [] for m in ('L1', 'L2', 'B1', 'B1_rule', 'B3')}; probe = {K: [] for K in KS}; mae = []
            for asset, a in A.items():
                idx = a['idx']; ev = idx >= ONSET; p = a['p'].copy(); ok = a['ok'].copy()
                if isinstance(lt, dict):
                    g = lt[asset].reindex(idx[ev]); q = g['p'].astype(float).values
                    ok[ev] = g['parse_ok'].fillna(False).values; p[ev] = np.where(np.isnan(q), .5, q)
                elif isinstance(lt, str):
                    p = corrupt_matrix(p[:, None], (0, lt, int((~ev).sum()), None))[0][:, 0]
                st = stats(asset, p, ok)
                # for CUSUMs, restart at the evaluation start (fair to the loss monitors)
                k = np.where(ev)[0]; b = (p - a['y']) ** 2; bl = np.r_[np.nan, b[:-1]]
                L1 = np.zeros(len(p)); L2 = np.zeros(len(p))
                L1[k] = cusum(bl[k], a['mu'], a['sd'], False); L2[k] = cusum(bl[k], a['mu'], a['sd'], True)
                row['L1'].append(first(L1 > thr['L1'], idx)); row['L2'].append(first(L2 > thr['L2'], idx))
                row['B1'].append(first(st['B1'] > thr['B1'], idx)); row['B1_rule'].append(first(st['B1'] > PSI_RULE, idx))
                row['B3'].append(first(st['B3'] > .05, idx))
                # probes: drawn from this asset's 2025 prompts; recorded answers = clean configuration
                pc, pf, okf = a['p'][ev], p[ev], ok[ev]; mae.append(float(np.mean(np.abs(pf - pc))))
                for K in KS:
                    hits = 0
                    for _ in range(NSETS):
                        j = rng.choice(len(pc), K, replace=False)
                        if sname in ('news_alarm', 'news_calm'): continue      # injected via live feed, not via probes
                        hits += (np.mean(np.abs(pf[j] - pc[j])) > TOL) or (not okf[j].all())
                    probe[K].append(hits / NSETS)
            summ = {}
            for m, v in row.items():
                d = [x for x in v if x is not None]
                summ[m] = dict(assets=len(d), median_days=float(np.median(d)) if d else None, per_asset=v)
            summ['B2_probe_alarm_share'] = {str(K): float(np.mean(v)) for K, v in probe.items()}
            summ['mean_abs_change_vs_clean'] = float(np.mean(mae))
            out['scenarios'][sname] = summ
            print(cname, sname, {m: (s['assets'], s['median_days']) for m, s in summ.items() if isinstance(s, dict) and 'assets' in s},
                  'B2', summ['B2_probe_alarm_share'], flush=True)
        res['configs'][cname] = out
    (HERE / 'monitors_results.json').write_text(json.dumps(res, indent=1)); print('wrote monitors_results.json')

if __name__ == '__main__':
    main()
