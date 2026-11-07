"""Tables for the third revision: ablation (abl_*.json), input faults (input_faults.json), volatility targeting
(econ_nse.json) and pooled NSE evidence (pool_nse.json; printed summary for the text). Writes ../../tab{ABL,IF,ECON}.tex."""
import json, numpy as np
from pathlib import Path
R = Path(__file__).resolve().parent; OUT = R.parent.parent
FAULTS = ['inversion', 'stuck', 'overconfident', 'transient']

# ---------------- ablation ----------------
names = {'full': 'FinTrust (full)', 'no_shadow': 'no shadow start', 'no_cap': 'no probation cap', 'one_window': 'one window, not three',
         'no_gap': 'no hysteresis gap', 'operator': 'operator clears all', 'short_window': '$W=50$, not $50m$', 'no_states': 'no lifecycle (LW)'}
A = {n: json.loads((R / f'abl_{n}.json').read_text()) for n in ['S1', 'S2', 'S3', 'BTC']}
def loss(r, v): return np.mean([r['fixed'][s][v]['brier'] for s in FAULTS] + [d[v]['brier'] for k in r['random'] for d in r['random'][k]])
def dets(r, v):
    x = [r['fixed'][s][v].get('detect') for s in FAULTS[:3]] + [d[v].get('detect') for k in r['random'] for d in r['random'][k]]
    return [z for z in x if z is not None], len(x)
def hq(r, v): return (r['fixed']['clean'][v].get('healthy_q') or 0) + sum(d[v].get('healthy_q') or 0 for k in r['random'] for d in r['random'][k])
rows = []
for v, lab in names.items():
    fi = np.mean([loss(A[n], v) for n in ['S1', 'S2', 'S3']]); bt = loss(A['BTC'], v)
    if v == 'no_states': rows.append(f'{lab} & {fi:.4f} & {bt:.4f} & --- & --- & ---\\\\'); continue
    d = [dets(A[n], v) for n in ['S1', 'S2', 'S3']]; dd = sum((x for x, _ in d), []); nn = sum(n for _, n in d)
    h = sum(hq(A[n], v) for n in ['S1', 'S2', 'S3'])
    ab = np.mean([A[n]['fixed'][s][v]['abstain'] for n in ['S1', 'S2', 'S3'] for s in A[n]['fixed']])
    rows.append(f'{lab} & {fi:.4f} & {bt:.4f} & {len(dd)}/{nn}, {int(np.median(dd))} & {h} & {100 * ab:.1f}\\\\')
(OUT / 'tabABL.tex').write_text('\n'.join(rows) + '\n'); print('\n'.join(rows))

# ---------------- input faults ----------------
I = json.loads((R / 'input_faults.json').read_text()); rows = []
cols = ['EW', 'LW|clean', 'FT|clean', 'STK|clean', 'FTR|clean', 'ORACLE']
def row(lab, sc, agent_delta, det, tot):
    m = [np.mean([s['brier'][c] for s in sc]) for c in cols]; lo = int(np.argmin(m))
    cells = [(f'\\textbf{{{x:.4f}}}' if i == lo else f'{x:.4f}') for i, x in enumerate(m)]
    return f'{lab} & {agent_delta} & ' + ' & '.join(cells) + f' & {det}/{tot}\\\\'
for kind, lab in [('delayed', 'FI delayed feed'), ('dropped', 'FI dropped volumes'), ('scaled', 'FI volumes $\\times10$')]:
    sc = [v for n in ['S1', 'S2', 'S3'] for k, v in I[n]['scen'].items() if k.startswith(kind)]
    dl = [v['agent_after'] - v['agent_before'] for v in sc]
    det = sum(1 for v in sc if v['aux']['FT|clean']['detect'][0][0] is not None)
    rows.append(row(lab, sc, f'{min(dl):.3f}--{max(dl):.3f}', det, len(sc)))
N = I['NSE']['scen']
for keys, lab in [(['stale_vix'], 'NSE stale VIX'), ([k for k in N if k.startswith('decay')], 'NSE no refit')]:
    sc = [N[k] for k in keys]; dl = [b - a for s in sc for a, b in s['agents'].values()]
    det = sum(1 for s in sc for inst in s['aux']['FT|clean']['detect'] for x in inst if x is not None); tot = sum(len(inst) for s in sc for inst in s['aux']['FT|clean']['detect'])
    rows.append(row(lab, sc, f'{min(dl):.3f}--{max(dl):.3f}', det, tot))
(OUT / 'tabIF.tex').write_text('\n'.join(rows) + '\n'); print('\n'.join(rows))

# ---------------- volatility targeting ----------------
E = json.loads((R / 'econ_nse.json').read_text())['1.0']; rows = []
meth = ['VT', 'EW', 'LW', 'FS', 'FT', 'FT-B', 'FT-R', 'ORACLE']; key = {'FT-B': 'FTB', 'FT-R': 'FTR'}
for sc, lab in [('clean', 'Clean'), ('inversion', 'Inversion'), ('stuck', 'Stuck'), ('overconfident', 'Overconf.')]:
    cells = [f"{100 * E[sc][key.get(m, m)]['vol']:.1f}/{100 * E[sc][key.get(m, m)]['maxdd']:.1f}" for m in meth]
    rows.append(f'{lab} & ' + ' & '.join(cells) + '\\\\')
(OUT / 'tabECON.tex').write_text('\n'.join(rows) + '\n'); print('\n'.join(rows))
for g in ['0.5', '2.0']:
    e = json.loads((R / 'econ_nse.json').read_text())[g]
    print('gamma', g, {sc: {m: round(100 * e[sc][m]['vol'], 1) for m in ['VT', 'EW', 'LW', 'FT', 'ORACLE']} for sc in e})
print('exposure', {sc: {m: round(E[sc][m]['exposure'], 3) for m in ['VT', 'EW', 'LW', 'FT', 'ORACLE']} for sc in E})

# ---------------- pooled evidence (for the text) ----------------
Pn = json.loads((R / 'pool_nse.json').read_text())
for v in ['per_half', 'pool50_half', 'pool300_half', 'per_clim', 'pool50_clim', 'pool300_clim']:
    for kind in ['inversion', 'stuck', 'overconfident']:
        d = [x for k, s in Pn.items() if k.startswith(kind) for x in s[v]['detect_days']]; ok = [x for x in d if x is not None]
        print(v, kind, f'{len(ok)}/{len(d)}', 'median days', np.median(ok) if ok else None, 'range', (min(ok), max(ok)) if ok else None)
    print(v, 'healthy q', sum(s[v]['healthy_q'] for s in Pn.values()), 'max |brier diff vs per_half|', max(abs(s[v]['brier'] - s['per_half']['brier']) for s in Pn.values()))
