"""Writes LaTeX table bodies for the round-2 paper sections to ../../tables_r2.tex and prints summary counts."""
import json, numpy as np
from pathlib import Path
R = Path(__file__).resolve().parent; OUT = R.parent.parent / 'tables_r2.tex'
D = ['S1', 'S2', 'S3', 'BTC']; r = {n: json.loads((R / f'r2_{n}.json').read_text()) for n in D}
lab = {'S1': 'FI S1', 'S2': 'FI S2', 'S3': 'FI S3', 'BTC': 'BTC'}
scn = {'clean': 'Clean', 'inversion': 'Inversion', 'stuck': 'Stuck', 'overconfident': 'Overconf.', 'transient': 'Transient'}
A = ['EW', 'LW', 'FS', 'DWM', 'LWC', 'STK', 'FT', 'FTB', 'FTR']
f4 = lambda x: f'{x:.4f}'
def sg(x): return ('$+' if x > 5e-5 else ('$-' if x < -5e-5 else '$')) + f'{abs(x):.4f}$'
lines = ['% Table A: fixed faults, clean tuning rule']
for n in D:
    for i, (sc, sl) in enumerate(scn.items()):
        b = r[n]['fixed'][sc]['brier']; v = {m: (b['EW'] if m == 'EW' else b[f'{m}|clean']) for m in A}; mn = min(v.values())
        ci = r[n]['fixed'][sc]['aux']['ci|clean']
        cells = [('\\textbf{' + f4(v[m]) + '}') if abs(v[m] - mn) < 5e-5 else f4(v[m]) for m in A]
        first = f'\\multirow{{5}}{{*}}{{{lab[n]}}}' if i == 0 else ''
        cis = sg(ci[0]) if sc == 'clean' else f'{sg(ci[0])} [{sg(ci[1])}, {sg(ci[2])}]'
        lines.append(f'{first} & {sl} & ' + ' & '.join(cells) + f' & {f4(b["ORACLE"])} & {cis}\\\\')
    lines.append('\\midrule' if n != 'BTC' else '')
lines.append('% Table B: random draws, both rules')
tot = {}
for n in D:
    for i, kind in enumerate(['inversion', 'stuck', 'overconfident']):
        ds = r[n]['random'][kind]; row = []
        for mode in ['clean', 'fault']:
            lw_ = np.array([x['brier'][f'LW|{mode}'] for x in ds]); ft_ = np.array([x['brier'][f'FT|{mode}'] for x in ds]); d = ft_ - lw_
            w = int((d < -5e-5).sum()); l = int((d > 5e-5).sum()); tot.setdefault((mode, kind), []).append((w, l))
            row += [f4(lw_.mean()), f4(ft_.mean()), f'{sg(d.mean())} ({w}/{l})']
        stk = np.mean([x['brier']['STK|fault'] for x in ds]); ftr = np.mean([x['brier']['FTR|fault'] for x in ds])
        first = f'\\multirow{{3}}{{*}}{{{lab[n]}}}' if i == 0 else ''
        lines.append(f'{first} & {scn[kind]} & ' + ' & '.join(row) + f' & {f4(stk)} & {f4(ftr)}\\\\')
    lines.append('\\midrule' if n != 'BTC' else '')
# NSE table
r1 = json.loads((R.parent / 'fair' / 'nse_vol_results.json').read_text())['scenarios']; r2 = json.loads((R / 'r2_NSE.json').read_text())['scenarios']
lines.append('% Table NSE')
for key, nm in [('clean', 'Clean'), ('inversion:RV', 'Inversion (RV)'), ('inversion:GBT', 'Inversion (GBT)'), ('inversion:VIX', 'Inversion (VIX)'),
                ('stuck:RV', 'Stuck (RV)'), ('stuck:VIX', 'Stuck (VIX)'), ('overconfident:GBT', 'Overconf.\\ (GBT)')]:
    b1 = r1[key]['brier']; b2 = r2[key]['brier']
    v = {'EW': b1['EW'], 'LW': b1['LW-tuned'], 'FS': b1['FS-tuned'], 'DWM': b1['DWM-tuned'], 'STK': b2['STK'], 'FT': b1['FT-tuned'], 'FTB': b2['FTB'], 'FTR': b2['FTR']}
    mn = min(v.values()); lines.append(nm + ' & ' + ' & '.join(('\\textbf{' + f4(x) + '}') if abs(x - mn) < 5e-5 else f4(x) for x in v.values()) + '\\\\')
OUT.write_text('\n'.join(lines) + '\n'); print('wrote', OUT)
print('wins/losses FT vs LW (|d|>=5e-5):', {k: (v, sum(a for a,b in v), sum(b for a,b in v)) for k, v in tot.items()})
# extra summary numbers
for n in D:
    for mode in ['clean', 'fault']:
        ds = sum(r[n]['random'].values(), [])
        ftr_best = sum(1 for x in ds if x['brier'][f'FTR|{mode}'] <= min(x['brier'][f'{m}|{mode}'] for m in ['LW', 'FS', 'DWM', 'LWC', 'FT', 'FTB']) + 1e-12)
        stk_best = sum(1 for x in ds if x['brier'][f'STK|{mode}'] <= min(x['brier'][f'{m}|{mode}'] for m in ['LW', 'FS', 'DWM', 'LWC', 'FT', 'FTB']) + 1e-12)
        ft_vs_excl = sum(1 for x in ds if x['brier'][f'FT|{mode}'] < min(x['brier'][f'{m}|{mode}'] for m in ['LW', 'FS', 'DWM', 'LWC']) - 1e-12)
        print(n, mode, 'FTR beats all non-exploiting', ftr_best, '/30; STK', stk_best, '/30; FT beats LW,FS,DWM,LWC', ft_vs_excl, '/30')
    for sc in ['inversion', 'stuck', 'overconfident', 'transient']:
        print(n, sc, 'fault-rule FT-LW CI', [round(x, 4) for x in r[n]['fixed'][sc]['aux']['ci|fault']], 'detect', r[n]['fixed'][sc]['aux']['FT|clean']['detect'])
    for sc, g in r[n]['grid'].items():
        lw_, ft_ = np.array(g['LW']), np.array(g['FT']); print(n, 'grid', sc, 'FT<LW', int((ft_ < lw_ - 1e-9).sum()), 'LW range', round(lw_.min(), 4), round(lw_.max(), 4), 'FT range', round(ft_.min(), 4), round(ft_.max(), 4))
t = OUT.read_text(); a, rest = t.split('% Table B: random draws, both rules'); b, c = rest.split('% Table NSE')
(OUT.parent / 'tabA.tex').write_text(a.replace('% Table A: fixed faults, clean tuning rule\n', '').rstrip() + '\n')
(OUT.parent / 'tabB.tex').write_text(b.strip() + '\n'); (OUT.parent / 'tabNSE.tex').write_text(c.strip() + '\n')
