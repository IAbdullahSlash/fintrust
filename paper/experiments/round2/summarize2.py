"""Prints the round-2 tables from r2_*.json."""
import sys, json, numpy as np
from pathlib import Path
from math import comb
R = Path(__file__).resolve().parent
M = ['LW', 'FS', 'DWM', 'LWC', 'STK', 'FT', 'FTB', 'LWR', 'FTR']
def sign_p(w, n):  # two-sided sign test
    k = max(w, n - w); return min(1., 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)
for name in sys.argv[1:]:
    r = json.loads((R / f'r2_{name}.json').read_text()); print(f'===== {name}  n_sel={r["n_sel"]} n_test={r["n_test"]} best1={r["best_single"]}')
    for mode in ['clean', 'fault']:
        print(f'-- tuned: {mode}', {m: r['best'][mode][m] for m in M}, 'rec_lr', r['best'][mode]['REC_lr'])
        print('scen'.ljust(14), 'EW     ', '  '.join(f'{m:6s}' for m in M), 'ORACLE  FT-LW [CI]   detect cov healthyQ')
        for sc, d in r['fixed'].items():
            b = d['brier']; a = d['aux']; ci = a[f'ci|{mode}']; fa = a[f'FT|{mode}']
            print(sc.ljust(14), f"{b['EW']:.4f}", '  '.join(f"{b[f'{m}|{mode}']:.4f}" for m in M), f"{b['ORACLE']:.4f}",
                  f"{ci[0]:+.4f} [{ci[1]:+.4f},{ci[2]:+.4f}]", fa['detect'], round(fa['coverage'], 3), len(fa['healthy_q']))
        for kind, ds in r['random'].items():
            d = np.array([x['brier'][f'FT|{mode}'] - x['brier'][f'LW|{mode}'] for x in ds]); w = int((d < 0).sum())
            bm = np.array([min(x['brier'][f'{m}|{mode}'] for m in M if m not in ('FT',)) for x in ds])
            dbest = np.array([x['brier'][f'FT|{mode}'] for x in ds]) - bm
            per = {m: round(float(np.mean([x['brier'][f'{m}|{mode}'] for x in ds])), 4) for m in M}
            det = [x['aux'][f'FT|{mode}']['detect'] for x in ds]; hq = sum(len(x['aux'][f'FT|{mode}']['healthy_q']) for x in ds)
            se = d.std(ddof=1) / np.sqrt(len(d)); print(f'  random {kind:13s} FT-LW mean {d.mean():+.4f} 95%CI[{d.mean()-2.262*se:+.4f},{d.mean()+2.262*se:+.4f}] wins {w}/10 sign p={sign_p(w,10):.3f} | FT-best other {dbest.mean():+.4f} wins {(dbest<0).sum()} | det med {np.median([x for x in det if x is not None]) if any(x is not None for x in det) else None} ({sum(x is not None for x in det)}/10) hq {hq} | means {per}')
    print('-- ref', {sc: (round(d['brier']['LW-ref'], 4), round(d['brier']['FT-ref'], 4), round(d['brier']['BEST1'], 4)) for sc, d in r['fixed'].items()})
    print('-- lastday clean-tuned', r['best'].get('clean_lastday'), {sc: {k: round(v, 4) for k, v in d.items()} for sc, d in r['lastday'].items()})
    for sc, g in r['grid'].items():
        lw_, ft_ = np.array(g['LW']), np.array(g['FT'])
        print(f'-- grid {sc}: LW min {lw_.min():.4f} med {np.median(lw_):.4f} max {lw_.max():.4f} | FT min {ft_.min():.4f} med {np.median(ft_):.4f} max {ft_.max():.4f} | FT<LW in {(ft_<lw_).sum()}/{len(lw_)}')
