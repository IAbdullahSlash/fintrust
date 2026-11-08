"""Governance replay for the LLM study (protocol: PREREGISTRATION.md).

Reads the LLM answers in outputs/*.jsonl, adds the LLM agent to the four NSE volatility agents of
../fair/nse_vol_forecasts.csv, builds every fault scenario that has the outputs it needs, runs the combiners with the
hyper-parameters already chosen in ../fair/nse_vol_results.json, and writes llm_results.json.

usage: python run_llm.py [--outputs outputs] [--main llama31_8b] [--swap llama32_1b groq_gptoss120b zoho_qwen]
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; FAIR = HERE.parent / 'fair'
sys.path.insert(0, str(FAIR))
from engine import ft, lw, fixed_share, dwm, ew, oracle_excl, corrupt_matrix  # noqa: E402

AG = ['RV', 'RANGE', 'GBT', 'VIX']; LLM = len(AG)
STUDY = ('2024-01-01', '2025-12-29'); ONSET = pd.Timestamp('2025-01-01')

def load_outputs(d):
    rows = []
    for f in sorted(Path(d).glob('*__*.jsonl')):
        rows += [json.loads(l) for l in open(f) if l.strip()]
    if not rows: return pd.DataFrame(columns=['tag', 'condition', 'asset', 'date', 'p', 'parse_ok', 'raw'])
    o = pd.DataFrame(rows); o['date'] = pd.to_datetime(o['date']); return o.drop_duplicates(['tag', 'condition', 'asset', 'date'], keep='last')

def series(o, tag, cond):
    s = o[(o.tag == tag) & (o.condition == cond)]
    return {a: g.set_index('date')['p'] for a, g in s.groupby('asset')}

def fill(out, Q):
    p = out['p'].copy(); m = np.isnan(p); p[m] = Q[m].mean(1); return p

def block_boot(diffs, block=20, n=2000, seed=0):
    """diffs: list of per-asset arrays (loss_FT - loss_LW per day). Moving-block bootstrap of the pooled mean."""
    rng = np.random.default_rng(seed); means = []
    for _ in range(n):
        tot = []
        for d in diffs:
            L = len(d); k = int(np.ceil(L / block)); st = rng.integers(0, max(1, L - block + 1), k)
            tot.append(np.concatenate([d[s:s + block] for s in st])[:L])
        means.append(np.concatenate(tot).mean())
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]

def main(a):
    lab = pd.read_csv(FAIR / 'nse_vol_forecasts.csv', index_col=0, parse_dates=True)
    lab = lab[(lab.index >= STUDY[0]) & (lab.index <= STUDY[1])]
    best = json.loads((FAIR / 'nse_vol_results.json').read_text())['best']
    o = load_outputs(a.outputs); res = dict(protocol='PREREGISTRATION.md', hyper=best, main=a.main, available={}, agents={}, scenarios={})
    res['available'] = {f'{t}:{c}': int(n) for (t, c), n in o.groupby(['tag', 'condition']).size().items()}
    res['parse_fail'] = {f'{t}:{c}': float(1 - g['parse_ok'].mean()) for (t, c), g in o.groupby(['tag', 'condition'])}
    main_clean = series(o, a.main, 'clean')
    if not main_clean: sys.exit(f'no outputs for {a.main}:clean in {a.outputs}')

    # LLM column for 2024 is always the clean main model; from ONSET it is replaced according to the scenario.
    def llm_col(asset, idx, late):
        base = main_clean[asset].reindex(idx)
        if late is not None: base[idx >= ONSET] = late[asset].reindex(idx[idx >= ONSET]).values
        return base.values
    scen = {'clean': None}
    for c, key in [('news_alarm', 'F1_news_alarm'), ('news_calm', 'F2_news_calm'), ('persona', 'F3_persona')]:
        s = series(o, a.main, c)
        if s: scen[key] = s
    for t in a.swap:
        s = series(o, t, 'clean')
        if s: scen[f'F4_swap_{t}'] = s
    for k in ['inversion', 'stuck', 'overconfident']: scen[f'OUT_{k}'] = k

    start = ['ACTIVE'] * len(AG) + ['SHADOW']
    methods = {'EW': None, 'LW': (lw, best['LW']), 'FS': (fixed_share, best['FS']), 'DWM': (dwm, best['DWM']), 'FT': (ft, best['FT'])}
    for name, late in scen.items():
        rows, diffs, det, detf, adm, fq, cov, miss = {}, [], [], [], [], 0, [], []
        for asset, g in lab.groupby('asset', sort=False):
            idx = g.index; y = g['y'].values.astype(int); k0 = int((idx < ONSET).sum()); ev = idx >= ONSET
            col = llm_col(asset, idx, late if isinstance(late, dict) else None)
            if np.isnan(col).any(): miss.append(f'{asset}:{int(np.isnan(col).sum())}'); continue
            P = np.column_stack([g[AG].values, col])
            if isinstance(late, str): Q, M = corrupt_matrix(P, (LLM, late, k0, None))
            else:
                Q = P; M = np.zeros_like(P, bool)
                if late is not None: M[k0:, LLM] = True
            losses = {}
            for with_llm in (True, False):
                X = Q if with_llm else Q[:, :LLM]; sfx = '' if with_llm else '-noLLM'
                for m, spec in methods.items():
                    if spec is None: p = ew(X)
                    elif spec[0] is ft:
                        out = ft(X, y, 1, start=start if with_llm else 'ACTIVE', **spec[1]); p = fill(out, X)
                        if with_llm:
                            tr = out['trans']; cov.append(float(out['cov'][ev].mean()))
                            adm.append(next((str(idx[t].date()) for t, k, x, n in tr if k == LLM and x == 'SHADOW' and n == 'ACTIVE'), None))
                            det.append(next((int(t - k0) for t, k, x, nn in tr if k == LLM and nn == 'QUARANTINED' and t >= k0), None) if late is not None else None)
                            fq += sum(1 for t, k, x, nn in tr if nn == 'QUARANTINED' and ev[t] and k != LLM)
                    else: p = spec[0](X, y, 1, **spec[1])['p']
                    l = (p[ev] - y[ev]) ** 2; rows.setdefault(m + sfx, []).extend(l.tolist()); losses[m + sfx] = l
                if with_llm:   # secondary arm: the operator force-admits the LLM (starts ACTIVE), so fault detection can be measured
                    out = ft(X, y, 1, start='ACTIVE', **best['FT']); p = fill(out, X)
                    rows.setdefault('FT-forced', []).extend(((p[ev] - y[ev]) ** 2).tolist())
                    detf.append(next((int(t - k0) for t, k, x, nn in out['trans'] if k == LLM and nn == 'QUARANTINED' and t >= k0), None) if late is not None else None)
            rows.setdefault('ORACLE', []).extend(((oracle_excl(Q, M)[ev] - y[ev]) ** 2).tolist())
            diffs.append(losses['FT'] - losses['LW'])
            if name == 'clean':
                res['agents'][asset] = {nm: float(np.mean((P[ev, i] - y[ev]) ** 2)) for i, nm in enumerate(AG + ['LLM'])}
                res['agents'][asset]['LLM_2024'] = float(np.mean((P[~ev, LLM] - y[~ev]) ** 2)); res['agents'][asset]['base_rate_2025'] = float(y[ev].mean())
        if miss: res['scenarios'][name] = dict(skipped='missing LLM answers: ' + ', '.join(miss)); print(name, 'skipped', miss); continue
        res['scenarios'][name] = dict(brier={m: float(np.mean(v)) for m, v in rows.items()}, ft_minus_lw_ci=block_boot(diffs),
                                      llm_admitted=adm, detect_days=det, detect_days_forced=detf, false_quarantines=fq, coverage=float(np.mean(cov)))
        print(name, {m: round(float(np.mean(v)), 4) for m, v in rows.items()}, 'admitted', adm, 'detect', det, 'forced', detf, flush=True)

    # leakage probe: named minus anonymised Brier, before and after the model's knowledge cutoff
    full = pd.read_csv(FAIR / 'nse_vol_forecasts.csv', index_col=0, parse_dates=True)
    def brier_of(cond):
        s = series(o, a.main, cond); L = []
        for asset, p in s.items():
            y = full[full.asset == asset]['y'].reindex(p.index); L += ((p - y) ** 2).dropna().tolist()
        return float(np.mean(L)) if L else None
    b = {c: brier_of(c) for c in ['clean_pre', 'named_pre', 'clean', 'named']}
    res['leakage'] = dict(brier=b, gap_pre=(b['named_pre'] - b['clean_pre']) if b['named_pre'] and b['clean_pre'] else None,
                          gap_post=(b['named'] - b['clean']) if b['named'] and b['clean'] else None)

    # hallucinated-reasoning diagnostic on the 'rationale' condition
    rq = Path(a.requests) / 'rationale.jsonl'; rat = o[(o.tag == a.main) & (o.condition == 'rationale')]
    if len(rat) and rq.exists():
        prompts = {json.loads(l)['id']: json.loads(l)['user'] for l in open(rq)}; cited = hall = rows_h = 0
        for _, r in rat.iterrows():
            pid = f"rationale|{r.asset}|{r.date:%Y-%m-%d}"; reason = re.split(r'Probability', str(r.raw), flags=re.I)[0]
            nums = [float(x) for x in re.findall(r'(?<![\d.])\d+(?:\.\d+)?', reason)]
            nums = [x for x in nums if not (x.is_integer() and x <= 20)]      # day counts / window lengths are structural
            pnums = [float(x) for x in re.findall(r'\d+(?:\.\d+)?', prompts.get(pid, ''))]
            bad = [x for x in nums if not any(abs(x - q) <= .011 for q in pnums)]
            cited += len(nums); hall += len(bad); rows_h += bool(bad)
        res['hallucination'] = dict(answers=int(len(rat)), numbers_cited=cited, numbers_not_in_prompt=hall,
                                    share_answers_with_unsupported_number=rows_h / len(rat))
    (Path(a.out)).write_text(json.dumps(res, indent=1)); print('wrote', a.out)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--outputs', default=str(HERE / 'outputs')); ap.add_argument('--requests', default=str(HERE / 'requests'))
    ap.add_argument('--main', default='llama31_8b'); ap.add_argument('--swap', nargs='*', default=['llama32_1b', 'groq_gptoss120b', 'zoho_qwen'])
    ap.add_argument('--out', default=str(HERE / 'llm_results.json'))
    main(ap.parse_args())
