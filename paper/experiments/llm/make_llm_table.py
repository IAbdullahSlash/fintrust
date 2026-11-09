"""Writes tabLLM.tex (Table: LLM study, 2025 Brier) from the three result files and the model answers."""
import json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
lab = pd.read_csv(HERE.parent / 'fair' / 'nse_vol_forecasts.csv', index_col=0, parse_dates=True)
lab = lab[(lab.index >= '2025-01-01') & (lab.index <= '2025-12-29')]

def llm_brier(tag, cond):
    if cond == 'stuck':
        y = lab['y'].values; return float(np.mean((0.95 - y) ** 2))
    L = []
    for l in open(HERE / 'outputs' / f'{tag}__{cond}.jsonl'):
        r = json.loads(l); L.append((r['asset'], pd.Timestamp(r['date']), r['p']))
    o = pd.DataFrame(L, columns=['asset', 'date', 'p']).drop_duplicates(['asset', 'date'], keep='last')
    m = o.merge(lab.reset_index().rename(columns={lab.index.name or 'index': 'date'})[['date', 'asset', 'y']], on=['date', 'asset'])
    return float(((m.p - m.y) ** 2).mean())

EXPS = [('1: Llama-3.1-8B', 'llm_results.json', 'llama31_8b', '', 'llama32_1b', False),
        ('2: + signals, reasoning', 'llm_results_feat.json', 'llama31_8b', 'feat_', 'llama32_1b', False),
        ('3: Phi-3-medium', 'llm_results_phi3.json', 'phi3_medium', '', 'llama32_1b', True)]
ROWS = [('clean', 'Clean', 'clean'), ('F1_news_alarm', 'Alarm headline', 'news_alarm'), ('F3_persona', 'Persona drift', 'persona'),
        ('F4_swap_llama32_1b', 'Swap to 1B', 'SWAP'), ('OUT_stuck', 'Stuck at 0.95', 'stuck')]
out = []
for title, f, tag, pre, swap, forced_ok in EXPS:
    r = json.loads((HERE / f).read_text())
    out.append(r'\multicolumn{8}{@{}l}{\emph{Experiment ' + title + '}}' + chr(92) * 2)
    for key, name, cond in ROWS:
        s = r['scenarios'][key]; b = s['brier']
        lb = llm_brier(swap, pre + 'clean') if cond == 'SWAP' else llm_brier(tag, cond if cond == 'stuck' else pre + cond)
        adm = sum(x is not None for x in s['llm_admitted'])
        if not forced_ok or key == 'clean': det = '--'
        else:
            d = [x for x in s['detect_days_forced'] if x is not None]
            det = f'{min(d)}--{max(d)} ({len(d)})' if d and min(d) != max(d) else (f'{d[0]} ({len(d)})' if d else 'none')
        out.append(f"{name} & {lb:.3f} & {b['EW']:.4f} & {b['DWM']:.4f} & {b['LW']:.4f} & {b['FT']:.4f} & {adm}/6 & {det}\\\\")
(HERE / 'tabLLM.tex').write_text('\n'.join(out) + '\n'); print('\n'.join(out))
