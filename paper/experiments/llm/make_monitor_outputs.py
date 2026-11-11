"""Table and figure for the behaviour-monitor study, from monitors_results.json.

usage: python make_monitor_outputs.py  -> ../../tabMON.tex, ../../figures/monitor_swap.pdf/.png
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; PAPER = HERE.parent.parent; FAIR = HERE.parent / 'fair'
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(FAIR))
from monitors import cusum, psi_series, R1, ONSET  # noqa: E402
from run_llm import load_outputs  # noqa: E402

r = json.loads((HERE / 'monitors_results.json').read_text())
NAMES = {'E1_llama8b': 'E1: Llama-3.1-8B', 'E2_llama8b_signals': 'E2: Llama-3.1-8B + signals', 'E3_phi3': 'E3: Phi-3-medium'}
FAULTS = [('news_alarm', 'Alarm headline'), ('news_calm', 'Calm headline'), ('persona', 'Persona drift'), ('swap_1b', 'Swap to 1B'),
          ('inversion', 'Inversion'), ('stuck', 'Stuck at 0.95'), ('overconfident', 'Overconfident'), ('clean', 'Clean (false alarms)')]

def cell(s):
    return '--' if s['assets'] == 0 else f"{s['assets']} ({s['median_days']:g})"

lines, tot = [], {m: [0, 0, []] for m in ('L1', 'L2', 'B1', 'B2')}
for c, cname in NAMES.items():
    lines.append(r'\multicolumn{7}{@{}l}{\emph{' + cname + r'}}\\')
    for f, fname in FAULTS:
        s = r['configs'][c]['scenarios'][f]; ch = s['mean_abs_change_vs_clean']
        b2 = s['B2_probe_alarm_share']['10']
        if f == 'clean': b2s = '0'
        elif f.startswith('news'): b2s = 'n/a'
        else: b2s = f'{100 * b2:.0f}\\%'
        if ch < 1e-9 and f != 'clean':
            lines.append(f'{fname} & 0 & \\multicolumn{{5}}{{c}}{{answers unchanged: no monitor can or needs to see it}}\\\\'); continue
        lines.append(f"{fname} & {'--' if f == 'clean' else f'{ch:.2f}'} & {cell(s['L1'])} & {cell(s['L2'])} & {cell(s['B1'])} & {b2s} & {cell(s['B1_rule'])}\\\\")
        if f != 'clean':
            for m in ('L1', 'L2', 'B1'):
                tot[m][0] += s[m]['assets']; tot[m][1] += 6; tot[m][2] += [d for d in s[m]['per_asset'] if d is not None]
            tot['B2'][0] += 6 * b2; tot['B2'][1] += 6
lines.append(r'\midrule')
def tc(m): a, n, d = tot[m]; return f"{a:.0f}/{n} ({np.median(d):g})" if d else f"{a:.0f}/{n}"
lines.append(f"All faults that change the answers & & {tc('L1')} & {tc('L2')} & {tc('B1')} & {tot['B2'][0]:.0f}/{tot['B2'][1]} (1) & \\\\")
(PAPER / 'tabMON.tex').write_text('\n'.join(lines) + '\n')
summary = {m: dict(caught=round(tot[m][0]), of=tot[m][1], median_days=float(np.median(tot[m][2])) if tot[m][2] else 1.0) for m in tot}
(HERE / 'monitors_summary.json').write_text(json.dumps(summary, indent=1)); print(summary)

# figure: experiment 2, silent swap, one instrument
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt  # noqa: E402
lab = pd.read_csv(FAIR / 'nse_vol_forecasts.csv', index_col=0, parse_dates=True); o = load_outputs(HERE / 'outputs')
thr = r['configs']['E2_llama8b_signals']['thresholds']; asset = '^NSEI'
def ser(tag, cond):
    s = o[(o.tag == tag) & (o.condition == cond) & (o.asset == asset)].set_index('date')['p'].astype(float); return s.fillna(.5)
clean = ser('llama31_8b', 'feat_clean'); clean = clean[clean.index >= R1[0]]; swap = ser('llama32_1b', 'feat_clean')
p = clean.copy(); p[p.index >= ONSET] = swap.reindex(p.index[p.index >= ONSET]).values
y = lab[lab.asset == asset]['y'].reindex(p.index).values; idx = p.index; ev = idx >= ONSET; r1 = (idx >= R1[0]) & (idx <= R1[1])
b0 = (clean.values - y) ** 2; mu, sd = b0[r1].mean(), b0[r1].std()
b = (p.values - y) ** 2; bl = np.r_[np.nan, b[:-1]]; k = np.where(ev)[0]
L1 = cusum(bl[k], mu, sd, False) / thr['L1']; L2 = cusum(bl[k], mu, sd, True) / thr['L2']
B1 = np.minimum(psi_series(p.values, p.values[r1])[k] / thr['B1'], 3.1)
fig, ax = plt.subplots(2, 1, figsize=(3.4, 3.2), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.3]))
d = idx[k]
ax[0].plot(d, pd.Series(clean.reindex(d).values).rolling(5, min_periods=1).mean(), color='#555', lw=1, label='original 8B model')
ax[0].plot(d, pd.Series(p.values[k]).rolling(5, min_periods=1).mean(), color='#c0392b', lw=1, label='after silent swap to 1B')
ax[0].set_ylabel('forecast\n(5-day mean)', fontsize=7); ax[0].legend(fontsize=6, frameon=False, loc='center right'); ax[0].tick_params(labelsize=6)
ax[1].plot(d, B1, color='#2471a3', lw=1.2, label='answer drift (PSI)')
ax[1].plot(d, L1, color='#7d3c98', lw=1, ls='--', label='loss CUSUM, one-sided')
ax[1].plot(d, L2, color='#d68910', lw=1, ls=':', label='loss CUSUM, two-sided')
ax[1].axhline(1, color='k', lw=.6); ax[1].set_ylabel('statistic / threshold', fontsize=7); ax[1].set_ylim(-0.1, 3.2); ax[1].text(d[len(d)//2], 1.05, 'alarm threshold', fontsize=6, ha='center', va='bottom')
ax[1].legend(fontsize=6, frameon=False, loc='center right', bbox_to_anchor=(1, .66)); ax[1].tick_params(labelsize=6)
fig.autofmt_xdate(); fig.tight_layout(pad=.3)
for ext in ('pdf', 'png'): fig.savefig(PAPER / 'figures' / f'monitor_swap.{ext}', dpi=200)
print('alarm days: B1', int(np.argmax(B1 > 1)) if (B1 > 1).any() else None, 'L1', int(np.argmax(L1 > 1)) if (L1 > 1).any() else None,
      'L2', int(np.argmax(L2 > 1)) if (L2 > 1).any() else None)

# exploratory sensitivity table (monitors_sensitivity.json)
sp = HERE / 'monitors_sensitivity.json'
if sp.exists():
    sens = json.loads(sp.read_text()); L = []
    for key, s in sens.items():
        w, mlt = key.split('_'); w = w[3:]; mlt = mlt[1:]
        c = lambda m: f"{s[m]['caught']} ({s[m]['median_days']:g}) / {s[m]['clean_false_alarms']}"
        L.append(f"{w} & $\\times${float(mlt):g} & {c('L1')} & {c('L2')} & {c('B1')}\\\\")
    (PAPER / 'tabSENS.tex').write_text('\n'.join(L) + '\n')
