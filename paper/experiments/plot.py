"""Fig: influence assigned to the faulty agent D (seed 0). Synthetic data."""
import json
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
D = Path(__file__).resolve().parent
rows = json.loads((D / 'results_raw.json').read_text())
get = lambda s: next(r for r in rows if r['scenario'] == s and r['seed'] == 0)
plt.rcParams.update({'font.family': 'serif', 'font.size': 8})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
for ax, scn, title in zip(axes, ['inversion', 'transient'], ['(a) Persistent sign inversion', '(b) Transient inversion, cleared at t=3000']):
    r = get(scn); t = [500 + 10 * i for i in range(len(r['trace_wD_ft']))]
    ax.plot(t, r['trace_wD_hedge'], color='#888888', lw=1.0, label='Loss-weighted softmax (no states)')
    ax.plot(t, r['trace_wD_ft'], color='#1f4e8c', lw=1.2, label='FinTrust governor')
    ax.axvline(2000, color='#b22222', ls='--', lw=.8); ax.text(2030, .40, 'fault', color='#b22222')
    if scn == 'transient': ax.axvline(3000, color='#2e7d32', ls=':', lw=.9); ax.text(3030, .40, 'repaired', color='#2e7d32')
    ax.set_title(title, fontsize=8); ax.set_xlabel('decision step'); ax.set_xlim(500, 4000); ax.set_ylim(0, .45)
    ax.grid(alpha=.25, lw=.5)
axes[0].set_ylabel('influence on agent D')
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc='upper center', ncol=2, frameon=False, fontsize=7)
fig.tight_layout(rect=(0, 0, 1, .9)); fig.savefig(D.parent / 'figures' / 'fault_weights.pdf'); fig.savefig(D.parent / 'figures' / 'fault_weights.png', dpi=200)
print('ok')
