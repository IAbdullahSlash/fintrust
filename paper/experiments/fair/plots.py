"""Figures for the revised paper: synthetic influence band (30 seeds) and FI-2010 S3 influence trace."""
import json, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from engine import ft, lw, corrupt_matrix
D = Path(__file__).resolve().parent; F = D.parent.parent / 'figures'
plt.rcParams.update({'font.family': 'serif', 'font.size': 8})
tr = json.loads((D / 'synth_traces.json').read_text())
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
for a, sc, title in zip(ax, ['inversion', 'transient'], ['(a) Persistent inversion', '(b) Transient inversion, ends at t=3000']):
    for m, col, lab in [('LW-tuned', '#888888', 'Loss weighting (tuned)'), ('FT-tuned', '#1f4e8c', 'FinTrust (tuned)')]:
        d = tr[sc][m]; t = np.arange(len(d['mean'])) * 10
        a.fill_between(t, d['lo'], d['hi'], color=col, alpha=.2, lw=0); a.plot(t, d['mean'], color=col, lw=1.1, label=lab)
    a.axvline(2000, color='#b22222', ls='--', lw=.8)
    if sc == 'transient': a.axvline(3000, color='#2e7d32', ls=':', lw=.9)
    a.set_title(title, fontsize=8); a.set_xlabel('decision step'); a.set_xlim(500, 4000); a.set_ylim(0, .6); a.grid(alpha=.25, lw=.5)
ax[0].set_ylabel('influence on agent D'); h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc='upper center', ncol=2, frameon=False, fontsize=7)
fig.tight_layout(rect=(0, 0, 1, .9)); fig.savefig(F / 'synth_band.pdf')
r = json.loads((D / 'fi_fair_S3.json').read_text()); z = dict(np.load(D / 'fi_S3_forecasts.npz'))
P = np.stack([z[a] for a in ['LOGIT40', 'GBT', 'MLP']], 1); y = z['y'].astype(int); day = z['day']; t0 = int(np.argmax(day == 9))
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
for a, kind, title in zip(ax, ['inversion', 'overconfident'], ['(a) MLP inverted from day 9', '(b) MLP overconfident from day 9']):
    Q, _ = corrupt_matrix(P, (2, kind, t0, None))
    o = ft(Q, y, 10, **r['best']['FT']['params']); l = lw(Q, y, 10, **r['best']['LW']['params'])
    x = np.arange(len(y)); s = slice(t0 - 20000, len(y), 50)
    sm = lambda v: np.convolve(v, np.ones(1000) / 1000, mode='same')
    l['W'][:, 2] = sm(l['W'][:, 2]); o['W'][:, 2] = sm(o['W'][:, 2])
    a.plot(x[s] - t0, l['W'][s, 2], color='#888888', lw=.8, label='Loss weighting (tuned)')
    a.plot(x[s] - t0, o['W'][s, 2], color='#1f4e8c', lw=1.0, label='FinTrust (tuned)')
    for tt, k, old, new in o['trans']:
        if k == 2 and tt >= t0 - 20000: a.axvline(tt - t0, color={'QUARANTINED': '#b22222', 'PROBATION': '#e08a00', 'ACTIVE': '#2e7d32'}.get(new, 'k'), lw=.6, ls=':')
    a.axvline(0, color='#b22222', ls='--', lw=.8); a.axvline(int(np.argmax(day == 10)) - t0, color='k', lw=.5, alpha=.5)
    a.set_title(title, fontsize=8); a.set_xlabel('samples from start of day 9 (day 10 marked)'); a.set_ylim(0, 1); a.grid(alpha=.25, lw=.5)
ax[0].set_ylabel('influence on MLP (1000-sample mean)'); h, l_ = ax[0].get_legend_handles_labels(); fig.legend(h, l_, loc='upper center', ncol=2, frameon=False, fontsize=7)
fig.tight_layout(rect=(0, 0, 1, .9)); fig.savefig(F / 'fi_influence.pdf'); fig.savefig(F / 'fi_influence.png', dpi=150)
print('ok')
