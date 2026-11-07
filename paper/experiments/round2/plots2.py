"""Round-2 figures: tuning robustness over the (rho, tau) grid, and the FI-2010 S1 influence traces."""
import json, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from engine2 import lw_fast, ft2, corrupt_matrix
R = Path(__file__).resolve().parent; F = R.parent.parent / 'figures'
plt.rcParams.update({'font.family': 'serif', 'font.size': 8})
res = {n: json.loads((R / f'r2_{n}.json').read_text()) for n in ['S1', 'S2', 'S3', 'BTC']}
panels = [('S1', 'inversion'), ('S1', 'stuck'), ('S3', 'stuck'), ('BTC', 'inversion')]
fig, ax = plt.subplots(1, 4, figsize=(7.0, 2.1))
for a, (n, sc) in zip(ax, panels):
    g = res[n]['grid'][sc]; RT = res[n]['grid_params']
    for m, col, lab in [('LW', '#888888', 'Loss weighting'), ('FT', '#1f4e8c', 'FinTrust')]:
        v = np.sort(g[m]); a.plot(np.arange(1, len(v) + 1), v, color=col, lw=1.2, label=lab)
        cb = res[n]['best']['clean'][m]; i = RT.index({'decay': cb['decay'], 'temperature': cb['temperature']})
        a.plot(int(np.searchsorted(v, g[m][i])) + 1, g[m][i], 'o', color=col, ms=3.5, mfc='white', mew=1.1)
    a.axhline(res[n]['fixed'][sc]['brier']['ORACLE'], color='#b22222', lw=.7, ls='--')
    a.set_title(f'({"abcd"[panels.index((n, sc))]}) {n.replace("BTC", "BTC/USD")}, {sc}', fontsize=8)
    a.set_xlabel('rank of the 56 settings'); a.grid(alpha=.25, lw=.5)
ax[0].set_ylabel('test Brier loss'); h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=2, frameon=False, fontsize=7); fig.tight_layout(rect=(0, 0, 1, .88))
fig.savefig(F / 'robust_grid.pdf'); fig.savefig(F / 'robust_grid.png', dpi=150)

# S1 influence traces with clean-tuned LW and FT (round-2 parameters)
D = R.parent / 'fair'; z = dict(np.load(D / 'fi_S1_forecasts.npz')); P = np.stack([z[a] for a in ['LOGIT40', 'GBT', 'MLP']], 1)
y = z['y'].astype(int); day = z['day']; t0 = int(np.argmax(day == 5)); b = res['S1']['best']['clean']
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.2), sharey=True)
for a, kind in zip(ax, ['inversion', 'stuck', 'overconfident']):
    Q, _ = corrupt_matrix(P, (2, kind, t0, None))
    o = ft2(Q, y, 10, window=500, minimum=2000, **b['FT']); l = lw_fast(Q, y, 10, **b['LW'])
    sm = lambda v: np.convolve(v, np.ones(1000) / 1000, mode='same'); x = np.arange(len(y)); s = slice(0, len(y), 50)
    a.plot(x[s] - t0, sm(l['W'][:, 2])[s], color='#888888', lw=.8, label='Loss weighting (tuned)')
    a.plot(x[s] - t0, sm(o['W'][:, 2])[s], color='#1f4e8c', lw=1.0, label='FinTrust (tuned)')
    for tt, k, old, new in o['trans']:
        if k == 2 and tt >= t0: a.axvline(tt - t0, color={'QUARANTINED': '#b22222', 'PROBATION': '#e08a00', 'ACTIVE': '#2e7d32'}.get(new, 'k'), lw=.6, ls=':')
    a.axvline(0, color='#b22222', ls='--', lw=.8); a.axvline(int(np.argmax(day == 6)) - t0, color='k', lw=.5, alpha=.5)
    a.set_title(f'({"abc"[["inversion", "stuck", "overconfident"].index(kind)]}) MLP {kind} from day 5', fontsize=8)
    a.set_xlabel('samples from start of day 5'); a.set_ylim(0, 1); a.grid(alpha=.25, lw=.5)
ax[0].set_ylabel('influence on MLP'); h, l_ = ax[0].get_legend_handles_labels()
fig.legend(h, l_, loc='upper center', ncol=2, frameon=False, fontsize=7); fig.tight_layout(rect=(0, 0, 1, .88))
fig.savefig(F / 'fi_influence_s1.pdf'); fig.savefig(F / 'fi_influence_s1.png', dpi=150)
print('ok')
