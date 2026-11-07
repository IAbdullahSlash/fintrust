"""Round-2 engine: the baselines the first revision promised, and a faster loss-weighting combiner.

lw_fast    exact vectorised version of fair/engine.lw (checked in check2.py)
ft2        fair/engine.ft with a choice of reference forecast for the quality windows:
           ref='half' (constant 0.5, the original rule) or ref='clim' (running base rate pi, reference loss pi(1-pi))
lw_cusum   loss weighting plus a Page CUSUM per agent on its excess loss over 0.25; weight 0 while the alarm is on
stack      online logistic regression on the agents' log-odds (can learn a negative weight, i.e. a sign correction)
recal      per-agent online Platt scaling; the recalibrated matrix is then fed to any combiner
All are causal: the outcome of step s is used only from step s + h on.
"""
import sys, math, numpy as np
from pathlib import Path
from scipy.signal import lfilter
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'fair'))
from engine import corrupt_matrix, fixed_share, dwm, ew, oracle_excl, brier, ft as ft_ref  # noqa: F401

def _lag(x, h):
    z = np.zeros_like(x); z[h:] = x[:len(x) - h]; return z

def lw_scores(P, y, h, decay, prior_mass=20):
    T, K = P.shape; L = (P - y[:, None]) ** 2
    x = _lag(L, h); n = _lag(np.ones((T, 1)), h)
    pw = decay ** np.arange(T)[:, None]
    ls = lfilter([1.], [1., -decay], x, axis=0) + prior_mass * .25 * pw
    ms = lfilter([1.], [1., -decay], n, axis=0) + prior_mass * pw
    return 1 - ls / ms

def softmax_rows(S, mask=None):
    S = S.copy()
    if mask is not None: S[~mask] = -np.inf
    S -= S.max(1, keepdims=True); E = np.exp(S); return E / E.sum(1, keepdims=True)

def lw_fast(P, y, h, decay=.9, temperature=.2, prior_mass=20):
    W = softmax_rows(lw_scores(P, y, h, decay, prior_mass) / temperature); return dict(p=(W * P).sum(1), W=W)

def lw_cusum(P, y, h, decay=.9, temperature=.2, k=.05, H=5., prior_mass=20):
    T, K = P.shape; S = lw_scores(P, y, h, decay, prior_mass) / temperature
    on = np.ones((T, K), bool); c = [0.] * K; alarm = [False] * K
    D = (P - y[:, None]) ** 2 - .25
    for t in range(T):
        s = t - h
        if s >= 0:
            for j in range(K):
                c[j] = max(0., c[j] + D[s, j] - k)
                if c[j] > H: alarm[j] = True
                elif c[j] == 0.: alarm[j] = False
        if any(alarm) and not all(alarm): on[t] = [not a for a in alarm]
    W = softmax_rows(S, on); return dict(p=(W * P).sum(1), W=W, on=on)

def _logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))

def stack(P, y, h, lr=1e-3, l2=0.):
    T, K = P.shape; X = _logit(P).tolist(); w = [1. / K] * K; b = 0.; out = np.empty(T); yl = y.tolist()
    for t in range(T):
        s = t - h
        if s >= 0:
            xs = X[s]; z = b + sum(w[j] * xs[j] for j in range(K)); q = 1 / (1 + math.exp(-max(-30., min(30., z))))
            g = q - yl[s]; b -= lr * g
            for j in range(K): w[j] -= lr * (g * xs[j] + l2 * w[j])
        xt = X[t]; z = b + sum(w[j] * xt[j] for j in range(K)); out[t] = 1 / (1 + math.exp(-max(-30., min(30., z))))
    return dict(p=out)

def recal(P, y, h, lr=1e-3):
    """Per-agent online Platt scaling q = sigmoid(a*logit(p) + c), a=1, c=0 at start; returns the T x K matrix."""
    T, K = P.shape; X = _logit(P); Q = np.empty((T, K)); a = np.ones(K); c = np.zeros(K)
    for t in range(T):
        s = t - h
        if s >= 0:
            q = 1 / (1 + np.exp(-(a * X[s] + c))); g = q - y[s]; a -= lr * g * X[s]; c -= lr * g
        Q[t] = 1 / (1 + np.exp(-(a * X[t] + c)))
    return Q

def ft2(P, y, h, ref='half', **kw):
    """FinTrust with a choice of window reference. ref='half' calls the verified engine.ft unchanged."""
    if ref == 'half': return ft_ref(P, y, h, **kw)
    # 'clim': excess loss measured against pi(1-pi), pi = running mean of matured outcomes (prior 0.5, weight 1)
    T = len(y); cs = np.cumsum(np.r_[0., y.astype(float)]); n = np.arange(T + 1)
    pi_at = (cs + .5) / (n + 1)                        # base rate using outcomes 0..s-1 ... index s+? handled below
    # the outcome of step s arrives at step s+h; when it arrives the reference uses outcomes 0..s (all matured)
    refloss = np.array([pi_at[s + 1] * (1 - pi_at[s + 1]) for s in range(T)])
    shift = .25 - refloss                              # loss - refloss = (loss + shift) - 0.25
    return ft_ref(P, y, h, loss_shift=shift, **kw)

def recal_fast(P, y, h, lr=1e-3):
    """Same as recal(), written with Python floats for speed."""
    T, K = P.shape; X = _logit(P).tolist(); yl = y.tolist(); a = [1.] * K; c = [0.] * K; Q = [[0.] * K for _ in range(T)]
    sg = lambda z: 1 / (1 + math.exp(-max(-30., min(30., z))))
    for t in range(T):
        s = t - h
        if s >= 0:
            xs = X[s]
            for j in range(K):
                g = sg(a[j] * xs[j] + c[j]) - yl[s]; a[j] -= lr * g * xs[j]; c[j] -= lr * g
        xt = X[t]; Q[t] = [sg(a[j] * xt[j] + c[j]) for j in range(K)]
    return np.array(Q)

def fs_fast(P, y, h, eta=2., alpha=.01):
    """Same as engine.fixed_share, written with Python floats for speed."""
    T, K = P.shape; Pl = P.tolist(); yl = y.tolist(); w = [1. / K] * K; out = np.empty(T)
    for t in range(T):
        s = t - h
        if s >= 0:
            ps = Pl[s]; u = [w[j] * math.exp(-eta * (ps[j] - yl[s]) ** 2) for j in range(K)]; tot = sum(u)
            w = [(1 - alpha) * v / tot + alpha / K for v in u]
        out[t] = sum(w[j] * Pl[t][j] for j in range(K))
    return dict(p=out)
