"""Fast engine for the fair-baseline study.

ft(): a line-by-line re-implementation of prototype/fintrust.py Governor semantics (scoring, windows,
lifecycle, capped softmax) without the audit log, for speed. verify.py checks it reproduces the
prototype exactly. Baselines: EW, LW (decayed Brier softmax), Fixed Share, Dynamic Weighted Majority,
and an exclusion oracle. All combiners see the same (possibly corrupted) forecast matrix P (T x K)
and receive outcome y[s] at step s+h.
"""
import math, numpy as np

def corrupt_matrix(P, fault):
    """fault = (k, kind, start, end) ; returns corrupted copy and boolean mask of faulty cells."""
    Q = P.copy(); M = np.zeros_like(P, dtype=bool)
    if fault is None: return Q, M
    k, kind, a, b = fault; b = len(P) if b is None else b
    x = P[a:b, k]
    if kind == 'inversion': x = 1 - x
    elif kind == 'stuck': x = np.full_like(x, .95)
    elif kind == 'overconfident': x = np.clip(.5 + 4 * (x - .5), 0, 1)
    Q[a:b, k] = x; M[a:b, k] = True
    return Q, M

def ft(P, y, h, decay=.9, temperature=.2, prior_mass=20, window=50, minimum=200, consecutive=3,
       bad_thr=.05, good_thr=0., auto_clear=True, clear_at=None, clear_agent=None, probation_cap=.1, loss_shift=None, start='SHADOW'):
    T, K = P.shape
    st = list(start) if isinstance(start, (list, tuple)) else [start] * K; ls = [prior_mass * .25] * K; ms = [float(prior_mass)] * K
    cnt = [0] * K; since = [0] * K; good = [0] * K; bad = [0] * K; win = [[] for _ in range(K)]; clr = [True] * K
    out = np.empty(T); cov = np.zeros(T, bool); W = np.zeros((T, K)); trans = []
    for t in range(T):
        if t > 0:
            for k in range(K): ms[k] *= decay; ls[k] *= decay
        s = t - h
        if s >= 0:
            ys = y[s]
            for k in range(K):
                loss = (P[s, k] - ys) ** 2
                ms[k] += 1; ls[k] += loss; cnt[k] += 1; since[k] += 1; win[k].append(loss - .25 + (loss_shift[s] if loss_shift is not None else 0.))
                if len(win[k]) == window:
                    d = sum(win[k]) / window; win[k] = []
                    good[k] = good[k] + 1 if d < good_thr else 0
                    bad[k] = bad[k] + 1 if d > bad_thr else 0
                    old = st[k]
                    if st[k] == 'ACTIVE' and bad[k] >= consecutive: st[k] = 'QUARANTINED'; clr[k] = auto_clear
                    elif st[k] == 'PROBATION' and bad[k]: st[k] = 'QUARANTINED'; clr[k] = auto_clear
                    elif st[k] == 'SHADOW' and cnt[k] >= minimum and good[k] >= consecutive: st[k] = 'ACTIVE'
                    elif st[k] == 'QUARANTINED' and clr[k] and since[k] >= minimum and good[k] >= consecutive: st[k] = 'PROBATION'
                    elif st[k] == 'PROBATION' and good[k] >= consecutive: st[k] = 'ACTIVE'
                    if old != st[k]:
                        since[k] = good[k] = bad[k] = 0; trans.append((t, k, old, st[k]))
        if clear_at is not None and t == clear_at and st[clear_agent] == 'QUARANTINED': clr[clear_agent] = True
        el = [k for k in range(K) if st[k] in ('ACTIVE', 'PROBATION')]
        caps = {k: (probation_cap if st[k] == 'PROBATION' else 1.) for k in el}
        if not el or sum(caps.values()) < 1 - 1e-12:
            out[t] = np.nan; continue
        sc = {k: (1 - (ls[k] / ms[k] if ms[k] > 1e-12 else .25)) / temperature for k in el}
        mx = max(sc.values()); raw = {k: math.exp(v - mx) for k, v in sc.items()}
        rem = set(raw); budget = 1.; w = {}
        while rem:
            tot = sum(raw[k] for k in rem); trial = {k: budget * raw[k] / tot for k in rem}
            cl = {k for k in rem if trial[k] > caps[k] + 1e-12}
            if not cl: w.update(trial); break
            for k in cl: w[k] = caps[k]; budget -= caps[k]
            rem -= cl
        for k, v in w.items(): W[t, k] = v
        out[t] = sum(w[k] * P[t, k] for k in w); cov[t] = True
    return dict(p=out, cov=cov, W=W, trans=trans)

def lw(P, y, h, decay=.9, temperature=.2, prior_mass=20):
    T, K = P.shape; ls = np.full(K, prior_mass * .25); ms = np.full(K, float(prior_mass)); out = np.empty(T); W = np.empty((T, K))
    for t in range(T):
        if t > 0: ls *= decay; ms *= decay
        s = t - h
        if s >= 0: ls += (P[s] - y[s]) ** 2; ms += 1
        sc = (1 - ls / ms) / temperature; e = np.exp(sc - sc.max()); w = e / e.sum()
        W[t] = w; out[t] = w @ P[t]
    return dict(p=out, W=W)

def fixed_share(P, y, h, eta=2., alpha=.01):
    """Herbster & Warmuth (1998) Fixed Share with Brier loss, updated when each delayed outcome arrives."""
    T, K = P.shape; lw_ = np.zeros(K); out = np.empty(T); W = np.empty((T, K))
    for t in range(T):
        s = t - h
        if s >= 0:
            lw_ = lw_ - eta * (P[s] - y[s]) ** 2
            w = np.exp(lw_ - lw_.max()); w /= w.sum(); w = (1 - alpha) * w + alpha / K; lw_ = np.log(w)
        w = np.exp(lw_ - lw_.max()); w /= w.sum(); W[t] = w; out[t] = w @ P[t]
    return dict(p=out, W=W)

def dwm(P, y, h, beta=.9, theta=.05, period=10):
    """Dynamic Weighted Majority (Kolter & Maloof, 2007) adapted to fixed probabilistic experts:
    every `period` outcomes, experts whose thresholded prediction was wrong have weight x beta;
    weights are normalised; experts below theta are removed; when the ensemble errs and an expert
    is removed, the removed expert with the lowest recent Brier loss is re-admitted at weight 1/K."""
    T, K = P.shape; w = np.full(K, 1. / K); act = np.ones(K, bool); rec = np.full(K, .25); out = np.empty(T); W = np.empty((T, K)); n = 0
    for t in range(T):
        s = t - h
        if s >= 0:
            n += 1; ys = y[s]; rec = .99 * rec + .01 * (P[s] - ys) ** 2
            if n % period == 0:
                wrong = ((P[s] > .5).astype(int) != ys) & act
                w[wrong] *= beta; w[act] /= w[act].sum()
                drop = act & (w < theta)
                if drop.sum() < act.sum(): act[drop] = False; w[drop] = 0
                ens = (w[act] @ P[s, act]) / w[act].sum()
                if (int(ens > .5) != ys) and (~act).any():
                    j = np.where(~act)[0][np.argmin(rec[~act])]; act[j] = True; w[j] = 1. / K; w[act] /= w[act].sum()
        ww = np.where(act, w, 0.); ww = ww / ww.sum(); W[t] = ww; out[t] = ww @ P[t]
    return dict(p=out, W=W)

def ew(P): return P.mean(1)

def oracle_excl(P, M):
    """Equal weight over cells not marked faulty (hindsight; not implementable)."""
    good = ~M; return (P * good).sum(1) / good.sum(1)

def brier(p, y): return float(np.mean((p - y) ** 2))
