"""Replay saved out-of-sample agent forecasts through the unmodified FinTrust prototype.

F: list of dicts {agent: p} per decision step, y: list of 0/1 outcomes, h: label delay in steps.
Outcome for step s is delivered at step s+h. Baselines use identical forecasts.
"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'prototype'))
from fintrust import Governor, Config

def corrupt(kind, p):
    if kind == 'inversion': return 1 - p
    if kind == 'stuck': return .95
    if kind == 'overconfident': return min(1., max(0., .5 + 4 * (p - .5)))
    return p

def replay(F, y, h, agents, fault=None, clear_at=None, cfg=None, auto_clear_statistical=False):
    """fault = (agent, kind, start, end) with end exclusive or None. Returns per-step arrays."""
    cfg = cfg or Config()
    g = Governor(cfg)
    for a in agents: g.register(a)
    lw = {a: [cfg.prior_mass * .25, cfg.prior_mass] for a in agents}
    pend = {}
    out = dict(p_eq=[], p_lw=[], p_ft=[], cov=[], w_lw=[], w_ft=[], states=[])
    events = []
    T = len(y)
    for t in range(T):
        g.advance(t)
        for a in agents: lw[a][0] *= cfg.decay; lw[a][1] *= cfg.decay
        s = t - h
        if s >= 0:
            for a in agents:
                before = g.agents[a].state
                g.outcome(f'{s}:{a}', y[s], t)
                # variant: a purely statistical quarantine needs no operator clearance (integrity faults still would)
                if auto_clear_statistical and before != 'QUARANTINED' and g.agents[a].state == 'QUARANTINED':
                    g.clear(a)
                lw[a][0] += (pend.pop((s, a)) - y[s]) ** 2; lw[a][1] += 1
        if clear_at is not None and t == clear_at and fault and g.agents[fault[0]].state == 'QUARANTINED':
            g.clear(fault[0])
        f = dict(F[t])
        if fault and t >= fault[2] and (fault[3] is None or t < fault[3]):
            f[fault[0]] = corrupt(fault[1], f[fault[0]])
        p_eq = sum(f.values()) / len(f)
        sc = {a: (1 - lw[a][0] / lw[a][1]) / cfg.temperature for a in agents}
        mx = max(sc.values()); ex = {a: math.exp(v - mx) for a, v in sc.items()}; Z = sum(ex.values())
        w = {a: ex[a] / Z for a in agents}
        pol = g.policy(f)
        cov = pol['probability'] is not None
        out['p_eq'].append(p_eq); out['p_lw'].append(sum(w[a] * f[a] for a in agents))
        out['p_ft'].append(pol['probability'] if cov else p_eq); out['cov'].append(cov)
        out['w_lw'].append(w); out['w_ft'].append(dict(pol['weights']))
        out['states'].append({a: g.agents[a].state for a in agents})
        for a in agents:
            g.forecast(f'{t}:{a}', a, f[a], t, h, t + 1)
            pend[(t, a)] = f[a]
    out['transitions'] = [r['payload'] for r in g.audit.records if r['payload'].get('kind') == 'transition']
    out['audit_ok'] = g.audit.verify(g.audit.checkpoint()); out['audit_n'] = len(g.audit.records)
    return out
