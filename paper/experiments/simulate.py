"""Synthetic fault-injection study of the FinTrust governance core.

Uses prototype/fintrust.py UNCHANGED, with its default Config. All data are
synthetic Bernoulli streams; nothing here is market data or a trading result.
Run from the Resources folder:  python paper/experiments/simulate.py
"""
import sys, json, math, time, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'prototype'))
from fintrust import Governor, Config

T, FAULT, RECOVER, EVAL_START, SEEDS = 4000, 2000, 3000, 500, 30
AGENTS = ['A', 'B', 'C', 'D']           # D is the agent that may become faulty
NOISE = {'A': .6, 'B': .8, 'C': 1.0, 'D': .7}
SCENARIOS = ['clean', 'inversion', 'stuck', 'overconfident', 'transient', 'inversion_delayed']
sig = lambda x: 1 / (1 + math.exp(-x))

def stream(seed):
    """Latent AR(1) signal; outcome y ~ Bernoulli(sigmoid(1.5 z))."""
    r = random.Random(seed); z = 0.; out = []
    for t in range(T):
        z = .95 * z + math.sqrt(1 - .95**2) * r.gauss(0, 1)
        y = 1 if r.random() < sig(1.5 * z) else 0
        obs = {a: sig(1.5 * (z + r.gauss(0, NOISE[a])) / math.sqrt(1 + NOISE[a]**2 * 0.6)) for a in AGENTS}
        out.append((y, obs))
    return out

def corrupt(scn, t, p):
    """Return agent D's emitted forecast under the scenario."""
    if scn == 'clean' or t < FAULT: return p
    if scn == 'transient' and t >= RECOVER: return p
    if scn in ('inversion', 'transient', 'inversion_delayed'): return 1 - p
    if scn == 'stuck': return .95
    if scn == 'overconfident': return min(1., max(0., .5 + 4 * (p - .5)))
    raise ValueError(scn)

def brier(p, y): return (p - y) ** 2

def run(scn, seed, cfg=None):
    h = 5 if scn == 'inversion_delayed' else 1
    data = stream(seed)
    cfg = cfg or Config()
    g = Governor(cfg)
    for a in AGENTS: g.register(a)
    # baseline state: decayed loss with the same prior/decay/temperature, no states
    hedge = {a: [cfg.prior_mass * .25, cfg.prior_mass] for a in AGENTS}
    pend_hedge = {}
    res = dict(eq=[], hedge=[], ft=[], ft_cov=[], w_hedge_D=[], w_ft_D=[], oracle=[])
    events = []; quarantined_at = None; probation_at = None; reactivated_at = None
    cycle_time = 0.
    for t in range(T):
        g.advance(t)
        for a in hedge: hedge[a][0] *= cfg.decay; hedge[a][1] *= cfg.decay
        # deliver matured outcomes for decisions made at t-h
        s = t - h
        if s >= 0:
            ys = data[s][0]
            for a in AGENTS:
                g.outcome(f'{s}:{a}', ys, t)
                hedge[a][0] += brier(pend_hedge.pop((s, a)), ys); hedge[a][1] += 1
        if scn == 'transient' and t == RECOVER and g.agents['D'].state == 'QUARANTINED':
            g.clear('D')   # operator confirms root cause fixed
        y, obs = data[t]
        f = {a: (corrupt(scn, t, obs[a]) if a == 'D' else obs[a]) for a in AGENTS}
        # Equal weight
        p_eq = sum(f.values()) / len(f)
        # Loss-weighted softmax (FinTrust score without state machine)
        sc = {a: (1 - hedge[a][0] / hedge[a][1]) / cfg.temperature for a in AGENTS}
        mx = max(sc.values()); ex = {a: math.exp(v - mx) for a, v in sc.items()}; Z = sum(ex.values())
        w_h = {a: ex[a] / Z for a in AGENTS}
        p_h = sum(w_h[a] * f[a] for a in AGENTS)
        # FinTrust governor (actual prototype)
        t0 = time.perf_counter()
        pol = g.policy(f)
        cycle_time += time.perf_counter() - t0
        covered = pol['probability'] is not None
        p_ft = pol['probability'] if covered else p_eq   # frozen equal-weight fallback
        # oracle: equal weight over agents that are actually healthy
        healthy = [a for a in AGENTS if not (a == 'D' and corrupt(scn, t, obs[a]) != obs[a])]
        p_or = sum(f[a] for a in healthy) / len(healthy)
        for a in AGENTS:
            g.forecast(f'{t}:{a}', a, f[a], t, h, t + 1)
            pend_hedge[(t, a)] = f[a]
        if t >= EVAL_START:
            res['eq'].append(brier(p_eq, y)); res['hedge'].append(brier(p_h, y))
            res['ft'].append(brier(p_ft, y)); res['ft_cov'].append(covered)
            res['oracle'].append(brier(p_or, y))
            res['w_hedge_D'].append(w_h['D']); res['w_ft_D'].append(pol['weights'].get('D', 0.))
        st = g.agents['D'].state
        if quarantined_at is None and st == 'QUARANTINED' and t >= FAULT: quarantined_at = t
        if quarantined_at is not None and probation_at is None and st == 'PROBATION': probation_at = t
        if probation_at is not None and reactivated_at is None and st == 'ACTIVE': reactivated_at = t
    # false quarantines: transitions into QUARANTINED for healthy agents, or for D before fault/in clean
    false_q = 0
    for r in g.audit.records:
        pl = r['payload']
        if pl.get('kind') == 'transition' and pl['new'] == 'QUARANTINED':
            if pl['agent'] != 'D' or scn == 'clean' or pl['time'] < FAULT: false_q += 1
    first_active = min((r['payload']['time'] for r in g.audit.records
                        if r['payload'].get('kind') == 'transition' and r['payload']['new'] == 'ACTIVE'), default=None)
    def mean(x): return sum(x) / len(x)
    post = slice(FAULT - EVAL_START, (RECOVER if scn == 'transient' else T) - EVAL_START)
    return dict(
        scenario=scn, seed=seed,
        brier_eq=mean(res['eq']), brier_hedge=mean(res['hedge']), brier_ft=mean(res['ft']), brier_oracle=mean(res['oracle']),
        post_eq=mean(res['eq'][post]), post_hedge=mean(res['hedge'][post]), post_ft=mean(res['ft'][post]), post_oracle=mean(res['oracle'][post]),
        coverage=mean(res['ft_cov']),
        wD_hedge_post=mean(res['w_hedge_D'][post]), wD_ft_post=mean(res['w_ft_D'][post]),
        detect_delay=(quarantined_at - FAULT) if quarantined_at is not None and scn != 'clean' else None,
        probation_delay=(probation_at - RECOVER) if probation_at is not None else None,
        reactivate_delay=(reactivated_at - RECOVER) if reactivated_at is not None else None,
        false_quarantine=false_q, first_active=first_active,
        audit_records=len(g.audit.records), audit_ok=g.audit.verify(g.audit.checkpoint()),
        policy_us=1e6 * cycle_time / T,
        trace_wD_ft=res['w_ft_D'][::10] if seed == 0 else None,
        trace_wD_hedge=res['w_hedge_D'][::10] if seed == 0 else None,
    )

if __name__ == '__main__':
    out = [run(s, k) for s in SCENARIOS for k in range(SEEDS)]
    p = ROOT / 'paper' / 'experiments' / 'results_raw.json'
    p.write_text(json.dumps(out, indent=1)); print('wrote', p, len(out))
