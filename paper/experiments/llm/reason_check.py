"""Experiment 2 diagnostic: do the LLM's stated reasons match the signals it was given?
A claim counts as contradicted when the prompt's own numbers say the opposite:
'high volatility' while today's and the 5-day |return| ratios are both below 1; 'wide range(s)' while both range ratios are
below 1; 'rising volatility index' while the 5-day index change is negative. Adds 'reason_check' to llm_results_feat.json.
usage: python reason_check.py <requests dir> [tag]"""
import json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; req_dir = Path(sys.argv[1]); tag = sys.argv[2] if len(sys.argv) > 2 else 'llama31_8b'
def sig(u, k): return float(re.search(re.escape(k) + r': (-?[\d.]+)', u).group(1))
req = {}
for l in open(req_dir / 'feat_clean.jsonl'):
    r = json.loads(l); u = r['user']
    req[r['id']] = dict(a1=sig(u, "today's |return| / 20-day mean |return|"), a5=sig(u, "5-day mean |return| / 20-day mean |return|"),
                        rg1=sig(u, "today's high-low range / 20-day mean range"), rg5=sig(u, "5-day mean range / 20-day mean range"),
                        dv=sig(u, "volatility index change over 5 days (%)"))
rules = {'high_volatility': (('high recent volatility', 'high volatility', 'elevated'), lambda s: s['a1'] < 1 and s['a5'] < 1),
         'wide_range': (('wide range',), lambda s: s['rg1'] < 1 and s['rg5'] < 1),
         'rising_index': (('rising volatility index', 'rising vol'), lambda s: s['dv'] < 0)}
out = [json.loads(l) for l in open(HERE / 'outputs' / f'{tag}__feat_clean.jsonl')]
res = {'answers': len(out)}
for name, (words, contra) in rules.items():
    claims = [o for o in out if any(w in o['raw'].lower().split('probability')[0] for w in words)]
    bad = sum(contra(req[o['id']]) for o in claims)
    res[name] = dict(claims=len(claims), contradicted=bad, share_contradicted=bad / max(1, len(claims)))
print(json.dumps(res, indent=1))
p = HERE / 'llm_results_feat.json'; d = json.loads(p.read_text()); d['reason_check'] = res; p.write_text(json.dumps(d, indent=1))
