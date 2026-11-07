"""Aggregate results_raw.json into results_summary.json (paired bootstrap over seeds)."""
import json, random, statistics as st
from pathlib import Path
D = Path(__file__).resolve().parent
rows = json.loads((D / 'results_raw.json').read_text())
def ci(diffs, B=10000, seed=1):
    r = random.Random(seed); n = len(diffs)
    m = sorted(sum(r.choice(diffs) for _ in range(n)) / n for _ in range(B))
    return st.mean(diffs), m[int(.025 * B)], m[int(.975 * B)]
summary = {}
for scn in dict.fromkeys(r['scenario'] for r in rows):
    R = [r for r in rows if r['scenario'] == scn]
    s = {k: st.mean(r[k] for r in R) for k in ['brier_eq','brier_hedge','brier_ft','brier_oracle','post_eq','post_hedge','post_ft','post_oracle','coverage','wD_hedge_post','wD_ft_post','false_quarantine','policy_us','audit_records']}
    s['sd_post_ft'] = st.stdev(r['post_ft'] for r in R)
    s['sd_post_hedge'] = st.stdev(r['post_hedge'] for r in R)
    s['diff_post_ft_minus_hedge'] = ci([r['post_ft'] - r['post_hedge'] for r in R])
    s['diff_post_ft_minus_eq'] = ci([r['post_ft'] - r['post_eq'] for r in R])
    s['diff_all_ft_minus_hedge'] = ci([r['brier_ft'] - r['brier_hedge'] for r in R])
    for k in ['detect_delay','probation_delay','reactivate_delay','first_active']:
        v = [r[k] for r in R if r[k] is not None]
        s[k] = dict(n=len(v), median=st.median(v) if v else None, min=min(v) if v else None, max=max(v) if v else None)
    s['audit_all_ok'] = all(r['audit_ok'] for r in R); s['seeds'] = len(R)
    summary[scn] = s
(D / 'results_summary.json').write_text(json.dumps(summary, indent=1))
for k, s in summary.items():
    print(f"\n== {k}")
    for kk, v in s.items(): print(f"  {kk}: {v if not isinstance(v,float) else round(v,4)}")
