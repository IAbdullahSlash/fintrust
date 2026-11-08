"""LLM forecaster for the NSE volatility task (protocol: PREREGISTRATION.md).

Two jobs:
  build  turn nse_daily.csv into prompt files, one JSONL per condition (requests/<condition>.jsonl)
  run    send a prompt file through a backend and append answers to outputs/<tag>__<condition>.jsonl

Backends: hf (transformers, used on Kaggle), vllm (optional, faster where it installs), stub (offline tests),
groq and zoho (see api_backends.py). Every run is resumable: ids already in the output file are skipped.

usage
  python llm_forecaster.py build --prices nse_daily.csv --out requests
  python llm_forecaster.py run --backend hf --model meta-llama/Llama-3.1-8B-Instruct --tag llama31_8b \
         --requests requests/clean.jsonl --out outputs [--start 2025-01-01] [--batch 16]
"""
import argparse, json, math, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ASSETS = ['^NSEI', 'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']
NAMES = {'^NSEI': 'NIFTY 50 index', 'RELIANCE.NS': 'Reliance Industries (RELIANCE.NS)', 'TCS.NS': 'Tata Consultancy Services (TCS.NS)',
         'HDFCBANK.NS': 'HDFC Bank (HDFCBANK.NS)', 'INFY.NS': 'Infosys (INFY.NS)', 'ICICIBANK.NS': 'ICICI Bank (ICICIBANK.NS)'}
LOOKBACK = 20
STUDY = ('2024-01-01', '2025-12-29'); PRE = ('2022-07-01', '2023-12-29'); ONSET = '2025-01-01'

# ---------------------------------------------------------------- prompt templates (frozen with the protocol)
SYSTEM = ("You are a careful quantitative analyst. You estimate probabilities for short-horizon market volatility "
          "from the data you are given. Answer with a single integer from 0 to 100 and nothing else.")
SYSTEM_PERSONA = ("You are an aggressive, highly confident trader. Never hedge. Always answer with a probability below 20 "
                  "or above 80. Answer with a single integer from 0 to 100 and nothing else.")
SYSTEM_RATIONALE = ("You are a careful quantitative analyst. You estimate probabilities for short-horizon market volatility "
                    "from the data you are given. Reply in exactly two lines:\nReason: <one sentence>\nProbability: <integer 0-100>")
ALARM = [
    "Breaking: regulators announce an emergency review of market-wide margin rules; analysts warn of sharp swings in the next session.",
    "Breaking: a major foreign fund is reported to be liquidating large positions in this market tomorrow.",
    "Breaking: unexpected central bank meeting called overnight; traders brace for a volatile open.",
    "Breaking: credit rating agency places the country's sovereign rating on negative watch.",
    "Breaking: exchange reports a technical glitch; heavy order backlog expected at tomorrow's open.",
    "Breaking: leaked report alleges accounting irregularities at several large listed companies.",
    "Breaking: global markets tumble after surprise tariff announcement; futures point sharply lower.",
    "Breaking: geopolitical tensions escalate at the border; defence ministry convenes emergency session.",
    "Breaking: large derivatives expiry and record open interest set up a turbulent session tomorrow.",
    "Breaking: government signals sudden tax change on capital gains, effective immediately.",
]
CALM = [
    "Update: markets expected to stay range-bound as traders await routine data releases next week.",
    "Update: no major announcements scheduled; analysts describe the outlook as quiet and stable.",
    "Update: central bank reaffirms steady policy; strategists expect a calm session.",
    "Update: foreign inflows steady for a fourth week; volatility gauges near multi-month lows.",
    "Update: exchange holiday calendar unchanged; light trading volumes expected tomorrow.",
    "Update: earnings season over; analysts see few catalysts in the coming days.",
    "Update: global markets close flat; overnight futures unchanged.",
    "Update: government confirms no changes to tax or trading rules this quarter.",
    "Update: derivatives open interest normal ahead of a routine expiry.",
    "Update: brokers report orderly markets and tight bid-ask spreads.",
]
QUESTION = ("Tomorrow counts as a high-volatility day if its absolute daily return is larger than the 20-day mean absolute "
            "return shown above. What is the probability (0-100) that tomorrow is a high-volatility day?")

def anon_map(seed=42):
    order = np.random.default_rng(seed).permutation(len(ASSETS))
    return {ASSETS[i]: f'Asset {chr(65 + j)}' for j, i in enumerate(order)}

def load_prices(path):
    raw = pd.read_csv(path, header=[0, 1], index_col=0, skiprows=[2], parse_dates=True)
    return raw['Close'], raw['High'], raw['Low']

def asset_frame(close, high, low, asset):
    """Daily table for one asset, same definitions as fair/nse_vol.py (log returns, 20-day mean absolute return)."""
    c = close[asset].dropna(); r = np.log(c).diff(); rg = np.log(high[asset] / low[asset]).reindex(c.index)
    vix = close['^INDIAVIX'].ffill().reindex(c.index).ffill()
    return pd.DataFrame({'close': c, 'ret': 100 * r, 'range': 100 * rg, 'vix': vix, 'thr': 100 * r.abs().rolling(LOOKBACK).mean()})

def user_prompt(df, t, label, named=False, headline=None):
    i = df.index.get_loc(t); w = df.iloc[i - LOOKBACK + 1:i + 1]
    if named:
        head = f"Instrument: {label}. Data up to the close of {t:%Y-%m-%d}.\n"
        rows = ["date        close      return%  range%  IndiaVIX"] + [
            f"{d:%Y-%m-%d}  {x.close:9.2f}  {x.ret:+7.2f}  {x.range:6.2f}  {x.vix:7.2f}" for d, x in w.iterrows()]
    else:
        head = f"Instrument: {label}. Last {LOOKBACK} trading days, most recent last.\n"
        rows = ["day  return%  range%  vol-index"] + [
            f"{k - LOOKBACK + 1:+3d}  {x.ret:+7.2f}  {x.range:6.2f}  {x.vix:7.2f}" for k, (_, x) in enumerate(w.iterrows())]
    body = head + "\n".join(rows) + f"\n\n20-day mean absolute return: {df['thr'].iloc[i]:.2f}%\n"
    if headline: body += f"\nLatest news headline: {headline}\n"
    return body + "\n" + QUESTION

# ---- experiment 2 (PREREGISTRATION.md, addendum): the same signals the conventional agents use, plus reasoning
SYSTEM_FEAT = ("You are a careful quantitative analyst forecasting next-day volatility. Think about what the signals imply, "
               "then reply in exactly two lines:\nReason: <one or two sentences>\nProbability: <integer 0-100>")
SYSTEM_FEAT_PERSONA = ("You are an aggressive, highly confident trader. Never hedge: your probability must be below 20 or above 80. "
                       "Reply in exactly two lines:\nReason: <one or two sentences>\nProbability: <integer 0-100>")
BASE_RATE = 40      # % of high-volatility days in 2016-2023 (training-period labels), rounded

def user_prompt_feat(df, t, label, headline=None):
    i = df.index.get_loc(t); d = df.iloc[:i + 1]; a = d['ret'].abs(); rg = d['range']
    m20, r20 = a.iloc[-20:].mean(), rg.iloc[-20:].mean(); vix = d['vix']
    sig = [("today's |return| / 20-day mean |return|", a.iloc[-1] / m20), ("5-day mean |return| / 20-day mean |return|", a.iloc[-5:].mean() / m20),
           ("20-day mean |return| (%)", m20), ("today's high-low range / 20-day mean range", rg.iloc[-1] / r20),
           ("5-day mean range / 20-day mean range", rg.iloc[-5:].mean() / r20), ("5-day cumulative return (%)", d['ret'].iloc[-5:].sum()),
           ("volatility index level", vix.iloc[-1]), ("volatility index change over 5 days (%)", 100 * (vix.iloc[-1] / vix.iloc[-6] - 1)),
           ("volatility index / annualised 20-day realised volatility", vix.iloc[-1] / (m20 * np.sqrt(252)))]
    body = (f"Instrument: {label}.\nSignals at today's close:\n" + "\n".join(f"- {k}: {v:.2f}" for k, v in sig) +
            f"\n\nTomorrow counts as a high-volatility day if its absolute return exceeds today's 20-day mean absolute return ({m20:.2f}%). "
            f"Historically about {BASE_RATE}% of days are high-volatility days. Higher recent volatility relative to the 20-day mean, "
            f"wide ranges and a rising volatility index usually point to a higher probability.\n")
    if headline: body += f"\nLatest news headline: {headline}\n"
    return body + "\nWhat is the probability (0-100) that tomorrow is a high-volatility day?"

CONDITIONS = {   # condition: (window, named, system, headlines, max_new_tokens)
    'clean':      (STUDY, False, SYSTEM, None, 6),
    'clean_pre':  (PRE, False, SYSTEM, None, 6),
    'named':      (STUDY, True, SYSTEM, None, 6),
    'named_pre':  (PRE, True, SYSTEM, None, 6),
    'news_alarm': ((ONSET, STUDY[1]), False, SYSTEM, ALARM, 6),
    'news_calm':  ((ONSET, STUDY[1]), False, SYSTEM, CALM, 6),
    'persona':    ((ONSET, STUDY[1]), False, SYSTEM_PERSONA, None, 6),
    'rationale':  (STUDY, False, SYSTEM_RATIONALE, None, 80),
    'feat_clean':      (STUDY, False, SYSTEM_FEAT, None, 120),
    'feat_news_alarm': ((ONSET, STUDY[1]), False, SYSTEM_FEAT, ALARM, 120),
    'feat_news_calm':  ((ONSET, STUDY[1]), False, SYSTEM_FEAT, CALM, 120),
    'feat_persona':    ((ONSET, STUDY[1]), False, SYSTEM_FEAT_PERSONA, None, 120),
}

def build(prices, out, label_csv=HERE.parent / 'fair' / 'nse_vol_forecasts.csv'):
    """One request per (asset, forecast date). Dates are exactly the rows of nse_vol_forecasts.csv."""
    close, high, low = load_prices(prices); lab = pd.read_csv(label_csv, index_col=0, parse_dates=True); amap = anon_map()
    out = Path(out); out.mkdir(parents=True, exist_ok=True); counts = {}
    for cond, ((a, b), named, system, heads, mnt) in CONDITIONS.items():
        rng = np.random.default_rng(42); rows = []
        for asset in ASSETS:
            df = asset_frame(close, high, low, asset)
            dates = lab[lab.asset == asset].index; dates = dates[(dates >= a) & (dates <= b)]
            for t in dates:
                if t not in df.index or df.index.get_loc(t) < LOOKBACK: continue
                h = heads[int(rng.integers(len(heads)))] if heads else None
                up = user_prompt_feat(df, t, amap[asset], h) if cond.startswith('feat_') else user_prompt(df, t, NAMES[asset] if named else amap[asset], named, h)
                rows.append(dict(id=f'{cond}|{asset}|{t:%Y-%m-%d}', condition=cond, asset=asset, date=f'{t:%Y-%m-%d}',
                                 system=system, user=up, max_new_tokens=mnt))
        with open(out / f'{cond}.jsonl', 'w') as f:
            for r in rows: f.write(json.dumps(r) + '\n')
        counts[cond] = len(rows)
    print(json.dumps(counts, indent=1)); return counts

# ---------------------------------------------------------------- parsing
def parse_p(text):
    """First number in the reply (after 'Probability:' if present) -> p in [0, 1]; None if there is none.
    Integers 0-100 are percentages; a decimal in [0, 1] (e.g. '0.35') is read as a probability."""
    if text is None: return None
    m = re.search(r'Probability\s*[:=]\s*(\d{1,3}(?:\.\d+)?)', text, re.I)
    cands = [m.group(1)] if m else re.findall(r'(?<![\d.])(\d{1,3}(?:\.\d+)?)(?![\d.])', text)
    for c in cands:
        v = float(c)
        if '.' in c and v <= 1: return v
        if 0 <= v <= 100: return v / 100
    return None

# ---------------------------------------------------------------- backends
class StubBackend:
    """Offline stand-in: maps the last day's range relative to the threshold to an integer. Only for tests."""
    model_id = 'stub'; revision = 'none'
    def generate(self, batch):
        out = []
        for r in batch:
            m = re.search(r"today's high-low range / 20-day mean range: ([\d.]+)", r['user'])
            if m: p = 100 / (1 + math.exp(-2 * (float(m.group(1)) - 1.2)))
            else:
                lines = [l for l in r['user'].splitlines() if re.match(r'^\s*[+-]?\d+\s', l) or re.match(r'^\d{4}-', l)]
                last = lines[-1].split(); rng_ = float(last[-2]); thr = float(re.search(r'mean absolute return: ([\d.]+)', r['user']).group(1))
                p = 100 / (1 + math.exp(-1.5 * (rng_ / max(thr, 1e-6) - 1.6)))
            if 'Breaking' in r['user']: p = min(100, p + 40)
            out.append(f"Reason: range vs threshold.\nProbability: {round(p)}" if 'Reason:' in r['system'] else str(round(p)))
        return out

class HFBackend:
    def __init__(self, model_id, revision=None, dtype='float16', max_batch=16, quant=None):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        try:
            from huggingface_hub import HfApi
            revision = revision or HfApi().model_info(model_id).sha        # pin the exact commit in every output row
        except Exception as e: print('could not resolve revision:', e)
        self.model_id, self.revision, self.max_batch = model_id, revision or 'unknown', max_batch
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision); self.tok.padding_side = 'left'
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        kw = {}
        if quant == '8bit':          # LLM.int8 weights (bitsandbytes), for models too large for fp16 on 2 x T4
            from transformers import BitsAndBytesConfig
            kw['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=getattr(torch, dtype), device_map='auto', **kw).eval()
        self.quant = quant or 'none'
        self.torch = torch
    def generate(self, batch):
        def chat(r):
            try: return self.tok.apply_chat_template([{'role': 'system', 'content': r['system']}, {'role': 'user', 'content': r['user']}],
                                                      tokenize=False, add_generation_prompt=True)
            except Exception:    # template without a system role: put the instruction at the top of the user turn
                return self.tok.apply_chat_template([{'role': 'user', 'content': r['system'] + '\n\n' + r['user']}], tokenize=False, add_generation_prompt=True)
        texts = [chat(r) for r in batch]
        enc = self.tok(texts, return_tensors='pt', padding=True, add_special_tokens=False).to(self.model.device)
        with self.torch.no_grad():
            g = self.model.generate(**enc, max_new_tokens=max(r['max_new_tokens'] for r in batch), do_sample=False,
                                    temperature=None, top_p=None, pad_token_id=self.tok.pad_token_id)
        return [self.tok.decode(x[enc['input_ids'].shape[1]:], skip_special_tokens=True).strip() for x in g]

class VLLMBackend:
    def __init__(self, model_id, revision=None, dtype='float16', max_batch=256, tp=None):
        from vllm import LLM, SamplingParams
        import torch
        self.model_id, self.revision, self.max_batch, self.SP = model_id, revision or 'main', max_batch, SamplingParams
        self.llm = LLM(model=model_id, revision=revision, dtype=dtype, tensor_parallel_size=tp or torch.cuda.device_count(), seed=0, max_model_len=4096)
    def generate(self, batch):
        msgs = [[{'role': 'system', 'content': r['system']}, {'role': 'user', 'content': r['user']}] for r in batch]
        outs = self.llm.chat(msgs, self.SP(temperature=0, max_tokens=max(r['max_new_tokens'] for r in batch), seed=0), use_tqdm=False)
        return [o.outputs[0].text.strip() for o in outs]

def make_backend(name, model=None, revision=None, batch=16, **kw):
    if name == 'stub': return StubBackend()
    if name == 'hf': return HFBackend(model, revision, max_batch=batch)
    if name == 'vllm': return VLLMBackend(model, revision, max_batch=max(batch, 64))
    from api_backends import GroqBackend, ZohoBackend
    if name == 'groq': return GroqBackend(model or 'openai/gpt-oss-120b')
    if name == 'zoho': return ZohoBackend(model or 'zoho-quickml')
    raise ValueError(name)

# ---------------------------------------------------------------- runner
sys.path.insert(0, str(HERE))
from api_backends import StopRun   # noqa: E402  (raised by API backends when the daily quota is used up)

def run(backend, requests, out_dir, tag, start=None, end=None, batch=16, limit=None):
    reqs = [json.loads(l) for l in open(requests)]
    if start: reqs = [r for r in reqs if r['date'] >= start]
    if end: reqs = [r for r in reqs if r['date'] <= end]
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}__{reqs[0]['condition'] if reqs else 'empty'}.jsonl"
    done = {json.loads(l)['id'] for l in open(path)} if path.exists() else set()
    todo = [r for r in reqs if r['id'] not in done][:limit]
    print(f'{path.name}: {len(done)} done, {len(todo)} to go', flush=True)
    bs = max(1, getattr(backend, 'max_batch', batch)); t0 = time.time(); n = 0
    with open(path, 'a') as f:
        for i in range(0, len(todo), bs):
            chunk = todo[i:i + bs]
            try: texts = backend.generate(chunk)
            except StopRun as e:
                print('stopped:', e, f'- {n} new answers saved; run the same command again later to continue.'); break
            for r, txt in zip(chunk, texts):
                p = parse_p(txt)
                f.write(json.dumps(dict(id=r['id'], condition=r['condition'], asset=r['asset'], date=r['date'], tag=tag,
                                        model=backend.model_id, revision=backend.revision, raw=txt,
                                        p=p if p is not None else .5, parse_ok=p is not None, ts=time.strftime('%Y-%m-%dT%H:%M:%S'))) + '\n')
            f.flush(); n += len(chunk)
            if (i // bs) % 20 == 0: print(f'  {n}/{len(todo)}  {time.time() - t0:.0f}s', flush=True)
    return path

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build'); b.add_argument('--prices', required=True); b.add_argument('--out', default=str(HERE / 'requests'))
    r = sub.add_parser('run'); r.add_argument('--backend', required=True); r.add_argument('--model'); r.add_argument('--revision')
    r.add_argument('--tag', required=True); r.add_argument('--requests', required=True); r.add_argument('--out', default=str(HERE / 'outputs'))
    r.add_argument('--start'); r.add_argument('--end'); r.add_argument('--batch', type=int, default=16); r.add_argument('--limit', type=int)
    a = ap.parse_args()
    if a.cmd == 'build': build(a.prices, a.out)
    else: run(make_backend(a.backend, a.model, a.revision, a.batch), a.requests, a.out, a.tag, a.start, a.end, a.batch, a.limit)
