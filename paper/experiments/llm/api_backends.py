"""Free-tier API backends for llm_forecaster.py (standard library only, so they run anywhere, even on a weak laptop).

Groq   (console.groq.com, free plan, checked 2026-11-08): openai/gpt-oss-120b 30 requests/min, 1,000 requests/day,
       200,000 tokens/day. The token cap binds first. Set GROQ_API_KEY.
Zoho   Catalyst QuickML LLM serving endpoint (free tier about 1M input + 500K output tokens/month, checked 2026-11-08).
       Set ZOHO_CLIENT_ID, ZOHO_CLIENT_SECRET, ZOHO_REFRESH_TOKEN (self-client, scope QuickML.deployment.READ),
       ZOHO_ORG_ID, ZOHO_ENDPOINT_URL (copied from the endpoint page) and optionally ZOHO_DC (default 'in').
       The request body is not documented in text form; ZOHO_BODY_TEMPLATE (a JSON string or a path to a JSON file)
       lets you paste the sample from the endpoint's "JSON format" test view, with {system}, {user}, {max_tokens}
       placeholders. NOT yet tested against a live endpoint.

Both stop cleanly with StopRun when a daily quota is reached; rerun the same command later to continue.
"""
import json, os, time, urllib.request, urllib.error, urllib.parse

class StopRun(Exception):
    pass

def _post(url, body, headers, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r: return r.status, dict(r.headers), json.loads(r.read() or b'{}')
    except urllib.error.HTTPError as e:
        txt = e.read().decode(errors='replace')
        try: data = json.loads(txt)
        except ValueError: data = {'error': {'message': txt}}
        return e.code, dict(e.headers), data

class GroqBackend:
    URL = 'https://api.groq.com/openai/v1/chat/completions'
    def __init__(self, model='openai/gpt-oss-120b', rpm=28, max_tokens=None):
        self.key = os.environ.get('GROQ_API_KEY') or exit('set GROQ_API_KEY')
        self.model_id, self.revision, self.max_batch = model, 'api', 1
        self.gap = 60. / rpm; self.last = 0.; self.max_tokens = max_tokens
        self.reasoning = model.startswith('openai/gpt-oss')      # reasoning model: keep reasoning short, it counts against the token cap
    def generate(self, batch):
        return [self._one(r) for r in batch]
    def _one(self, r):
        body = dict(model=self.model_id, temperature=0, seed=0,
                    messages=[{'role': 'system', 'content': r['system']}, {'role': 'user', 'content': r['user']}],
                    max_completion_tokens=self.max_tokens or (400 if self.reasoning else r['max_new_tokens'] + 10))
        if self.reasoning: body['reasoning_effort'] = 'low'
        for attempt in range(8):
            wait = self.last + self.gap - time.time()
            if wait > 0: time.sleep(wait)
            self.last = time.time()
            code, hdr, data = _post(self.URL, body, {'Authorization': f'Bearer {self.key}', 'Content-Type': 'application/json',
                                                      'User-Agent': 'fintrust-research/1.0'})
            if code == 200: return (data['choices'][0]['message'].get('content') or '').strip()
            msg = str(data.get('error', {}).get('message', data))
            if code == 429:
                if 'per day' in msg or '(RPD)' in msg or '(TPD)' in msg: raise StopRun(f'Groq daily limit reached: {msg[:200]}')
                time.sleep(float(hdr.get('retry-after') or hdr.get('Retry-After') or 5 * (attempt + 1))); continue
            if code >= 500: time.sleep(5 * (attempt + 1)); continue
            raise RuntimeError(f'Groq HTTP {code}: {msg[:300]}')
        raise RuntimeError('Groq: too many retries')

class ZohoBackend:
    def __init__(self, model='zoho-quickml', gap=2.):
        e = os.environ; need = ['ZOHO_CLIENT_ID', 'ZOHO_CLIENT_SECRET', 'ZOHO_REFRESH_TOKEN', 'ZOHO_ORG_ID', 'ZOHO_ENDPOINT_URL']
        miss = [k for k in need if not e.get(k)]
        if miss: exit('set ' + ', '.join(miss))
        self.e = e; self.model_id, self.revision, self.max_batch = model, 'api', 1
        self.dc = e.get('ZOHO_DC', 'in'); self.token = None; self.exp = 0; self.gap = gap; self.last = 0.
        t = e.get('ZOHO_BODY_TEMPLATE')
        self.template = (json.load(open(t)) if t and os.path.exists(t) else json.loads(t)) if t else \
            {'prompt': '{user}', 'system_prompt': '{system}', 'temperature': 0, 'max_tokens': '{max_tokens}'}
    def _token(self):
        if self.token and time.time() < self.exp - 60: return self.token
        q = urllib.parse.urlencode(dict(refresh_token=self.e['ZOHO_REFRESH_TOKEN'], client_id=self.e['ZOHO_CLIENT_ID'],
                                        client_secret=self.e['ZOHO_CLIENT_SECRET'], grant_type='refresh_token'))
        code, _, data = _post(f'https://accounts.zoho.{self.dc}/oauth/v2/token?{q}', None, {})
        if code != 200 or 'access_token' not in data: raise RuntimeError(f'Zoho token error {code}: {data}')
        self.token, self.exp = data['access_token'], time.time() + float(data.get('expires_in', 3600)); return self.token
    def _fill(self, x, r):
        if isinstance(x, dict): return {k: self._fill(v, r) for k, v in x.items()}
        if isinstance(x, list): return [self._fill(v, r) for v in x]
        if x == '{max_tokens}': return r['max_new_tokens'] + 10
        if isinstance(x, str): return x.replace('{system}', r['system']).replace('{user}', r['user'])
        return x
    @staticmethod
    def _text(d):
        for path in (('response',), ('output',), ('text',), ('data', 'response'), ('data', 'output'), ('choices', 0, 'message', 'content')):
            v = d
            try:
                for k in path: v = v[k]
                if isinstance(v, str): return v.strip()
            except (KeyError, IndexError, TypeError): pass
        return json.dumps(d)[:500]      # unknown shape: keep it, the parser will flag it
    def generate(self, batch):
        out = []
        for r in batch:
            for attempt in range(6):
                wait = self.last + self.gap - time.time()
                if wait > 0: time.sleep(wait)
                self.last = time.time()
                code, hdr, data = _post(self.e['ZOHO_ENDPOINT_URL'], self._fill(self.template, r),
                                        {'Authorization': f'Zoho-oauthtoken {self._token()}', 'CATALYST-ORG': self.e['ZOHO_ORG_ID'],
                                         'Content-Type': 'application/json'})
                if code == 200: out.append(self._text(data)); break
                msg = json.dumps(data)[:300]
                if code == 401: self.token = None; continue
                if code == 429 or 'limit' in msg.lower() or 'credit' in msg.lower():
                    if attempt >= 2: raise StopRun(f'Zoho quota or rate limit: {msg}')
                    time.sleep(30 * (attempt + 1)); continue
                if code >= 500: time.sleep(5 * (attempt + 1)); continue
                raise RuntimeError(f'Zoho HTTP {code}: {msg}')
            else: raise RuntimeError('Zoho: too many retries')
        return out
